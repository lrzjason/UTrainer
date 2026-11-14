"""ComfyUI quantized checkpoint decoding (asym_w4a8_int8 + NVFP4-AWQ).

Comfy-Org's MiniMax-H3-Image release ships two ComfyUI-specific formats:

``asym_w4a8_int8`` (DiT, 11.7 GB)
    every quantized Linear is stored as four tensors::

        <p>.weight            int8 [out, in/2]   two 4-bit codes per byte
        <p>.weight_codebook   f32  [16]          the 16 quantisation levels
        <p>.weight_s_channel  f32  [out]         per-output-channel scale
        <p>.weight_s_rel      f8e4m3 [out, in/16] per-16-input-feature relative scale
        <p>.comfy_quant       json ``{"format": "asym_w4a8_int8"}``

    Verified on the real file (blocks.0.attn.out_proj): the low nibble holds the even
    feature index and the high nibble the odd one; dequantising with
    ``codebook[codes] * s_channel[:, None] * s_rel.repeat_interleave(16, 1)`` yields
    mean 0.0, std 0.085, range +/-1.30 -- a plausible weight distribution.

``NVFP4-AWQ`` (text encoder, 14.6 GB)
    ``<p>.weight`` uint8 [out, in/2]  two E2M1 values per byte
    ``<p>.weight_scale`` f8e4m3 [out, in/32]  per-16-value group scale (nibble pair)
    ``<p>.weight_scale_2`` f32 scalar         per-tensor second-level scale

This module only decodes; the callers decide when to materialise a layer (the 33B encoder
must never be dequantised in one piece).
"""
from __future__ import annotations

from typing import Iterator

import torch

_W4A8 = "asym_w4a8_int8"


def _f8e4m3_to_float(t: torch.Tensor) -> torch.Tensor:
    """Decode a raw uint8 buffer holding E4M3 values (PyTorch lacks a portable cast)."""
    b = t.view(torch.uint8).to(torch.int32)
    sign = torch.where((b >> 7) & 1 != 0, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    man = (b & 0x7).to(torch.float32) / 8.0
    normal = (1.0 + man) * torch.pow(2.0, (exp - 7).to(torch.float32))
    subnormal = man * (2.0 ** -6)
    return (sign * torch.where(exp == 0, subnormal, normal)).to(torch.float32)


def dequant_asym_w4a8(
    weight: torch.Tensor,
    codebook: torch.Tensor,
    s_channel: torch.Tensor,
    s_rel: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``int8 [out, in/2]`` + codebook + two scales -> ``[out, in]`` weight."""
    w = weight.view(torch.uint8).to(torch.int64)
    low = (w & 0x0F).to(torch.long)
    high = ((w >> 4) & 0x0F).to(torch.long)

    out_features, half = low.shape
    codes = torch.empty((out_features, half * 2), dtype=torch.long, device=low.device)
    codes[:, 0::2] = low
    codes[:, 1::2] = high

    values = codebook.to(torch.float32)[codes]
    n_codes = codes.shape[1]
    n_groups = s_rel.shape[1]
    if n_codes % n_groups != 0:
        raise ValueError(f"w4a8 layout mismatch: {n_codes} codes vs {n_groups} group scales")
    group = n_codes // n_groups
    scale = s_rel.to(torch.float32).repeat_interleave(group, dim=1)
    return (values * s_channel.to(torch.float32).unsqueeze(1) * scale).to(dtype)


def dequant_nvfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale_2: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_rows: int = 4096,
) -> torch.Tensor:
    """NVFP4-AWQ -> ``[out, in]``, decoded in row blocks so no giant intermediate exists."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32, device=weight.device,
    )
    b = weight.view(torch.uint8)
    out_features, half = b.shape
    in_features = half * 2
    scalar = float(scale_2.reshape(-1)[0]) if scale_2.numel() else 1.0
    scale_f = _f8e4m3_to_float(scale) if scale.dtype == torch.uint8 else scale.to(torch.float32)
    groups = scale_f.shape[1]
    group = in_features // groups
    if in_features % groups != 0:
        raise ValueError(f"nvfp4 layout mismatch: {in_features} values vs {groups} scales")

    out = torch.empty((out_features, in_features), dtype=dtype, device=weight.device)
    for r0 in range(0, out_features, block_rows):
        r1 = min(r0 + block_rows, out_features)
        chunk = b[r0:r1]
        lo = (chunk & 0x0F).to(torch.uint8)
        hi = ((chunk >> 4) & 0x0F).to(torch.uint8)
        codes = torch.empty((r1 - r0, in_features), dtype=torch.uint8, device=weight.device)
        codes[:, 0::2] = lo
        codes[:, 1::2] = hi
        vals = table[codes.long()]                       # float32, one block only
        gs = scale_f[r0:r1].repeat_interleave(group, dim=1)
        out[r0:r1] = (vals * gs * scalar).to(dtype)
        del lo, hi, codes, vals, gs
    return out


# ── comfy_kitchen backend (what ComfyUI itself uses) ────────────────────────
try:  # pragma: no cover - optional dependency, present on the training box
    from comfy_kitchen.tensor import (
        AsymW4A8Int8Layout as _CK_W4A8,
        TensorCoreNVFP4Layout as _CK_NVFP4,
    )
except Exception:  # noqa: BLE001
    _CK_W4A8 = _CK_NVFP4 = None


def dequant_nvfp4_ck(weight: torch.Tensor, block_scale: torch.Tensor, scale: torch.Tensor,
                     dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """NVFP4 decode via comfy_kitchen: identical to ComfyUI's own loader path."""
    if _CK_NVFP4 is None:
        raise RuntimeError("comfy_kitchen is not installed")
    out_features, half = weight.shape
    params = _CK_NVFP4.Params(
        scale=scale.reshape(1).to(torch.float32),
        block_scale=block_scale,
        orig_dtype=dtype,
        orig_shape=(out_features, half * 2),
    )
    return _CK_NVFP4.dequantize(weight, params).to(dtype)


def dequant_w4a8_ck(weight: torch.Tensor, codebook: torch.Tensor, s_channel: torch.Tensor,
                    s_rel: torch.Tensor, group_size: int = 16,
                    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """asym_w4a8_int8 decode via comfy_kitchen."""
    if _CK_W4A8 is None:
        raise RuntimeError("comfy_kitchen is not installed")
    out_features, half = weight.shape
    params = _CK_W4A8.Params(
        scale=s_rel,
        s_channel=s_channel,
        codebook=codebook,
        group_size=group_size,
        convrot_groupsize=256,
        orig_dtype=dtype,
        orig_shape=(out_features, half * 2),
    )
    return _CK_W4A8.dequantize(weight, params).to(dtype)
