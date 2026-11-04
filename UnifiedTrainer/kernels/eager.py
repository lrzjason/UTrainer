# =============================================================================
# DERIVED CODE - kernel techniques learned from KohakuMoss.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# The reference implementation of every op in this module is
# `src/kohakumoss_kernels/backends/eager/*` in that repository: the eager ops
# there are the normative definition every other backend is checked against.
# The functions below are re-expressed against UnifiedTrainer's tensor layout
# ((B, L, H, D) attention, diffusers-style blocks), not copied verbatim.
# See kernels/ATTRIBUTION.md and md/14-kernel-optimizations-applied.md.
# =============================================================================
"""Reference (pure torch) implementations of the fused block ops.

These are the definition: the Triton kernels in this package must match them
within the precision bar, and every selector falls back here.  Semantics match
what the model adapters already compute, so swapping implementations cannot
change the numbers beyond floating-point ordering.

Layout: every op is a last-dimension op on a ``(B, L, D)`` tensor; attention is
``(B, L, H, D)`` with ``q``/``k``/``v`` on the same sequence axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "add_gated",
    "attention",
    "rms_norm",
    "rms_norm_modulate",
    "rope_apply",
    "sdpa_gqa",
    "silu_mul",
]


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-5,
    offset: float = 0.0,
) -> torch.Tensor:
    """RMSNorm with the scale applied as ``weight + offset`` (Krea-2 uses 1.0).

    The normalisation runs in float32 (the statistics and the reciprocal square
    root), the result is cast back to ``x.dtype`` - the reduction-in-fp32 rule.
    """
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * (weight.float() + offset)
    return normed.to(x.dtype)


def rms_norm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    eps: float = 1e-5,
    offset: float = 0.0,
) -> torch.Tensor:
    """``(1 + scale) * rms_norm(x, weight + offset) + shift`` in one call.

    ``scale``/``shift`` are broadcastable over ``(B, L, D)`` (the adaLN
    modulation is ``(B, 1, D)``).  Without them this is plain RMSNorm.
    """
    out = rms_norm(x, weight=weight, eps=eps, offset=offset).float()
    if scale is not None:
        out = out * (1.0 + scale.float())
    if shift is not None:
        out = out + shift.float()
    return out.to(x.dtype)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: ``silu(gate) * up``."""
    return F.silu(gate) * up


def rope_apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved rotary embedding on ``(B, L, H, D)``.

    ``cos``/``sin`` are ``(L, D)`` (``D`` is the head dimension).  Pair ``i`` is
    ``(x[..., 2i], x[..., 2i+1])`` - the ``use_real_unbind_dim=-1`` convention
    diffusers uses for Flux / Krea-2 / Qwen-Image:

        out[2i]   = x[2i] * cos[2i] - x[2i+1] * sin[2i]
        out[2i+1] = x[2i+1] * cos[2i+1] + x[2i] * sin[2i+1]

    Math in float32.
    """
    xf = x.float()
    cos = cos.float()[None, :, None, :]
    sin = sin.float()[None, :, None, :]
    x_even = xf[..., 0::2]
    x_odd = xf[..., 1::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
    out[..., 1::2] = x_odd * cos[..., 1::2] + x_even * sin[..., 1::2]
    return out.to(x.dtype)


def add_gated(
    residual: torch.Tensor,
    gate: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Residual with a per-token gate: ``residual + gate * value``."""
    return residual + gate * value


def sdpa_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Grouped-query attention that keeps SDPA on a fused kernel.

    ``enable_gqa=True`` is a trap on CUDA builds without flash attention: the
    fused kernels reject differing head counts, silently fall back to the fp32
    math backend, and the attention becomes several times slower (measured
    67.7 ms vs 24.2 ms forward+backward for one Krea-2 block at L=2048 on an
    RTX 5060 Ti).  Repeating the KV heads first keeps the memory-efficient
    kernel, at the cost of one small copy per call.
    """
    heads = query.shape[-2]
    kv_heads = key.shape[-2]
    if kv_heads != heads:
        if heads % kv_heads:
            raise ValueError(f"query heads {heads} not divisible by kv heads {kv_heads}")
        key = key.repeat_interleave(heads // kv_heads, dim=-2)
        value = value.repeat_interleave(heads // kv_heads, dim=-2)
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return out.transpose(1, 2)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Eager reference for the attention selector.

    Deliberately the same signature as the Triton entry point so the two are
    interchangeable behind :func:`UnifiedTrainer.kernels.attention`.
    """
    return sdpa_gqa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
