# Reference: adapted from ai-toolkit/toolkit/optimizers/optimizer_utils.py
# See aitoolkit_acc.md for full technique analysis.
"""
Stochastic rounding utilities for mixed-precision training.

When parameters live in bf16 / fp8 but gradients are accumulated in fp32,
naive rounding introduces systematic bias.  Stochastic rounding eliminates
that bias by rounding up or down probabilistically based on the truncated
mantissa bits.

Key functions:
 - ``copy_stochastic_bf16``  — bit-manipulation bf16 rounding
 - ``copy_stochastic``       — generic dispatcher (fp32 passthrough)
 - ``stochastic_grad_accummulation`` — hook for ``register_post_accumulate_grad_hook``
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

# ── Conditional import for QBytesTensor (optimum.quanto) ────────────────
try:
    from optimum.quanto import QBytesTensor  # type: ignore
    _HAS_QUANTO = True
except ImportError:
    QBytesTensor = None  # type: ignore
    _HAS_QUANTO = False


# ── Format helpers ──────────────────────────────────────────────────────

def get_format_params(dtype: torch.dtype) -> tuple[int, int]:
    """Return ``(mantissa_bits, total_bits)`` for *dtype*.

    ``mantissa_bits`` excludes the implicit leading 1.
    """
    if dtype == torch.float32:
        return 23, 32
    if dtype == torch.bfloat16:
        return 7, 16
    if dtype == torch.float16:
        return 10, 16
    if dtype == torch.float8_e4m3fn:
        return 3, 8
    if dtype == torch.float8_e5m2:
        return 2, 8
    if dtype == torch.int8:
        return 0, 8
    raise ValueError(f"Unsupported dtype: {dtype}")


def compute_scale_for_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> float:
    """Compute appropriate scale for quantizing *tensor* to *dtype*."""
    if dtype == torch.int8:
        abs_max = torch.max(torch.abs(tensor))
        return abs_max / 127.0 if abs_max > 0 else 1.0
    if dtype == torch.uint8:
        max_val = torch.max(tensor)
        min_val = torch.min(tensor)
        range_val = max_val - min_val
        return range_val / 255.0 if range_val > 0 else 1.0
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        abs_max = torch.max(torch.abs(tensor))
        max_repr = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
        return abs_max / max_repr if abs_max > 0 else 1.0
    raise ValueError(f"Unsupported dtype for quantization: {dtype}")


def quantize_tensor(
    tensor: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, float]:
    """Quantize a float tensor to *dtype* with appropriate scaling."""
    scale = compute_scale_for_dtype(tensor, dtype)
    if dtype == torch.int8:
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(dtype)
    elif dtype == torch.uint8:
        q = torch.clamp(torch.round(tensor / scale), 0, 255).to(dtype)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        q = (tensor / scale).to(dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return q, scale


def update_parameter(target, result_float: torch.Tensor) -> None:
    """Write *result_float* back to *target*, re-quantizing if needed."""
    if _HAS_QUANTO and isinstance(target, QBytesTensor):
        target_dtype = target._data.dtype
        device = target._data.device
        result_float = result_float.to(device)
        quantized_data, new_scale = quantize_tensor(result_float, target_dtype)
        target._data.copy_(quantized_data)
        target._scale.copy_(new_scale)
    else:
        target.copy_(result_float)


# ── Core stochastic rounding ────────────────────────────────────────────

def copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor) -> None:
    """Stochastically round *source* (fp32) into *target* (bf16 view).

    Uses bit-manipulation: add random noise to the lower 16 mantissa bits
    of the fp32 representation, then mask them off.
    """
    # create a random 16-bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bits of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bits of the mantissa (0xFFFF0000 == -65536 signed)
    result.bitwise_and_(-65536)

    # copy the higher 16 bits into the target bf16 tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def copy_stochastic(
    target: torch.Tensor, source: torch.Tensor, eps: Optional[float] = None
) -> None:
    """Generic stochastic-rounding copy.

   - fp32 target → plain copy
   - bf16 target → ``copy_stochastic_bf16``
   - fp8 / int8 target → mantissa-based rounding + clamp + ``update_parameter``
    """
    with torch.no_grad():
        assert target.device.type != "cpu", "Target is on cpu!"
        assert source.device.type != "cpu", "Source is on cpu!"

        if target.dtype == torch.float32:
            target.copy_(source)
            return
        if target.dtype == torch.bfloat16:
            copy_stochastic_bf16(target, source)
            return

        mantissa_bits, _ = get_format_params(target.dtype)
        round_factor = 2 ** (23 - mantissa_bits)

        noise = torch.rand_like(source) - 0.5
        rounded = torch.round(source * round_factor + noise)
        result_float = rounded / round_factor

        if target.dtype == torch.float8_e4m3fn:
            result_float.clamp_(-448.0, 448.0)
        elif target.dtype == torch.float8_e5m2:
            result_float.clamp_(-57344.0, 57344.0)

        update_parameter(target, result_float)


# ── Gradient accumulation hook ──────────────────────────────────────────

def stochastic_grad_accummulation(param: torch.nn.Parameter) -> None:
    """``register_post_accumulate_grad_hook`` callback.

    Accumulates gradient in fp32 with stochastic rounding so that bf16/fp8
    parameters benefit from unbiased gradient accumulation across micro-batches.
    """
    if hasattr(param, "_accum_grad"):
        grad_fp32 = param._accum_grad.clone().to(torch.float32)
        grad_fp32.add_(param.grad.to(torch.float32))
        copy_stochastic(param._accum_grad, grad_fp32)
        del grad_fp32
        del param.grad
    else:
        param._accum_grad = param.grad.clone()
        del param.grad
