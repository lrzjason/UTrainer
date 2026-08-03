# Reference: adapted from ai-toolkit/toolkit/optimizers/optimizer_utils.py
# See aitoolkit_acc.md for full technique analysis.
"""
Auto8bitTensor — int8-quantized tensor wrapper for optimizer state.

Stores the exponential moving average of gradients (exp_avg, exp_avg_sq) in
int8 + a scalar scale, cutting optimizer VRAM by ~4x compared to fp32 state.
"""
from __future__ import annotations

import torch
from torch import Tensor


class Auto8bitTensor:
    """Wraps a tensor as int8 data + float scale.

    The constructor accepts either a plain tensor (which gets quantized)
    or a state-dict (for checkpoint loading).
    """

    def __init__(self, data: Tensor, *args, **kwargs):
        if isinstance(data, dict):
            self._load_from_state_dict(data)
        else:
            abs_max = data.abs().max().item()
            scale = abs_max / 127.0 if abs_max > 0 else 1.0

            self.quantized = (data / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale = scale
            self.orig_dtype = data.dtype

    def dequantize(self) -> Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        dtype = None
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        elif "dtype" in kwargs:
            dtype = kwargs["dtype"]
            del kwargs["dtype"]

        if dtype is not None:
            return self.dequantize().to(dtype=dtype, *args, **kwargs)
        return self.dequantize().to(*args, **kwargs)

    def state_dict(self) -> dict:
        return {
            "quantized": self.quantized,
            "scale": self.scale,
            "orig_dtype": self.orig_dtype,
        }

    def _load_from_state_dict(self, state_dict: dict) -> None:
        self.quantized = state_dict["quantized"]
        self.scale = state_dict["scale"]
        self.orig_dtype = state_dict["orig_dtype"]

    def __str__(self):
        return f"Auto8bitTensor({self.dequantize()})"
