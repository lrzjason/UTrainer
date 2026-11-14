"""ComfyUI-style lazy quantised linear: weights stay packed in host RAM, decode per forward.

That is how ComfyUI runs this 33B encoder on a 16 GB card: the module keeps the uint8 packed
weights plus the two scale tensors (16 GB total for the whole encoder, not 66 GB), and each
forward dequantises only its own matrix onto the GPU. Per-layer transients are tens of MB,
so VRAM stays a few GB and host RAM stays at the packed size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Nvfp4Linear(nn.Module):
    """Drop-in nn.Linear whose weight is NVFP4-packed and decoded on demand."""

    def __init__(self, out_features: int, in_features: int, packed, block_scale, scale,
                 bias=None):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        # plain attributes: `.to()` must not move the 16 GB packed stack to the GPU
        self.packed = packed                                 # uint8 [out, in/2]
        self.block_scale = block_scale                       # e4m3 [out, in/16]
        self.scale = scale.reshape(1).float()
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .comfy_quant import dequant_nvfp4_ck

        device = x.device
        weight = dequant_nvfp4_ck(
            self.packed.to(device, non_blocking=True),
            self.block_scale.to(device, non_blocking=True),
            self.scale.to(device, non_blocking=True),
            dtype=x.dtype,
        )
        out = torch.nn.functional.linear(x, weight, self.bias)
        del weight
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features} (nvfp4)"


class W4A8Linear(nn.Module):
    """Same idea for ComfyUI's ``asym_w4a8_int8`` format (int4 codes + codebook + 2 scales)."""

    def __init__(self, out_features: int, in_features: int, packed, codebook, s_channel, s_rel,
                 bias=None, group_size: int = 16):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.packed = packed
        self.codebook = codebook
        self.s_channel = s_channel
        self.s_rel = s_rel
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .comfy_quant import dequant_w4a8_ck

        device = x.device
        weight = dequant_w4a8_ck(
            self.packed.to(device, non_blocking=True),
            self.codebook.to(device, non_blocking=True),
            self.s_channel.to(device, non_blocking=True),
            self.s_rel.to(device, non_blocking=True),
            group_size=self.group_size,
            dtype=x.dtype,
        )
        out = torch.nn.functional.linear(x, weight, self.bias)
        del weight
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features} (w4a8)"
