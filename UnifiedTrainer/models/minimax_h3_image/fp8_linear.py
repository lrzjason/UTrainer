"""FP8 (e4m3) linear layer for the frozen MiniMax-H3 DiT.

The base model stays frozen, so its weight is stored once as `float8_e4m3fn` plus one scale per
output row: 33 GB for a 33B transformer, versus 16 GB for the NF4 snapshot. The forward runs
`torch._scaled_mm` (Ada/Hopper e4m3 tensor cores); the backward only needs the *input* gradient,
which is computed from the weight dequantised on the fly, so no bf16 copy of the model is kept.

`_scaled_mm` wants `mat1` `[M, K]`, `mat2` `[K, N]` with `stride(0) == 1`, and float32 scales of
shape `[M, 1]` / `[1, N]`. Keeping the weight in its natural `[out, in]` row-major layout and
passing the transpose *view* satisfies that without ever materialising a transposed copy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_FP8_MAX = 448.0  # largest magnitude representable in e4m3fn


def to_fp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` bf16 -> (``[out, in]`` fp8 e4m3, ``[1, out]`` float32 scale)."""
    w = weight.detach().to(torch.float32)
    scale = (w.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-8)   # [out, 1]
    q = (w / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.t().contiguous()                           # [out,in], [1,out]


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """fp8 ``[out, in]`` + ``[1, out]`` scale -> bf16 ``[out, in]`` (transient)."""
    return q.to(dtype) * scale.t().to(dtype)


class _Fp8Matmul(torch.autograd.Function):
    """``y = x @ W.T`` with W stored as fp8; only the input gradient is required."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor):
        ctx.save_for_backward(q, scale)
        # _scaled_mm rejects bf16 scales -- an amax on a bf16 input yields bf16, so force fp32.
        a_scale = (x.detach().to(torch.float32).abs().amax(dim=-1, keepdim=True)
                   .clamp_min(1e-8) / _FP8_MAX)                            # [M, 1]
        x_fp8 = (x.detach().to(torch.float32) / a_scale).clamp(
            -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        return torch._scaled_mm(x_fp8, q.t(), scale_a=a_scale, scale_b=scale,
                                out_dtype=x.dtype, use_fast_accum=True)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, scale = ctx.saved_tensors
        w = dequantize(q, scale, grad_out.dtype)                           # [out, in]
        return grad_out @ w, None, None


class Fp8Linear(nn.Module):
    """Drop-in replacement for a frozen `nn.Linear` whose weight lives in fp8 e4m3."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight_fp8", torch.empty(out_features, in_features,
                                                       dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale", torch.empty(1, out_features,
                                                         dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device,
                                             dtype=dtype)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Fp8Linear":
        out_f, in_f = linear.weight.shape
        mod = cls(in_f, out_f, bias=linear.bias is not None,
                  device=linear.weight.device, dtype=linear.weight.dtype)
        q, s = to_fp8_weight(linear.weight)
        mod.weight_fp8.copy_(q)
        mod.weight_scale.copy_(s)
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        # `_scaled_mm` supports only bf16/fp16 output for row-wise scaling, but the block stack
        # can hand this layer float32 activations (the checkpoint's `proj_in` / `time_embedder`
        # stay float32, so the first rows of the packed sequence arrive as fp32). Run the GEMM
        # in bf16 -- exactly what bitsandbytes does internally for a 4-bit layer -- and cast the
        # result back; the cast is a differentiable op, so the gradient keeps its fp32 path.
        compute_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        y = _Fp8Matmul.apply(x.reshape(-1, shape[-1]).to(compute_dtype),
                             self.weight_fp8, self.weight_scale)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, weight=fp8_e4m3")
