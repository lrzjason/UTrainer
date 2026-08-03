# Reference: adapted from ai-toolkit/toolkit/memory_management/manager_modules.py
# See aitoolkit_acc.md for full technique analysis.
# Original inspiration: https://github.com/lodestone-rock/RamTorch
"""
Bouncing weights offload -ring-buffer pipelined layer offloading.

Weights live in pinned CPU RAM and are bounced to the GPU per-layer via
custom autograd functions.  Three CUDA streams (transfer, transfer_grad,
and the default compute stream) overlap H2D copies with compute, using a
depth-N ring buffer so multiple layers' weights can be in-flight at once.

Key concepts:
 - ``PIPELINE_DEPTH``  -ring buffer depth (default 4, overridable via env)
 - ``_stage_forward_weight``  -H2D next layer's weight into ring slot
 - ``_BouncingLinearFn`` / ``_BouncingConv2dFn``  -autograd functions that
    stream weight H2D and grad D2H asynchronously
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import has_torch_function_unary

if TYPE_CHECKING:
    from UnifiedTrainer.utils.memory_management.manager import MemoryManager


# ── Per-device global state registry ────────────────────────────────────

_DEVICE_STATE: dict = {}

# How many layers deep to prefetch weights. A deeper ring lets Python enqueue
# several layers ahead so the H2D stream stays saturated instead of stalling
# on a per-layer sync. Override with UnifiedTrainer_OFFLOAD_DEPTH.
PIPELINE_DEPTH = int(os.environ.get("UnifiedTrainer_OFFLOAD_DEPTH", "4"))


def _get_device_state(device: torch.device):
    """Get or initialize per-device CUDA state (streams, ring buffers, events)."""
    if isinstance(device, str):
        device = torch.device(device)

    # CPU path needs no CUDA state
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        d = max(2, PIPELINE_DEPTH)
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "depth": d,
                # streams
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                # forward weight ring: slot_ready = H2D done, slot_free = compute done
                "w_buffers": [None] * d,
                "b_buffers": [None] * d,
                "fwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "fwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "forward_clk": 0,
                # backward weight ring (re-fetch for grad-input)
                "w_bwd_buffers": [None] * d,
                "bwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "bwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "backward_clk": 0,
                # backward grad-staging ring (device-side grads -> CPU)
                "w_grad_buffers": [None] * d,
                "b_grad_buffers": [None] * d,
                "grad_compute_done": [torch.cuda.Event() for _ in range(d)],
                "grad_xfer_done": [torch.cuda.Event() for _ in range(d)],
            }
    return _DEVICE_STATE[device]


# ── Ring-buffer staging helpers ─────────────────────────────────────────
#
# Each transfer waits only on the event for the *specific slot* it is about to
# overwrite (the compute that used that slot D layers ago), not on a single
# global "compute started" event. With D slots that prior compute is long done,
# so the transfer stream never actually stalls and stays D layers ahead of
# compute. This is the deeper-pipeline + relaxed-dependency change in one.


def _stage_forward_weight(state, device, materialize, weight_cpu, bias_cpu):
    """H2D the next forward weight (+bias) into its ring slot; return (idx, w, b).

    Caller runs compute, then calls _release_forward_slot(state, idx).
    """
    d = state["depth"]
    idx = state["forward_clk"]
    state["forward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    with torch.cuda.stream(ts):
        ts.wait_event(state["fwd_slot_free"][idx])
        state["w_buffers"][idx] = materialize(weight_cpu, device)
        state["b_buffers"][idx] = (
            bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
        )
        state["fwd_slot_ready"][idx].record()
    torch.cuda.current_stream().wait_event(state["fwd_slot_ready"][idx])
    return idx, state["w_buffers"][idx], state["b_buffers"][idx]


def _release_forward_slot(state, idx):
    """Slot is reusable once the compute stream finishes the op that read it."""
    state["fwd_slot_free"][idx].record()


def _stage_backward_weight(state, device, materialize, weight_cpu):
    """H2D the next backward weight into its ring slot; return (idx, w).

    Caller runs grad-input compute, then _release_backward_weight_slot.
    """
    d = state["depth"]
    idx = state["backward_clk"]
    state["backward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    with torch.cuda.stream(ts):
        ts.wait_event(state["bwd_slot_free"][idx])
        state["w_bwd_buffers"][idx] = materialize(weight_cpu)
        state["bwd_slot_ready"][idx].record()
    torch.cuda.current_stream().wait_event(state["bwd_slot_ready"][idx])
    return idx, state["w_bwd_buffers"][idx]


def _release_backward_weight_slot(state, idx):
    state["bwd_slot_free"][idx].record()


def _stage_grads_to_cpu(state, idx, grad_w_gpu, grad_b_gpu):
    """Copy freshly-computed device grads (in staging slot idx) to CPU on the
    grad stream, overlapping the next H2D. Returns (grad_w_cpu, grad_b_cpu)."""
    gs = state["transfer_grad_stream"]
    state["grad_compute_done"][idx].record()  # on the compute stream
    grad_w_cpu = grad_b_cpu = None
    with torch.cuda.stream(gs):
        gs.wait_event(state["grad_compute_done"][idx])
        if grad_w_gpu is not None:
            grad_w_cpu = grad_w_gpu.to("cpu", non_blocking=True)
        if grad_b_gpu is not None:
            grad_b_cpu = grad_b_gpu.to("cpu", non_blocking=True)
        state["grad_xfer_done"][idx].record()
    return grad_w_cpu, grad_b_cpu


# ── Quantized tensor detection ──────────────────────────────────────────

def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    """Detect torchao wrapper tensors."""
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in ("_scale", "_scales", "_zero_point", "_zp",
                 "_block_size", "_group_size", "_pack_dim"):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    """Check if *t* is any kind of quantized tensor (torch or torchao)."""
    if t is None:
        return False
    try:
        if torch.is_quantized(t):
            return True
    except Exception:
        pass
    if _is_ao_quantized_tensor(t):
        return True
    return not t.dtype.is_floating_point


def _pin_inner_tensors(t: torch.Tensor) -> None:
    """Pin the leaf storage of a tensor-subclass (e.g. torchao float8) in place.

    Quantized wrappers can't be pin_memory()'d directly, but they expose their
    real data as inner tensors via __tensor_flatten__. Pinning those lets the
    per-layer H2D bounce run async and overlap with compute.
    """
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return
    for name in names:
        inner = getattr(t, name, None)
        if inner is None:
            continue
        if hasattr(inner, "__tensor_flatten__"):
            _pin_inner_tensors(inner)
        elif (isinstance(inner, torch.Tensor)
              and inner.device.type == "cpu"
              and not inner.is_pinned()):
            try:
                setattr(t, name, inner.pin_memory())
            except Exception:
                pass


def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Ensure *t* is on CPU and pinned for fast async H2D transfer."""
    if t is None:
        return None
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    if _is_quantized_tensor(t):
        if torch.cuda.is_available():
            _pin_inner_tensors(t)
        return t
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


def _move_params_to_cpu_and_pin(module: nn.Module):
    """Force parameters to CPU (+pinned) so we can 'bounce' them per forward/backward."""
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(module, name, None)
            if not isinstance(param, nn.Parameter):
                continue
            cpu_data = _ensure_cpu_pinned(param.data).detach()
            if _is_quantized_tensor(param.data):
                # Tensor-subclass weights: replace the whole Parameter so device move sticks
                setattr(
                    module,
                    name,
                    nn.Parameter(cpu_data, requires_grad=param.requires_grad),
                )
            else:
                param.data = cpu_data


# ════════════════════════════════════════════════════════════════════════
# Autograd functions (CUDA bouncing)
# ════════════════════════════════════════════════════════════════════════


class _BouncingLinearFn(torch.autograd.Function):
    """Bounces a Linear layer's weight CPU→GPU per forward, GPU→CPU per backward."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        def _materialize_linear_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            return cpu_w.to(dev, non_blocking=True)

        if device.type != "cuda":
            out = F.linear(
                x.to("cpu"),
                _materialize_linear_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_linear_weight, weight_cpu, bias_cpu
        )
        out = F.linear(x, w_gpu, b_gpu)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)

        if device.type != "cuda":
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_mat.dtype != target_dtype and target_dtype in (
                torch.bfloat16, torch.float16, torch.float32,
            ):
                w_mat = w_mat.to(target_dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None

        state = _get_device_state(device)

        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            return cpu_w.to(device, non_blocking=True)

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu
        )

        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd
        _release_backward_weight_slot(state, idx)

        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
                state["b_grad_buffers"][idx] = b_grad_gpu
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None


class _BouncingConv2dFn(torch.autograd.Function):
    """Bounces a Conv2d layer's weight CPU→GPU per forward, GPU→CPU per backward."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        def _materialize_conv_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            return cpu_w.to(dev, non_blocking=True)

        if device.type != "cuda":
            out = F.conv2d(
                x.to("cpu"),
                _materialize_conv_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
                stride, padding, dilation, groups,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype)
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_conv_weight, weight_cpu, bias_cpu
        )
        out = F.conv2d(x, w_gpu, b_gpu, stride, padding, dilation, groups)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype = ctx.meta

        if (
            isinstance(device, torch.device) and device.type != "cuda"
        ) or device == "cpu":
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_cpu.dtype != target_dtype and target_dtype in (
                torch.bfloat16, torch.float16, torch.float32,
            ):
                w_cpu = w_cpu.to(target_dtype)
            from torch.nn.grad import conv2d_input, conv2d_weight

            grad_input = conv2d_input(
                x_cpu.shape, w_cpu, go,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
            )
            grad_weight = (
                conv2d_weight(
                    x_cpu, w_cpu.shape, go,
                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                )
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go.sum(dim=(0, 2, 3))
                if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None, None, None, None, None

        state = _get_device_state(device)

        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            return cpu_w.to(device, non_blocking=True)

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu
        )

        from torch.nn.grad import conv2d_input, conv2d_weight

        grad_input = conv2d_input(
            x.shape, w_bwd, grad_out.to(dtype=target_dtype),
            stride=stride, padding=padding, dilation=dilation, groups=groups,
        )
        _release_backward_weight_slot(state, idx)

        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = conv2d_weight(
                    x, weight_cpu.shape, grad_out,
                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                )
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=(0, 2, 3))
                state["b_grad_buffers"][idx] = b_grad_gpu
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight, grad_bias, None, None, None, None, None,
        )


# ════════════════════════════════════════════════════════════════════════
# Layer memory managers
# ════════════════════════════════════════════════════════════════════════


class BaseLayerMemoryManager:
    """Base class for per-layer memory managers."""

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        self.module: nn.Module = module
        self.manager: "MemoryManager" = manager

    @classmethod
    def attach(cls, module: nn.Module, manager: "MemoryManager"):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, manager)

        # Mark parameters as memory managed
        for param in module.parameters(recurse=False):
            param._is_memory_managed = True


class LinearLayerMemoryManager(BaseLayerMemoryManager):
    """Manages a Linear layer's weight on CPU, bouncing it per forward/backward."""

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # 2) Hijack forward
        self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingLinearFn.apply(x, weight_cpu, bias_cpu, device)

        self.module.forward = _mm_forward
        self.module._memory_management_device = self.manager.process_device


class ConvLayerMemoryManager(BaseLayerMemoryManager):
    """Manages a Conv2d layer's weight on CPU, bouncing it per forward/backward."""

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory
        _move_params_to_cpu_and_pin(self.module)

        # Cache static conv attributes
        stride = (
            self.module.stride
            if isinstance(self.module.stride, tuple)
            else (self.module.stride, self.module.stride)
        )
        padding = (
            self.module.padding
            if isinstance(self.module.padding, tuple)
            else (self.module.padding, self.module.padding)
        )
        dilation = (
            self.module.dilation
            if isinstance(self.module.dilation, tuple)
            else (self.module.dilation, self.module.dilation)
        )
        groups = self.module.groups

        # 2) Hijack forward
        self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingConv2dFn.apply(
                x, weight_cpu, bias_cpu, device,
                stride, padding, dilation, groups,
            )

        self.module.forward = _mm_forward
        self.module._memory_management_device = self.manager.process_device
