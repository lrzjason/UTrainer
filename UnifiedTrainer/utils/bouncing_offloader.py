"""Bouncing layer offloader for torchao-quantized models.

Ported from ai-toolkit's ``toolkit.memory_management.manager_modules``.

Instead of swapping entire transformer blocks (kohya block-swap, which is
fundamentally incompatible with torchao ``AffineQuantizedTensor``), this
module attaches a ``_BouncingLinearFn`` custom autograd function to every
``nn.Linear`` in the model. The quantized weights stay permanently on CPU
(pinned for fast H2D). On each forward/backward, the weight is copied to
GPU, dequantized on-GPU, used for the matmul, then immediately discarded.

This achieves the same VRAM savings as block-swap without ever touching
``.data`` on a torchao tensor — the key operation that crashes.
"""

import os
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import has_torch_function_unary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-device CUDA stream state (ring-buffer staging)
# ---------------------------------------------------------------------------
_DEVICE_STATE = {}
PIPELINE_DEPTH = int(os.environ.get("TORCHAO_OFFLOAD_DEPTH", "4"))


def _get_device_state(device: torch.device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]
    if device not in _DEVICE_STATE:
        d = max(2, PIPELINE_DEPTH)
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "depth": d,
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "w_buffers": [None] * d,
                "b_buffers": [None] * d,
                "fwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "fwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "forward_clk": 0,
                "w_bwd_buffers": [None] * d,
                "bwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "bwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "backward_clk": 0,
                "w_grad_buffers": [None] * d,
                "b_grad_buffers": [None] * d,
                "grad_compute_done": [torch.cuda.Event() for _ in range(d)],
                "grad_xfer_done": [torch.cuda.Event() for _ in range(d)],
            }
    return _DEVICE_STATE[device]


def _stage_forward_weight(state, device, materialize, weight_cpu, bias_cpu):
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
    state["fwd_slot_free"][idx].record()


def _stage_backward_weight(state, device, materialize, weight_cpu):
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


# ---------------------------------------------------------------------------
# Quantized-tensor detection (never touches .data)
# ---------------------------------------------------------------------------


def _is_ao_quantized_tensor(t) -> bool:
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in ("_scale", "_scales", "_zero_point", "_zp", "_block_size",
                 "_group_size", "_pack_dim"):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t) -> bool:
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


# ---------------------------------------------------------------------------
# CPU pinning helpers
# ---------------------------------------------------------------------------


def _pin_inner_tensors(t) -> None:
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


def _ensure_cpu_pinned(t):
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
    """Force parameters to CPU (+pinned) so we can bounce them per forward/backward."""
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(module, name, None)
            if not isinstance(param, nn.Parameter):
                continue
            cpu_data = _ensure_cpu_pinned(param.data).detach()
            if _is_quantized_tensor(param.data):
                # Tensor-subclass weights ignore param.data = ... assignment
                setattr(module, name,
                        nn.Parameter(cpu_data, requires_grad=param.requires_grad))
            else:
                param.data = cpu_data


# ---------------------------------------------------------------------------
# Bouncing autograd functions
# ---------------------------------------------------------------------------


class _BouncingLinearFn(torch.autograd.Function):
    """Bounce a quantized weight CPU→GPU→compute→discard per forward/backward."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device):
        target_dtype = (
            x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        def _materialize(cpu_w, dev):
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
                _materialize(weight_cpu, torch.device("cpu")),
                bias_cpu,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize, weight_cpu, bias_cpu
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
                torch.bfloat16, torch.float16, torch.float32
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
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None

        state = _get_device_state(device)

        def _materialize_bwd(cpu_w):
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

        idx, w_bwd = _stage_backward_weight(state, device, _materialize_bwd, weight_cpu)

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
            gs = state["transfer_grad_stream"]
            state["grad_compute_done"][idx].record()
            with torch.cuda.stream(gs):
                gs.wait_event(state["grad_compute_done"][idx])
                if w_grad_gpu is not None:
                    grad_weight = w_grad_gpu.to("cpu", non_blocking=True)
                if b_grad_gpu is not None:
                    grad_bias = b_grad_gpu.to("cpu", non_blocking=True)
                state["grad_xfer_done"][idx].record()

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None


# ---------------------------------------------------------------------------
# LinearLayerMemoryManager — attaches bouncing to each nn.Linear
# ---------------------------------------------------------------------------

LINEAR_MODULES = {"Linear", "LoRACompatibleLinear", "QLinear"}
UNMANAGED_MODULES = {
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "Embedding",
    "EmbeddingBag", "RNNBase", "LSTM", "GRU", "RNN", "Conv3d",
}
UNMANAGED_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class BouncingOffloader:
    """Attach bouncing forward/backward to all nn.Linear in a model.

    Quantized weights are moved to CPU (pinned), and non-Linear modules
    (norms, embeddings) are moved to GPU. During forward/backward, the
    bouncing autograd function copies each weight to GPU, dequantizes,
    runs the matmul, then discards the GPU copy.
    """

    def __init__(self, device: torch.device, offload_percent: float = 1.0):
        self.device = device
        self.offload_percent = offload_percent
        self._original_forwards = {}

    def attach(self, model: nn.Module):
        """Attach bouncing to all base nn.Linear modules in model.

        For PEFT-wrapped modules (LoraLinear), only the ``base_layer`` is
        bounced — the LoRA adapter weights (lora_A, lora_B) are small and
        stay on GPU.

        When ``offload_percent < 1.0``, a random fraction of Linear layers
        are kept on GPU (not bounced) to reduce transfer overhead, matching
        ai-toolkit's ``layer_offloading_transformer_percent``.

        Returns a list of unmanaged modules (norms, embeddings) that
        the caller should move to GPU.
        """
        import random as _random
        unmanaged = []
        count = 0
        kept_on_gpu = 0
        visited = set()
        for name, module in model.named_modules():
            if id(module) in visited:
                continue
            cls_name = module.__class__.__name__

            # Detect PEFT LoraLinear wrapper (has base_layer + lora_A + lora_B)
            if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
                # This is a PEFT-wrapped Linear. Bounce ONLY the base_layer
                # (frozen quantized weight). LoRA adapters stay on GPU.
                base = module.base_layer

                # offload_percent: randomly keep some layers fully on GPU
                if self.offload_percent < 1.0 and _random.random() > self.offload_percent:
                    # Keep this layer on GPU — no bouncing
                    kept_on_gpu += 1
                    visited.add(id(base))
                    for adapter_name, adapter in module.lora_A.items():
                        adapter.to(self.device)
                    for adapter_name, adapter in module.lora_B.items():
                        adapter.to(self.device)
                    continue

                self._attach_linear(base)
                visited.add(id(base))
                count += 1
                # Move lora_A/lora_B to GPU (they were created on whatever
                # device the parent was on at add_adapter time).
                for adapter_name, adapter in module.lora_A.items():
                    adapter.to(self.device)
                for adapter_name, adapter in module.lora_B.items():
                    adapter.to(self.device)
                continue

            # Skip PEFT LoRA sub-modules entirely (lora_A.default, lora_B.default)
            if "lora_A" in name or "lora_B" in name:
                continue

            if isinstance(module, nn.Linear) or cls_name in LINEAR_MODULES:
                # offload_percent: randomly keep some layers fully on GPU
                if self.offload_percent < 1.0 and _random.random() > self.offload_percent:
                    kept_on_gpu += 1
                    visited.add(id(module))
                    continue
                self._attach_linear(module)
                visited.add(id(module))
                count += 1
            elif (cls_name in UNMANAGED_MODULES
                  or any(inc in cls_name for inc in UNMANAGED_INCLUDES)):
                unmanaged.append(module)

        logger.info(
            f"BouncingOffloader: attached to {count} Linear layers "
            f"(offload_percent={self.offload_percent}, {kept_on_gpu} kept on GPU), "
            f"{len(unmanaged)} unmanaged modules"
        )
        return unmanaged

    def _attach_linear(self, module: nn.Linear):
        """Attach bouncing to a single nn.Linear."""
        # Move weight + bias to CPU and pin
        _move_params_to_cpu_and_pin(module)
        # Save original forward
        self._original_forwards[id(module)] = module.forward

        def _mm_forward(x, *args, **kwargs):
            if args or kwargs:
                return self._original_forwards[id(module)](x, *args, **kwargs)
            weight_cpu = module.weight
            bias_cpu = getattr(module, "bias", None)
            # Defensive: ensure x is on the compute device. When an unmanaged
            # parent module (e.g. time_embed) is moved to GPU via
            # move_unmanaged_to_device, the torchao AffineQuantizedTensor params
            # in child Linear layers may stay on CPU depending on the AQT's
            # _apply() implementation.  If the parent's forward creates intermediate
            # tensors on CPU, the first bounced Linear gets a CPU input and
            # _BouncingLinearFn.forward crashes with "mat1 is on cpu".
            if x.device != self.device:
                x = x.to(self.device)
            return _BouncingLinearFn.apply(x, weight_cpu, bias_cpu, self.device)

        module.forward = _mm_forward
        module._memory_management_device = self.device

    def move_unmanaged_to_device(self, unmanaged_modules):
        """Move non-Linear modules (norms, embeddings, etc.) to GPU.

        Calls module.to(device) which may raise for torchao AQT child Linear
        layers. We catch the exception and fall back to selective param
        movement that skips bounced children.
        """
        for module in unmanaged_modules:
            try:
                module.to(self.device)
            except Exception:
                logger.debug(
                    f"module.to() failed for {module.__class__.__name__}, "
                    f"falling back to selective movement"
                )
                self._safe_to_device(module, self.device)

    def _safe_to_device(self, module: nn.Module, device: torch.device):
        """Move module params/buffers to device, skipping bounced Linear children.

        Only processes parameters and buffers that belong DIRECTLY to this module
        (not children). This prevents torchao-quantized child Linear layers from
        being moved back to GPU, which would undo the BouncingOffloader placement.
        """
        # Move own parameters (not children's — avoid hitting bounced Linear layers)
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            if _is_quantized_tensor(param):
                # torchao quantized tensor — BouncingOffloader handles its placement.
                # module.to() calls _apply which would try to move it to GPU, but
                # AQT._apply() behavior is unstable (may fail silently or create
                # inconsistent device state). Skip entirely.
                continue
            try:
                param.data = param.data.to(device)
            except Exception:
                logger.debug(
                    f"Skipping param '{name}' on {module.__class__.__name__} "
                    f"(could not move to {device})"
                )

        # Move own buffers
        for name, buf in list(module._buffers.items()):
            if buf is None or _is_quantized_tensor(buf):
                continue
            try:
                module._buffers[name] = buf.to(device)
            except Exception:
                pass

        # Recurse into children, but SKIP any child that was bounced
        # (i.e., its forward was replaced by _mm_forward). Those Linear
        # layers are managed by the BouncingOffloader — their weights must
        # stay on CPU (pinned) for the CPU→GPU bounce mechanism to work.
        for child in module.children():
            if id(child) in self._original_forwards:
                # This child is a bounced Linear — skip it
                continue
            self._safe_to_device(child, device)
