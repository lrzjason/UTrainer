# Reference: adapted from ai-toolkit/toolkit/memory_management/manager.py
# See aitoolkit_acc.md for full technique analysis.
"""
MemoryManager -orchestrates bouncing weights offload for a model.

``MemoryManager.attach(model, device, offload_percent)`` walks the model tree
and replaces Linear / Conv2d forwards with ``_BouncingLinearFn`` /
``_BouncingConv2dFn`` autograd functions that stream weights CPU→GPU per layer
and overlap transfers with compute using a depth-N ring buffer.

``offload_percent`` controls what fraction of layers are offloaded (0.0 = none,
1.0 = all). At < 1.0, layers are randomly selected for offload.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from UnifiedTrainer.utils.memory_management.manager_modules import (
    LinearLayerMemoryManager,
    ConvLayerMemoryManager,
    _DEVICE_STATE,
)

# Module class names that should be bounced to CPU
LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
]

CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
]

# Modules that should NOT be offloaded (small, frequently used)
UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d",
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class MemoryManager:
    """Manages bouncing weights offload for a model.

    Use ``MemoryManager.attach(model, device, offload_percent)`` to enable
    offloading, and ``MemoryManager.detach(model)`` to disable.
    """

    def __init__(
        self,
        module: nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list = []

    def memory_managed_to(self, *args, **kwargs):
        """Custom ``.to()`` that handles memory-managed modules."""
        # First move all unmanaged modules normally
        for module in self.unmanaged_modules:
            if isinstance(module, torch.nn.Parameter):
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)

        # Check for dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls,
        module: nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
        ignore_modules: list = None,
    ):
        """Attach bouncing offload to *module*.

        Args:
            module: the model to offload
            device: the GPU device to bounce weights to
            offload_percent: fraction of layers to offload (0.0 -.0).
                At 1.0, all eligible layers are offloaded.
                At 0.5, ~50% are randomly selected.
            ignore_modules: modules to skip (kept on device)
        """
        if hasattr(module, "_memory_manager"):
            return  # already attached

        ignore_modules = ignore_modules or []
        module._memory_manager = cls(module, device)

        # Override .to() to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # Add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        modules_processed = list(ignore_modules)

        # Attach to all eligible submodules
        for _name, sub_module in module.named_modules():
            for _child_name, child_module in sub_module.named_modules():
                if child_module in modules_processed:
                    continue

                class_name = child_module.__class__.__name__

                if class_name in LINEAR_MODULES:
                    skip = False
                    if offload_percent < 1.0:
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        LinearLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                    modules_processed.append(child_module)

                elif class_name in CONV_MODULES:
                    skip = False
                    if offload_percent < 1.0:
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                    modules_processed.append(child_module)

                elif (
                    class_name in UNMANAGED_MODULES
                    or any(inc in class_name for inc in UNMANAGED_MODULES_INCLUDES)
                ):
                    module._memory_manager.unmanaged_modules.append(child_module)

    @classmethod
    def detach(cls, module: nn.Module):
        """Reverse of ``attach()``: restore original forwards, unpin, clear state.

        Call this before unloading or replacing a module that had ``attach()`` applied.
        """
        if not hasattr(module, "_memory_manager"):
            return

        # Move unmanaged modules back to CPU
        for unmanaged in module._memory_manager.unmanaged_modules:
            try:
                if isinstance(unmanaged, torch.nn.Parameter):
                    unmanaged.data = unmanaged.data.to("cpu")
                else:
                    unmanaged.to("cpu")
            except Exception:
                pass

        # Restore original .to()
        if hasattr(module, "_mm_to"):
            module.to = module._mm_to
            del module._mm_to

        del module._memory_manager

        # Restore original forwards and unpin memory for all managed layers
        for child in module.modules():
            lmm = getattr(child, "_layer_memory_manager", None)
            if lmm is None:
                continue

            original_forward = getattr(lmm, "_original_forward", None)
            if original_forward is not None:
                child.forward = original_forward

            # Unpin memory: clone pinned tensors back to normal memory
            for param_name in ("weight", "bias"):
                param = getattr(child, param_name, None)
                if param is None or not isinstance(param, torch.nn.Parameter):
                    continue
                try:
                    if param.data.is_pinned():
                        object.__setattr__(
                            child,
                            param_name,
                            torch.nn.Parameter(
                                param.data.clone(),
                                requires_grad=param.requires_grad,
                            ),
                        )
                except Exception:
                    pass

            del child._layer_memory_manager
            if hasattr(child, "_memory_management_device"):
                del child._memory_management_device
            if hasattr(child, "_is_memory_managed"):
                del child._is_memory_managed

        # Clear global CUDA device state
        keys_to_delete = [
            dev for dev in _DEVICE_STATE
            if isinstance(dev, torch.device) and dev.type == "cuda"
        ]
        for key in keys_to_delete:
            del _DEVICE_STATE[key]

        torch.cuda.empty_cache()
