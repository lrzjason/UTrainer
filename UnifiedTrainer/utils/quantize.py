# Reference: adapted from ai-toolkit/toolkit/util/quantize.py
# See aitoolkit_acc.md for full technique analysis.
"""
Per-block weight quantization using torchao.

Quantizes transformer blocks one at a time to avoid VRAM peaks:
  move block to GPU -quantize -freeze -move back to CPU.

Supports torchao qtypes: float8, uint4, uint8, int8.
"""
from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Conditional torchao import ──────────────────────────────────────────
try:
    from torchao.quantization.quant_api import (
        quantize_ as torchao_quantize_,
        Float8WeightOnlyConfig,
        Int8WeightOnlyConfig,
        Int4WeightOnlyConfig,
    )
    try:
        from torchao.quantization.quant_api import UIntXWeightOnlyConfig
        _TORCHAO_AVAILABLE = True
    except ImportError:
        # Older torchao may not have UIntXWeightOnlyConfig
        UIntXWeightOnlyConfig = None  # type: ignore
        _TORCHAO_AVAILABLE = True
except ImportError:
    torchao_quantize_ = None  # type: ignore
    Float8WeightOnlyConfig = None  # type: ignore
    Int8WeightOnlyConfig = None  # type: ignore
    Int4WeightOnlyConfig = None  # type: ignore
    UIntXWeightOnlyConfig = None  # type: ignore
    _TORCHAO_AVAILABLE = False
    logger.warning(
        "torchao not installed -quantization features disabled. "
        "Install with: pip install torchao"
    )

# ── torchao qtype registry ──────────────────────────────────────────────

torchao_qtypes = {}
if _TORCHAO_AVAILABLE:
    torchao_qtypes = {
        "float8": Float8WeightOnlyConfig(),
        "int8": Int8WeightOnlyConfig(),
        "int4": Int4WeightOnlyConfig(),
    }
    if UIntXWeightOnlyConfig is not None:
        for bits in range(2, 8):
            dtype = getattr(torch, f"uint{bits}", None)
            if dtype is not None:
                torchao_qtypes[f"uint{bits}"] = UIntXWeightOnlyConfig(dtype)


class AOType:
    """Wrapper around a torchao quantization config."""

    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


def get_qtype(qtype: Union[str, AOType]) -> Optional[AOType]:
    """Resolve a qtype string to an AOType, or None if not available."""
    if qtype in torchao_qtypes:
        return AOType(qtype)
    return None


def get_torchao_config(qtype: str):
    """Return the torchao config for *qtype*, or None if not torchao."""
    if qtype is None:
        return None
    try:
        q = get_qtype(qtype)
    except Exception:
        return None
    return q.config if isinstance(q, AOType) else None


# ── Tensor utilities ────────────────────────────────────────────────────

def is_quantized_tensor(t) -> bool:
    """Check if *t* is a torchao quantized tensor."""
    if t is None:
        return False
    return "torchao" in type(t).__module__ and hasattr(t, "dequantize")


def dequantize_if_quantized(t):
    """Dequantize *t* if it is a quantized tensor, else return as-is."""
    return t.dequantize() if is_quantized_tensor(t) else t


def requantize_module_weight(module: nn.Module, fp_weight: torch.Tensor,
                              orig_dtype: torch.dtype, config) -> None:
    """Write a full-precision weight back, re-quantizing if config is provided."""
    module.weight = nn.Parameter(fp_weight.to(orig_dtype), requires_grad=False)
    if config is not None and _TORCHAO_AVAILABLE:
        torchao_quantize_(module, config)


# ── Core quantization ───────────────────────────────────────────────────

def quantize(
    model: nn.Module,
    weights: Optional[Union[str, AOType]] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    """Quantize the model's Linear / Conv2d submodules using torchao.

    Args:
        model: the model to quantize (in-place)
        weights: qtype string ("float8", "uint4", "int8", ...) or AOType
        include: fnmatch patterns for modules to include
        exclude: fnmatch patterns for modules to exclude
    """
    if not _TORCHAO_AVAILABLE:
        logger.warning("torchao not available -skipping quantization.")
        return

    if isinstance(weights, str):
        weights = get_qtype(weights)
    if weights is None:
        return

    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude

    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue  # Only quantize Linear layers (matches T2ITrainer behaviour)
        if is_quantized_tensor(getattr(m, "weight", None)):
            continue  # already quantized in an earlier pass (e.g., per-block)
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(
            fnmatch(name, pattern) for pattern in exclude
        ):
            continue
        try:
            if isinstance(weights, AOType):
                torchao_quantize_(m, weights.config)
        except Exception as e:
            logger.warning(f"Failed to quantize {name}: {e}")


def freeze(module: nn.Module) -> None:
    """Freeze a quantized module (set requires_grad=False on all params)."""
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


# ── Per-block quantization (the key VRAM-saving pattern) ─────────────────

def quantize_model(
    model: nn.Module,
    qtype: str = "float8",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    transformer_block_names: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    low_vram: bool = True,
):
    """Quantize a model block-by-block to avoid VRAM peaks.

    For each transformer block:
      1. Move to GPU
      2. Quantize weights (torchao)
      3. Freeze (requires_grad = False)
      4. Move back to CPU (if low_vram)

    This is the core VRAM optimization: never have the full model in GPU
    memory at once.

    Args:
        model: the transformer model to quantize
        qtype: quantization type ("float8", "int8", "uint4", etc.)
        device: target GPU device
        dtype: model dtype
        transformer_block_names: attribute paths to block lists
            (e.g. ["transformer_blocks"]). If None, auto-detect.
        exclude_modules: fnmatch patterns for modules to skip
        low_vram: if True, move blocks back to CPU after quantizing
    """
    if not _TORCHAO_AVAILABLE:
        logger.error("torchao not available -cannot quantize model.")
        return

    from UnifiedTrainer.utils.flush import flush

    quantization_type = get_qtype(qtype)
    if quantization_type is None:
        logger.error(f"Unknown qtype: {qtype}")
        return

    exclude_modules = exclude_modules or []

    # Auto-detect transformer blocks if not specified
    if transformer_block_names is None:
        transformer_block_names = _detect_block_names(model)

    all_blocks: List[nn.Module] = []
    for name in transformer_block_names:
        block_list = model
        for part in name.split("."):
            block_list = getattr(block_list, part, None)
            if block_list is None:
                break
        if block_list is not None and hasattr(block_list, "__iter__"):
            all_blocks += list(block_list)

    if not all_blocks:
        logger.warning(
            f"No transformer blocks found via {transformer_block_names}. "
            "Quantizing entire model at once."
        )
        quantize(model, weights=quantization_type, exclude=exclude_modules)
        freeze(model)
        return

    logger.info(f"Quantizing {len(all_blocks)} transformer blocks one at a time...")

    for block in all_blocks:
        # Move single block to GPU
        block.to(device, dtype=dtype, non_blocking=True)

        # Quantize this block's weights
        quantize(block, weights=quantization_type)

        # Freeze quantized weights
        freeze(block)

        # Move back to CPU to free VRAM for the next block
        if low_vram:
            block.to("cpu", non_blocking=True)
            flush()

    # Quantize remaining (non-block) modules
    logger.info("Quantizing remaining modules...")
    block_module_names = []
    for name in transformer_block_names:
        # fnmatch needs a wildcard to exclude the block sub-modules
        # (e.g. "transformer_blocks.0.attn.to_q"); the bare name only
        # matches the ModuleList itself, not its children.
        block_module_names.append(name)
        block_module_names.append(name + ".*")
    quantize(
        model,
        weights=quantization_type,
        exclude=exclude_modules + block_module_names,
    )
    flush()


def _detect_block_names(model: nn.Module) -> List[str]:
    """Auto-detect transformer block attribute paths."""
    common_names = [
        "transformer_blocks",
        "blocks",
        "layers",
        "h",
        "model.transformer_blocks",
        "model.layers",
    ]
    found = []
    for name in common_names:
        obj = model
        for part in name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__iter__"):
            found.append(name)
    return found or ["transformer_blocks"]
