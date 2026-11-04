# =============================================================================
# DERIVED CODE - training-side glue for the official MOSS-Image transformer.
#
# Original author : OpenMOSS Team - openmoss/MOSS-Image-Base (ModelScope)
# Wrapper         : UnifiedTrainer, 2026-10-19
#
# The model is the vendored official file (see
# `vendor/openmoss_moss_image/ATTRIBUTION.md`).  This module adds what the
# inference-oriented original lacks, without editing it:
#
#   * `models/moss/forward.py` replaces `forward` (reference-latent spans on the
#     reserved `r` RoPE axis + quantization-safe dtype casts; identical math when
#     neither is used);
#   * gradient checkpointing per joint block, toggled through the usual diffusers
#     `enable_gradient_checkpointing()` entry point.
# =============================================================================
"""Wire the training forward and per-block gradient checkpointing into the model."""

from __future__ import annotations

import types

from UnifiedTrainer.models.moss.forward import install_reference_support

__all__ = ["install_training_hooks", "gradient_checkpointing_enabled"]


def install_training_hooks(model) -> None:
    """Bind the training forward, the block entry points and the checkpointing flags."""
    if getattr(model, "_utrainer_moss_hooks", False):
        return
    install_reference_support(model)

    def enable_gradient_checkpointing(self):
        model._moss_checkpoint_blocks = True

    def disable_gradient_checkpointing(self):
        model._moss_checkpoint_blocks = False

    model.enable_gradient_checkpointing = types.MethodType(enable_gradient_checkpointing, model)
    model.disable_gradient_checkpointing = types.MethodType(disable_gradient_checkpointing, model)
    model.gradient_checkpointing_enabled = types.MethodType(
        lambda self: bool(getattr(model, "_moss_checkpoint_blocks", False)), model
    )
    # diffusers' ModelMixin checks this flag before calling the method above.
    model._supports_gradient_checkpointing = True
    model._utrainer_moss_hooks = True


def gradient_checkpointing_enabled(model) -> bool:
    """Whether the blocks are currently checkpointed."""
    return bool(getattr(model, "_moss_checkpoint_blocks", False))
