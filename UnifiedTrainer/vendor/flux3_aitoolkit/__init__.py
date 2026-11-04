# =============================================================================
# VENDORED COPIES + DERIVED CODE from ostris/ai-toolkit - read ATTRIBUTION.md.
#
# Original author : Ostris, LLC ("ostris") - AI Toolkit
# Upstream repo   : https://github.com/ostris/ai-toolkit
# Upstream commit : 12395a103fd0ca499c631dc6f460a8778e11490d (2026-10-15)
# Licence         : MIT - Copyright (c) 2024 Ostris, LLC
# Vendored        : 2026-10-16 into UnifiedTrainer (see md/12-aitoolkit-flux3-compare.md)
#
# This package is deliberately import-light: importing it must not pull
# torch/diffusers/einops just to read a constant. Import the submodule you need.
# =============================================================================
"""ai-toolkit's FLUX.3 Image Klein implementation, vendored for reference.

Nothing here is imported by ai-toolkit at runtime and nothing here imports
ai-toolkit: each module is self-contained and usable from UnifiedTrainer.

===========================  ==================================================
Module                       What it is
===========================  ==================================================
``positional``               verbatim: 4-axis (t,h,w,l) RoPE id packers
``checkpoints``              verbatim: BFL package / ComfyUI repack file names
``pipeline``                 verbatim: inference recipe + latent/cond packers
``image_vae``                verbatim: standalone FLUX.2/3 packed VAE
``text_encoder``             adapted: Qwen3-VL stack, Comfy single-file keys,
                             vision-tower drop, Conv3d->GEMM patch
``recipe``                   extracted: training/sampling constants
                             (timeshift alpha 6.93, reference sizing,
                             quantization excludes, LoRA prefix mapping)
``keymap``                   derived: native(BFL/Comfy) <-> diffusers key map,
                             fused-qkv split, output-adaLN shift/scale swap
===========================  ==================================================

Provenance, the upstream file behind each module, and the exact changes made are
listed in ``ATTRIBUTION.md``; the comparison itself is in
``md/12-aitoolkit-flux3-compare.md``. Verification:
``tests/test_flux3_aitoolkit_keymap.py``.
"""

__all__ = [
    "positional",
    "checkpoints",
    "pipeline",
    "image_vae",
    "text_encoder",
    "recipe",
    "keymap",
]
