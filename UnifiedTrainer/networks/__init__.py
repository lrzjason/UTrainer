"""Network adapter utilities for UnifiedTrainer.

Provides LoKR (Kronecker-product) adapter support via a self-contained,
VRAM-optimized implementation (no third-party lycoris-lora dependency).

Usage in config JSON:
    {"training": {"network_type": "lokr", "lokr_factor": -1, ...}}
    {"training": {"network_type": "lokr", "lokr_full_rank": true, ...}}
    # lokr_full_rank: musubi-compatible full-rank LoKr. Forces full-matrix
    # W1/W2 and overrides rank/alpha to the 9999 sentinel (scale = 1.0).

`lokr_model_type` presets (or pass `lokr_target_modules` fnmatch patterns):
    krea2         -> None -> ALL Linear modules (musubi krea2 convention)
    qwen          -> _QWEN_PATTERNS (qwen_image / qwen_image_edit)
    flux2_klein   -> _FLUX2_KLEIN_PATTERNS (flux2_klein 4B)
    flux          -> _FLUX_PATTERNS (original FLUX naming)
    minimax_h3    -> _H3_PATTERNS
    <unknown>     -> falls back to _KREA2_PATTERNS list
"""
from UnifiedTrainer.networks.lokr_module import (
    LokrConfig,
    LokrLayer,
    LokrNetwork,
    apply_lokr,
    factorization,
)

__all__ = ["LokrConfig", "LokrLayer", "LokrNetwork", "apply_lokr", "factorization"]
