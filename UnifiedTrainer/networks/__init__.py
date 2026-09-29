"""Network adapter utilities for UnifiedTrainer.

Provides LoKR (Kronecker-product) adapter support via a self-contained,
VRAM-optimized implementation (no third-party lycoris-lora dependency).

Usage in config JSON:
    {"training": {"network_type": "lokr", "lokr_factor": -1, ...}}
    {"training": {"network_type": "lokr", "lokr_full_rank": true, ...}}
    # lokr_full_rank: musubi-compatible full-rank LoKr. Forces full-matrix
    # W1/W2 and overrides rank/alpha to the 9999 sentinel (scale = 1.0).

Targeting: `lokr_target_modules=null` (default) → ALL nn.Linear modules get
adapters (musubi krea2 convention, now the default for every model_type).
An explicit list of fnmatch patterns restricts targeting. The per-model
preset lists (`_QWEN_PATTERNS`, `_QWEN21_PATTERNS`, `_FLUX_PATTERNS`,
`_FLUX2_KLEIN_PATTERNS`, `_H3_PATTERNS`) are reference-only — copy one into
`lokr_target_modules` for a restricted set.
"""
from UnifiedTrainer.networks.lokr_module import (
    LokrConfig,
    LokrLayer,
    LokrNetwork,
    apply_lokr,
    factorization,
)

__all__ = ["LokrConfig", "LokrLayer", "LokrNetwork", "apply_lokr", "factorization"]
