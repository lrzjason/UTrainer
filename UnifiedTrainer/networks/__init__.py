"""Network adapter utilities for UnifiedTrainer.

Provides LoKR (Kronecker-product) adapter support via a self-contained,
VRAM-optimized implementation (no third-party lycoris-lora dependency).

Usage in config JSON:
    {"training": {"network_type": "lokr", "lokr_factor": -1, ...}}
    {"training": {"network_type": "lokr", "lokr_full_rank": true, ...}}
    # lokr_full_rank: musubi-compatible full-rank LoKr. Forces full-matrix
    # W1/W2 and overrides rank/alpha to the 9999 sentinel (scale = 1.0).
"""
from UnifiedTrainer.networks.lokr_module import (
    LokrConfig,
    LokrLayer,
    LokrNetwork,
    apply_lokr,
    factorization,
)

__all__ = ["LokrConfig", "LokrLayer", "LokrNetwork", "apply_lokr", "factorization"]
