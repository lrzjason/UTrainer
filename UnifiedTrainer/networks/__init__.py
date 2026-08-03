"""Network adapter utilities for UnifiedTrainer.

Provides LoKR (Kronecker-product) adapter support via a self-contained,
VRAM-optimized implementation (no third-party lycoris-lora dependency).

Usage in config JSON:
    {"training": {"network_type": "lokr", "lokr_factor": 4, ...}}
"""
from UnifiedTrainer.networks.lokr_module import (
    LokrConfig,
    LokrLayer,
    LokrNetwork,
    apply_lokr,
    factorization,
)

__all__ = ["LokrConfig", "LokrLayer", "LokrNetwork", "apply_lokr", "factorization"]
