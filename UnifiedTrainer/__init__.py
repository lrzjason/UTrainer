"""
UnifiedTrainer - A modular, protocol-based training framework for image/video diffusion models.

Design principle:
    Core training logic (loop, optimizer, checkpointing) is SHARED across all models.
    Model-specific behaviors (data representation, forward pass, noise schedule, loss
    computation) are ISOLATED into interchangeable protocol-based backends.

Usage:
    from UnifiedTrainer.registry import ModelRegistry, LossRegistry
    from UnifiedTrainer.engine.trainer import Trainer
"""

__version__ = "0.1.0"
