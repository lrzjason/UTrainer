# Reference: adapted from ai-toolkit/toolkit/memory_management/manager.py and manager_modules.py
"""Bouncing weights offload -pipelined layer offloading for low-VRAM training."""
from UnifiedTrainer.utils.memory_management.manager import MemoryManager
from UnifiedTrainer.utils.memory_management.manager_modules import (
    PIPELINE_DEPTH,
    _DEVICE_STATE,
)

__all__ = ["MemoryManager", "PIPELINE_DEPTH", "_DEVICE_STATE"]
