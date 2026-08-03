# Reference: adapted from ai-toolkit/toolkit/basic.py
# See aitoolkit_acc.md for full technique analysis.
"""
Centralized memory flush utility.

Call flush() after every large ``del`` to release VRAM promptly.
This is the single most habituated VRAM-reclamation pattern across
AI Toolkit — gc.collect() + empty_cache().
"""
from __future__ import annotations

import gc

import torch


def flush(garbage_collect: bool = True) -> None:
    """Release cached memory back to the allocator.

    Args:
        garbage_collect: If True, run ``gc.collect()`` before emptying caches.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if garbage_collect:
        gc.collect()
