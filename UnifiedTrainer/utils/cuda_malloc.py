# Reference: adapted from ai-toolkit/toolkit/cuda_malloc.py
# See aitoolkit_acc.md for full technique analysis.
"""
CUDA Malloc Async -enable the async CUDA allocator *before* PyTorch is imported.

Setting ``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`` reduces memory
fragmentation on NVIDIA GPUs (Kepler+ / driver -515).  The env var **must**
be set before the first CUDA operation, so this module should be imported
very early -ideally as the first import in ``train.py``.

Usage::

    from UnifiedTrainer.utils import cuda_malloc  # side-effect import

Or explicitly::

    from UnifiedTrainer.utils.cuda_malloc import enable_cuda_malloc_async
    enable_cuda_malloc_async()
"""
from __future__ import annotations

import importlib.util
import os


# ── GPU name detection (pre-torch) ──────────────────────────────────────

def _get_gpu_names() -> set[str]:
    """Return display-adapter names using the OS API (no torch dependency)."""
    if os.name == "nt":
        import ctypes

        class _DISPLAY_DEVICEA(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("DeviceName", ctypes.c_char * 32),
                ("DeviceString", ctypes.c_char * 128),
                ("StateFlags", ctypes.c_ulong),
                ("DeviceID", ctypes.c_char * 128),
                ("DeviceKey", ctypes.c_char * 128),
            ]

        user32 = ctypes.windll.user32

        def _enum() -> set[str]:
            info = _DISPLAY_DEVICEA()
            info.cb = ctypes.sizeof(info)
            idx = 0
            names: set[str] = set()
            while user32.EnumDisplayDevicesA(None, idx, ctypes.byref(info), 0):
                idx += 1
                names.add(info.DeviceString.decode("utf-8", errors="replace"))
            return names

        return _enum()
    else:
        return set()


# GPUs known to crash or corrupt with cudaMallocAsync.
_BLACKLIST = {
    "GeForce GTX TITAN X", "GeForce GTX 980", "GeForce GTX 970",
    "GeForce GTX 960", "GeForce GTX 950",
    "GeForce 945M", "GeForce 940M", "GeForce 930M", "GeForce 920M",
    "GeForce 910M", "GeForce GTX 750", "GeForce GTX 745",
    "Quadro K620", "Quadro K1200", "Quadro K2200",
    "Quadro M500", "Quadro M520", "Quadro M600", "Quadro M620",
    "Quadro M1000", "Quadro M1200", "Quadro M2000", "Quadro M2200",
    "Quadro M3000", "Quadro M4000", "Quadro M5000", "Quadro M5500",
    "Quadro M6000",
    "GeForce MX110", "GeForce MX130",
    "GeForce 830M", "GeForce 840M",
    "GeForce GTX 850M", "GeForce GTX 860M",
    "GeForce GTX 1650", "GeForce GTX 1630",
}


def cuda_malloc_supported() -> bool:
    """Check whether the current GPU supports the async allocator."""
    try:
        names = _get_gpu_names()
    except Exception:
        return False
    for name in names:
        if "NVIDIA" in name:
            for b in _BLACKLIST:
                if b in name:
                    return False
    return bool(names)  # need at least one adapter


# ── Public API ──────────────────────────────────────────────────────────

_enabled = False


def enable_cuda_malloc_async(force: bool = False) -> bool:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` if the GPU is compatible.

    Returns True if enabled (or already enabled).
    """
    global _enabled
    if _enabled and not force:
        return True

    if not force and not cuda_malloc_supported():
        return False

    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)
    if existing is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    elif "cudaMallocAsync" not in existing:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            existing + ",backend:cudaMallocAsync"
        )

    _enabled = True
    return True


# ── Auto-enable on import ───────────────────────────────────────────────
# Mirror AIToolkit behaviour: enable immediately when this module is imported
# (which should happen before ``import torch`` in the entry point).
enable_cuda_malloc_async()
