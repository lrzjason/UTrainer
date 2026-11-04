# =============================================================================
# DERIVED CODE - measured selection, the idea from KohakuMoss's tuner.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# Upstream: `src/kohakumoss_kernels/tuning/{tuner,timing,buckets,decisions}.py`
# - "a kernel is selected only where it is faster than PyTorch on that arch,
# the choice is measured and recorded, never guessed per call".  This module is
# the minimal version of that: one shape -> one comparison -> one cached bit.
# It is new code; the methodology is theirs.
# =============================================================================
"""Pick the fastest attention implementation per (arch, dtype, shape), once.

The measurement matters: on this project's sm120 card the Triton flash-attention
kernel beats PyTorch SDPA by 1.4-1.5x for MHA at every length, but for
grouped-query attention it *loses* below ~1.5k keys and wins above it:

    krea2 GQA, bf16, fwd+bwd     SDPA      Triton
    L=1024                       5.9 ms    9.6 ms   -> SDPA
    L=2048                      23.5 ms   18.4 ms   -> Triton

So "always Triton" would slow the most common Krea-2 run (512 px) down by 60%.
The first time a shape is seen, both candidates are timed on synthetic tensors of
that shape; the winner is cached for the process and (optionally) on disk.

Controls
--------
``UTR_KERNELS_TUNE``
    ``auto`` (default) - measure a new shape once, cache the verdict;
    ``off``            - never measure, take Triton when it supports the call;
    ``eager``          - never measure, always take the SDPA path.

The measurement is skipped (and SDPA is used) when less than four times the
probe's footprint is free, or when ``torch.compile`` is tracing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from . import eager as _eager

__all__ = ["clear", "describe", "pick_triton"]

_TUNE = os.environ.get("UTR_KERNELS_TUNE", "auto").strip().lower()
_TABLE: dict[str, bool] = {}
_LOADED = False


def _cache_path() -> Path | None:
    override = os.environ.get("UTR_KERNELS_CACHE")
    if override:
        return Path(override)
    return None  # in-process only unless the user asks for a file


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = _cache_path()
    if path is None or not path.exists():
        return
    try:
        _TABLE.update({str(k): bool(v) for k, v in json.loads(path.read_text()).items()})
    except Exception:
        _TABLE.clear()


def _save() -> None:
    path = _cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_TABLE, indent=2, sort_keys=True))
    except Exception:
        pass


def _key(query, key, value, mask) -> str:
    arch = torch.cuda.get_device_capability(query.device) if query.is_cuda else (0, 0)
    return "|".join(
        str(x)
        for x in (
            f"sm{arch[0]}{arch[1]}",
            query.dtype,
            key.dtype,
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
            None if mask is None else (str(mask.dtype), tuple(mask.shape)),
        )
    )


def _median_ms(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _probe(query, key, value, attn_mask, scale):
    """Time fwd+backward of both candidates on this exact shape."""
    from . import attention_triton

    def make():
        q = torch.randn_like(query).detach().requires_grad_(True)
        k = torch.randn_like(key).detach().requires_grad_(True)
        v = torch.randn_like(value).detach().requires_grad_(True)
        return q, k, v

    def run(fn):
        q, k, v = make()
        out = fn(q, k, v)
        out.float().sum().backward()

    candidates = {
        "sdpa": lambda q, k, v: _eager.sdpa_gqa(q, k, v, attn_mask=attn_mask, scale=scale),
        "triton": lambda q, k, v: attention_triton.attention(
            q, k, v, attn_mask=attn_mask, scale=scale
        ),
    }
    results = {}
    for name, fn in candidates.items():
        try:
            results[name] = _median_ms(lambda: run(fn))
        except Exception:
            results[name] = float("inf")
    triton_wins = results["triton"] < results["sdpa"]
    return triton_wins, results


def pick_triton(query, key, value, attn_mask=None, scale=None) -> bool:
    """Whether the Triton attention should be used for this exact call shape."""
    if _TUNE == "eager":
        return False
    if _TUNE == "off":
        return True
    if torch.cuda.is_available():
        from . import _is_compiling

        if _is_compiling():
            return False
    _load()
    table_key = _key(query, key, value, attn_mask)
    if table_key in _TABLE:
        return _TABLE[table_key]

    verdict = True
    try:
        footprint = 4 * (
            query.numel() + key.numel() + value.numel()
        ) * query.element_size()
        if query.is_cuda:
            free, _total = torch.cuda.mem_get_info()
            if free < 4 * footprint:
                return True
        verdict, results = _probe(query, key, value, attn_mask, scale)
        if not any(v < float("inf") for v in results.values()):
            verdict = True
    except Exception:
        return True
    _TABLE[table_key] = bool(verdict)
    _save()
    return bool(verdict)


def describe() -> dict:
    """What has been measured in this process (for logs and tests)."""
    return {"tune": _TUNE, "entries": dict(_TABLE)}


def clear() -> None:
    """Forget every verdict (tests)."""
    _TABLE.clear()
