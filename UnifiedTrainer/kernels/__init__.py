# =============================================================================
# DERIVED CODE - kernel techniques learned from KohakuMoss.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# Design taken from that repository's kernel package (`src/kohakumoss_kernels`):
#   * one normative eager implementation per op, every other backend equals it
#     within the precision bar (`backends/eager/*`);
#   * the backend for an op is a build-time choice, never a branch inside the
#     per-step path (`registry.py` -> `Bound op`);
#   * a backend that cannot run on this arch/dtype/shape degrades to the next
#     candidate instead of raising (`errors.py`, the `RuntimeError: next
#     backend` chain).
# The ops themselves are re-expressed for UnifiedTrainer's layouts.
# See kernels/ATTRIBUTION.md, md/13-kohakumoss-kernels.md,
# md/14-kernel-optimizations-applied.md.
# =============================================================================
"""Fused attention and block ops for UnifiedTrainer's diffusion transformers.

Why this package exists
-----------------------
The model adapters under ``UnifiedTrainer/models`` are diffusers-style
transformers: every block is a chain of small eager ops, and attention goes
through ``diffusers.models.attention_dispatch.dispatch_attention_fn``.  On the
Windows + Blackwell (sm120) target the fused SDPA kernels are unavailable, so
that path degrades badly: ``enable_gqa=True`` drops to the fp32 *math* backend,
which costs 2.8x a plain expanded-KV call (measured, see
``md/14-kernel-optimizations-applied.md``).  This package supplies

* :func:`attention` - a GQA-aware flash-attention (Triton, forward+backward)
  with an optional key-padding mask, plus a correct pure-torch fallback;
* :func:`rms_norm_modulate`, :func:`silu_mul`, :func:`rope_apply`,
  :func:`add_gated` - the block elementwise ops, fused;
* :func:`install_attention_backend` - one bind-time hook that routes *every*
  adapter's ``dispatch_attention_fn`` through :func:`attention`.

Selection
---------
``UTR_KERNELS`` (env var) or :func:`set_mode`:

===========  ==========================================================
``off``      eager fallbacks only (bit-exact baseline)
``auto``     Triton where it is supported on this device, else eager
``triton``   Triton only; an unsupported call raises (CI / benchmarks)
===========  ==========================================================

Nothing here imports another project's code at runtime: the Triton kernels are
local, and the eager ops are plain PyTorch.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from . import eager as _eager
from . import select as _select

__all__ = [
    "HAS_TRITON",
    "attention",
    "attention_backend_installed",
    "available",
    "add_gated",
    "mode",
    "rms_norm_modulate",
    "rope_apply",
    "set_mode",
    "silu_mul",
    "install_attention_backend",
    "uninstall_attention_backend",
]

_VALID_MODES = ("off", "auto", "triton")
_mode = os.environ.get("UTR_KERNELS", "auto").strip().lower()
if _mode not in _VALID_MODES:
    _mode = "auto"

try:  # Triton is optional: the framework must run without it.
    from . import attention_triton as _attn_triton
    from . import blockops_triton as _block_triton

    HAS_TRITON = bool(getattr(_attn_triton, "HAS_TRITON", False)) and bool(
        getattr(_block_triton, "HAS_TRITON", False)
    )
except Exception:  # pragma: no cover - missing triton, or a partial checkout
    _attn_triton = None
    _block_triton = None
    HAS_TRITON = False


def mode() -> str:
    """Current kernel mode: ``off`` | ``auto`` | ``triton``."""
    return _mode


def set_mode(new_mode: str) -> None:
    """Set the kernel mode for subsequent calls (see the module docstring)."""
    global _mode
    new_mode = str(new_mode).strip().lower()
    if new_mode not in _VALID_MODES:
        raise ValueError(f"UTR_KERNELS must be one of {_VALID_MODES}, got {new_mode!r}")
    _mode = new_mode  # the installed hook re-reads this on every call


def available() -> dict[str, Any]:
    """What this build can actually run - for logs and the doctor output."""
    return {
        "mode": _mode,
        "tune": _select.describe()["tune"],
        "triton": HAS_TRITON,
        "cuda": torch.cuda.is_available(),
        "attention_triton": _attn_triton is not None,
        "blockops_triton": _block_triton is not None,
    }


def _wants_triton() -> bool:
    return _mode != "off" and HAS_TRITON and torch.cuda.is_available()


# ── attention ────────────────────────────────────────────────────────────


def _is_compiling() -> bool:
    """``torch.compiler.is_compiling()`` without hard-depending on the namespace."""
    compiler = getattr(torch, "compiler", None)
    try:
        return bool(compiler is not None and compiler.is_compiling())
    except Exception:  # pragma: no cover - exotic torch builds
        return False


_WARNED: set[str] = set()


def _warn_once(topic: str, message: str) -> None:
    """One warning per topic per process: a fallback must not spam every step."""
    if topic in _WARNED:
        return
    _WARNED.add(topic)
    import warnings

    warnings.warn(f"[UnifiedTrainer.kernels] {message}", RuntimeWarning, stacklevel=3)


def _attention_triton_ok(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> bool:
    if _attn_triton is None or not _wants_triton():
        return False
    if dropout_p:
        return False
    checker = getattr(_attn_triton, "is_supported", None)
    if checker is None:
        return False
    return bool(checker(query, key, value, attn_mask, is_causal))


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **_ignored: Any,
) -> torch.Tensor:
    """Flash attention over ``(B, L, H, D)`` with grouped-query KV heads.

    ``query`` is ``(B, Lq, H, D)``, ``key``/``value`` are ``(B, Lk, Hkv, D)``
    with ``H`` a multiple of ``Hkv``.  ``attn_mask`` is ``None``, a boolean mask
    or an additive mask broadcastable to ``(B, H, Lq, Lk)``.  Returns
    ``(B, Lq, H, D)`` in ``query.dtype``.

    Falls back to :func:`eager.sdpa_gqa` whenever the Triton path does not
    support the call, so the result is defined for every input.
    """
    supported = _attention_triton_ok(query, key, value, attn_mask, dropout_p, is_causal)
    if not supported and _mode == "triton" and HAS_TRITON:
        raise RuntimeError(
            "UTR_KERNELS=triton but the Triton attention kernel does not support "
            f"this call: q={tuple(query.shape)} k={tuple(key.shape)} "
            f"dtype={query.dtype} device={query.device} "
            f"mask={None if attn_mask is None else tuple(attn_mask.shape)} "
            f"dropout={dropout_p} causal={is_causal}"
        )
    if supported:
        # Inside torch.compile the custom autograd function would graph-break or
        # be traced into something slower than inductor's own fusion; let the
        # compiler see plain SDPA (which still picks the memory-efficient kernel,
        # see eager.sdpa_gqa) and fuse around it.
        if not _is_compiling() and _select.pick_triton(query, key, value, attn_mask, scale):
            try:
                return _attn_triton.attention(
                    query, key, value, attn_mask=attn_mask, scale=scale, is_causal=is_causal
                )
            except Exception:
                if _mode == "triton":
                    raise
                _warn_once("attention", "Triton attention failed; using SDPA")
    return _eager.sdpa_gqa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


# ── block ops ────────────────────────────────────────────────────────────


def rms_norm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    eps: float = 1e-5,
    offset: float = 0.0,
) -> torch.Tensor:
    """``(1 + scale) * rms_norm(x, weight + offset) + shift``, fused.

    Equivalent to :func:`eager.rms_norm_modulate`; the Triton path runs the
    whole chain in one pass (fp32 statistics, one HBM write).
    """
    if _block_triton is not None and _wants_triton():
        ok = getattr(_block_triton, "is_supported_rms_norm_modulate", None)
        if ok is not None and ok(x, weight, scale, shift):
            try:
                return _block_triton.rms_norm_modulate(
                    x, weight=weight, scale=scale, shift=shift, eps=eps, offset=offset
                )
            except Exception:
                if _mode == "triton":
                    raise
                _warn_once("rms_norm_modulate", "Triton rms_norm_modulate failed; using eager")
    return _eager.rms_norm_modulate(
        x, weight=weight, scale=scale, shift=shift, eps=eps, offset=offset
    )


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` fused into one kernel."""
    if _block_triton is not None and _wants_triton():
        ok = getattr(_block_triton, "is_supported_silu_mul", None)
        if ok is not None and ok(gate, up):
            try:
                return _block_triton.silu_mul(gate, up)
            except Exception:
                if _mode == "triton":
                    raise
                _warn_once("silu_mul", "Triton silu_mul failed; using eager")
    return _eager.silu_mul(gate, up)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary embedding on ``(B, L, H, D)`` with ``cos``/``sin`` ``(L, D)``."""
    if _block_triton is not None and _wants_triton():
        ok = getattr(_block_triton, "is_supported_rope_apply", None)
        if ok is not None and ok(x, cos, sin):
            try:
                return _block_triton.rope_apply(x, cos, sin)
            except Exception:
                if _mode == "triton":
                    raise
                _warn_once("rope_apply", "Triton rope_apply failed; using eager")
    return _eager.rope_apply(x, cos, sin)


def add_gated(
    residual: torch.Tensor, gate: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """``residual + gate * value`` fused (the gated residual of a DiT block)."""
    if _block_triton is not None and _wants_triton():
        ok = getattr(_block_triton, "is_supported_add_gated", None)
        if ok is not None and ok(residual, gate, value):
            try:
                return _block_triton.add_gated(residual, gate, value)
            except Exception:
                if _mode == "triton":
                    raise
                _warn_once("add_gated", "Triton add_gated failed; using eager")
    return _eager.add_gated(residual, gate, value)


# ── the one hook that covers every adapter ───────────────────────────────

_INSTALLED = False
_ORIGINAL_DISPATCH = None
_PATCHED_MODULES: list[tuple[Any, Any]] = []


def _make_dispatch():
    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                 scale=None, enable_gqa=False, attention_kwargs=None, *,
                 backend=None, parallel_config=None, **_kw):
        # diffusers passes (B, L, H, D) and expects the same layout back.
        # Anything we do not model - an explicit backend request (e.g. "flex"),
        # a `flex_attention` BlockMask, a parallel config - goes to diffusers
        # untouched so the fused kernels cannot silently change semantics.
        mask_is_tensor = attn_mask is None or isinstance(attn_mask, torch.Tensor)
        if backend not in (None, "native", "sdpa") or not mask_is_tensor or parallel_config:
            return _ORIGINAL_DISPATCH(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
                attention_kwargs=attention_kwargs, backend=backend,
                parallel_config=parallel_config,
            )
        return attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale,
        )

    return dispatch


def install_attention_backend(force: bool = False) -> bool:
    """Route ``dispatch_attention_fn`` through :func:`attention`.

    Patches the definition site and every already-imported module that holds a
    ``dispatch_attention_fn`` reference (the model adapters do
    ``from diffusers.models.attention_dispatch import dispatch_attention_fn``,
    which binds the name at import time).  Idempotent; returns whether the hook
    is in place.
    """
    global _INSTALLED, _ORIGINAL_DISPATCH
    import sys
    import types

    try:
        from diffusers.models import attention_dispatch as _ad
    except Exception:
        return False

    if _INSTALLED:
        if not force:
            return True
        # Re-install cleanly: the recorded "original" of a second pass would be
        # the hook itself, and uninstall would then leave the hook in place.
        uninstall_attention_backend()
    if _ORIGINAL_DISPATCH is None:
        _ORIGINAL_DISPATCH = _ad.dispatch_attention_fn

    patched = _make_dispatch()
    setattr(patched, "__utrainer_hook__", True)
    for _name, module in list(sys.modules.items()):
        # `vars()` only: several torch/diffusers modules resolve attributes
        # dynamically (`torch.classes` builds a namespace for any name), and a
        # plain getattr would instantiate those.
        if not isinstance(module, types.ModuleType):
            continue
        current = vars(module).get("dispatch_attention_fn")
        if current is None:
            continue
        if current is _ORIGINAL_DISPATCH or getattr(current, "__utrainer_hook__", False):
            _PATCHED_MODULES.append((module, current))
            setattr(module, "dispatch_attention_fn", patched)
    _INSTALLED = True
    return True


def uninstall_attention_backend() -> None:
    """Undo :func:`install_attention_backend` (tests, and mode ``off``)."""
    global _INSTALLED
    for module, original in _PATCHED_MODULES:
        setattr(module, "dispatch_attention_fn", original)
    _PATCHED_MODULES.clear()
    if _ORIGINAL_DISPATCH is not None:
        from diffusers.models import attention_dispatch as _ad

        _ad.dispatch_attention_fn = _ORIGINAL_DISPATCH
    _INSTALLED = False


def attention_backend_installed() -> bool:
    """Whether :func:`install_attention_backend` has patched this process."""
    return _INSTALLED
