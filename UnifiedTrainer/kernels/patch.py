# =============================================================================
# DERIVED CODE - bind-time patching in the style of KohakuMoss.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# The idea is theirs (`src/kohakumoss_kernels/registry.py`: a kernel is bound to
# an op once, at build time, and the per-step path calls a plain attribute - no
# `if backend == ...` inside the block).  Here the "binding" is a method swap on
# the model classes, installed once before the training loop and removed again
# on request.  The wrappers themselves are new code for UnifiedTrainer's blocks.
# =============================================================================
"""Bind the fused ops into the model blocks, once, at setup time.

The model adapters are diffusers-style: a block is a chain of small modules whose
``forward`` is called hundreds of times per step.  Replacing three of those
``forward`` methods at build time is enough to use the fused kernels everywhere
without touching the model code, and the replacement is reversible:

    handle = patch.install()      # after the transformer is loaded
    ...
    patch.remove(handle)          # tests, or training.kernels: off

What is patched today:

======================================  ==========================  ==============
target                                  replaced with               measured
======================================  ==========================  ==============
``Krea2RMSNorm.forward``                ``kernels.rms_norm_modulate``  4.87 -> 0.32 ms
``Krea2SwiGLU.forward``                 ``kernels.silu_mul``           1.77 -> 1.07 ms
``Flux2SwiGLU.forward``                 ``kernels.silu_mul``           (same op)
``apply_rotary_emb`` in adapter modules ``kernels.rope_apply``          4.84 -> 0.25 ms
======================================  ==========================  ==============

The numbers are one call at ``(1, 2048, 6144)`` / ``(1, 2048, 48, 128)`` bf16 on
an RTX 5060 Ti; see ``md/14-kernel-optimizations-applied.md``.

Under ``torch.compile`` every wrapper defers to the original implementation: a
custom autograd Function inside a compiled region would graph-break, and inductor
fuses these pointwise chains itself.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable

import torch

from . import _is_compiling
from . import rms_norm_modulate, rope_apply, silu_mul

__all__ = ["install", "remove", "installed"]

# (holder, attribute, original) - `holder` is a module for function swaps and a
# class for method swaps.  Restored in reverse order by `remove`.
_PATCHED: list[tuple[Any, str, Any]] = []


def installed() -> bool:
    """Whether the fused forwards are currently bound into the models."""
    return bool(_PATCHED)


def _set(holder: Any, name: str, value: Any) -> None:
    _PATCHED.append((holder, name, getattr(holder, name)))
    setattr(holder, name, value)


def _method(original: Callable, new_forward: Callable) -> Callable:
    """Wrap a bound method so ``torch.compile`` falls back to the original."""

    def wrapper(self, *args, **kwargs):
        if _is_compiling():
            return original(self, *args, **kwargs)
        return new_forward(self, original, *args, **kwargs)

    wrapper.__name__ = getattr(original, "__name__", "forward")
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper.__utrainer_patch__ = True
    return wrapper


# ── the replacements ────────────────────────────────────────────────────


def _krea2_rms_norm(self, original, hidden_states):
    # Same contract as Krea2RMSNorm.forward: fp32 statistics, scale = weight + 1.
    if hidden_states.device != self.weight.device:
        hidden_states = hidden_states.to(self.weight.device)
    return rms_norm_modulate(
        hidden_states, weight=self.weight, eps=self.eps, offset=1.0
    )


def _krea2_swiglu(self, original, hidden_states):
    # down(silu(gate(x)) * up(x)) - the fused kernel computes the middle term.
    return self.down(silu_mul(self.gate(hidden_states), self.up(hidden_states)))


def _flux2_swiglu(self, original, x):
    # Flux 2 fuses gate/up into one projection and splits it with chunk().
    x1, x2 = x.chunk(2, dim=-1)
    return silu_mul(x1, x2)


def _make_rope_shim(original):
    """Adapt ``diffusers.models.embeddings.apply_rotary_emb`` to the fused op.

    Only the layout ``rope_apply`` implements is taken over: ``sequence_dim=1``
    (sequence at dim 1), ``use_real``, ``use_real_unbind_dim=-1`` and a
    ``(cos, sin)`` tuple with 2-D tables.  Everything else - including the
    4-D-table / ``sequence_dim=2`` form used by FLUX.3 - is delegated.
    """

    def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
        if (
            use_real
            and use_real_unbind_dim == -1
            and sequence_dim == 1
            and isinstance(freqs_cis, tuple)
            and len(freqs_cis) == 2
            and x.ndim == 4
            and freqs_cis[0].ndim == 2
            and not _is_compiling()
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            return rope_apply(x, freqs_cis[0], freqs_cis[1])
        return original(
            x, freqs_cis, use_real=use_real,
            use_real_unbind_dim=use_real_unbind_dim, sequence_dim=sequence_dim,
        )

    apply_rotary_emb.__name__ = "apply_rotary_emb"
    apply_rotary_emb.__utrainer_patch__ = True
    return apply_rotary_emb


# ── install / remove ────────────────────────────────────────────────────


def install() -> bool:
    """Bind the fused forwards into every already-imported model module.

    Idempotent; returns whether anything is bound.
    """
    if _PATCHED:
        return True

    # 1. Krea-2: the custom RMSNorm and the SwiGLU of every block.
    krea2 = sys.modules.get("UnifiedTrainer.models.krea2.transformer_krea2")
    if krea2 is not None:
        for cls_name, new in (
            ("Krea2RMSNorm", _krea2_rms_norm),
            ("Krea2SwiGLU", _krea2_swiglu),
        ):
            cls = getattr(krea2, cls_name, None)
            if cls is not None and not getattr(cls.forward, "__utrainer_patch__", False):
                _set(cls, "forward", _method(cls.forward, new))

    # 2. Flux 2: the SwiGLU that splits a fused projection.
    flux2 = sys.modules.get("UnifiedTrainer.models.flux2_klein.transformer_flux2")
    if flux2 is not None:
        cls = getattr(flux2, "Flux2SwiGLU", None)
        if cls is not None and not getattr(cls.forward, "__utrainer_patch__", False):
            _set(cls, "forward", _method(cls.forward, _flux2_swiglu))

    # 3. Rotary embeddings: every adapter module that imported the diffusers
    #    helper by name (that binding is what its blocks call).
    for name, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType):
            continue
        if not name.startswith("UnifiedTrainer.models."):
            continue
        original = vars(module).get("apply_rotary_emb")
        if original is None or getattr(original, "__utrainer_patch__", False):
            continue
        _set(module, "apply_rotary_emb", _make_rope_shim(original))

    return bool(_PATCHED)


def remove() -> None:
    """Restore every patched attribute (safe to call when nothing is patched)."""
    while _PATCHED:
        holder, name, original = _PATCHED.pop()
        setattr(holder, name, original)
