# =============================================================================
# DERIVED CODE - kernel techniques learned from KohakuMoss.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Derived from    : src/kohakumoss_kernels/backends/triton/attention/_kernels.py
#                   (and .../triton/attention/varlen.py for the launch frame and
#                   the fused-vs-split backward decision)
# Re-implemented for UnifiedTrainer's dense (B, L, H, D) layout.
# Date            : 2026-10-19
#
# Their ideas, kept and re-expressed for the dense layout:
#   * the two-plane row statistics: the raw score max `m` and `log2(l)` as two
#     fp32 planes instead of one fused LSE, so a probability is
#     `exp2((s - m) * c - log2 l)` with `c = scale * log2(e)` and the largest
#     probability in a row is exactly 1;
#   * shift-before-scale, and `scale` left on the fp32 accumulators: `dS` is
#     rounded to 16 bits *unscaled* and `scale` is applied once to the fp32 dQ /
#     dK accumulators, so `delta = rowsum(dO * O)` cancels the `p * delta` term;
#   * 16-bit MMA operands with fp32 accumulation, P and dS each rounded once,
#     `delta` taken from the 16-bit O;
#   * the full-tile loop plus one masked tail tile instead of masking every
#     tile (their `MASKED` constexpr; `PAD` here, because the attention mask is
#     a second, independent thing);
#   * the fused backward: one program per key block that emits dK/dV and adds
#     its dQ partials into an fp32 buffer with `tl.atomic_add`, the `dq_cast`
#     epilogue that applies `scale` and casts, and `delta` in its own pass;
#   * the split backward (separate dkdv and dq kernels, no atomics) and the
#     notion of choosing between the two strategies;
#   * row-major `[B, H, L]` statistic planes, the transposed `qT = [D, BM]` load
#     in the dK/dV kernel, and int64 for the segment-start token offset.
#
# New here, not in the reference:
#   * grouped-query heads resolved inside the kernel: `kv_head = h // G` in the
#     forward and dQ kernels, and a static loop over the `G` query heads of one
#     KV head in the dK/dV kernels, so expanded K/V is never materialised (the
#     reference has no GQA; its packed layout keeps H == Hkv);
#   * the attention mask - bool or additive, broadcast to (B, H, Lq, Lk) - with
#     a constexpr no-mask variant, a 1-D key-column variant (the real call
#     site's (B, 1, 1, Lk) padding mask) and a general 2-D variant;
#   * the empty-row guard (`m` clamped away from -inf, the row sum floored, the
#     final divide guarded) so a row with no visible key yields 0, not NaN; the
#     reference never needs it because its only mask is the length tail;
#   * `torch.autograd.Function` with the mask replayed in the backward, the
#     fused-vs-split policy keyed on the fp32 dQ buffer size, the tile heuristic
#     (there is no tuning harness here) and the benchmark harness.
#
# See md/13-kohakumoss-kernels.md 3 for the distilled design notes.
# =============================================================================
"""Triton flash attention (forward + backward) over UnifiedTrainer's (B, L, H, D).

``attention`` is a drop-in replacement for
:func:`UnifiedTrainer.kernels.eager.sdpa_gqa` on the diffusion transformers'
shapes: ``query`` is ``(B, Lq, H, D)``, ``key``/``value`` are ``(B, Lk, Hkv, D)``
with ``H`` a multiple of ``Hkv`` (grouped-query), ``D`` in ``{64, 128}``,
``bf16``/``fp16`` in, fp32 accumulation, ``(B, Lq, H, D)`` out in
``query.dtype``.

What is fused
-------------
* forward: one program per (query tile, head, batch); online softmax with the
  two fp32 row planes (``m``, ``log2 l``) kept for the backward;
* backward: recomputation-based (q/k/v, ``O`` and the row planes are saved, the
  probability matrix never is), the GQA KV head resolved in the kernel, dQ
  accumulated in fp32 and cast once at the end.

The no-mask path carries no mask work at all (``MASKED`` is a ``constexpr``);
the grouped-query path never expands K/V (the KV head is ``h // (H // Hkv)``).

Unsupported - ``is_supported`` returns ``False`` and ``attention`` raises:
fp32 and other dtypes, ``D`` outside ``{64, 128}``, ``is_causal=True``,
non-CUDA tensors, empty sequences, a mask that requires grad or is not
broadcastable to ``(B, H, Lq, Lk)``, and any tensor whose token offsets would
exceed 2**31 elements (see ``_max_offset``).  The attention mask is a constant.

The fused backward accumulates dQ with ``tl.atomic_add``, so its summation
order across key blocks is not fixed; ``_TILE_OVERRIDE["bwd_impl"] = "split"``
selects the deterministic two-kernel path instead.
"""

from __future__ import annotations

import torch

try:  # Triton is optional: importing this module must never fail.
    import triton
    import triton.language as tl

    HAS_TRITON = True
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - no triton in this environment
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False
    _TRITON_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

__all__ = ["HAS_TRITON", "attention", "benchmark_cases", "is_supported"]

_LOG2E = 1.4426950408889634
_HEAD_DIMS = (64, 128)
_DTYPES = (torch.bfloat16, torch.float16)
#: An empty row's score max: large and negative, so it can never become a stale
#: reference for a later tile (see ``_fwd_tile``).
_NEG_BIG = -1e30
#: Floor of the softmax row sum, so ``log2`` of a fully masked row is finite.
_L_FLOOR = 1e-30
#: Above this size the fused backward's fp32 dQ buffer costs more (zero + cast
#: traffic) than the split path's two extra GEMMs.
_FUSED_MAX_BYTES = 512 << 20

# Tile table: (BM, BN, num_warps, num_stages) per phase and head dim, measured
# on the RTX 5060 Ti (sm120) at B=1, L=2048, H=48/12; see benchmark_cases.
_TILES: dict[str, dict[int, tuple[int, int, int, int]]] = {
    "fwd": {64: (128, 64, 4, 2), 128: (128, 64, 8, 2)},
    "bwd": {64: (64, 64, 4, 1), 128: (64, 64, 8, 1)},
    "dkdv": {64: (32, 64, 4, 2), 128: (64, 64, 8, 1)},
    "dq": {64: (128, 32, 4, 2), 128: (128, 32, 8, 1)},
}

#: Benchmark / tuning hook: a phase name ("fwd", "bwd", "dkdv", "dq") mapped to
#: an explicit (BM, BN, num_warps, num_stages), or "bwd_impl" to "fused"/"split".
#: Empty in production.
_TILE_OVERRIDE: dict[str, object] = {}


# ── Triton kernels ───────────────────────────────────────────────────────
if HAS_TRITON:

    @triton.jit
    def _mask_tile(MP, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                   T: tl.constexpr, MASK_COL: tl.constexpr):  # fmt: skip
        """One attention-mask tile, laid out like the score tile of its caller.

        ``T`` is the transpose flag: ``False`` gives ``[BM, BN]`` (query-major;
        the forward and dQ kernels), ``True`` gives ``[BN, BM]`` (key-major; the
        dK/dV kernels).  ``MASK_COL`` loads the key-column vector instead of a
        plane and broadcasts it over the query axis.
        """
        if MASK_COL:
            v = tl.load(MP + offs_n * sm_n, mask=cols, other=0)
            if T:
                return v[:, None]
            else:
                return v[None, :]
        else:
            # Rows past the query length are clamped; their results are dropped.
            mm = tl.minimum(offs_m, Lq - 1)
            if T:
                return tl.load(
                    MP + offs_n[:, None] * sm_n + mm[None, :] * sm_m,
                    mask=cols[:, None], other=0,
                )  # fmt: skip
            else:
                return tl.load(
                    MP + mm[:, None] * sm_m + offs_n[None, :] * sm_n,
                    mask=cols[None, :], other=0,
                )  # fmt: skip

    @triton.jit
    def _apply_mask(s, mb, MASKED: tl.constexpr, MASK_BOOL: tl.constexpr):
        """Add an additive mask to raw scores, or send masked keys to -inf.

        An additive mask is often written with a finite extreme instead of
        ``-inf`` (``torch.finfo(dtype).min`` and friends, as HuggingFace does).
        That value is not a mask: ``s + finfo.min`` stays finite, so the row max
        and the row sum disagree by ~1e38 and the backward lands on ``inf -
        inf``.  Any weight at or below ``-1e30`` is therefore treated as a hard
        mask, which is what the caller meant and what the bool path does; real
        additive biases (-5, -1e9) are untouched.
        """
        if MASKED:
            if MASK_BOOL:
                # `mb != 0`: triton's bool masks arrive as uint8, and `tl.where`
                # with a non-boolean condition is deprecated (future error).
                s = tl.where(mb != 0, s, float("-inf"))
            else:
                mb32 = mb.to(tl.float32)
                s = tl.where(mb32 <= -1e30, float("-inf"), s + mb32)
        return s

    @triton.jit
    def _fwd_tile(acc, m_i, l_i, q, k_ptrs, v_ptrs, mb, cols, c,
                  PAD: tl.constexpr, MASKED: tl.constexpr,
                  MASK_BOOL: tl.constexpr):  # fmt: skip
        """One key tile of the online softmax; returns (acc, m, l)."""
        if PAD:
            k = tl.load(k_ptrs, mask=cols[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=cols[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        s = tl.dot(q, tl.trans(k))
        s = _apply_mask(s, mb, MASKED, MASK_BOOL)
        if PAD:
            s = tl.where(cols[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # A row with no visible key has max -inf; pin it to a finite value so it
        # is never a stale reference for a later tile (which would NaN alpha/p).
        m_new = tl.where(m_new == float("-inf"), -1e30, m_new)
        alpha = tl.exp2((m_i - m_new) * c)
        p = tl.exp2((s - m_new[:, None]) * c)
        acc = tl.dot(p.to(k.dtype), v, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        return acc, m_new, l_i

    @triton.jit
    def _attn_fwd_kernel(
        Q, K, V, MASK, OUT, MST, LST,
        q_b, q_l, q_h, k_b, k_l, k_h, v_b, v_l, v_h, o_b, o_l, o_h,
        sm_b, sm_h, sm_m, sm_n, st_b, st_h,
        c, Lq, Lk,
        D: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        MASKED: tl.constexpr, MASK_COL: tl.constexpr, MASK_BOOL: tl.constexpr,
    ):  # fmt: skip
        """Grid (query tiles, H, B): O, the raw row max and log2(row sum)."""
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        b = tl.program_id(2)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        rows = offs_m < Lq
        q = tl.load(
            Q + b * q_b + h * q_h + offs_m[:, None] * q_l + offs_d[None, :],
            mask=rows[:, None], other=0.0,
        )  # fmt: skip
        kv_h = h // G
        k_base = K + b * k_b + kv_h * k_h
        v_base = V + b * v_b + kv_h * v_h
        if MASKED:
            mask_base = MASK + b * sm_b + h * sm_h
        else:
            mask_base = MASK
        acc = tl.zeros([BM, D], tl.float32)
        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        full = Lk // BN * BN
        for n0 in range(0, full, BN):
            offs_n = n0 + tl.arange(0, BN)
            cols = offs_n < Lk
            if MASKED:
                mb = _mask_tile(mask_base, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                                False, MASK_COL)  # fmt: skip
            else:
                mb = tl.zeros([1, 1], tl.float32)
            acc, m_i, l_i = _fwd_tile(
                acc, m_i, l_i, q,
                k_base + offs_n[:, None] * k_l + offs_d[None, :],
                v_base + offs_n[:, None] * v_l + offs_d[None, :],
                mb, cols, c, False, MASKED, MASK_BOOL,
            )  # fmt: skip
        if full < Lk:
            offs_n = full + tl.arange(0, BN)
            cols = offs_n < Lk
            if MASKED:
                mb = _mask_tile(mask_base, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                                False, MASK_COL)  # fmt: skip
            else:
                mb = tl.zeros([1, 1], tl.float32)
            acc, m_i, l_i = _fwd_tile(
                acc, m_i, l_i, q,
                k_base + offs_n[:, None] * k_l + offs_d[None, :],
                v_base + offs_n[:, None] * v_l + offs_d[None, :],
                mb, cols, c, True, MASKED, MASK_BOOL,
            )  # fmt: skip
        l_safe = tl.maximum(l_i, 1e-30)
        o = acc / l_safe[:, None]
        st = b * st_b + h * st_h + offs_m
        tl.store(LST + st, tl.log2(l_safe), mask=rows)
        tl.store(MST + st, m_i, mask=rows)
        tl.store(
            OUT + b * o_b + h * o_h + offs_m[:, None] * o_l + offs_d[None, :],
            o.to(OUT.dtype.element_ty), mask=rows[:, None],
        )  # fmt: skip

    @triton.jit
    def _attn_delta_kernel(
        OUT, DO, DELTA, o_b, o_l, o_h, d_b, d_l, d_h, de_b, de_h, Lq,
        D: tl.constexpr, BM: tl.constexpr,
    ):  # fmt: skip
        """Grid (query tiles, H, B): DELTA = rowsum(O * dO), fp32."""
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        b = tl.program_id(2)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        rows = offs_m < Lq
        o = tl.load(
            OUT + b * o_b + h * o_h + offs_m[:, None] * o_l + offs_d[None, :],
            mask=rows[:, None], other=0.0,
        ).to(tl.float32)  # fmt: skip
        do = tl.load(
            DO + b * d_b + h * d_h + offs_m[:, None] * d_l + offs_d[None, :],
            mask=rows[:, None], other=0.0,
        ).to(tl.float32)  # fmt: skip
        tl.store(DELTA + b * de_b + h * de_h + offs_m, tl.sum(o * do, 1), mask=rows)

    @triton.jit
    def _bwd_tile(dk, dv, k, v, qt_ptrs, do_ptrs, dqa_ptrs,
                  m_ptr, l_ptr, de_ptr, mb, qrow, cols, c,
                  PAD: tl.constexpr, MASKED: tl.constexpr, MASK_BOOL: tl.constexpr,
                  ATOMIC: tl.constexpr):  # fmt: skip
        """One query tile against the resident key tile: dK, dV and the dQ part."""
        if PAD:
            qt = tl.load(qt_ptrs, mask=qrow[None, :], other=0.0)
            do = tl.load(do_ptrs, mask=qrow[:, None], other=0.0)
            m = tl.load(m_ptr, mask=qrow, other=0.0)
            lg = tl.load(l_ptr, mask=qrow, other=0.0)
            delta = tl.load(de_ptr, mask=qrow, other=0.0)
        else:
            qt = tl.load(qt_ptrs)
            do = tl.load(do_ptrs)
            m = tl.load(m_ptr)
            lg = tl.load(l_ptr)
            delta = tl.load(de_ptr)
        s = tl.dot(k, qt)
        s = _apply_mask(s, mb, MASKED, MASK_BOOL)
        if PAD:
            s = tl.where(cols[:, None], s, float("-inf"))
        p = tl.exp2((s - m[None, :]) * c - lg[None, :])
        # A row with no visible key has its row sum floored (lg = log2(1e-30)
        # ~ -99.7) and therefore no weight at all; pin it to zero so that the
        # recomputed exponent cannot leak a stale +-inf into dS / dQ.
        p = tl.where(lg[None, :] > -1.0, p, 0.0)
        pt = p.to(k.dtype)
        dv = tl.dot(pt, do, dv)
        dp = tl.dot(v, tl.trans(do))
        ds = (pt * (dp - delta[None, :])).to(k.dtype)
        if tl.min(m) < -1e29:
            tl.device_print("dbg", tl.min(s), tl.min(p), tl.max(p), tl.min(ds),
                            tl.max(ds), tl.min(lg), tl.min(m))
        dk = tl.dot(ds, tl.trans(qt), dk)
        if ATOMIC:
            part = tl.dot(tl.trans(ds), k)
            if PAD:
                tl.atomic_add(dqa_ptrs, part, mask=qrow[:, None], sem="relaxed")
            else:
                tl.atomic_add(dqa_ptrs, part, sem="relaxed")
        return dk, dv

    @triton.jit
    def _attn_bwd_dkdv_kernel(
        Q, K, V, DO, MASK, MST, LST, DELTA, DK, DV, DQACC,
        q_b, q_l, q_h, k_b, k_l, k_h, v_b, v_l, v_h, d_b, d_l, d_h,
        g_b, g_l, g_h, sm_b, sm_h, sm_m, sm_n, st_b, st_h, dq_b, dq_l, dq_h,
        c, scale, Lq, Lk,
        D: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        MASKED: tl.constexpr, MASK_COL: tl.constexpr, MASK_BOOL: tl.constexpr,
        ATOMIC: tl.constexpr,
    ):  # fmt: skip
        """Grid (key tiles, Hkv, B): dK/dV of one KV head over its G query heads.

        With ``ATOMIC`` the program also emits this key tile's dQ partials (the
        fused backward); without it dQ comes from ``_attn_dq_kernel``.
        """
        pid_n = tl.program_id(0)
        hkv = tl.program_id(1)
        b = tl.program_id(2)
        offs_n = pid_n * BN + tl.arange(0, BN)
        cols = offs_n < Lk
        offs_d = tl.arange(0, D)
        k = tl.load(
            K + b * k_b + hkv * k_h + offs_n[:, None] * k_l + offs_d[None, :],
            mask=cols[:, None], other=0.0,
        )  # fmt: skip
        v = tl.load(
            V + b * v_b + hkv * v_h + offs_n[:, None] * v_l + offs_d[None, :],
            mask=cols[:, None], other=0.0,
        )  # fmt: skip
        dk = tl.zeros([BN, D], tl.float32)
        dv = tl.zeros([BN, D], tl.float32)
        full = Lq // BM * BM
        for g in tl.static_range(G):
            h = hkv * G + g
            q_base = Q + b * q_b + h * q_h
            do_base = DO + b * d_b + h * d_h
            st = b * st_b + h * st_h
            dqa_base = DQACC + b * dq_b + h * dq_h
            if MASKED:
                mask_base = MASK + b * sm_b + h * sm_h
            else:
                mask_base = MASK
            for m0 in range(0, full, BM):
                offs_m = m0 + tl.arange(0, BM)
                qrow = offs_m < Lq
                if MASKED:
                    mb = _mask_tile(mask_base, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                                    True, MASK_COL)  # fmt: skip
                else:
                    mb = tl.zeros([1, 1], tl.float32)
                dk, dv = _bwd_tile(
                    dk, dv, k, v,
                    q_base + offs_m[None, :] * q_l + offs_d[:, None],
                    do_base + offs_m[:, None] * d_l + offs_d[None, :],
                    dqa_base + offs_m[:, None] * dq_l + offs_d[None, :],
                    MST + st + offs_m, LST + st + offs_m, DELTA + st + offs_m,
                    mb, qrow, cols, c, False, MASKED, MASK_BOOL, ATOMIC,
                )  # fmt: skip
            if full < Lq:
                offs_m = full + tl.arange(0, BM)
                qrow = offs_m < Lq
                if MASKED:
                    mb = _mask_tile(mask_base, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                                    True, MASK_COL)  # fmt: skip
                else:
                    mb = tl.zeros([1, 1], tl.float32)
                dk, dv = _bwd_tile(
                    dk, dv, k, v,
                    q_base + offs_m[None, :] * q_l + offs_d[:, None],
                    do_base + offs_m[:, None] * d_l + offs_d[None, :],
                    dqa_base + offs_m[:, None] * dq_l + offs_d[None, :],
                    MST + st + offs_m, LST + st + offs_m, DELTA + st + offs_m,
                    mb, qrow, cols, c, True, MASKED, MASK_BOOL, ATOMIC,
                )  # fmt: skip
        dk = dk * scale
        out = b * g_b + hkv * g_h + offs_n[:, None] * g_l + offs_d[None, :]
        tl.store(DK + out, dk.to(DK.dtype.element_ty), mask=cols[:, None])
        tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=cols[:, None])

    @triton.jit
    def _dq_tile(dq, q, do, m, lg, delta, k_ptrs, v_ptrs, mb, cols, c,
                 PAD: tl.constexpr, MASKED: tl.constexpr,
                 MASK_BOOL: tl.constexpr):  # fmt: skip
        """One key tile of dQ = sum_n dS K for a resident query tile."""
        if PAD:
            k = tl.load(k_ptrs, mask=cols[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=cols[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        s = tl.dot(q, tl.trans(k))
        s = _apply_mask(s, mb, MASKED, MASK_BOOL)
        if PAD:
            s = tl.where(cols[None, :], s, float("-inf"))
        p = tl.exp2((s - m[:, None]) * c - lg[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p.to(k.dtype) * (dp - delta[:, None])).to(k.dtype)
        return tl.dot(ds, k, dq)

    @triton.jit
    def _attn_dq_kernel(
        Q, K, V, DO, MASK, MST, LST, DELTA, DQ,
        q_b, q_l, q_h, k_b, k_l, k_h, v_b, v_l, v_h, d_b, d_l, d_h,
        g_b, g_l, g_h, sm_b, sm_h, sm_m, sm_n, st_b, st_h,
        c, scale, Lq, Lk,
        D: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        MASKED: tl.constexpr, MASK_COL: tl.constexpr, MASK_BOOL: tl.constexpr,
    ):  # fmt: skip
        """Grid (query tiles, H, B): dQ of one query head over every key tile."""
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        b = tl.program_id(2)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        rows = offs_m < Lq
        q = tl.load(
            Q + b * q_b + h * q_h + offs_m[:, None] * q_l + offs_d[None, :],
            mask=rows[:, None], other=0.0,
        )  # fmt: skip
        do = tl.load(
            DO + b * d_b + h * d_h + offs_m[:, None] * d_l + offs_d[None, :],
            mask=rows[:, None], other=0.0,
        )  # fmt: skip
        st = b * st_b + h * st_h + offs_m
        m = tl.load(MST + st, mask=rows, other=0.0)
        lg = tl.load(LST + st, mask=rows, other=0.0)
        delta = tl.load(DELTA + st, mask=rows, other=0.0)
        kv_h = h // G
        k_base = K + b * k_b + kv_h * k_h
        v_base = V + b * v_b + kv_h * v_h
        if MASKED:
            mask_base = MASK + b * sm_b + h * sm_h
        else:
            mask_base = MASK
        dq = tl.zeros([BM, D], tl.float32)
        full = Lk // BN * BN
        for n0 in range(0, full, BN):
            offs_n = n0 + tl.arange(0, BN)
            cols = offs_n < Lk
            if MASKED:
                mb = _mask_tile(mask_base, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                                False, MASK_COL)  # fmt: skip
            else:
                mb = tl.zeros([1, 1], tl.float32)
            dq = _dq_tile(
                dq, q, do, m, lg, delta,
                k_base + offs_n[:, None] * k_l + offs_d[None, :],
                v_base + offs_n[:, None] * v_l + offs_d[None, :],
                mb, cols, c, False, MASKED, MASK_BOOL,
            )  # fmt: skip
        if full < Lk:
            offs_n = full + tl.arange(0, BN)
            cols = offs_n < Lk
            if MASKED:
                mb = _mask_tile(mask_base, sm_m, sm_n, offs_m, offs_n, cols, Lq,
                                False, MASK_COL)  # fmt: skip
            else:
                mb = tl.zeros([1, 1], tl.float32)
            dq = _dq_tile(
                dq, q, do, m, lg, delta,
                k_base + offs_n[:, None] * k_l + offs_d[None, :],
                v_base + offs_n[:, None] * v_l + offs_d[None, :],
                mb, cols, c, True, MASKED, MASK_BOOL,
            )  # fmt: skip
        dq = dq * scale
        tl.store(
            DQ + b * g_b + h * g_h + offs_m[:, None] * g_l + offs_d[None, :],
            dq.to(DQ.dtype.element_ty), mask=rows[:, None],
        )  # fmt: skip

    @triton.jit
    def _dq_cast_kernel(DQACC, DQ, scale, NUMEL, BLOCK: tl.constexpr):
        """DQ = (DQACC * scale) cast to DQ's dtype; both contiguous."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        keep = offs < NUMEL
        x = tl.load(DQACC + offs, mask=keep, other=0.0)
        tl.store(DQ + offs, (x * scale).to(DQ.dtype.element_ty), mask=keep)

else:  # pragma: no cover - no triton
    _attn_fwd_kernel = None
    _attn_delta_kernel = None
    _attn_bwd_dkdv_kernel = None
    _attn_dq_kernel = None
    _dq_cast_kernel = None


# ── host side ────────────────────────────────────────────────────────────


def _tile(phase: str, D: int) -> tuple[int, int, int, int]:
    """(BM, BN, num_warps, num_stages) for a phase and head dim."""
    override = _TILE_OVERRIDE.get(phase)
    if override is not None:
        return tuple(override)  # type: ignore[arg-type,return-value]
    table = _TILES[phase]
    return table.get(D, table[128])


def _max_offset(t: torch.Tensor) -> int:
    """Largest element offset a kernel forms for a 4D tensor (int32 tiles)."""
    b, l, h, d = t.shape
    return (b - 1) * t.stride(0) + (l - 1) * t.stride(1) + (h - 1) * t.stride(2) + d


def _mask_info(mask: torch.Tensor | None, B: int, H: int, Lq: int, Lk: int):
    """(mask, sb, sh, sm, sn, masked, col, is_bool) for the kernels."""
    if mask is None:
        return None, 0, 0, 0, 0, False, False, False
    mb = torch.broadcast_to(mask, (B, H, Lq, Lk))
    sm_b, sm_h, sm_m, sm_n = mb.stride()
    return (
        mb,
        sm_b,
        sm_h,
        sm_m,
        sm_n,
        True,
        sm_h == 0 and sm_m == 0,
        mb.dtype == torch.bool,
    )


def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
    """Reinterpret a bool mask as uint8 so Triton loads it as plain bytes."""
    if mask.dtype != torch.bool:
        return mask
    try:
        return mask.view(torch.uint8)
    except RuntimeError:  # a stride-0 last dim; materialise the bytes
        return mask.to(torch.uint8)


def _strides(t: torch.Tensor) -> tuple[int, int, int]:
    return t.stride(0), t.stride(1), t.stride(2)


def is_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> bool:
    """Whether :func:`attention` can run this call on this device.

    Returns ``False`` (never raises) for any shape, dtype, device or mask the
    Triton path does not implement, so the caller can use the eager reference.
    """
    if not HAS_TRITON or not torch.cuda.is_available():
        return False
    if is_causal:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.dtype not in _DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if key.shape != value.shape:
        return False
    B, Lq, H, D = query.shape
    Bk, Lk, Hkv, Dk = key.shape
    if Bk != B or Dk != D or Lq == 0 or Lk == 0 or H == 0 or Hkv == 0:
        return False
    if D not in _HEAD_DIMS:
        return False
    if H % Hkv:
        return False
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        return False
    # int32 tile offsets: the kernels keep the per-tile offset arithmetics in
    # int32 (like the reference); that is only sound while one row fits.
    limit = 2 ** 31 - 1
    for t in (query, key, value):
        if _max_offset(t) > limit:
            return False
    if attn_mask is not None:
        if attn_mask.device != query.device or attn_mask.requires_grad:
            return False
        if attn_mask.dtype != torch.bool and not (
            attn_mask.dtype.is_floating_point and attn_mask.dtype.itemsize <= 4
        ):
            return False
        try:
            mb = torch.broadcast_to(attn_mask, (B, H, Lq, Lk))
        except RuntimeError:
            return False
        for s, n in zip(mb.stride(), mb.shape):
            if s and abs(s) * (n - 1) > limit:
                return False
    return True


def _check(query, key, value, attn_mask, is_causal) -> None:
    if not is_supported(query, key, value, attn_mask, is_causal):
        size = lambda t: tuple(t.shape) if torch.is_tensor(t) else t  # noqa: E731
        raise RuntimeError(
            "attention_triton.attention does not support this call: "
            f"q={size(query)} {getattr(query, 'dtype', '?')} "
            f"{getattr(query, 'device', '?')}, kv={size(key)} "
            f"{getattr(key, 'dtype', '?')}, mask="
            f"{None if attn_mask is None else (size(attn_mask), attn_mask.dtype)}, "
            f"causal={is_causal}. Check is_supported() or use "
            "UnifiedTrainer.kernels.eager.sdpa_gqa."
        )


def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the forward kernel; returns (o, m, log2 l)."""
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    G = H // k.shape[2]
    mb, sm_b, sm_h, sm_m, sm_n, masked, col, is_bool = _mask_info(mask, B, H, Lq, Lk)
    if masked:
        mb = _prepare_mask(mb)
    BM, BN, warps, stages = _tile("fwd", D)
    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    m = torch.empty((B, H, Lq), device=q.device, dtype=torch.float32)
    lg = torch.empty((B, H, Lq), device=q.device, dtype=torch.float32)
    st_b, st_h = H * Lq, Lq
    with torch.cuda.device(q.device):
        _attn_fwd_kernel[(triton.cdiv(Lq, BM), H, B)](
            q, k, v, mb, o, m, lg,
            *_strides(q), *_strides(k), *_strides(v), *_strides(o),
            sm_b, sm_h, sm_m, sm_n, st_b, st_h,
            scale * _LOG2E, Lq, Lk,
            D=D, G=G, BM=BM, BN=BN,
            MASKED=masked, MASK_COL=col, MASK_BOOL=is_bool,
            num_warps=warps, num_stages=stages,
        )  # fmt: skip
    return o, m, lg


def _backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    m: torch.Tensor,
    lg: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the backward kernels; returns (dq, dk, dv) in the inputs' dtypes."""
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    Hkv = k.shape[2]
    G = H // Hkv
    mb, sm_b, sm_h, sm_m, sm_n, masked, col, is_bool = _mask_info(mask, B, H, Lq, Lk)
    if masked:
        mb = _prepare_mask(mb)
    do = do if do.stride(-1) == 1 else do.contiguous()
    impl = _TILE_OVERRIDE.get("bwd_impl")
    if impl is not None:
        fused = impl == "fused"
    else:
        fused = B * H * Lq * D * 4 <= _FUSED_MAX_BYTES
    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk = torch.empty_like(k, memory_format=torch.contiguous_format)
    dv = torch.empty_like(v, memory_format=torch.contiguous_format)
    delta = torch.empty((B, H, Lq), device=q.device, dtype=torch.float32)
    st_b, st_h = H * Lq, Lq
    c = scale * _LOG2E
    with torch.cuda.device(q.device):
        _attn_delta_kernel[(triton.cdiv(Lq, 64), H, B)](
            o, do, delta, *_strides(o), *_strides(do), st_b, st_h, Lq,
            D=D, BM=64, num_warps=8,
        )  # fmt: skip
        if fused:
            dq_acc = torch.zeros_like(dq, dtype=torch.float32)
            BM, BN, warps, stages = _tile("bwd", D)
            _attn_bwd_dkdv_kernel[(triton.cdiv(Lk, BN), Hkv, B)](
                q, k, v, do, mb, m, lg, delta, dk, dv, dq_acc,
                *_strides(q), *_strides(k), *_strides(v), *_strides(do),
                *_strides(dk), sm_b, sm_h, sm_m, sm_n, st_b, st_h,
                *_strides(dq_acc),
                c, scale, Lq, Lk,
                D=D, G=G, BM=BM, BN=BN,
                MASKED=masked, MASK_COL=col, MASK_BOOL=is_bool, ATOMIC=True,
                num_warps=warps, num_stages=stages,
            )  # fmt: skip
            numel = dq_acc.numel()
            _dq_cast_kernel[(triton.cdiv(numel, 4096),)](
                dq_acc, dq, scale, numel, BLOCK=4096, num_warps=4,
            )  # fmt: skip
        else:
            BM, BN, warps, stages = _tile("dkdv", D)
            _attn_bwd_dkdv_kernel[(triton.cdiv(Lk, BN), Hkv, B)](
                q, k, v, do, mb, m, lg, delta, dk, dv, dk,
                *_strides(q), *_strides(k), *_strides(v), *_strides(do),
                *_strides(dk), sm_b, sm_h, sm_m, sm_n, st_b, st_h, 0, 0, 0,
                c, scale, Lq, Lk,
                D=D, G=G, BM=BM, BN=BN,
                MASKED=masked, MASK_COL=col, MASK_BOOL=is_bool, ATOMIC=False,
                num_warps=warps, num_stages=stages,
            )  # fmt: skip
            BM, BN, warps, stages = _tile("dq", D)
            _attn_dq_kernel[(triton.cdiv(Lq, BM), H, B)](
                q, k, v, do, mb, m, lg, delta, dq,
                *_strides(q), *_strides(k), *_strides(v), *_strides(do),
                *_strides(dq), sm_b, sm_h, sm_m, sm_n, st_b, st_h,
                c, scale, Lq, Lk,
                D=D, G=G, BM=BM, BN=BN,
                MASKED=masked, MASK_COL=col, MASK_BOOL=is_bool,
                num_warps=warps, num_stages=stages,
            )  # fmt: skip
    return dq, dk, dv


class _FlashAttention(torch.autograd.Function):
    """Recompute-based flash attention.

    Saves q, k, v, O, the two row planes and the mask; the backward replays the
    mask and recomputes the score matrix.  Second-order gradients are not
    supported.
    """

    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        o, m, lg = _forward(q, k, v, mask, scale)
        ctx.save_for_backward(q, k, v, o, m, lg, mask)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, lg, mask = ctx.saved_tensors
        if do is None:
            return None, None, None, None, None
        dq, dk, dv = _backward(q, k, v, o, do, m, lg, mask, ctx.scale)
        return dq, dk, dv, None, None


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Flash attention over ``(B, L, H, D)`` with grouped-query KV heads.

    ``query`` is ``(B, Lq, H, D)``, ``key``/``value`` are ``(B, Lk, Hkv, D)``
    with ``H`` a multiple of ``Hkv``.  ``attn_mask`` is ``None``, a bool mask or
    an additive float mask broadcastable to ``(B, H, Lq, Lk)``; ``scale``
    defaults to ``D ** -0.5``.  Returns ``(B, Lq, H, D)`` in ``query.dtype``.
    Raises ``RuntimeError`` for a call :func:`is_supported` rejects.
    """
    _check(query, key, value, attn_mask, is_causal)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scale = float(scale)
    q = query if query.stride(-1) == 1 else query.contiguous()
    k = key if key.stride(-1) == 1 else key.contiguous()
    v = value if value.stride(-1) == 1 else value.contiguous()
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _FlashAttention.apply(q, k, v, attn_mask, scale)
    return _forward(q, k, v, attn_mask, scale)[0]


# ── benchmark harness ────────────────────────────────────────────────────

#: The shapes the report measures: (B, Lq, Lk, H, Hkv, D).
_BENCH_SHAPES = (
    (1, 1024, 1024, 48, 12, 128),
    (1, 2048, 2048, 48, 12, 128),
    (1, 4608, 4608, 48, 12, 128),
)


def _sdpa_gqa(q, k, v, scale):
    """The model's current path: enable_gqa, i.e. SDPA's math backend here."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=scale, enable_gqa=True,
    ).transpose(1, 2)  # fmt: skip


def _sdpa_expanded(q, k, v, scale):
    """SDPA's fused memory-efficient kernel (the KV heads arrive pre-expanded)."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
    ).transpose(1, 2)  # fmt: skip


def _compiled_flex():
    """``torch.compile(flex_attention)`` on pre-expanded KV, or None."""
    try:
        from torch.nn.attention.flex_attention import flex_attention

        return torch.compile(flex_attention, dynamic=False)
    except Exception:  # pragma: no cover - no flex / no compiler
        return None


def _gpu_snapshot() -> str:
    free, total = torch.cuda.mem_get_info()
    try:
        util = f"{torch.cuda.utilization()}%"
    except Exception:  # pragma: no cover - no nvml
        util = "?"
    return f"free={free / 2**30:.1f}/{total / 2**30:.1f} GiB util={util}"


def benchmark_cases(
    shapes: Any = None,
    dtype: torch.dtype = torch.bfloat16,
    with_baselines: bool = True,
    rounds: int = 3,
    rep: float = 100.0,
) -> list[dict]:
    """Time this kernel and the three baselines; print a table, return the rows.

    TFLOP/s counts 2 GEMMs for the forward and 5 for forward+backward, i.e.
    ``4 Lq Lk H D`` and ``10 Lq Lk H D`` FLOPs.  Each cell is the minimum over
    ``rounds`` interleaved rounds of ``do_bench(..., return_mode="min")``: the
    implementations are sampled round-robin so a busy GPU inflates every
    candidate alike, and the min discards the samples it hit.

    The baselines are the three paths measured for the report: SDPA with
    ``enable_gqa=True`` (the model's current call), SDPA on expanded KV, and
    ``torch.compile(flex_attention)``.  K/V are expanded once, before timing.
    """
    if not HAS_TRITON or not torch.cuda.is_available():
        print("attention_triton: no triton/CUDA; nothing to benchmark")
        return []
    device = torch.device("cuda", torch.cuda.current_device())
    shapes = tuple(shapes) if shapes is not None else _BENCH_SHAPES
    flex = _compiled_flex() if with_baselines else None
    impls: list[tuple[str, Any]] = [("triton", attention)]
    if with_baselines:
        impls.append(("sdpa enable_gqa", _sdpa_gqa))
        impls.append(("sdpa kv-expanded", _sdpa_expanded))
        if flex is not None:
            impls.append(("flex compiled", flex))
    print(f"\nGPU before: {_gpu_snapshot()}")
    rows: list[dict] = []
    header = (
        f"{'shape':30s} {'impl':18s} {'fwd ms':>9s} {'fwd T':>7s} "
        f"{'f+b ms':>8s} {'f+b T':>7s} {'speedup':>8s}"
    )
    for B, Lq, Lk, H, Hkv, D in shapes:
        q = torch.randn(B, Lq, H, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, Lk, Hkv, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, Lk, Hkv, D, device=device, dtype=dtype, requires_grad=True)
        g = H // Hkv
        kx = k.repeat_interleave(g, dim=2)
        vx = v.repeat_interleave(g, dim=2)
        dout = torch.ones(B, Lq, H, D, device=device, dtype=dtype)
        flops = 2.0 * 2.0 * Lq * Lk * H * D
        scale = D ** -0.5
        label = f"B{B} L{Lq}/{Lk} H{H}/{Hkv} D{D} {str(dtype)[6:]}"
        calls: dict[str, tuple[Any, Any]] = {}
        for name, fn in impls:
            kk, vv = (kx, vx) if name != "sdpa enable_gqa" else (k, v)
            if name == "flex compiled":
                def call(q=q, kk=kk, vv=vv, fn=fn):
                    return fn(q.transpose(1, 2), kk.transpose(1, 2),
                              vv.transpose(1, 2), scale=scale).transpose(1, 2)
            else:
                def call(q=q, kk=kk, vv=vv, fn=fn):
                    return fn(q, kk, vv, scale=scale)

            def full(call=call):
                out = call()
                torch.autograd.grad(out, (q, k, v), grad_outputs=dout)

            try:  # compile and allocate before anything is timed
                call()
                full()
                torch.cuda.synchronize()
            except Exception as exc:  # pragma: no cover - a baseline may fail
                print(f"{label:30s} {name:18s} unavailable: {type(exc).__name__}: "
                      f"{str(exc)[:80]}")  # fmt: skip
                continue
            calls[name] = (call, full)
        ms_f = {name: float("inf") for name in calls}
        ms_fb = {name: float("inf") for name in calls}
        for _ in range(rounds):
            for name, (call, full) in calls.items():
                ms_f[name] = min(ms_f[name], triton.testing.do_bench(
                    call, warmup=10, rep=rep, return_mode="min"))  # fmt: skip
                ms_fb[name] = min(ms_fb[name], triton.testing.do_bench(
                    full, warmup=10, rep=rep, return_mode="min"))  # fmt: skip
        base = ms_fb.get("sdpa enable_gqa")
        print(header)
        for name, (call, _) in calls.items():
            speed = f"{base / ms_fb[name]:7.2f}x" if base else "     -"
            rows.append({"shape": (B, Lq, Lk, H, Hkv, D), "impl": name,
                         "fwd_ms": ms_f[name], "fwd_bwd_ms": ms_fb[name]})  # fmt: skip
            print(f"{label:30s} {name:18s} {ms_f[name]:9.3f} "
                  f"{flops / ms_f[name] / 1e9:7.1f} {ms_fb[name]:8.3f} "
                  f"{2.5 * flops / ms_fb[name] / 1e9:7.1f} {speed:>8s}")  # fmt: skip
        if "sdpa enable_gqa" in calls:
            ref = _sdpa_gqa(q, k, v, scale)
            got = attention(q, k, v, scale=scale)
            cos = torch.nn.functional.cosine_similarity(
                got.float().flatten(), ref.float().flatten(), dim=0
            ).item()  # fmt: skip
            print(f"{label:30s} {'cos vs sdpa':18s} {cos:9.7f}")
    print(f"GPU after: {_gpu_snapshot()}")
    return rows


if __name__ == "__main__":  # pragma: no cover
    benchmark_cases()
