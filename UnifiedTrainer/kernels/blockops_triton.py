# =============================================================================
# DERIVED CODE - fused block ops learned from KohakuMoss.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# Upstream design this module comes from (`src/kohakumoss_kernels/backends/`):
#   triton/blocks/norm.py       rms_norm_mod   - one row tile per program, fp32
#                               statistics, `rstd` kept in fp32 for the backward,
#                               no `[N, D]` intermediate tensor
#   triton/blocks/rope.py       qk_norm_rope   - sin/cos read per row tile, the
#                               backward rotates the gradient back before the
#                               norm's backward
#   triton/blocks/ffn.py        situ_glu       - one read of each half of the
#                               projection, one write of the activation
#   triton/blocks/residual.py   gated_residual - gate read through strides,
#                               `d_x = g` in the backward
#   triton/blocks/_blockops.py  the per-tile fp32 column partials and the
#                               fixed-order reduction that makes the
#                               d_scale / d_shift / d_weight path atomic-free
#   eager/{norm,ffn,rope}.py    the normative semantics (mirrored in
#                               UnifiedTrainer/kernels/eager.py)
#
# What is theirs (the ideas):
#   * one program = one row tile `[ROWS, next_pow2(D)]`, row reduction in fp32
#     registers, one rounding at the store;
#   * `rstd` saved in fp32 for the backward instead of recomputing statistics;
#   * per-tile fp32 column partials for the modulation / weight gradients, summed
#     afterwards in a fixed order (deterministic, no atomics);
#   * the modulation (adaLN shift/scale, residual gate) is read through its
#     strides - an expanded `[B, L, D]` copy is never materialised;
#   * `d_x = g` and `d_o = g * gate` for the gated residual.
#
# What is new here (this file, for UnifiedTrainer):
#   * the re-expression against the dense `(B, L, D)` layout and against
#     diffusers' `use_real_unbind_dim=-1` pair convention (upstream's rope uses
#     split-half pairs, and fuses q/k norm into the same program);
#   * the `(B, 1, D)` / `(1, 1, D)` per-sample modulation contract, including how
#     the broadcast is folded back into the caller's gradient shape;
#   * a standalone rope backward that applies the transpose rotation (upstream
#     has no standalone rope op);
#   * the `(B, L, H, D)` rope tiling - a program owns `HB` heads of one token;
#   * one two-slot tile reduction shared by the norm and the gated-residual
#     backward.
# Nothing from that repository is imported at runtime: the kernels below are
# local, and the semantics they must equal live in UnifiedTrainer/kernels/eager.py.
#
# Measure before believing a win: this file's `__main__` prints eager vs
# torch.compile vs Triton for the Krea-2 shapes (see also
# md/14-kernel-optimizations-applied.md).
# =============================================================================
"""Fused block ops for UnifiedTrainer's diffusion transformer blocks (Triton).

Four ops, each the fused form of a chain the model adapters otherwise run as
several eager kernels:

======================================  ====================================
:func:`rms_norm_modulate`               ``(1+scale) * rms_norm(x, w) + shift``
:func:`silu_mul`                        ``silu(gate) * up``
:func:`rope_apply`                      interleaved rotary embedding
:func:`add_gated`                       ``residual + gate * value``
======================================  ====================================

Layout / dtype contract
-----------------------
Every op is a last-dimension op on a contiguous CUDA tensor: ``(B, L, D)`` for
the three elementwise ops (``(B, L, H, D)`` with ``cos``/``sin`` ``(L, D)`` for
:func:`rope_apply`).  Inputs are bf16 / fp16 / fp32 - a mix is allowed where the
model mixes them (an fp32 norm weight with bf16 activations) - and all arithmetic
inside a kernel is fp32, rounded once when the output is stored.

Statistics (the RMS mean, its reciprocal square root) are fp32, as the eager
reference computes them.

Broadcast contract
------------------
``scale``/``shift``/``gate`` may be ``None`` (only the norm's ``scale``/``shift``)
or broadcast to the activation as a **per-sample** modulation: ``(B, 1, D)`` or
``(1, 1, D)`` (a 2-D ``(B, D)`` is read as ``(B, 1, D)``).  Their last-dim stride
must be 1; the row stride may be anything (a ``mod[:, k]`` slice of a
``[B, 6, D]`` tensor is read in place, never copied).  A full per-token
``(B, L, D)`` modulation of the *norm* is not supported and
:func:`is_supported_rms_norm_modulate` returns ``False`` for it, so the caller
falls back to eager; the gated residual does accept a full-size ``gate``.

Autograd
--------
Each op is a :class:`torch.autograd.Function` with a Triton backward; the
backward writes the activation gradient and, when the modulation needs one, the
per-tile fp32 column partials that a second small kernel reduces into
``d_scale`` / ``d_shift`` / ``d_weight`` / ``d_gate``.  Where a gradient is the
upstream gradient unchanged (``d_residual``) nothing is launched at all.

``torch.compiler.is_compiling()`` is not special-cased: these are plain
autograd.Functions, so ``torch.compile`` traces them as a graph break unless the
caller keeps them out of the compiled region.  The measured alternative for a
compiled region is inductor's own fusion of ``kernels/eager.py``; ``__main__``
prints both, per op.
"""

__all__ = [
    "HAS_TRITON",
    "is_supported_rms_norm_modulate",
    "rms_norm_modulate",
    "is_supported_silu_mul",
    "silu_mul",
    "is_supported_rope_apply",
    "rope_apply",
    "is_supported_add_gated",
    "add_gated",
]

import torch

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

try:  # Triton is optional: importing this module must not raise.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - no triton on this box
    triton = None
    tl = None
    HAS_TRITON = False


# =============================================================================
# kernels
# =============================================================================

if HAS_TRITON:

    # ── rms_norm_modulate ────────────────────────────────────────────────
    #
    # Grid: (row tiles of one batch entry, batch).  A program owns `ROWS` rows
    # of `BLOCK_D` (masked to D) columns; `scale`/`shift` are per-(batch, column)
    # so they are loaded as one `[BLOCK_D]` vector and broadcast over the rows.

    @triton.jit
    def _rnm_fwd_kernel(
        x_ptr,
        w_ptr,
        sc_ptr,
        sh_ptr,
        y_ptr,
        rstd_ptr,
        L,
        D,
        eps,
        offset,
        sc_sb,
        sh_sb,
        HAS_W: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_SHIFT: tl.constexpr,
        STORE_RSTD: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        b = tl.program_id(1)
        rows = t * ROWS + tl.arange(0, ROWS)
        rok = rows < L
        cols = tl.arange(0, BLOCK_D)
        cok = cols < D
        mask = rok[:, None] & cok[None, :]
        row = b * L + rows
        offs = row.to(tl.int64)[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=1) / D
        rstd = tl.div_rn(1.0, tl.sqrt_rn(ms + eps))
        y = x * rstd[:, None]
        if HAS_W:
            w = tl.load(w_ptr + cols, mask=cok, other=0.0).to(tl.float32)
            y = y * (w + offset)[None, :]
        if HAS_SCALE:
            sc = tl.load(sc_ptr + b * sc_sb + cols, mask=cok, other=0.0)
            y = y * (1.0 + sc.to(tl.float32))[None, :]
        if HAS_SHIFT:
            sh = tl.load(sh_ptr + b * sh_sb + cols, mask=cok, other=0.0)
            y = y + sh.to(tl.float32)[None, :]
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
        if STORE_RSTD:
            tl.store(rstd_ptr + row, rstd, mask=rok)

    # The backward of `y = (1+s) * (w * xhat) + sh`, xhat = x * rstd:
    #   m   = (1+s) * w
    #   c   = mean_D(dy * m * xhat)
    #   dx  = rstd * (dy * m - xhat * c)
    #   d_w = sum_rows dy * (1+s) * xhat        (slot 0, summed over the batch)
    #   d_s = w * sum_rows dy * xhat
    #   d_sh= sum_rows dy                       (slot 1)
    # Slots 0/1 hold fp32 column partials of the rows this program owns; the
    # reduce kernel sums the tiles of a sample in a fixed order.

    @triton.jit
    def _rnm_bwd_kernel(
        dy_ptr,
        x_ptr,
        rstd_ptr,
        w_ptr,
        sc_ptr,
        dx_ptr,
        part_ptr,
        L,
        D,
        P,
        CHUNK,
        sc_sb,
        HAS_W: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        NEED_A: tl.constexpr,
        NEED_B: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        start = t * CHUNK
        stop = tl.minimum(start + CHUNK, L)
        cols = tl.arange(0, BLOCK_D)
        cok = cols < D
        m = tl.full([BLOCK_D], 1.0, tl.float32)
        if HAS_W:
            m = tl.load(w_ptr + cols, mask=cok, other=0.0).to(tl.float32)
        if HAS_SCALE:
            sc = tl.load(sc_ptr + b * sc_sb + cols, mask=cok, other=0.0)
            m = m * (1.0 + sc.to(tl.float32))
        acc_a = tl.zeros([BLOCK_D], tl.float32)
        acc_b = tl.zeros([BLOCK_D], tl.float32)
        base = b * L
        for r0 in range(start, stop, ROWS):
            rows = r0 + tl.arange(0, ROWS)
            rok = rows < stop
            mask = rok[:, None] & cok[None, :]
            offs = (base + rows).to(tl.int64)[:, None] * D + cols[None, :]
            dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            rs = tl.load(rstd_ptr + base + rows, mask=rok, other=0.0)
            xhat = x * rs[:, None]
            gm = dy * m[None, :]
            c = tl.sum(gm * xhat, axis=1) / D
            dx = rs[:, None] * (gm - xhat * c[:, None])
            tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
            if NEED_A:
                acc_a += tl.sum(dy * xhat, axis=0)
            if NEED_B:
                acc_b += tl.sum(dy, axis=0)
        if NEED_A:
            tl.store(part_ptr + (b * 2) * P * D + t * D + cols, acc_a, mask=cok)
        if NEED_B:
            tl.store(part_ptr + (b * 2 + 1) * P * D + t * D + cols, acc_b, mask=cok)

    # One reduction for both op backs: `part [B, SLOTS, P, D]` fp32 tile partials
    # -> the caller's gradient tensors.  Slot a is `sum dy * aux` (xhat for the
    # norm, value for the residual), slot b is `sum dy`.
    #   d_w  = sum_b (1+scale_b) * a_b        NEED_DW
    #   d_s  = w * a_b     /  sum_b w * a_b   STORE_A (+ACC_A)
    #   d_sh = b_b         /  sum_b b_b       STORE_B (+ACC_B)

    @triton.jit
    def _col_reduce_kernel(
        part_ptr,
        w_ptr,
        sc_ptr,
        out_a_ptr,
        out_b_ptr,
        dw_ptr,
        D,
        B,
        P,
        SLOTS,
        sc_sb,
        HAS_W: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        LOAD_A: tl.constexpr,
        STORE_A: tl.constexpr,
        LOAD_B: tl.constexpr,
        STORE_B: tl.constexpr,
        NEED_DW: tl.constexpr,
        ACC_A: tl.constexpr,
        ACC_B: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        cols = tl.program_id(0) * BD + tl.arange(0, BD)
        ok = cols < D
        w = tl.full([BD], 1.0, tl.float32)
        if HAS_W:
            w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
        acc_dw = tl.zeros([BD], tl.float32)
        acc_both_a = tl.zeros([BD], tl.float32)
        acc_both_b = tl.zeros([BD], tl.float32)
        for b in range(B):
            base = part_ptr + b.to(tl.int64) * (SLOTS * P * D)
            a = tl.zeros([BD], tl.float32)
            s = tl.zeros([BD], tl.float32)
            for p0 in range(0, P, BT):
                ps = p0 + tl.arange(0, BT)
                pm = (ps < P)[:, None] & ok[None, :]
                idx = ps.to(tl.int64)[:, None] * D + cols[None, :]
                if LOAD_A:
                    a += tl.sum(tl.load(base + idx, mask=pm, other=0.0), axis=0)
                if LOAD_B:
                    s += tl.sum(tl.load(base + P * D + idx, mask=pm, other=0.0), axis=0)
            if NEED_DW:
                mod = 1.0
                if HAS_SCALE:
                    sc = tl.load(sc_ptr + b * sc_sb + cols, mask=ok, other=0.0)
                    mod = 1.0 + sc.to(tl.float32)
                acc_dw += mod * a
            if ACC_A:
                acc_both_a += w * a
            else:
                if STORE_A:
                    tl.store(
                        out_a_ptr + b * D + cols,
                        (w * a).to(out_a_ptr.dtype.element_ty),
                        mask=ok,
                    )
            if ACC_B:
                acc_both_b += s
            else:
                if STORE_B:
                    tl.store(
                        out_b_ptr + b * D + cols,
                        s.to(out_b_ptr.dtype.element_ty),
                        mask=ok,
                    )
        if NEED_DW:
            tl.store(dw_ptr + cols, acc_dw.to(dw_ptr.dtype.element_ty), mask=ok)
        if ACC_A:
            if STORE_A:
                tl.store(
                    out_a_ptr + cols, acc_both_a.to(out_a_ptr.dtype.element_ty), mask=ok
                )
        if ACC_B:
            if STORE_B:
                tl.store(
                    out_b_ptr + cols, acc_both_b.to(out_b_ptr.dtype.element_ty), mask=ok
                )

    # ── silu_mul ─────────────────────────────────────────────────────────
    #
    # Flat elementwise, BLOCK elements per program: one read of each input, one
    # write.  The backward reads `gate`/`up` again (both are already needed) and
    # writes both gradients in the same pass.

    @triton.jit
    def _silu_mul_fwd_kernel(g_ptr, u_ptr, y_ptr, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (g * tl.sigmoid(g)) * u
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_mul_bwd_kernel(
        dy_ptr, g_ptr, u_ptr, dg_ptr, du_ptr, N, BLOCK: tl.constexpr
    ):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        # d/dg [g * sigmoid(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        dg = dy * u * (sig * (1.0 + g * (1.0 - sig)))
        du = dy * (g * sig)
        tl.store(dg_ptr + offs, dg.to(dg_ptr.dtype.element_ty), mask=mask)
        tl.store(du_ptr + offs, du.to(du_ptr.dtype.element_ty), mask=mask)

    # ── rope_apply ───────────────────────────────────────────────────────
    #
    # x (B, L, H, D) contiguous, cos/sin (L, D).  A program owns `HB` heads of
    # one token: the cos/sin row is loaded once and broadcast over the heads.
    # A pair (2j, 2j+1) is loaded as one 2-wide inner tile, so the loads and the
    # stores stay coalesced and vectorised:
    #   out[2j]   = x[2j]   * cos[2j]   - x[2j+1] * sin[2j]
    #   out[2j+1] = x[2j+1] * cos[2j+1] + x[2j]   * sin[2j+1]
    # which is the linear map [[c_e, -s_e], [s_o, c_o]] on the pair, so the
    # backward applies its transpose (no sin/cos gradient: they are tables).

    @triton.jit
    def _rope_fwd_kernel(
        x_ptr,
        c_ptr,
        s_ptr,
        y_ptr,
        L,
        H,
        D,
        HB: tl.constexpr,
        BHALF: tl.constexpr,
    ):
        bl = tl.program_id(0)  # b * L + l
        l = bl % L
        hs = tl.program_id(1) * HB + tl.arange(0, HB)
        hok = hs < H
        j = tl.arange(0, BHALF)
        jok = 2 * j < D
        eo = tl.arange(0, 2)
        offs = (
            bl.to(tl.int64) * (H * D)
            + hs.to(tl.int64)[:, None, None] * D
            + (2 * j)[None, :, None]
            + eo[None, None, :]
        )
        mask = hok[:, None, None] & jok[None, :, None]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        coff = l.to(tl.int64) * D + (2 * j)[:, None] + eo[None, :]
        cmask = jok[:, None]
        ce, co = tl.split(tl.load(c_ptr + coff, mask=cmask, other=0.0).to(tl.float32))
        se, so = tl.split(tl.load(s_ptr + coff, mask=cmask, other=0.0).to(tl.float32))
        xe, xo = tl.split(x)
        out = tl.join(xe * ce[None, :] - xo * se[None, :], xo * co[None, :] + xe * so[None, :])
        tl.store(y_ptr + offs, out.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rope_bwd_kernel(
        dy_ptr,
        c_ptr,
        s_ptr,
        dx_ptr,
        L,
        H,
        D,
        HB: tl.constexpr,
        BHALF: tl.constexpr,
    ):
        bl = tl.program_id(0)
        l = bl % L
        hs = tl.program_id(1) * HB + tl.arange(0, HB)
        hok = hs < H
        j = tl.arange(0, BHALF)
        jok = 2 * j < D
        eo = tl.arange(0, 2)
        offs = (
            bl.to(tl.int64) * (H * D)
            + hs.to(tl.int64)[:, None, None] * D
            + (2 * j)[None, :, None]
            + eo[None, None, :]
        )
        mask = hok[:, None, None] & jok[None, :, None]
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        coff = l.to(tl.int64) * D + (2 * j)[:, None] + eo[None, :]
        cmask = jok[:, None]
        ce, co = tl.split(tl.load(c_ptr + coff, mask=cmask, other=0.0).to(tl.float32))
        se, so = tl.split(tl.load(s_ptr + coff, mask=cmask, other=0.0).to(tl.float32))
        ge, go = tl.split(dy)
        out = tl.join(ce[None, :] * ge + so[None, :] * go, co[None, :] * go - se[None, :] * ge)
        tl.store(dx_ptr + offs, out.to(dx_ptr.dtype.element_ty), mask=mask)

    # ── add_gated ────────────────────────────────────────────────────────
    #
    # y = residual + gate * value.  `gate` is either full-size (then g_sb/g_sl
    # are the activation's strides) or a per-sample (B|1, 1, D) modulation
    # (g_sl == 0).  Backward: d_residual is the upstream gradient itself,
    # d_value = dy * gate, and d_gate = dy * value - per row when gate is
    # full-size, otherwise summed over the rows of the sample in tile partials.

    @triton.jit
    def _ag_fwd_kernel(
        r_ptr,
        g_ptr,
        v_ptr,
        y_ptr,
        L,
        D,
        g_sb,
        FULL: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        b = tl.program_id(1)
        rows = t * ROWS + tl.arange(0, ROWS)
        rok = rows < L
        cols = tl.arange(0, BLOCK_D)
        cok = cols < D
        mask = rok[:, None] & cok[None, :]
        offs = (b * L + rows).to(tl.int64)[:, None] * D + cols[None, :]
        r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if FULL:
            g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = r + g * v
        else:
            gv = tl.load(g_ptr + b * g_sb + cols, mask=cok, other=0.0).to(tl.float32)
            y = r + gv[None, :] * v
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _ag_bwd_kernel(
        dy_ptr,
        g_ptr,
        v_ptr,
        dv_ptr,
        dg_ptr,
        part_ptr,
        L,
        D,
        P,
        CHUNK,
        g_sb,
        FULL: tl.constexpr,
        NEED_DV: tl.constexpr,
        NEED_G: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        start = t * CHUNK
        stop = tl.minimum(start + CHUNK, L)
        cols = tl.arange(0, BLOCK_D)
        cok = cols < D
        acc = tl.zeros([BLOCK_D], tl.float32)
        gv = tl.zeros([BLOCK_D], tl.float32)
        if not FULL:
            gv = tl.load(g_ptr + b * g_sb + cols, mask=cok, other=0.0).to(tl.float32)
        for r0 in range(start, stop, ROWS):
            rows = r0 + tl.arange(0, ROWS)
            rok = rows < stop
            mask = rok[:, None] & cok[None, :]
            offs = (b * L + rows).to(tl.int64)[:, None] * D + cols[None, :]
            dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            if FULL:
                g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                if NEED_DV:
                    tl.store(dv_ptr + offs, (dy * g).to(dv_ptr.dtype.element_ty), mask=mask)
                if NEED_G:
                    v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                    tl.store(dg_ptr + offs, (dy * v).to(dg_ptr.dtype.element_ty), mask=mask)
            else:
                if NEED_DV:
                    tl.store(dv_ptr + offs, (dy * gv[None, :]).to(dv_ptr.dtype.element_ty),
                             mask=mask)
                if NEED_G:
                    v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                    acc += tl.sum(dy * v, axis=0)
        if NEED_G:
            if not FULL:
                tl.store(part_ptr + b * P * D + t * D + cols, acc, mask=cok)


# =============================================================================
# host side: layouts, launches, autograd
# =============================================================================


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


_NUM_SMS = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = max(1, torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    return _NUM_SMS


def _plain(t) -> bool:
    """A contiguous CUDA tensor of a supported dtype with at least one element."""
    return (
        t is not None
        and t.is_cuda
        and t.dtype in _DTYPES
        and t.is_contiguous()
        and t.numel() > 0
    )


def _needs_grad(*ts) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in ts)


def _row_cfg(block_d: int) -> tuple[int, int]:
    """(ROWS, num_warps) for a row-tile op: about 4k-8k fp32 elements per tile.

    Measured on the RTX 5060 Ti / sm120 (bf16, Krea-2 shapes): 4 warps and one
    row per tile is fastest for the 6144-column norm, 16 rows of 256 columns for
    the small one.  The tile is a register budget: ROWS * BLOCK_D fp32 values
    live at once in the row reduction.
    """
    if block_d <= 256:
        return 16, 4
    if block_d <= 512:
        return 8, 4
    if block_d <= 1024:
        return 4, 4
    if block_d <= 2048:
        return 2, 4
    if block_d <= 4096:
        return 2, 8
    if block_d <= 8192:
        return 1, 4
    return 1, 8


def _tile_plan(length: int, batch: int) -> tuple[int, int]:
    """(P, CHUNK): P programs per batch entry, CHUNK rows each, for a backward.

    Two programs per SM measured fastest on sm120 for both row-tiled backwards
    (one per SM leaves the kernel short of warps to hide memory latency; four
    and up only buys a larger fp32 partial buffer for the reduction to read).
    """
    p = min(_cdiv(length, 1), max(1, (2 * _num_sms()) // max(1, batch)))
    return p, _cdiv(length, p)


def _bcast(t, x):
    """(ok, (sb, acc_b)) for a per-sample modulation of ``x`` (B, L, D).

    ``sb`` is the element stride between batch entries after expanding to
    ``(B, L, D)`` (0 for a global modulation); ``acc_b`` says the caller's
    gradient must be summed over the batch as well.
    """
    if t is None:
        return True, None
    if not _plain(t):
        return False, None
    B, L, D = x.shape
    if t.dim() == 2:
        if t.shape[1] != D or t.shape[0] not in (1, B):
            return False, None
        t = t.unsqueeze(1)
    if t.dim() != 3 or t.shape[0] not in (1, B) or t.shape[1] != 1 or t.shape[2] != D:
        return False, None
    if t.stride(2) != 1:
        return False, None
    return True, (t.expand(B, L, D).stride(0), t.shape[0] == 1)


def _act(t) -> bool:
    """A contiguous CUDA ``(B, L, D)`` activation of a supported dtype."""
    return _plain(t) and t.dim() == 3


# ── rms_norm_modulate ────────────────────────────────────────────────────────


def is_supported_rms_norm_modulate(x, weight, scale, shift) -> bool:
    """Whether :func:`rms_norm_modulate` can run this call in one Triton pass.

    False for anything the kernel cannot equal exactly: a non-contiguous
    activation, an unsupported device/dtype, a weight that is not ``(D,)``, a
    modulation whose broadcast is not the per-sample ``(B, 1, D)`` / ``(1, 1, D)``
    one, or a ``(B, L, D)`` per-token modulation (eager handles that).
    """
    try:
        if not HAS_TRITON or not torch.cuda.is_available():
            return False
        if not _act(x):
            return False
        D = x.shape[2]
        if weight is not None:
            if not _plain(weight) or weight.dim() != 1 or weight.shape[0] != D:
                return False
        return _bcast(scale, x)[0] and _bcast(shift, x)[0]
    except Exception:  # pragma: no cover - a pathological input, never a crash
        return False


def _rnm_forward(x, weight, scale, shift, eps, offset, need_rstd):
    B, L, D = x.shape
    block_d = triton.next_power_of_2(D)
    rows, warps = _row_cfg(block_d)
    y = torch.empty_like(x)
    rstd = (
        torch.empty(B * L, device=x.device, dtype=torch.float32)
        if need_rstd
        else torch.empty(0, device=x.device, dtype=torch.float32)
    )
    sc_sb = _bcast(scale, x)[1][0] if scale is not None else 0
    sh_sb = _bcast(shift, x)[1][0] if shift is not None else 0
    _rnm_fwd_kernel[(_cdiv(L, rows), B)](
        x,
        weight if weight is not None else x,
        scale if scale is not None else x,
        shift if shift is not None else x,
        y,
        rstd,
        L,
        D,
        float(eps),
        float(offset),
        sc_sb,
        sh_sb,
        HAS_W=weight is not None,
        HAS_SCALE=scale is not None,
        HAS_SHIFT=shift is not None,
        STORE_RSTD=need_rstd,
        ROWS=rows,
        BLOCK_D=block_d,
        num_warps=warps,
    )
    return y, rstd


def _rnm_backward(dy, x, rstd, weight, scale, shift_shape, need_w, need_s, need_sh):
    """(dx, dweight, dscale, dshift) for one rms_norm_modulate forward.

    ``need_*`` come from ``ctx.needs_input_grad``: the backward must not infer
    them from ``requires_grad`` (a saved tensor may not carry it) or from the
    grad mode (autograd runs this with grad disabled).
    """
    B, L, D = x.shape
    dy = dy.contiguous()
    block_d = triton.next_power_of_2(D)
    rows, warps = _row_cfg(block_d)
    p, chunk = _tile_plan(L, B)

    dx = torch.empty_like(x)
    slots = 2 if (need_s or need_w or need_sh) else 1
    part = torch.empty(
        B * slots * p * D if slots > 1 else 1, device=x.device, dtype=torch.float32
    )
    sc_sb = _bcast(scale, x)[1][0] if scale is not None else 0
    _rnm_bwd_kernel[(B, p)](
        dy,
        x,
        rstd,
        weight if weight is not None else x,
        scale if scale is not None else x,
        dx,
        part,
        L,
        D,
        p,
        chunk,
        sc_sb,
        HAS_W=weight is not None,
        HAS_SCALE=scale is not None,
        NEED_A=need_s or need_w,
        NEED_B=need_sh,
        ROWS=rows,
        BLOCK_D=block_d,
        num_warps=warps,
    )
    dweight = dscale = dshift = None
    if need_s or need_w or need_sh:
        w_dtype = weight.dtype if weight is not None else torch.float32
        s_dtype = scale.dtype if scale is not None else torch.float32
        sh_dtype = shift_shape[2] if shift_shape is not None else torch.float32
        # One row per batch entry, or a single row for a global modulation.
        n_s = (1 if scale.shape[0] == 1 else B) if need_s else 0
        n_sh = (1 if shift_shape[0] == 1 else B) if need_sh else 0
        dw = torch.empty(D, device=x.device, dtype=w_dtype) if need_w else dx
        da = torch.empty(n_s * D, device=x.device, dtype=s_dtype) if need_s else dx
        db = torch.empty(n_sh * D, device=x.device, dtype=sh_dtype) if need_sh else dx
        acc_a = need_s and scale.shape[0] == 1
        acc_b = need_sh and shift_shape[0] == 1
        bd = min(128, triton.next_power_of_2(D))
        _col_reduce_kernel[(_cdiv(D, bd),)](
            part,
            weight if weight is not None else dx,
            scale if scale is not None else dx,
            da if need_s else dx,
            db if need_sh else dx,
            dw if need_w else dx,
            D,
            B,
            p,
            slots,
            sc_sb,
            HAS_W=weight is not None,
            HAS_SCALE=scale is not None,
            LOAD_A=need_s or need_w,
            STORE_A=need_s,
            LOAD_B=need_sh,
            STORE_B=need_sh,
            NEED_DW=need_w,
            ACC_A=acc_a,
            ACC_B=acc_b,
            BT=max(8, 4096 // bd),
            BD=bd,
            num_warps=4,
        )
        if need_w:
            dweight = dw
        if need_s:
            dscale = da.view(scale.shape)
        if need_sh:
            dshift = db.view(shift_shape[0], 1, D)
    return dx, dweight, dscale, dshift


class _RmsNormModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, shift, eps, offset):
        # `rstd` is kept fp32 for every differentiable call: d_weight and
        # d_scale need `xhat` even when `x` itself does not require a gradient,
        # and autograd runs `forward` with grad mode off, so the decision cannot
        # be made from the grad mode.
        y, rstd = _rnm_forward(x, weight, scale, shift, eps, offset, True)
        ctx.save_for_backward(x, rstd, weight, scale)
        ctx.shift_shape = None if shift is None else (shift.shape[0], 1, shift.dtype)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rstd, weight, scale = ctx.saved_tensors
        need = ctx.needs_input_grad
        dx, dweight, dscale, dshift = _rnm_backward(
            dy,
            x,
            rstd,
            weight,
            scale,
            ctx.shift_shape,
            need[1],
            need[2],
            need[3],
        )
        return (
            dx if need[0] else None,
            dweight if need[1] else None,
            dscale if need[2] else None,
            dshift if need[3] else None,
            None,
            None,
        )


def rms_norm_modulate(x, weight=None, scale=None, shift=None, eps=1e-5, offset=0.0):
    """``(1 + scale) * rms_norm(x, weight + offset, eps) + shift``, fused.

    ``x`` is a contiguous ``(B, L, D)`` CUDA tensor; ``weight`` is ``(D,)`` or
    ``None``; ``scale``/``shift`` are ``None`` or a per-sample ``(B, 1, D)`` /
    ``(1, 1, D)`` modulation.  Statistics and the modulation run in fp32; the
    output is ``x.dtype``.  Check :func:`is_supported_rms_norm_modulate` first.
    """
    if _needs_grad(x, weight, scale, shift):
        return _RmsNormModulate.apply(x, weight, scale, shift, eps, offset)
    return _rnm_forward(x, weight, scale, shift, eps, offset, False)[0]


# ── silu_mul ────────────────────────────────────────────────────────────────


def is_supported_silu_mul(gate, up) -> bool:
    """Whether ``silu(gate) * up`` can run as one Triton kernel (same-shape,
    contiguous, CUDA, bf16/fp16/fp32)."""
    try:
        return HAS_TRITON and _plain(gate) and _plain(up) and gate.shape == up.shape
    except Exception:  # pragma: no cover
        return False


def _silu_mul_fwd(gate, up):
    y = torch.empty_like(gate)
    n = gate.numel()
    block = 4096
    _silu_mul_fwd_kernel[(_cdiv(n, block),)](gate, up, y, n, BLOCK=block, num_warps=4)
    return y


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return _silu_mul_fwd(gate, up)

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        need = ctx.needs_input_grad
        dy = dy.contiguous()
        n = gate.numel()
        block = 4096
        dg = torch.empty_like(gate) if need[0] else gate
        du = torch.empty_like(up) if need[1] else up
        _silu_mul_bwd_kernel[(_cdiv(n, block),)](
            dy, gate, up, dg, du, n, BLOCK=block, num_warps=4
        )
        return dg if need[0] else None, du if need[1] else None


def silu_mul(gate, up):
    """``silu(gate) * up`` in one pass: one read of each input, one write."""
    if _needs_grad(gate, up):
        return _SiluMul.apply(gate, up)
    return _silu_mul_fwd(gate, up)


# ── rope_apply ──────────────────────────────────────────────────────────────


def is_supported_rope_apply(x, cos, sin) -> bool:
    """Whether the interleaved rotary embedding can run in one Triton kernel.

    False for a non-contiguous ``x``, an odd head dim, ``cos``/``sin`` that are
    not contiguous ``(L, D)``, or a length mismatch.
    """
    try:
        if not HAS_TRITON or not torch.cuda.is_available():
            return False
        if not _plain(x) or x.dim() != 4:
            return False
        B, L, H, D = x.shape
        if D % 2 or H < 1 or B < 1:
            return False
        if not _plain(cos) or not _plain(sin):
            return False
        return cos.shape == (L, D) and sin.shape == (L, D)
    except Exception:  # pragma: no cover
        return False


def _rope_cfg(block_d: int) -> tuple[int, int]:
    """(HB heads per program, num_warps) keeping a tile near 4k elements."""
    hb = max(1, min(16, 4096 // block_d))
    warps = 4 if hb * block_d <= 2048 else 8
    return hb, warps


def _rope_forward(x, cos, sin):
    B, L, H, D = x.shape
    block_d = triton.next_power_of_2(D)
    hb, warps = _rope_cfg(block_d)
    y = torch.empty_like(x)
    _rope_fwd_kernel[(B * L, _cdiv(H, hb))](
        x,
        cos,
        sin,
        y,
        L,
        H,
        D,
        HB=hb,
        BHALF=block_d // 2,
        num_warps=warps,
    )
    return y


class _RopeApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rope_forward(x, cos, sin)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not need[0]:
            return None, None, None
        B, L, H, D = dy.shape
        block_d = triton.next_power_of_2(D)
        hb, warps = _rope_cfg(block_d)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        _rope_bwd_kernel[(B * L, _cdiv(H, hb))](
            dy,
            cos,
            sin,
            dx,
            L,
            H,
            D,
            HB=hb,
            BHALF=block_d // 2,
            num_warps=warps,
        )
        return dx, None, None


def rope_apply(x, cos, sin):
    """Interleaved rotary embedding of a contiguous ``(B, L, H, D)`` tensor.

    ``cos``/``sin`` are contiguous ``(L, D)`` tables; pair ``i`` is
    ``(x[..., 2i], x[..., 2i+1])``::

        out[..., 2i]   = x[..., 2i]   * cos[..., 2i]   - x[..., 2i+1] * sin[..., 2i]
        out[..., 2i+1] = x[..., 2i+1] * cos[..., 2i+1] + x[..., 2i]   * sin[..., 2i+1]

    The tables do not receive gradients (they are position tables).
    """
    if _needs_grad(x):
        return _RopeApply.apply(x, cos, sin)
    return _rope_forward(x, cos, sin)


# ── add_gated ───────────────────────────────────────────────────────────────


def _gate_layout(gate, x):
    """(ok, full, sb) for the residual gate of ``x`` (B, L, D).

    ``full`` is True when the gate has the activation's own shape (read with the
    activation's offsets); otherwise it is a per-sample modulation and ``sb`` is
    the element stride between batch entries.
    """
    if gate is None or not _plain(gate):
        return False, False, 0
    if gate.shape == x.shape:
        return True, True, 0
    ok, lay = _bcast(gate, x)
    if not ok:
        return False, False, 0
    return True, False, lay[0]


def is_supported_add_gated(residual, gate, value) -> bool:
    """Whether ``residual + gate * value`` can run in one Triton kernel.

    ``gate`` is a full-size tensor or a per-sample ``(B, 1, D)`` / ``(1, 1, D)``
    modulation; ``residual`` and ``value`` must be contiguous and the same shape.
    """
    try:
        if not HAS_TRITON or not torch.cuda.is_available():
            return False
        if not _act(residual) or not _act(value):
            return False
        if residual.shape != value.shape:
            return False
        return _gate_layout(gate, residual)[0]
    except Exception:  # pragma: no cover
        return False


def _ag_forward(residual, gate, value):
    B, L, D = residual.shape
    block_d = triton.next_power_of_2(D)
    rows, warps = _row_cfg(block_d)
    _, full, g_sb = _gate_layout(gate, residual)
    y = torch.empty_like(residual)
    _ag_fwd_kernel[(_cdiv(L, rows), B)](
        residual,
        gate,
        value,
        y,
        L,
        D,
        g_sb,
        FULL=full,
        ROWS=rows,
        BLOCK_D=block_d,
        num_warps=warps,
    )
    return y


class _AddGated(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, gate, value):
        ctx.save_for_backward(gate, value)
        ctx.gate_shape = gate.shape
        ctx.gate_full = gate.shape == residual.shape
        return _ag_forward(residual, gate, value)

    @staticmethod
    def backward(ctx, dy):
        gate, value = ctx.saved_tensors
        need = ctx.needs_input_grad
        need_v, need_g = need[2], need[1]
        if not need_v and not need_g:
            return dy, None, None  # d_residual is the upstream gradient itself
        B, L, D = dy.shape
        dy = dy.contiguous()
        _, full, g_sb = _gate_layout(gate, dy)
        block_d = triton.next_power_of_2(D)
        rows, warps = _row_cfg(block_d)
        p, chunk = _tile_plan(L, B) if (need_g and not full) else (1, L)
        dv = torch.empty_like(value) if need_v else dy
        part = (
            torch.empty(B * p * D, device=dy.device, dtype=torch.float32)
            if (need_g and not full)
            else dy
        )
        # One column per batch entry, or one row for a global gate.
        dg = (
            torch.empty_like(dy)
            if (need_g and full)
            else (
                torch.empty(gate.numel(), device=dy.device, dtype=gate.dtype)
                if need_g
                else dy
            )
        )
        _ag_bwd_kernel[(B, p)](
            dy,
            gate,
            value,
            dv,
            dg,
            part,
            L,
            D,
            p,
            chunk,
            g_sb,
            FULL=full,
            NEED_DV=need_v,
            NEED_G=need_g,
            ROWS=rows,
            BLOCK_D=block_d,
            num_warps=warps,
        )
        dgate = None
        if need_g:
            if full:
                dgate = dg
            else:
                acc_b = gate.shape[0] == 1
                bd = min(128, triton.next_power_of_2(D))
                _col_reduce_kernel[(_cdiv(D, bd),)](
                    part,
                    dy,
                    dy,
                    dg,
                    dg,
                    dg,
                    D,
                    B,
                    p,
                    1,
                    0,
                    HAS_W=False,
                    HAS_SCALE=False,
                    LOAD_A=True,
                    STORE_A=True,
                    LOAD_B=False,
                    STORE_B=False,
                    NEED_DW=False,
                    ACC_A=acc_b,
                    ACC_B=False,
                    BT=max(8, 4096 // bd),
                    BD=bd,
                    num_warps=4,
                )
                dgate = dg.view(gate.shape)
        return dy, dgate, dv if need_v else None


def add_gated(residual, gate, value):
    """``residual + gate * value`` in one pass (the gated residual of a DiT block).

    ``gate`` is a full-size tensor or a per-sample ``(B, 1, D)`` / ``(1, 1, D)``
    modulation, read through its strides.  ``d_residual`` is the upstream
    gradient; nothing is launched for it.
    """
    if _needs_grad(residual, gate, value):
        return _AddGated.apply(residual, gate, value)
    return _ag_forward(residual, gate, value)


# =============================================================================
# measurements:  python -m UnifiedTrainer.kernels.blockops_triton
# =============================================================================


def _timeit(fn, iters: int) -> float:
    """Mean milliseconds of ``iters`` calls, CUDA events around the whole loop."""
    for _ in range(max(3, iters // 8)):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _is_grad_arg(arg, first) -> bool:
    """Whether this argument is an input whose gradient the row measures."""
    if not isinstance(arg, torch.Tensor) or not arg.is_floating_point():
        return False
    return arg.shape == first.shape or (arg.dim() == 1 and arg.shape[0] > 1)


def _dtype_tag(dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(
        dtype, str(dtype)
    )


def _fmt_row(name, label, dtype, direction, eager_ms, compile_ms, triton_ms) -> str:
    base = eager_ms if not (compile_ms == compile_ms) else min(eager_ms, compile_ms)
    comp = f"{compile_ms:9.3f}" if compile_ms == compile_ms else "      n/a"
    return (
        f"{name:19} {label:18} {direction + ' ' + _dtype_tag(dtype):8} "
        f"{eager_ms:9.3f} {comp} {triton_ms:9.3f} "
        f"{eager_ms / triton_ms:8.2f}x {base / triton_ms:9.2f}x"
    )


def _gpu_snapshot() -> str:
    """Free memory / utilization, so a contended measurement is visible in the table."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.used,utilization.gpu,clocks.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - a snapshot must never break a measurement
        return f"(nvidia-smi unavailable: {type(exc).__name__})"
    return out or "(no nvidia-smi output)"


def _measure(fns: dict, iters: int, rounds: int = 9) -> dict:
    """Milliseconds per candidate: the **min** over interleaved rounds.

    Interleaved so a contended card slows every candidate of a round about
    equally, and min rather than median because this box's GPU is shared: the
    minimum is the cleanest window each candidate got, which reflects the kernel
    rather than the neighbour process.  Spikes inflate all three columns
    together, so the comparison stays meaningful while the absolute numbers keep
    the caveat the harness prints (the GPU snapshot).
    """
    samples = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            samples[name].append(_timeit(fn, iters))
    return {name: min(v) for name, v in samples.items()}


def _bench_suite(dtype_name: str = "bf16", iters_scale: float = 1.0, compile_ok: bool = True):
    """Measure every op at the Krea-2 shapes (plus one small-D shape) against
    eager and torch.compile; prints the table and returns the rows."""
    from . import eager

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    dev = "cuda"
    dim, ffn, heads, head_dim = 6144, 16384, 48, 128

    def mk(*shape):
        return torch.randn(*shape, device=dev, dtype=dtype)

    def mk_mod(b, d):
        return torch.randn(b, 1, d, device=dev, dtype=dtype)

    def mk_cos_sin(length, d):
        ang = torch.rand(length, d, device=dev, dtype=torch.float32) * 6.2831853
        return ang.cos(), ang.sin()

    cases = []
    for length, d in ((2048, dim), (2048, 256)):
        cases.append(
            dict(
                name="rms_norm_modulate",
                label=f"(1,{length},{d})",
                big=d == dim,
                args=(mk(1, length, d), torch.zeros(d, device=dev, dtype=torch.float32),
                      mk_mod(1, d), mk_mod(1, d)),
                eager=eager.rms_norm_modulate,
                tri=rms_norm_modulate,
            )
        )
    for length, f2 in ((2048, ffn), (2048, 256)):
        cases.append(
            dict(name="silu_mul", label=f"(1,{length},{f2})", big=f2 == ffn,
                 args=(mk(1, length, f2), mk(1, length, f2)),
                 eager=eager.silu_mul, tri=silu_mul)
        )
    for length, h, hd in ((2048, heads, head_dim), (2048, 4, 64)):
        cos, sin = mk_cos_sin(length, hd)
        cases.append(
            dict(name="rope_apply", label=f"(1,{length},{h},{hd})", big=hd == head_dim,
                 args=(mk(1, length, h, hd), cos, sin),
                 eager=eager.rope_apply, tri=rope_apply)
        )
    for length, d in ((2048, dim), (2048, 256)):
        cases.append(
            dict(name="add_gated", label=f"(1,{length},{d})", big=d == dim,
                 args=(mk(1, length, d), mk_mod(1, d), mk(1, length, d)),
                 eager=eager.add_gated, tri=add_gated)
        )

    header = (
        f"{'op':19} {'shape':18} {'dir':8} {'eager':>9} {'compile':>9} {'triton':>9} "
        f"{'vs eager':>9} {'vs best':>9}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    compiled_fns = {}
    for case in cases:
        args = case["args"]
        n = args[0].numel()
        iters = max(6, min(120, int(iters_scale * max(6, 2097152 / max(1, n)))))
        key = (case["name"], tuple(a.shape for a in args))
        if compile_ok and key not in compiled_fns:
            try:
                fn = torch.compile(case["eager"], dynamic=False)
                fn(*args)
                torch.cuda.synchronize()
                compiled_fns[key] = fn
            except Exception as exc:  # noqa: BLE001
                print(f"  (torch.compile failed for {case['name']}: {type(exc).__name__}: {exc})")
                compiled_fns[key] = None

        # forward only
        cands = {"eager": lambda: case["eager"](*args), "triton": lambda: case["tri"](*args)}
        if compile_ok and compiled_fns.get(key) is not None:
            cands["compile"] = lambda: compiled_fns[key](*args)
        res = _measure(cands, iters)
        t_eager, t_tri = res["eager"], res["triton"]
        t_comp = res.get("compile", float("nan"))
        rows.append(dict(op=case["name"], label=case["label"], big=case["big"], direction="fwd",
                         eager=t_eager, compile=t_comp, triton=t_tri))
        print(_fmt_row(case["name"], case["label"], args[0].dtype, "fwd", t_eager, t_comp, t_tri))

        # forward + backward, the same leaves for all three candidates
        grad_args = [
            a.detach().clone().requires_grad_(True) if _is_grad_arg(a, args[0]) else a for a in args
        ]
        leaves = [a for a in grad_args if isinstance(a, torch.Tensor) and a.requires_grad]

        def one(fn):
            out = fn(*grad_args)
            torch.autograd.grad(out, leaves, torch.ones_like(out))

        cands = {"eager": lambda: one(case["eager"]), "triton": lambda: one(case["tri"])}
        if compile_ok and compiled_fns.get(key) is not None:
            cands["compile"] = lambda: one(compiled_fns[key])
        res = _measure(cands, max(4, iters // 2))
        rows.append(dict(op=case["name"], label=case["label"], big=case["big"], direction="f+b",
                         eager=res["eager"], compile=res.get("compile", float("nan")),
                         triton=res["triton"]))
        print(
            _fmt_row(case["name"], case["label"], args[0].dtype, "f+b",
                     res["eager"], res.get("compile", float("nan")), res["triton"])
        )
        del args, grad_args, leaves
        torch.cuda.empty_cache()

    print()
    print("verdict per op (fwd+bwd): ratio = triton / fastest of eager, compile")
    for name in dict.fromkeys(c["name"] for c in cases):
        big = [x for x in rows if x["op"] == name and x["big"] and x["direction"] == "f+b"][0]
        small = [x for x in rows if x["op"] == name and not x["big"] and x["direction"] == "f+b"]
        for r in [big] + small:
            base = r["eager"] if not (r["compile"] == r["compile"]) else min(r["eager"], r["compile"])
            ratio = r["triton"] / base
            verdict = "wins clearly" if ratio < 0.9 else ("loses" if ratio > 1.1 else "about equal")
            print(
                f"  {name:19} {r['label']:16}{'(big)  ' if r['big'] else '(small)'} "
                f"triton {r['triton']:7.3f}  eager {r['eager']:7.3f}  compile {r['compile']:7.3f}"
                f"  vs fastest {ratio:5.2f}x -> {verdict}"
            )
    return rows


def _bench_main(argv=None):  # pragma: no cover - measurement, not a code path
    import argparse

    parser = argparse.ArgumentParser(description="block ops: eager vs torch.compile vs Triton")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--iters", type=float, default=1.0, help="scale the repeat count")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--quiet-gpu", type=float, default=0.0,
                        help="wait (s) up to this long for a mostly idle card before measuring")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required for these measurements")
        return 2
    if not HAS_TRITON:
        print("triton is not importable on this box")
        return 2
    if args.quiet_gpu:
        import time

        deadline = time.time() + args.quiet_gpu
        while time.time() < deadline:
            snap = _gpu_snapshot()
            if " % " in snap and int(snap.split(",")[3].strip().split()[0]) < 10:
                break
            time.sleep(2.0)

    torch.manual_seed(0)
    print(f"torch {torch.__version__}  triton {triton.__version__}  dtype {args.dtype}")
    print(f"gpu snapshot: {_gpu_snapshot()}")
    print()
    _bench_suite(args.dtype, args.iters, not args.no_compile)
    print()
    print(f"gpu snapshot after: {_gpu_snapshot()}")
    print("timing: min over interleaved rounds; eager/compile/triton see the same card.")
    print("        fwd = forward only; f+b = forward + autograd backward (same leaves).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_bench_main())

