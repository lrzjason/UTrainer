# Attribution — `UnifiedTrainer/kernels`

Every file in this package is either an **independent re-expression** of, or
**derived from**, code and design from **KohakuMoss** by **Kohaku-Blueleaf**:

| | |
|---|---|
| Original author | **Kohaku-Blueleaf** (KohakuMoss, KohakUwULLM, LyCORIS) |
| Upstream repository | <https://github.com/KohakuBlueleaf/KohakuMossTrainer> |
| Upstream version | `kohakumoss` 0.1.0 (repository `CLAUDE.md`, `pyproject.toml`) |
| Licence | **KohakuMoss License 1.0** — Apache-2.0 plus one additional term (a derivative work must carry "Kohaku" in its name; this directory keeps the attribution and is not redistributed as a separate product) |
| Local reference clone | `E:\UnifiedTrainer\ref\KohakuMossTrainer` (read-only, never imported) |
| Ported on | 2026-10-19, into `UnifiedTrainer/kernels/` |
| Why | KohakuMoss is the reference for a *measured* kernel-selection stack for diffusion transformers: one normative eager implementation, per-arch kernels, and a per-(op, arch, dtype, shape) tuning table. UnifiedTrainer's adapters had no such layer; on this project's target (Windows + sm120, torch built without flash attention) their attention-path finding applies directly. |

## What "derived" means here, precisely

Nothing is imported from the reference clone at runtime, and no file is a
verbatim copy of an upstream file: the ops are re-expressed for
UnifiedTrainer's layouts (`(B, L, H, D)` attention, diffusers-style blocks,
no packed `SeqInfo` axis) and for `torch`/`triton` only. What is taken from
KohakuMoss is the **design**, and it is credited per file:

| File here | Upstream source of the design | Change |
|---|---|---|
| `eager.py` | `src/kohakumoss_kernels/backends/eager/*` — "the eager ops are the reference", and its `rms_norm` / `rope` / `ffn` / `residual` semantics | new code: the same ops over `(B, L, D)` / `(B, L, H, D)` with diffusers' rotary convention; adds `sdpa_gqa`, the KV-expand fallback that keeps SDPA on a fused kernel |
| `__init__.py` | `src/kohakumoss_kernels/registry.py` (`resolve` -> `Bound op`), `errors.py` (the "next backend" fallback chain), `constraints.py` | new code: a three-mode selector (`off`/`auto`/`triton`) instead of a registry, because UnifiedTrainer has one arch family and eight adapters; `install_attention_backend()` is the equivalent of their bind-time binding for a third-party call site |
| `attention_triton.py` | `src/kohakumoss_kernels/backends/triton/attention/_kernels.py`, `.../varlen.py` — the online-softmax formulation, the two-plane `m` / `log2(l)` row statistics, `exp2`-based exponentials, the fp32-accumulate / output-dtype-store discipline, the `dS` cast rule | re-implemented for dense `(B, L, H, D)` with grouped-query KV heads, a broadcast key mask, and no `cu_seqlens` packing layer |
| `blockops_triton.py` | `src/kohakumoss_kernels/backends/triton/blocks/{norm, rope, ffn, residual}.py` and the `_blockops.py` hub — fused single-pass block ops, fp32 statistics inside the kernel, rowwise gradient without materialised intermediates | re-implemented for `(B, L, D)` / `(B, L, H, D)`; the Krea-2 "weight + 1" RMSNorm offset and the diffusers interleaved rotary convention are this project's requirements |
| `bench.py` | `src/kohakumoss_kernels/tuning/{tuner, buckets, timing, prewarm}.py` — the methodology: a candidate is selected only from a measurement on this card, per (op, arch, dtype, shape); the eager op is always the reference column | new code: a small CLI that prints the table the report quotes; no persisted cache yet (see "Not ported") |
| `select.py` | `src/kohakumoss_kernels/tuning/{decisions, window}.py` and `registry.py`'s "measured, never guessed per call" rule | new code: the minimal form — one synthetic fwd+bwd probe per shape, one cached bit, `UTR_KERNELS_TUNE=auto\|off\|eager`. Needed because the Triton attention wins for MHA at every length but loses for GQA below ~1.5k keys, so a static "always Triton" would slow Krea-2 512 px runs |
| `patch.py` | `src/kohakumoss_kernels/registry.py` — a kernel is bound to an op once at build time, the per-step path calls a plain attribute | new code: per-block `forward` swaps (reversible) plus a shim over diffusers' `apply_rotary_emb` |

## Not ported (and why)

- `backends/tilelang/**` — TileLang is not installed here, and sm120 is served by
  Triton; the TileLang backend exists upstream for sm70/sm75 (V100, 2080 Ti).
- `backends/triton/textenc/**`, `fp8/**`, `quant/**` — the text-encoder and
  fp8/int8 weight-only kernels. UnifiedTrainer already quantizes the frozen base
  through `torchao` (`training.quantize: torchao_float8`), and the reference's own
  measurement is that fp8 *compute* loses for LoRA training because the 16-bit
  activation still has to be quantized per call.
- `tuning/replay.py`, the persisted 240-entry table, `hardware/toolchain.py` —
  the per-card table is only worth persisting when a run tunes many shapes; this
  port selects per call-site with a shape check instead. `bench.py` is the entry
  point if that changes.
- `kohakumoss_common` (logger/registry) and `kohakumoss_data` — UnifiedTrainer
  has its own.

## Verification

- `tests/test_kernels_patch.py` — hook install/uninstall, eager parity, the
  Krea-2 GQA fix against the previous `enable_gqa=True` result, trainer wiring.
- `tests/test_kernels_attention.py` — the Triton attention against
  `kernels/eager.py` and a float64 ruler (cosine + abs/rel error), masks, GQA,
  dtype and length edge cases.
- `tests/test_kernels_blockops.py` — the fused block ops against `eager.py` and
  a float64 ruler, gradients included.
- Measurements and the verdicts they produced: `md/14-kernel-optimizations-applied.md`.

## Maintenance

If KohakuMoss's kernels change, re-read the upstream file, re-derive, and keep
the numeric contract pinned by the tests above. Do not copy upstream code that
depends on `kohakumoss_common`, `kohakumoss_data` or the tuning store — those
dependencies are deliberately absent here.
