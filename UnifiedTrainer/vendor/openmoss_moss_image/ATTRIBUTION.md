# Attribution — vendored official MOSS-Image model

Every file in this directory is either a **verbatim copy** of, or **derived from**,
code released with **MOSS-Image** by the **OpenMOSS Team**:

| | |
|---|---|
| Original author | **OpenMOSS Team** |
| Upstream | `openmoss/MOSS-Image-Base` on ModelScope — <https://modelscope.cn/models/openmoss/MOSS-Image-Base> (also mirrored as `OpenMOSS-Team/MOSS-Image-Base` on Hugging Face: <https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base>) |
| Upstream file | `pipeline_moss_image.py` (repository root, self-contained: "copy this file anywhere and import it normally") |
| Model version | **MOSS-Image-Base**, the dense text-to-image model published in that repository (`model_index.json`: `_diffusers_version` 0.38.0); snapshot pulled 2026-10-19 |
| Licence | **Apache-2.0** — the repository `LICENSE` is the Apache License, Version 2.0 (202 lines, <http://www.apache.org/licenses/LICENSE-2.0>) |
| Local reference copy | `E:\hf_models\MOSS-Image-Base\pipeline_moss_image.py` (read-only for us; nothing is imported from it at runtime) |
| Vendored on | 2026-10-19, into `UnifiedTrainer/vendor/openmoss_moss_image/` |
| Why | UnifiedTrainer's MOSS-Image adapter needs the official `MOSSImageTransformer` graph (RoPE layout, shared modulation, SiTU FFN, qk-norm, final head) as the single source of truth. The file is the model author's own release, so it is vendored instead of re-implemented. |
| Changes | **Attribution banner only.** The body below the banner is byte-identical to the released file: source SHA-256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31` (25 060 bytes); vendored file SHA-256 `e68151c4b087c778edc807302e637b473cd64a408dc4a1ed14e2ec4a4b476fd0` (25 862 bytes = banner + body). |

## Upstream licence

The MOSS-Image-Base repository states Apache-2.0:

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
```

The full 202-line licence text ships as `LICENSE` in the model repository
(`E:\hf_models\MOSS-Image-Base\LICENSE`). It contains the stock Apache-2.0
"Copyright [yyyy] [name of copyright owner]" appendix line and no additional
copyright holder statement.

## Per-file provenance

| File here | Upstream file (repo-relative) | Change |
|---|---|---|
| `pipeline_moss_image.py` | `pipeline_moss_image.py` | verbatim; attribution banner prepended only (body byte-identical, SHA-256 above) |
| `__init__.py` | — (no upstream counterpart) | new file: re-exports `MOSSImageTransformer`, `MOSSImageScheduler`, `MOSSImagePipeline`, `_RoPE`, `_Block` |
| *(not vendored here)* timestep / sigma sampling semantics | `ref/KohakuMossTrainer`, `src/kohakumoss/diffusion/timestep/moss.py` | **Separate attribution, not part of this file.** The *training* sigma sampler semantics used by `UnifiedTrainer/models/moss/pack.py` — a discrete 1000-step sigma grid `j/steps`, a truncated logit-normal pmf, a uniform mix, mass moved to `sigma = 1`, and a static `shift_table` shift — come from **Kohaku-Blueleaf's KohakuMossTrainer** (<https://github.com/KohakuBlueleaf/KohakuMossTrainer>; KohakuMoss 1.0 = Apache-2.0 + naming term). They are distinct from the *inference* `MOSSImageScheduler` vendored here (which snaps to the training grid with shift 5.0 / training_shift 1.0). No code from KohakuMossTrainer is copied into this directory. |

## Notes on the vendored file

- The file imports only `torch`, `diffusers` and `transformers` (plus the
  standard library). The sampling-only imports (`AutoencoderKLFlux2`,
  `DiffusionPipeline`, `SchedulerMixin`, `AutoModel`, `AutoTokenizer`, …) are
  kept unchanged as shipped; the model graph does not require them at call
  time, but importing the module does import those third-party packages.
- `_RoPE`, `_Block`, `_TextBlock`, `_TextProjection` and `_RMSNorm` are private
  by name and public de facto: the checkpoint's state-dict keys and every
  positional argument of the graph depend on them.
- The timestep convention of `MOSSImageTransformer.forward` is **not** coded
  into the file: `timestep` is used as a raw angle scale (`timestep * 10000^(-k/half)`),
  so it expects **sigma * 1000** (the inference pipeline's `FlowMatchEuler`
  timesteps), not sigma in [0, 1]. Verified empirically in the P1 gate report
  (scheduler `timesteps / sigmas` = 1000).

## Maintenance

Re-vendor as a whole file if the model repository updates
`pipeline_moss_image.py`; update both SHA-256 values above and re-run the
P1 load/forward gates (Gate A: 0 missing / 0 unexpected keys; Gate B: forward
shape, finiteness, determinism, mask semantics).
