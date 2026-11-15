# Attribution — vendored ai-toolkit FLUX.3 code

Every file in this directory is either a **verbatim copy** of, or **derived from**,
code from **AI Toolkit** by **Ostris, LLC**:

| | |
|---|---|
| Original author | **Ostris, LLC** ("ostris") |
| Upstream repository | <https://github.com/ostris/ai-toolkit> |
| Upstream commit | `12395a103fd0ca499c631dc6f460a8778e11490d` (2026-10-15, *"Load comfy files if they exist in flux.3 image"*) |
| Licence | **MIT** — Copyright (c) 2024 Ostris, LLC |
| Local reference clone | `<repo>/ref\ai-toolkit` (read-only) |
| Vendored on | 2026-10-16, into `UnifiedTrainer/vendor/flux3_aitoolkit/` |
| Why | UnifiedTrainer's `flux3_klein` adapter and ai-toolkit's `flux3` extension both implement FLUX.3 Image Klein; the upstream code is the second independent implementation and the reference for the shipped checkpoint's native (BFL/ComfyUI) weight layout. Nothing is imported from ai-toolkit at runtime. |

## Upstream licence text

```
MIT License

Copyright (c) 2024 Ostris, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Per-file provenance

| File here | Upstream file (repo-relative) | Change |
|---|---|---|
| `positional.py` | `extensions_built_in/diffusion_models/flux3/src/positional.py` | verbatim (attribution banner prepended only) |
| `checkpoints.py` | `extensions_built_in/diffusion_models/flux3/src/checkpoints.py` | verbatim |
| `pipeline.py` | `extensions_built_in/diffusion_models/flux3/src/pipeline.py` | verbatim |
| `image_vae.py` | `extensions_built_in/diffusion_models/flux3/src/image_vae.py` | verbatim |
| `text_encoder.py` | `extensions_built_in/diffusion_models/flux3/src/text_encoder.py` + `toolkit/models/v2/text_encoders/qwen3_vl.py` | adapted: the `Flux3TextEncoder`/`Qwen3VLTextEncoder` classes (which depend on `toolkit.*` mixins) are replaced by free functions; `encode_flux3_prompt`, `padded_token_length` and the Comfy key conversion are unchanged; `drop_vision_tower` / `patch_qwen_vl_patch_embed` are re-implemented to take a plain model |
| `recipe.py` | `extensions_built_in/diffusion_models/flux3/flux3.py` + `src/pipeline.py` | extracted: constants and the `_fit_reference` / LoRA-prefix / quantization-exclude helpers, values unchanged |
| `keymap.py` | `extensions_built_in/diffusion_models/flux3/src/transformer.py` (module tree) + `toolkit/util/comfy_quant_import.py` (`split_fused_quantized_keys`) | new code: the key mapping written down explicitly, the reverse direction, the fused-qkv split for the plain case, layout/flow-map sniffing. Cross-checked against diffusers' `convert_flux3_transformer_checkpoint_to_diffusers` (`diffusers/loaders/single_file_utils.py:4017`), including its `swap_scale_shift` on the output adaLN weight |

The banner comment at the head of each file repeats the author, the upstream
repository/files and the commit, and states whether the file is verbatim,
adapted or derived.

## Maintenance

If ai-toolkit's FLUX.3 extension changes, re-vendor as a whole file and re-run
`python -m pytest tests/test_flux3_aitoolkit_keymap.py -q` (the key map is pinned
against a real module tree and against diffusers' own converter).
