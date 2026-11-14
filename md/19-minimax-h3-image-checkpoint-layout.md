# MiniMax-H3-Image: checkpoint-native blocks, one-file LoKR, fp8 base

Status: implemented and validated on the RTX 4090 D box (2026-11-14).
Scope: `models/minimax_h3_image/*`, `networks/lokr_convert.py`, `networks/lokr_module.py`,
`engine/checkpoint.py`, `train.py`, `configs/minimax_h3_image_*.json`.
Validation: three short runs on the training box — see §3 (numbers) and §6 (what was checked).

## 1. The problem this fixes

The vendored DiT used the diffusers module spelling, while MiniMax ships a different one:

| release (`transformer/`) | trainer, before | trainer, now |
|---|---|---|
| `attn.qkv_proj` Linear(hidden, 3·inner), rows head-major `q|k|v` | `attn.to_q`/`to_k`/`to_v`, rows permuted to `[q; k; v]` | `attn.qkv_proj`, unchanged |
| `attn.out_proj` | `attn.to_out.0` (inside a ModuleList) | `attn.out_proj` |
| `mlp.fc1` Linear(hidden, 2·ffn), rows `[gate; value]` | `ff.net.0.proj` (diffusers `FeedForward`), rows `[value; gate]` | `mlp.fc1`, unchanged |
| `mlp.fc2` | `ff.net.2` | `mlp.fc2` |
| `blocks.N.*` | `transformer_blocks.N.*` | `transformer_blocks.N.*` |
| `token_refiner.blocks.N.*` | `token_refiner.refiner_blocks.N.*` | `token_refiner.refiner_blocks.N.*` |

Those two intra-block conversions (the fused-QKV row permutation, the MLP half swap) are
reversible on a dense weight but **not** on a Kronecker-factorised LoKR delta. Exporting a
trainable checkpoint for ComfyUI therefore had to materialise three separate `to_q`/`to_k`/`to_v`
deltas, stack them into ComfyUI's `[q; k; v]` row order, and re-factorise that sum of three
Kronecker products with an SVD — measured 46–73 % relative error, and the result could not be
resumed.

## 2. What the layout change buys

* **One file per checkpoint.** Written straight into the release's spelling
  (`blocks.9.attn.qkv_proj.lokr_w1`, no `lora_unet_` prefix, no `_comfyui` sibling). ComfyUI's
  MiniMaxH3 branch registers exactly those dotted paths in its LoRA key map, and
  `LokrNetwork.load_weights` maps them back with two prefix rewrites, so the same file resumes
  training and loads in `LoraLoaderModelOnly`.
* **Lossless.** Full-rank factor 8: 626 MB instead of 1.87 GB (the difference is the rank-512
  SVD pair the fused qkv used to need), and the factors are the trained ones exactly
  (`lokr_w1 [8,8]`, `lokr_w2 [2688,672]` for the `[21504,5376]` qkv).
* **No second pass over the weights** on save: `save_lokr` writes the file and stops.
* **Correct default targets.** `lokr_target_modules: null` now resolves to
  `attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2` on both block stacks (208 modules for
  50+2 blocks) instead of *every* `nn.Linear`, which used to adapt the 13 B AdaLN projections.

## 3. fp8 base (`training.quantize: "fp8"`)

`Fp8Linear` (`models/minimax_h3_image/fp8_linear.py`) keeps the frozen weight as
`float8_e4m3fn` plus one `float32` scale per output row and runs `torch._scaled_mm`; the
backward only needs the input gradient, which uses the weight dequantised on the fly.

Wiring (`quantize.py`):

```text
build_meta_model_fp8(...)   meta build; every quantizable Linear -> meta Fp8Linear
                            (`_keep_in_fp32_modules` = proj_in / time_embedder /
                            pdd_proj_out / rope stay full precision)
load_streaming_fp8(...)     stream the bf16 shards, quantize each layer on the GPU
                            (one shard in RAM; ~4 min per start)
load_fp8_into(...)          optional: restore a snapshot (transformer_fp8.safetensors)
save_fp8(...)               optional: write that snapshot
```

**Default: no snapshot, no extra path.** `training.quantize: "fp8"` alone quantizes the base
from the bf16 shards at every start and writes nothing to disk. A 33 GB fp8 snapshot is only
produced when the config explicitly names `training.quantize_path` *and* sets
`save_prequantized: true`; the same pair makes a later start load it instead.

Measured on the 48 GB card, 1024², LoKR full-rank factor 8:

| base | weights | peak VRAM | s/step | file |
|---|---|---|---|---|
| nf4 snapshot (16 GB) | 16.0 GB | 27.1 GB | ~8 | 626 MB + json |
| fp8 snapshot (33 GB) | 30.9 GB | 40.3 GB | ~7 | 626 MB + json |

fp8 needs two things to fit at 1024²: EMA off (`training.ema_decay: 0`, the shadow copy is
another 1.25 GB) and `mixed_precision: bf16`. Above 1024² it does not fit.

Three hard constraints of `_scaled_mm` (already encoded in the layer, do not "simplify"):

1. scales must be **float32** — an `amax` on a bf16 activation yields bf16 and `_scaled_mm`
   rejects it;
2. `mat2` must have `stride(0) == 1` — pass the transpose **view** of a row-major `[out, in]`
   weight, never a materialised transpose;
3. the output dtype must be bf16/fp16 for row-wise scaling, while the block stack can hand the
   layer **fp32** activations (the checkpoint's `proj_in`/`time_embedder` stay fp32). The layer
   therefore runs the GEMM in bf16 and casts back (a differentiable cast), which is what
   bitsandbytes does internally for its 4-bit layers.

## 4. Migration (old artifacts are not loadable)

The layout change renames modules, so anything written by the old code cannot be read:

* an **nf4 snapshot** from the old layout fails in `load_nf4_into` with
  `nf4 snapshot ... has weights for unknown module 'token_refiner.refiner_blocks.0.attn.to_k'`;
* an old **LoKR file** (`lycoris_*` with `attn.to_q`) matches no target and
  `load_weights` warns `matched NO tensors`; `convert_h3_state_dict` raises instead of silently
  producing an empty file.

Rebuild path: point `training.quantize_path` at a fresh directory (or delete the old one) and
run once with `quantize: "prequantized"` + `save_prequantized: true`. The run streams the bf16
shards, writes the new snapshot (16 GB, ~2 min) and continues training. For fp8 use
`quantize: "fp8"`; its snapshot is 33 GB and takes ~4 min to build.

## 5. Pitfalls fixed while validating (do not regress)

| symptom | cause | fix |
|---|---|---|
| `The expanded size of the tensor (917) must match the existing size (2688)` on the first step, from deep inside `bnb.matmul_4bit` | `load_streaming_quantized` wrapped **every** tensor of a 4-bit module in `Params4bit`, so a `[5376]` bias became a packed `uint8 [2688,1]` | only wrap `attr == "weight"`; the same guard was added to `stream_hf_shards_into`, and `load_nf4_into` now rejects a malformed bias with an actionable message |
| fp8: `CUDA out of memory` right after the model loads, "47.21 GiB in use" | `train.py`'s device move did `.to(device, dtype=bf16)`; `float8_e4m3fn` **is** a floating dtype, so the weight buffers were upcast (33 → 66 GB) | `quantize` in (`nf4`,`int8`,`prequantized`,`fp8`) now takes the device-only branch |
| fp8: `Only bf16 and fp16 high precision output types are supported for row-wise scaling` | the first rows of the packed sequence arrive as fp32 (`proj_in` is a float32 module), and the layer passed `out_dtype=x.dtype` | run the GEMM in bf16 and cast back (see §3 rule 3) |
| `Cannot copy out of meta tensor` when a meta-built model is moved | `rope.inv_freq` is **non-persistent**, so it is neither in the snapshot nor replaced by `assign=True` | `MiniMaxH3ImageRotaryPosEmbed.reset_inv_freq()`; the streamed loaders call it with the device |

## 6. What was checked end-to-end

1. **nf4, 1 epoch (51 steps)**: 208 LoKR modules / 313.1M params, avg_loss 0.28–0.30,
   peak 27.1 GB, saved `h3i_layout-0/h3i_layout_epoch0.safetensors` (626 MB) + json and
   `h3i_layout_final.safetensors`; no `_comfyui` sibling.
2. **Key inventory vs the previously ComfyUI-verified export**: same 208 modules, same
   624 keys (`lokr_w1`, `lokr_w2`, `alpha`), identical module set — the only difference is
   that the fused qkv module now carries exact Kronecker factors instead of a rank-512
   `lora_down`/`lora_up` pair (hence 1.87 GB → 626 MB).
3. **Resume**: `--resume <that file>` logs `Loaded LoKR weights ... (416 tensors into 208
   modules)` and trains on.
4. **fp8, 12 steps**: 261 fp8 modules / 30.9 GB weights, peak 40.3 GB, ~7 s/step, avg_loss
   0.30, same single-file checkpoints.
5. **Unit-level checks** (run in a scratch session, not shipped): the fused split equals the
   release's head-major `q|k|v` rows, `mlp.fc1` is `[gate; value]` with `value*silu(gate)`,
   `map_transformer_keys` is one-to-one, and an fp32 save reloads bit-exactly.

## 7. ComfyUI

`E:\models\loras` — drop the single epoch file in, pick it in `LoraLoaderModelOnly`, no
`_comfyui` suffix and no second file. The key inventory is bit-identical to the previously
verified export: 208 modules × (`lokr_w1`, `lokr_w2`, `alpha`), dotted
`blocks.N.*` / `token_refiner.blocks.N.*`.
