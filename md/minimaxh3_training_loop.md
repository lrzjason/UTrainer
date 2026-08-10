# MiniMax-H3 — Training Architecture & Training Loop

Companion to [minimaxh3_diff.md](./minimaxh3_diff.md) (implementation review vs.
diffusers PR-14355 / ComfyUI). This document summarizes **what is trained** and
**how the training loop runs**, as implemented in `UnifiedTrainer` (adapter
`models/minimax_h3/__init__.py`, engine `engine/trainer.py`, loss
`losses/flow_matching.py`).

---

## 1. What is being trained

MiniMax-H3 is a **33B joint video+audio diffusion transformer** (50 DiT blocks,
hidden size 5376). UTrainer trains a **LoRA or LoKR** on the frozen transformer,
using the **image-pair path** (milestone 1: reference image R → target image T)
and the **video path** (milestone 2: short video clips). Audio is **excluded**
this phase (D6): the packed sequence carries zero audio rows.

The forward pass runs over a **single packed 1-D sequence**:

```
[ text (L) | keyframe condition rows (C) | target audio (A=0) | target video rows (V) ]
```

| Component | Role | Loaded from | Used in training? |
|---|---|---|---|
| `MiniMaxH3Transformer3DModel` (33B) | the LoRA base | `<model_path>/transformer_ref` (Ref2VA omni-reference checkpoint; config `transformer_subfolder`, default `transformer`) | ✅ forward/backward |
| `AutoencoderKLMiniMaxH3` | pixels ↔ 24-ch latents | `<model_path>/vae` | cache build only (offloaded after) |
| Qwen3-VL text encoder | caption → embeddings | `<model_path>/text_encoder` | cache build only (offloaded after) |
| tokenizer / processor | text tokenization | `<model_path>/tokenizer`, `processor` | cache build only |
| `MiniMaxH3Scheduler(shift=12)` | σ schedule (train + val) | constructed directly | ✅ |
| audio VAE / audio scheduler | — | not downloaded (D6) | ❌ excluded |

**Checkpoint variants** (`transformer/` vs `transformer_ref/`): same architecture,
different weights. `transformer/` = FL2VA (text + keyframe → video);
`transformer_ref/` = Ref2VA (omni-reference: up to 9 images / 3 videos / 3
audio as labeled in-sequence blocks). For single-reference image editing both
pack the condition the same way (keyframe rows at `t_cond = 0.999`), so the
ref checkpoint is the flexible base — it also supports the multi-reference
packing upgrade later (order-aware rotary clock, labeled blocks). The adapter
reads `training.adaln_surrogate` and the loader reads `transformer_subfolder`
(`transformer_ref` in `configs/minimax_h3_train.json`).

### 1.1 AdaLN modulation surrogate (built inside training)

~13B of the 33B params are **frozen AdaLN projections**: each
`transformer_blocks[i].adaln_proj.linear` (2688 → 96768, bf16) plus
`norm_out.linear` (2688 → 10752) project `silu(temb(t))` into the per-row
modulations. They never receive gradients, so `adaln_surrogate.py` replaces
them at load time with a **low-rank continuous function of the timestep**:

```
M[t, :] ~= u(t) @ V,   u(t) = cheb(t) @ C     (V fp16, C fp32)
```

- **Build runs inside training** — `adapter.load_transformer` calls
  `install_surrogate` right after `from_pretrained` (first run ~5-15 min:
  streams the shards block by block, peak RAM ~2-3 GB, transient temp ~10 GB;
  fp64 eigh of the grid Gram matrix). Cached as
  `<transformer_dir>/adaln_surrogate_r{rank}.safetensors` + JSON sidecar —
  later runs load the cache (validated against `config.json`).
- **GPU per-block build**: `device` (default `"auto"`) runs each block's
  matmuls on the GPU when CUDA is available — the block's `W/bias` is loaded,
  moved to the GPU, `E @ W.T` computed, result transferred back; the fp64
  Gram accumulation stays on the CPU either way (exactness). `"cpu"` forces
  the CPU path, `"cuda"` forces GPU (falls back with a warning).
- **Progress bars**: pass1 (sample grid), pass2 (build V), pass3 (held-out)
  and the bf16 baseline each show a tqdm bar over blocks; the nf4/int8/torchao
  weight-quantization loops show bars over linear layers too.
- **Config**: `training.adaln_surrogate = {enabled (default true), rank
  (default 64, clamped [32,128]), grid (default 1024), device (default
  "auto")}`. The fitting grid is
  Chebyshev nodes (uniform points + high-degree basis suffer Runge-like
  off-grid error); the surrogate stays continuous in t — no snap/lerp table.
  Directions below 1e-6 relative singular value are dropped (numerical rank;
  they would overflow fp16 V).
- **Runtime**: a forward pre-hook reads the packed sequence's distinct
  `timestep` tensor (kwargs or positional), computes `V @ (cheb(t) @ C)`
  (~1 ms GPU at R=64, K=2) and fills a per-forward cache; each `linear` is
  replaced by a parameter-free lookup module (invisible to state_dict, PEFT,
  quantization, gradient checkpointing).
- **Net effect**: ~26 GB bf16 freed, ~620 MB fp16 added (R=64). Held-out
  reconstruction error is at/below the bf16 table baseline (~1e-3 relative);
  the tiny `time_embedder` MLP (~16K params) stays — the diffusers forward
  calls it every step and the lookups ignore its output.
- The swap happens **before** NF4/int8 quantization, so the 13B AdaLN weights
  are never quantized and never reach the GPU.

Key semantics (all PR-exact):

- **Data-ward velocity**: the model predicts `v = x0 - x_t`
  (`adapter.velocity_sign == "data_ward"`; `compute_target` returns
  `learning_target - noise`).
- **Timestep convention**: the engine samples sigma (`t = 1 - σ` in PR terms);
  interpolation `(1-σ)·x0 + σ·noise` is **bit-identical** to the PR's
  `t·x0 + (1-t)·noise`.
- **Spatial-only patch**: patch `(1, 2, 2)` — no temporal patch.
- **Modality tags**: video = 0, text = 1 (audio = 2 unused); position IDs are a
  float64 3-D grid shared across the batch.
- **Batch-uniform contract**: one packed layout per forward — one sigma per
  batch, same text length/tag pattern across samples.

## 2. Keyframe conditioning (the C rows)

- The source image S is the **keyframe condition**; its clean latent is
  **noise-augmented**, not fed in clean:
  `scale_noise(clean, 0.999) = 0.999·x0 + 0.001·noise`
  (`MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999`, packing.py L82-84).
- The mix happens in **float32** — in bf16 the 0.001 term would be swallowed by
  the ulp (~0.0078).
- Condition rows are **pinned at timestep `t_cond = max(t_target, 0.999)`** for
  every denoising step (`before_denoise.py` L417 parity).
- **RNG policy** (this is the notable design point, fixed in the review):
  - *Training / eval-loss*: condition noise is drawn **per forward** with a
    sigma-derived seed (`base_seed + round(σ·1e6)`) — CFG-consistent at a given
    σ, fresh noise at every distinct training σ.
  - *Validation generation*: condition rows are built **once per request**
    (`build_condition_rows`, fixed seed) and reused across all 20 denoising
    steps — PR parity (`PrepareLatentsStep` freezes the condition for the whole
    trajectory). One tensor serves both CFG forwards (the uncond batch shares
    the same latents).

## 3. Data & cache pipeline

Config-driven (`minimax_h3_train.json`):

```json
"image_configs":    { "T": {"suffix": "_t", "media": "image"},
                      "R": {"suffix": "_r", "media": "image"} },
"target_configs":   { "T": [{"image": "T"}] },
"reference_configs": { "R": [{"image": "R", "sample_type": "from_same_name"}] },
"caption_configs":  { "C": {"ext": ".txt", "image": "R"} }
```

- **Image pair**: `T` (target, `_t`) and `R` (reference/keyframe, `_r`) from
  the same base name; the caption `.txt` is paired with **R** (vision-block
  presentation: `encode_text_accepts_image == True`).
- **Video** (milestone 2): `V` = `.mp4`, `video_frames: 124`, caption paired
  with the video.
- **Cache build** (one-time, per `cache_dir`): VAE encodes every image/video
  frame-group to 5-D latents `(B, 24, T, H, W)` (`vae_scale=8`, spatial-only
  `_encode_clip` for images — never the 17-frame-clip `_encode`); the text
  encoder produces prompt embeddings from `hidden_states[50]`
  (`MINIMAX_H3_TEXT_ENCODER_LAYER`), stored as `.npz`. Empty-embedding and
  suffix-embedding files are created alongside.
- **Text encoder quantization**: the cache process loads the Qwen3-VL encoder
  (~15 GB bf16); `training.text_encoder_quantize` (`none|nf4|int8`, default
  `nf4` in the example config) quantizes it with the same bnb helpers as the
  transformer (progress bar over the Linear layers), keeping the cache phase
  within VRAM — ~4 GB at nf4 vs ~15 GB bf16. Only the frozen cached
  embeddings are affected; set `"none"` for max embedding fidelity.
  Quantization happens after load, before `.to(device)` (Params4bit/Int8Params
  quantize on the GPU move).
- **Buckets**: three `RESOLUTION_CONFIG` tiers — 768 (19 buckets, short edge
  768, area ≤ 768×1344), 1024 (9 buckets), 1280 (9 buckets); every axis a
  multiple of 16 (`vae_scale 8 × patch 2`); aspect-preserving cover-crop. One
  bucket per batch. Image training can go beyond 768p by setting the dataset
  (and validation) `resolution` to the tier's short edge, e.g. 1024.
- After cache build the VAE and text encoder are **offloaded** (default
  `offload_vae: true`, `offload_text_encoder: true`); training touches only the
  transformer.

## 4. Training loop — per micro-batch

The loop is the unified `trainer.train_epoch` (generic engine) + H3-specific
behaviors injected by the adapter. Per batch:

1. **Move & cast**: latents → CUDA, compute dtype = bf16 (forced bf16 when the
   base is quantized nf4/int8 — parameter storage dtype is never used as the
   compute dtype).
2. **Resolve targets**: `T` target latents (multi-target supported; H3 uses one).
3. **Optional perturbations** (off by default): Helios frame-aware corrupt on
   reference latents; dynamic caption dropout.
4. **Noise/timestep selection** (`noise_selector.select`): default is one σ per
   batch from a **logit-normal** with configurable `timestep_mu` (default 0.0):
   `u ~ N(mu, 1)`, `σ = sigmoid(u)`, clamped `[1e-5, 1-1e-5]`; broadcast over the
   batch. Alternatively `{"type": "explorative", "K_cond": 2, "K_uncond": 5,
   ...}` selects the **Explorative Noise Selector** (engine/noise_selector.py)
   — it re-samples σ around the current optimum per step (CFG-aware, K_cond
   candidates for the conditional forward, K_uncond for the unconditional), and
   works for H3 because it only relies on `adapter.velocity_sign`
   (`"data_ward"`) and the batch's `_caption_dropped` flag. Noise
   `ε ~ randn_like(target)`.
5. **Flow-matching interpolation**:
   `noisy = (1-σ)·x0 + σ·ε` (one noisy latent per target).
6. **`prepare_model_input(batch, noisy, sigmas)`** — the H3 core:
   - condition rows: noise-aug `0.999·x0 + 0.001·ε` (float32), seed derived
     from `base + round(σ·1e6)`;
   - patchify all latents `(1,2,2)`, build the packed sequence
     `[text | cond | target]`, per-row timesteps `t = 1-σ`,
     `t_cond = max(t, 0.999)`, `keyframe_anchors=("first",)`;
   - emit `hidden_states`, empty `audio_hidden_states (B, 0, 32)`, prompt
     embeds, `timestep`/`timestep_indices`, `token_tags`, float64
     `position_ids`, `video/audio/text_indices`.
7. **Forward**: `transformer(**model_input)` under `accumulate()` (gradient
   accumulation; autocast only for non-bf16). Output `(video, audio)` tuple.
8. **Unpack**: slice the target rows (`video[:, C:C+V]`), unpatchify back to
   5-D `(B, 24, 1, H, W)` per target.
9. **Loss** (`flow_matching`, weight 1.0):
   `target = learning_target - noise` (data-ward flip), MSE
   `((pred - target)²).mean()`; optional SD3-style weighting
   `1/(σ²+1)` (`use_weighting`); summed across targets then **normalised by
   target count** so the scale is independent of target count.
10. **Backward → clip → step**: `accelerator.backward`, grad-norm clip
    (`max_grad_norm > 0`), `optimizer.step()`, `lr_scheduler.step()`,
    `zero_grad(set_to_none=True)`, EMA update (if configured).
11. **Aggressive cleanup**: delete intermediate tensors, `batch.clear()`,
    `gc.collect()` + `torch.cuda.empty_cache()` every 50 steps or on batch
    shape change (CUDA caching-allocator fragmentation control).
12. **Bookkeeping**: step counter advances only on sync (accumulation-aware);
    progress bar with per-loss breakdown; slow-step timing log
    (data/swap/fwd/bwd/optim/cleanup, VRAM); callbacks (`on_step_end` → WandB);
    optional ScheduledTrainer hook manager.

## 5. Optimizer / LoRA-LoKR specifics

From `minimax_h3_train.json`:

| Setting | Value | Note |
|---|---|---|
| `network_type` | `lokr` | LoKR on the frozen base (fall back to PEFT LoRA by removing this key) |
| `lokr_model_type` / `lokr_target_modules` | `minimax_h3` / `null` | `_H3_PATTERNS`: `to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2`; `null` = patterns |
| `lokr_alpha` / `lokr_factor` | 1.0 / 4 | factorization scale `alpha/rank`, decomposition factor |
| `lora_rank` / `lora_alpha` | 8 / 8 | PEFT LoRA on the frozen base (when not using LoKR) |
| `lora_target_modules` | `to_q, to_k, to_v, to_out, ff.net.0.proj, ff.net.2` | H3-pinned list; diffusers swiglu FFN names, not `ff.gate/up/down` |
| `noise_selector` | explorative | `K_cond: 2, K_uncond: 5, warmup_steps: 0, schedule: constant, log_stats: true` |
| `optimizer` | `adamw8bit` | 8-bit AdamW |
| `quantize` | `nf4` | base must fit the GPU Guard (`free > total·3/4`) — 61.7 GB transformer needs NF4 on 1×80 GB, or 2×80 GB DDP |
| `mixed_precision` / `weight_dtype` | `bf16` | no autocast on bf16-on-bf16 (avoids LayerNorm dtype warnings) |
| `gradient_checkpointing` | true | engages via PEFT `enable_input_require_grads` |
| `batch_size` / `gradient_accumulation_steps` | 1 / 1 | — |
| `lr` / `lr_scheduler` | 1e-4 / constant | — |
| `multi_gpu` | `reserve` (default) | `ddp` via accelerate for >1 GPU |

## 6. Validation

Two distinct paths (both `torch.no_grad`, RNG frozen with `val_seed=42` and
restored afterwards):

- **`val_loss`** (per epoch): same interpolation + forward as training, one σ
  per batch sampled via `adapter.sample_timesteps`; EMA weights swapped in;
  per-loss breakdown accumulated across batches.
- **Image/video generation** (`generate_images: true`, 4 samples, 20 steps):
  1. initial noise: `randn`, generator seeded `val_seed + sample_idx`;
  2. `noise_scheduler.set_timesteps(20)` (`MiniMaxH3Scheduler`, shift=12;
     no dynamic-shift `mu` — the scheduler has no `mu` param);
  3. **condition rows built once** before the loop (`build_condition_rows`,
     fixed seed) — reused at every step (the review fix);
  4. loop over `t`: σ → `_prepare_model_input(..., condition_rows)` → forward →
     unpack velocities → **Euler step** `scheduler.step(v, t, latent)`;
  5. decode: VAE spatial decode → PIL PNG for images, **silent mp4** at
     24 fps (`MINIMAX_H3_FPS`) for videos (`decode_validation_video`);
  6. `guidance_scale` **must stay 1.0** — MiniMax-H3 has no CFG (D8); the CFG
     branch exists generically but is off for H3.

## 7. Checkpointing & resume

- Save every `save_every_epoch` (1) from `save_start_epoch` (0) to
  `output/<dir>/<save_name>`; weights + optimizer state + `_training_state.pt`.
- **ComfyUI copy**: every save also writes `<name>_comfyui.safetensors` — for
  LoRA the H3 key map is converted exactly (fused `attn.qkv_proj` = concat of
  `to_q/to_k/to_v` down/up, `out_proj` ← `to_out.0`, `mlp.fc1/fc2` ←
  `ff.net.0.proj/ff.net.2`, `transformer_blocks` → `blocks`); for LoKR the
  layer is first converted to an **exact** LoRA pair via the Kronecker
  mixed-product (`kron(w1,w2) = kron(X1,X2)·kron(Y1,Y2)`, `R = in1·rank`, no
  delta materialization) and then the same key conversion is applied. Load in
  ComfyUI with the built-in LoRA loader (regular `lora_down/lora_up` format +
  explicit `.alpha`).
- **Full resume** (krea2-style): `--resume-full <checkpoint.safetensors>` or
  config `"resume": {"checkpoint": ..., "full": true}` — restores weights +
  optimizer + scheduler + RNG + step/epoch from the sibling
  `_training_state.pt` (saved with every checkpoint).
- LoRA-only resume: `--resume <checkpoint>` (config- or CLI-driven; epoch
  off-by-one and `num_epochs` bump rules apply per the unified trainer
  conventions).

## 8. Key constants (from the locked diffusers PR packing module)

| Constant | Value | Meaning |
|---|---|---|
| `MINIMAX_H3_KEYFRAME_NOISE_AUG` | 0.999 | keyframe condition noise level |
| `MINIMAX_H3_KEYFRAME_ENCODE_SEED` | 42 | (PR reference encode seed) |
| `MINIMAX_H3_TEXT_ENCODER_LAYER` | 50 | prompt embeds = `hidden_states[50]` |
| `MINIMAX_H3_TEXT_TAG` / `VIDEO_TAG` | 1 / 0 | modality tags in the packed sequence |
| `MINIMAX_H3_FPS` | 24 | validation video frame rate |
| `_H3_AUDIO_IN_CHANNELS` | 32 | empty audio rows are `(B, 0, 32)` |

## 9. Reference files

| File | Role |
|---|---|
| `UnifiedTrainer/models/minimax_h3/__init__.py` | adapter: packing, condition rows, LoRA contract, decode |
| `UnifiedTrainer/engine/trainer.py` | `train_epoch`, `validate_epoch`, `_generate_from_val_sample` |
| `UnifiedTrainer/losses/flow_matching.py` | data-ward velocity MSE |
| `UnifiedTrainer/data/video_utils.py` | bucket cover-crop, keyframe preprocessing |
| `UnifiedTrainer/configs/minimax_h3_train.json` | production image-pair config |
| `UnifiedTrainer/configs/minimax_h3_video_smoke.json` | video smoke config |
| `md/07-minimaxh3-training.md` | as-built training doc (milestones) |
| `md/minimaxh3_diff.md` | review vs. diffusers PR-14355 / ComfyUI |

*Status: image-pair (M1) + video (M2) code delivered; G2–G7 runtime acceptance
pending model weights. Audio (D6) deferred.*
