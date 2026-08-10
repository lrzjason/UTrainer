# MiniMax-H3 Implementation Diff Review

**Scope.** Code review of the `minimax_h3` adapter at
`UnifiedTrainer/models/minimax_h3/__init__.py` (1249 lines), compared against
the two reference implementations:

| Reference | Files reviewed | Role |
|---|---|---|
| diffusers PR-14355 modular pipeline | `src/diffusers/modular_pipelines/minimax_h3/` (packing, before_denoise, before_encoder, encoders, decoders, denoise, modular_pipeline, modular_blocks), `schedulers/scheduling_minimax_h3.py`, `models/transformers/transformer_minimax_h3.py`, `models/autoencoders/autoencoder_kl_minimax_h3.py` | The upstream inference implementation the adapter was built against |
| ComfyUI | `comfy/ldm/minimax/model.py`, `comfy/ldm/minimax/vae.py`, `comfy/text_encoders/minimax.py` | Independent production inference implementation (release-weights compatible) |

**Method.** Static comparison of training-path semantics (UTrainer-only) and
inference/validation-path semantics (UTrainer vs PR vs ComfyUI). No weights
available, so G2–G7 runtime acceptance from
[07-minimaxh3-training.md](07-minimaxh3-training.md) has not run; everything
below is a code-level comparison.

---

## 1. Executive summary

- **Training path: semantically correct and PR-consistent.** Timestep
  convention, interpolation, data-ward velocity target, condition-row
  construction, text presentation and packing geometry all match the PR.
  Training does one forward per sampled σ, so the per-step RNG differences
  that matter at inference do not affect training.
- **One genuine inference bug found and FIXED: condition rows were re-noised
  every denoising step during validation** ([#1](#findings)). The PR prepares
  the keyframe condition once per request and freezes it as an anchor; the
  trainer called `prepare_model_input` inside the denoising loop, which
  re-drew the condition noise at every step. The trainer now builds the
  condition rows once via the adapter's new `build_condition_rows` and reuses
  them for the whole trajectory. Training was never affected (one forward
  per σ).
- **All other divergences are documented scope cuts (audio, fp32 decode,
  CFG) or unreachable edge cases** (t_cond pinning, σ < 1e-3).
- ComfyUI differs from both UTrainer and the PR on several points of its own
  (posterior-mean VAE encode, last-frame decode trick, closed-form audio
  sigma map, noise-ward velocity) — those are ComfyUI's conventions and do
  not define the PR contract UTrainer implements.

---

## 2. Common ground (used verbatim from diffusers)

The adapter does not reimplement the PR's core machinery — it imports and
reuses it directly, which eliminates whole classes of drift:

- `build_packed_sequence` / `build_row_timesteps` (float64 position grids,
  anchor "first" = `num_text_tokens`, modality tags 0=video/1=text/2=audio,
  AdaLN row = `timestep_index * 3 + token_tag`)
- `patchify_video_latents` / `unpatchify_video_tokens` (patch (1,2,2), hidden
  dim `24·4 = 96`), `align_num_frames` (17n+5), `video_latent_num_frames`
  (5n+2) — the UTrainer media pipeline wraps these via `data/video_utils.py`
- `keyframe_condition_noise` / `MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999`
- `MiniMaxH3Scheduler` (sigma grid, Euler step, float32 blend) — the same
  class instance runs in both training and validation
- `MiniMaxH3Transformer3DModel` — forward signature, internal fp32 module
  list (`proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`,
  `audio_proj_out`, `rope`), padless sequence → `attention_mask=None`
- `AutoencoderKLMiniMaxH3` `_encode_clip` / `_decode_clip` (spatial, tiled),
  `_encode` / `_decode` (temporal 17n+5), `MiniMaxH3VAENormalization`
- Qwen3-VL text encoder, `hidden_states[50]` of 64 layers, no LM head

Because these are shared code, any parity question about them reduces to
"is the PR itself correct", which is out of scope here.

---

## 3. Training path comparison (UTrainer vs PR forward semantics)

### 3.1 Timestep / sigma convention — **equivalent (bit-identical)**

| | UTrainer | PR (packing / scheduler) |
|---|---|---|
| Engine sigma σ ∈ [0,1] | `sample_timesteps`: logit-normal `sigmoid(u).clamp(1e-5, 1-1e-5)`, broadcast over batch | — |
| H3 t | `t = 1 - σ` (adapter `build_row_timesteps(video_timestep=1-σ)`) | `t` directly (scheduler timesteps = `1 - σ_grid`) |
| Interpolation | `(1-σ)·x0 + σ·noise` (trainer.py L620) | `scale_noise`: `t·x0 + (1-t)·noise` |

Same expression after substitution → bit-identical noisy latents (same op
order, assuming same dtype). Verified algebraically; no weights needed.

### 3.2 Velocity / loss target — **consistent, guard-railed**

- Adapter `velocity_sign = "data_ward"`, `compute_target = learning_target - noise`.
- `losses/flow_matching.py` L48–57 dispatches on `velocity_sign` and
  **rejects unknown values** (a silent sign flip would still decrease loss).
- `noise_selector._eval_velocity_mse` and `compute_x0_hat` (`noise +
  velocity` for data_ward) use the same convention — no sign mismatch found
  anywhere on the training path.
- `MiniMaxH3Scheduler.step` consumes the data-ward velocity directly
  (`denoised = x_t + (1-t)·v`), so training and validation agree by
  construction.

### 3.3 Condition rows — **same math, different RNG policy**

Construction is identical: `scale_noise(clean, 0.999)` =
`0.999·x0 + 0.001·noise`, computed in **float32** in both (bf16 ulp ≈ 0.0078
would swallow the 0.001 term), rows pinned at `t_cond = 0.999`.

RNG policy for the 0.001 noise differs:

| | UTrainer | PR |
|---|---|---|
| Seed | `_keyframe_noise_seed + round(σ·1e6)` (per-sigma deterministic, fresh per level) | drawn from the request `torch.Generator`, before the video noise, once per request |
| Effect | deterministic per σ — same σ ⇒ same condition rows | fresh every request |

For **training** (one forward per σ) this is irrelevant — each forward gets a
valid condition draw. It only becomes a parity break at multi-step inference
(see [#1](#findings)).

### 3.4 Batch handling — **PR-consistent**

- Transformer batch dim is pure replication (verified in
  `transformer_minimax_h3.py`); UTrainer enforces batch-uniformity of tags
  and sigmas before packing.
- One packed sequence per batch; single σ per batch (broadcast in
  `sample_timesteps`).

### 3.5 Text conditioning — **identical**

- PR presentation `<Picture i>: ` + `<|vision_start|>` + image_pad×N +
  `<|vision_end|>`, no chat template; token-type ids via
  `processor.create_mm_token_type_ids`.
- Same presentation in UTrainer `encode_text`; vision block tagged
  VIDEO(0), text tag 1; `hidden_states[50]` output, `(L, 5120)`.

### 3.6 Audio — **omitted (documented D6)**

- UTrainer: `audio_hidden_states (B, 0, 32)` (input channels, **not** the
  5376 projection dim), `num_audio_latents = 0` → empty packed rows. Matches
  PR t2va-without-audio shape contract exactly.
- PR: second scheduler + audio rows; ComfyUI: closed-form
  `time_shift_sigma` + `time_shift_slope` velocity scaling, no second
  scheduler. Neither is implemented in UTrainer; audio training is deferred
  per 07-minimaxh3-training.md.

### 3.7 LoRA scope

Frozen targets `["to_q","to_k","to_v","to_out","ff.net.0.proj","ff.net.2"]`
(attention + MLP LoRA). `proj_in`/`proj_out`/`time_embedder` are not
LoRA-adapted — consistent with their fp32-keep status in the PR, and with
full-finetune not being supported (documented scope cut).

---

## 4. Inference / validation path comparison

### 4.1 Scheduler step — **identical code**

UTrainer and PR both run `MiniMaxH3Scheduler.step`: float32 Euler blend
`ratio·x_t + (1-ratio)·x0_hat`, ratio from the σ grid, denoised from the
timestep. Same class, same behavior. UTrainer skips dynamic `mu` shifting
(`set_timesteps` has no `mu` param — handled in trainer.py L1392–1405).

### 4.2 Condition-row lifecycle — **DIFFERS → FIXED**

| | UTrainer validation | PR |
|---|---|---|
| When condition rows are built | **once per request** (before the loop) via `adapter.build_condition_rows`, then frozen | once, in `PrepareLatentsStep`, before the loop |
| Noise realization | fixed seed (keyframe noise base, no σ offset) — same rows for the whole trajectory | fixed for the whole trajectory (anchors) |
| Loop | `prepare_model_input(..., condition_rows=rows)` per step; only target rows change | only `latents[num_condition_video_rows:]` are stepped; condition rows untouched |

**Before the fix** the trainer called `prepare_model_input` inside the
loop, so the 0.001-noise component of the condition (seed = base +
`round(σ·1e6)`) was re-drawn at every step: the model was trained with one
fixed condition per sample but saw a wobbling condition across the
validation trajectory. Deterministic run-to-run (same σ grid ⇒ same seeds),
so it did not show up as nondeterminism. CFG is off
(`val_guidance_scale = 1`), so the "CFG-consistent same seed" rationale did
not rescue it either.

**Fix (adapter + trainer):**
- `MiniMaxH3Adapter.build_condition_rows(batch, device, dtype)` — builds the
  packed condition rows once per request, mixing at the fixed keyframe-noise
  base (no σ offset, PR-style once-per-request draw). Returns `None` for
  pure-t2v batches.
- `prepare_model_input(..., condition_rows=None)` — pre-built rows bypass the
  per-step draw; shape (`(B, rows_per_frame, dim)` and source/target bucket
  equality) is re-validated against the target.
- `Trainer._generate_from_val_sample` builds the rows once before the loop
  and forwards them through the guarded `Trainer._prepare_model_input`
  helper (generic adapters unaffected). The uncond batch shares the same
  latents, so one tensor serves both CFG forwards.
- Verified at packing level (CPU, no weights): rows identical across σ with
  pre-built path; per-step path (training) still re-draws per σ; guards
  raise on wrong shape / bucket mismatch.

### 4.3 t_cond pinning — **edge case → now PR-exact**

- UTrainer: `condition_video_timestep = max(t_target, 0.999)` — matches the
  PR exactly (`before_denoise.py` L417: `max(float(timestep), 0.999)`).
- The `0.999` floor only binds when `σ < 1e-3` (logit-normal sampling
  probability ≈ 2.4e-12 — practically unreachable), so the change is
  behavior-neutral in reachable territory; applied as part of the #1 fix.

### 4.4 Initial latents — **consistent**

Both initialize the video latents as standard-normal `randn` at the first
step (UTrainer: `torch.Generator(device).manual_seed(val_seed + idx)`,
PR: request generator). UTrainer draws only video noise; PR additionally
draws condition noise from the same generator first — ordering affects
generated output but not correctness.

### 4.5 Decode T == 1 (single frame)

| | UTrainer | ComfyUI |
|---|---|---|
| Input | `z.repeat(1,1,5,1,1)` → `_decode_clip` → 20 pixel frames | single latent frame → decoder |
| Frame pick | crop `frame_pre_padding=3`, take **first** valid (index 3 of 20) | take **last** of 4 (`dec[:,:,-1:,:,:]`, index 3 of 4) |

Both select pixel position 3, but the causal temporal conv / RoPE context
differs: UTrainer feeds 5 identical latent frames (uniform context), ComfyUI
decodes with a single-frame context. Same target pixel, different receptive
field → numerically different output. The PR path for "up to one frame" goes
through the full temporal decode (`vae.decode`) and was not exercised here
(no weights). **Unverified numerically; flag for a weights-based check.**

### 4.6 Decode T > 1 — **documented fp32 divergence**

UTrainer runs `vae._decode` in fp32; the PR runs `vae.decode` under fp16
autocast on CUDA. Documented in 07-minimaxh3-training.md as intentional
(fp32 claimed cleaner / deterministic). Outputs will differ in low bits from
PR samples; not a correctness issue for training.

### 4.7 CFG

UTrainer validation runs guidance-free (`val_guidance_scale = 1`). The
trainer's CFG branch exists generically but is not exercised for minimax_h3;
the PR's t2va/fl2va pipeline likewise has no CFG path. Consistent.

---

## 5. ComfyUI contrast (inference conventions)

ComfyUI is a valid production reference but encodes several deliberate
conventions that differ from both UTrainer and the PR:

| Aspect | ComfyUI | PR / UTrainer |
|---|---|---|
| Velocity returned | **noise-ward** (`[-video_out, -slope_a·audio_out]`, negated) | data-ward `x0 - x_t` |
| Audio timestep | closed form `time_shift_sigma(video_σ)` + `time_shift_slope` on the velocity | second scheduler (PR) / none (UTrainer) |
| VAE encode | **posterior MEAN**, no sampling; single frame → `moments[:,:,-1:]` | `posterior.sample(seed 42)` + fp16 rounding (PR) — UTrainer matches PR |
| VAE decode | last-of-4 frame trick (see 4.5) | full temporal decode / clip decode |
| Condition noise RNG | `torch.Generator("cpu").manual_seed(payload seed)` restarted **per condition** (same noise for all conditions) | request generator (PR), σ-derived seed (UTrainer) |
| Spatial padding | `pad_to_patch_size` before packing | 16-divisible bucket latents (UTrainer); 32-multiple canvas (PR) |
| Text | Qwen3-VL truncated at 50 layers, `enable_attention_masks=False`, `layer_norm_hidden_state=False`, pad 151643; same presentation; vision block tag 0 (video), text tag 1 | same presentation & tags; `hidden_states[50]` |

Key takeaway: for VAE encode, ComfyUI's posterior-mean choice is the
**only** one of the three that is deterministic; PR and UTrainer sample with
seed 42. Any bit-level parity test against ComfyUI encode outputs will fail
by design.

---

## 6. Findings

| # | Severity | Finding | Location | Impact |
|---|---|---|---|---|
| 1 | **High (inference) — FIXED** | Condition rows re-noised every validation step; PR freezes them once | trainer.py val loop → `build_condition_rows` + `condition_rows=` kwarg (adapter), `_prepare_model_input` (trainer) | Validation conditioning now frozen per request (PR parity). Verified at packing level; weights-based acceptance pending |
| 2 | Low — FIXED | `t_cond` pinned 0.999 vs PR `max(t, 0.999)` | adapter `prepare_model_input` `build_row_timesteps` call | Now `max(t_target, 0.999)` — exactly PR; behavior-neutral for reachable σ |
| 3 | Medium (unverified) | Decode T==1: repeat-5-clip vs ComfyUI single-latent decode; same pixel index, different temporal context | adapter `decode_latent` vs comfy vae.py `decode` | Numerically different single-frame decode; needs weights to judge which (if either) matches the PR |
| 4 | Low (documented) | Decode T>1 fp32 vs PR fp16 autocast | adapter `decode_validation_video` vs PR decoders.py | Low-bit output differences; documented as intentional |
| 5 | Info (documented) | Audio omitted; `(B,0,32)` rows | adapter `prepare_model_input` | Scope cut D6; matches PR t2va shape contract |
| 6 | Info | 16-divisibility (bucket) vs PR 32-multiple canvas | `RESOLUTION_CONFIG` vs PR `resolve_canvas_size` | Non-canvas-aspect keyframes: UTrainer cover-crops, PR stretches the first keyframe (geometry anchor). Different conditioning pixels for mismatched aspect ratios; training distribution note |
| 7 | Info | No CFG path exercised; no full finetune | trainer.py val loop, LoRA targets | Scope cuts; consistent with PR |

---

## 7. Verified parity (code-level, no weights needed)

- Interpolation bit-equivalence: `(1-σ)x0 + σ·noise` ≡ `t·x0 + (1-t)·noise`, `t = 1-σ`.
- Data-ward target `x0 - noise` consistent across adapter, loss, noise
  selector and `compute_x0_hat`; unknown velocity signs rejected.
- Packing geometry: float64 position grids (temporal `5/3·(1,4,4,4,4)`,
  spatial `linspace(endpoint=False)·32`), anchors, tags, AdaLN table.
- Text presentation, `hidden_states[50]`, token-type ids, text tags.
- Condition mix `0.999·x0 + 0.001·noise` in float32, `t_cond = 0.999`.
- Forward kwargs match `MiniMaxH3Transformer3DModel.forward` exactly;
  mixed-precision fp32 module list handled inside the transformer.
- Batch-uniformity enforcement; single-σ-per-batch semantics.
- Scheduler class shared verbatim between training and validation.
- Empty-audio shape contract `(B, 0, 32)`.

## 8. Unverified (needs weights / runtime)

- G2–G7 runtime acceptance (07-minimaxh3-training.md) — never run.
- End-to-end validation generation after fix #1 (packing-level verification
  passed; full denoising trajectory needs weights).
- Decode T==1 numerical parity vs PR/ComfyUI (finding #3).
- fp16-autocast decode difference magnitude (finding #4).

---

*Generated from static review of the three codebases; line references are to
the current workspace state (2026-08).*
