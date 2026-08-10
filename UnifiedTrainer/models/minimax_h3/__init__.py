"""MiniMax-H3 model adapter — packed-sequence image-conditioned flow matching.

MiniMax-H3 (diffusers PR-14355) is a text + keyframe-conditioned video
diffusion transformer that runs its forward pass over a single packed 1-D
sequence:

    [ text (L) | keyframe condition rows (C) | target audio (A=0) | target video rows (V) ]

This adapter implements the **image-pair training path** (milestone 1) and the
**video training path** (milestone 2, P2):

- A source image (``S``) is the keyframe condition; a target image (``P``) is
  the single generated frame.  Both are single-frame latents in the unified
  5D ``(B, 24, 1, H, W)`` convention built by P1.3's media pipeline.
- The transformer consumes MiniMax-H3's ``t = 1 - sigma`` convention and
  predicts a **data-ward** velocity (``v = x0 - x_t``).
- Condition rows are not fully clean: the clean source latent is mixed with
  the PR's keyframe noise-augmentation, ``scale_noise(clean, 0.999)`` =
  ``0.999 * x0 + 0.001 * noise`` (``MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999``,
  packing.py L82-84; engine-sigma terms: ``sigma_cond = 0.001``) and pinned
  at ``t_cond = 0.999`` for every denoising step (before_denoise.py L417:
  ``condition_video_timestep = max(t, 0.999)``; the scheduler's
  ``scale_noise`` docstring takes the ``noise_aug`` level at face value).

Only the video VAE's *spatial* encoder/decoder is used for images:
``_encode_clip`` / ``_decode_clip`` — never ``_encode`` / ``_decode``, which
chunk 17-frame clips and would turn a single frame into two latent frames.

The adapter is batch-uniform by construction: ``position_ids`` / ``token_tags``
/ ``timestep_indices`` are shared across the batch axis, so every sample in a
forward must share the same packed layout (identical text length + tag pattern)
and the same timestep.  ``sample_timesteps`` therefore emits **one** sigma per
forward, broadcast over the batch.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

# Packed-sequence geometry and constants, reused verbatim from the locked
# diffusers PR (import path is NOT re-exported from the package __init__).
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

logger = logging.getLogger(__name__)

# Bucket tiers — isomorphic to krea2's RESOLUTION_CONFIG: short edge fixed
# per tier, area capped at the widest bucket, every axis a multiple of the 16
# divisibility (vae_scale=8 x patch_size=2).  The PR's inference canvas rounds
# to 32; the bucket system only requires vae_scale*patch divisibility, so 16
# suffices.  The 768 tier gained intermediate aspect ratios; the 1024/1280
# tiers let image training go beyond 768p (set data/validation resolution to
# the tier's short edge, e.g. 1024).
RESOLUTION_CONFIG = {
    768: [
        (768, 768),
        (768, 832), (768, 896), (768, 960), (768, 1024),
        (768, 1088), (768, 1152), (768, 1216), (768, 1280), (768, 1344),
        (832, 768), (896, 768), (960, 768), (1024, 768),
        (1088, 768), (1152, 768), (1216, 768), (1280, 768), (1344, 768),
    ],
    1024: [
        (1024, 1024),
        (1024, 1152), (1024, 1280), (1024, 1408), (1024, 1536),
        (1152, 1024), (1280, 1024), (1408, 1024), (1536, 1024),
    ],
    1280: [
        (1280, 1280),
        (1280, 1408), (1280, 1536), (1280, 1664), (1280, 1792),
        (1408, 1280), (1536, 1280), (1664, 1280), (1792, 1280),
    ],
}

# Audio input channels of the transformer's ``audio_proj_in``
# (``nn.Linear(audio_in_channels=32, hidden_size=5376)``).  With zero audio
# rows the linear projection still validates the feature dim, so an empty
# ``audio_hidden_states`` must be ``(B, 0, 32)`` — NOT ``(B, 0, 5376)`` which
# is the *output* dim of the projection (see the deviations report).
_H3_AUDIO_IN_CHANNELS = 32

# Conditioning timestep floor, t_cond = 0.999.
# PR semantics (packing.py L82-84: the released model noises keyframe
# latents to `t = 0.999` and runs them at that timestep for every denoising
# step; before_denoise.py L417: condition_video_timestep = max(t, 0.999);
# scheduler scale_noise docstring: x_t = t*x0 + (1 - t)*noise with `t` the
# noise_aug level, taken at face value).  prepare_model_input applies
# ``max(t_target, _H3_CONDITION_TIMESTEP)`` — the floor only binds for
# sigma < 1e-3, which logit-normal sampling never reaches in practice.
_H3_CONDITION_TIMESTEP = MINIMAX_H3_KEYFRAME_NOISE_AUG


@ModelRegistry.register("minimax_h3")
class MiniMaxH3Adapter(BaseModelAdapter):
    """Adapter for the MiniMax-H3 video transformer (image-pair + video training)."""

    name = "minimax_h3"
    patch_size = 2  # spatial patch; the temporal patch (1) is applied in prepare_model_input

    def __init__(self, config: dict):
        self.config = config or {}
        self._model_path = self.config.get("model_path", "")

        # Logit-normal timestep sampling mu (configurable).  mu=0.0 is the
        # plain logit-normal used by the base adapter; positive mu shifts the
        # density toward noisier steps.
        self.timestep_mu: float = float(self.config.get("timestep_mu", 0.0))

        # Base seed for the keyframe-conditioning noise.  The per-forward seed
        # is derived from this base plus the current sigma value, so the same
        # noise level reproduces the same conditioning rows (CFG-consistent)
        # while every distinct training sigma draws fresh noise.
        self._keyframe_noise_seed: int = int(
            self.config.get("seed")
            or self.config.get("training", {}).get("seed", 0)
        )

        # Packed-geometry cache shared between prepare_model_input and
        # unpack_prediction (mirrors krea2's ``_target_grids``).
        self._packed_geometry: Optional[dict] = None

        # Spatial shape of the source behind the last build_condition_rows
        # call; prepare_model_input validates pre-built condition rows
        # against the target's (H, W) (one-bucket-per-batch contract).
        self._condition_rows_shape: Optional[tuple] = None

    # ── Model loading ──────────────────────────────────────────────────

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import MiniMaxH3Transformer3DModel

        transformer = MiniMaxH3Transformer3DModel.from_pretrained(path, torch_dtype=dtype)

        # AdaLN modulation surrogate (adaln_surrogate.py): the ~13B frozen
        # AdaLN projections are replaced by a low-rank function of the
        # timestep, built/loaded inside training and cached next to the
        # checkpoint.  Config: training.adaln_surrogate = {enabled (default
        # true), rank (default 64, clamped [32, 128]), grid (default 1024),
        # device ("auto" default: per-block GPU build when CUDA is available;
        # "cuda" to force, "cpu" to force CPU)}.
        cfg = (self.config.get("training") or {}).get("adaln_surrogate") or {}
        if cfg.get("enabled", True):
            from UnifiedTrainer.models.minimax_h3.adaln_surrogate import (
                install_surrogate,
            )

            install_surrogate(
                transformer,
                path,
                rank=cfg.get("rank", 64),
                grid=cfg.get("grid", 1024),
                device=cfg.get("device", "auto"),
            )
        return transformer

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import AutoencoderKLMiniMaxH3
        return AutoencoderKLMiniMaxH3.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        from diffusers import MiniMaxH3Scheduler
        # The released model uses shift=12 for the video schedule.
        return MiniMaxH3Scheduler(shift=12)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import AutoConfig, Qwen3VLForConditionalGeneration
            config = AutoConfig.from_pretrained(path)
            # Patch rope_scaling if missing — transformers has a bug where it
            # calls config.rope_scaling.get() without null-checking.  The
            # rope_scaling lives on the text_config sub-config.
            mrope = {"rope_type": "default", "mrope_section": [24, 20, 20]}
            if hasattr(config, "text_config"):
                if getattr(config.text_config, "rope_scaling", None) is None:
                    config.text_config.rope_scaling = mrope
            elif getattr(config, "rope_scaling", None) is None:
                config.rope_scaling = mrope
            return Qwen3VLForConditionalGeneration.from_pretrained(
                path, config=config, torch_dtype=dtype
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load MiniMax-H3 text encoder: {e}")
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        """Load the Qwen2 tokenizer (krea2 fallback construction).

        The released Qwen3-VL tokenizer_config.json has ``extra_special_tokens``
        as a list, which makes ``AutoTokenizer.from_pretrained`` fail with
        ``'list' object has no attribute 'keys'``.  We patch the config by
        renaming it to ``additional_special_tokens`` (the standard field) and
        load via ``Qwen2TokenizerFast``.
        """
        import json
        import os
        import shutil
        import tempfile

        # First, try the standard loading path.
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(path)
        except Exception:  # noqa: BLE001
            pass

        # Fallback: patch tokenizer_config.json and load Qwen2TokenizerFast.
        try:
            from transformers import Qwen2TokenizerFast

            config_path = os.path.join(path, "tokenizer_config.json")
            if not os.path.exists(config_path):
                return None

            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)

            # Rename extra_special_tokens -> additional_special_tokens.
            extra = cfg.pop("extra_special_tokens", [])
            if extra:
                existing = cfg.get("additional_special_tokens", [])
                cfg["additional_special_tokens"] = list(existing) + list(extra)

            # Write the patched config to a temp directory.
            tmp_dir = tempfile.mkdtemp(prefix="minimax_h3_tok_")
            try:
                for f in os.listdir(path):
                    shutil.copy2(os.path.join(path, f), os.path.join(tmp_dir, f))
                with open(
                    os.path.join(tmp_dir, "tokenizer_config.json"),
                    "w", encoding="utf-8",
                ) as f:
                    json.dump(cfg, f, indent=2)
                tokenizer = Qwen2TokenizerFast.from_pretrained(tmp_dir)
            finally:
                shutil.rmtree(tmp_dir)
            logger.info(f"Loaded Qwen2TokenizerFast (patched config) from {path}")
            return tokenizer
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load MiniMax-H3 tokenizer from {path}: {e}")
            # Last resort: PreTrainedTokenizerFast (won't work with the
            # Qwen3VLProcessor, but keeps text-only encoding alive).
            try:
                from transformers import PreTrainedTokenizerFast
                return PreTrainedTokenizerFast(
                    tokenizer_file=os.path.join(path, "tokenizer.json")
                )
            except Exception:  # noqa: BLE001
                return None

    def load_processor(self, path: str, tokenizer_path: str = "") -> Optional[Any]:
        """Load the Qwen3VL processor (krea2 fallback construction).

        The released model ships without a ``preprocessor_config.json``, so
        ``AutoProcessor.from_pretrained`` fails.  We construct the processor
        manually from a ``Qwen2VLImageProcessor``, a ``Qwen3VLVideoProcessor``,
        and the tokenizer loaded from ``tokenizer_path``.
        """
        # First, try the standard loading path.
        try:
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(path)
        except Exception:  # noqa: BLE001
            pass

        tok_path = tokenizer_path or path
        try:
            import json
            import os

            from transformers import Qwen3VLProcessor
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                Qwen2VLImageProcessor,
            )
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
                Qwen3VLVideoProcessor,
            )

            # Qwen3VL's vision tower uses patch_size=16 (not Qwen2VL's default 14).
            patch_size = 16
            temporal_patch_size = 2
            merge_size = 2
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    model_config = json.load(f)
                vision_config = model_config.get("vision_config", {})
                patch_size = vision_config.get("patch_size", patch_size)
                temporal_patch_size = vision_config.get(
                    "temporal_patch_size", temporal_patch_size
                )
                merge_size = vision_config.get("spatial_merge_size", merge_size)

            image_processor = Qwen2VLImageProcessor(
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            video_processor = Qwen3VLVideoProcessor()

            tokenizer = self.load_tokenizer(tok_path)
            if tokenizer is None:
                logger.warning(
                    f"Cannot construct MiniMax-H3 processor: tokenizer is None "
                    f"from {tok_path}"
                )
                return None

            processor = Qwen3VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
            )
            logger.info(
                f"Constructed Qwen3VLProcessor (tokenizer from {tok_path}, "
                f"patch_size={patch_size})"
            )
            return processor
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to construct Qwen3VLProcessor from {path}: {e}"
            )
            return None

    # ── Architecture specs ─────────────────────────────────────────────

    @property
    def latent_channels(self) -> int:
        return 24

    @property
    def vae_scale_factor(self) -> int:
        return 8

    @property
    def embedding_dim(self) -> int:
        return 5120

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def velocity_sign(self) -> str:
        return "data_ward"

    @property
    def encode_text_accepts_image(self) -> bool:
        return True

    # ── VAE normalization helpers ──────────────────────────────────────

    @staticmethod
    def _latents_normalization(
        vae: nn.Module, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The (mean, std) the cache stores its normalized latents in.

        ``AutoencoderKLMiniMaxH3`` ships ``latents_mean`` / ``latents_std`` in
        its config (24 values each, viewed as ``(1, C, 1, 1, 1)``).  Diffusers
        configs are FrozenDicts, so both dict and attribute access are covered.
        """
        cfg = getattr(vae, "config", None)
        if isinstance(cfg, dict):
            mean_vals = cfg.get("latents_mean", (0.0,) * 24)
            std_vals = cfg.get("latents_std", (1.0,) * 24)
        else:
            mean_vals = getattr(cfg, "latents_mean", (0.0,) * 24)
            std_vals = getattr(cfg, "latents_std", (1.0,) * 24)
        mean = torch.tensor(list(mean_vals), device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(list(std_vals), device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        return mean, std

    @staticmethod
    def _pixel_normalization(
        device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ImageNet pixel mean/std used by the video VAE (PR constants)."""
        pixel_mean = torch.tensor(
            MINIMAX_H3_PIXEL_MEAN, device=device, dtype=dtype
        ).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(
            MINIMAX_H3_PIXEL_STD, device=device, dtype=dtype
        ).view(1, -1, 1, 1, 1)
        return pixel_mean, pixel_std

    # ── Encoding ───────────────────────────────────────────────────────

    def encode_image(self, vae: nn.Module, image: torch.Tensor) -> dict:
        """Encode one keyframe image into a normalized single latent frame.

        The image follows the unified media convention — a 5D
        ``(1, 3, 1, H, W)`` float32 tensor in ``[0, 1]`` (what ``encode_video``
        receives).  A legacy 4D ``(1, 3, H, W)`` diffusion-normalized ``[-1, 1]``
        tensor is also accepted and converted.

        Encoding goes through ``vae._encode_clip`` — the PR's keyframe path.
        ``vae._encode`` is forbidden here: its 17-frame temporal chunking turns
        a single frame into two latent frames.
        """
        if image.ndim == 4:
            # Legacy diffusion-normalized [-1, 1] -> unified [0, 1] 5D frames.
            image = ((image.clamp(-1.0, 1.0) / 2.0) + 0.5).unsqueeze(2)
        if image.ndim != 5 or image.shape[0] != 1 or image.shape[2] != 1:
            raise ValueError(
                f"MiniMax-H3 encode_image expects (1, 3, 1, H, W) in [0, 1], "
                f"got shape {tuple(image.shape)}"
            )

        device = next(vae.parameters()).device
        pixel_mean, pixel_std = self._pixel_normalization(device, torch.float32)
        pixels = (image.to(device=device, dtype=torch.float32) - pixel_mean) / pixel_std

        with torch.no_grad():
            moments = vae._encode_clip(pixels)
            posterior = DiagonalGaussianDistribution(moments)
            # Fixed seed, independent of any request seed (PR parity).
            latents = posterior.sample(
                generator=torch.Generator(device=device).manual_seed(
                    MINIMAX_H3_KEYFRAME_ENCODE_SEED
                )
            )
        # fp16 rounding before normalization — ~11 bits of every conditioning
        # latent, so the released model's conditioning cannot be reproduced
        # without it (PR parity).
        latents = latents.to(torch.float16).float()

        mean, std = self._latents_normalization(vae, device, torch.float32)
        latents = (latents - mean) / std  # (1, 24, 1, h, w)
        # B folds into the per-sample cache dimension -> (24, 1, h, w).
        return {"latent": latents[0]}

    def encode_video(self, vae: nn.Module, frames: torch.Tensor) -> dict:
        """Encode a 5D ``(B=1, C, T, H, W)`` frame tensor to latent space.

        ``T == 1`` (image) delegates to ``encode_image``'s single-frame path.
        ``T > 1`` (video, P2-ready) goes through ``vae._encode`` — the PR's
        video-reference recipe: 17n+5 frame chunks with token_drop, a posterior
        sample under the fixed keyframe seed, fp16 rounding, and normalization.
        """
        if frames.ndim != 5:
            raise ValueError(
                f"encode_video expects 5D (B=1, C, T, H, W) frames, "
                f"got shape {tuple(frames.shape)}"
            )
        if frames.shape[0] != 1:
            raise ValueError(
                f"encode_video supports B=1 batches only, got B={frames.shape[0]}"
            )
        if frames.shape[2] == 1:
            return self.encode_image(vae, frames)

        device = next(vae.parameters()).device
        pixel_mean, pixel_std = self._pixel_normalization(device, torch.float32)
        pixels = (frames.to(device=device, dtype=torch.float32) - pixel_mean) / pixel_std

        with torch.no_grad():
            # 17n+5 frame chunks -> 5n+2 latent frames (token_drop=3).
            moments = vae._encode(pixels)
            posterior = DiagonalGaussianDistribution(moments)
            latents = posterior.sample(
                generator=torch.Generator(device=device).manual_seed(
                    MINIMAX_H3_KEYFRAME_ENCODE_SEED
                )
            )
        latents = latents.to(torch.float16).float()

        mean, std = self._latents_normalization(vae, device, torch.float32)
        latents = (latents - mean) / std  # (1, 24, T_lat, h, w)
        return {"latent": latents[0]}  # (24, T_lat, h, w)

    def encode_text(
        self,
        text_encoder: Optional[nn.Module],
        tokenizer: Optional[Any],
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        reference_image: Optional[Any] = None,
        processor: Optional[Any] = None,
        condition_image: Optional[Any] = None,
    ) -> dict:
        """Encode MiniMax-H3's presentation of a prompt into hidden states.

        When the associated keyframe image is available (``condition_image``,
        passed by the cache builder when ``encode_text_accepts_image`` is
        true), the presentation prepends a ``"<Picture i>: "`` label and a
        vision block (``<|vision_start|>`` + one ``<|image_pad|>`` per vision
        patch + ``<|vision_end|>``) — no chat template, matching the PR.  The
        vision-block rows are tagged *video* (``MINIMAX_H3_VIDEO_TAG``), which
        is what the transformer's AdaLN modulation keys off.

        With ``condition_image is None`` the prompt is encoded verbatim with an
        all-text tag layout (used by the empty-string embedding).

        Returns:
            dict with ``prompt_embed`` (``(L, 5120)``), ``prompt_embeds_mask``
            (``(L,)`` bool) and ``text_token_tags`` (``(L,)`` long) — the last
            is persisted by the cache (int64 survives EmbeddingCache.save) and
            consumed by ``prepare_model_input`` to build the packed layout.
        """
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "MiniMax-H3 requires a loaded Qwen3VL text encoder and tokenizer."
            )

        images: list[Any] = []
        if condition_image is not None:
            images = (
                condition_image
                if isinstance(condition_image, list)
                else [condition_image]
            )
        elif reference_image is not None:
            refs = (
                reference_image
                if isinstance(reference_image, list)
                else [reference_image]
            )
            images = [r for r in refs if r is not None]

        num_layers = text_encoder.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] "
                f"of its Qwen3-VL conditioner, which needs more than "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but the text encoder has "
                f"{num_layers}. The last hidden state of a stack truncated to exactly "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the "
                f"conditioning MiniMax-H3 expects."
            )

        # ── Build the presentation: [<Picture i>: + vision block]* + prompt ──
        pixel_values, image_grid_thw = None, None
        token_ids: list[int] = []
        token_tags: list[int] = []
        if images and processor is not None:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = processor.image_processor.merge_size ** 2
            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = tokenizer(
                    f"<Picture {index + 1}>: ", add_special_tokens=False
                )["input_ids"]
                vision_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += (
                    [MINIMAX_H3_TEXT_TAG] * len(label_ids)
                    + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
                )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, which it
        # reads off the token type ids the processor derives from the vision pad
        # ids (`0` text, `1` image, `2` video).
        try:
            mm_token_type_ids = torch.tensor(
                processor.create_mm_token_type_ids([token_ids]),
                dtype=torch.long,
                device=device,
            )
        except (AttributeError, TypeError):  # noqa: BLE001
            # Defensive: some transformers versions lack this helper; zeros are
            # correct for a text-only presentation and safe for vision blocks.
            mm_token_type_ids = torch.zeros(
                len(token_ids), dtype=torch.long, device=device
            )

        # ``text_encoder.model`` is a submodule, and a CPU-offload hook
        # (accelerate's) wraps the *top-level* module's forward alone, so
        # calling the submodule directly would leave the conditioner on the
        # CPU.  Fire the hook by hand instead of routing through
        # ``text_encoder(...)``: MiniMax-H3 reads ``hidden_states[50]`` and
        # never uses the language-model head (PR parity).
        hook = getattr(text_encoder, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(text_encoder)

        with torch.no_grad():
            outputs = text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=(
                    None
                    if pixel_values is None
                    else pixel_values.to(device, text_encoder.dtype)
                ),
                image_grid_thw=(
                    None
                    if image_grid_thw is None
                    else image_grid_thw.to(device)
                ),
                use_cache=False,
                output_hidden_states=True,
            )

        prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(
            device=device, dtype=dtype
        )  # (1, L, 5120)
        prompt_embed = prompt_embeds[0]  # (L, 5120)

        return {
            "prompt_embed": prompt_embed,
            "prompt_embeds_mask": torch.ones(
                prompt_embed.shape[0], dtype=torch.bool, device=device
            ),
            "text_token_tags": torch.tensor(token_tags, dtype=torch.long),
        }

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent into pixels in ``[-1, 1]``.

        ``T == 1`` (image path, milestone 1 — bit-for-bit unchanged):
        denormalize (the cache stores normalized latents) -> repeat the frame
        to a 5-latent decode clip -> ``_decode_clip`` -> 20 pixel frames ->
        crop the ``frame_pre_padding`` leading frames -> take the first real
        frame -> convert from the VAE's ImageNet-normalized pixel space to
        ``[-1, 1]`` (the engine's image-validation contract), returning a 4D
        ``(B, 3, H, W)`` image.

        ``T > 1`` (video path, P2): denormalize -> ``vae._decode`` (the PR's
        chunked temporal decoder) -> ``(B, 3, T', H*8, W*8)`` pixel frames ->
        the same ImageNet-inverse conversion, keeping the 5D shape (no time
        axis squeeze) — returns ``(B, 3, T', H, W)`` in ``[-1, 1]``.
        """
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        if latent.ndim != 5:
            raise ValueError(
                f"MiniMax-H3 decode_latent expects 5D (B, C, T, H, W) latents, "
                f"got shape {tuple(latent.shape)}"
            )

        vae_dtype = getattr(vae, "dtype", torch.float32)
        device = latent.device

        # 1. Undo the cache-side normalization (encode stores normalized).
        mean, std = self._latents_normalization(vae, device, vae_dtype)
        z = (latent.to(dtype=vae_dtype) * std + mean)  # (B, 24, T, h, w)

        if latent.shape[2] == 1:
            # 2. Repeat the single latent frame to a full decode clip.  _decode_clip
            # turns 5 latent frames into 20 pixel frames: 17 real frames plus the
            # `frame_pre_padding` leading frames the encoder's implicit padding left.
            z = z.repeat(1, 1, 5, 1, 1)
            with torch.no_grad():
                pixels = vae._decode_clip(z)  # (B, 3, 20, H*8, W*8)

            # 3. Crop the encoder pre-padding and take the first real frame.
            frame_pre_padding = int(getattr(vae, "frame_pre_padding", 3))
            pixels = pixels[:, :, frame_pre_padding:]
            image = pixels[:, :, 0]  # (B, 3, H*8, W*8)

            # 4. VAE pixel convention (ImageNet-normalized) -> [0, 1] -> [-1, 1].
            # _pixel_normalization views the stats as (1, C, 1, 1, 1) but the
            # frame is 4D (B, C, H, W); restore the singleton temporal axis for
            # the conversion and squeeze it back off.
            pixel_mean, pixel_std = self._pixel_normalization(device, image.dtype)
            image = image.unsqueeze(2)  # (B, C, 1, H, W)
            image = image * pixel_std + pixel_mean
            image = image.squeeze(2)  # (B, C, H, W)
            image = image * 2.0 - 1.0
            return image

        # T > 1 (video): the PR's chunked temporal decoder.
        # fp32 decode (vs the PR's fp16 autocast) is an intentional
        # divergence: the adapter decodes without autocast — results
        # verified bit-exact — at a higher transient cost (a 768px val
        # clip is (1, 3, 124, 768, 768) fp32 ≈ 2.8 GB).  Keep the
        # dtype; the engine's VAE stays fp32.
        with torch.no_grad():
            pixels = vae._decode(z)  # (B, 3, T', H*8, W*8)
        # 4'. VAE pixel convention (ImageNet-normalized) -> [0, 1] -> [-1, 1].
        # Keep the 5D (B, C, T', H, W) shape — the engine's video validation
        # dispatch expects per-batch temporal frames, not a squeezed image.
        pixel_mean, pixel_std = self._pixel_normalization(device, pixels.dtype)
        pixels = pixels * pixel_std + pixel_mean
        pixels = pixels * 2.0 - 1.0
        return pixels

    def decode_validation_video(
        self,
        vae: nn.Module,
        latent: torch.Tensor,
        output_dir: str,
        prefix: str = "",
    ) -> list[str]:
        """Decode a video latent to one silent h264 mp4 per batch item.

        ``decode_latent`` (T > 1 branch) produces ``(B, 3, T', H, W)`` pixels in
        ``[-1, 1]``; each batch item is converted to ``[0, 1]`` uint8 RGB frames
        (the engine's existing PIL conversion: ``/2 + 0.5`` clamp, permute to
        HWC, uint8) and written with PyAV as a **silent** mp4 — no audio track
        (audio training is out of this phase).  The frame rate is
        ``MINIMAX_H3_FPS = 24``, the PR's fixed generation rate: 37 latent
        frames decode to 124 pixel frames (the decoder expands ×4
        temporally), written at rate=24 → a 5.17 s mp4 matching the
        124-frame source clip.

        Args:
            vae: the video VAE (must expose ``_decode`` for T > 1).
            latent: ``(B, 24, T, H, W)`` normalized latent with T > 1.
            output_dir: directory to write the mp4s (created if missing).
            prefix: optional artifact prefix; when given the file is named
                ``{prefix}_video_<b>.mp4`` (the validation identity, e.g.
                ``{save_name}_val_epoch{epoch}_{sample_idx}``), otherwise
                the legacy ``val_video_<b>.mp4`` is kept (backward
                compatible).

        Returns:
            list of mp4 paths, one per batch item.
        """
        import os

        import av

        frames = self.decode_latent(vae, latent)  # (B, 3, T', H, W) in [-1, 1]
        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []
        for b in range(frames.shape[0]):
            clip = frames[b].cpu()  # (3, T', H, W)
            clip = (clip / 2 + 0.5).clamp(0, 1)  # [0, 1]
            clip = clip.permute(1, 2, 3, 0).float().numpy()  # (T', H, W, 3)
            vid_path = os.path.join(
                output_dir,
                f"{prefix}_video_{b}.mp4" if prefix else f"val_video_{b}.mp4",
            )
            container = av.open(vid_path, mode="w")
            stream = container.add_stream("libx264", rate=MINIMAX_H3_FPS)
            stream.width, stream.height = clip.shape[2], clip.shape[1]
            stream.pix_fmt = "yuv420p"
            for t in range(clip.shape[0]):
                frame = av.VideoFrame.from_ndarray(
                    (clip[t] * 255).round().astype("uint8"), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():  # flush
                container.mux(packet)
            container.close()
            paths.append(vid_path)
        return paths

    # ── Batch extraction helpers ───────────────────────────────────────

    def _extract_encoder_hidden_states(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Stack per-sample cached ``prompt_embed`` into ``(B, L, 5120)``.

        The data pipeline stores per-sample embeddings as numpy dicts with
        ``prompt_embed`` of shape ``(L, 5120)``.  Items are zero-padded to the
        batch-max length; ``prepare_model_input``'s tag-uniformity check then
        rejects batches whose layouts would conflict anyway.
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            raise ValueError(
                "MiniMax-H3 requires cached encoder_hidden_states in "
                "batch['embeddings']. Ensure the caption cache was built with "
                "this adapter (encode_text)."
            )

        prompt_embeds: list[torch.Tensor] = []
        for emb in embeddings:
            if emb is None:
                raise ValueError(
                    "MiniMax-H3 requires a caption embedding for every sample "
                    "in the batch; got None."
                )
            pe = emb["prompt_embed"] if isinstance(emb, dict) else emb
            if isinstance(pe, np.ndarray):
                pe = torch.from_numpy(pe)
            if pe.ndim != 2 or pe.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"MiniMax-H3 expects prompt_embed of shape (L, {self.embedding_dim}), "
                    f"got {tuple(pe.shape)}. The caption cache may be stale — "
                    f"rebuild it with the minimax_h3 adapter."
                )
            prompt_embeds.append(pe.to(device=device, dtype=dtype))

        max_len = max(pe.shape[0] for pe in prompt_embeds)
        padded = []
        for pe in prompt_embeds:
            if pe.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - pe.shape[0], pe.shape[1], dtype=dtype, device=device
                )
                pe = torch.cat([pe, pad], dim=0)
            padded.append(pe)
        return torch.stack(padded)

    def _extract_text_token_tags(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Stack per-sample ``text_token_tags`` into ``(B, L)`` long, ``-1`` padded.

        Captions encoded with a vision block carry video-tagged rows; the
        empty-string (dropout/CFG-uncond) embedding path strips ``text_token_tags``
        (the trainer's ``_apply_caption_dropout`` keeps only ``prompt_embed`` and
        ``prompt_embeds_mask``), so a missing tag array falls back to an
        all-text layout of the embedding's length.
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            raise ValueError(
                "MiniMax-H3 requires batch['embeddings'] to build the packed "
                "text layout."
            )

        tag_lists: list[torch.Tensor] = []
        for emb in embeddings:
            if emb is None:
                raise ValueError(
                    "MiniMax-H3 requires a caption embedding for every sample "
                    "in the batch; got None."
                )
            pe = emb["prompt_embed"] if isinstance(emb, dict) else emb
            length = int(pe.shape[0])
            tags = emb.get("text_token_tags") if isinstance(emb, dict) else None
            if tags is None:
                # Caption-dropout / CFG-uncond path: no tags cached, and the
                # presentation is plain text (no vision block) by construction.
                tags = torch.full(
                    (length,), MINIMAX_H3_TEXT_TAG, dtype=torch.long
                )
            else:
                if isinstance(tags, np.ndarray):
                    tags = torch.from_numpy(tags)
                tags = tags.to(device=device, dtype=torch.long)
            tag_lists.append(tags)

        max_len = max(t.shape[0] for t in tag_lists)
        padded = []
        for t in tag_lists:
            if t.shape[0] < max_len:
                pad = torch.full(
                    (max_len - t.shape[0],), -1, dtype=torch.long, device=device
                )
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def _resolve_condition_latent(
        self,
        batch: dict,
        latents: dict,
        resolved_bc: Optional[dict],
        target_key: Optional[str],
    ) -> torch.Tensor:
        """Resolve the clean source-image (keyframe) latent from the batch.

        ``batch['latents']`` is role-keyed: the target lives under
        ``batch_configs[0]['target_config']`` and the source (keyframe
        condition) under ``batch_configs[0]['reference_config']``.  The
        reference key is preferred; any non-target role is a fallback; else an
        actionable error explains the required config wiring.
        """
        if resolved_bc:
            ref_key = resolved_bc.get("reference_config")
            if ref_key and ref_key in latents:
                return latents[ref_key]
        # Reserved keys = the target-config name AND every image key it
        # resolves to (target_configs[target_config][*].image).  The dataset
        # keys latents by the target-config name (e.g. "T") — a pure-t2v
        # batch is {"T"} only; the image-key union (e.g. "V") is kept as
        # defense against both keyings.  The fallback must never pick the
        # target itself up as a "condition" — that would silently break
        # the image-pair e12 contract and corrupt i2v conditioning.
        reserved = self._reserved_latent_keys(resolved_bc, target_key)
        for k, v in latents.items():
            if k not in reserved and isinstance(v, torch.Tensor):
                return v
        raise ValueError(
            "MiniMax-H3 image-pair training requires a source-image (keyframe) "
            "latent in batch['latents'], but none was found. "
            "batch_configs[0] must reference a reference_config that carries "
            "the source image, e.g.: "
            'reference_configs: {"S": [{"image": "S", "sample_type": "from_same_name"}]} '
            'and batch_configs: [{"target_config": "T", "caption_config": "C", '
            '"reference_config": "S"}]. '
            f"got batch_configs={batch.get('batch_configs')!r}, "
            f"latent roles={list(latents.keys())}."
        )

    def _reserved_latent_keys(
        self, resolved_bc: Optional[dict], target_key: Optional[str]
    ) -> set:
        """Keys never eligible as a condition latent: the target itself.

        The dataset keys ``latents`` by the target-config name (e.g. ``"T"``)
        — a pure-t2v batch is ``{"T"}`` only; the resolved image-key union
        (e.g. ``"V"``) is kept as defense against both keyings.  Mirrors
        ``Trainer._resolve_target_keys``: a ``target_configs[target_config]``
        entry maps a target-config name to its image key(s) (e.g.
        ``{"T": [{"image": "V"}]}``).  Returns ``{target_config name} ∪
        {resolved image keys}``; when the config mapping is unavailable the set
        degrades to ``{target_key}`` (legacy single-name behavior).
        """
        reserved = {target_key} if target_key else set()
        if not resolved_bc or not resolved_bc.get("target_config"):
            return reserved
        tc_key = resolved_bc["target_config"]
        dataset_configs = self.config.get("data", {}).get("dataset_configs", [])
        for ds_cfg in dataset_configs:
            target_configs = ds_cfg.get("target_configs", {})
            if isinstance(target_configs, dict) and tc_key in target_configs:
                entries = target_configs[tc_key]
                if isinstance(entries, list):
                    for e in entries:
                        reserved.add(e.get("image", e) if isinstance(e, dict) else e)
                else:
                    reserved.add(entries)
                break
        return reserved

    def _condition_noise_generator(
        self, sigmas: torch.Tensor, device: torch.device
    ) -> torch.Generator:
        """A reproducible per-noise-level generator for the condition noise.

        The seed is derived from the batch sigma, so the same noise level
        reproduces the same conditioning rows — which keeps the conditional and
        unconditional forwards of a CFG step aligned — while every distinct
        training sigma draws fresh noise ("每步重抽").  This is correct for
        training (one forward per sigma); multi-step inference must build the
        rows once via ``build_condition_rows`` instead, or every denoising
        step would see a differently noised condition.
        """
        sigma_ref = (
            float(sigmas.reshape(-1)[0]) if sigmas.numel() else 0.0
        )
        seed = (self._keyframe_noise_seed + int(round(sigma_ref * 1_000_000))) & 0xFFFFFFFF
        return torch.Generator(device=device).manual_seed(seed)

    def _resolve_condition_rows_source(
        self,
        batch: dict,
        latents: dict,
        resolved_bc: Optional[dict],
        target_key: Optional[str],
        required: bool,
    ) -> Optional[torch.Tensor]:
        """Resolve the clean keyframe-condition latent, or ``None`` when optional.

        ``required=True`` (image-pair, target T == 1) preserves the
        milestone-1 contract: a missing source raises the actionable
        ``_resolve_condition_latent`` error.  ``required=False`` (video,
        T > 1) treats the source as optional — a genuine i2v source is a
        single-frame latent; a resolved multi-frame tensor (the t2v fallback
        can surface the target itself) is not a keyframe condition and yields
        ``None`` (pure t2v), never an error.
        """
        if required:
            return self._resolve_condition_latent(
                batch, latents, resolved_bc, target_key
            )
        try:
            cond_clean = self._resolve_condition_latent(
                batch, latents, resolved_bc, target_key
            )
        except ValueError as e:
            logger.debug(
                f"MiniMax-H3 T>1 video: no condition latent, pure t2v "
                f"layout (keyframe_anchors=()); {e}"
            )
            return None
        if (
            cond_clean.ndim != 5
            or cond_clean.shape[1] != self.latent_channels
            or cond_clean.shape[2] != 1
        ):
            # t2v: the fallback resolver can surface the target video itself
            # when its image key differs from target_config (e.g. latents
            # keyed "V" with target_config "T").  That is NOT a keyframe
            # condition — treat as no condition (pure t2v), never error.  A
            # genuine i2v source is single-frame and still reaches the
            # spatial validation in prepare_model_input.
            logger.debug(
                f"MiniMax-H3 T>1 video: resolved latent is not a "
                f"single-frame keyframe ({tuple(cond_clean.shape)}); "
                "treating as pure t2v (no condition rows)."
            )
            return None
        return cond_clean

    def _mix_condition_rows(
        self,
        cond_clean: torch.Tensor,
        patch: tuple,
        rows_per_frame: int,
        dim_per_row: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Patchify a clean keyframe latent and mix it to the condition noise
        level, PR semantics (packing.py L82-84; scheduler scale_noise
        docstring: x_t = t*x0 + (1 - t)*noise with t = noise_aug):
        x_cond = 0.999*x0 + 0.001*n.

        The mix runs in float32: the 0.001·noise term (~0.001 magnitude) is
        below the bf16 ulp (~0.0078 @ 1.0) and would be rounded away if mixed
        in the working dtype, degrading the rows to a deterministic 0.999·x0.
        The packed sequence still enters the transformer in bf16 (PR
        behavior), but the noise-aug semantics stay real.
        """
        cond_rows = patchify_video_latents(cond_clean, patch).reshape(
            cond_clean.shape[0], rows_per_frame, dim_per_row
        )
        cond_noise = keyframe_condition_noise(
            ((1, cond_clean.shape[3], cond_clean.shape[4]),),
            patch,
            self.latent_channels,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )  # (rows_per_frame, dim_per_row) float32
        cond_rows = (
            MINIMAX_H3_KEYFRAME_NOISE_AUG * cond_rows.to(torch.float32)
            + (1.0 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * cond_noise
        ).to(dtype)
        return cond_rows

    def build_condition_rows(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Build the packed keyframe-condition rows ONCE per generation request.

        PR semantics (before_denoise.py ``PrepareLatentsStep``): the keyframe
        condition is prepared once and frozen for the whole denoising
        trajectory — the model must see the SAME conditioning at every step.
        ``prepare_model_input``'s per-step draw is correct for training (one
        forward per sigma), but its sigma-derived seed would re-noise the
        condition at every validation step; multi-step inference must call
        this once before the denoising loop and pass the result to
        ``prepare_model_input(..., condition_rows=rows)``.

        The noise-aug component is drawn at the adapter's fixed keyframe
        noise base (no sigma offset) — deterministic per request and across
        epochs, mirroring the PR's once-per-request draw.

        Returns:
            ``(B, rows_per_frame, dim_per_row)`` rows, or ``None`` when the
            batch has no keyframe condition (pure t2v video).
        """
        latents = batch.get("latents", {})
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        target_key = resolved_bc.get("target_config")
        cond_clean = self._resolve_condition_rows_source(
            batch, latents, resolved_bc, target_key, required=False
        )
        if cond_clean is None:
            self._condition_rows_shape = None
            return None
        cond_clean = cond_clean.to(device=device, dtype=dtype)
        patch = (1, self.patch_size, self.patch_size)
        rows_per_frame = (
            (cond_clean.shape[3] // self.patch_size)
            * (cond_clean.shape[4] // self.patch_size)
        )
        dim_per_row = cond_clean.shape[1] * patch[0] * patch[1] * patch[2]
        rows = self._mix_condition_rows(
            cond_clean,
            patch,
            rows_per_frame,
            dim_per_row,
            generator=torch.Generator(device=device).manual_seed(
                self._keyframe_noise_seed & 0xFFFFFFFF
            ),
            device=device,
            dtype=dtype,
        )
        self._condition_rows_shape = (cond_clean.shape[3], cond_clean.shape[4])
        return rows

    # ── Training hooks ─────────────────────────────────────────────────

    def prepare_model_input(
        self,
        batch: dict,
        noise: torch.Tensor | list[torch.Tensor],
        sigmas: torch.Tensor,
        condition_rows: Optional[torch.Tensor] = None,
    ) -> dict:
        """Assemble the kwargs dict passed to ``MiniMaxH3Transformer3DModel``.

        Packed layout (no audio; condition rows optional):
            [ text (L) | (condition rows)? | target video rows (T x rows_per_frame) ]

        The target is a 5D ``(B, 24, T, H, W)`` latent with any ``T >= 1``:
        images use ``T == 1`` (image-pair), videos use ``T = 5n+2`` (e.g. 37
        latent frames from 124 pixel frames); ``num_video_rows = T x
        rows_per_frame``.

        Condition-row mode is decided by the batch:
        - ``T == 1`` (image-pair, milestone 1 — unchanged): a source-image
          (keyframe) latent is **required**; ``_resolve_condition_latent``
          raises an actionable error when the batch has none (the e12 negative
          contract is preserved).
        - ``T > 1`` (video, P2): the condition is **optional** — an i2v
          extension uses a single-frame source as the first-frame keyframe
          (``keyframe_anchors=("first",)`` with the mixed-noise condition
          rows); pure t2v (P2 default) has no condition rows
          (``keyframe_anchors=()``).  Never errors on a missing source.
          A video condition source must share the target's spatial size
          (same bucket) — validated like the image path.

        Steps (plan §P1/P2, PR-verified):
        1. Target rows: ``noisy_latents[0]`` -> ``patchify_video_latents((1,2,2))``
           -> ``T * rows_per_frame`` rows (frame-major).
        2. Condition rows (when present): the clean source latent, mixed per
           the PR (packing.py L82-84) to ``0.999 * x0 + 0.001 * noise`` with
           freshly drawn, reproducible noise, then patchified.  Multi-step
           inference passes PRE-BUILT rows via ``condition_rows=`` (built once
           per request by ``build_condition_rows`` — the PR freezes the
           condition for the whole denoising trajectory; the per-step draw
           below is correct only for single-forward training steps).
        3. ``build_packed_sequence(..., num_latent_frames=T, keyframe_anchors=...)``.
        4. Timesteps: ``t_target = 1 - sigma``, ``t_cond = max(t_target, 0.999)``
           (before_denoise.py L417), ``unique([t_cond, t_target])`` with
           per-row-group indices.
        5. ``audio_hidden_states = empty(B, 0, 32)`` (no audio rows).
        6. Forward kwargs dict.

        The batch must be uniform (one shared packed layout per forward):
        uniform sigmas, and identical text length + tag pattern across samples.
        """
        latents = batch.get("latents", {})
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        target_key = resolved_bc.get("target_config")

        # ── 1. Target rows ───────────────────────────────────────────────
        noises = noise if isinstance(noise, list) else [noise]
        if len(noises) != 1:
            raise ValueError(
                f"MiniMax-H3 supports exactly one target per forward, "
                f"got {len(noises)}. Multi-target H3 training is not supported."
            )
        noisy = noises[0]
        if (
            noisy.ndim != 5
            or noisy.shape[1] != self.latent_channels
            or noisy.shape[2] < 1
        ):
            raise ValueError(
                f"MiniMax-H3 expects a 5D (B, {self.latent_channels}, T, H, W) "
                f"noisy target with T >= 1 (image T==1, video T=5n+2), "
                f"got shape {tuple(noisy.shape)}"
            )
        B, C, T, H, W = noisy.shape
        patch = (1, self.patch_size, self.patch_size)
        dim_per_row = C * patch[0] * patch[1] * patch[2]
        rows_per_frame = (H // self.patch_size) * (W // self.patch_size)
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"MiniMax-H3 latent height/width must be divisible by "
                f"patch_size={self.patch_size}, got {H}x{W}."
            )
        num_video_rows = T * rows_per_frame

        target_rows = patchify_video_latents(noisy, patch)
        target_rows = target_rows.reshape(B, num_video_rows, dim_per_row)

        # ── 2. Condition rows (optional for video) ───────────────────────
        # T==1 (image-pair) keeps the milestone-1 contract: a source latent is
        # REQUIRED and _resolve_condition_latent raises an actionable error.
        # T>1 (video) treats the source as OPTIONAL: i2v (with a source) uses
        # keyframe_anchors=("first",); pure t2v (P2 default) has no condition
        # rows at all (keyframe_anchors=()) and never errors.
        #
        # Multi-step inference (validation generation) passes PRE-BUILT rows
        # via ``condition_rows=`` — built once per request by
        # ``build_condition_rows`` and reused at every denoising step (PR
        # semantics).  The per-step branch below re-draws the noise-aug
        # component at every call with a sigma-derived seed, which is correct
        # for training (one forward per sigma) but would wobble the
        # conditioning across an inference trajectory.
        cond_rows = condition_rows
        keyframe_anchors: tuple[str, ...] = ()
        if cond_rows is not None:
            keyframe_anchors = ("first",)
            if (
                cond_rows.ndim != 3
                or cond_rows.shape[0] != B
                or cond_rows.shape[1] != rows_per_frame
                or cond_rows.shape[2] != dim_per_row
            ):
                raise ValueError(
                    f"MiniMax-H3 pre-built condition rows must be "
                    f"(B={B}, rows_per_frame={rows_per_frame}, "
                    f"dim_per_row={dim_per_row}), got {tuple(cond_rows.shape)}. "
                    "Rebuild with adapter.build_condition_rows(batch) — rows "
                    "built for another batch/target cannot be reused."
                )
            if (
                self._condition_rows_shape is not None
                and tuple(self._condition_rows_shape) != (H, W)
            ):
                raise ValueError(
                    f"MiniMax-H3 source and target latents must share the same "
                    f"spatial size (one bucket per batch); source "
                    f"{self._condition_rows_shape} vs target ({H}, {W})."
                )
        else:
            cond_clean = self._resolve_condition_rows_source(
                batch, latents, resolved_bc, target_key, required=(T == 1)
            )
            if cond_clean is not None:
                cond_clean = cond_clean.to(device=noisy.device, dtype=noisy.dtype)
                if cond_clean.ndim != 5 or cond_clean.shape[1] != self.latent_channels:
                    raise ValueError(
                        f"MiniMax-H3 condition latent must be 5D "
                        f"(B, {self.latent_channels}, 1, H, W), "
                        f"got {tuple(cond_clean.shape)}"
                    )
                if tuple(cond_clean.shape[2:]) != (1, H, W):
                    raise ValueError(
                        f"MiniMax-H3 source and target latents must share the same "
                        f"spatial size (one bucket per batch); source "
                        f"{tuple(cond_clean.shape[2:])} vs target ({T}, {H}, {W})."
                    )
                cond_rows = self._mix_condition_rows(
                    cond_clean,
                    patch,
                    rows_per_frame,
                    dim_per_row,
                    generator=self._condition_noise_generator(sigmas, noisy.device),
                    device=noisy.device,
                    dtype=noisy.dtype,
                )
                keyframe_anchors = ("first",)

        # ── 3. Text + packed layout ──────────────────────────────────────
        prompt_embeds = self._extract_encoder_hidden_states(
            batch, noisy.device, noisy.dtype
        )  # (B, L, 5120)
        text_tags = self._extract_text_token_tags(batch, noisy.device)  # (B, L) long
        first_tags = text_tags[0]
        for i in range(1, B):
            if not torch.equal(first_tags, text_tags[i]):
                raise ValueError(
                    "MiniMax-H3 packs one shared layout per forward: every "
                    "sample in a batch must share the same text length AND "
                    "token-tag pattern. Sample 0 and sample "
                    f"{i} differ (e.g. one caption has a vision block and "
                    "another does not, or caption lengths differ). Use "
                    "batch_size=1, or batch samples with identical caption "
                    "token layouts."
                )

        layout = build_packed_sequence(
            text_token_tags=first_tags,
            num_latent_frames=T,
            latent_height=H,
            latent_width=W,
            num_audio_latents=0,
            patch_size=patch,
            keyframe_anchors=keyframe_anchors,
        )

        # ── 4. Timesteps ────────────────────────────────────────────────
        sigmas_flat = sigmas.reshape(-1)
        if sigmas_flat.numel() == 0 or not torch.all(sigmas_flat == sigmas_flat[0]):
            raise ValueError(
                "MiniMax-H3 packs one shared layout per forward: sigmas must "
                f"be batch-uniform, got {sigmas_flat.tolist()[:8]}. The "
                "adapter's sample_timesteps already emits one sigma per forward."
            )
        sigma = float(sigmas_flat[0])
        t_target = 1.0 - sigma
        timestep, timestep_indices = build_row_timesteps(
            layout,
            video_timestep=t_target,
            audio_timestep=t_target,
            # PR: condition rows run at max(t, 0.999) (before_denoise.py
            # L417) — the pinned level for every reachable sigma (>= 1e-3).
            condition_video_timestep=max(t_target, _H3_CONDITION_TIMESTEP),
            condition_audio_timestep=max(t_target, _H3_CONDITION_TIMESTEP),
        )

        # ── 5. Empty audio rows ─────────────────────────────────────────
        # audio_proj_in is nn.Linear(audio_in_channels=32, hidden_size=5376);
        # with zero rows the linear projection still validates the feature dim,
        # so the empty tensor carries 32 features (NOT 5376 — see module doc).
        audio_hidden_states = torch.empty(
            B, 0, _H3_AUDIO_IN_CHANNELS, device=noisy.device, dtype=noisy.dtype
        )

        # ── 6. Forward kwargs ───────────────────────────────────────────
        dev = noisy.device
        self._packed_geometry = {
            "latent_height": H,
            "latent_width": W,
            "num_latent_frames": T,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "num_video_rows": num_video_rows,
        }
        hidden_states = (
            torch.cat([cond_rows, target_rows], dim=1)
            if cond_rows is not None
            else target_rows
        )
        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep.to(dev),
            "timestep_indices": timestep_indices.to(dev),
            "token_tags": layout.token_tags.to(dev),
            "position_ids": layout.position_ids.to(dev),
            "video_indices": layout.video_indices.to(dev),
            "audio_indices": layout.audio_indices.to(dev),
            "text_indices": layout.text_indices.to(dev),
            "return_dict": False,
        }

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> list[torch.Tensor]:
        """Unpack the transformer output into per-target velocity tensors.

        ``return_dict=False`` yields a ``(video, audio)`` tuple; the video rows
        are ordered ``[condition; target]`` matching ``video_indices`` (with
        ``num_condition_video_rows == 0`` for pure-t2v video, the
        ``video[:, 0:num_video]`` slice naturally starts at row 0).  The
        target rows are unpatchified back to 5D ``(B, 24, T, H, W)`` —
        ``T == 1`` for image-pair, ``T = 5n+2`` for video — so the loss
        modules can compare against the (5D) noises.
        """
        if hasattr(model_pred, "sample"):
            video = model_pred.sample
        elif isinstance(model_pred, (tuple, list)):
            video = model_pred[0]
        else:
            raise ValueError(
                f"MiniMax-H3 transformer output must be a "
                f"MiniMaxH3TransformerOutput or (video, audio) tuple, got "
                f"{type(model_pred).__name__}"
            )

        geo = self._packed_geometry
        if geo is None:
            raise ValueError(
                "MiniMax-H3 unpack_prediction requires prepare_model_input to "
                "have run first (the packed geometry is resolved there)."
            )
        num_cond = geo["num_condition_video_rows"]
        num_video = geo["num_video_rows"]
        if video.shape[1] != num_cond + num_video:
            raise ValueError(
                f"Model returned {video.shape[1]} video rows but the packed "
                f"layout has {num_cond} condition + {num_video} target rows. "
                f"Is the layout stale? Call prepare_model_input first."
            )

        target_rows = video[:, num_cond:num_cond + num_video]
        latent = unpatchify_video_tokens(
            target_rows.reshape(-1, target_rows.shape[-1]),
            num_latent_frames=geo["num_latent_frames"],
            latent_height=geo["latent_height"],
            latent_width=geo["latent_width"],
            channels=self.latent_channels,
            patch_size=(1, self.patch_size, self.patch_size),
        )  # (B, 24, 1, H, W)
        return [latent]

    def compute_target(
        self, noise: torch.Tensor, learning_target: torch.Tensor
    ) -> torch.Tensor:
        # MiniMax-H3's scheduler is data-ward: v = x0 - x_t, the opposite of
        # the standard noise - x0 convention (see base.velocity_sign).
        return learning_target - noise

    # ── Logit-normal timestep sampling ─────────────────────────────────

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample sigma from a logit-normal with a configurable mu.

        MiniMax-H3 shares one packed layout across the batch, so a single sigma
        is drawn and broadcast over ``batch_size`` (the engine's flow
        interpolation then mixes every target at the same noise level).  The
        returned value is the engine's sigma; the H3 transformer's actual
        timestep ``t = 1 - sigma`` is built inside ``prepare_model_input``.
        """
        u = torch.normal(mean=self.timestep_mu, std=1.0, size=(1,), device=device)
        sigma = torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5)
        sigmas = sigma.expand(batch_size).to(dtype=dtype)
        return sigmas, sigmas
