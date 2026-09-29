"""Qwen-Image 2.1 model adapter — single-stream block-causal DiT, 64-ch latent VAE.

Architecture (verified against diffusers transformer_qwenimage21 /
pipeline_qwenimage21, mirrored by transformer_qwenimage21.py in this package):

- Transformer consumes latents UNPATCHED (patch_size=1); packing is a plain
  spatial flatten (B, C, H, W) -> view(B, C, H*W).transpose(1, 2) — pipeline
  _pack_latents (pipeline_qwenimage21.py:410-412). Prediction slicing takes
  the LAST H*W rows of the joint output (pipeline: noise_pred[:,
  -latents.size(1):]).
- Joint sequence: [VLM text tokens (image slots expanded 4x), target slots
  (one per 2x2 latent-token group)]. _IMG_TOKENS_PER_SLOT = 4 — one VLM
  image slot == one 2x2 latent group == a 32x32 pixel tile (16x VAE x 2x2
  grouping), hence bucket_divisibility = 32 (pipeline enforces
  multiple_of = vae_scale_factor * 2).
- Text encoder: Qwen3VLForConditionalGeneration. Hidden states must be the
  LAST decoder layer BEFORE the final RMSNorm — captured with a forward hook on
  text_model.norm returning its input (transformers 5.x ties
  hidden_states[-1] to the normalized last_hidden_state; pipeline
  pipeline_qwenimage21.py:297-310).
- Prompt: raw template string (NOT apply_chat_template), left padding,
  empty prompt -> " " (Qwen has no BOS). _drop_idx = token count of the
  tokenized system message; stripped from every extracted row.
- Timestep: the transformer receives timestep in [0, 1] and multiplies by
  1000 internally (QwenImage21TemporalTimesteps; pipeline passes t / 1000).

Multimodal cache dual-key trick
-------------------------------
cache_builder._encode_caption POPS image_token_mask from the embedding dict
before saving the npz (data/cache_builder.py:896), so that key alone would
NOT survive into the cache. encode_text therefore stores the SAME slot mask
tensor under a second key img_mask (never popped); prepare_model_input reads
img_mask first, falls back to image_token_mask (in-memory path), then
all-False (text-only). The npz keeps img_mask as a raw bool array
(EmbeddingCache.save only quantizes float arrays).

Edit path status: implemented end-to-end (multimodal encode_text + VAE
reference packing in prepare_model_input), but the shipped example configs
are text-to-image only — T2I is the verified milestone; edit-config examples
follow once model weights are available.

Transparency (RGBA) training:
    ``vae_pixel_channels = 4`` — the data pipeline loads source PNG/WebP
    images as RGBA (``load_image_frames(..., channels=4)``) and feeds the
    real alpha channel to the 4-channel VAE, matching the official pipeline's
    ``img.convert("RGBA")`` input.  Opaque sources get alpha=255 and produce
    bit-identical latents to the historical 3-channel + padded-alpha path.
    See md/08-qwenimage21-training.md §5.1 for the full chain.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter, sample_sigma_uniform
from UnifiedTrainer.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Aspect buckets in krea2 style (models/krea2/__init__.py RESOLUTION_CONFIG):
# {base_resolution: [(w, h), ...]} — landscape pairs first, portrait mirrors
# after. EVERY dimension is a multiple of 32 (the pipeline rounds height/width
# down to multiple_of = vae_scale_factor * 2 = 32; see
# pipeline_qwenimage21.py:631-633). 512/1024/1536/2048 reuse the krea2 lists
# verbatim (all dims already 32-aligned); the 768 bucket scales the 512 pairs
# by 1.5 rounded to the 32 grid.
RESOLUTION_CONFIG = {
    512: [
        (512, 512),
        (576, 448), (640, 384), (704, 320), (768, 288),
        (832, 256), (896, 224), (960, 192),
        (448, 576), (384, 640), (320, 704), (288, 768),
        (256, 832), (224, 896), (192, 960),
    ],
    768: [
        (768, 768),
        (864, 672), (960, 576), (1056, 480), (1152, 448),
        (1248, 384), (1344, 320), (1440, 288),
        (672, 864), (576, 960), (480, 1056), (448, 1152),
        (384, 1248), (320, 1344), (288, 1440),
    ],
    1024: [
        (1024, 1024),
        (1120, 960), (1152, 896), (1216, 832),
        (1280, 768), (1344, 704), (1408, 640), (1472, 576),
        (1536, 544), (1600, 512), (1664, 480), (1728, 448),
        (960, 1120), (896, 1152), (832, 1216),
        (768, 1280), (704, 1344), (640, 1408), (576, 1472),
        (544, 1536), (512, 1600), (480, 1664), (448, 1728),
    ],
    1536: [
        (1536, 1536),
        (1664, 1440), (1728, 1344), (1824, 1248),
        (1920, 1152), (2016, 1056), (2112, 960), (2208, 864),
        (2304, 832), (2400, 768), (2496, 704), (2592, 672),
        (1440, 1664), (1344, 1728), (1248, 1824),
        (1152, 1920), (1056, 2016), (960, 2112), (864, 2208),
        (832, 2304), (768, 2400), (704, 2496), (672, 2592),
    ],
    2048: [
        (2048, 2048),
        (2240, 1920), (2304, 1792), (2432, 1664),
        (2560, 1536), (2688, 1408), (2816, 1280), (2944, 1152),
        (1920, 2240), (1792, 2304), (1664, 2432),
        (1536, 2560), (1408, 2688), (1280, 2816), (1152, 2944),
    ],
}

# Dynamic-shifting FlowMatchEulerDiscreteScheduler config — Qwen-Image 2.1.
#
# These MUST equal the model's own scheduler/scheduler_config.json, because the
# pipeline derives mu from exactly these numbers: pipeline_qwenimage21.py:724-729
# calls calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
# base_shift, max_shift) reading ALL of them out of the scheduler config. The
# 4096 / 1.15 pair is merely calculate_shift's *signature default* and is NOT
# what this model ships — using it over-shifts training vs inference
# (at 1024x1024, seq=4096: mu_train=1.15 vs mu_infer=0.6935, +66%).
#
# Values below are verbatim from Qwen/Qwen-Image-2.1
# scheduler/scheduler_config.json (verified byte-identical in both the local
# E:\hf_models and remote /home/waas/hf_models copies):
#   base_image_seq_len=256, max_image_seq_len=8192,
#   base_shift=0.5,         max_shift=0.9
# The scheduler instance itself is built in load_scheduler and ignores the
# on-disk scheduler path (krea2 convention).
QWEN21_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "base_shift": 0.5,
    "max_shift": 0.9,
}

# Qwen-Image 2.1 prompt-template constants — VERBATIM from
# pipeline_qwenimage21.py:206-220. Raw strings passed straight to the
# processor; apply_chat_template tokenizes differently and the checkpoint
# expects these. The ti2i template carries ONE <image1> placeholder that is
# expanded n-fold at encode time (pipeline _get_qwen_prompt_embeds 249-259).
_SYS_PROMPT = "Comprehend and analyze the provided prompt."
_PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{_SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
_PROMPT_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{_SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
_IMG_PLACEHOLDER = "<image{i}><|vision_start|><|image_pad|><|vision_end|>"
# One VLM image slot == one 2x2 group of latent tokens (transformer
# _IMG_TOKENS_PER_SLOT; transformer_qwenimage21.py:938-939).
_IMG_TOKENS_PER_SLOT = 4
# Caption token budget for the TEXT-ONLY path (multimodal prompts are never
# truncated — pipeline parity). Default matches krea2's rationale: no model
# text-length limit, 1024 keeps sequence cost negligible while matching what
# inference conditions on. Override with training.text_max_length (alias:
# top-level text_max_length).
_DEFAULT_MAX_SEQ_LEN = 1024
# Chat-template scaffold headroom on top of text_max_length when truncating
# the RAW template string: covers the system message (dropped again by
# _drop_idx after extraction) plus the user opener / assistant suffix that
# stay in the extracted rows (pipeline parity — the pipeline never strips
# them). 16 is generous; truncation only bounds worst-case memory.
_TEMPLATE_HEADROOM = 16


@ModelRegistry.register("qwen_image21")
class QwenImage21Adapter(BaseModelAdapter):
    """Adapter for Qwen-Image 2.1 (T2I + image-conditioned editing)."""

    name = "qwen_image21"
    # 2.1 consumes latents UNPATCHED — packing is a plain spatial flatten.
    patch_size = 1

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

        # Caption token budget for the text-only encode path (see
        # _DEFAULT_MAX_SEQ_LEN).
        _training_cfg = config.get("training", {}) or {}
        self.text_max_length: int = int(
            config.get(
                "text_max_length",
                _training_cfg.get("text_max_length", _DEFAULT_MAX_SEQ_LEN),
            )
        )
        if self.text_max_length < 64:
            raise ValueError(
                f"qwen_image21: text_max_length must be >= 64, got {self.text_max_length!r}"
            )

        # Timestep shift mode for training-time sigma sampling:
        #   "dynamic_shift" — DEFAULT, pipeline-parity: logit-normal with mu
        #                     linearly interpolated 0.5->0.9 over 256->8192
        #                     image tokens (calculate_shift +
        #                     FlowMatchEulerDiscreteScheduler dynamic
        #                     shifting; values mirror the model's own
        #                     scheduler_config.json).
        #   "flow_shift"    — fixed-s logit-normal sigma = sigmoid(N(ln s,
        #                     scale)) (krea2 recipe; s = flow_shift, default
        #                     2.5).
        #   "sigma"         — uniform sigma in [1/1000, 1] via the shared
        #                     base-adapter helper (musubi recipe).
        self.timestep_shift_mode: str = config.get("timestep_shift_mode", "dynamic_shift")
        self.flow_shift: float = float(config.get("flow_shift", 2.5))
        self.sigmoid_scale: float = float(config.get("sigmoid_scale", 1.0))
        if self.timestep_shift_mode not in ("dynamic_shift", "flow_shift", "sigma"):
            raise ValueError(
                f"Unknown timestep_shift_mode '{self.timestep_shift_mode}'. "
                "Expected 'dynamic_shift', 'flow_shift' or 'sigma'."
            )

        # Target grid dims cached from prepare_model_input — used by
        # unpack_prediction to slice the LAST target rows (pipeline parity,
        # no sqrt guesses). Latent-space (H, W); patch_size=1 so latent dims
        # == token dims.
        self._target_grids: list[tuple[int, int]] = []
        self._ref_grids: list[tuple[int, int]] = []

        # Template metadata derived from the processor/tokenizer once
        # (pipeline __init__ 223-226 computes the same values per instance).
        self._drop_idx: Optional[int] = None
        self._img_token_id: Optional[int] = None
        self._im_end_id: Optional[int] = None

    # ── Model loading ──────────────────────────────────────────────────

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        # Standalone in-package copy (Deliverable A) — lazy import so this
        # module stays importable before that file exists.
        from .transformer_qwenimage21 import BlockSwapQwenImage21Transformer2DModel

        return BlockSwapQwenImage21Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        # Generic diffusers VAE infra import — allowed; only QwenImage21
        # TRANSFORMER classes are vendored (spec constraint).
        from diffusers import AutoencoderKLQwenImage21

        return AutoencoderKLQwenImage21.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        # Dynamic-shifting flow-match schedule, built from constants (path
        # ignored — krea2 convention): mu interpolation 0.5->0.9 over
        # 256->8192 image tokens happens through use_dynamic_shifting at
        # inference; training samples sigmas itself (sample_timesteps).
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=QWEN21_SCHEDULER_CONFIG["num_train_timesteps"],
            shift=1.0,
            use_dynamic_shifting=True,
            base_image_seq_len=QWEN21_SCHEDULER_CONFIG["base_image_seq_len"],
            max_image_seq_len=QWEN21_SCHEDULER_CONFIG["max_image_seq_len"],
            base_shift=QWEN21_SCHEDULER_CONFIG["base_shift"],
            max_shift=QWEN21_SCHEDULER_CONFIG["max_shift"],
        )

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import AutoConfig, Qwen3VLForConditionalGeneration

            config = AutoConfig.from_pretrained(path)
            # Patch rope_scaling if missing — transformers calls
            # config.rope_scaling.get() without null-checking. The
            # rope_scaling lives on the text_config sub-config (krea2
            # pattern, models/krea2/__init__.py load_text_encoder).
            mrope = {"rope_type": "default", "mrope_section": [24, 20, 20]}
            if hasattr(config, "text_config"):
                if getattr(config.text_config, "rope_scaling", None) is None:
                    config.text_config.rope_scaling = mrope
            elif getattr(config, "rope_scaling", None) is None:
                config.rope_scaling = mrope
            return Qwen3VLForConditionalGeneration.from_pretrained(
                path, config=config, torch_dtype=dtype
            )
        except Exception as e:
            logger.warning(f"Failed to load Qwen-Image 2.1 text encoder: {e}")
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        # Plain fallback contract (spec): no config patching — a tokenizer
        # that fails to load surfaces as None and the trainer reports it.
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(path)
        except Exception as e:
            logger.warning(f"Failed to load Qwen-Image 2.1 tokenizer from {path}: {e}")
            return None

    def load_processor(self, path: str, tokenizer_path: str = "") -> Optional[Any]:
        """Load the Qwen3VL processor for multimodal text+image encoding.

        Uses AutoProcessor when the model directory ships a
        preprocessor_config.json; otherwise constructs a Qwen3VLProcessor
        manually from a Qwen2VLImageProcessor, a Qwen3VLVideoProcessor and
        the tokenizer loaded from tokenizer_path (falls back to path) —
        mirrors qwen_image/krea2 load_processor, reading the vision config
        for patch/merge sizes (Qwen3VL vision expects patch_size=16).
        """
        try:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(path)
        except Exception:
            pass

        # Fallback: construct Qwen3VLProcessor manually (krea2 pattern).
        tok_path = tokenizer_path or path
        try:
            from transformers import Qwen3VLProcessor
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

            import json
            import os

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
                    f"Cannot construct Qwen3VLProcessor: tokenizer is None from {tok_path}"
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
            logger.warning(f"Failed to construct Qwen3VLProcessor from {path}: {e}")
            return None

    # ── Architecture specs ─────────────────────────────────────────────

    @property
    def latent_channels(self) -> int:
        return 64

    @property
    def vae_pixel_channels(self) -> int:
        # 4 = RGBA: the 2.1 VAE is a 4-channel causal video VAE
        # (vae.config in_channels=4 / out_channels=4) and the official
        # pipeline feeds it `img.convert("RGBA")` — the alpha channel is a
        # real input the VAE encodes (transparency training), not padding.
        # Declaring 4 makes the data pipeline load source PNG/WebP images
        # as RGBA (transforms.to_tensor channels=4) so real alpha reaches
        # encode_image; opaque sources simply get alpha=255, byte-identical
        # to the historical 3-channel + padded-alpha path.
        return 4

    @property
    def vae_scale_factor(self) -> int:
        return 16

    @property
    def embedding_dim(self) -> int:
        return 4096

    @property
    def bucket_divisibility(self) -> int:
        # Override of vae_scale_factor * patch_size (=16): the pipeline
        # enforces 32 (multiple_of = vae_scale_factor * 2) because one VLM
        # image slot covers a 2x2 group of latent tokens — i.e. a 32x32
        # pixel tile. Misaligned dims would desync img_shapes from the VLM
        # slot mask.
        return 32

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return True

    # ── Encoding ───────────────────────────────────────────────────────

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            # The 2.1 VAE is a 4-channel causal video VAE (vae.config
            # in_channels=4 / out_channels=4), and the official pipeline feeds it
            # RGBA: `pipeline_qwenimage21.py` does `img.convert("RGBA")` before
            # `image_processor.preprocess(...).unsqueeze(2)` (lines 653-663).
            # A 4-channel input carries a REAL alpha channel (PNG/WebP
            # transparency loaded by the data pipeline via
            # `vae_pixel_channels=4`) and is fed through unchanged.  A legacy
            # 3-channel tensor would crash conv_in ("weight of size [96, 4, 3, 3]
            # ... expected input[...] to have 4 channels, but got 3"), so pad a
            # fully opaque alpha channel — `.convert("RGBA")` also sets alpha to
            # 255, i.e. 1.0, reproducing the pipeline's RGBA input exactly.
            if image_tensor.shape[1] == 3:
                alpha = torch.ones_like(image_tensor[:, :1])
                image_tensor = torch.cat([image_tensor, alpha], dim=1)
            # AutoencoderKLQwenImage21 is a video VAE expecting 5D input
            # (B, C, T, H, W).
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(2)
            latent = vae.encode(image_tensor).latent_dist.sample()
            # Squeeze temporal dim back to 4D (B, C, H, W).
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            # 64-channel per-channel normalization with the VAE's OWN
            # calibration values (pipeline _encode_vae_image 422-444). No
            # hardcoded T2I override — weights are unreleased, vae.config
            # is the only truth for 2.1.
            latents_mean = torch.tensor(
                vae.config.latents_mean, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latents_std = torch.tensor(
                vae.config.latents_std, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latent = (latent - latents_mean) / latents_std
        return {"latent": latent}

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        # Cast to the VAE's dtype/device FIRST — cached latents arrive as
        # fp32 while the validation VAE is bf16 (krea2 convention; a
        # mismatch raises "Input type (float) and bias type (BFloat16)
        # should be the same" in the conv bias add).
        latent = latent.to(device=vae.device, dtype=vae.dtype)
        # Denormalize (inverse of encode_image).
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latents_std = torch.tensor(
            vae.config.latents_std, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latent = latent * latents_std + latents_mean
        # 5D input (B, C, T, H, W).
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        with torch.no_grad():
            image = vae.decode(latent).sample
        if image.ndim == 5:
            image = image.squeeze(2)
        return image

    def decode_latent_differentiable(
        self, vae: nn.Module, latent: torch.Tensor, tile_size: int = 0
    ) -> Any:
        """Gradient-enabled decode for perceptual losses (losses/pfm.py).

        Same math as decode_latent (config mean/std denormalization + 5D
        decode) but WITHOUT torch.no_grad() so autograd can backprop from
        pixel space into x0_hat -> model_pred. tile_size is accepted for the
        base-class signature but deliberately unused: a tiled branch would
        have to keep the grad and no-grad loss paths bit-identical
        (BaseModelAdapter.decode_latent_differentiable contract) and the 16x
        VAE decodes within budget at supported bucket resolutions.
        """
        latent = latent.to(device=vae.device, dtype=vae.dtype)
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latents_std = torch.tensor(
            vae.config.latents_std, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latent = latent * latents_std + latents_mean
        squeeze_frame = latent.ndim == 4
        if squeeze_frame:
            latent = latent.unsqueeze(2)
        image = vae.decode(latent).sample
        if squeeze_frame and image.ndim == 5:
            image = image.squeeze(2)
        return image

    # ── Text encoding (pipeline parity) ────────────────────────────────

    def _resolve_template_meta(self, processor: Any, tokenizer: Any) -> None:
        """Derive and cache _drop_idx / _img_token_id once.

        _drop_idx = token count of the tokenized system message (pipeline
        __init__ 223-225: apply_chat_template(sys_message, tokenize=True,
        return_dict=False)) — derived rather than hardcoded so it tracks the
        processor's template. _img_token_id = the <|image_pad|> id (pipeline
        line 226). Falls back from the processor's tokenizer to the bare
        tokenizer when no processor is available (same jinja template either
        way).
        """
        if (
            self._drop_idx is not None
            and self._img_token_id is not None
            and self._im_end_id is not None
        ):
            return
        holder = processor if processor is not None else tokenizer
        if holder is None:
            raise RuntimeError(
                "qwen_image21 encode_text needs a processor or tokenizer to "
                "resolve the chat-template metadata (_drop_idx / image pad id)."
            )
        try:
            sys_message = [
                {"role": "system", "content": [{"type": "text", "text": _SYS_PROMPT}]}
            ]
            sys_tokens = holder.apply_chat_template(
                sys_message, tokenize=True, return_dict=False
            )
            drop_idx = len(sys_tokens[0])
            tok = getattr(holder, "tokenizer", holder)
            img_token_id = tok.encode("<|image_pad|>")[0]
            im_end_id = tok.encode("<|im_end|>")[0]
        except Exception as e:
            raise RuntimeError(
                "qwen_image21: failed to derive chat-template metadata "
                f"(_drop_idx / <|image_pad|> id) from {type(holder).__name__}: {e}"
            ) from e
        self._drop_idx = drop_idx
        self._img_token_id = img_token_id
        self._im_end_id = im_end_id
        logger.debug(
            f"qwen_image21 template metadata: _drop_idx={drop_idx}, "
            f"_img_token_id={img_token_id}"
        )

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> list:
        """Pipeline-parity masked-row extraction (pipeline 228-232).

        Left-padding safe: hidden[bool_mask] keeps only valid rows, then
        splits them back per sample by the valid lengths.
        """
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def encode_text(
        self,
        text_encoder: Optional[nn.Module],
        tokenizer: Optional[Any],
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        reference_image: Optional[Any] = None,
        processor: Optional[Any] = None,
    ) -> dict:
        """Encode a prompt (and optionally reference images) via Qwen3-VL.

        Mirrors QwenImage21Pipeline._get_qwen_prompt_embeds
        (pipeline_qwenimage21.py:234-331): raw template strings, empty
        prompt -> " ", left padding, masked-row extraction, system-prefix
        drop, pre-final-RMSNorm hook.

        Returns
        -------
        dict
            prompt_embed        — (seq_len, 4096) float
            prompt_embeds_mask  — (seq_len,) bool ALL-TRUE (no padding
                                  remains after masked extraction —
                                  pipeline parity)
            image_token_mask    — (seq_len,) bool (all-False for
                                  text-only) — popped by cache_builder
                                  before the npz save
            img_mask            — SAME tensor under a key the cache
                                  builder does NOT pop, so the multimodal
                                  slot mask survives into the cache npz
                                  (see module docstring)
        """
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "Qwen-Image 2.1 requires a loaded Qwen3VL text encoder and tokenizer."
            )

        is_multimodal = reference_image is not None and processor is not None
        if reference_image is not None and processor is None:
            logger.warning(
                "qwen_image21 encode_text: reference image(s) provided but no "
                "processor — falling back to TEXT-ONLY encoding."
            )

        self._resolve_template_meta(processor, tokenizer)

        # ── 1. Build prompt text (raw template strings — pipeline 241-259) ─
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompts = [" " if not p else p for p in prompt]  # Qwen has no BOS

        refs: list[Any] = []
        images_kw: dict[str, Any] = {}
        if is_multimodal:
            from PIL import Image as PILImage

            refs = reference_image if isinstance(reference_image, list) else [reference_image]
            # Expand the single <image1> placeholder n-fold, exactly like
            # the pipeline (249-259).
            base_ph = _IMG_PLACEHOLDER.format(i=1)
            replace = base_ph
            for i in range(2, len(refs) + 1):
                replace += f" {_IMG_PLACEHOLDER.format(i=i)}"
            template = _PROMPT_TEMPLATE_TI2I.replace(base_ph, replace)
            prompts = [template.format(t) for t in prompts]

            # One set of images per prompt, in placeholder order. RGBA is
            # flattened over white for the VISION encoder only (pipeline
            # 262-272) — the VAE still reads all four channels.
            condition_pil_list = []
            for _ in prompts:
                for img in refs:
                    if not isinstance(img, PILImage.Image):
                        img = PILImage.fromarray(img)
                    if img.mode == "RGBA":
                        white = PILImage.new("RGB", img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel("A"))
                        img = white
                    condition_pil_list.append(img)
            images_kw = {"images": condition_pil_list}
        else:
            raw_texts = list(prompts)  # pre-template texts (opener metadata)
            prompts = [_PROMPT_TEMPLATE_T2I.format(t) for t in prompts]

        # ── 2. Tokenize (pipeline 276-285) ─────────────────────────────────
        processor_kwargs: dict[str, Any] = dict(
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            **images_kw,
        )
        if not is_multimodal:
            # Text-only: truncate at the configured budget + template
            # headroom (see _TEMPLATE_HEADROOM). NO truncation on the
            # multimodal path (pipeline parity).
            processor_kwargs.update(
                truncation=True,
                max_length=self.text_max_length + self._drop_idx + _TEMPLATE_HEADROOM,
            )

        if processor is not None:
            model_inputs = processor(**processor_kwargs).to(device)
        else:
            # No processor available: bare-tokenizer fallback, text-only.
            tok_kwargs: dict[str, Any] = dict(
                text=processor_kwargs["text"],
                padding=True,
                padding_side="left",
                truncation=True,
                max_length=processor_kwargs.get("max_length"),
                return_tensors="pt",
            )
            model_inputs = tokenizer(**tok_kwargs).to(device)

        # ── 3. Forward kwargs (pipeline 287-295) ───────────────────────────
        forward_kwargs: dict[str, Any] = dict(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )
        if is_multimodal and getattr(model_inputs, "pixel_values", None) is not None:
            forward_kwargs["pixel_values"] = model_inputs.pixel_values.to(text_encoder.dtype)
            forward_kwargs["image_grid_thw"] = model_inputs.image_grid_thw
        if getattr(model_inputs, "mm_token_type_ids", None) is not None:
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        # ── 4. Forward with pre-final-RMSNorm hook (pipeline 297-310) ──────
        # hidden_states[-1] must be the last decoder layer BEFORE the final
        # RMSNorm. transformers 5.x ties that entry to the normalized
        # last_hidden_state, so a forward hook returning the module's INPUT
        # neutralizes the norm for this call on either transformers version.
        hook_handle = None
        try:
            text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
            hook_handle = text_model.norm.register_forward_hook(
                lambda module, args, output: args[0]
            )
        except Exception as e:
            logger.warning(
                "qwen_image21: could not register pre-final-norm hook "
                f"({e}); falling back to outputs.hidden_states[-1] — on "
                "transformers 5.x this may be RMSNorm-normalized."
            )
        try:
            with torch.no_grad():
                outputs = text_encoder(**forward_kwargs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        hidden_states = outputs.hidden_states[-1]

        # ── 5. Masked-row extraction + system-prefix drop (pipeline 312-313)
        split_hidden = list(
            self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        )
        split_hidden = [e[self._drop_idx:] for e in split_hidden]

        # image_pad_mask per sample: <|image_pad|> positions among valid
        # tokens, sliced past the system prefix likewise (pipeline 315-319).
        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [e[self._drop_idx:] for e in image_pad_mask]

        # ── 6. Stack (pipeline 321-329) — one prompt per call, so the
        # padded length is the row's own length and no padding survives:
        # the mask is ALL-TRUE (pipeline encode_prompt 382-385 drops the
        # mask entirely in this case; prepare_model_input rebuilds the
        # padded mask from per-sample lengths).
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden
        ]
        max_seq_len = max(e.size(0) for e in split_hidden)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask]
        )

        slot_mask = image_pad_mask[0].to(torch.bool)

        # Where the user text starts inside the extracted row (tokens). The
        # raw template is [system (dropped by _drop_idx)][user opener][TEXT]
        # [<|im_end|>]..., so opener_len = first <|im_end|> at/after
        # _drop_idx − _drop_idx − len(TEXT tokens). The edit-uncond slot
        # synthesis (prepare_model_input) uses this to place synthesized
        # reference slots at the exact ti2i position (the template puts
        # <image_i> right before {text}). Only meaningful for the text-only
        # path — the empty embedding is the only consumer.
        user_opener_len = -1
        if not is_multimodal:
            try:
                holder = processor if processor is not None else tokenizer
                tok = getattr(holder, "tokenizer", holder)
                row_ids = model_inputs.input_ids[0].tolist()
                im_end_idx = next(
                    (i for i, t in enumerate(row_ids)
                     if i >= self._drop_idx and t == self._im_end_id),
                    -1,
                )
                n_text_tokens = len(
                    tok(text=raw_texts[0], add_special_tokens=False).input_ids
                )
                if im_end_idx >= 0 and n_text_tokens > 0:
                    user_opener_len = im_end_idx - self._drop_idx - n_text_tokens
            except Exception as e:  # metadata only — never fail the encode
                logger.debug(f"qwen_image21: user_opener_len derivation skipped: {e}")

        return {
            "prompt_embed": prompt_embeds[0].to(dtype),
            "prompt_embeds_mask": encoder_attention_mask[0].to(torch.bool),
            "image_token_mask": slot_mask,
            "img_mask": slot_mask,
            "user_opener_len": user_opener_len,
        }

    # ── Latent packing / training hooks ────────────────────────────────

    def prepare_model_input(
        self, batch: dict, noise: torch.Tensor | list[torch.Tensor], sigmas: torch.Tensor
    ) -> dict:
        """Assemble the joint-sequence forward kwargs (transformer contract).

        Single-target only: Qwen-Image 2.1 defines exactly ONE noisy target
        image per sequence — the LAST image block (build_token_metadata
        hard-marks only the last block via
        image_positions[-block_lengths[-1]:]).
        """
        if isinstance(noise, list):
            if len(noise) != 1:
                raise NotImplementedError(
                    f"qwen_image21 supports exactly one noisy target image per sequence "
                    f"(the last image block), got {len(noise)} targets. Configure a "
                    "single-target batch_config."
                )
            target = noise[0]
        else:
            target = noise

        B, C, H, W = target.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                f"qwen_image21 target latent dims must be even (2x2 latent-token "
                f"grouping), got H={H}, W={W}. Check bucket_divisibility=32 upstream."
            )
        device, dtype = target.device, target.dtype

        # Plain spatial pack — NO patchify (pipeline _pack_latents 410-412).
        target_seq = target.view(B, C, H * W).transpose(1, 2)  # (B, H*W, 64)

        encoder_hidden_states = self._extract_encoder_hidden_states(batch, device, dtype)
        if encoder_hidden_states is None:
            raise ValueError(
                "Qwen-Image 2.1 requires encoder_hidden_states in the batch. "
                "Ensure the data pipeline provides pre-computed text embeddings "
                "(cached via encode_text)."
            )
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                f"Qwen-Image 2.1 expects encoder_hidden_states of shape "
                f"(B, seq_len, dim), got shape {tuple(encoder_hidden_states.shape)} "
                f"with ndim={encoder_hidden_states.ndim}"
            )
        text_len = encoder_hidden_states.shape[1]
        encoder_attention_mask = self._extract_encoder_attention_mask(batch, device)
        # (B, text_len) bool — True at VLM image SLOTS (pre-expansion).
        img_mask = self._extract_image_token_mask(batch, device, text_len)

        # ── Reference (edit) path — driven by the cached slot mask ────────
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        ref_key = resolved_bc.get("reference_config")
        latents = batch.get("latents", {})
        ref = latents.get(ref_key) if ref_key else None
        refs = ref if isinstance(ref, list) else ([ref] if ref is not None else [])

        # ── krea2-style uncond composition: empty text + refs kept ────────
        # _build_uncond_batch swaps in the global text-only empty embedding
        # but keeps batch["latents"] (references included). The transformer
        # writes the latent rows INTO the text stream's image-slot positions
        # — the VLM slot content is overwritten
        # (transformer_qwenimage21.py:950) — so the uncond branch only needs
        # the slot LAYOUT, synthesized here from the reference latents
        # (one VLM slot per 2x2 latent group). Gated on the explicit uncond
        # marker: the conditional path never changes behaviour.
        if (
            batch.get("_uncond_empty_text")
            and refs
            and int(img_mask.sum()) == 0
        ):
            ref_latent_tokens = sum(
                int(r.shape[-2] * r.shape[-1]) for r in refs
            )
            if ref_latent_tokens % _IMG_TOKENS_PER_SLOT != 0:
                raise ValueError(
                    f"qwen_image21 uncond: reference latent tokens "
                    f"({ref_latent_tokens}) are not divisible by "
                    f"{_IMG_TOKENS_PER_SLOT} (one VLM slot per 2x2 latent "
                    f"group); reference latent dims must be even."
                )
            num_synth = ref_latent_tokens // _IMG_TOKENS_PER_SLOT
            # Exact ti2i placement when the empty embedding recorded its
            # user-opener length (encode_text caches it as user_opener_len);
            # otherwise fall back to the stream head.
            insert_at = int(batch.get("_uncond_ref_slot_pos") or 0)
            insert_at = min(max(insert_at, 0), text_len)
            pad_ehs = encoder_hidden_states.new_zeros(
                B, num_synth, encoder_hidden_states.shape[2]
            )
            encoder_hidden_states = torch.cat(
                [
                    encoder_hidden_states[:, :insert_at],
                    pad_ehs,
                    encoder_hidden_states[:, insert_at:],
                ],
                dim=1,
            )
            if encoder_attention_mask is not None:
                pad_m = encoder_attention_mask.new_ones(B, num_synth)
                encoder_attention_mask = torch.cat(
                    [
                        encoder_attention_mask[:, :insert_at],
                        pad_m,
                        encoder_attention_mask[:, insert_at:],
                    ],
                    dim=1,
                )
            img_mask = torch.cat(
                [
                    img_mask[:, :insert_at],
                    img_mask.new_ones(B, num_synth),
                    img_mask[:, insert_at:],
                ],
                dim=1,
            )
            text_len = encoder_hidden_states.shape[1]
            logger.debug(
                f"qwen_image21 uncond: synthesized {num_synth} reference "
                f"slot(s) at text position {insert_at} (empty-text embedding, "
                f"refs kept via latents)."
            )

        slot_counts = img_mask.sum(dim=1).tolist()
        if len(set(slot_counts)) > 1:
            raise ValueError(
                f"qwen_image21: img_mask slot counts differ across the batch "
                f"({slot_counts}). The transformer reads the layout from row 0 "
                "(transformer_qwenimage21.py:938), so samples must share it — "
                "the batch sampler must keep multimodal/text-only embeddings "
                "homogeneous per batch."
            )
        num_cond = int(slot_counts[0]) if slot_counts else 0

        ref_seqs: list[torch.Tensor] = []
        self._ref_grids = []
        ref_tokens = 0
        if num_cond > 0:
            if not refs:
                raise ValueError(
                    f"qwen_image21: cached embedding marks {num_cond} VLM image "
                    f"slot(s) but no reference latents found in batch['latents']"
                    f"[{ref_key!r}]. The batch_config reference_config key must "
                    "match caption reference_list.reference_config so the VAE "
                    "latents are cached under the role the text encoder saw."
                )
            for ref_tensor in refs:
                rb, rc, rh, rw = ref_tensor.shape
                if rh % 2 != 0 or rw % 2 != 0:
                    raise ValueError(
                        f"qwen_image21 reference latent dims must be even, got "
                        f"H={rh}, W={rw} (shape {tuple(ref_tensor.shape)})."
                    )
                self._ref_grids.append((rh, rw))
                ref_seqs.append(ref_tensor.reshape(rb, rc, rh * rw).transpose(1, 2))
                ref_tokens += rh * rw
            if ref_tokens != _IMG_TOKENS_PER_SLOT * num_cond:
                raise ValueError(
                    f"qwen_image21: reference latent tokens ({ref_tokens}) != "
                    f"{_IMG_TOKENS_PER_SLOT} * VLM slots ({num_cond}) = "
                    f"{_IMG_TOKENS_PER_SLOT * num_cond}. Reference images fed to "
                    "the text encoder and the VAE must share dims — configure "
                    "caption reference_list.resize == image resolution so both "
                    "see identically-sized images."
                )
            # Conditions first, target last (transformer img_shapes contract;
            # build_token_metadata marks the LAST block as the target).
            hidden_states = torch.cat([*ref_seqs, target_seq], dim=1)
            img_shapes_entry = [(1, rh, rw) for rh, rw in self._ref_grids] + [(1, H, W)]
        else:
            hidden_states = target_seq
            img_shapes_entry = [(1, H, W)]

        # Slots appended for the TARGET image: one VLM slot per 2x2 group of
        # target latent tokens. Pipeline parity — append_target_slots cats
        # latents.shape[1] // 4 (target noise only; pipeline 740-743), and
        # the transformer builds its joint zeros from
        # math.prod(img_shapes[0][-1]) // 4, also target-only (transformer
        # 914-918). NOTE: the PM spec's (ref_tokens + H*W) // 4 equals
        # num_cond_slots + H*W//4 and would DOUBLE-COUNT the reference slots
        # (each ref slot is already True in img_mask) — build_token_metadata
        # would reject the sequence (sum(block_lengths) != image positions).
        # Identical to the spec formula in the T2I case (ref_tokens == 0).
        target_slots = (H * W) // _IMG_TOKENS_PER_SLOT
        img_mask_full = torch.cat(
            [
                img_mask,
                torch.ones(B, target_slots, dtype=torch.bool, device=device),
            ],
            dim=1,
        )

        # Per-sample lists of (frame, h, w) TUPLES — the rope/prod code
        # iterates and unpacks them; plain ints from a list-of-lists would
        # crash. forward reads img_shapes[0] (samples share the layout).
        img_shapes = [list(img_shapes_entry) for _ in range(B)]

        # Cache grids for unpack_prediction (krea2-style, no sqrt guesses).
        self._target_grids = [(H, W)]

        # Timestep in [0, 1] — the transformer multiplies by 1000
        # internally (QwenImage21TemporalTimesteps; pipeline passes
        # t / 1000).
        timesteps = sigmas.to(device=device, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_attention_mask,
            "img_shapes": img_shapes,
            "img_mask": img_mask_full,
            "return_dict": False,
        }

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids=None
    ) -> torch.Tensor | list[torch.Tensor]:
        # Handle Transformer2DModelOutput or tuple (when return_dict=False).
        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample
        elif isinstance(model_pred, (tuple, list)):
            model_pred = model_pred[0]

        if model_pred.dim() != 3:
            return model_pred

        if not self._target_grids:
            raise ValueError(
                f"_target_grids is empty — cannot unpack prediction of "
                f"seq_len={model_pred.shape[1]}. Call prepare_model_input before "
                "unpack_prediction."
            )

        B, seq_len, channels = model_pred.shape
        th, tw = self._target_grids[0]
        target_tokens = th * tw
        if seq_len < target_tokens:
            raise ValueError(
                f"Prediction seq_len={seq_len} shorter than target tokens "
                f"{target_tokens} (grid {th}x{tw}) — wrong model input?"
            )

        # Prediction for the target image = LAST target_tokens rows of the
        # joint output (pipeline parity: noise_pred[:, -latents.size(1):]).
        pred_target = model_pred[:, -target_tokens:, :]
        # Inverse of the plain spatial pack: (B, H*W, 64) -> (B, 64, H, W).
        latents = pred_target.transpose(1, 2).reshape(B, channels, th, tw)
        return [latents]

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        # Standard velocity parameterization: v = noise - x0 (velocity_sign
        # "standard" — inherited; compute_x0_hat inherited accordingly).
        return noise - learning_target

    # ── Timestep sampling ──────────────────────────────────────────────

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample flow-matching sigmas, honoring timestep_shift_mode.

        - "dynamic_shift" (DEFAULT, pipeline parity): logit-normal with mu
          linearly interpolated 0.5 -> 0.9 over 256 -> 8192 image tokens
          (the model's own scheduler_config.json values, so training and
          inference share one schedule).
          Image token count = latent_height * latent_width (patch_size=1 —
          latents are consumed unpatched, every latent token IS an image
          token). Falls back to the base logit-normal (mu=0) when the latent
          dims are unavailable.
        - "flow_shift": fixed-s logit-normal, s = flow_shift (krea2 recipe).
        - "sigma": uniform sigma via the shared base helper (musubi recipe).

        Returns (sigmas * 1000, sigmas): timesteps for scheduler
        bookkeeping, sigmas in [0, 1] for the noise interpolation and the
        transformer forward (which rescales by 1/1000 internally).
        """
        n = QWEN21_SCHEDULER_CONFIG["num_train_timesteps"]

        if self.timestep_shift_mode == "sigma":
            return sample_sigma_uniform(batch_size, device, dtype, num_timesteps=n)

        if self.timestep_shift_mode == "flow_shift":
            mu = math.log(self.flow_shift)
            u = torch.normal(mean=mu, std=self.sigmoid_scale, size=(batch_size,), device=device)
            sigmas = torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5).to(dtype=dtype)
            return sigmas * n, sigmas

        # dynamic_shift
        if latent_height is None or latent_width is None:
            # Base logit-normal (mu=0) when no shape info is available.
            # Implemented inline: BaseModelAdapter.sample_timesteps rejects any
            # timestep_shift_mode other than "sigma", so super() would raise.
            u = torch.randn(batch_size, device=device, dtype=torch.float32)
            sigmas = torch.sigmoid(u).to(dtype=dtype)
            return sigmas * n, sigmas

        cfg = QWEN21_SCHEDULER_CONFIG
        m = (cfg["max_shift"] - cfg["base_shift"]) / (
            cfg["max_image_seq_len"] - cfg["base_image_seq_len"]
        )
        b = cfg["base_shift"] - m * cfg["base_image_seq_len"]
        mu = (latent_height * latent_width) * m + b  # = calculate_shift(image_seq_len)

        # Logit-normal: u ~ N(mu, sigmoid_scale), sigma = sigmoid(u).
        u = torch.normal(mean=mu, std=self.sigmoid_scale, size=(batch_size,), device=device)
        sigmas = torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5).to(dtype=dtype)
        return sigmas * n, sigmas

    # ── Helpers ────────────────────────────────────────────────────────

    def _extract_encoder_hidden_states(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Extract and stack per-sample prompt_embed to (B, L, 4096).

        Cached embeddings hold prompt_embed of shape (seq_len, dim);
        captions vary in seq_len, so all rows are padded to the batch max
        (qwen_image helper convention).
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            return None

        prompt_embeds = []
        for emb in embeddings:
            if emb is None:
                return None
            pe = emb["prompt_embed"] if isinstance(emb, dict) else emb
            if isinstance(pe, np.ndarray):
                pe = torch.from_numpy(pe)
            # Cast to target dtype immediately — avoids float32 intermediates
            # accumulating in the padded list.
            pe = pe.to(device=device, dtype=dtype)
            prompt_embeds.append(pe)

        if not prompt_embeds:
            return None

        max_len = max(pe.shape[0] for pe in prompt_embeds)
        padded = []
        for pe in prompt_embeds:
            if pe.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - pe.shape[0], *pe.shape[1:], dtype=dtype, device=device
                )
                pe = torch.cat([pe, pad], dim=0)
            padded.append(pe)

        return torch.stack(padded)

    def _extract_encoder_attention_mask(
        self, batch: dict, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Extract and stack prompt_embeds_mask to (B, L) bool.

        Pads masks to the batch-max sequence length with False so padded
        token slots are ignored (cached masks are all-TRUE rows after the
        pipeline-parity masked extraction; the False tail is pure padding).
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            return None

        masks = []
        for emb in embeddings:
            if emb is None:
                return None
            mask = emb.get("prompt_embeds_mask") if isinstance(emb, dict) else None
            if mask is None:
                # No mask cached — derive an all-ones row from prompt_embed.
                pe = emb.get("prompt_embed") if isinstance(emb, dict) else emb
                if isinstance(pe, np.ndarray):
                    pe = torch.from_numpy(pe)
                mask = torch.ones(pe.shape[0], dtype=torch.bool)
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            masks.append(mask.to(torch.bool))

        if not masks:
            return None

        max_len = max(m.shape[0] for m in masks)
        padded = []
        for m in masks:
            if m.shape[0] < max_len:
                pad = torch.zeros(max_len - m.shape[0], dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded).to(device=device)

    def _extract_image_token_mask(
        self, batch: dict, device: torch.device, seq_len: int
    ) -> torch.Tensor:
        """Extract the VLM image-slot mask (B, seq_len) from cached embeddings.

        Reads img_mask first (the npz-surviving dual key — cache_builder pops
        image_token_mask before saving), falls back to image_token_mask
        (in-memory dicts), then all-False (text-only / caption-dropout empty
        embedding). Padded positions are False.
        """
        embeddings = batch.get("embeddings")
        masks: list[torch.Tensor] = []
        for emb in embeddings or []:
            mask = None
            if isinstance(emb, dict):
                mask = emb.get("img_mask", emb.get("image_token_mask"))
            if mask is None:
                mask = torch.zeros(seq_len, dtype=torch.bool)
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(torch.bool).to(device=device)
            if mask.shape[0] < seq_len:
                pad = torch.zeros(seq_len - mask.shape[0], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=0)
            elif mask.shape[0] > seq_len:
                raise ValueError(
                    f"qwen_image21: img_mask length {mask.shape[0]} exceeds padded "
                    f"prompt_embed length {seq_len} — embedding cache is "
                    "inconsistent (mismatched npz for this caption?)."
                )
            masks.append(mask)
        if not masks:
            return torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        return torch.stack(masks)
