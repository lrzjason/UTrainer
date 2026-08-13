"""Krea-2 model adapter — MMDiT with optional reference image conditioning.

The Krea-2 Raw model is a single-stream MMDiT that uses:
- Qwen3VL (4B) as the text encoder (multi-layer hidden state taps)
- AutoencoderKLQwenImage VAE (16 latent channels, shift+scaling)
- Patchified latents (patch_size=2, so in_channels = 16 * 4 = 64)
- Explicit (t, h, w) rotary position IDs for the combined text+image sequence

Reference image conditioning uses a **RoPE frame-index** design
(matching ai-toolkit / Flux ``index_timestep_zero`` convention):
1. Reference latents are packed and appended as extra sequence tokens
   (not channel-concatenated — no weight expansion needed).
2. References receive **positive** incrementing frame indices (1, 2, …) on
   RoPE axis 0; generation targets start at frame 0.
3. Per-span timestep modulation: reference tokens get t=0 modulation
   across all transformer blocks; noisy target tokens get the training timestep.
4. Qwen3VL multimodal conditioning — pass reference images to the text encoder
   for text+image conditioned hidden states (handled in ``encode_text``).
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

RESOLUTION_CONFIG = {
    512: [
        (512, 512),
        (576, 448), (640, 384), (704, 320), (768, 288),
        (832, 256), (896, 224), (960, 192),
        (448, 576), (384, 640), (320, 704), (288, 768),
        (256, 832), (224, 896), (192, 960),
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

# Default layer taps from the Qwen3-VL-4B text encoder (12 layers, matching
# ``num_text_layers=12`` in the transformer config).
DEFAULT_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Krea-2 chat-template constants — matches the ComfyUI ``KREA2_TEMPLATE``
# and Qwen-Image template exactly (same system instruction, user-opening,
# and assistant suffix).  The full template is:
#   "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
#
# At encoding time the system+user preamble is stripped from the hidden-states
# output (matching ComfyUI's ``encode_token_weights`` prefix stripping), so the
# DiT receives only the user-content tokens plus any suffix tokens the text
# encoder retains.
_PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_TEMPLATE_START_IDX = 34  # token count of _PROMPT_TEMPLATE_PREFIX
_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS: int = 5  # "<|im_end|>\n<|im_start|>assistant\n"
_DEFAULT_MAX_SEQ_LEN = 512
_IMG_PLACEHOLDER = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


@ModelRegistry.register("krea2")
class Krea2Adapter(BaseModelAdapter):
    """Adapter for Krea-2 Raw MMDiT model with reference image support."""

    name = "krea2"
    patch_size = 2

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

        # Reference conditioning via VAE latent concatenation.
        self.reference_conditioning: bool = config.get("reference_conditioning", False)

        # Multimodal reference: pass reference image to Qwen3VL text encoder.
        self.multimodal_reference: bool = config.get("multimodal_reference", False)



        # Text encoder layer taps.
        self.text_encoder_select_layers: tuple = tuple(
            config.get("text_encoder_select_layers", DEFAULT_TEXT_ENCODER_SELECT_LAYERS)
        )

        # Suffix embedding — cached globally (like empty embedding) and appended
        # to per-sample encoder_hidden_states at training time.  Set by the
        # trainer after loading from cache; None = suffix not included.
        self.suffix_embed: Optional[torch.Tensor] = None
        self.suffix_mask: Optional[torch.Tensor] = None

        # Timestep shift mode for training-time sigma sampling:
        #   "sigma"            — uniform σ ∈ [0.001, 1] (musubi timestep_sampling=
        #                        sigma recipe: u ~ U[0,1), t = floor(u*1000)+1,
        #                        σ = t/1000). DEFAULT — musubi-aligned.
        #   "comfy_fixed"      — logit-normal μ=1.15 at every resolution, exactly
        #                        matching ComfyUI inference (supported_models.py
        #                        Krea2 sampling_settings shift=1.15).
        #   "pretrain_dynamic" — logit-normal, mu linearly interpolated 0.5→1.15
        #                        over 256→6400 image tokens, matching the model's
        #                        pretraining distribution (ai-toolkit /
        #                        T2ITrainer convention).
        # All cover sigma ∈ (0,1]; only the sampling density differs.
        self.timestep_shift_mode: str = config.get("timestep_shift_mode", "sigma")
        if self.timestep_shift_mode not in ("sigma", "comfy_fixed", "pretrain_dynamic"):
            raise ValueError(
                f"Unknown timestep_shift_mode '{self.timestep_shift_mode}'. "
                "Expected 'sigma', 'comfy_fixed' or 'pretrain_dynamic'."
            )

        # Target grid dimensions cached from prepare_model_input — used by unpack_prediction.
        # Single-target: list with one (gh, gw) pair.  Multi-target: one pair per target.
        self._target_grids: list[tuple[int, int]] = []

    # ── Model loading ──────────────────────────────────────────────────

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from .transformer_krea2 import BlockSwapKrea2Transformer2DModel
        model = BlockSwapKrea2Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

        # Reference conditioning via RoPE frame-index design: reference tokens
        # enter as extra sequence tokens with negative frame indices (no weight
        # expansion needed — img_in always receives in_channels=64).
        return model

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import AutoencoderKLQwenImage
        return AutoencoderKLQwenImage.from_pretrained(path, torch_dtype=dtype)

    # Krea2 mu-shift flow-matching schedule config.
    # Matches ComfyUI's ModelSamplingFlux which uses fixed shift=1.15
    # (see comfy/supported_models.py Krea2 sampling_settings).
    # Both base_shift and max_shift are set to 1.15 so _krea2_calculate_shift
    # always returns 1.15 regardless of resolution — exactly matching ComfyUI.
    # (When config timestep_shift_mode="pretrain_dynamic", sample_timesteps
    # instead interpolates mu 0.5→1.15 over base/max_image_seq_len, matching
    # the model's pretraining distribution. With "sigma", sample_timesteps
    # samples uniform σ∈[0.001,1] directly and ignores the shift entirely.)
    KREA2_SCHEDULER_CONFIG = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 6400,
        "base_shift": 1.15,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
    }

    @staticmethod
    def _krea2_calculate_shift(image_seq_len: int) -> float:
        """Linear interpolation of shift mu between base and max seq lengths."""
        cfg = Krea2Adapter.KREA2_SCHEDULER_CONFIG
        m = (cfg["max_shift"] - cfg["base_shift"]) / (
            cfg["max_image_seq_len"] - cfg["base_image_seq_len"]
        )
        b = cfg["base_shift"] - m * cfg["base_image_seq_len"]
        return image_seq_len * m + b

    @staticmethod
    def _krea2_time_shift(mu: float, t: float) -> float:
        """Exponential time-shift used by Krea2 dynamic shifting."""
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))

    def load_scheduler(self, path: str) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.KREA2_SCHEDULER_CONFIG["num_train_timesteps"],
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=self.KREA2_SCHEDULER_CONFIG["base_shift"],
            max_shift=self.KREA2_SCHEDULER_CONFIG["max_shift"],
            base_image_seq_len=self.KREA2_SCHEDULER_CONFIG["base_image_seq_len"],
            max_image_seq_len=self.KREA2_SCHEDULER_CONFIG["max_image_seq_len"],
        )

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoConfig
            config = AutoConfig.from_pretrained(path)
            # Patch rope_scaling if missing -transformers has a bug where it
            # calls config.rope_scaling.get() without null-checking.
            # The rope_scaling lives on the text_config sub-config.
            mrope = {"rope_type": "default", "mrope_section": [24, 20, 20]}
            if hasattr(config, "text_config"):
                if getattr(config.text_config, "rope_scaling", None) is None:
                    config.text_config.rope_scaling = mrope
            elif getattr(config, "rope_scaling", None) is None:
                config.rope_scaling = mrope
            return Qwen3VLForConditionalGeneration.from_pretrained(path, config=config, torch_dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to load Krea2 text encoder: {e}")
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        """Load the Qwen2 tokenizer.

        The Krea-2-Raw tokenizer_config.json has ``extra_special_tokens`` as a
        list, which causes ``AutoTokenizer.from_pretrained`` to fail with
        ``'list' object has no attribute 'keys'``.  We patch the config by
        renaming it to ``additional_special_tokens`` (the standard field) and
        loading via ``Qwen2TokenizerFast``.
        """
        import json
        import os
        import tempfile
        import shutil

        # First, try the standard loading path
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(path)
        except Exception:
            pass

        # Fallback: patch tokenizer_config.json and load Qwen2TokenizerFast
        try:
            from transformers import Qwen2TokenizerFast

            config_path = os.path.join(path, "tokenizer_config.json")
            if not os.path.exists(config_path):
                return None

            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)

            # Rename extra_special_tokens -additional_special_tokens
            extra = cfg.pop("extra_special_tokens", [])
            if extra:
                existing = cfg.get("additional_special_tokens", [])
                cfg["additional_special_tokens"] = list(existing) + list(extra)

            # Write patched config to a temp directory
            tmp_dir = tempfile.mkdtemp(prefix="krea2_tok_")
            for f in os.listdir(path):
                shutil.copy2(os.path.join(path, f), os.path.join(tmp_dir, f))
            with open(os.path.join(tmp_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            tokenizer = Qwen2TokenizerFast.from_pretrained(tmp_dir)
            shutil.rmtree(tmp_dir)
            logger.info(f"Loaded Qwen2TokenizerFast (patched config) from {path}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {path}: {e}")
            # Last resort: PreTrainedTokenizerFast (won't work with Qwen3VLProcessor)
            try:
                from transformers import PreTrainedTokenizerFast
                return PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
            except Exception:
                return None

    def load_processor(self, path: str, tokenizer_path: str = "") -> Optional[Any]:
        """Load the Qwen3VL processor for multimodal text+image encoding.

        The Krea-2-Raw model ships without a ``preprocessor_config.json``, so
        ``AutoProcessor.from_pretrained`` fails.  We construct the processor
        manually from a ``Qwen2VLImageProcessor``, a ``Qwen3VLVideoProcessor``,
        and the tokenizer loaded from ``tokenizer_path`` (falls back to ``path``).
        """
        # First, try the standard loading path
        try:
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(path)
        except Exception:
            pass

        # Fallback: construct Qwen3VLProcessor manually
        tok_path = tokenizer_path or path
        try:
            from transformers import Qwen3VLProcessor
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
            import json
            import os

            # Read vision config to get patch_size, temporal_patch_size, merge_size
            # The Qwen3VL vision model expects patch_size=16 (not Qwen2VL's default 14).
            patch_size = 16
            temporal_patch_size = 2
            merge_size = 2
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    model_config = json.load(f)
                vision_config = model_config.get("vision_config", {})
                patch_size = vision_config.get("patch_size", patch_size)
                temporal_patch_size = vision_config.get("temporal_patch_size", temporal_patch_size)
                merge_size = vision_config.get("spatial_merge_size", merge_size)

            image_processor = Qwen2VLImageProcessor(
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            video_processor = Qwen3VLVideoProcessor()

            # Load tokenizer from tokenizer_path (not text_encoder path)
            tokenizer = self.load_tokenizer(tok_path)
            if tokenizer is None:
                logger.warning(f"Cannot construct processor: tokenizer is None from {tok_path}")
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
        except Exception as e:
            logger.warning(f"Failed to construct Qwen3VLProcessor from {path}: {e}")
            return None

    # ── Architecture specs ─────────────────────────────────────────────

    @property
    def latent_channels(self) -> int:
        return 16

    @property
    def vae_scale_factor(self) -> int:
        return 8

    @property
    def embedding_dim(self) -> int:
        return 2560

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return True

    # T2ITrainer hardcoded VAE normalization constants.
    # These are the values used during Krea2 training — they differ from
    # vae.config.latents_mean/latents_std because T2ITrainer calibrates on
    # training data, not the generic VAE calibration set.
    _T2I_LATENTS_MEAN: list[float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    _T2I_LATENTS_STD: list[float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ]

    # ── Encoding ───────────────────────────────────────────────────────

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            # AutoencoderKLQwenImage is a video VAE expecting 5D input (B, C, T, H, W).
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(2)
            latent = vae.encode(image_tensor).latent_dist.sample()
            # Squeeze temporal dim back to 4D (B, C, H, W)
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            # Use T2ITrainer-calibrated normalization constants (matching training).
            latents_mean = torch.tensor(
                Krea2Adapter._T2I_LATENTS_MEAN, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latents_std = torch.tensor(
                Krea2Adapter._T2I_LATENTS_STD, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latent = (latent - latents_mean) / latents_std
        return {"latent": latent}

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
        """Encode a text prompt (and optionally reference images) via Qwen3VL.

        When *reference_image* and *processor* are both provided the reference
        images are embedded as ``Picture N:`` placeholders in the user message
        and forwarded to the Qwen3VL vision encoder.  The text-only path uses
        the standard Krea-2 chat template.

        Returns
        -------
        dict
            ``prompt_embed``        — ``(seq_len, num_text_layers, dim)``
            ``prompt_embeds_mask``  — ``(seq_len,)`` bool
            ``image_token_mask``    — ``(seq_len,)`` bool  *(multimodal only)*
        """
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "Krea2 requires a loaded Qwen3VL text encoder and tokenizer."
            )

        is_multimodal = reference_image is not None and processor is not None

        # ── 1. Build prompt text (matches ComfyUI KREA2_TEMPLATE) ─────────
        # Full chat template: system + user_preamble + {content} + suffix.
        # After encoding the system+user preamble is stripped (see step 4),
        # mirroring ComfyUI's ``encode_token_weights`` prefix stripping.
        #
        # NOTE: Qwen3VLForConditionalGeneration (HuggingFace) drops the
        # 5-token suffix from hidden_states internally; ComfyUI's custom
        # Qwen3VL implementation does NOT.  The defensive length-clamping
        # below handles both cases transparently.

        prompt_text = _PROMPT_TEMPLATE_PREFIX  # shared system + user preamble
        refs: list[Any] = []

        if is_multimodal:
            refs = reference_image if isinstance(reference_image, list) else [reference_image]
            prompt_text += "".join(
                _IMG_PLACEHOLDER.format(i + 1) for i in range(len(refs))
            )
        prompt_text += prompt + _PROMPT_TEMPLATE_SUFFIX

        # ── 2. Tokenize ────────────────────────────────────────────────────

        proc_kwargs: dict[str, Any] = dict(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
        )

        if is_multimodal:
            proc_kwargs.update(images=refs, do_rescale=False)
        else:
            # T2ITrainer parity: do NOT pad to max_length.
            proc_kwargs.update(
                truncation=True,
                max_length=(
                    _DEFAULT_MAX_SEQ_LEN
                    + _PROMPT_TEMPLATE_START_IDX
                    + _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS
                ),
            )

        main_inputs = processor(**proc_kwargs).to(device)
        input_ids = main_inputs.input_ids
        attention_mask = main_inputs.attention_mask.bool()

        # ── 2b. Trim suffix tokens before Qwen3VL forward ──────────────────
        # The suffix ("<|im_end|>\n<|im_start|>assistant\n" — 5 tokens) is
        # included in the prompt text for structural alignment with ComfyUI's
        # KREA2_TEMPLATE, but MUST be stripped before the Qwen3VL forward
        # pass.  Qwen3VLForConditionalGeneration drops suffix tokens from
        # hidden_states internally but references the original (longer)
        # attention_mask inside its forward(), causing a
        # "mask [N] does not match tensor [N-5]" shape error.
        # Pre-trimming avoids this crash — Qwen3VL sees only the prefix +
        # user-content span, and the output hidden-states length matches the
        # system-preamble stripping below.
        valid_lens = attention_mask.sum(dim=1)  # non-pad token count per item
        for b in range(input_ids.shape[0]):
            vl = int(valid_lens[b].item())
            if vl >= _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS:
                attention_mask[b, vl - _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS:vl] = False
        # Also trim input_ids to the max valid length (minus suffix) so
        # Qwen3VL's internal sequence-length tracking matches the mask.
        new_max_len = int(valid_lens.max().item()) - _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS
        input_ids = input_ids[:, :new_max_len]
        attention_mask = attention_mask[:, :new_max_len]
        # Trim multimodal metadata if present (same length as input_ids).
        if hasattr(main_inputs, "mm_token_type_ids") and main_inputs.mm_token_type_ids is not None:
            main_inputs.mm_token_type_ids = main_inputs.mm_token_type_ids[:, :new_max_len]

        # ── 3. Assemble encoder kwargs ─────────────────────────────────────

        encoder_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if is_multimodal:
            _pixel = getattr(main_inputs, "pixel_values", None)
            if _pixel is not None:
                encoder_kwargs["pixel_values"] = _pixel.to(dtype=text_encoder.dtype)
            _grid = getattr(main_inputs, "image_grid_thw", None)
            if _grid is not None:
                encoder_kwargs["image_grid_thw"] = _grid
            _mm = getattr(main_inputs, "mm_token_type_ids", None)
            if _mm is not None:
                encoder_kwargs["mm_token_type_ids"] = _mm
        else:
            # mRoPE position IDs — cumulative valid-token count over 3 axes
            pos_ids = attention_mask.long().cumsum(dim=-1) - 1
            encoder_kwargs["position_ids"] = (
                pos_ids.clamp(min=0).unsqueeze(0).expand(3, -1, -1)
            )

        # ── 4. Run encoder & stack hidden states ───────────────────────────

        with torch.no_grad():
            outputs = text_encoder(**encoder_kwargs)

        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in self.text_encoder_select_layers],
            dim=2,
        )[:, _PROMPT_TEMPLATE_START_IDX:]  # strip system+user preamble (matches ComfyUI)

        # Clamp masks to actual hidden-states length (defensive — handles
        # any residual suffix trimming by the model).
        valid_len = hidden_states.shape[1]
        attention_mask = attention_mask[:, _PROMPT_TEMPLATE_START_IDX:_PROMPT_TEMPLATE_START_IDX + valid_len]

        # ── 5. Build result ────────────────────────────────────────────────

        result: dict[str, Any] = dict(
            prompt_embed=hidden_states[0].to(dtype),
            prompt_embeds_mask=attention_mask[0].to(torch.bool),
        )

        if is_multimodal:
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            result["image_token_mask"] = (
                (input_ids[0, _PROMPT_TEMPLATE_START_IDX:_PROMPT_TEMPLATE_START_IDX + valid_len] == image_pad_id)
                if image_pad_id is not None
                else torch.zeros(valid_len, dtype=torch.bool)
            )

        return result

    @staticmethod
    def encode_suffix_embedding(
        text_encoder: nn.Module,
        tokenizer: Any,
        processor: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        """Encode the 5 chat-template suffix tokens once for global caching.

        The suffix ("<|im_end|>\\n<|im_start|>assistant\\n") is encoded with
        the system preamble as context, matching how it would appear in a
        full template.  The result is cached to ``suffix_embedding.npkrea2``
        and appended to per-sample encoder_hidden_states at training time.

        This is a text-only path — no images are involved.
        """
        # Build: system preamble + suffix (no user content).
        prompt_text = _PROMPT_TEMPLATE_PREFIX + _PROMPT_TEMPLATE_SUFFIX

        proc_kwargs: dict[str, Any] = dict(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=_PROMPT_TEMPLATE_START_IDX + _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
        )
        main_inputs = processor(**proc_kwargs).to(device)
        input_ids = main_inputs.input_ids
        attention_mask = main_inputs.attention_mask.bool()

        # mRoPE position IDs
        pos_ids = attention_mask.long().cumsum(dim=-1) - 1

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=pos_ids.clamp(min=0).unsqueeze(0).expand(3, -1, -1),
                output_hidden_states=True,
            )

        # Stack selected layer taps, then strip the system preamble.
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in DEFAULT_TEXT_ENCODER_SELECT_LAYERS],
            dim=2,
        )[:, _PROMPT_TEMPLATE_START_IDX:]

        return {
            "prompt_embed": hidden_states[0].to(dtype),
            "prompt_embeds_mask": attention_mask[0, _PROMPT_TEMPLATE_START_IDX:].to(torch.bool),
        }

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        # Denormalize using T2ITrainer-calibrated constants (inverse of encode_image).
        latents_mean = torch.tensor(
            Krea2Adapter._T2I_LATENTS_MEAN, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latents_std = torch.tensor(
            Krea2Adapter._T2I_LATENTS_STD, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latent = latent * latents_std + latents_mean
        # AutoencoderKLQwenImage expects 5D input (B, C, T, H, W).
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        with torch.no_grad():
            image = vae.decode(latent).sample
        # Squeeze temporal dim back to 4D
        if image.ndim == 5:
            image = image.squeeze(2)
        return image

    # ── Latent packing ─────────────────────────────────────────────────

    def _pack_latents(
        self, latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int
    ) -> torch.Tensor:
        """Pack (B, C, H, W) latents into (B, (H//p)*(W//p), C*p*p) patchified sequence."""
        p = self.patch_size
        latents = latents.view(batch_size, num_channels, height // p, p, width // p, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // p) * (width // p), num_channels * p * p
        )
        return latents


    # ── Position IDs ───────────────────────────────────────────────────

    @staticmethod
    def _build_grid_positions(
        grid_height: int, grid_width: int, frame: float, device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build ``(grid_h * grid_w, 3)`` rotary coordinates for a single image grid.

        Args:
            grid_height: Latent grid height (H // patch_size).
            grid_width:  Latent grid width  (W // patch_size).
            frame:      Value for RoPE axis 0 (0 for targets, positive for refs).
            device:     Target device.
            dtype:      Tensor dtype (should match model compute dtype).

        Returns:
            ``(grid_h * grid_w, 3)`` tensor of ``(frame, y, x)`` positions.
        """
        pos = torch.zeros(grid_height, grid_width, 3, device=device, dtype=dtype)
        pos[..., 0] = frame
        pos[..., 1] = torch.arange(grid_height, device=device, dtype=dtype)[:, None]
        pos[..., 2] = torch.arange(grid_width, device=device, dtype=dtype)[None, :]
        return pos.reshape(grid_height * grid_width, 3)

    @staticmethod
    def prepare_position_ids(
        text_seq_len: int,
        device: torch.device,
        target_grids: list[tuple[int, int]],
        ref_grids: list[tuple[int, int]] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build rotary coordinates for the full ``[text, target_imgs, ref_imgs]`` sequence.

        Text tokens sit at ``(0, 0, 0)``.
        Target image tokens use ``(frame, y, x)`` with ``frame = 0, 1, …``
        (one per entry in ``target_grids``).
        Reference image tokens use **positive** incrementing frame indices
        ``(1, 2, …)`` (one per entry in ``ref_grids``), matching the ai-toolkit
        / Flux ``index_timestep_zero`` convention.
        Each grid may have a different ``(grid_h, grid_w)``.

        Args:
            text_seq_len: Number of text tokens.
            device:       Target device.
            target_grids: List of ``(grid_h, grid_w)`` for each target image.
            ref_grids:    List of ``(grid_h, grid_w)`` for each reference image.
                          ``None`` or empty list = no references.
            dtype:        Tensor dtype (should match model compute dtype).

        Returns:
            ``(total_seq_len, 3)`` position tensor — always the full sequence.
        """
        text_ids = torch.zeros(text_seq_len, 3, device=device, dtype=dtype)

        target_parts = [
            Krea2Adapter._build_grid_positions(gh, gw, float(f), device, dtype)
            for f, (gh, gw) in enumerate(target_grids)
        ]
        target_ids = torch.cat(target_parts, dim=0)

        if ref_grids:
            ref_parts = [
                Krea2Adapter._build_grid_positions(rh, rw, float(i + 1), device, dtype)
                for i, (rh, rw) in enumerate(ref_grids)
            ]
            ref_ids = torch.cat(ref_parts, dim=0)
            return torch.cat([text_ids, target_ids, ref_ids], dim=0)

        return torch.cat([text_ids, target_ids], dim=0)

    # ── Training hooks ─────────────────────────────────────────────────

    def prepare_model_input(
        self,
        batch: dict,
        noise: torch.Tensor | list[torch.Tensor],
        sigmas: torch.Tensor,
    ) -> dict:
        latents = batch.get("latents", {})

        # Resolve reference from batch_config — user-defined keys, NOT hardcoded.
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        ref_key = resolved_bc.get("reference_config")
        ref = latents.get(ref_key) if ref_key else None

        p = self.patch_size

        # ── Target (noisy) latents — support single or multiple targets ──
        noises = noise if isinstance(noise, list) else [noise]
        packed_parts: list[torch.Tensor] = []
        target_grids: list[tuple[int, int]] = []
        for n in noises:
            B, C, H, W = n.shape
            gh = H // p
            gw = W // p
            target_grids.append((gh, gw))
            packed_parts.append(self._pack_latents(n, B, C, H, W))

        # Cache target grids for unpack_prediction
        self._target_grids = target_grids



        # ── Reference latents — extra sequence tokens with negative frame indices ──
        refs = ref if isinstance(ref, list) else ([ref] if ref is not None else [])
        ref_grids: list[tuple[int, int]] = []
        reflen = 0
        if refs and self.reference_conditioning:
            for ref_tensor in refs:
                ref_B, ref_C, ref_H, ref_W = ref_tensor.shape
                ref_gh = ref_H // p
                ref_gw = ref_W // p
                ref_grids.append((ref_gh, ref_gw))
                ref_packed = self._pack_latents(ref_tensor, ref_B, ref_C, ref_H, ref_W)
                packed_parts.append(ref_packed)
                reflen += ref_gh * ref_gw

        model_input = torch.cat(packed_parts, dim=1)

        # Extract encoder_hidden_states from cached embeddings in the batch.
        encoder_hidden_states = self._extract_encoder_hidden_states(batch, noises[0].device, noises[0].dtype)
        encoder_attention_mask = self._extract_encoder_attention_mask(batch, noises[0].device)

        if encoder_hidden_states is not None:
            if encoder_hidden_states.ndim != 4:
                raise ValueError(
                    f"Krea2 expects encoder_hidden_states of shape (B, seq_len, num_text_layers, dim), "
                    f"got shape {tuple(encoder_hidden_states.shape)} with ndim={encoder_hidden_states.ndim}"
                )
            text_seq_len = encoder_hidden_states.shape[1]
        else:
            raise ValueError(
                "Krea2 requires encoder_hidden_states in the batch. "
                "Ensure the data pipeline provides pre-computed text encoder hidden states "
                "(cached via encode_text)."
            )

        position_ids = self.prepare_position_ids(
            text_seq_len, noises[0].device,
            target_grids=target_grids,
            ref_grids=ref_grids if ref_grids else None,
            dtype=noises[0].dtype,
        )

        # Krea2 transformer expects timestep in [0, 1] -it scales by 1e3 internally.
        timesteps = sigmas.to(dtype=noises[0].dtype)

        result = {
            "hidden_states": model_input,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "position_ids": position_ids,
            "return_dict": False,
            "reflen": reflen,
        }

        if encoder_attention_mask is not None:
            result["encoder_attention_mask"] = encoder_attention_mask

        return result

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor | list[torch.Tensor]:
        # Handle Transformer2DModelOutput or tuple (when return_dict=False)
        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample
        elif isinstance(model_pred, (tuple, list)):
            model_pred = model_pred[0]

        # Unpack packed prediction: (B, seq_len, C*p*p) -> (B, C, H, W) per target.
        if model_pred.dim() == 3:
            B, seq_len, channels = model_pred.shape
            p = self.patch_size
            C = channels // (p * p)

            # Use cached target grids (set by prepare_model_input).
            # Multi-target: split prediction by grid sizes, unpack each separately.
            if self._target_grids:
                results = []
                offset = 0
                for gh, gw in self._target_grids:
                    chunk_len = gh * gw
                    chunk = model_pred[:, offset : offset + chunk_len, :]
                    latents = chunk.view(B, gh, gw, C, p, p)
                    latents = latents.permute(0, 3, 1, 4, 2, 5)
                    latents = latents.reshape(B, C, 1, gh * p, gw * p)
                    latents = latents.squeeze(2)
                    results.append(latents)
                    offset += chunk_len
                return results

            # No cached grids — shouldn't happen in training (prepare_model_input always
            # runs first), but keep a clear error rather than a wrong sqrt-based guess.
            raise ValueError(
                f"_target_grids is empty — cannot unpack prediction of seq_len={seq_len}. "
                f"Call prepare_model_input before unpack_prediction."
            )

        return model_pred

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target

    # ── Krea2 timestep sampling (uniform-σ / logit-normal modes) ──────

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps for flow-matching, honoring timestep_shift_mode.

        - "sigma": uniform σ ∈ [0.001, 1] — the musubi timestep_sampling=sigma
          recipe (u ~ U[0,1), integer timestep = floor(u*1000)+1 ∈ [1, 1000], σ = t/1000).
          Resolution-independent; DEFAULT.
        - "comfy_fixed": logit-normal σ = sigmoid(N(1.15, 1)) at every resolution.
        - "pretrain_dynamic": logit-normal with mu interpolated 0.5→1.15 over
          256→6400 image tokens (pretraining distribution).

        Returns timesteps = sigma * num_train_timesteps (for the transformer
        to normalize to [0,1] via timestep/1000).
        """
        n = self.KREA2_SCHEDULER_CONFIG["num_train_timesteps"]

        # Uniform σ ∈ [0.001, 1] — musubi-aligned (resolution-independent).
        # Shared helper in models/base.py; other adapters honor the same
        # timestep_shift_mode config key (opt-in there, default here).
        if self.timestep_shift_mode == "sigma":
            return sample_sigma_uniform(
                batch_size,
                device,
                dtype,
                num_timesteps=self.KREA2_SCHEDULER_CONFIG["num_train_timesteps"],
            )

        if latent_height is None or latent_width is None:
            # Fallback to base logit-normal (mu=0) if no shape info available
            return super().sample_timesteps(batch_size, device, dtype)

        # Compute mu from image token count, honoring timestep_shift_mode:
        # comfy_fixed → mu=1.15 always; pretrain_dynamic → 0.5→1.15 by tokens.
        image_seq_len = (latent_height * latent_width) // (self.patch_size ** 2)
        if self.timestep_shift_mode == "pretrain_dynamic":
            cfg = self.KREA2_SCHEDULER_CONFIG
            m = (1.15 - 0.5) / (cfg["max_image_seq_len"] - cfg["base_image_seq_len"])
            b = 0.5 - m * cfg["base_image_seq_len"]
            mu = image_seq_len * m + b
        else:
            mu = self._krea2_calculate_shift(image_seq_len)

        # Logit-normal: u ~ N(mu, 1), sigma = sigmoid(u)
        u = torch.normal(mean=mu, std=1.0, size=(batch_size,), device=device)
        sigmas = torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5).to(dtype=dtype)
        selected_timesteps = sigmas * n

        return selected_timesteps, sigmas

    # ── Helpers ────────────────────────────────────────────────────────

    def _extract_encoder_hidden_states(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Extract and stack encoder_hidden_states from cached embeddings.

        The data pipeline stores per-sample embeddings as dicts with
        ``prompt_embed`` of shape (seq_len, num_text_layers, dim).
        Different captions may have different seq_len, so we pad all
        to the batch-max before stacking.

        Returns:
            (B, max_seq_len, num_text_layers, dim) tensor.
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

            # Append cached suffix embedding (matches ComfyUI output length).
            if self.suffix_embed is not None:
                se = self.suffix_embed.to(device=device, dtype=dtype)
                pe = torch.cat([pe, se], dim=0)

            prompt_embeds.append(pe)

        if not prompt_embeds:
            return None

        # Pad all to max sequence length in the batch
        max_len = max(pe.shape[0] for pe in prompt_embeds)
        padded = []
        for pe in prompt_embeds:
            if pe.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - pe.shape[0], *pe.shape[1:],
                    dtype=dtype, device=device,
                )
                pe = torch.cat([pe, pad], dim=0)
            padded.append(pe)

        return torch.stack(padded)

    def _extract_encoder_attention_mask(
        self, batch: dict, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Extract and stack encoder_attention_mask from cached embeddings.

        Pads masks to batch-max sequence length (False for padding positions)
        so the model ignores padded token slots.
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            return None

        masks = []
        for emb in embeddings:
            if emb is None:
                return None
            if isinstance(emb, dict):
                mask = emb.get("prompt_embeds_mask")
            else:
                mask = None
            if mask is None:
                # If no mask, create all-ones mask
                pe = emb.get("prompt_embed") if isinstance(emb, dict) else emb
                if isinstance(pe, np.ndarray):
                    pe = torch.from_numpy(pe)
                mask = torch.ones(pe.shape[0], dtype=torch.bool)
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            # Append suffix mask if suffix embedding is active.
            if self.suffix_mask is not None:
                sm = self.suffix_mask.to(device=mask.device)
                mask = torch.cat([mask, sm], dim=0)

            masks.append(mask)

        if not masks:
            return None

        # Pad all masks to max length with False
        max_len = max(m.shape[0] for m in masks)
        padded = []
        for m in masks:
            if m.shape[0] < max_len:
                pad = torch.zeros(max_len - m.shape[0], dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded).to(device=device)
