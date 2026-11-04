"""FLUX.3 klein model adapter — per-mode MMDiT with joint single-stream blocks.

Architecture (verified against the diffusers reference checkout
``E:\\diffusers`` @ 7ecb1e1b4 "Flux3 image (#15046)"; the transformer itself is
vendored in ``transformer_flux3.py`` next to this file):

- **Transformer** ``Flux3Transformer2DModel`` — three token *modes* share one
  model: the noisy image tokens (``hidden_states``), the optional clean
  reference-image tokens (``image_cond_hidden_states``) and the text context
  (``encoder_hidden_states``).  Each mode first runs its own per-mode blocks
  (``mode_blocks["image"]``, ``mode_blocks["image_cond"]``, ``txt_mode_blocks``
  — each self-attends *within* its own mode only), then text + content are
  concatenated for the joint ``single_transformer_blocks``; only the image
  tokens reach ``norm_out`` / ``proj_out``.
- **Defaults** for the shipped klein checkpoints: ``in_channels=128``,
  ``cond_in_channels=128``, ``num_layers=5``, ``num_single_layers=28``,
  ``attention_head_dim=128``, ``num_attention_heads=24`` (inner dim 3072),
  ``joint_attention_dim=20480``, ``timestep_channels=256``, ``mlp_ratio=3.0``,
  ``axes_dims_rope=(32,32,32,32)``, ``rope_theta=10000``.
- **Latents**: the FLUX.2 packed autoencoder (``AutoencoderKLFlux2``) — a
  32-channel latent at 8x downscale, which the transformer consumes *patchified
  2x2 and packed* into 128 channels on a ``height/16 x width/16`` grid.  Hence
  ``vae_scale_factor = 8``, ``patch_size = 2`` and ``bucket_divisibility = 16``.
  Unlike FLUX.2 klein there is **no** ``scaling_factor`` / ``shift_factor``:
  normalization is the VAE's BatchNorm statistics,
  ``(x - bn.running_mean) / sqrt(bn.running_var + batch_norm_eps)``, applied
  *after* the 2x2 patchify (reference ``Flux3VaeEncoderStep``).  This adapter
  does that work in ``encode_image``, so the cache stores **transformer-space**
  latents and ``latent_channels`` reports 128.
- **Text encoder**: ``Qwen3VLForConditionalGeneration`` (Qwen3-VL-4B,
  ``hidden_size=2560``).  Eight hidden-state taps ``(4, 8, 12, 16, 20, 24, 28,
  32)`` are stacked along the channel axis →
  ``8 * 2560 = 20480 = joint_attention_dim``.
  The text window is right-padded to a **multiple of 80 tokens** (capped at
  ``max_sequence_length``), and the padding is *attended to* by the transformer
  (the FLUX.3 forward takes no text mask), so the window size is part of the
  model recipe — see ``_DEFAULT_PAD_MULTIPLE``.
- **RoPE**: 4-axis ``(t, h, w, l)`` ids.  Text tokens sit at ``(0, 0, 0, idx)``.
  Image tokens use ``t=0``, ``h``/``w`` scaled by ``image_rope_scale = 0.5``
  (the factor the model was trained with) and an ``l`` coordinate that
  distinguishes the modes: ``l = 0`` for the target image and ``l = -(1 + i)``
  (``-1, -2, ...``) for the i-th reference image.
- **Timesteps**: the transformer scales by 1000 internally, so training sigmas
  in ``[0, 1]`` are passed through unchanged.  The image tokens are modulated
  by the flow timestep while the **reference tokens and the text context sit at
  timestep 0** (handled inside the transformer — the adapter only supplies the
  image-token timestep).  Training sigma sampling uses the fixed-s
  ``flow_shift`` recipe (logit-normal), and the DEFAULT shift is **16**
  (``mu = ln(16) = 2.77259``) — the official workflow's inference alpha, so
  training is centred on the alpha the model is sampled with (mean sigma ~0.92,
  ~72% of draws above 0.9).  Other recipes stay selectable per config:
  ``flow_shift = 6.93`` is FLUX.3's reference shift
  (``mu = ln(6.93) = 1.93586``, ComfyUI's documented Flux3 ``shift``) and
  ``flow_shift = 2.5`` is krea2's training distribution
  (``mu = ln(2.5) = 0.91629``).  **Inference/validation** is separate: it uses
  ``scheduler_shift`` (also 16 by default) and ``scheduler.py``.
- **Flow-map (step-distilled) checkpoints**: ``flux-3-image-klein`` is a flow
  map, so its config sets ``flow_map=True`` and the forward *requires*
  ``target_timestep`` (the timestep the step jumps to).  Set
  ``"flow_map": true`` in the adapter config to pass it; the base checkpoint
  (``flux-3-image-klein-base``) leaves it ``False``.

Reference conditioning
----------------------
``supports_image_conditioning = True``.  The reference image latents are read
from ``batch["latents"][batch_configs[0]["reference_config"]]`` (the same
user-defined key convention the flux2/krea2 adapters use) and are fed as
``image_cond_hidden_states`` — a *separate mode*, not extra target tokens.  The
reference is clean (no noise), gets timestep-0 modulation, and its RoPE ids use
negative ``l``.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter, sample_sigma_uniform
from UnifiedTrainer.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Aspect buckets, krea2/flux2-klein style: {base_resolution: [(w, h), ...]}.
# Every dimension is a multiple of 16 (vae_scale_factor 8 x latent patch 2),
# which is what the reference pipeline enforces
# (`check_inputs(..., vae_scale_factor * 2)`).
RESOLUTION_CONFIG = {
    1024: [
        (512, 1280), (576, 1408), (640, 1344), (704, 1280),
        (768, 1216), (832, 1152), (896, 1088), (960, 1024),
        (1024, 960), (1088, 896), (1152, 832), (1216, 768),
        (1280, 704), (1344, 640), (1408, 576), (1280, 512),
    ],
}

# Qwen3-VL hidden-state layer taps — the FLUX.3 convention
# (`Flux3TextEncoderStep` default `text_encoder_out_layers`).
# 8 taps x 2560 = 20480 == `joint_attention_dim`.
DEFAULT_TEXT_ENCODER_SELECT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)

# Qwen3-VL hidden size (Qwen3-VL-4B).  Only used for a sanity check on the
# stacked width, so a differently-sized Qwen3-VL would still work as long as
# the transformer's `joint_attention_dim` matches.
_QWEN3VL_HIDDEN_SIZE = 2560

# The text window is right-padded to a multiple of this many tokens and the
# transformer attends to that padding, so it is part of the model recipe
# (`Flux3TextEncoderStep(pad_multiple=80)`).
_DEFAULT_PAD_MULTIPLE = 80

# Matches the reference `max_sequence_length` default.
_DEFAULT_MAX_SEQ_LEN = 8192

# Latents are patchified 2x2 by the pipeline and packed into the channel axis:
# (B, 32, H, W) -> (B, 128, H/2, W/2).  The transformer's own `patch_size` does
# not exist — this is a pipeline-level operation, expressed here through the
# framework's `patch_size` hook so bucket divisibility stays correct.
FLUX3_LATENT_PATCH_SIZE = 2

# The model was trained with the image H/W RoPE ids scaled by this factor
# (`Flux3PrepareLatentsStep(image_rope_scale=0.5)`).
_DEFAULT_IMAGE_ROPE_SCALE = 0.5

# Inference (validation) timestep shift — the official recipe's *alpha*, i.e.
# the factor in sigma(t) = alpha * t / (1 + (alpha - 1) * t) over t = 1..0.
#
#   alpha = 16.0 — THE OFFICIAL WORKFLOW DEFAULT.  The workflow node "Timestep
#                  shift (from alpha)" feeds mu = ln(16) = 2.77259 into
#                  ComfyUI's ModelSamplingFlux with base_shift == max_shift, so
#                  alpha is 16 at every resolution (not just 1 MP).  BFL's
#                  reference repo does the same (`Pipeline.generate(alpha=16)`),
#                  and the 16-step distilled workflow ships alpha 16.
#   alpha = 6.93 — the model's *reference* shift (ComfyUI's built-in Flux3
#                  sampling_settings are mu = ln(6.93) = 1.93586, and the
#                  50-step base template leaves the shift at that).  Override
#                  with `"scheduler_shift": 6.93` in the adapter config.
#
# The schedule is built by `scheduler.Flux3FlowMatchEulerDiscreteScheduler`
# (diffusers' own FlowMatchEuler applies `shift` twice and ends on a
# shifted sigma_min floor, which does NOT reproduce the recipe).
_FLUX3_SCHEDULER_SHIFT = 16.0
_FLUX3_NUM_TRAIN_TIMESTEPS = 1000

# Default shift for training-time logit-normal sigma sampling (the "flow_shift"
# scheme: mu = ln(flow_shift), sigma = sigmoid(N(mu, sigmoid_scale))).
#
# 16.0 == THE OFFICIAL WORKFLOW'S ALPHA: mu = ln(16) = 2.77259, the value the
# "Timestep shift (from alpha)" node feeds to ModelSamplingFlux (base_shift ==
# max_shift).  Training therefore samples from the same high-noise-heavy centre
# the model is sampled with at inference: mean sigma ~0.92, ~72% of draws above
# 0.9, ~0.3% below 0.5.
#
# Historical alternatives, still selectable per config:
#   2.5  — KREA2'S ACTUAL VALUE (mu = ln(2.5) = 0.91629, mean sigma ~0.68),
#          what every krea2 config launched in this project uses;
#   6.93 — FLUX.3's own reference shift (``Flux3SetTimestepsStep(shift=6.93)``,
#          mu = ln(6.93) = 1.93586, ComfyUI's built-in Flux3 sampling_settings,
#          mean sigma ~0.84);
#   1.0  — plain logit-normal centred at mu = 0 (mean sigma 0.5).
_DEFAULT_FLOW_SHIFT = 16.0


@ModelRegistry.register("flux3_klein")
class Flux3KleinAdapter(BaseModelAdapter):
    """Adapter for the FLUX.3 klein family (base + step-distilled)."""

    name = "flux3_klein"

    # Latents are 2x2-patchified and packed before the transformer, so bucket
    # divisibility must account for the patch grid too
    # (vae_scale_factor * patch_size = 8 * 2 = 16).
    patch_size = FLUX3_LATENT_PATCH_SIZE

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

        # Text-encoder layer taps (Qwen3-VL convention for FLUX.3).
        self.text_encoder_select_layers: tuple = tuple(
            config.get("text_encoder_select_layers", DEFAULT_TEXT_ENCODER_SELECT_LAYERS)
        )

        # Text-window padding granularity (right-padded to a multiple of this,
        # capped at max_sequence_length).  The transformer attends to the
        # padding, so this must match the checkpoint's recipe.
        self.pad_multiple: int = int(config.get("pad_multiple", _DEFAULT_PAD_MULTIPLE))
        self.max_sequence_length: int = int(
            config.get("max_sequence_length", _DEFAULT_MAX_SEQ_LEN)
        )

        # Image H/W RoPE scale — part of the checkpoint's training recipe.
        self.image_rope_scale: float = float(
            config.get("image_rope_scale", _DEFAULT_IMAGE_ROPE_SCALE)
        )

        # Flow-map (step-distilled) checkpoint: the forward REQUIRES
        # `target_timestep`.  `None` = infer from the loaded transformer's
        # config (`flow_map`), which is what the shipped checkpoints declare.
        self.flow_map = config.get("flow_map", None)
        if self.flow_map is not None:
            self.flow_map = bool(self.flow_map)

        # Timestep-shift mode for training sigma sampling — the flow_shift
        # scheme, defaulting to the official alpha (16).
        #
        #   "flow_shift"    — fixed-s logit-normal,
        #                     sigma = sigmoid(N(ln s, sigmoid_scale)), s =
        #                     "flow_shift" (DEFAULT 16 = the official
        #                     workflow's inference alpha).  This is the krea2
        #                     recipe form (musubi `--timestep_sampling shift
        #                     --discrete_flow_shift s`), resolution
        #                     independent.
        #   "dynamic_shift" — logit-normal whose mu is linearly interpolated
        #                     over [base_image_seq_len, max_image_seq_len]
        #                     between base_shift and max_shift, i.e. larger
        #                     images (more tokens) sample higher noise.  Same
        #                     form as Qwen21Adapter's dynamic_shift and
        #                     Krea2Adapter's pretrain_dynamic.
        #   "sigma"         — uniform sigma in [1/1000, 1] (musubi
        #                     timestep_sampling=sigma), via the shared base
        #                     helper.
        #
        # NOTE on the flow_shift default value: it is 16 — the official
        # workflow's alpha, mu = ln(16) = 2.77259 (mean sigma ~0.92, ~72% of
        # draws above 0.9, ~0.3% below 0.5).  Training is deliberately centred
        # on the alpha the model is sampled with at inference.  Other recipes,
        # selectable per config:
        #   "flow_shift": 6.93 — FLUX.3's reference shift
        #                        (Flux3SetTimestepsStep, mu = ln(6.93) = 1.93586,
        #                        the value ComfyUI documents for Flux3);
        #   "flow_shift": 2.5  — krea2's training distribution
        #                        (mu = ln(2.5) = 0.91629, mean sigma ~0.68);
        #   "flow_shift": 1.0  — plain logit-normal (mu = 0, mean sigma 0.5).
        self.timestep_shift_mode: str = config.get("timestep_shift_mode", "flow_shift")
        self.flow_shift: float = float(config.get("flow_shift", _DEFAULT_FLOW_SHIFT))
        self.sigmoid_scale: float = float(config.get("sigmoid_scale", 1.0))

        # Inference/validation timestep shift (the official recipe's alpha —
        # unrelated to the training `flow_shift` above).  Built by
        # `scheduler.Flux3FlowMatchEulerDiscreteScheduler` in `load_scheduler`,
        # and only used when the checkpoint ships no real scheduler directory.
        #   default 16.0 — official workflow ("Timestep shift (from alpha)":
        #                  mu = ln 16 = 2.77259, base_shift == max_shift);
        #   override 6.93 — the model's reference shift (base 50-step template).
        self.scheduler_shift: float = float(
            config.get("scheduler_shift", _FLUX3_SCHEDULER_SHIFT)
        )

        # dynamic_shift endpoints (mu-space, NOT the s of flow_shift).  The
        # defaults are the project's "dynamic + focus high noise" policy: at
        # 1024x1024 (4096 image tokens) they land at mu ~1.98, i.e. mean sigma
        # ~0.84 with ~72% of draws above 0.8.  FLUX.3's own reference is the
        # CONSTANT mu = ln(6.93) = 1.93586 -- use flow_shift for that.
        self.base_shift: float = float(config.get("base_shift", 1.5))
        self.max_shift: float = float(config.get("max_shift", 2.5))
        self.base_image_seq_len: int = int(config.get("base_image_seq_len", 256))
        self.max_image_seq_len: int = int(config.get("max_image_seq_len", 8192))

        if self.timestep_shift_mode not in ("flow_shift", "dynamic_shift", "sigma"):
            raise ValueError(
                f"flux3_klein: unknown timestep_shift_mode "
                f"'{self.timestep_shift_mode}'. Expected 'flow_shift', "
                "'dynamic_shift' or 'sigma'."
            )
        if self.flow_shift <= 0.0:
            raise ValueError(
                f"flux3_klein: flow_shift must be > 0 (mu = ln(flow_shift)), "
                f"got {self.flow_shift}"
            )
        if self.sigmoid_scale <= 0.0:
            raise ValueError(
                f"flux3_klein: sigmoid_scale must be > 0, got {self.sigmoid_scale}"
            )
        if self.scheduler_shift <= 0.0:
            raise ValueError(
                f"flux3_klein: scheduler_shift (inference alpha) must be > 0, "
                f"got {self.scheduler_shift}"
            )
        if self.timestep_shift_mode == "dynamic_shift":
            if self.max_image_seq_len <= self.base_image_seq_len:
                raise ValueError(
                    f"flux3_klein: max_image_seq_len "
                    f"({self.max_image_seq_len}) must exceed base_image_seq_len "
                    f"({self.base_image_seq_len}) for dynamic_shift."
                )

        # Target grid dimensions cached from prepare_model_input — used by
        # unpack_prediction (krea2-style, no sqrt guesses).
        self._target_grids: list[tuple[int, int]] = []
        self._ref_grids: list[tuple[int, int]] = []

    # ── Model loading ──────────────────────────────────────────────────

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from .transformer_flux3 import BlockSwapFlux3Transformer2DModel

        transformer = BlockSwapFlux3Transformer2DModel.from_pretrained(
            path, torch_dtype=dtype
        )
        # The checkpoint's own config is the source of truth for `flow_map`
        # (the step-distilled `flux-3-image-klein` package ships
        # `"flow_map": true`).  An explicit adapter config value still wins, so
        # a user can override a mis-declared package.
        if self.flow_map is None:
            self.flow_map = bool(getattr(transformer.config, "flow_map", False))
            logger.info(
                f"flux3_klein: flow_map={self.flow_map} read from the "
                f"transformer config ({path})."
            )
        return transformer

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers.models import AutoencoderKLFlux2

        return AutoencoderKLFlux2.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        """Build the FLUX.3 inference scheduler.

        The FLUX.3 module ships **no scheduler component**, and the official
        workflow does not rely on one either: the "Timestep shift (from
        alpha)" node feeds ``mu = ln(alpha)`` (alpha 16, ``mu = 2.77259``) into
        ComfyUI's ``ModelSamplingFlux`` with ``base_shift == max_shift``, i.e.
        the plain ``sigma(t) = alpha*t / (1 + (alpha-1)*t)`` grid over
        ``t = 1..0``.  Falling through to the trainer's ``shift=1.0`` default
        would be nothing like that recipe, so the adapter builds
        :class:`Flux3FlowMatchEulerDiscreteScheduler` (diffusers'
        ``FlowMatchEulerDiscreteScheduler`` double-applies ``shift`` and ends
        on a ``sigma_min`` floor — see ``scheduler.py``).  A real scheduler
        directory is still honoured when the checkpoint actually ships one.

        ``self.scheduler_shift`` is the recipe alpha: 16.0 by default (official
        workflow / BFL reference), 6.93 via ``"scheduler_shift": 6.93`` for the
        model's reference shift.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        if path and os.path.isdir(path):
            try:
                sched = FlowMatchEulerDiscreteScheduler.from_pretrained(path)
                # An explicit scheduler directory wins, but it is then NOT the
                # official recipe — say so, because FLUX.3 packages normally
                # have no scheduler component and this path is easy to trip
                # over (a leftover shift=1.0 config renders badly).
                logger.info(
                    f"flux3_klein: using the scheduler at {path} "
                    f"(shift={getattr(sched.config, 'shift', None)}); "
                    f"scheduler_shift={self.scheduler_shift} only applies to "
                    f"the built-in official recipe."
                )
                return sched
            except Exception as e:  # noqa: BLE001 - fall through to the recipe
                logger.warning(
                    f"flux3_klein: could not load scheduler from {path} ({e}); "
                    f"using the FLUX.3 recipe "
                    f"(alpha={self.scheduler_shift}, "
                    f"mu=ln(alpha)={math.log(self.scheduler_shift):.5f})."
                )

        from .scheduler import Flux3FlowMatchEulerDiscreteScheduler

        return Flux3FlowMatchEulerDiscreteScheduler(
            alpha=self.scheduler_shift,
            num_train_timesteps=_FLUX3_NUM_TRAIN_TIMESTEPS,
        )

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        """Load the FLUX.3 text encoder — ``Qwen3VLForConditionalGeneration``."""
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load FLUX.3 text encoder (Qwen3-VL): {e}")
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        """Load the Qwen2 tokenizer family used by Qwen3-VL."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load FLUX.3 tokenizer from {path}: {e}")
            return None

    def load_processor(self, path: str, tokenizer_path: str = "") -> Optional[Any]:
        """Load the Qwen3-VL processor (its tokenizer drives the chat template).

        FLUX.3 text conditioning is text-only (reference images enter through
        the VAE as ``image_cond_*`` tokens), so only the processor's tokenizer
        half is used — but ``apply_chat_template`` lives on the processor, which
        is why the reference pipeline passes one.
        """
        try:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load FLUX.3 processor from {path}: {e}")
            return None

    # ── Architecture specs ─────────────────────────────────────────────

    @property
    def latent_channels(self) -> int:
        # Width of the latent as STORED IN THE CACHE: the VAE's raw 32 channels
        # patchified 2x2 into the channel axis (32 * 4 = 128) — i.e. what the
        # transformer's `x_embedders["image"]` consumes directly.  `encode_image`
        # does the patchify, so the cache holds transformer-space latents
        # (see that docstring).
        return 32 * FLUX3_LATENT_PATCH_SIZE * FLUX3_LATENT_PATCH_SIZE

    @property
    def raw_latent_channels(self) -> int:
        """The VAE's own latent width (pre-patchify) — for reference only."""
        return 32

    @property
    def vae_scale_factor(self) -> int:
        return 8

    @property
    def embedding_dim(self) -> int:
        # joint_attention_dim = len(taps) * Qwen3-VL hidden size.
        return len(self.text_encoder_select_layers) * _QWEN3VL_HIDDEN_SIZE

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return True

    # ── Encoding ───────────────────────────────────────────────────────

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C*4, H//2, W//2) — FLUX.2/3 2x2 patchify.

        Mirrors ``Flux3VaeEncoderStep._patchify_latents`` (and
        ``Flux2VaeEncoderStep``): the 2x2 spatial neighbourhood is folded into
        the channel axis, so a 32-channel latent becomes 128 channels on a
        half-resolution grid — exactly the transformer's ``in_channels``.
        """
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_patchify_latents`."""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @staticmethod
    def _bn_stats(vae: nn.Module, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel BatchNorm mean/std of the packed latents.

        FLUX.3 (like FLUX.2's reference-image path) normalizes *packed* latents
        with the VAE's own BatchNorm running statistics rather than with
        ``scaling_factor`` / ``shift_factor``.
        """
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=latent.device, dtype=latent.dtype
        )
        eps = float(getattr(vae.config, "batch_norm_eps", 1e-5))
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1).to(
                device=latent.device, dtype=latent.dtype
            )
            + eps
        )
        return bn_mean, bn_std

    def _normalize_latents(self, vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
        """BatchNorm-normalize *packed* latents (the FLUX.3 VAE convention).

        The BatchNorm statistics have one entry per *packed* channel
        (``32 * 4 = 128``), so the caller must patchify first — exactly as the
        reference ``Flux3VaeEncoderStep`` does.
        """
        bn_mean, bn_std = self._bn_stats(vae, latents)
        return (latents - bn_mean) / bn_std

    @staticmethod
    def _denormalize_latents(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
        """Inverse of the BatchNorm normalization (input: packed latents)."""
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype
        )
        eps = float(getattr(vae.config, "batch_norm_eps", 1e-5))
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1).to(
                device=latents.device, dtype=latents.dtype
            )
            + eps
        )
        return latents * bn_std + bn_mean

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        """Encode an image to the transformer-space latent the cache stores.

        Follows the reference ``Flux3VaeEncoderStep`` exactly:

            latent = vae.encode(img).latent_dist.mode()      # (B, 32, H, W)
            latents = _patchify_latents(latent)              # (B, 128, H/2, W/2)
            latents = (latents - bn_mean) / bn_std

        The result is what the transformer's ``x_embedders["image"]`` consumes,
        so the heavy per-image work (patchify + BatchNorm) happens once at cache
        build time instead of every training step, and ``latent_channels``
        reports the stored width (``32 * 4 = 128``) so the cache/placeholder
        bookkeeping stays consistent.
        """
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.mode()
            packed = self._patchify_latents(latent)
            packed = self._normalize_latents(vae, packed)
        return {"latent": packed}

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        """Decode a cached (packed, BN-normalized) latent back to pixels."""
        latent = latent.to(device=vae.device, dtype=vae.dtype)
        latent = self._denormalize_latents(vae, latent)
        latent = self._unpatchify_latents(latent)
        with torch.no_grad():
            image = vae.decode(latent).sample
        return image

    def decode_latent_differentiable(
        self, vae: nn.Module, latent: torch.Tensor, tile_size: int = 0
    ) -> Any:
        """Gradient-enabled decode for perceptual losses (``losses/pfm.py``).

        Same math as :meth:`decode_latent` (BN denormalize + unpatchify) without
        the ``no_grad`` guard, so autograd can reach the model prediction.
        ``tile_size`` is accepted for the base-class signature but unused: a
        tiled branch would have to stay bit-identical to the no-grad path and
        the 8x VAE decodes within budget at supported bucket resolutions.
        """
        latent = latent.to(device=vae.device, dtype=vae.dtype)
        latent = self._denormalize_latents(vae, latent)
        latent = self._unpatchify_latents(latent)
        return vae.decode(latent).sample

    # ── Text encoding ──────────────────────────────────────────────────

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
        """Encode a prompt with the Qwen3-VL text encoder (FLUX.3 recipe).

        Mirrors ``Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds``:

        1. chat template (``add_generation_prompt=True``),
        2. measure the real token length, round it **up to a multiple of
           ``pad_multiple``** (capped at ``max_sequence_length``),
        3. tokenize right-padded to that window,
        4. run with explicit ``position_ids`` (padding slots pinned to 1 —
           Qwen3-VL versions disagree on the padding position id and FLUX.3
           attends to the padding, so it is passed explicitly),
        5. stack the configured hidden-state taps along the channel axis.

        ``reference_image`` is accepted (the cache builder passes it
        unconditionally) but ignored: FLUX.3 conditioning is text-only, and
        reference images enter as VAE latents in ``prepare_model_input``.

        Returns
        -------
        dict
            ``prompt_embed``       — ``(seq_len, len(taps) * hidden)``
            ``prompt_embeds_mask`` — ``(seq_len,)`` bool (real tokens True)
        """
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "FLUX.3 klein requires a loaded Qwen3-VL text encoder and tokenizer."
            )

        # `apply_chat_template` lives on the processor when one is available
        # (the reference passes `AutoProcessor`); fall back to the tokenizer.
        holder = processor if processor is not None else tokenizer

        messages = [{"role": "user", "content": prompt}]
        try:
            text = holder.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            # Processor/tokenizer without a chat template — feed the raw prompt.
            logger.warning(
                "FLUX.3 klein: chat template unavailable, encoding the raw prompt."
            )
            text = prompt

        # ── 2. Round the window up to the pad multiple ──────────────────
        # NO truncation: the encoder is Qwen3-VL (long context), so the caption
        # keeps its natural length and is only rounded UP to the pad multiple
        # (the reference checkpoint was trained with multiples of 8/16).
        inner = getattr(holder, "tokenizer", holder)
        try:
            real_length = inner(
                text,
                return_tensors="pt",
                truncation=False,
            )["input_ids"].shape[1]
        except Exception:  # noqa: BLE001 - fall back to the configured cap
            real_length = self.max_sequence_length

        padded_length = max(
            math.ceil(real_length / self.pad_multiple) * self.pad_multiple, 1)

        inputs = inner(
            text,
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            truncation=False,
            max_length=padded_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Padding positions get position id 1, matching the training
        # environment of the reference checkpoint.
        position_ids = torch.arange(padded_length, device=device)[None].masked_fill(
            attention_mask == 0, 1
        )

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        # ── 4. Stack the layer taps: (B, taps, L, D) -> (B, L, taps*D) ──
        hidden_states = getattr(outputs, "hidden_states", None)
        if self.text_encoder_select_layers and hidden_states:
            out = torch.stack(
                [hidden_states[i] for i in self.text_encoder_select_layers], dim=1
            )
        elif getattr(outputs, "last_hidden_state", None) is not None:
            out = outputs.last_hidden_state.unsqueeze(1)
        elif hidden_states:
            out = hidden_states[-1].unsqueeze(1)
        else:
            raise RuntimeError(
                "FLUX.3 klein text encoder output has neither hidden_states nor "
                "last_hidden_state — cannot build prompt embeddings."
            )

        batch_size, num_layers, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )

        return {
            "prompt_embed": prompt_embeds[0].to(dtype=dtype),
            "prompt_embeds_mask": attention_mask[0].to(torch.bool),
        }

    # ── Token packing / position ids ───────────────────────────────────

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C) — plain spatial flatten.

        The 2x2 patchify already happened in ``encode_image``, so this is just
        ``flatten(2).transpose(1, 2)`` exactly like the reference
        ``Flux3PrepareLatentsStep``.
        """
        return latents.flatten(2).transpose(1, 2)

    def _prepare_txt_ids(self, text_seq_len: int, device: torch.device) -> torch.Tensor:
        """4-axis RoPE ids for text tokens: ``(seq_len, 4)`` as ``(0, 0, 0, i)``."""
        t = torch.arange(1, device=device)
        h = torch.arange(1, device=device)
        w = torch.arange(1, device=device)
        l = torch.arange(text_seq_len, device=device)
        return torch.cartesian_prod(t, h, w, l)

    def _prepare_image_ids(
        self,
        height: int,
        width: int,
        l_coord: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """4-axis RoPE ids for one image token grid: ``(height*width, 4)``.

        ``t = 0``, the H/W axes scaled by ``image_rope_scale``, and a constant
        ``l`` coordinate (0 for the generated/target image, negative for
        references) — mirrors ``_prepare_image_ids`` in the reference.
        """
        t = torch.arange(1, device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        l = torch.tensor([l_coord], device=device)

        ids = torch.cartesian_prod(t, h, w, l).to(dtype=torch.float32)
        ids[:, 1:3] = ids[:, 1:3] * self.image_rope_scale
        return ids

    # ── Training hooks ─────────────────────────────────────────────────

    def _resolve_flow_map(self, batch: dict) -> bool:
        """Whether the loaded transformer is a flow map (needs ``target_timestep``).

        Set from the checkpoint config by :meth:`load_transformer`; an explicit
        adapter-config ``flow_map`` overrides it.  ``batch["_flow_map"]`` is a
        last-resort fallback for callers that construct the adapter without
        loading a transformer (e.g. unit tests).
        """
        if self.flow_map is not None:
            return self.flow_map
        return bool(batch.get("_flow_map", False))

    def prepare_model_input(
        self,
        batch: dict,
        noise: torch.Tensor | list[torch.Tensor],
        sigmas: torch.Tensor,
    ) -> dict:
        """Assemble the kwargs for ``Flux3Transformer2DModel.forward``.

        The target image tokens go in ``hidden_states``; reference image
        latents (from the batch's configured ``reference_config`` key) go in
        ``image_cond_hidden_states`` as a separate mode with their own RoPE ids
        using negative ``l`` coordinates.  Cached text embeddings are padded to
        the batch-max length and stacked.

        Multiple noisy targets are NOT supported by this architecture: the
        transformer marks exactly one image mode, so only the first target is
        used (an explicit error is raised if more are supplied).

        Latents arrive from the cache already packed and BatchNorm-normalized
        (see :meth:`encode_image`), so this is a pure tokenization step.
        """
        if isinstance(noise, list):
            if len(noise) != 1:
                raise NotImplementedError(
                    f"flux3_klein supports exactly one noisy target image per "
                    f"sequence (the 'image' mode of the transformer), got "
                    f"{len(noise)} targets. Configure a single-target "
                    "batch_config."
                )
            target = noise[0]
        else:
            target = noise

        if target.ndim != 4:
            raise ValueError(
                f"flux3_klein expects a 4D target latent (B, C, H, W), got shape "
                f"{tuple(target.shape)}. This adapter is image-only."
            )

        B, C, H, W = target.shape
        device, dtype = target.device, target.dtype
        if C != self.latent_channels:
            raise ValueError(
                f"flux3_klein expects {self.latent_channels}-channel latents "
                f"(patchified FLUX.2/3 VAE space), got {C}. The cache was "
                "probably built by a different model — rebuild it."
            )

        # ── Target tokens ───────────────────────────────────────────────
        target_tokens = self._pack_latents(target)
        # Grid of the token sequence the transformer sees (drives the RoPE
        # ids and the prediction unpack).
        self._target_grids = [(H, W)]

        # ── Reference tokens (image-conditioned / edit path) ────────────
        latents = batch.get("latents", {})
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        ref_key = resolved_bc.get("reference_config")
        ref = latents.get(ref_key) if ref_key else None
        refs = ref if isinstance(ref, list) else ([ref] if ref is not None else [])

        cond_tokens: list[torch.Tensor] = []
        cond_ids: list[torch.Tensor] = []
        self._ref_grids = []
        for index, ref_tensor in enumerate(refs):
            if ref_tensor is None:
                continue
            ref_tensor = ref_tensor.to(device=device, dtype=dtype)
            if ref_tensor.ndim == 3:
                ref_tensor = ref_tensor.unsqueeze(0)
            _rb, rc, rh, rw = ref_tensor.shape
            if rc != self.latent_channels:
                raise ValueError(
                    f"flux3_klein reference latent has {rc} channels, expected "
                    f"{self.latent_channels} — rebuild the cache for this model."
                )
            self._ref_grids.append((rh, rw))
            cond_tokens.append(self._pack_latents(ref_tensor))
            # References are told apart on the `l` axis: -1, -2, ...
            cond_ids.append(
                self._prepare_image_ids(rh, rw, l_coord=-(1 + index), device=device)
            )

        # ── Text conditioning from cached embeddings ─────────────────────
        encoder_hidden_states = self._extract_encoder_hidden_states(batch, device, dtype)
        if encoder_hidden_states is None:
            raise ValueError(
                "FLUX.3 klein requires encoder_hidden_states in the batch. "
                "Ensure the data pipeline provides pre-computed text encoder "
                "hidden states (cached via encode_text)."
            )
        text_seq_len = encoder_hidden_states.shape[1]

        # ── Position ids ────────────────────────────────────────────────
        # `Flux3PosEmbed` indexes `pos[..., i]` per axis, so both id tensors
        # are (seq_len, 4).  The reference passes them unbatched — the rotary
        # embedding is shared across the batch.
        txt_ids = self._prepare_txt_ids(text_seq_len, device)
        img_ids = self._prepare_image_ids(H, W, l_coord=0, device=device)

        result: dict[str, Any] = {
            "hidden_states": target_tokens,
            "encoder_hidden_states": encoder_hidden_states,
            # The transformer multiplies by 1000 internally — pass [0, 1].
            "timestep": sigmas.to(device=device, dtype=dtype),
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "return_dict": False,
        }

        if cond_tokens:
            # Reference latents may have different grids across samples in a
            # batch; the transformer builds one sequence, so all rows must
            # share the layout (take row 0's token order for all rows).
            image_cond_hidden_states = torch.cat(
                [t if t.shape[0] == B else t.expand(B, -1, -1) for t in cond_tokens],
                dim=1,
            )
            result["image_cond_hidden_states"] = image_cond_hidden_states
            result["image_cond_ids"] = torch.cat(cond_ids, dim=0)

        # ── Flow map (step-distilled) checkpoints ───────────────────────
        if self._resolve_flow_map(batch):
            # Inference passes the timestep the step jumps to.  During training
            # the sampled sigma IS the current step and the target is the clean
            # sample, so the jump target is 0 (t -> 0 in one step), which is
            # exactly the flow-map training objective: predict the velocity that
            # reaches x0 in a single jump from sigma.
            result["target_timestep"] = torch.zeros(
                B, device=device, dtype=dtype
            )

        # NOTE: the FLUX.3 forward accepts NO text attention mask (the joint
        # attention over text + content is unmasked and the model relies on the
        # padding being present in the text window).  The mask is still
        # extracted above for batch bookkeeping but deliberately NOT forwarded
        # — an unexpected kwarg would TypeError.

        return result

    def _finalize_prediction(self, model_pred: Any) -> torch.Tensor:
        """Unwrap a ``Transformer2DModelOutput`` / tuple into the raw tensor."""
        if hasattr(model_pred, "sample"):
            return model_pred.sample
        if isinstance(model_pred, (tuple, list)):
            return model_pred[0]
        return model_pred

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor | list[torch.Tensor]:
        """Unpack the transformer output back to a ``(B, C, H, W)`` velocity.

        The forward returns only the *image* tokens (``content_states[:,
        :image_seq_len]``), so this is the inverse of the spatial pack: a
        ``view``/``permute`` with no slicing required.
        """
        model_pred = self._finalize_prediction(model_pred)

        if model_pred.dim() != 3:
            return model_pred

        if not self._target_grids:
            raise ValueError(
                f"_target_grids is empty — cannot unpack a prediction of "
                f"seq_len={model_pred.shape[1]}. Call prepare_model_input "
                "before unpack_prediction."
            )

        B, seq_len, channels = model_pred.shape
        gh, gw = self._target_grids[0]
        if seq_len != gh * gw:
            raise ValueError(
                f"flux3_klein: prediction seq_len {seq_len} does not match the "
                f"cached target grid {gh}x{gw} = {gh * gw} tokens. The "
                "transformer should return exactly the image tokens."
            )

        latents = model_pred.transpose(1, 2).reshape(B, channels, gh, gw)
        return [latents]

    def compute_target(
        self, noise: torch.Tensor, learning_target: torch.Tensor
    ) -> torch.Tensor:
        # Standard velocity parameterization: v = noise - x0.
        return noise - learning_target

    # ── Timestep sampling (official alpha default; krea2 recipe form) ──

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample flow-matching sigmas with the flow_shift recipe.

        - ``"flow_shift"`` (DEFAULT) — fixed-s logit-normal::

              mu    = ln(flow_shift)
              u     ~ N(mu, sigmoid_scale)
              sigma = sigmoid(u).clamp(1e-5, 1 - 1e-5)

          Identical in form to ``Krea2Adapter.sample_timesteps`` in its
          ``flow_shift`` mode (musubi ``--timestep_sampling shift
          --discrete_flow_shift s``), and resolution-independent.

          With the default ``flow_shift = 16`` this gives
          ``mu = ln(16) = 2.77259`` — the official workflow's inference alpha
          (mean sigma ~0.92, ~72% of draws above 0.9, ~0.3% below 0.5), so
          training is centred on the alpha the model is sampled with.
          ``flow_shift = 6.93`` gives FLUX.3's own reference shift
          (``mu = ln(6.93) = 1.93586``, mean sigma ~0.84);
          ``flow_shift = 2.5`` gives krea2's training distribution
          (``mu = ln(2.5) = 0.91629``, mean sigma ~0.68);
          ``flow_shift = 1.0`` gives the plain logit-normal (mu 0, sigma 0.5).

        - ``"sigma"`` — uniform sigma in ``[1/1000, 1]`` via the shared
          :func:`sample_sigma_uniform` helper (musubi ``timestep_sampling=
          sigma``).

        Returns ``(timesteps, sigmas)``: timesteps are ``sigmas * 1000`` for
        scheduler bookkeeping, sigmas in ``[0, 1]`` and are what
        ``prepare_model_input`` hands the transformer (which multiplies by
        1000 itself).
        """
        n = _FLUX3_NUM_TRAIN_TIMESTEPS

        if self.timestep_shift_mode == "sigma":
            return sample_sigma_uniform(batch_size, device, dtype, num_timesteps=n)

        if self.timestep_shift_mode == "dynamic_shift":
            if latent_height is None or latent_width is None:
                # No shape info: use the high-noise end rather than the bare
                # base-class default (mu = 0).
                mu = self.max_shift
            else:
                mu = self._dynamic_shift_mu(int(latent_height) * int(latent_width))
        else:
            # flow_shift — fixed-s logit-normal (default s = 16).
            mu = math.log(self.flow_shift)

        u = torch.normal(
            mean=mu, std=self.sigmoid_scale, size=(batch_size,), device=device
        )
        sigmas = torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5).to(dtype=dtype)
        return sigmas * n, sigmas

    def _dynamic_shift_mu(self, image_seq_len: int) -> float:
        """Linear interpolation of mu between the base and max seq lengths.

        Mirrors ``Krea2Adapter._krea2_calculate_shift`` and Qwen21Adapter's
        dynamic_shift exactly (same formula, no clamping).
        """
        m = (self.max_shift - self.base_shift) / (
            self.max_image_seq_len - self.base_image_seq_len
        )
        b = self.base_shift - m * self.base_image_seq_len
        return image_seq_len * m + b

    # ── Helpers ────────────────────────────────────────────────────────

    def _extract_encoder_hidden_states(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Extract and stack cached ``prompt_embed`` rows to ``(B, max_seq, dim)``.

        Captions have different (pad-multiple-quantized) lengths, so rows are
        zero-padded to the batch max before stacking.
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
            pe = pe.to(device=device, dtype=dtype)
            if pe.ndim != 2:
                raise ValueError(
                    "flux3_klein expects cached prompt_embed of shape "
                    f"(seq_len, dim), got shape {tuple(pe.shape)} with "
                    f"ndim={pe.ndim}. Recreate the embedding cache for this model."
                )
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
        """Extract and stack ``prompt_embeds_mask`` to ``(B, max_seq)`` bool.

        Only used for batch bookkeeping — the FLUX.3 forward takes no text
        mask.  Padded slots are False.
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
