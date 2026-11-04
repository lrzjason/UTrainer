"""MOSS-Image adapter (MOSS-Image-Base / -SFT) for UnifiedTrainer.

Architecture under the hood (from the checkpoint's config, verified):

- transformer: ``MOSSImageTransformer`` — dim 3072, ffn 8192, 24 heads, 42 joint
  blocks, ``latent_channels=128`` (32 VAE channels patchified 2x2), ``text_dim=4096``;
- text encoder: Qwen3.5-9B (``Qwen3_5Model``, 32 text layers, 4096 hidden), which
  is multimodal — the vision tower is what a reference image goes through;
- vae: ``diffusers.AutoencoderKLFlux2`` (32 latent channels, 2x2 patch).

Layout: the official ``MOSSImageTransformer`` is a normal diffusers-style model —
``forward(hidden_states [B,128,H,W], timestep = sigma*1000,
encoder_hidden_states [B,S,4096], encoder_attention_mask [B,S] bool)`` — so the
adapter is a straight bridge from UnifiedTrainer's ``(B, C, H, W)`` batches; no
packed token axis is involved (the model loops over the batch and drops padded
text tokens itself).

Attribution: the transformer/scheduler/pipeline are the **OpenMOSS Team**'s
official implementation, vendored under ``UnifiedTrainer/vendor/openmoss_moss_image/``.
The training recipe details we ported from **Kohaku-Blueleaf**'s KohakuMoss
(timestep pmf, encoder handling) are marked per file. Nothing is imported from a
reference clone at runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

logger = logging.getLogger(__name__)

#: MOSS-Image latent channels are the VAE's 32 channels patchified 2x2.
_VAE_CHANNELS = 32
_PATCH = 2
_LATENT_CHANNELS = _VAE_CHANNELS * _PATCH * _PATCH
_TEXT_DIM = 4096


@ModelRegistry.register("moss")
class MossAdapter(BaseModelAdapter):
    """Adapter for MOSS-Image (packed DiT + Qwen3.5-9B + Flux2 VAE)."""

    name = "moss"
    patch_size = _PATCH

    # Wide diptych canvases (two panels butt-joined left-right) need buckets
    # beyond the standard ratios, otherwise centre-crop cuts into both panels.
    # Keyed by base resolution; merged with the auto-generated standard buckets.
    _DIPETYCH_BUCKETS = {
        1024: [(1088, 960), (1456, 720), (1680, 624), (1776, 592), (1936, 544)],
    }

    # ── construction ──────────────────────────────────────────────────

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        # Reference-image conditioning: the reference goes through the text
        # encoder's vision tower (no extra DiT tokens - the MOSS backbone has
        # no reference-token path).
        self.reference_conditioning: bool = bool(
            self.config.get("reference_conditioning", False)
        )
        self.multimodal_reference: bool = bool(
            self.config.get("multimodal_reference", False)
        )

    # ── specs ─────────────────────────────────────────────────────────

    @property
    def resolution_config(self) -> dict:
        """Standard buckets merged with the wide diptych buckets."""
        from UnifiedTrainer.data.bucket import STANDARD_RATIOS, _generate_buckets

        merged: dict = {}
        for base, extras in self._DIPETYCH_BUCKETS.items():
            buckets = list(_generate_buckets(base, self.bucket_divisibility, STANDARD_RATIOS))
            for bucket in extras:
                if bucket not in buckets:
                    buckets.append(bucket)
            merged[base] = buckets
        return merged

    @property
    def latent_channels(self) -> int:
        return _LATENT_CHANNELS

    @property
    def vae_scale_factor(self) -> int:
        return 8

    @property
    def embedding_dim(self) -> int:
        return _TEXT_DIM

    @property
    def supports_image_conditioning(self) -> bool:
        """References are supported as latent spans *and* as VL tokens."""
        return bool(self.reference_conditioning or self.multimodal_reference)

    # ── loading ───────────────────────────────────────────────────────

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        """Build and load the official ``MOSSImageTransformer``.

        ``path`` may be the ModelScope/HF model directory (``<path>/transformer``)
        or the transformer directory itself.
        """
        import os as _os

        from UnifiedTrainer.vendor.openmoss_moss_image import MOSSImageTransformer

        sub = _os.path.join(path, "transformer")
        transformer_dir = sub if _os.path.isdir(sub) else path
        if self.config.get("init_from_config"):
            # Plumbing-only path: exercises the adapter/trainer/data wiring before
            # (or without) the checkpoint.  Never use it for a real run.
            logger.warning(
                "MOSS: init_from_config=True - the transformer is RANDOMLY initialised "
                "from %s/config.json, NOT the trained checkpoint.",
                transformer_dir,
            )
            cfg = MOSSImageTransformer.load_config(transformer_dir)
            model = MOSSImageTransformer.from_config(cfg)
        else:
            model = MOSSImageTransformer.from_pretrained(transformer_dir, torch_dtype=dtype)
        model = model.to(dtype=dtype)
        from UnifiedTrainer.models.moss.train_forward import install_training_hooks

        install_training_hooks(model)
        logger.info(
            "MOSS transformer loaded: %.1fM params from %s",
            sum(p.numel() for p in model.parameters()) / 1e6,
            transformer_dir,
        )
        return model

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from UnifiedTrainer.models.moss.encoders import load_vae

        return load_vae(path, dtype=dtype)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        from UnifiedTrainer.models.moss.encoders import load_text_encoder

        return load_text_encoder(path, dtype=dtype)

    def load_tokenizer(self, path: str) -> Optional[Any]:
        from UnifiedTrainer.models.moss.encoders import load_tokenizer

        return load_tokenizer(path)

    def load_scheduler(self, path: str) -> Any:
        """The official MOSS scheduler (Euler + the training V2 nearest-grid schedule).

        Training does not use it (the adapter's ``sample_timesteps`` owns the
        sigma recipe); it is loaded for the validation / sampling path.
        """
        import os as _os

        from UnifiedTrainer.vendor.openmoss_moss_image import MOSSImageScheduler

        sub = _os.path.join(path, "scheduler")
        scheduler_dir = sub if _os.path.isdir(sub) else path
        return MOSSImageScheduler.from_pretrained(scheduler_dir)

    def load_processor(self, path: str) -> Optional[Any]:
        """The multimodal processor (needed only for reference-image encoding).

        MOSS's own HF directory does not ship ``preprocessor_config.json``, so the
        processor comes from ``processor_path`` (train.py's config key) or, as a
        fallback, from the ``text_processor_path`` the adapter config may set —
        normally the local Qwen3.5-9B checkout. See ``md/15-moss-utrainer-port.md``.
        """
        from UnifiedTrainer.models.moss.encoders import load_processor

        return load_processor(
            self.config.get("processor_path")
            or self.config.get("text_processor_path")
            or path
        )

    # ── encoding ──────────────────────────────────────────────────────

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        """``(B, 3, H, W)`` in [-1, 1] -> cached latent ``(B, 128, H/16, W/16)``."""
        from UnifiedTrainer.models.moss.encoders import encode_latent

        return {"latent": encode_latent(vae, image_tensor)}

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
        """Caption (plus optional reference images) -> ``prompt_embed`` ``(L, 4096)``.

        Mirrors the Krea-2 adapter's contract: ``reference_image`` is a PIL image
        or a list of them, already resized by the cache builder; with ``processor``
        present they are embedded as ``Picture N: <|vision_start|>…`` in the user
        message and go through the vision tower.
        """
        from UnifiedTrainer.models.moss.encoders import encode_prompt

        return encode_prompt(
            text_encoder,
            tokenizer,
            prompt,
            device,
            dtype,
            processor=processor,
            reference_image=reference_image,
        )

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        from UnifiedTrainer.models.moss.encoders import decode_latent

        return decode_latent(vae, latent)

    # ── training hooks ────────────────────────────────────────────────

    def prepare_model_input(
        self,
        batch: dict,
        noise: torch.Tensor | list[torch.Tensor],
        sigmas: torch.Tensor,
    ) -> dict:
        from UnifiedTrainer.models.moss.pack import prepare_model_input

        return prepare_model_input(self, batch, noise, sigmas)

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        from UnifiedTrainer.models.moss.pack import unpack_prediction

        return unpack_prediction(self, model_pred)

    def compute_target(
        self, noise: torch.Tensor, learning_target: torch.Tensor
    ) -> torch.Tensor:
        return noise - learning_target

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MOSS shift sampling (``diffusion/timestep/moss.py`` semantics)."""
        from UnifiedTrainer.models.moss.pack import sample_timesteps

        return sample_timesteps(self, batch_size, device, dtype,
                                latent_height, latent_width)
