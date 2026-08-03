"""Flux2 Klein 4B model adapter."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

RESOLUTION_CONFIG = {
    1024: [
        (512, 1280), (576, 1408), (640, 1344), (704, 1280),
        (768, 1216), (832, 1152), (896, 1088), (960, 1024),
        (1024, 960), (1088, 896), (1152, 832), (1216, 768),
        (1280, 704), (1344, 640), (1408, 576), (1280, 512),
    ],
}


@ModelRegistry.register("flux2_klein")
class Flux2KleinAdapter(BaseModelAdapter):
    """Adapter for Flux2 Klein 4B model."""

    name = "flux2_klein"

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from .transformer_flux2 import Flux2Transformer2DModel
        return Flux2Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers.models import AutoencoderKLFlux2
        return AutoencoderKLFlux2.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_pretrained(path)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype)
        except Exception:
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        try:
            from transformers import Qwen2Tokenizer
            return Qwen2Tokenizer.from_pretrained(path)
        except Exception:
            return None

    @property
    def latent_channels(self) -> int:
        return 16

    @property
    def vae_scale_factor(self) -> int:
        return 8

    @property
    def embedding_dim(self) -> int:
        return 7680

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return True

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        return {"latent": latent}

    def encode_text(self, text_encoder, tokenizer, prompt, device, dtype) -> dict:
        return {"prompt": prompt, "device": device}

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        latent = latent / vae.config.scaling_factor + vae.config.shift_factor
        with torch.no_grad():
            image = vae.decode(latent).sample
        return image

    def prepare_model_input(
        self, batch: dict, noise: torch.Tensor | list[torch.Tensor], sigmas: torch.Tensor
    ) -> dict:
        latents = batch.get("latents", {})

        # Resolve reference from batch_config — user-defined keys, NOT hardcoded.
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        ref_key = resolved_bc.get("reference_config")
        ref = latents.get(ref_key) if ref_key else None

        # Support single or multiple target tensors (channel-concat design).
        noises = noise if isinstance(noise, list) else [noise]
        target = noises[0]

        # Concatenate reference + noise for image conditioning
        if ref is not None:
            model_input = torch.cat([ref, target], dim=1)
        else:
            model_input = target

        timesteps = (sigmas * 1000).long()
        return {
            "hidden_states": model_input,
            "timestep": timesteps,
        }

    def unpack_prediction(self, model_pred: torch.Tensor, input_ids=None) -> torch.Tensor:
        # Flux packs latents into a sequence; unpack to [B, C, H, W]
        if model_pred.dim() == 3:
            B, L, D = model_pred.shape
            H = W = int(L ** 0.5)
            model_pred = model_pred.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return model_pred

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target
