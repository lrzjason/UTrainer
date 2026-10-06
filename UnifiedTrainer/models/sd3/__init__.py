"""Stable Diffusion 3 model adapter -text-to-image with 16-channel latents."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

RESOLUTION_CONFIG = {
    1024: [
        (1024, 1024), (1120, 960), (1152, 896), (1216, 832),
        (1280, 768), (1344, 704), (1408, 640), (1472, 576),
        (1536, 544), (1600, 512), (1664, 480), (1728, 448),
        (1792, 416), (1856, 384), (1920, 352), (1984, 320),
    ],
}


@ModelRegistry.register("sd3")
class SD3Adapter(BaseModelAdapter):
    """Adapter for Stable Diffusion 3 text-to-image model."""

    name = "sd3"
    patch_size = 2  # SD3Transformer2DModel patches latents in 2x2 blocks

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import SD3Transformer2DModel
        return SD3Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_pretrained(path)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import CLIPTextModelWithProjection
            return CLIPTextModelWithProjection.from_pretrained(path, torch_dtype=dtype)
        except Exception:
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        try:
            from transformers import CLIPTokenizer
            return CLIPTokenizer.from_pretrained(path)
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
        return 2048

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return False

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
        target = noise[0] if isinstance(noise, list) else noise
        timesteps = (sigmas * 1000).long()
        return {
            "hidden_states": target,
            "timestep": timesteps,
        }

    def unpack_prediction(self, model_pred: torch.Tensor, input_ids=None) -> torch.Tensor:
        return model_pred

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target
