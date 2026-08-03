"""HiDream model adapter -text-to-image with 256 base resolution buckets."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

RESOLUTION_CONFIG = {
    256: [
        (256, 256), (384, 256), (288, 224), (304, 256),
        (304, 208), (512, 256), (512, 288), (512, 320),
    ],
    512: [
        (512, 512), (768, 512), (576, 448), (608, 512),
        (608, 416), (1024, 512), (1024, 576), (1024, 640),
    ],
    1024: [
        (1024, 1024), (1344, 1024), (1152, 896), (1216, 832),
        (1344, 768), (1536, 640), (1536, 1024), (2048, 1024),
    ],
    2048: [
        (2048, 2048), (2304, 1792), (2432, 1664), (2688, 1536),
        (3072, 1280),
    ],
}


@ModelRegistry.register("hidream")
class HiDreamAdapter(BaseModelAdapter):
    """Adapter for HiDream text-to-image model."""

    name = "hidream"

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import Transformer2DModel
        return Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_pretrained(path)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        except Exception:
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(path)
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
        return 4096

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
