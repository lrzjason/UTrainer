"""Qwen-Image-Edit model adapter -image editing with reference conditioning."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

RESOLUTION_CONFIG = {
    2048: [
        (1024, 2560), (1152, 2816), (1280, 3072), (1280, 2816),
        (1408, 3008), (1536, 2816), (1664, 2688), (1792, 2368),
        (1920, 2240), (2048, 2560), (2176, 2304), (2048, 2048),
        (2048, 2304),
    ],
}


@ModelRegistry.register("qwen_image_edit")
class QwenImageEditAdapter(BaseModelAdapter):
    """Adapter for Qwen-Image-Edit image editing model."""

    name = "qwen_image_edit"
    patch_size = 2  # Qwen transformer patches latents in 2x2 blocks

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from ..qwen_image.transformer_qwenimage import BlockSwapQwenImageTransformer2DModel
        return BlockSwapQwenImageTransformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import AutoencoderKLQwenImage
        return AutoencoderKLQwenImage.from_pretrained(path, torch_dtype=dtype)

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
            # AutoencoderKLQwenImage is a video VAE expecting 5D input (B, C, T, H, W).
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(2)
            latent = vae.encode(image_tensor).latent_dist.sample()
            # Squeeze temporal dim back to 4D (B, C, H, W)
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            # Qwen VAE uses per-channel latents_mean/latents_std, not shift_factor/scaling_factor.
            latents_mean = torch.tensor(
                vae.config.latents_mean, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latents_std = torch.tensor(
                vae.config.latents_std, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latent = (latent - latents_mean) / latents_std
        return {"latent": latent}

    def encode_text(self, text_encoder, tokenizer, prompt, device, dtype) -> dict:
        return {"prompt": prompt, "device": device}

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        # Denormalize using latents_mean/latents_std (inverse of encode_image).
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latents_std = torch.tensor(
            vae.config.latents_std, device=latent.device, dtype=latent.dtype
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

    def prepare_model_input(
        self, batch: dict, noise: torch.Tensor | list[torch.Tensor], sigmas: torch.Tensor
    ) -> dict:
        latents = batch.get("latents", {})
        # Resolve reference from batch_config — user-defined keys, NOT hardcoded.
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        ref_key = resolved_bc.get("reference_config")
        ref = latents.get(ref_key) if ref_key else None

        # Support single or multiple target tensors (channel-concat design:
        # if multiple targets are passed, concat them along batch dim).
        noises = noise if isinstance(noise, list) else [noise]
        target = noises[0]

        # Qwen-Image-Edit concatenates reference image latent with noise
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
        if model_pred.dim() == 3:
            B, L, D = model_pred.shape
            H = W = int(L ** 0.5)
            model_pred = model_pred.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return model_pred

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target
