"""
BaseModelAdapter — the protocol that every model backend must implement.

All model-specific behavior (loading, encoding, forward pass, prediction unpacking,
target computation) lives here. The training engine is model-agnostic and calls
only these methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters.

    A model adapter encapsulates everything that differs between diffusion model
    architectures: how to load components, encode images/text, build the forward
    pass input, unpack the model prediction, and compute the training target.

    The shared Trainer never accesses model internals directly — it always goes
    through the adapter.
    """

    # ── Identity ──────────────────────────────────────────────────────

    name: str = "base"

    # ── Model loading ─────────────────────────────────────────────────

    @abstractmethod
    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        """Load the main transformer / U-Net model."""
        ...

    @abstractmethod
    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        """Load the VAE encoder/decoder."""
        ...

    @abstractmethod
    def load_scheduler(self, path: str) -> Any:
        """Load the noise scheduler."""
        ...

    def load_text_encoder(
        self, path: str, dtype: torch.dtype
    ) -> Optional[nn.Module]:
        """Load the text encoder. Override for models that use one."""
        return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        """Load the tokenizer. Override for models that use one."""
        return None

    # ── Architecture specs ────────────────────────────────────────────

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """Number of latent channels (16 for Flux, 4 for SD, 0 for pixel-diffusion)."""
        ...

    @property
    @abstractmethod
    def vae_scale_factor(self) -> int:
        """VAE downscale factor (typically 8)."""
        ...

    @property
    def patch_size(self) -> int:
        """Transformer patch size for latent patchification (default 1).

        Combined with vae_scale_factor, determines bucket divisibility:
        image dimensions must be divisible by vae_scale_factor * patch_size.

        Examples:
            Flux2 Klein: vae=8, patch=1 → divisibility=8
            Qwen/Krea2: vae=8, patch=2 → divisibility=16
        """
        return 1

    @property
    def bucket_divisibility(self) -> int:
        """Image dimensions must be divisible by this value (vae_scale * patch_size)."""
        return self.vae_scale_factor * self.patch_size

    @property
    def embedding_dim(self) -> int:
        """Text encoder output dimension. Override per model."""
        return 0

    @property
    def empty_embedding_suffix(self) -> str:
        """File suffix for the model-specific empty embedding.

        Each model produces different embedding shapes, so the empty-string
        embedding must be stored with a model-specific name to avoid
        collisions when multiple models share the same cache dir.
        """
        return self.name

    @property
    def resolution_config(self) -> dict:
        """Custom bucket config: {base_resolution: [(w, h), ...]}.

        Return empty dict to use auto-generation from bucket_divisibility.
        Override per model only when custom buckets are needed.
        """
        return {}

    @property
    def supports_image_conditioning(self) -> bool:
        """Whether this model takes a reference image as conditioning."""
        return False

    @property
    def is_pixel_diffusion(self) -> bool:
        """True for pixel-space diffusion (no VAE latent encoding)."""
        return False

    # ── Encoding ──────────────────────────────────────────────────────

    @abstractmethod
    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        """Encode an image to latent space. Returns dict with 'latent' key."""
        ...

    @abstractmethod
    def encode_text(
        self,
        text_encoder: Optional[nn.Module],
        tokenizer: Optional[Any],
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        """Encode a text prompt. Returns dict with embedding tensors."""
        ...

    @abstractmethod
    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        """Decode a latent back to image space."""
        ...

    # ── Pipeline ──────────────────────────────────────────────────────

    def build_pipeline(self, components: dict) -> Any:
        """Assemble an inference pipeline from loaded components."""
        raise NotImplementedError

    def get_empty_embedding(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return the empty/unconditional embedding for this model."""
        raise NotImplementedError

    # ── Training hooks ────────────────────────────────────────────────

    @abstractmethod
    def prepare_model_input(
        self, batch: dict, noise: torch.Tensor | list[torch.Tensor], sigmas: torch.Tensor
    ) -> dict:
        """Prepare the kwargs dict passed to transformer.forward().

        This is where model-specific input assembly happens: concatenating
        reference latents, adding timesteps, building attention masks, etc.
        """
        ...

    @abstractmethod
    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Unpack the raw model output into a velocity / noise prediction.

        Some models (e.g. Flux) pack latents into a sequence and need unpacking;
        others return the prediction directly.
        """
        ...

    @abstractmethod
    def compute_target(
        self, noise: torch.Tensor, learning_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the velocity-matching target.

        For flow-matching: target = noise - learning_target
        Models with different parameterizations override this.
        """
        ...

    # ── Timestep sampling ─────────────────────────────────────────────

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps and sigmas for flow-matching.

        Default: logit-normal distribution (sigma = sigmoid(N(0,1))).
        Override per-model for custom distributions (e.g. Krea2 mu-shift).

        Args:
            batch_size: Number of samples.
            device: Target device.
            dtype: Output dtype.
            latent_height: Latent grid height (may be used by model-specific sampling).
            latent_width: Latent grid width (may be used by model-specific sampling).

        Returns:
            Tuple of (timesteps, sigmas) — both shape (batch_size,).
        """
        u = torch.randn(batch_size, device=device, dtype=torch.float32)
        sigmas = torch.sigmoid(u).to(dtype=dtype)
        timesteps = sigmas  # default: sigma is the timestep
        return timesteps, sigmas
