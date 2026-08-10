"""
BaseModelAdapter — the protocol that every model backend must implement.

All model-specific behavior (loading, encoding, forward pass, prediction unpacking,
target computation) lives here. The training engine is model-agnostic and calls
only these methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters.

    A model adapter encapsulates everything that differs between diffusion model
    architectures: how to load components, encode images/text, build the forward
    pass input, unpack the model prediction, and compute the training target.

    The shared Trainer never accesses model internals directly — it always goes
    through the adapter.

    Optional video-validation hook (documented here, deliberately NOT defined as
    a base method so ``hasattr``-based dispatch keeps working):

    - ``decode_validation_video(vae, latent, output_dir, prefix="") -> list[str]``
      — implemented by video-capable adapters (MiniMax-H3, P2).  Decodes a 5D
      ``(B, C, T, H, W)`` latent with ``T > 1`` to one silent h264 mp4 per batch
      item (no audio track) and returns the mp4 paths.  The engine's validation
      loop dispatches via ``hasattr(self.adapter, "decode_validation_video")``
      plus a ``latent.ndim == 5 and latent.shape[2] > 1`` guard: adapters that
      do not define the hook keep the legacy ``decode_latent`` -> PIL image path
      unchanged, so the *absence* of this method is the default implementation.
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

    @property
    def supports_video(self) -> bool:
        """Whether this adapter consumes unified 5D (B, C, T, H, W) latents.

        The unified media pipeline stores every latent as (C, T, H, W) — an
        image is simply (C, 1, H, W).  Legacy image adapters (krea2, ...)
        keep receiving 4D (B, C, H, W) runtime tensors: the dataset squeezes
        the singleton temporal dim before collate.  Video-capable adapters
        (MiniMax-H3, P1.4) override this to ``True`` so the full 5D path is
        preserved — P2 video samples only change T, no data-layer changes.
        """
        return False

    @property
    def velocity_sign(self) -> str:
        """Velocity parameterization sign convention.

        "standard": velocity points from data toward noise
        (flow target = noise - x0), matching most flow-matching trainers
        (Flux, Krea2).
        "data_ward": velocity points from noise toward data
        (MiniMax-H3 scheduler convention, flow target = x0 - noise).
        """
        return "standard"

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

    def encode_video(self, vae: nn.Module, frames: torch.Tensor) -> dict:
        """Encode a 5D frame tensor ``(B=1, C, T, H, W)`` to latent space.

        Default implementation — single-frame images (``T == 1``):
        The 5D [0, 1] pixel frames are converted back to a PIL image (the
        input convention the cache pipeline historically fed ``encode_image``:
        a diffusion-normalized 4D tensor derived from a PIL image) and the
        call is delegated to ``self.encode_image``.  This keeps every existing
        image adapter's behavior exactly unchanged.

        Video-capable adapters (MiniMax-H3, P1.4) override this method for
        ``T > 1`` and return a dict with ``latent`` of shape ``(C, T, H, W)``
        (B already folded) — or ``(1, C, T, H, W)`` which the cache builder
        folds.

        Args:
            vae: Loaded VAE module.
            frames: ``(1, C, T, H, W)`` float32 pixel frames in [0, 1].

        Returns:
            dict with ``latent`` key (canonical ``(C, T, H, W)`` for the
            default single-frame path).

        Raises:
            ValueError: frames is not ``(B=1, C, T, H, W)``.
            NotImplementedError: when ``T > 1`` (video encoding requires
                adapter support — the MiniMax-H3 adapter overrides this).
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
        if frames.shape[2] != 1:
            raise NotImplementedError(
                "video encoding requires adapter support — override encode_video "
                "for T > 1 (e.g. MiniMax-H3 in P1.4)"
            )

        # ── T == 1: image path — reconstruct encode_image's input convention ──
        # frames (1, C, 1, H, W) float32 [0,1] → PIL RGB → to_tensor_universal
        # (diffusion [-1,1]) → (1, C, H, W) — byte-equivalent to what
        # cache_builder previously fed encode_image for every existing adapter.
        from PIL import Image

        from UnifiedTrainer.data.transforms import to_tensor_universal

        arr = (
            frames[0, :, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        ).round().astype(np.uint8)
        pil_image = Image.fromarray(arr)
        image_tensor = to_tensor_universal(np.array(pil_image)).unsqueeze(0)
        # Legacy cache pipeline moved the tensor to the VAE device/dtype; the
        # VAE's own dtype attribute (diffusers ModelMixin) is the same one the
        # old `_cache_image` used.  Defensive fallbacks keep plain nn.Module
        # (and stub) VAEs working.
        try:
            vae_dtype = vae.dtype
        except (AttributeError, TypeError):
            vae_dtype = torch.float32
        try:
            vae_device = next(vae.parameters()).device
        except (AttributeError, TypeError, StopIteration):
            vae_device = frames.device
        image_tensor = image_tensor.to(device=vae_device, dtype=vae_dtype)

        latent_dict = self.encode_image(vae, image_tensor)
        latent = latent_dict["latent"]

        # Normalize to the canonical (C, T, H, W) contract: fold B, keep T=1.
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # (C, T, H, W)
        elif latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0).unsqueeze(1)  # (C, H, W) → (C, 1, H, W)
        elif latent.ndim == 3:
            latent = latent.unsqueeze(1)  # (C, H, W) → (C, 1, H, W)
        return {"latent": latent}

    @property
    def encode_text_accepts_image(self) -> bool:
        """Whether ``encode_text`` accepts a ``condition_image`` keyword argument.

        MiniMax-H3 (P1.4) overrides this to ``True``: its caption encoding
        builds a vision-block presentation from the associated keyframe image,
        so the cache builder passes the keyframe PIL as ``condition_image``.

        Default ``False`` — existing adapters keep their exact ``encode_text``
        contract (the cache builder never adds the extra keyword).
        """
        return False

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

    def compute_x0_hat(
        self, noise: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the clean latent from a predicted velocity.

        Standard (noise-ward): v = noise - x0  → x0_hat = noise - v.
        Data-ward (MiniMax-H3): v = x0 - noise → x0_hat = noise + v.

        符号约定与 ``velocity_sign`` / ``compute_target`` /
        ``losses/flow_matching.py`` 同源——trainer 不得写死公式（写反会静默
        训错方向：loss 下降但模型发散）。MiniMaxH3Adapter 继承默认实现即正确。
        """
        if self.velocity_sign == "data_ward":
            return noise + velocity
        return noise - velocity

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
