"""
Transforms — image preprocessing utilities for the unified data pipeline.

Provides:
   - ToTensorUniversal: PIL/numpy → normalized torch tensor
   - normalize: (x - mean) / std
   - denormalize: inverse of normalize
   - crop_center: center crop to target size

Alpha/transparency support (Qwen-Image 2.1):
   ``to_tensor`` / ``to_tensor_universal`` accept ``channels=4``, which loads
   the source as RGBA instead of RGB so the alpha channel of PNG/WebP files
   survives into the VAE encoder.  Every channel — RGB *and* alpha — is
   normalized to [-1, 1] with the diffusion constants, matching diffusers'
   ``VaeImageProcessor.normalize`` (``2x - 1`` per channel), which is what
   the official Qwen-Image pipeline feeds its 4-channel VAE.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image


# Common normalization constants (ImageNet-style for RGB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Diffusion-style normalization ([-1, 1] range)
DIFFUSION_MEAN = [0.5, 0.5, 0.5]
DIFFUSION_STD = [0.5, 0.5, 0.5]

# Alpha-channel constants: normalized exactly like RGB (diffusers
# VaeImageProcessor.normalize maps EVERY channel with 2x - 1 — there is no
# per-channel mean/std broadcast, so the alpha channel lands in [-1, 1] too).
DIFFUSION_ALPHA_MEAN = 0.5
DIFFUSION_ALPHA_STD = 0.5


def _pad_constants(values: list, channels: int, fill: float) -> list:
    """Extend a per-channel constants list to ``channels`` entries.

    The alpha channel reuses the diffusion constant (0.5) — identical
    normalization to the RGB channels (pipeline parity).
    """
    if len(values) >= channels:
        return list(values[:channels])
    return list(values) + [fill] * (channels - len(values))


def to_tensor(image: Image.Image | np.ndarray, channels: int = 3) -> torch.Tensor:
    """Convert PIL image or numpy HWC array to CHW float tensor in [0, 1].

    Args:
        channels: 3 loads/keeps RGB; 4 loads/keeps RGBA (alpha preserved —
            see module docstring).  Numpy inputs carry their own channel
            count, so the flag only affects the PIL conversion.
    """
    if isinstance(image, Image.Image):
        mode = "RGBA" if channels == 4 else "RGB"
        arr = np.array(image.convert(mode))
    else:
        arr = image
    # HWC → CHW, float32, [0, 1]
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    return tensor


def to_tensor_universal(
    image: Image.Image | np.ndarray,
    mean: list = DIFFUSION_MEAN,
    std: list = DIFFUSION_STD,
    channels: int = 3,
) -> torch.Tensor:
    """Convert to tensor and normalize to mean/std range.

    Default: diffusion-style [-1, 1] normalization.  With ``channels=4``
    (RGBA), ``mean``/``std`` are padded with the alpha constant (0.5) so the
    alpha channel is normalized to [-1, 1] as well — matching the official
    pipeline's 4-channel VAE input.
    """
    tensor = to_tensor(image, channels=channels)
    n_ch = tensor.shape[0]
    mean_t = torch.tensor(
        _pad_constants(mean, n_ch, DIFFUSION_ALPHA_MEAN)
    ).view(-1, 1, 1)
    std_t = torch.tensor(
        _pad_constants(std, n_ch, DIFFUSION_ALPHA_STD)
    ).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def normalize(
    tensor: torch.Tensor,
    mean: list = DIFFUSION_MEAN,
    std: list = DIFFUSION_STD,
) -> torch.Tensor:
    """Normalize a CHW tensor: (x - mean) / std."""
    mean_t = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def denormalize(
    tensor: torch.Tensor,
    mean: list = DIFFUSION_MEAN,
    std: list = DIFFUSION_STD,
) -> torch.Tensor:
    """Inverse normalization: x * std + mean."""
    mean_t = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
    return tensor * std_t + mean_t


def crop_center(
    image: Image.Image | np.ndarray,
    target_w: int,
    target_h: int,
) -> Image.Image | np.ndarray:
    """Center crop to target dimensions."""
    if isinstance(image, Image.Image):
        w, h = image.size
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))
    else:
        h, w = image.shape[:2]
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        return image[top : top + target_h, left : left + target_w]


def resize_to_fit(
    image: Image.Image,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """Resize image so both dimensions >= target, maintaining aspect ratio."""
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [-1, 1] to PIL image."""
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2 * 255
    arr = tensor.byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)
