"""
Transforms — image preprocessing utilities for the unified data pipeline.

Provides:
   - ToTensorUniversal: PIL/numpy → normalized torch tensor
   - normalize: (x - mean) / std
   - denormalize: inverse of normalize
   - crop_center: center crop to target size
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


def to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    """Convert PIL image or numpy HWC array to CHW float tensor in [0, 1]."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = image
    # HWC → CHW, float32, [0, 1]
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    return tensor


def to_tensor_universal(
    image: Image.Image | np.ndarray,
    mean: list = DIFFUSION_MEAN,
    std: list = DIFFUSION_STD,
) -> torch.Tensor:
    """Convert to tensor and normalize to mean/std range.

    Default: diffusion-style [-1, 1] normalization.
    """
    tensor = to_tensor(image)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
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
