"""
BucketSystem — aspect-ratio bucketing with divisibility awareness.

Image dimensions must be divisible by the model's bucket_divisibility
(vae_scale_factor * patch_size). Buckets are generated dynamically from
the target aspect ratios and divisibility, so they are always valid.

Models can still override with a custom resolution_config for full control.

Usage:
    # Auto-generate from divisibility (recommended)
    bs = BucketSystem(divisibility=16)  # Qwen/Krea2: vae=8, patch=2
    bucket = bs.find_bucket(512, 1.5)   # -> (640, 384)

    # Custom buckets (overrides auto-generation)
    bs = BucketSystem(resolution_config={512: [(512, 512), ...]})
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image


# Standard aspect ratios to generate buckets for (width/height).
# Covers common photography and digital art ratios.
# Each entry is (numerator, denominator) to avoid float precision issues.
STANDARD_RATIOS = [
    (1, 1),     # square
    (4, 3),     # landscape
    (3, 4),     # portrait
    (3, 2),     # landscape (DSLR)
    (2, 3),     # portrait (DSLR)
    (16, 9),    # widescreen
    (9, 16),    # vertical (phone)
    (16, 10),   # widescreen monitor
    (10, 16),   # vertical monitor
    (21, 9),    # ultrawide
    (9, 21),    # vertical ultrawide
]


def _generate_buckets(
    base_resolution: int,
    divisibility: int,
    ratios: list = None,
) -> List[Tuple[int, int]]:
    """Generate valid bucket dimensions for a base resolution and divisibility.

    For each target aspect ratio r = w_num / w_den:
     - target_pixels = base_resolution^2
     - w = round(sqrt(target_pixels * r) / div) * div
     - h = round(sqrt(target_pixels / r) / div) * div

    Ensures w * h is close to base_resolution^2 and both are divisible.
    """
    if ratios is None:
        ratios = STANDARD_RATIOS

    target_pixels = base_resolution * base_resolution
    buckets = []
    seen = set()

    for w_num, w_den in ratios:
        aspect = w_num / w_den
        # Compute ideal dimensions at this resolution
        ideal_w = (target_pixels * aspect) ** 0.5
        ideal_h = (target_pixels / aspect) ** 0.5

        # Snap to divisibility
        w = max(divisibility, round(ideal_w / divisibility) * divisibility)
        h = max(divisibility, round(ideal_h / divisibility) * divisibility)

        key = (w, h)
        if key not in seen:
            seen.add(key)
            buckets.append((w, h))

    # Sort by area then aspect ratio for readability
    buckets.sort(key=lambda wh: (wh[0] * wh[1], wh[0] / wh[1]))
    return buckets


class BucketSystem:
    """Aspect-ratio bucket system with divisibility awareness.

    Priority for bucket lookup:
      1. Custom resolution_config entries (if the base_resolution key exists)
      2. Auto-generated buckets from divisibility

    Args:
        divisibility: Image dims must be divisible by this (vae_scale * patch_size).
        resolution_config: Optional override: {base_resolution: [(w, h), ...]}.
        ratios: Custom aspect ratios for auto-generation (default: STANDARD_RATIOS).
    """

    def __init__(
        self,
        divisibility: int = 8,
        resolution_config: dict = None,
        ratios: list = None,
    ):
        self.divisibility = divisibility
        self._custom_config = resolution_config
        self._ratios = ratios
        self._cache: dict = {}

    def get_buckets(self, base_resolution: int) -> List[Tuple[int, int]]:
        """Return the list of (width, height) buckets for a base resolution.

        Uses custom config if available, otherwise auto-generates from divisibility.
        """
        if base_resolution in self._cache:
            return self._cache[base_resolution]

        # Priority 1: explicit custom config
        if self._custom_config and base_resolution in self._custom_config:
            buckets = self._custom_config[base_resolution]
        else:
            # Priority 2: auto-generate
            buckets = _generate_buckets(
                base_resolution, self.divisibility, self._ratios
            )

        self._cache[base_resolution] = buckets
        return buckets

    def find_bucket(
        self, base_resolution: int, aspect_ratio: float
    ) -> Tuple[int, int]:
        """Find the nearest bucket for a given aspect ratio.

        aspect_ratio = width / height
        """
        buckets = self.get_buckets(base_resolution)
        best = buckets[0]
        best_diff = float("inf")
        for w, h in buckets:
            bucket_ratio = w / h
            diff = abs(bucket_ratio - aspect_ratio)
            if diff < best_diff:
                best_diff = diff
                best = (w, h)
        return best

    def find_bucket_for_image(
        self, base_resolution: int, image: Image.Image
    ) -> Tuple[int, int]:
        """Find the nearest bucket for a PIL image."""
        w, h = image.size
        return self.find_bucket(base_resolution, w / h)

    def find_bucket_for_size(
        self, base_resolution: int, width: int, height: int
    ) -> Tuple[int, int]:
        """Find the nearest bucket for explicit width/height."""
        return self.find_bucket(base_resolution, width / height)

    def crop_to_bucket(
        self, image: Image.Image, bucket: Tuple[int, int]
    ) -> Image.Image:
        """Center-crop a PIL image to exact bucket dimensions."""
        target_w, target_h = bucket
        src_w, src_h = image.size

        # Scale image so both dimensions >= bucket dimensions
        scale = max(target_w / src_w, target_h / src_h)
        scaled_w = int(round(src_w * scale))
        scaled_h = int(round(src_h * scale))
        image = image.resize((scaled_w, scaled_h), Image.LANCZOS)

        # Center crop
        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))

    def crop_numpy_to_bucket(
        self, arr: np.ndarray, bucket: Tuple[int, int]
    ) -> np.ndarray:
        """Center-crop a numpy HWC array to exact bucket dimensions."""
        target_w, target_h = bucket
        src_h, src_w = arr.shape[:2]

        scale = max(target_w / src_w, target_h / src_h)
        scaled_w = int(round(src_w * scale))
        scaled_h = int(round(src_h * scale))

        # Resize via PIL for quality
        pil = Image.fromarray(arr)
        pil = pil.resize((scaled_w, scaled_h), Image.LANCZOS)
        arr = np.array(pil)

        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        return arr[top : top + target_h, left : left + target_w]
