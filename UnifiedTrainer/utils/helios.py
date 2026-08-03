"""
Helios Anti-Drifting Utilities — Frame-Aware Corrupt

Based on Helios paper (Section 3.2.3): perturbs condition/reference latents
during training to simulate "imperfect history", improving model robustness
against color/pixel drift during inference.

Key design:
    - Operates in LATENT space (post-VAE encoding), not pixel space
    - Perturbations are sampled INDEPENDENTLY per sample in the batch
    - Only reference/condition latents are perturbed — target stays clean

Corruption types (v2 — extended for super-resolution / deblurring):
    1. Exposure adjust     — multiply latent by random factor
    2. Gaussian noise      — add Gaussian noise
    3. Strong downsample   — downsample then upsample (simulates low-res)
    4. Gaussian blur       — separable Gaussian convolution
    5. Color blocks        — random color patches (simulates compression artifacts)
    6. Combo blur + jaggy  — pixelation + blur
    7. Clean               — no perturbation (keeps ~20-30% of data clean)

Reference: T2ITrainer/utils/Helios/helios_utils.py (ported to UnifiedTrainer)
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Core perturbation primitives ──────────────────────────────────


def gaussian_blur2d(
    latent: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Separable Gaussian blur in latent space.

    Args:
        latent: [B, C, H, W]
        kernel_size: odd integer ≥ 3
        sigma: Gaussian standard deviation
    """
    if kernel_size < 3:
        return latent

    device = latent.device
    dtype = latent.dtype
    C = latent.shape[1]

    x = torch.arange(kernel_size, dtype=dtype, device=device)
    x = x - kernel_size // 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2 + 1e-8))
    gauss_1d = gauss_1d / (gauss_1d.sum() + 1e-8)

    padding = kernel_size // 2

    # Horizontal pass: [C, 1, 1, kernel_size]
    kernel_h = gauss_1d.view(1, 1, -1).repeat(C, 1, 1).unsqueeze(2)
    latent_h = F.conv2d(latent, kernel_h, padding=(0, padding), groups=C)

    # Vertical pass: [C, 1, kernel_size, 1]
    kernel_v = gauss_1d.view(1, 1, -1).repeat(C, 1, 1).unsqueeze(3)
    latent_blur = F.conv2d(latent_h, kernel_v, padding=(padding, 0), groups=C)

    return latent_blur


def add_jaggy_effect(
    latent: torch.Tensor,
    downsample_range: Tuple[float, float] = (0.25, 0.5),
    blur_range: Tuple[float, float] = (0.0, 1.5),
    quantize_bits: int = 0,
) -> torch.Tensor:
    """Simulate pixelation from low-resolution upscaling."""
    B, C, H, W = latent.shape
    device = latent.device
    dtype = latent.dtype
    latent_out = latent.clone()

    for b in range(B):
        scale = torch.empty(1, device=device, dtype=dtype).uniform_(
            downsample_range[0], downsample_range[1]
        ).item()
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))

        latent_down = F.interpolate(
            latent[b:b + 1], size=(new_h, new_w), mode="area"
        )
        latent_up = F.interpolate(
            latent_down, size=(H, W), mode="nearest"
        )

        if blur_range[1] > 0:
            blur_sigma = torch.empty(1, device=device, dtype=dtype).uniform_(
                blur_range[0], blur_range[1]
            ).item()
            if blur_sigma > 0:
                kernel_size = int(6 * blur_sigma + 1) | 1
                kernel_size = min(kernel_size, min(H, W) // 2 * 2 - 1)
                kernel_size = max(3, kernel_size)
                latent_up = gaussian_blur2d(latent_up, kernel_size, blur_sigma)

        if quantize_bits > 0:
            levels = 2 ** quantize_bits
            latent_up = torch.round(latent_up * levels) / levels

        latent_out[b:b + 1] = latent_up

    return latent_out


def add_color_blocks(
    latent: torch.Tensor,
    num_blocks_range: Tuple[int, int] = (1, 5),
    block_size_range: Tuple[int, int] = (4, 16),
    intensity_range: Tuple[float, float] = (0.3, 0.7),
    block_type: str = "random",
) -> torch.Tensor:
    """Add random color patches to simulate compression artifacts."""
    B, C, H, W = latent.shape
    device = latent.device
    dtype = latent.dtype
    latent_out = latent.clone()

    for b in range(B):
        num_blocks = torch.randint(
            num_blocks_range[0], num_blocks_range[1] + 1, (1,)
        ).item()

        for _ in range(num_blocks):
            block_h = torch.randint(
                block_size_range[0], block_size_range[1] + 1, (1,)
            ).item()
            block_w = torch.randint(
                block_size_range[0], block_size_range[1] + 1, (1,)
            ).item()
            y = torch.randint(0, max(1, H - block_h), (1,)).item()
            x = torch.randint(0, max(1, W - block_w), (1,)).item()

            block_h = min(block_h, H - y)
            block_w = min(block_w, W - x)
            if block_h <= 0 or block_w <= 0:
                continue

            intensity = torch.empty(1, device=device, dtype=dtype).uniform_(
                *intensity_range
            ).item()

            if block_type == "random":
                random_color = torch.randn(C, device=device, dtype=dtype) * intensity
                latent_out[b, :, y:y + block_h, x:x + block_w] += random_color.view(C, 1, 1)
            elif block_type == "mean":
                local_mean = latent[b, :, y:y + block_h, x:x + block_w].mean(dim=(1, 2), keepdim=True)
                latent_out[b, :, y:y + block_h, x:x + block_w] = (
                    latent_out[b, :, y:y + block_h, x:x + block_w] * (1 - intensity)
                    + local_mean * intensity
                )
            elif block_type == "jpeg":
                y_aligned = (y // 8) * 8
                x_aligned = (x // 8) * 8
                block_h_jpeg = min(8, H - y_aligned)
                block_w_jpeg = min(8, W - x_aligned)
                if block_h_jpeg > 0 and block_w_jpeg > 0:
                    block_mean = latent[
                        b, :, y_aligned:y_aligned + block_h_jpeg,
                        x_aligned:x_aligned + block_w_jpeg
                    ].mean()
                    quantized = torch.round(block_mean * 8) / 8
                    latent_out[
                        b, :, y_aligned:y_aligned + block_h_jpeg,
                        x_aligned:x_aligned + block_w_jpeg
                    ] = (
                        latent_out[
                            b, :, y_aligned:y_aligned + block_h_jpeg,
                            x_aligned:x_aligned + block_w_jpeg
                        ] * (1 - intensity)
                        + quantized * intensity
                    )
            elif block_type == "shift":
                shift_color = torch.zeros(C, device=device, dtype=dtype)
                shift_channel = torch.randint(0, C, (1,)).item()
                shift_color[shift_channel] = intensity * (
                    1 if torch.rand(1) > 0.5 else -1
                )
                latent_out[b, :, y:y + block_h, x:x + block_w] += shift_color.view(C, 1, 1)

    return latent_out


# ── Frame-Aware Corrupt (main entry point) ────────────────────────


def frame_aware_corrupt(
    latent: torch.Tensor,
    prob_exposure: float = 0.15,
    prob_noise: float = 0.15,
    prob_downsample: float = 0.15,
    prob_gaussian_blur: float = 0.15,
    prob_color_blocks: float = 0.10,
    prob_combo_blur_jaggy: float = 0.0,
    prob_clean: float = 0.30,
    exposure_range: Tuple[float, float] = (0.5, 1.5),
    noise_range: Tuple[float, float] = (0.0, 0.1),
    downsample_range: Tuple[float, float] = (0.25, 0.75),
    gaussian_blur_range: Tuple[float, float] = (0.5, 3.0),
    color_block_num_range: Tuple[int, int] = (1, 5),
    color_block_size_range: Tuple[int, int] = (4, 16),
    color_block_intensity_range: Tuple[float, float] = (0.3, 0.7),
    color_block_type: str = "random",
    jaggy_downsample_range: Tuple[float, float] = (0.25, 0.5),
    jaggy_blur_range: Tuple[float, float] = (0.0, 1.5),
) -> torch.Tensor:
    """Apply Helios Frame-Aware Corrupt to a latent tensor.

    Supports [C, H, W], [B, C, H, W], and [B, C, T, H, W] shapes.
    Each sample in the batch independently receives one perturbation type.
    """
    input_shape = latent.shape
    original_ndim = len(input_shape)

    if original_ndim == 3:
        latent = latent.unsqueeze(0)
        need_squeeze = True
    elif original_ndim == 4:
        need_squeeze = False
    elif original_ndim == 5:
        need_squeeze = False
    else:
        raise ValueError(
            f"Unsupported latent shape: {input_shape}. "
            f"Expected [C,H,W], [B,C,H,W], or [B,C,T,H,W]"
        )

    input_shape = latent.shape
    is_video = original_ndim == 5

    if is_video:
        B, C, T, H, W = input_shape
        latent = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        B, C, H, W = input_shape

    device = latent.device
    dtype = latent.dtype

    # Normalize probabilities if they don't sum to 1.0
    total_prob = (
        prob_exposure + prob_noise + prob_downsample
        + prob_gaussian_blur + prob_color_blocks
        + prob_combo_blur_jaggy + prob_clean
    )
    if abs(total_prob - 1.0) > 0.01:
        prob_exposure /= total_prob
        prob_noise /= total_prob
        prob_downsample /= total_prob
        prob_gaussian_blur /= total_prob
        prob_color_blocks /= total_prob
        prob_combo_blur_jaggy /= total_prob
        prob_clean /= total_prob

    N = B * T if is_video else B
    rand = torch.rand(N, 1, 1, 1, device=device)
    latent_out = latent.clone()

    # Probability boundaries
    p1 = prob_exposure
    p2 = p1 + prob_noise
    p3 = p2 + prob_downsample
    p4 = p3 + prob_gaussian_blur
    p5 = p4 + prob_color_blocks
    p6 = p5 + prob_combo_blur_jaggy

    # 1. Exposure Adjust
    mask_exp = rand < p1
    if mask_exp.any():
        alpha = torch.where(
            mask_exp,
            torch.empty_like(mask_exp, dtype=dtype).uniform_(*exposure_range),
            torch.ones_like(mask_exp, dtype=dtype),
        )
        latent_out = latent_out * alpha

    # 2. Gaussian Noise
    mask_noise = (rand >= p1) & (rand < p2)
    if mask_noise.any():
        sigma = torch.where(
            mask_noise,
            torch.empty_like(mask_noise, dtype=dtype).uniform_(*noise_range),
            torch.zeros_like(mask_noise, dtype=dtype),
        )
        latent_out = latent_out + torch.randn_like(latent) * sigma

    # 3. Strong Downsample
    mask_ds = (rand >= p2) & (rand < p3)
    if mask_ds.any():
        scale = torch.empty(1, device=device, dtype=dtype).uniform_(
            downsample_range[0], downsample_range[1]
        ).item()
        latent_down = F.interpolate(
            latent, scale_factor=scale, mode="bilinear", align_corners=False
        )
        latent_down = F.interpolate(
            latent_down, size=(H, W), mode="bilinear", align_corners=False
        )
        latent_out = torch.where(mask_ds, latent_down, latent_out)

    # 4. Gaussian Blur
    mask_blur = (rand >= p3) & (rand < p4)
    if mask_blur.any():
        sigma = torch.empty(1, device=device, dtype=dtype).uniform_(
            gaussian_blur_range[0], gaussian_blur_range[1]
        ).item()
        kernel_size = int(6 * sigma + 1) | 1
        kernel_size = min(kernel_size, min(H, W) // 2 * 2 - 1)
        kernel_size = max(3, kernel_size)
        for i in range(latent.shape[0]):
            if mask_blur[i, 0, 0, 0]:
                latent_out[i:i + 1] = gaussian_blur2d(
                    latent[i:i + 1], kernel_size, sigma
                )

    # 5. Color Blocks
    mask_color = (rand >= p4) & (rand < p5)
    if mask_color.any():
        for i in range(latent.shape[0]):
            if mask_color[i, 0, 0, 0]:
                latent_out[i:i + 1] = add_color_blocks(
                    latent_out[i:i + 1],
                    num_blocks_range=color_block_num_range,
                    block_size_range=color_block_size_range,
                    intensity_range=color_block_intensity_range,
                    block_type=color_block_type,
                )

    # 6. Combo Blur + Jaggy
    mask_jaggy = (rand >= p5) & (rand < p6)
    if mask_jaggy.any():
        for i in range(latent.shape[0]):
            if mask_jaggy[i, 0, 0, 0]:
                latent_out[i:i + 1] = add_jaggy_effect(
                    latent[i:i + 1],
                    downsample_range=jaggy_downsample_range,
                    blur_range=jaggy_blur_range,
                )

    # 7. Clean: no operation (p6 <= rand < 1.0)

    if is_video:
        latent_out = latent_out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    if need_squeeze:
        latent_out = latent_out.squeeze(0)

    return latent_out


# ── Config parsing ────────────────────────────────────────────────


def parse_helios_config(config: Dict) -> Dict:
    """Parse Helios config from a JSON config dict.

    Expected config structure (under top-level "helios" key):
        {
            "enabled": true,
            "prob_exposure": 0.15,
            "prob_noise": 0.15,
            ...
        }

    Returns a flat dict of kwargs suitable for frame_aware_corrupt().
    """
    helios_cfg = config.get("helios", {})
    if not helios_cfg.get("enabled", False):
        return {"enabled": False}

    def _range(key, default):
        val = helios_cfg.get(key, list(default))
        if isinstance(val, list):
            return tuple(val)
        return tuple(default)

    def _irange(key, default):
        val = helios_cfg.get(key, list(default))
        if isinstance(val, list):
            return tuple(int(v) for v in val)
        return tuple(default)

    return {
        "enabled": True,
        "prob_exposure": helios_cfg.get("prob_exposure", 0.15),
        "prob_noise": helios_cfg.get("prob_noise", 0.15),
        "prob_downsample": helios_cfg.get("prob_downsample", 0.15),
        "prob_gaussian_blur": helios_cfg.get("prob_gaussian_blur", 0.15),
        "prob_color_blocks": helios_cfg.get("prob_color_blocks", 0.10),
        "prob_combo_blur_jaggy": helios_cfg.get("prob_combo_blur_jaggy", 0.0),
        "prob_clean": helios_cfg.get("prob_clean", 0.30),
        "exposure_range": _range("exposure_range", (0.5, 1.5)),
        "noise_range": _range("noise_range", (0.0, 0.1)),
        "downsample_range": _range("downsample_range", (0.25, 0.75)),
        "gaussian_blur_range": _range("gaussian_blur_range", (0.5, 3.0)),
        "color_block_num_range": _irange("color_block_num_range", (1, 5)),
        "color_block_size_range": _irange("color_block_size_range", (4, 16)),
        "color_block_intensity_range": _range(
            "color_block_intensity_range", (0.3, 0.7)
        ),
        "color_block_type": helios_cfg.get("color_block_type", "random"),
        "jaggy_downsample_range": _range("jaggy_downsample_range", (0.25, 0.5)),
        "jaggy_blur_range": _range("jaggy_blur_range", (0.0, 1.5)),
    }


def apply_helios_corrupt(
    latent: torch.Tensor,
    config: Dict,
) -> torch.Tensor:
    """Apply Helios corrupt to a latent tensor using a parsed config dict.

    Args:
        latent: Reference/condition latent [B, C, H, W]
        config: Parsed config from parse_helios_config()

    Returns:
        Corrupted latent (same shape as input), or original if disabled.
    """
    if not config.get("enabled", False):
        return latent

    kwargs = {k: v for k, v in config.items() if k != "enabled"}
    return frame_aware_corrupt(latent, **kwargs)
