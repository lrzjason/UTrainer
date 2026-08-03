"""
Frequency-Domain Depth Loss (B5) — emphasise low-frequency alignment.

Concept:
    Global layout (low frequencies) matters more for depth adherence
    than fine texture detail (high frequencies).  This loss computes
    a weighted MSE in FFT space where low-frequency components get
    higher weight than high-frequency ones.

    Weight mask: radial frequency weight where
        w(r) = high_freq_weight + (1 - high_freq_weight) * exp(-r^2 / sigma^2)
    so DC and low-freq → weight ≈ 1.0, high-freq → weight ≈ high_freq_weight.

VRAM cost: ~0 (FFT on existing tensors).
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import List

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


def _radial_freq_weight(
    H: int, W: int,
    high_freq_weight: float = 0.3,
    sigma: float = 0.25,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Create a radial frequency weight mask for 2D FFT.

    Returns:
        (H, W) weight tensor, DC-centered (fftshifted).
    """
    # Frequency coordinates (centered)
    fy = torch.fft.fftfreq(H, device=device, dtype=torch.float32)
    fx = torch.fft.fftfreq(W, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(gx ** 2 + gy ** 2)  # normalised radius [0, ~0.707]

    # Gaussian falloff: low-freq → 1.0, high-freq → high_freq_weight
    weight = high_freq_weight + (1.0 - high_freq_weight) * torch.exp(
        -(r ** 2) / (2 * sigma ** 2)
    )
    if dtype is not None:
        weight = weight.to(dtype=dtype)
    return weight


@LossRegistry.register("freq_depth")
class FreqDepthLoss(BaseLoss):
    """Frequency-domain depth consistency loss.

    Config example::

        {"type": "freq_depth", "weight": 0.2, "params": {
            "high_freq_weight": 0.3,
            "freq_sigma": 0.25,
            "use_timestep_weighting": true,
            "timestep_center": 0.3,
            "timestep_width": 0.15
        }}
    """

    name = "freq_depth"

    def __init__(
        self,
        weight: float = 0.2,
        high_freq_weight: float = 0.3,
        freq_sigma: float = 0.25,
        use_timestep_weighting: bool = True,
        timestep_center: float = 0.3,
        timestep_width: float = 0.15,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.high_freq_weight = high_freq_weight
        self.freq_sigma = freq_sigma
        self.use_timestep_weighting = use_timestep_weighting
        self.timestep_center = timestep_center
        self.timestep_width = timestep_width
        self._freq_weight_cache: dict = {}

    def requires(self) -> List[str]:
        return ["model_pred", "noise", "sigmas", "reference_latent"]

    def _get_freq_weight(self, H: int, W: int, device, dtype) -> torch.Tensor:
        key = (H, W, device, dtype)
        if key not in self._freq_weight_cache:
            self._freq_weight_cache[key] = _radial_freq_weight(
                H, W, self.high_freq_weight, self.freq_sigma, device, dtype
            )
        return self._freq_weight_cache[key]

    def _compute_timestep_weight(self, sigmas: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(
            -((sigmas.float() - self.timestep_center) ** 2) / self.timestep_width
        )
        while weight.dim() < 4:
            weight = weight.unsqueeze(-1)
        return weight

    def compute(self, context: LossContext) -> torch.Tensor:
        if context.reference_latent is None:
            return torch.tensor(0.0, device=context.model_pred.device,
                                dtype=context.model_pred.dtype)

        x0_hat = context.x0_hat
        if x0_hat is None:
            x0_hat = context.noise - context.model_pred

        depth_latent = context.reference_latent.detach()

        if x0_hat.shape[-2:] != depth_latent.shape[-2:]:
            depth_latent = F.interpolate(
                depth_latent, size=x0_hat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        B, C, H, W = x0_hat.shape

        # 2D FFT per channel
        fft_pred = torch.fft.fft2(x0_hat.float())
        fft_ref = torch.fft.fft2(depth_latent.float())

        # Frequency-domain MSE (magnitude)
        diff = fft_pred - fft_ref
        power = (diff.real ** 2 + diff.imag ** 2)  # (B, C, H, W)

        # Radial frequency weight
        freq_w = self._get_freq_weight(H, W, power.device, power.dtype)
        # Broadcast: (H, W) → (1, 1, H, W)
        freq_w = freq_w.unsqueeze(0).unsqueeze(0)

        loss = (power * freq_w).mean()

        # Normalise by spatial size to keep loss scale comparable to MSE
        loss = loss / (H * W)

        if self.use_timestep_weighting:
            ts_weight = self._compute_timestep_weight(context.sigmas)
            loss = loss * ts_weight.mean()

        return loss
