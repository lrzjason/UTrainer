"""
Edge-Weighted Depth Loss (B4) — weight depth consistency by edge strength.

Concept:
    Regions with strong depth gradients (object boundaries) are more
    important for depth adherence than flat regions.  This loss computes
    per-pixel MSE between x0_hat and reference_latent, weighted by the
    edge magnitude of the reference latent.

    loss = mean(mse * (1 + alpha * edge_weight))

    where edge_weight is the normalised Sobel magnitude of the reference.

VRAM cost: ~0 (tensor ops on existing tensors).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-channel Sobel edge magnitude, averaged across channels.

    Args:
        x: (B, C, H, W)
    Returns:
        (B, 1, H, W) normalised edge weight map in [0, 1].
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)

    B, C, H, W = x.shape
    x_flat = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_flat, sobel_x, padding=1)
    gy = F.conv2d(x_flat, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    mag = mag.reshape(B, C, H, W)
    # Average across channels
    mag = mag.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    # Normalise to [0, 1] per sample
    B_size = mag.shape[0]
    mag_flat = mag.reshape(B_size, -1)
    mag_min = mag_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    mag_max = mag_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    mag = (mag - mag_min) / (mag_max - mag_min + 1e-8)
    return mag


@LossRegistry.register("edge_weighted_depth")
class EdgeWeightedDepthLoss(BaseLoss):
    """Edge-weighted depth consistency loss.

    Config example::

        {"type": "edge_weighted_depth", "weight": 0.3, "params": {
            "edge_alpha": 2.0,
            "use_timestep_weighting": true,
            "timestep_center": 0.3,
            "timestep_width": 0.15
        }}
    """

    name = "edge_weighted_depth"

    def __init__(
        self,
        weight: float = 0.3,
        edge_alpha: float = 2.0,
        use_timestep_weighting: bool = True,
        timestep_center: float = 0.3,
        timestep_width: float = 0.15,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.edge_alpha = edge_alpha
        self.use_timestep_weighting = use_timestep_weighting
        self.timestep_center = timestep_center
        self.timestep_width = timestep_width

    def requires(self) -> List[str]:
        return ["model_pred", "noise", "sigmas", "reference_latent"]

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

        # Upcast to fp32: x0_hat inherits the transformer output dtype (fp32)
        # while cached reference latents are bf16 — mixed-dtype elementwise
        # ops crash in backward ("Found dtype BFloat16 but expected Float").
        x0_hat = x0_hat.float()
        depth_latent = context.reference_latent.detach().float()

        if x0_hat.shape[-2:] != depth_latent.shape[-2:]:
            depth_latent = F.interpolate(
                depth_latent, size=x0_hat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        # Per-pixel MSE (no reduction yet)
        mse = (x0_hat - depth_latent) ** 2  # (B, C, H, W)

        # Edge weight from reference latent
        edge_weight = _sobel_magnitude(depth_latent)  # (B, 1, H, W)

        # Weighted loss: emphasise edges
        weighted_mse = mse * (1.0 + self.edge_alpha * edge_weight)
        loss = weighted_mse.mean()

        if self.use_timestep_weighting:
            ts_weight = self._compute_timestep_weight(context.sigmas)
            loss = loss * ts_weight.mean()

        return loss
