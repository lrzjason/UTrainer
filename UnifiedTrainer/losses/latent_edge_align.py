"""
Latent Edge Alignment Loss (B2) — compare structural edge features
between the predicted clean latent (x0_hat) and the depth reference
latent directly in latent space.

Concept (user's idea):
    VAE latent space encodes meaningful spatial structure.  Instead of
    decoding to pixel space (expensive), compute Sobel edge magnitude
    on both x0_hat and reference_latent per-channel, then measure
    alignment.  This encourages the generated image's spatial structure
    to follow the depth map's structure.

Two loss types:
    - "mse": MSE between edge maps (default)
    - "cosine": 1 - cosine_similarity between flattened edge maps

VRAM cost: ~0 (just tensor ops on existing tensors).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


def _sobel_edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute per-channel Sobel edge magnitude.

    Args:
        x: (B, C, H, W) latent tensor.

    Returns:
        (B, C, H, W) edge magnitude (same spatial size via padding).
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)

    B, C, H, W = x.shape
    # Process all channels at once via groups
    x_flat = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_flat, sobel_x, padding=1)
    gy = F.conv2d(x_flat, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return mag.reshape(B, C, H, W)


@LossRegistry.register("latent_edge_align")
class LatentEdgeAlignLoss(BaseLoss):
    """Latent-space edge alignment between x0_hat and depth reference.

    Config example::

        {"type": "latent_edge_align", "weight": 0.3, "params": {
            "edge_type": "sobel",
            "loss_type": "mse",
            "use_timestep_weighting": true,
            "timestep_center": 0.3,
            "timestep_width": 0.15
        }}
    """

    name = "latent_edge_align"

    def __init__(
        self,
        weight: float = 0.3,
        edge_type: str = "sobel",
        loss_type: str = "mse",
        use_timestep_weighting: bool = True,
        timestep_center: float = 0.3,
        timestep_width: float = 0.15,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.edge_type = edge_type
        self.loss_type = loss_type
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
        # while cached reference latents are bf16. mse_loss backward requires
        # matching dtypes ("Found dtype BFloat16 but expected Float").
        x0_hat = x0_hat.float()
        depth_latent = context.reference_latent.detach().float()

        # Shape compatibility
        if x0_hat.shape[-2:] != depth_latent.shape[-2:]:
            depth_latent = F.interpolate(
                depth_latent, size=x0_hat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        # Compute edge maps
        edge_pred = _sobel_edge_magnitude(x0_hat)
        edge_ref = _sobel_edge_magnitude(depth_latent)

        if self.loss_type == "cosine":
            # Flatten spatial dims, compute cosine similarity per sample
            B = edge_pred.shape[0]
            ep = edge_pred.reshape(B, -1)
            er = edge_ref.reshape(B, -1)
            cos_sim = F.cosine_similarity(ep, er, dim=1)
            loss = (1.0 - cos_sim).mean()
        else:
            # MSE between edge maps
            loss = F.mse_loss(edge_pred, edge_ref)

        if self.use_timestep_weighting:
            ts_weight = self._compute_timestep_weight(context.sigmas)
            loss = loss * ts_weight.mean()

        return loss
