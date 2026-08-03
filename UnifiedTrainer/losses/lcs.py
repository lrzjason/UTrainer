"""
LCS Loss -Latent Color Subspace projection MSE.

Projects predicted and target latents into a color subspace via PCA basis,
then computes MSE in that subspace. This constrains the color distribution
of generated images without penalizing spatial structure.

Projection: (latents - mean) @ basis -[B, L, lcs_dim]
Loss: MSE(pred_lcs, target_lcs) * timestep_weight
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


@LossRegistry.register("lcs")
class LCSLoss(BaseLoss):
    """LCS projection MSE loss.

    Requires LCS calibration data (basis + mean) from a safetensors file.
    The color target is the learning_target.
    """

    name = "lcs"

    def __init__(
        self,
        weight: float = 0.5,
        data_path: str = "",
        lcs_dim: int = 6,
        use_timestep_weighting: bool = False,
        timestep_center: float = 0.6,
        timestep_width: float = 0.15,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.data_path = data_path
        self.lcs_dim = lcs_dim
        self.use_timestep_weighting = use_timestep_weighting
        self.timestep_center = timestep_center
        self.timestep_width = timestep_width

        self._basis: Optional[torch.Tensor] = None
        self._mean: Optional[torch.Tensor] = None
        self._loaded = False

    def _load_data(self, device: torch.device, dtype: torch.dtype):
        """Lazily load LCS basis and mean from safetensors."""
        if self._loaded:
            return
        from safetensors.torch import load_file as load_safetensors

        data = load_safetensors(self.data_path)
        self._basis = data["basis"].to(device, dtype)  # [D, lcs_dim]
        self._mean = data["mean"].to(device, dtype)    # [D]

        # Auto-detect dimension
        if self._basis.shape[1] in (3, 6):
            self.lcs_dim = self._basis.shape[1]
        self._loaded = True

    def project_to_lcs(self, latents: torch.Tensor) -> torch.Tensor:
        """Project latents [B, C, H, W] or [B, L, D] to LCS space [B, L, lcs_dim]."""
        device, dtype = latents.device, latents.dtype
        self._load_data(device, dtype)

        original_shape = latents.shape
        if len(original_shape) == 4:
            B, C, H, W = original_shape
            latents = latents.permute(0, 2, 3, 1).reshape(B, -1, C)
        elif len(original_shape) == 5:
            B, C, T, H, W = original_shape
            latents = latents.permute(0, 3, 4, 1, 2).reshape(B, -1, C)

        # (latents - mean) @ basis
        projection = (latents - self._mean.unsqueeze(0).unsqueeze(0)) @ self._basis
        return projection

    def compute(self, context: LossContext) -> torch.Tensor:
        if context.x0_hat is None:
            context.x0_hat = context.noise - context.model_pred

        pred_lcs = self.project_to_lcs(context.x0_hat)

        # Color target is the learning target
        color_target = context.learning_target
        target_lcs = self.project_to_lcs(color_target)

        loss = torch.nn.functional.mse_loss(pred_lcs, target_lcs)

        if self.use_timestep_weighting:
            sigma = context.sigmas
            ts_weight = torch.exp(
               -((sigma - self.timestep_center) ** 2) / self.timestep_width
            )
            loss = loss * ts_weight.mean()

        return loss

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]
