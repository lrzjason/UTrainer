"""
LCS Saturation Loss -masked chroma/lightness saturation constraint.

Computes saturation as chroma / (|lightness| + eps) in LCS space, then
applies a mask that upweights vivid regions (sat > threshold) and
downweights gray regions (sat <= threshold).

Loss = MSE(s_pred, s_gt) with mask:
    s_gt >  threshold (0.15): weight = 1.0 (vivid -enhance)
    s_gt <= threshold (0.15): weight = 0.1 (gray -protect)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.losses.lcs import LCSLoss
from UnifiedTrainer.registry import LossRegistry


@LossRegistry.register("lcs_saturation")
class LCSSaturationLoss(BaseLoss):
    """Masked saturation loss in LCS color subspace."""

    name = "lcs_saturation"

    def __init__(
        self,
        weight: float = 2.0,
        data_path: str = "",
        lcs_dim: int = 6,
        sat_threshold: float = 0.15,
        vivid_weight: float = 1.0,
        gray_weight: float = 0.1,
        use_timestep_weighting: bool = False,
        timestep_center: float = 0.6,
        timestep_width: float = 0.15,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.sat_threshold = sat_threshold
        self.vivid_weight = vivid_weight
        self.gray_weight = gray_weight
        self.use_timestep_weighting = use_timestep_weighting
        self.timestep_center = timestep_center
        self.timestep_width = timestep_width

        # Reuse LCS projection logic
        self._lcs_projector = LCSLoss(
            data_path=data_path,
            lcs_dim=lcs_dim,
        )

    def compute(self, context: LossContext) -> torch.Tensor:
        if context.x0_hat is None:
            context.x0_hat = context.noise - context.model_pred

        pred_lcs = self._lcs_projector.project_to_lcs(context.x0_hat)

        color_target = context.learning_target
        target_lcs = self._lcs_projector.project_to_lcs(color_target)

        # Channel 0 = lightness, Channels 1..N = chromaticity
        l_pred = pred_lcs[..., 0]
        l_gt = target_lcs[..., 0]
        chroma_pred = torch.norm(pred_lcs[..., 1:], dim=-1)
        chroma_gt = torch.norm(target_lcs[..., 1:], dim=-1)

        eps = 1e-6
        s_pred = chroma_pred / (torch.abs(l_pred) + eps)
        s_gt = chroma_gt / (torch.abs(l_gt) + eps)

        # Masked MSE: upweight vivid, downweight gray
        mask = torch.where(
            s_gt > self.sat_threshold,
            torch.full_like(s_gt, self.vivid_weight),
            torch.full_like(s_gt, self.gray_weight),
        )
        loss = (mask * (s_pred - s_gt) ** 2).mean()

        if self.use_timestep_weighting:
            sigma = context.sigmas
            ts_weight = torch.exp(
               -((sigma - self.timestep_center) ** 2) / self.timestep_width
            )
            loss = loss * ts_weight.mean()

        return loss

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]
