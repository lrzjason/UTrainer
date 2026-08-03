"""
Flow-Matching Loss -standard velocity prediction MSE.

This is the base loss, always active in flow-matching diffusion training.
target = noise - learning_target
loss   = mean(weighting * (model_pred - target)^2)
"""
from __future__ import annotations

import torch

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


def compute_loss_weighting_for_sd3(sigmas: torch.Tensor) -> torch.Tensor:
    """SD3-style loss weighting based on noise level (sigma).

    Weighting = 1 / (sigma^2 + 1) — upweights low-noise (high-signal) steps.

    NOTE: this is a mild, nonstandard variant (true SD3/diffusers schemes are
    sigma_sqrt = 1/sigma^2 or cosmap; T2ITrainer's known-good Krea2 runs used
    uniform weighting, i.e. no weighting at all). For few-step turbo inference
    (e.g. 8-step Krea2 turbo), halving the gradient at high-sigma steps — where
    most of the inference trajectory lives — risks under-training the global
    structure phase and leaving residual noise. Prefer use_weighting=false
    (T2ITrainer parity) unless deliberately ablating.
    """
    return 1.0 / (sigmas.float() ** 2 + 1.0)


@LossRegistry.register("flow_matching")
class FlowMatchingLoss(BaseLoss):
    """Standard flow-matching velocity MSE loss."""

    name = "flow_matching"

    def __init__(self, weight: float = 1.0, use_weighting: bool = True, **params):
        super().__init__(weight=weight, **params)
        self.use_weighting = use_weighting

    def compute(self, context: LossContext) -> torch.Tensor:
        target = context.noise - context.learning_target

        if self.use_weighting:
            weighting = compute_loss_weighting_for_sd3(context.sigmas)
            # Expand weighting to match pred shape
            while weighting.dim() < context.model_pred.dim():
                weighting = weighting.unsqueeze(-1)
            loss = (weighting * (context.model_pred - target) ** 2).mean()
        else:
            loss = ((context.model_pred - target) ** 2).mean()

        return loss

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]
