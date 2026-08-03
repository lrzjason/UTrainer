"""
L2 Regularization Loss -penalizes large LoRA weights.

Prevents LoRA parameters from growing too large, which can cause
overfitting and instability during inference.

Loss = lambda_l2 * sum(p^2 for p in trainable params)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


@LossRegistry.register("l2_reg")
class L2RegLoss(BaseLoss):
    """L2 regularization on trainable (LoRA) parameters.

    This loss does not use the LossContext -it directly accesses the
    trainable parameters of the model. The trainer must set the model
    reference before calling this loss.
    """

    name = "l2_reg"

    def __init__(self, weight: float = 0.001, **params):
        super().__init__(weight=weight, **params)
        self._model: Optional[nn.Module] = None

    def set_model(self, model: nn.Module):
        """Set the model whose trainable parameters to regularize."""
        self._model = model

    def compute(self, context: LossContext) -> torch.Tensor:
        if self._model is None:
            return torch.tensor(0.0, device=context.model_pred.device,
                                dtype=context.model_pred.dtype)

        l2 = torch.tensor(0.0, device=context.model_pred.device,
                          dtype=torch.float32)
        for param in self._model.parameters():
            if param.requires_grad:
                l2 = l2 + param.float().pow(2).sum()

        return l2

    def requires(self) -> list:
        return ["model_pred"]
