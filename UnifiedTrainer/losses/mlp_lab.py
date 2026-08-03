"""
MLP-LAB Loss -direct latent→LAB color loss via frozen MLP.

Uses a pre-trained MLP (latent[128] -LAB[3]) to map predicted and target
latents into perceptual LAB color space, then computes weighted MSE per
channel (L, a, b).

Architecture (MLPLarge):
    Linear(128 -56) -LayerNorm -ReLU
    Linear(256 -56) -LayerNorm -ReLU
    Linear(256 -28) -LayerNorm -ReLU
    Linear(128 -28) -LayerNorm -ReLU
    Linear(128 -)   # LAB output, no activation

Loss = scale_L * MSE(L_pred, L_tgt)
     + scale_a * MSE(a_pred, a_tgt)
     + scale_b * MSE(b_pred, b_tgt)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


class _MLPBlock(nn.Module):
    """MLP block: Linear -> LayerNorm -> ReLU."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.linear(x)))


class MLPLarge(nn.Module):
    """Large MLP: latent(input_dim) -> hidden_dims -> LAB(3)."""

    def __init__(self, input_dim: int = 128, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(_MLPBlock(prev_dim, h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@LossRegistry.register("mlp_lab")
class MLPLabLoss(BaseLoss):
    """MLP-based latent→LAB color loss.

    The MLP is frozen (no gradients). Only the LoRA-adapted transformer
    produces gradients through the predicted latent.
    """

    name = "mlp_lab"

    def __init__(
        self,
        weight: float = 1.5,
        checkpoint: str = "",
        scale_lab_l: float = 3.0,
        scale_lab_a: float = 2.0,
        scale_lab_b: float = 3.0,
        weight_transition: str = "soft",  # "soft" or "hard"
        sigma_threshold: float = 0.5,
        weight_softness: float = 0.5,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.scale_lab_l = scale_lab_l
        self.scale_lab_a = scale_lab_a
        self.scale_lab_b = scale_lab_b
        self.weight_transition = weight_transition
        self.sigma_threshold = sigma_threshold
        self.weight_softness = weight_softness

        self._mlp: Optional[MLPLarge] = None
        self._checkpoint = checkpoint
        self._loaded = False

    def _load_mlp(self, device: torch.device, dtype: torch.dtype):
        """Lazily load the frozen MLP-Large model."""
        if self._loaded:
            return
        ckpt = torch.load(self._checkpoint, map_location="cpu", weights_only=False)

        state_dict = ckpt.get("state_dict", ckpt)
        input_dim = ckpt.get("input_dim", 128)
        hidden_dims = ckpt.get("hidden_dims", None)

        self._mlp = MLPLarge(input_dim=input_dim, hidden_dims=hidden_dims)
        self._mlp.load_state_dict(state_dict)
        self._mlp = self._mlp.to(device, dtype)
        self._mlp.eval()
        for param in self._mlp.parameters():
            param.requires_grad_(False)
        self._loaded = True

    def _compute_timestep_weight(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Compute timestep weighting for LAB loss."""
        if self.weight_transition == "hard":
            return (sigmas < self.sigma_threshold).float()
        else:
            # Soft: exp(-sigma^2 / softness) -emphasis at low noise
            return torch.exp(-sigmas.float() ** 2 / self.weight_softness)

    def compute(self, context: LossContext) -> torch.Tensor:
        if context.x0_hat is None:
            context.x0_hat = context.noise - context.model_pred

        device, dtype = context.x0_hat.device, context.x0_hat.dtype
        self._load_mlp(device, dtype)

        # Flatten spatial dims for MLP: [B, C, H, W] -> [B*H*W, C]
        x0_flat = context.x0_hat.flatten(2).transpose(1, 2)  # [B, L, C]
        pred_lab = self._mlp(x0_flat)  # [B, L, 3]

        color_target = context.learning_target
        target_flat = color_target.flatten(2).transpose(1, 2)
        target_lab = self._mlp(target_flat)  # [B, L, 3]

        # Per-channel weighted MSE
        loss_l = F.mse_loss(pred_lab[..., 0], target_lab[..., 0])
        loss_a = F.mse_loss(pred_lab[..., 1], target_lab[..., 1])
        loss_b = F.mse_loss(pred_lab[..., 2], target_lab[..., 2])

        loss = (
            self.scale_lab_l * loss_l
            + self.scale_lab_a * loss_a
            + self.scale_lab_b * loss_b
        )

        # Timestep weighting
        ts_weight = self._compute_timestep_weight(context.sigmas)
        loss = loss * ts_weight.mean()

        return loss

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]
