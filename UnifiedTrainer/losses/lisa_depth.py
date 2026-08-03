"""
LISA Depth Consistency Loss -Likelihood-Score-Inspired depth alignment.

Inspired by LISA (Likelihood Score Alignment, arXiv:2606.27192v1), this loss
regularises the model's predicted clean latent (x0_hat) to be structurally
consistent with the depth-reference latent.

Original LISA aligns a *side-network feature* with an approximated likelihood
score  ∇_{x_t} log p_t(c|x_t).  Krea-2 has no separate side network -the depth
condition enters via VAE-latent channel concatenation into a single MMDiT-
so we adapt the principle:

    Instead of aligning an intermediate side-net feature, we align the
    model's one-step predicted clean latent  x0_hat = noise - model_pred
    with the depth reference latent, projected through a lightweight
    learnable decoder  D_psi  (Conv ->SiLU ->Conv ->Upsample).

This decoder plays the same role as LISA's D_psi: it maps the model
prediction into the condition's latent space so we can measure consistency.
At inference the decoder is discarded -zero extra inference cost.

Gradient flow:
    L_lisa_depth -> D_psi -> x0_hat -> model_pred -> LoRA params

The loss also supports a parameter-free "structural" mode that uses spatial
gradient correlation instead of a learnable decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


class DepthAlignmentDecoder(nn.Module):
    """Lightweight decoder that projects an image latent into depth-consistency
    space.

    Architecture (mirrors LISA's D_psi -Conv, activation, upsampling):
        Conv2d(C, C, 3, pad=1) -> SiLU -> Conv2d(C, C, 3, pad=1) -> Upsample

    The decoder is intentionally tiny (~0.1% of transformer size).  It learns
    to extract "depth-relevant" structural features from image latents so that
    image regions with strong edges / depth boundaries are emphasised.
    """

    def __init__(self, channels: int = 16, hidden_multiplier: int = 1):
        super().__init__()
        hidden = channels * hidden_multiplier
        self.conv1 = nn.Conv2d(channels, hidden, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


def _spatial_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute per-channel spatial gradient magnitude.

    Args:
        x: (B, C, H, W) latent tensor.

    Returns:
        (B, C, H-1, W-1) gradient magnitude (sqrt of sum of squared finite diffs).
    """
    grad_h = x[:, :, 1:, :] - x[:, :, :-1, :]      # (B, C, H-1, W)
    grad_w = x[:, :, :, 1:] - x[:, :, :, :-1]      # (B, C, H, W-1)
    # Align to (B, C, H-1, W-1)
    grad_h = grad_h[:, :, :, :-1]
    grad_w = grad_w[:, :, :-1, :]
    return torch.sqrt(grad_h ** 2 + grad_w ** 2 + 1e-8)


@LossRegistry.register("lisa_depth")
class LISADepthLoss(BaseLoss):
    """LISA-inspired depth-consistency regularisation.

    Config example::

        {"type": "lisa_depth", "weight": 0.2, "params": {
            "mode": "decoder",
            "latent_channels": 16,
            "use_timestep_weighting": true,
            "timestep_center": 0.3,
            "timestep_width": 0.15
        }}

    Modes:
        - "decoder"    (default): learnable DepthAlignmentDecoder projects
                       x0_hat into depth latent space, then MSE.
        - "structural": parameter-free spatial-gradient-magnitude MSE.

    Timestep weighting (optional):
        Applies a Gaussian window centred on *timestep_center* (sigma in [0,1],
        0 = clean, 1 = noise).  Structure is most visible at low-to-mid noise,
        so default center=0.3 focuses the loss where it matters.
    """

    name = "lisa_depth"

    def __init__(
        self,
        weight: float = 0.2,
        mode: str = "decoder",
        latent_channels: int = 16,
        hidden_multiplier: int = 1,
        use_timestep_weighting: bool = True,
        timestep_center: float = 0.3,
        timestep_width: float = 0.15,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.mode = mode
        self.latent_channels = latent_channels
        self.use_timestep_weighting = use_timestep_weighting
        self.timestep_center = timestep_center
        self.timestep_width = timestep_width

        self._decoder: Optional[DepthAlignmentDecoder] = None
        if mode == "decoder":
            self._decoder = DepthAlignmentDecoder(
                channels=latent_channels,
                hidden_multiplier=hidden_multiplier,
            )
        elif mode == "structural":
            pass  # no learnable parameters
        else:
            raise ValueError(
                f"Unknown lisa_depth mode '{mode}'. Use 'decoder' or 'structural'."
            )

    # ── BaseLoss interface ────────────────────────────────────────────

    def parameters(self) -> List[torch.nn.Parameter]:
        """Expose decoder parameters for the optimizer."""
        if self._decoder is not None:
            return list(self._decoder.parameters())
        return []

    def to(self, device: torch.device, dtype: torch.dtype) -> "LISADepthLoss":
        if self._decoder is not None:
            self._decoder = self._decoder.to(device=device, dtype=dtype)
        return self

    def requires(self) -> list:
        return ["model_pred", "noise", "sigmas", "reference_latent"]

    # ── Loss computation ──────────────────────────────────────────────

    def _compute_timestep_weight(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Gaussian window over sigma (flow-matching time, 0=clean, 1=noise)."""
        weight = torch.exp(
            -((sigmas.float() - self.timestep_center) ** 2) / self.timestep_width
        )
        # Broadcast to match latent dimensions
        while weight.dim() < 4:
            weight = weight.unsqueeze(-1)
        return weight

    def compute(self, context: LossContext) -> torch.Tensor:
        if context.reference_latent is None:
            # No depth reference in this batch -skip silently.
            return torch.tensor(0.0, device=context.model_pred.device,
                                dtype=context.model_pred.dtype)

        # Predicted clean latent (already computed by trainer, but ensure).
        x0_hat = context.x0_hat
        if x0_hat is None:
            x0_hat = context.noise - context.model_pred

        # Upcast to fp32: x0_hat inherits the transformer output dtype (fp32)
        # while cached reference latents are bf16. mse_loss backward requires
        # matching dtypes ("Found dtype BFloat16 but expected Float").
        x0_hat = x0_hat.float()
        depth_latent = context.reference_latent.detach().float()

        # Ensure shape compatibility (spatial dims may differ due to bucketing).
        if x0_hat.shape[-2:] != depth_latent.shape[-2:]:
            depth_latent = F.interpolate(
                depth_latent, size=x0_hat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        if self.mode == "decoder":
            # LISA-style: project x0_hat through learnable decoder, align with
            # the depth reference latent.  Cast input to decoder dtype to avoid
            # float/bf16 mismatch (x0_hat can be float32 from noise - model_pred).
            dec_dtype = next(self._decoder.parameters()).dtype
            pred_projected = self._decoder(x0_hat.to(dec_dtype))
            loss = F.mse_loss(pred_projected, depth_latent.to(dec_dtype))
        else:
            # Structural mode: align spatial gradient magnitudes.
            pred_grad = _spatial_gradient_magnitude(x0_hat)
            depth_grad = _spatial_gradient_magnitude(depth_latent)
            loss = F.mse_loss(pred_grad, depth_grad)

        if self.use_timestep_weighting:
            ts_weight = self._compute_timestep_weight(context.sigmas)
            loss = loss * ts_weight.mean()

        return loss
