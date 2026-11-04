"""
Flow-Matching Loss -standard velocity prediction MSE.

This is the base loss, always active in flow-matching diffusion training.
target = noise - learning_target
target = learning_target - noise   (adapter.velocity_sign == "data_ward", MiniMax-H3)
loss   = mean(weighting * (model_pred - target)^2)
"""
from __future__ import annotations

import torch

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry


def compute_loss_weighting_for_sd3(sigmas: torch.Tensor) -> torch.Tensor:
    """SD3-style loss weighting based on noise level (sigma).

    Weighting = 1 / (sigma^2 + 1) — upweights low-noise (high-signal) steps.
    """
    return 1.0 / (sigmas.float() ** 2 + 1.0)


def compute_loss_weighting_for_min_snr(
    sigmas: torch.Tensor, gamma: float = 5.0
) -> torch.Tensor:
    """Min-SNR-γ weighting for flow matching (Hang et al., 2023).

    ``SNR(σ) = ((1-σ)/σ)^2`` for the flow-matching / rectified-flow
    parameterisation, and ``w = min(SNR, γ) / SNR``.  Unlike the ad-hoc
    ``1/(σ²+1)`` above this (a) caps the high-noise gradient instead of
    suppressing it, and (b) keeps the optimum of the loss unchanged — it only
    rebalances how much each noise level contributes.  γ=5 is the paper's
    default for pixel/latent diffusion.
    """
    s = sigmas.float().clamp(min=1e-4)
    snr = ((1.0 - s) / s) ** 2
    return torch.clamp(snr, max=gamma) / snr


def high_frequency_energy(residual: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Radially-weighted high-frequency energy of *residual* (B, C, H, W).

    Motivation: natural images follow a 1/f² spectrum, so a plain spatial MSE
    under-weights the sparse high-frequency band that carries edges/texture
    ("Balancing Frequencies and Pixels in Flow Matching", 2609.02748 — their
    frequency-weighted loss converges ~1.25x faster with better FID).  We keep
    the elementwise MSE for masking and add this term as a detail penalty.
    """
    r = residual.float()
    spec = torch.fft.rfft2(r, norm="ortho")
    h, w = r.shape[-2], r.shape[-1]
    fy = torch.fft.fftfreq(h, device=r.device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=r.device).abs().view(1, -1)
    radial = torch.sqrt(fy ** 2 + fx ** 2)
    radial = radial / radial.max().clamp(min=1e-8)
    weight = radial ** alpha
    energy = (spec.abs() ** 2 * weight).sum(dim=(-2, -1))       # (B, C)
    norm = weight.sum().clamp(min=1e-8)
    return energy.mean() / norm


@LossRegistry.register("flow_matching")
class FlowMatchingLoss(BaseLoss):
    """Standard flow-matching velocity MSE loss.

    Weighting (``weighting_mode``):
      ``none``    uniform
      ``sd3``     1/(σ²+1)                       (legacy ``use_weighting``)
      ``min_snr`` min(SNR,γ)/SNR                 (recommended, γ=5)

    ``freq_weight`` > 0 adds ``freq_weight * high_frequency_energy(...)``, a
    spectral detail penalty.  Two notes on why it is gated and applied to the
    predicted **x0** rather than to the velocity:

    * qwen-image 2.x latents are 16x spatially downsampled, so their spectrum only
      covers the low/mid band of the image - a latent radial penalty acts on
      mid-scale structure (edges, silhouettes), not on pixel texture.  The 1/f²
      prior of "Balancing Frequencies and Pixels" (2609.02748) is stated for
      *pixel-space* flow matching, so this is a heuristic adaptation, not the
      paper's setting;
    * at high sigma the velocity residual is dominated by white noise, so a
      high-frequency penalty there just re-weights noise and double-counts the
      Min-SNR weighting.  Hence ``freq_sigma_max`` gates the term to the
      detail phase, and it is evaluated on ``x0_pred - x0`` (a "denoised" residual
      whose spectrum is comparable across sigma).
    """

    name = "flow_matching"

    def __init__(
        self,
        weight: float = 1.0,
        use_weighting: bool = True,
        weighting_mode: str | None = None,
        min_snr_gamma: float = 5.0,
        freq_weight: float = 0.0,
        freq_alpha: float = 1.0,
        freq_sigma_max: float = 0.5,
        **params,
    ):
        super().__init__(weight=weight, **params)
        self.use_weighting = use_weighting
        self.weighting_mode = (weighting_mode or ("sd3" if use_weighting else "none")).lower()
        if self.weighting_mode not in ("none", "sd3", "min_snr"):
            raise ValueError(
                f"Unknown flow_matching weighting_mode {weighting_mode!r}; "
                "expected 'none', 'sd3' or 'min_snr'"
            )
        self.min_snr_gamma = float(min_snr_gamma)
        self.freq_weight = float(freq_weight)
        self.freq_alpha = float(freq_alpha)
        self.freq_sigma_max = float(freq_sigma_max)

    def compute(self, context: LossContext) -> torch.Tensor:
        # MiniMax-H3's scheduler is data-ward: v = x0 - x_t, so the flow
        # target flips sign relative to the standard convention.  A typo'd
        # or missing override would silently train the wrong direction
        # (loss still decreases), so reject unknown values outright.
        velocity_sign = getattr(context.adapter, "velocity_sign", "standard")
        if velocity_sign not in ("standard", "data_ward"):
            raise ValueError(
                f"Unsupported velocity_sign {velocity_sign!r}; "
                "expected 'standard' or 'data_ward'"
            )
        if velocity_sign == "data_ward":
            target = context.learning_target - context.noise
        else:
            target = context.noise - context.learning_target

        residual = context.model_pred - target
        weighting = None
        if self.weighting_mode == "sd3":
            weighting = compute_loss_weighting_for_sd3(context.sigmas)
        elif self.weighting_mode == "min_snr":
            weighting = compute_loss_weighting_for_min_snr(
                context.sigmas, self.min_snr_gamma)
        if weighting is not None:
            # Expand weighting to match pred shape
            while weighting.dim() < residual.dim():
                weighting = weighting.unsqueeze(-1)
        # Region/mask weighting is loss-agnostic (BaseLoss.reduce_masked);
        # with no mask this is bit-identical to the previous plain mean().
        loss = self.reduce_masked(residual ** 2, context, weighting=weighting)

        if self.freq_weight > 0.0 and residual.dim() == 4:
            # Denoised residual: x0_pred - x0 (velocity + x_t decomposition), whose
            # spectrum is comparable across sigma; gate to the detail phase only.
            try:
                x0_pred = context.noise - context.model_pred
                x0_res = x0_pred - context.learning_target
                sig = context.sigmas.float().view(-1)
                keep = sig <= self.freq_sigma_max
                if keep.any():
                    loss = loss + self.freq_weight * high_frequency_energy(
                        x0_res[keep] if x0_res.shape[0] == sig.shape[0] else x0_res,
                        self.freq_alpha)
            except Exception:  # noqa: BLE001 - the term is optional, never fatal
                pass

        return loss

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]
