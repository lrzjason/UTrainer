"""
Corrected Flow-Matching Objective for Guidance-Distilled Models.

Implements DC-Gen (arXiv:2509.25180), Appendix A, Eq. (4)-(10) — the training
objective for fine-tuning a GUIDANCE-DISTILLED checkpoint.

The problem: a guidance-distilled model v_eta is trained to mimic the
classifier-free-guidance (CFG) output of its teacher (Eq. 5):

    L_distill = E [ || v_eta(z_t, c, t, w) - v^w_theta(z_t, c, t) ||^2 ]
    v^w_theta(z_t, c, t) = (1 + w) v_theta(z_t, c, t) - w v_theta(z_t, c_hat, t)   (Eq. 6)

Applying the STANDARD flow-matching loss (Eq. 4) to such a checkpoint is
biased: the distilled model's output IS the CFG-combined velocity, not the
raw velocity v_theta, so matching it against the flow target
v_t = eps - x0 trains the wrong quantity.  DC-Gen Fig. 9 measures the damage
on MJHQ-30K 512x512: CLIP score 26.87 (L_FM) vs 27.38 (L_guide_FM).

The correction recovers the raw velocity from the distilled model's OWN
outputs under two conditions (Eq. 7-9):

    v_theta(z_t, c, t) ~= [ v_eta(z_t, c, t, w) + w * v_eta(z_t, c_hat, t, w) ] / (1 + w)

The unconditional pass feeds the empty-prompt embedding (Eq. 7: with an empty
condition the CFG combination collapses to the raw unconditional velocity for
ANY w, so v_eta(z_t, c_hat, t, w) ~= v_theta(z_t, c_hat, t)).  w is the
guidance scale, sampled per-sample from U[guidance_scale_min,
guidance_scale_max] per the paper's expectation.  The training loss is
(Eq. 10):

    L_guide_FM = E_{z0, eps, t, w} [ || [v_eta(c) + w * v_eta(c_hat)] / (1 + w)
                                     - v_t ||^2 ]

Gradient flows through BOTH forwards — both are outputs of the same network
being trained, and dropping either term changes the objective.

Trainer plumbing: this loss sets ``needs_uncond_forward = True``.  The
trainer detects the flag, runs a second forward on the same noisy latents
with the cached empty-prompt embedding, and hands the unpacked unconditional
velocity to this loss via ``LossContext.model_pred_uncond``.  When caption
dropout already fired for the step (the conditional pass IS unconditional),
the trainer reuses that prediction instead of running a second forward.

Scope note: the algebraic correction assumes the distilled model does NOT
take the guidance scale as a network input (Qwen-Image 2.1, Z-Image-Turbo).
For guidance-input checkpoints (FLUX-Krea style — adapter config
``"guidance"``), set that config value AND make the correction use the same
scale (``guidance_scale_min == guidance_scale_max == guidance``), so the
algebraic correction matches the scale the forwards were actually run at.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.losses.flow_matching import compute_loss_weighting_for_sd3
from UnifiedTrainer.registry import LossRegistry


@LossRegistry.register("guide_flow_matching")
class GuideFlowMatchingLoss(BaseLoss):
    """DC-Gen corrected flow-matching objective for guidance-distilled models.

    Config example (replaces the plain flow_matching loss when fine-tuning a
    guidance-distilled checkpoint, e.g. Qwen-Image 2.1):

        "losses": [
            {"type": "guide_flow_matching", "weight": 1.0, "params": {
                "guidance_scale_min": 1.0,
                "guidance_scale_max": 8.0
            }}
        ]
    """

    name = "guide_flow_matching"

    # Trainer hook: run a second (empty-prompt) forward each step and pass it
    # as LossContext.model_pred_uncond.
    needs_uncond_forward = True

    def __init__(
        self,
        weight: float = 1.0,
        guidance_scale_min: float = 1.0,
        guidance_scale_max: float = 8.0,
        fixed_guidance_scale: Optional[float] = None,
        use_weighting: bool = False,
        **params,
    ):
        super().__init__(weight=weight, **params)
        if fixed_guidance_scale is not None:
            # Degenerate range — single known effective distillation scale.
            guidance_scale_min = guidance_scale_max = float(fixed_guidance_scale)
        if guidance_scale_min < 0.0:
            raise ValueError(
                f"guide_flow_matching: guidance_scale_min must be >= 0 "
                f"(1 + w divides the correction), got {guidance_scale_min}"
            )
        if guidance_scale_max < guidance_scale_min:
            raise ValueError(
                f"guide_flow_matching: guidance_scale_max ({guidance_scale_max}) "
                f"must be >= guidance_scale_min ({guidance_scale_min})"
            )
        self.guidance_scale_min = float(guidance_scale_min)
        self.guidance_scale_max = float(guidance_scale_max)
        self.fixed_guidance_scale = fixed_guidance_scale
        self.use_weighting = use_weighting
        # Telemetry for callbacks/wandb (loss/guide_flow_matching/<component>).
        self.last_components: dict = {}

    # ── helpers ────────────────────────────────────────────────────────

    def _sample_guidance_scales(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Per-sample guidance scale w ~ U[w_min, w_max] (fp32 for a clean
        division; cast back by the caller)."""
        if self.fixed_guidance_scale is not None:
            return torch.full(
                (batch_size,), self.fixed_guidance_scale,
                device=device, dtype=torch.float32,
            )
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        return self.guidance_scale_min + (
            self.guidance_scale_max - self.guidance_scale_min
        ) * u

    @staticmethod
    def _broadcast_like(w: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B, 1, 1, ...) matching pred's dims (4D image / 5D video)."""
        while w.dim() < pred.dim():
            w = w.unsqueeze(-1)
        return w

    # ── loss ───────────────────────────────────────────────────────────

    def compute(self, context: LossContext) -> torch.Tensor:
        v_cond = context.model_pred
        v_uncond = context.model_pred_uncond
        if v_uncond is None:
            raise RuntimeError(
                "guide_flow_matching requires the unconditional velocity "
                "(LossContext.model_pred_uncond) — the trainer fills it when a "
                "loss sets needs_uncond_forward=True. If you see this, the "
                "trainer plumbing is missing or the loss was called directly."
            )
        if v_uncond.shape != v_cond.shape:
            raise ValueError(
                f"guide_flow_matching: cond/uncond prediction shapes differ: "
                f"{tuple(v_cond.shape)} vs {tuple(v_uncond.shape)}"
            )

        # Flow target with the adapter's sign dispatch — identical logic to
        # losses/flow_matching.py (a typo'd or missing override would silently
        # train the wrong direction, so reject unknown values outright).
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

        # ── DC-Gen Eq. 9/10: corrected raw-velocity estimate ──
        w = self._sample_guidance_scales(
            v_cond.shape[0], v_cond.device, v_cond.dtype
        ).to(dtype=v_cond.dtype)
        w_b = self._broadcast_like(w, v_cond)
        v_corrected = (v_cond + w_b * v_uncond) / (1.0 + w_b)

        if self.use_weighting:
            weighting = compute_loss_weighting_for_sd3(context.sigmas)
            while weighting.dim() < v_corrected.dim():
                weighting = weighting.unsqueeze(-1)
            loss = (weighting * (v_corrected - target) ** 2).mean()
        else:
            loss = ((v_corrected - target) ** 2).mean()

        self.last_components = {
            "guidance_scale_mean": float(w.mean().item()),
        }
        return loss

    def requires(self) -> List[str]:
        return [
            "model_pred",
            "model_pred_uncond",
            "noise",
            "learning_target",
            "sigmas",
        ]
