"""
Masked Flow-Matching Loss -region-weighted velocity MSE.

Extends the standard flow-matching loss with a per-latent weight built
from the sample's edit mask:

    w = base_weight + edit_weight * mask
    loss = mean(w * (model_pred - target)^2) / mean(w)

Key properties:

- **Only UP-weights the edited region.**  ``base_weight`` keeps the
  non-edited region at (or above) full strength, so the copy/alignment
  signal the alignment training relies on is never suppressed.  Set
  ``base_weight=2..3`` for the k3-style "non-edited region 2-3x" recipe,
  and ``edit_weight`` to taste for the edited region.
- **Normalised** (divide by ``mean(w)``) so the loss scale stays
  comparable to plain ``flow_matching`` — safe to mix the two losses in
  one config, and hyper-parameters (lr) transfer.
- **Graceful degradation**: when the batch carries no mask
  (``LossContext.loss_mask is None`` — t2i / noop views, or any config
  without ``mask_configs``), it computes exactly the plain flow-matching
  loss.  A config that never uses masks behaves bit-identically to
  ``{"type": "flow_matching"}``.

The mask grid matches the TARGET latent (bucket-cropped with the same
deterministic transform at cache-build time, then area-downsampled by the
VAE scale factor), so ``mask`` broadcasts against ``model_pred`` as
(B, 1, h, w) -> (B, C, h, w).

Config example:
    "losses": [
        {"type": "masked_flow_matching", "weight": 1.0, "params": {
            "edit_weight": 3.0, "base_weight": 1.0, "use_weighting": true}}
    ]
"""
from __future__ import annotations

import torch

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.losses.flow_matching import compute_loss_weighting_for_sd3
from UnifiedTrainer.registry import LossRegistry


@LossRegistry.register("masked_flow_matching")
class MaskedFlowMatchingLoss(BaseLoss):
    """Flow-matching velocity MSE with optional edit-region up-weighting."""

    name = "masked_flow_matching"

    def __init__(
        self,
        weight: float = 1.0,
        edit_weight: float = 3.0,
        base_weight: float = 1.0,
        use_weighting: bool = True,
        **params,
    ):
        super().__init__(weight=weight, **params)
        if edit_weight < 0:
            raise ValueError(
                f"masked_flow_matching: edit_weight must be >= 0, "
                f"got {edit_weight}"
            )
        if base_weight <= 0:
            raise ValueError(
                f"masked_flow_matching: base_weight must be > 0 "
                f"(a null base would erase the copy/alignment signal), "
                f"got {base_weight}"
            )
        self.edit_weight = float(edit_weight)
        self.base_weight = float(base_weight)
        self.use_weighting = use_weighting

    def compute(self, context: LossContext) -> torch.Tensor:
        # Same velocity_sign contract as flow_matching (fail loud on typos).
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

        sq_err = (context.model_pred - target) ** 2

        # ── No mask: plain flow matching (t2i / noop / mask-less configs) ──
        if context.loss_mask is None:
            if self.use_weighting:
                weighting = compute_loss_weighting_for_sd3(context.sigmas)
                while weighting.dim() < sq_err.dim():
                    weighting = weighting.unsqueeze(-1)
                return (weighting * sq_err).mean()
            return sq_err.mean()

        mask = context.loss_mask
        if mask.dim() == sq_err.dim() - 1:
            mask = mask.unsqueeze(1)          # (B, h, w) -> (B, 1, h, w)
        if mask.shape[-2:] != sq_err.shape[-2:]:
            raise ValueError(
                f"masked_flow_matching: mask grid {tuple(mask.shape)} does "
                f"not match model_pred grid {tuple(sq_err.shape)} — the mask "
                "is cached against the target bucket; check that the mask "
                "and target images share dimensions."
            )
        mask = mask.to(dtype=sq_err.dtype)

        w = self.base_weight + self.edit_weight * mask   # (B,1,h,w) 广播
        if self.use_weighting:
            weighting = compute_loss_weighting_for_sd3(context.sigmas)
            while weighting.dim() < sq_err.dim():
                weighting = weighting.unsqueeze(-1)
            w = w * weighting

        # Normalise so the loss scale matches plain flow matching:
        # w is (B, 1, h, w) while sq_err is (B, C, h, w), so the weight
        # only covers 1/C of the summed positions — scale the denominator
        # by the broadcast factor (numel ratio) instead of assuming the
        # shapes match.
        denom = w.sum() * (sq_err.numel() / w.numel())
        return (w * sq_err).sum() / denom.clamp_min(1e-8)

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]
