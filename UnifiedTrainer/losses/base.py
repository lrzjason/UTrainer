"""
BaseLoss -the protocol for composable loss modules.

Losses are config-driven: the trainer assembles a list of loss modules from
the config's "losses" array, and each module declares what context fields it
needs via requires(). The trainer calls loss.compute(context) generically.

Example config:
    "losses": [
        {"type": "flow_matching", "weight": 1.0},
    ]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import torch

if TYPE_CHECKING:
    from UnifiedTrainer.models.base import BaseModelAdapter


@dataclass
class LossContext:
    """Context bundle passed to every loss module's compute() method.

    Contains all model outputs, targets, and derived quantities that losses
    might need. Individual loss modules declare which fields they require
    via BaseLoss.requires().
    """

    # Model outputs
    model_pred: torch.Tensor           # unpacked velocity prediction
    noise: torch.Tensor                # noise sampled for this step
    sigmas: torch.Tensor               # noise level (timestep) for this step

    # Targets
    learning_target: torch.Tensor      # primary target latent

    # Derived (filled by trainer before loss computation)
    x0_hat: Optional[torch.Tensor] = None  # predicted clean latent = noise - model_pred

    # Second (empty-prompt / unconditional) velocity prediction, same shape
    # as model_pred.  Filled by the trainer when any configured loss sets
    # ``needs_uncond_forward = True`` (e.g. guide_flow_matching — the
    # DC-Gen corrected objective for guidance-distilled models needs the
    # distilled model's own output under the empty condition).  None when no
    # loss requested it.  Gradient flows through BOTH predictions: both are
    # outputs of the same network being trained.
    model_pred_uncond: Optional[torch.Tensor] = None

    # Reference / condition latent (e.g. depth map latent for depth-conditioned training).
    # Filled by the trainer when the batch contains a reference latent.
    reference_latent: Optional[torch.Tensor] = None

    # Mask (optional, for inpainting / region-specific losses)
    loss_mask: Optional[torch.Tensor] = None

    # Adapter reference (for model-specific operations inside losses)
    adapter: Optional["BaseModelAdapter"] = None

    # Extra fields for extensibility (e.g. representation alignment features)
    extra: dict = field(default_factory=dict)


class BaseLoss(ABC):
    """Abstract base class for composable loss modules.

    Subclasses must:
    1. Set `name` to a unique identifier matching the config "type" field.
    2. Implement compute(context) -> scalar tensor.
    3. Optionally override requires() to declare needed LossContext fields.
    """

    name: str = "base"

    def __init__(self, weight: float = 1.0, **params: Any):
        self.weight = weight
        self.params = params

    @abstractmethod
    def compute(self, context: LossContext) -> torch.Tensor:
        """Compute and return a scalar loss tensor (not yet weighted)."""
        ...

    def requires(self) -> List[str]:
        """Declare which LossContext fields this loss needs.

        The trainer can use this to skip unnecessary computation.
        Return field names from LossContext (e.g. ['x0_hat']).
        """
        return []

    def parameters(self) -> List[torch.nn.Parameter]:
        """Return trainable parameters owned by this loss module.

        Losses that create auxiliary modules (e.g. a lightweight decoder for
        LISA-style alignment) override this to expose their parameters so the
        trainer can add them to the optimizer.  Default: no extra parameters.
        """
        return []

    def to(self, device: torch.device, dtype: torch.dtype) -> "BaseLoss":
        """Move internal modules to the given device/dtype.  Override when
        the loss owns nn.Module sub-modules."""
        return self

    # ── Region / mask weighting (loss-agnostic) ─────────────────────────
    #
    # Shared entry point so mask weighting is NOT tied to any single loss.
    # Any loss opts in with one line:
    #
    #     loss = self.reduce_masked(sq_err, context, weighting=weighting)
    #
    # Semantics (identical to losses/masked_flow_matching.py, the framework's
    # reference implementation, and to losses/region_flow_matching.py):
    #
    #     w    = base_weight + edit_weight * loss_mask        # mask in [0,1]
    #     loss = sum(w * err2) / sum(w)                       # normalised
    #
    # where w is expanded to err2's shape first, so the divisor counts every
    # element of the broadcast weight (channel dim included).  That is what
    # keeps the loss scale comparable to a plain mean, so lr transfers between
    # masked and unmasked runs.
    #
    # No mask present -> exactly the pre-existing behaviour:
    #   (weighting * err2).mean()  or  err2.mean()
    #
    # Two weight sources are honoured:
    #   * LossContext.loss_mask    -> mask in [0,1], becomes base + edit * mask
    #   * extra["region_weight"]   -> used verbatim as the weight map (the hook
    #                                 region_flow_matching documents)
    def mask_base_weight(self, default: float = 1.0) -> float:
        return float(self.params.get("base_weight", default))

    def mask_edit_weight(self, default: float = 1.0) -> float:
        return float(self.params.get("edit_weight", default))

    @staticmethod
    def _fit_weight(w: torch.Tensor, like: torch.Tensor) -> Optional[torch.Tensor]:
        while w.dim() < like.dim():
            w = w.unsqueeze(0)
        try:
            return w.expand_as(like)
        except RuntimeError:
            return None

    def region_weight_map(
        self,
        context: LossContext,
        like: torch.Tensor,
        base_weight: Optional[float] = None,
        edit_weight: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Per-element weight map broadcast to ``like``, or None if unweighted."""
        mask = getattr(context, "loss_mask", None)
        if mask is not None:
            bw = self.mask_base_weight() if base_weight is None else float(base_weight)
            ew = self.mask_edit_weight() if edit_weight is None else float(edit_weight)
            if ew == 0.0:
                return None
            m = mask.to(device=like.device, dtype=torch.float32)
            if m.dim() == like.dim() - 1:            # (B,h,w) -> (B,1,h,w)
                m = m.unsqueeze(1)
            # Binary mask (>= 0.5).  Area-averaging the region down to the
            # latent grid leaves fractional values on boundary cells, which
            # the model then learns to reproduce as a semi-transparent rim.
            # Thresholding here ALSO covers masks already cached in float
            # form, so no cache rebuild is required.
            m = (m >= 0.5).to(m.dtype)
            if m.shape[-2:] != like.shape[-2:]:
                raise ValueError(
                    f"{self.name}: loss_mask grid {tuple(m.shape)} does not match "
                    f"the loss grid {tuple(like.shape)} - the mask is cached "
                    "against the target bucket; check that the mask and target "
                    "images share dimensions."
                )
            w = bw + ew * m
        else:
            raw = (context.extra or {}).get("region_weight")
            if raw is None:
                return None
            w = raw.to(device=like.device, dtype=torch.float32)
            if w.dim() == like.dim() - 1:
                w = w.unsqueeze(1)
        return self._fit_weight(w, like)

    def reduce_masked(
        self,
        sq_err: torch.Tensor,
        context: LossContext,
        weighting: Optional[torch.Tensor] = None,
        base_weight: Optional[float] = None,
        edit_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Region-weighted reduction of a per-element squared error.

        ``sq_err`` is the raw (unweighted) per-element error; ``weighting`` is
        an optional sigma/timestep weighting already broadcastable to it.
        Falls back to the plain-mean semantics when nothing is weighted.
        """
        if weighting is not None:
            while weighting.dim() < sq_err.dim():
                weighting = weighting.unsqueeze(-1)
        w = self.region_weight_map(context, sq_err, base_weight, edit_weight)
        if w is None:
            if weighting is not None:
                return (weighting * sq_err).mean()
            return sq_err.mean()
        if weighting is not None:
            w = w * weighting
        w_full = w.expand_as(sq_err)
        return (sq_err * w_full).sum() / w_full.sum().clamp_min(1e-6)

    def mask_telemetry(self, context: LossContext) -> dict:
        """wandb-friendly flags: is masking active, and at what weights."""
        mask = getattr(context, "loss_mask", None)
        if mask is None:
            return {"mask_active": 0.0}
        return {
            "mask_active": 1.0,
            "mask_base_weight": self.mask_base_weight(),
            "mask_edit_weight": self.mask_edit_weight(),
            "mask_region_frac": float((mask > 0.5).float().mean().item()),
        }

    def __call__(self, context: LossContext) -> torch.Tensor:
        """Compute weighted loss. This is what the trainer calls."""
        return self.weight * self.compute(context)
