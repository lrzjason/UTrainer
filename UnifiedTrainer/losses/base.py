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

    def __call__(self, context: LossContext) -> torch.Tensor:
        """Compute weighted loss. This is what the trainer calls."""
        return self.weight * self.compute(context)
