"""
EMA (Exponential Moving Average) manager for LoRA weights.

Maintains a shadow copy of trainable parameters updated as:
    ema_param = decay * ema_param + (1 - decay) * param

At validation time, swap in EMA weights for smoother evaluation.
VRAM cost: 1× LoRA params (~50MB for rank 64). Negligible.
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class EMAManager:
    """Manages EMA shadow weights for a set of model parameters.

    Usage::

        ema = EMAManager(model.parameters(), decay=0.999)
        # After each optimizer step:
        ema.update()
        # Before validation:
        ema.apply()
        # After validation:
        ema.restore()
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.999,
        warmup_steps: int = 0,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0

        # Store shadow copies (detached clones on same device)
        self._shadow: Dict[int, torch.Tensor] = {}
        self._backup: Dict[int, torch.Tensor] = {}
        self._param_refs: Dict[int, nn.Parameter] = {}

        for p in parameters:
            if p.requires_grad:
                pid = id(p)
                self._shadow[pid] = p.data.clone().detach()
                self._param_refs[pid] = p

    def _get_decay(self) -> float:
        """Optionally ramp up decay during warmup."""
        if self.warmup_steps > 0 and self._step < self.warmup_steps:
            return min(self.decay, (1 + self._step) / (10 + self._step))
        return self.decay

    @torch.no_grad()
    def update(self):
        """Update shadow weights. Call after each optimizer step."""
        decay = self._get_decay()
        self._step += 1
        for pid, param in self._param_refs.items():
            shadow = self._shadow[pid]
            # shadow = decay * shadow + (1 - decay) * param
            shadow.lerp_(param.data, 1.0 - decay)

    @torch.no_grad()
    def apply(self):
        """Swap EMA weights into the model. Call before validation."""
        for pid, param in self._param_refs.items():
            self._backup[pid] = param.data.clone()
            param.data.copy_(self._shadow[pid])

    @torch.no_grad()
    def restore(self):
        """Restore original weights after validation."""
        for pid, param in self._param_refs.items():
            if pid in self._backup:
                param.data.copy_(self._backup[pid])
        self._backup.clear()

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "shadow": {str(pid): t.cpu() for pid, t in self._shadow.items()},
            "step": self._step,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict):
        """Restore EMA state from checkpoint."""
        self._step = state.get("step", 0)
        self.decay = state.get("decay", self.decay)
        shadow_dict = state.get("shadow", {})
        for pid, param in self._param_refs.items():
            key = str(pid)
            if key in shadow_dict:
                self._shadow[pid] = shadow_dict[key].to(
                    device=param.device, dtype=param.dtype
                )
