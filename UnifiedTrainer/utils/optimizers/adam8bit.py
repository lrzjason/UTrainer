# Reference: adapted from ai-toolkit/toolkit/optimizers/adam8bit.py
# See aitoolkit_acc.md for full technique analysis.
"""
Adam8bit -Adam optimizer with 8-bit state storage + stochastic rounding.

Optimizer state (exp_avg, exp_avg_sq) is stored as ``Auto8bitTensor``
(int8 + scale), cutting state VRAM by ~4x.  Stochastic rounding ensures
unbiased parameter updates when working in bf16/fp8.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from UnifiedTrainer.utils.stochastic_rounding import (
    copy_stochastic,
    stochastic_grad_accummulation,
)
from UnifiedTrainer.utils.optimizers.optimizer_utils import Auto8bitTensor


class Adam8bit(Optimizer):
    """Adam with 8-bit state and stochastic rounding.

    Arguments:
        params: iterable of parameters or param-group dicts
        lr: learning rate (default 1e-3)
        betas: (beta1, beta2) for EMA (default (0.9, 0.999))
        eps: denominator stabilizer (default 1e-8)
        weight_decay: weight decay coefficient (default 0)
        decouple: AdamW-style decoupled weight decay if True (default True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        decouple: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, decouple=decouple,
        )
        super().__init__(params, defaults)

        self.is_stochastic_rounding_accumulation = False

        # Register stochastic gradient accumulation hooks for non-fp32 params
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    # ── PyTorch capability flags ─────────────────────────────────────────

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    # ── Pre-step: move stochastically-rounded grads into .grad ───────────

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # ── Main step ────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, closure=None):
        self.step_hook()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            decay = group["weight_decay"]
            decouple = group["decouple"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)

                # Coupled weight decay
                if decay != 0 and not decouple:
                    grad.add_(p_fp32.data, alpha=decay)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = Auto8bitTensor(
                        torch.zeros_like(p_fp32.data).detach()
                    )
                    state["exp_avg_sq"] = Auto8bitTensor(
                        torch.zeros_like(p_fp32.data).detach()
                    )

                exp_avg = state["exp_avg"].to(torch.float32)
                exp_avg_sq = state["exp_avg_sq"].to(torch.float32)

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Adam EMA updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Decoupled weight decay
                if decay != 0 and decouple:
                    p_fp32.data.mul_(1 - lr * decay)

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p_fp32.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Re-quantize state with stochastic rounding
                state["exp_avg"] = Auto8bitTensor(exp_avg)
                state["exp_avg_sq"] = Auto8bitTensor(exp_avg_sq)

                # Stochastic parameter update
                copy_stochastic(p.data, p_fp32.data)

        return loss

    # ── Checkpointing ────────────────────────────────────────────────────

    def state_dict(self):
        sd = super().state_dict()
        for _pid, pstate in sd["state"].items():
            for key, value in pstate.items():
                if isinstance(value, Auto8bitTensor):
                    pstate[key] = {
                        "_type": "Auto8bitTensor",
                        "state": value.state_dict(),
                    }
        return sd

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for _pid, pstate in self.state.items():
            for key, value in pstate.items():
                if isinstance(value, dict) and value.get("_type") == "Auto8bitTensor":
                    pstate[key] = Auto8bitTensor(value["state"])
