"""NoiseSelector — pluggable noise/timestep selection strategy for training.

The training loop delegates noise and sigma sampling to a NoiseSelector:
    noises, sigmas, timesteps = self.noise_selector.select(...)

Default: RandomNoiseSelector (standard random sampling, zero overhead).
Explorative: ExplorativeNoiseSelector (best-of-K noise exploration with
stop-gradient forwards — VRAM identical to standard training).

Config:
    "training": {
        "noise_selector": {
            "type": "explorative",   // "random" (default) | "explorative"
            "K_cond": 4,             // noise candidates for text-conditioned batches
            "K_uncond": 5,           // candidates for caption-dropped (uncond) batches
            "warmup_steps": 0,       // K=1 for first N steps (0 = explore from step 1)
            "schedule": "constant",  // "constant" | "linear_decay" | "cosine"
            "log_stats": true        // expose xm/* stats to callbacks
        }
    }

Reference: md/explorative_implementation.md
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Abstract base ────────────────────────────────────────────────────────


class NoiseSelector(ABC):
    """Strategy interface for selecting (noises, sigmas, timesteps) per step."""

    @abstractmethod
    def select(
        self,
        batch: dict,
        target_latents: List[torch.Tensor],
        adapter: Any,
        transformer: nn.Module,
        step: int,
        pre_forward_fn: Optional[Callable[[], None]] = None,
    ) -> tuple:
        """Select noise and timestep for this training step.

        Args:
            batch: Current data batch dict.
            target_latents: List of target latent tensors (one per target).
            adapter: Model adapter (has sample_timesteps, prepare_model_input,
                     unpack_prediction).
            transformer: The transformer model (for exploration forwards).
            step: Current global training step.
            pre_forward_fn: Optional callback invoked before each forward pass
                            (e.g. block_swap preparation).

        Returns:
            (noises, sigmas, timesteps) where:
            - noises: list[Tensor], one noise tensor per target
            - sigmas: Tensor [B], flow-matching noise levels
            - timesteps: Tensor [B], transformer-input timesteps
        """
        ...


# ── Default: random sampling ─────────────────────────────────────────────


class RandomNoiseSelector(NoiseSelector):
    """Standard random noise + timestep sampling. Equivalent to original trainer."""

    def select(
        self,
        batch: dict,
        target_latents: List[torch.Tensor],
        adapter: Any,
        transformer: nn.Module,
        step: int,
        pre_forward_fn: Optional[Callable[[], None]] = None,
    ) -> tuple:
        timesteps, sigmas = adapter.sample_timesteps(
            target_latents[0].shape[0],
            target_latents[0].device,
            target_latents[0].dtype,
            latent_height=target_latents[0].shape[2],
            latent_width=target_latents[0].shape[3],
        )
        noises = [torch.randn_like(tl) for tl in target_latents]
        return noises, sigmas, timesteps


# ── Explorative: best-of-K noise selection ───────────────────────────────


class ExplorativeNoiseSelector(NoiseSelector):
    """Best-of-K noise exploration with stop-gradient forwards.

    Sigma is sampled ONCE (standard distribution). K different noise vectors
    are evaluated via no_grad forwards; the noise yielding lowest velocity MSE
    is returned for the actual gradient-enabled training forward.

    Adaptive K (per the Explorative Modeling practical guide):
    - Text-conditioned batches have narrow p(x|prompt) → low K (default 4).
      The paper's own SOTA (XRAE on ImageNet) uses K=2; rich-text LoRA can
      afford a few more candidates.
    - Unconditional batches (caption dropped) match the WHOLE dataset → very
      high multimodality → high K (default 5) so the empty-text branch learns
      real modes instead of the blurry global mean (which is exactly what CFG
      compensates for).
    The selector reads batch["_caption_dropped"] (set by the trainer's caption
    dropout) to pick K_cond vs K_uncond per batch.

    VRAM: identical to standard training (no_grad stores no activations).
    Time: ~(K * 0.6 + 1.0) / 1.0 × standard step (exploration forwards are
          cheaper since no graph is built).
    """

    def __init__(self, config: dict):
        # K_cond / K_uncond take precedence; legacy single K is the fallback
        # for both when neither is specified.
        legacy_K = int(config.get("K", 4))
        self.K_cond: int = int(config.get("K_cond", config.get("K", 4)))
        self.K_uncond: int = int(config.get("K_uncond", 5))
        self.warmup_steps: int = int(config.get("warmup_steps", 0))
        self.schedule: str = config.get("schedule", "constant")
        self.log_stats: bool = config.get("log_stats", True)

        # Runtime stats (exposed to callbacks via trainer.last_loss_breakdown)
        self.last_stats: dict = {}
        # Human-readable per-step log line (consumed by the trainer's logger)
        self.last_log_line: str = ""

        logger.info(
            f"ExplorativeNoiseSelector: K_cond={self.K_cond}, "
            f"K_uncond={self.K_uncond}, warmup={self.warmup_steps}, "
            f"schedule={self.schedule}"
        )

    def select(
        self,
        batch: dict,
        target_latents: List[torch.Tensor],
        adapter: Any,
        transformer: nn.Module,
        step: int,
        pre_forward_fn: Optional[Callable[[], None]] = None,
    ) -> tuple:
        # Pick base K from the batch's conditioning state, then apply schedule.
        is_uncond = bool(batch.get("_caption_dropped", False))
        base_K = self.K_uncond if is_uncond else self.K_cond
        K = self._get_current_K(step, base_K)

        # ── Sigma/timestep: sampled ONCE (unchanged from standard training) ──
        timesteps, sigmas = adapter.sample_timesteps(
            target_latents[0].shape[0],
            target_latents[0].device,
            target_latents[0].dtype,
            latent_height=target_latents[0].shape[2],
            latent_width=target_latents[0].shape[3],
        )

        if K <= 1:
            # Warmup or schedule decayed to 1 → standard random noise
            noises = [torch.randn_like(tl) for tl in target_latents]
            if self.log_stats:
                self.last_stats = {
                    "xm/K_effective": 1,
                    "xm/uncond": float(is_uncond),
                }
                self.last_log_line = (
                    f"[XM] step={step} K=1 (warmup/decay) "
                    f"mode={'uncond' if is_uncond else 'cond'} — random noise"
                )
            return noises, sigmas, timesteps

        # ── Phase 1: Explore K noise candidates (no_grad, no activations) ──
        sigmas_b = sigmas.view(-1, 1, 1, 1) if target_latents[0].ndim >= 2 else sigmas
        best_loss = float("inf")
        worst_loss = float("-inf")
        best_noises = None
        best_k_idx = -1
        all_losses: List[float] = []

        for k in range(K):
            noises_k = [torch.randn_like(tl) for tl in target_latents]
            noisy_k = [
                (1.0 - sigmas_b) * tl + sigmas_b * n
                for tl, n in zip(target_latents, noises_k)
            ]

            # Prepare block swap (if applicable) before each forward
            if pre_forward_fn is not None:
                pre_forward_fn()

            with torch.no_grad():
                input_k = adapter.prepare_model_input(batch, noisy_k, sigmas)
                pred_k = transformer(**input_k)
                unpacked_k = adapter.unpack_prediction(pred_k)
                loss_k = self._eval_velocity_mse(unpacked_k, noises_k, target_latents)

            loss_val = loss_k.item()
            all_losses.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_noises = [n.clone() for n in noises_k]
                best_k_idx = k
            if loss_val > worst_loss:
                worst_loss = loss_val

            # Immediate release — no accumulation across K iterations
            del pred_k, unpacked_k, input_k, noisy_k, noises_k

        # ── Record exploration statistics ──
        if self.log_stats:
            import statistics
            mean_loss = sum(all_losses) / len(all_losses)
            self.last_stats = {
                "xm/K_effective": K,
                "xm/uncond": float(is_uncond),
                "xm/best_k": best_k_idx,
                "xm/loss_best": best_loss,
                "xm/loss_worst": worst_loss,
                "xm/loss_mean": mean_loss,
                "xm/gap": worst_loss - best_loss,
                "xm/loss_std": statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0,
            }
            self.last_log_line = (
                f"[XM] step={step} K={K} mode={'uncond' if is_uncond else 'cond'} "
                f"min={best_loss:.4f} max={worst_loss:.4f} mean={mean_loss:.4f} "
                f"gap={worst_loss - best_loss:.4f} best_k={best_k_idx}"
            )

        # ── Phase 2: Return winning noise (sigma unchanged) ──
        # The training loop will do the single gradient-enabled forward.
        return best_noises, sigmas, timesteps

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _eval_velocity_mse(
        unpacked: List[torch.Tensor],
        noises: List[torch.Tensor],
        target_latents: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute flow-matching velocity MSE for ranking.

        velocity target = noise - clean_latent
        loss = MSE(predicted_velocity, target_velocity)

        This is the primary training signal (always present, no auxiliary
        context needed). Used only for argmin ranking during exploration.
        """
        total = torch.tensor(0.0, device=target_latents[0].device)
        for unpacked_i, noise_i, target_i in zip(unpacked, noises, target_latents):
            velocity_target = noise_i - target_i
            total = total + F.mse_loss(unpacked_i.float(), velocity_target.float())
        return total / len(target_latents)

    def _get_current_K(self, step: int, base_K: int) -> int:
        """Compute effective K for this step (respects warmup + schedule).

        base_K is the per-batch target (K_cond or K_uncond); the schedule
        decays from base_K toward 1 over training.
        """
        if base_K <= 1:
            return 1
        if step < self.warmup_steps:
            return 1

        steps_past_warmup = step - self.warmup_steps

        if self.schedule == "constant":
            return base_K

        elif self.schedule == "linear_decay":
            # Decay from base_K to 1 over warmup_steps * 10 steps after warmup
            decay_duration = max(self.warmup_steps * 10, 1000)
            progress = min(1.0, steps_past_warmup / decay_duration)
            k = base_K - (base_K - 1) * progress
            return max(1, round(k))

        elif self.schedule == "cosine":
            # Cosine decay from base_K to 1
            decay_duration = max(self.warmup_steps * 10, 1000)
            progress = min(1.0, steps_past_warmup / decay_duration)
            k = 1 + (base_K - 1) * 0.5 * (1 + math.cos(math.pi * progress))
            return max(1, round(k))

        else:
            return base_K


# ── Factory ──────────────────────────────────────────────────────────────


def build_noise_selector(config: dict) -> NoiseSelector:
    """Instantiate the appropriate NoiseSelector from training config.

    Reads config["training"]["noise_selector"]. Absent or type="random"
    returns the default RandomNoiseSelector (zero overhead).
    """
    training_cfg = config.get("training", config)
    ns_cfg = training_cfg.get("noise_selector", {})
    selector_type = ns_cfg.get("type", "random")

    if selector_type == "explorative":
        return ExplorativeNoiseSelector(ns_cfg)
    elif selector_type == "random":
        return RandomNoiseSelector()
    else:
        logger.warning(
            f"Unknown noise_selector type '{selector_type}', "
            f"falling back to random"
        )
        return RandomNoiseSelector()
