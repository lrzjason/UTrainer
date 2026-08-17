"""NoiseSelector — pluggable noise/timestep selection strategy for training.

The training loop delegates noise and sigma sampling to a NoiseSelector:
    noises, sigmas, timesteps = self.noise_selector.select(...)

Default: RandomNoiseSelector (standard random sampling, zero overhead).
Explorative: ExplorativeNoiseSelector (best-of-K noise exploration with
stop-gradient forwards — VRAM identical to standard training).
Explorative Improved: ExplorativeImprovedNoiseSelector (best-of-K noise
exploration with per-sigma-bucket ``global_min_loss`` baselines and min-rule
early stopping — see md/xm_improved_search.md §4/§5).

Config:
    "training": {
        "noise_selector": {
            "type": "explorative_improved", // "random" (default) | "explorative" | "explorative_improved"
            "K_cond": 4,             // noise candidates for text-conditioned batches
            "K_uncond": 1,           // caption-dropped (uncond) batches: no exploration (empirically little gain)
            "warmup_steps": 0,       // K=1 for first N steps (0 = explore from step 1)
            "schedule": "constant",  // "constant" | "linear_decay" | "cosine"
            "log_stats": true,       // expose xm/* stats to callbacks
            "num_buckets": 20,       // per-sigma-bucket count for global_min_loss baselines
            "bucket_mode": "log",    // "linear" | "log" sigma bucketing
            "sigma_min": 0.001,      // sigma floor for the log bucket mapping
            "ema_alpha": 0.1,        // EMA smoothing for per-bucket baseline: g_b ← (1-α)·g_b + α·L_min
            "init_k": 0,             // first-hit candidate budget for uninitialized buckets (0 = disabled; e.g. 100 when K=10)
            "combo_norm": false,     // normalize loss by per-combo scale before the sigma baseline (md/xm_combo_normalization.md)
        }
    }

Reference: md/explorative_implementation.md, md/xm_improved_implementation.md
"""
from __future__ import annotations

import logging
import math
import zlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _combo_key_from_batch(batch: dict) -> str:
    """Canonical training-combination id from the batch's resolved batch_config.

    One batch_config == one combo (target × caption × reference). Missing
    fields fall back to "?" / "none" so keys stay stable across samples.
    Used by the improved selector's per-combo loss-scale normalization
    (md/xm_combo_normalization.md).
    """
    bc_list = batch.get("batch_configs", []) if isinstance(batch, dict) else []
    bc = bc_list[0] if bc_list else None
    if isinstance(bc, dict):
        target = bc.get("target_config", "?")
        caption = bc.get("caption_config", "?")
        reference = bc.get("reference_config") or "none"
        return f"{target}|{caption}|{reference}"
    return "__unknown__"


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

    Adaptive K (empirical):
    - Text-conditioned batches have narrow p(x|prompt) → exploration helps
      (K_cond, default 4; the paper's XRAE SOTA uses K=2).
    - Unconditional batches (caption dropped) have a single "average" mode,
      so best-of-K mostly re-picks similar noises and brings little gain —
      set K_uncond=1 to disable exploration (falls to the random-noise path).
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
                    f"[XM] step={step} K=1 "
                    f"mode={'uncond' if is_uncond else 'cond'} — random noise"
                )
            return noises, sigmas, timesteps

        # ── Phase 1: Explore K noise candidates (no_grad, no activations) ──
        sigmas_b = sigmas.view(-1, *(1,) * (target_latents[0].ndim - 1)) if target_latents[0].ndim >= 2 else sigmas
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
                loss_k = self._eval_velocity_mse(
                    unpacked_k,
                    noises_k,
                    target_latents,
                    velocity_sign=getattr(adapter, "velocity_sign", "standard"),
                )

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
                f"[XM] step={step} K={K} best_k={best_k_idx} "
                f"gap={worst_loss - best_loss:.4f} min={best_loss:.4f}"
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
        velocity_sign: str = "standard",
    ) -> torch.Tensor:
        """Compute flow-matching velocity MSE for ranking.

        The velocity-target direction follows the adapter's velocity_sign
        convention (same source as losses/flow_matching.py):
          - "standard" (noise-ward): velocity_target = noise - clean_latent
          - "data_ward" (e.g. MiniMax-H3): velocity_target = clean_latent - noise
        loss = MSE(predicted_velocity, target_velocity)

        Default "standard" keeps legacy 4D adapters (krea2, ...) bit-identical.
        Used only for argmin ranking during exploration.
        """
        if velocity_sign not in ("standard", "data_ward"):
            raise ValueError(
                f"Unsupported velocity_sign {velocity_sign!r}; "
                "expected 'standard' or 'data_ward'"
            )
        total = torch.tensor(0.0, device=target_latents[0].device)
        for unpacked_i, noise_i, target_i in zip(unpacked, noises, target_latents):
            if velocity_sign == "data_ward":
                velocity_target = target_i - noise_i
            else:
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


# ── Explorative Improved: per-sigma-bucket min-rule early stopping ───────


class ExplorativeImprovedNoiseSelector(ExplorativeNoiseSelector):
    """Best-of-K noise exploration with per-sigma-bucket min-rule early stopping.

    Extends ExplorativeNoiseSelector with adaptive early stopping
    (md/xm_improved_search.md §4/§5): the sigma range is split into
    ``num_buckets`` per-timestep buckets; each bucket independently tracks a
    ``global_min_loss`` baseline — the typical full-K best loss observed in that
    bucket (EMA update). Within a round, candidates are sampled one by
    one; as soon as the running min loss reaches the bucket baseline
    (min-rule), exploration stops early.

    Only the min-rule + per-bucket variant is implemented (the gap rule and
    single-tracker variants were rejected by simulation — see the research doc).

    ``init_k`` (first-hit exploration): an uninitialized bucket (g_b == 0)
    runs ``init_k`` candidates on its very first hit to seed a high-quality
    baseline instead of the normal K. This happens once per bucket — after
    adoption the baseline is non-zero, so the bucket reverts to normal K —
    and since g_b == 0 the min-rule can never fire, the init round always
    runs the full ``init_k`` budget (never stops early). ``init_k=0``
    disables the feature.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.num_buckets: int = int(config.get("num_buckets", 20))
        self.bucket_mode: str = config.get("bucket_mode", "log")
        if self.bucket_mode not in ("linear", "log"):
            raise ValueError(
                f"Unsupported bucket_mode {self.bucket_mode!r}; "
                "expected 'linear' or 'log'"
            )
        if self.num_buckets < 1:
            raise ValueError(
                f"num_buckets must be >= 1, got {self.num_buckets}"
            )
        self.sigma_min: float = float(config.get("sigma_min", 1e-3))
        self.ema_alpha: float = float(config.get("ema_alpha", 0.1))
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        self.init_k: int = int(config.get("init_k", 0))
        if self.init_k < 0:
            raise ValueError(f"init_k must be >= 0, got {self.init_k}")
        self.combo_norm: bool = bool(config.get("combo_norm", False))

        # Per-bucket baselines: global_min_loss[b] = typical full-K best loss
        # observed in bucket b. 0.0 = uninitialized → first round always runs
        # the full K and adopts the observed best (never stops).
        self.global_min_loss: List[float] = [0.0] * self.num_buckets
        self.total_candidates: int = 0
        self.total_stopped_early: int = 0
        # Per-combo raw-loss scale EMA (only populated when combo_norm is on).
        self.combo_mu: Dict[str, float] = {}

        logger.info(
            f"ExplorativeImprovedNoiseSelector: K_cond={self.K_cond}, "
            f"K_uncond={self.K_uncond}, warmup={self.warmup_steps}, "
            f"schedule={self.schedule}, num_buckets={self.num_buckets}, "
            f"bucket_mode={self.bucket_mode}, sigma_min={self.sigma_min}, "
            f"ema_alpha={self.ema_alpha}, init_k={self.init_k}, "
            f"combo_norm={self.combo_norm}"
        )

    # ── Bucket mapping ────────────────────────────────────────────────

    def _bucket_for_sigma(self, sigma: float) -> int:
        """Map a sigma value to its per-timestep bucket index (0..num_buckets-1).

        linear: u = clamp(sigma, 0, 1); b = min(B-1, floor(u * B))
        log:    u spreads log(sigma) evenly across [sigma_min, 1]
                (u = 0 for sigma <= sigma_min, u = 1 for sigma = 1)
        """
        if self.bucket_mode == "linear":
            u = min(1.0, max(0.0, sigma))
        else:  # "log" — validated in __init__
            u = min(
                1.0,
                max(
                    0.0,
                    (math.log(max(sigma, self.sigma_min)) - math.log(self.sigma_min))
                    / (-math.log(self.sigma_min)),
                ),
            )
        return min(self.num_buckets - 1, int(u * self.num_buckets))

    # ── Selection with early stopping ─────────────────────────────────

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
            # Warmup or schedule decayed to 1 → standard random noise.
            # NO baseline update on this path (random path must stay neutral).
            noises = [torch.randn_like(tl) for tl in target_latents]
            if self.log_stats:
                self.last_stats = {
                    "xm/K_effective": 1,
                    "xm/uncond": float(is_uncond),
                }
                self.last_log_line = (
                    f"[XM-IMP] step={step} K=1 "
                    f"mode={'uncond' if is_uncond else 'cond'} — random noise"
                )
            return noises, sigmas, timesteps

        # ── Bucket + baseline for this round's sigma ────────────────────
        # batch > 1: bucket from the FIRST sample's sigma — all candidates in
        # a round share the same sampled sigma (md/xm_improved_search.md §7).
        sigmas_b = sigmas.view(-1, *(1,) * (target_latents[0].ndim - 1)) if target_latents[0].ndim >= 2 else sigmas
        b = self._bucket_for_sigma(sigmas.flatten()[0].item())
        g_b = self.global_min_loss[b]

        # ── init round (first-hit exploration) ───────────────────────────
        # Uninitialized bucket (g_b == 0): first hit runs init_k candidates
        # to seed a high-quality baseline. g_b == 0 → the min-rule can
        # never fire, so the init round always runs the full init_k budget.
        # After adoption the baseline is non-zero → normal K for this bucket.
        loop_K = self.init_k if (g_b == 0 and self.init_k > 0) else K
        init_round = loop_K != K

        # ── Combo identity + per-combo loss scale (combo_norm) ───────────
        # loss ≈ mu[combo] × h[sigma] (md/xm_combo_normalization.md): the
        # sigma baseline below tracks h on the NORMALIZED loss, so early-stop
        # is comparable across training combinations.
        combo_key = _combo_key_from_batch(batch) if self.combo_norm else ""
        mu_pre = self.combo_mu.get(combo_key) if self.combo_norm else None

        # Early-stop threshold (pre-round values). combo_norm scales the
        # sigma baseline by the combo's loss scale; a combo's first round
        # (mu unknown) has no threshold → full-K exploration to seed mu.
        if self.combo_norm:
            thresh = (mu_pre * g_b) if (mu_pre is not None and g_b > 0) else None
        else:
            thresh = g_b if g_b > 0 else None

        best_loss = float("inf")
        worst_loss = float("-inf")
        best_noises = None
        best_k_idx = -1
        all_losses: List[float] = []
        stopped_early = False

        # ── Phase 1: Explore K noise candidates (no_grad, no activations) ──
        for i in range(loop_K):
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
                loss_k = self._eval_velocity_mse(
                    unpacked_k,
                    noises_k,
                    target_latents,
                    velocity_sign=getattr(adapter, "velocity_sign", "standard"),
                )

            loss_val = loss_k.item()
            all_losses.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_noises = [n.clone() for n in noises_k]
                best_k_idx = i
            if loss_val > worst_loss:
                worst_loss = loss_val

            # Min-rule early stop: stop as soon as the running min reaches
            # the (normalized) bucket baseline — no minimum-candidate floor.
            #   - threshold available (baseline initialized; combo scale known)
            #   - running min <= threshold (only a GOOD candidate triggers it)
            if thresh is not None and best_loss <= thresh:
                stopped_early = True
                del pred_k, unpacked_k, input_k, noisy_k, noises_k
                break

            # Immediate release — no accumulation across K iterations
            del pred_k, unpacked_k, input_k, noisy_k, noises_k

        # ── Bookkeeping after the round ────────────────────────────────
        # i + 1 == number of candidates actually evaluated (full run: K).
        self.total_candidates += (i + 1)
        # Per-combo loss-scale EMA (raw loss) — absorbs combo difficulty when
        # combo_norm is on (md/xm_combo_normalization.md).
        if self.combo_norm:
            if mu_pre is None:
                # First round for this combo: adopt its best as the scale.
                self.combo_mu[combo_key] = best_loss
                mu_now = best_loss
                l_norm = 1.0
            else:
                self.combo_mu[combo_key] = (
                    (1.0 - self.ema_alpha) * mu_pre + self.ema_alpha * best_loss
                )
                mu_now = mu_pre
                l_norm = best_loss / mu_pre
        else:
            mu_now = None
            l_norm = best_loss
        # EMA baseline update (no ratchet): g_b tracks the bucket's *typical*
        # normalized best, not its historical minimum.
        #   - first hit (g_b == 0): adopt l_norm directly.
        #   - otherwise: g_b ← (1-α)·g_b + α·l_norm.
        if g_b == 0:
            self.global_min_loss[b] = l_norm
        else:
            self.global_min_loss[b] = (
                (1.0 - self.ema_alpha) * g_b + self.ema_alpha * l_norm
            )
        if stopped_early:
            self.total_stopped_early += 1

        # ── Record exploration statistics ──
        if self.log_stats:
            import statistics
            mean_loss = sum(all_losses) / len(all_losses)
            self.last_stats = {
                "xm/K_effective": i + 1,
                "xm/uncond": float(is_uncond),
                "xm/best_k": best_k_idx,
                "xm/loss_best": best_loss,
                "xm/loss_worst": worst_loss,
                "xm/loss_mean": mean_loss,
                "xm/gap": worst_loss - best_loss,
                "xm/loss_std": statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0,
                "xm/bucket": b,
                "xm/baseline": self.global_min_loss[b],
                "xm/stopped_early": 1.0 if stopped_early else 0.0,
                "xm/init_round": 1.0 if init_round else 0.0,
            }
            if self.combo_norm:
                self.last_stats["xm/loss_norm"] = l_norm
                self.last_stats["xm/combo_mu"] = mu_now
                self.last_stats["xm/combo_id"] = (
                    zlib.crc32(combo_key.encode("utf-8")) & 0x7FFFFFFF
                )
            # Terminal line stays minimal (K, best_k, gap, min, thr); the
            # full breakdown lives in last_stats → wandb (loss/xm/*).
            # thr = the raw-unit threshold min is compared against
            # (combo_norm: mu[combo] × g_b; otherwise g_b). "-" = no
            # comparison this round (init round or combo first round).
            thr_str = f"{thresh:.4f}" if thresh is not None else "-"
            self.last_log_line = (
                f"[XM-IMP] step={step} K={i + 1}/{loop_K} "
                f"best_k={best_k_idx} "
                f"gap={worst_loss - best_loss:.4f} min={best_loss:.4f} "
                f"max={worst_loss:.4f} "
                f"thr={thr_str}"
            )

        # ── Phase 2: Return winning noise (sigma unchanged) ──
        return best_noises, sigmas, timesteps

    # ── State persistence (checkpoint extra_state) ────────────────────

    def state_dict(self) -> dict:
        """Serializable selector state (baselines + counters) for checkpoints."""
        return {
            "version": 2,
            "global_min_loss": list(self.global_min_loss),
            "total_candidates": self.total_candidates,
            "total_stopped_early": self.total_stopped_early,
            "combo_mu": dict(self.combo_mu),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore baselines + counters from a state_dict.

        Missing keys (including absent "version") fall back to defaults — no
        exception. If the saved baseline list length differs from num_buckets
        (bucket count changed between runs), the shared prefix is kept and the
        missing tail re-initialized to 0.0 (first-round adoptions re-run there).
        """
        if not isinstance(state, dict):
            logger.warning(
                f"load_state_dict: expected dict, got {type(state).__name__}; "
                "keeping current state"
            )
            return
        self.total_candidates = int(state.get("total_candidates", 0))
        self.total_stopped_early = int(state.get("total_stopped_early", 0))

        state_list = state.get("global_min_loss", [])
        if not isinstance(state_list, (list, tuple)):
            logger.warning(
                "load_state_dict: 'global_min_loss' is not a list/tuple; "
                "re-initializing baselines to 0.0"
            )
            state_list = []
        if len(state_list) != self.num_buckets:
            keep = min(len(state_list), self.num_buckets)
            logger.warning(
                f"load_state_dict: saved global_min_loss length "
                f"{len(state_list)} != num_buckets {self.num_buckets}; "
                f"keeping first {keep} entries, re-initializing the rest"
            )
            self.global_min_loss = (
                [float(v) for v in state_list[:keep]]
                + [0.0] * (self.num_buckets - keep)
            )
        else:
            self.global_min_loss = [float(v) for v in state_list]

        combo_mu = state.get("combo_mu", {})
        if isinstance(combo_mu, dict):
            self.combo_mu = {str(k): float(v) for k, v in combo_mu.items()}
        else:
            logger.warning(
                "load_state_dict: 'combo_mu' is not a dict; "
                "re-initializing combo scales to empty"
            )
            self.combo_mu = {}


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
    elif selector_type == "explorative_improved":
        return ExplorativeImprovedNoiseSelector(ns_cfg)
    elif selector_type == "random":
        return RandomNoiseSelector()
    else:
        logger.warning(
            f"Unknown noise_selector type '{selector_type}', "
            f"falling back to random"
        )
        return RandomNoiseSelector()
