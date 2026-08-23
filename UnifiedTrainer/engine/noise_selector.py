"""NoiseSelector — pluggable noise/timestep selection strategy for training.

The training loop delegates noise and sigma sampling to a NoiseSelector:
    noises, sigmas, timesteps = self.noise_selector.select(...)

Default: RandomNoiseSelector (standard random sampling, zero overhead).
Explorative: ExplorativeNoiseSelector (best-of-K noise exploration with
stop-gradient forwards — VRAM identical to standard training).
XMS: ExplorativeSequentialNoiseSelector (type "explorative_sequential";
best-of-K with within-round sequential early stopping — TARGET/EI/streak
rules. ALL stopping evidence is computed from candidates observed in the
current round; the only cross-round state is per-sigma-bucket sigma_hat
(EWMA of within-round candidate-loss std) plus visit counts).

Config:
    "training": {
        "noise_selector": {
            "type": "explorative",  // "random" (default) | "explorative"
            "K_cond": 4,             // noise candidates for text-conditioned batches
            "K_uncond": 1,           // caption-dropped (uncond) batches: no exploration (empirically little gain)
            "warmup_steps": 0,       // K=1 for first N steps (0 = explore from step 1)
            "schedule": "constant",  // "constant" | "linear_decay" | "cosine"
            "log_stats": true,       // expose xm/* stats to callbacks
        }
    }

    // XMS: sequential best-of-K with within-round early stopping
    "training": {
        "noise_selector": {
            "type": "explorative_sequential",
            "K_cond": 4,             // candidates per text-conditioned batch
            "K_uncond": 4,           // candidates per caption-dropped batch
            "warmup_steps": 0,       // K=1 for first N steps
            "schedule": "constant",  // "constant" | "linear_decay" | "cosine"
            "log_stats": true,
            "stop_rule": "target",   // "target" | "ei" | "streak" | "none"
            "gamma": 0.8,            // target rule: stop when mu_hat - best >= gamma * alpha(K) * sigma_eff
            "kappa": 0.05,           // ei rule: stop when EI of one more draw < kappa * sigma_eff
            "streak_s": 3,           // streak rule: consecutive candidates failing to improve best
            "m_min": 2,              // min candidates observed before any stop check
            "p_raw": 0.15,           // prob. to reinject a purely random candidate (anti-degeneracy)
            "sigma_floor_frac": 0.001,  // sigma_eff floor as fraction of |mu_hat|
                                        // (keep small: high-loss rounds inflate the
                                        // threshold and force full exploration)
            "sigma_ewma": 0.01,      // EWMA alpha for sigma_hat (0 < alpha <= 1)
            "sigma_mix": 0.5,        // blend cross-round sigma_hat with within-round stdev:
                                     // 1.0 = pure cross-round, 0.0 = pure within-round,
                                     // 0.5 = hybrid (tight rounds stop early, tail kept)
            "mu_estimator": "loo",   // "loo" (exclude incumbent, winner's-curse guard) | "mean"
            "per_sample_winner": false,  // B>1: each sample takes its own best candidate's noise
            "num_buckets": 20,       // log-scaled sigma buckets
            "sigma_min": 0.001,      // sigma floor for bucket mapping
        }
    }

Reference: md/explorative_implementation.md; XMS design:
.tmp/ref_xms/xm_exploration_research.md
"""
from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from statistics import NormalDist, stdev
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── XMS math helpers ──────────────────────────────────────────────────────


_NDIST = NormalDist()
_ALPHA_CACHE: dict = {}


def alpha_of_m(m: int) -> float:
    """alpha_m = -E[min of m iid N(0,1)] (expected best-of-m gain, in sigma units).

    E[min of m] = m * int_0^1 Phi^-1(u) (1-u)^(m-1) du; alpha_m = -that.
    Values: 1->0, 2->0.564, 3->0.846, 5->1.163, 10->1.539, 20->1.867, 50->2.249.
    Asymptotic ~ sqrt(2 ln m) (Hall 1979).
    """
    m = int(m)
    if m <= 1:
        return 0.0
    if m in _ALPHA_CACHE:
        return _ALPHA_CACHE[m]
    n = 20000
    acc = 0.0
    for i in range(n):
        u = (i + 0.5) / n
        acc += _NDIST.inv_cdf(u) * (1.0 - u) ** (m - 1)
    val = -(m * acc / n)
    _ALPHA_CACHE[m] = val
    return val


def _pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ei_one_draw(best: float, mu: float, sig: float) -> float:
    """E[(best - X)^+] for X~N(mu, sig^2) = sig * (z*Phi(z) + phi(z)), z=(best-mu)/sig.

    Closed form of the expected improvement of evaluating one more candidate
    against the incumbent best. Sanity: z=0 -> 0.399*sig; z=-1 -> 0.083*sig;
    z=-3 -> 0.0013*sig. Guard: z < -8 (deep tail) -> 0; sig <= 0 -> 0.
    """
    if sig <= 0.0:
        return 0.0
    z = (best - mu) / sig
    if z < -8.0:
        return 0.0
    return sig * (z * _cdf(z) + _pdf(z))


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

            # DIAG: per-candidate footprint (first run only) — shows if the
            # fp8->bf16 materialized weights accumulate across K forwards.
            if k < 3 and getattr(self, "_diag_xm_k", True):
                try:
                    self._diag_xm_k = False
                    import torch as _t
                    _a = _t.cuda.memory_allocated() / 1024**3
                    _r = _t.cuda.memory_reserved() / 1024**3
                    _p = _t.cuda.max_memory_allocated() / 1024**3
                    # batch 关键张量
                    _bt = ""
                    try:
                        _nos = noises_k[0] if isinstance(noises_k, list) else noises_k
                        _bt = f"noise_shape={tuple(_nos.shape)} "
                    except Exception:
                        pass
                    try:
                        _tl0 = target_latents[0]
                        _bt += f"tgt={tuple(_tl0.shape)}@{_tl0.device.type} "
                    except Exception:
                        pass
                    try:
                        _embs = batch.get("embeddings")
                        if _embs and _embs[0] is not None:
                            _pe = _embs[0].get("prompt_embed") if isinstance(_embs[0], dict) else _embs[0]
                            import numpy as _np
                            if isinstance(_pe, _np.ndarray):
                                _bt += f"emb_shape={_pe.shape} "
                    except Exception:
                        pass
                    logger.info(
                        f"[DIAG-XM-K] k={k} after_forward alloc={_a:.2f}G "
                        f"resv={_r:.2f}G peak={_p:.2f}G {_bt}"
                    )
                except Exception as _de:
                    logger.warning(f"[DIAG-XM-K] failed: {_de}")

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


# ── XMS: sequential best-of-K with within-round early stopping ─────────────


class ExplorativeSequentialNoiseSelector(ExplorativeNoiseSelector):
    """XMS: best-of-K noise exploration with within-round sequential stopping.

    Design (research: .tmp/ref_xms/xm_exploration_research.md, §5):
      1. ALL stopping evidence is WITHIN the round: mu_hat is estimated from
         the candidates observed so far in this round (leave-one-out: excludes
         the incumbent best to avoid winner's curse). The only cross-round
         state is per-sigma-bucket sigma_hat (EWMA of within-round candidate
         loss std) plus visit counts. NO cross-round absolute loss levels,
         NO global_min_loss, NO dead zones, NO combo normalization.
      2. Stop rules (checked after each candidate, m = candidates so far):
           target: mu_hat_loo - best >= gamma * alpha(K) * sigma_eff
           ei:     ei_one_draw(best, mu_hat, sigma_eff) < kappa * sigma_eff
           streak: s consecutive candidates failing to improve best
           none:   always run full K
         m_min floor (default 2): only check rules when m >= m_min.
         sigma_eff = max(sigma_hat[b], sigma_floor_frac * abs(mu_hat)).
         First visit to a bucket (sigma_hat[b] is None): run FULL K
         (calibration, no stop checks), then set sigma_hat[b] = stdev(observed).
      3. Guards: p_raw probability of reinjecting a purely random candidate
         (anti-degeneracy; keeps the unselected noise distribution in the
         gradient stream). sigma_hat EWMA alpha = sigma_ewma (default 0.01).
    """

    def __init__(self, config: dict):
        super().__init__(config)  # inherits K_cond/K_uncond/warmup/schedule/log_stats
        self.stop_rule: str = config.get("stop_rule", "target")
        if self.stop_rule not in ("target", "ei", "streak", "none"):
            raise ValueError(
                f"stop_rule must be one of target/ei/streak/none, got {self.stop_rule!r}"
            )
        self.gamma: float = float(config.get("gamma", 0.8))
        self.kappa: float = float(config.get("kappa", 0.05))
        self.streak_s: int = int(config.get("streak_s", 3))
        # m_min floor: mu_hat needs >= 2 observed losses to be meaningful.
        self.m_min: int = max(2, int(config.get("m_min", 2)))
        self.p_raw: float = float(config.get("p_raw", 0.15))
        self.sigma_floor_frac: float = float(config.get("sigma_floor_frac", 0.005))
        self.sigma_ewma: float = float(config.get("sigma_ewma", 0.01))
        if not (0.0 < self.sigma_ewma <= 1.0):
            raise ValueError(
                f"sigma_ewma must satisfy 0 < sigma_ewma <= 1, got {self.sigma_ewma!r}"
            )
        # sigma_mix: blend cross-round sigma_hat with the within-round stdev.
        #   sigma_eff = sigma_mix * sigma_hat + (1 - sigma_mix) * stdev(observed)
        # 1.0 = pure cross-round EWMA (original); 0.0 = pure within-round;
        # 0.5 = hybrid (default): within-round component lets tight rounds stop
        # early, cross-round component guards against single-round stdev noise
        # (avoids premature stops that miss tail winners).
        self.sigma_mix: float = float(config.get("sigma_mix", 0.5))
        if not (0.0 <= self.sigma_mix <= 1.0):
            raise ValueError(
                f"sigma_mix must satisfy 0 <= sigma_mix <= 1, got {self.sigma_mix!r}"
            )
        self.mu_estimator: str = config.get("mu_estimator", "loo")
        if self.mu_estimator not in ("loo", "mean"):
            raise ValueError(
                f"mu_estimator must be 'loo' or 'mean', got {self.mu_estimator!r}"
            )
        self.per_sample_winner: bool = bool(config.get("per_sample_winner", False))
        self.num_buckets: int = int(config.get("num_buckets", 20))
        self.sigma_min: float = float(config.get("sigma_min", 1e-3))

        # Cross-round state: ONLY per-bucket sigma_hat (stable scale) + counts.
        self.sigma_hat: List[Optional[float]] = [None] * self.num_buckets
        self.visit_count: List[int] = [0] * self.num_buckets
        self.total_candidates: int = 0
        self.total_rounds: int = 0
        self.total_stopped: int = 0
        self.total_p_raw: int = 0

        logger.info(
            f"ExplorativeSequentialNoiseSelector: stop_rule={self.stop_rule}, "
            f"gamma={self.gamma}, kappa={self.kappa}, streak_s={self.streak_s}, "
            f"m_min={self.m_min}, p_raw={self.p_raw}, "
            f"sigma_floor_frac={self.sigma_floor_frac}, sigma_ewma={self.sigma_ewma}, "
            f"sigma_mix={self.sigma_mix}, "
            f"mu_estimator={self.mu_estimator}, per_sample_winner={self.per_sample_winner}, "
            f"num_buckets={self.num_buckets}, sigma_min={self.sigma_min}"
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _bucket_for_sigma(self, sigma: float) -> int:
        """Log-scaled bucket index (same mapping the old bucket-min selector used).

        u = clamp((log(max(sigma, sigma_min)) - log(sigma_min)) / (-log(sigma_min)), 0, 1);
        b = min(B-1, int(u * B)).
        """
        s = max(float(sigma), self.sigma_min)
        u = (math.log(s) - math.log(self.sigma_min)) / (-math.log(self.sigma_min))
        u = max(0.0, min(1.0, u))
        return min(self.num_buckets - 1, int(u * self.num_buckets))

    def _mu_hat(self, observed: List[float], best: float) -> float:
        """Within-round mean estimate.

        "loo" (default) excludes the incumbent best (winner's-curse guard,
        Efron 2011): including the selected min biases mu_hat downward, and the
        stop-triggering event is exactly a low-mu round. Falls back to the plain
        mean when there is only one observation or mu_estimator="mean".
        """
        if self.mu_estimator == "mean" or len(observed) <= 1:
            return sum(observed) / len(observed)
        return (sum(observed) - best) / (len(observed) - 1)

    @staticmethod
    def _eval_velocity_mse_per_sample(
        unpacked: List[torch.Tensor],
        noises: List[torch.Tensor],
        target_latents: List[torch.Tensor],
        velocity_sign: str = "standard",
    ) -> torch.Tensor:
        """Per-sample flow-matching velocity MSE vector [B].

        Same velocity convention as _eval_velocity_mse, but returns one loss
        per sample (mean over targets of the per-position MSE averaged over
        non-batch dims). Used to assemble the per_sample_winner noise.
        """
        if velocity_sign not in ("standard", "data_ward"):
            raise ValueError(
                f"Unsupported velocity_sign {velocity_sign!r}; "
                "expected 'standard' or 'data_ward'"
            )
        B = target_latents[0].shape[0]
        total = torch.zeros(B, device=target_latents[0].device)
        for unpacked_i, noise_i, target_i in zip(unpacked, noises, target_latents):
            if velocity_sign == "data_ward":
                velocity_target = target_i - noise_i
            else:
                velocity_target = noise_i - target_i
            per_pos = (unpacked_i.float() - velocity_target.float()) ** 2
            total = total + per_pos.flatten(1).mean(dim=1)
        return total / len(target_latents)

    # ── Main selection loop ──────────────────────────────────────────────

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
            # Warmup or schedule decayed to 1 → standard random noise, no state
            # updates (same minimal stats as the parent's warmup path).
            noises = [torch.randn_like(tl) for tl in target_latents]
            if self.log_stats:
                self.last_stats = {
                    "xm/K_effective": 1,
                    "xm/uncond": float(is_uncond),
                }
                self.last_log_line = (
                    f"[XMS] step={step} K=1 "
                    f"mode={'uncond' if is_uncond else 'cond'} — random noise"
                )
            return noises, sigmas, timesteps

        b = self._bucket_for_sigma(sigmas.flatten()[0].item())
        sig_hat = self.sigma_hat[b]
        first_visit = sig_hat is None
        alpha_K = alpha_of_m(K)

        sigmas_b = (
            sigmas.view(-1, *(1,) * (target_latents[0].ndim - 1))
            if target_latents[0].ndim >= 2
            else sigmas
        )
        velocity_sign = getattr(adapter, "velocity_sign", "standard")

        observed: List[float] = []
        best = float("inf")
        best_noise = None
        best_k = -1
        ps_best = None    # [B] per-sample best losses (per_sample_winner mode)
        ps_noise = None   # list of per-target [B, ...] per-sample winner noise
        used_rule = "full"

        # ── Phase 1: sequential exploration with within-round stop checks ──
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
                    velocity_sign=velocity_sign,
                )
                if self.per_sample_winner:
                    loss_vec = self._eval_velocity_mse_per_sample(
                        unpacked_k,
                        noises_k,
                        target_latents,
                        velocity_sign=velocity_sign,
                    )

            loss_val = loss_k.item()
            observed.append(loss_val)
            if loss_val < best:
                best = loss_val
                best_noise = [n.clone() for n in noises_k]
                best_k = k

            if self.per_sample_winner:
                # Per-sample winner bookkeeping: each sample tracks its own
                # best candidate (reuses the [B] loss vector, 0 extra forwards).
                if ps_best is None:
                    ps_best = loss_vec.clone()
                    ps_noise = [n.clone() for n in noises_k]
                else:
                    mask = loss_vec < ps_best
                    if bool(mask.any()):
                        ps_best = ps_best.masked_scatter(mask, loss_vec[mask])
                        for t in range(len(noises_k)):
                            ps_noise[t][mask] = noises_k[t][mask]

            # Immediate release — no accumulation across K iterations
            del pred_k, unpacked_k, input_k, noisy_k, noises_k
            if self.per_sample_winner:
                del loss_vec, loss_k
            else:
                del loss_k

            m = len(observed)
            if first_visit:
                continue  # calibration round: full K, no stop checks
            if m < self.m_min or m >= K:
                continue
            if sig_hat is None or sig_hat <= 0.0:
                break  # no usable scale evidence: stop at the m_min floor
            mu_hat = self._mu_hat(observed, best)
            # sigma_mix: blend cross-round sigma_hat with within-round stdev.
            # Within-round component lets tight rounds stop early; cross-round
            # component guards against single-round stdev noise (premature
            # stops that miss tail winners). First-visit (sig_hat None) and
            # sigma_mix=0 fall back to pure within-round scale.
            within_sd = stdev(observed) if len(observed) >= 2 else 0.0
            if self.sigma_mix <= 0.0 or sig_hat is None:
                scale = within_sd
            else:
                scale = self.sigma_mix * sig_hat + (1.0 - self.sigma_mix) * within_sd
            sig_eff = max(scale, self.sigma_floor_frac * abs(mu_hat))
            if sig_eff <= 0.0:
                continue  # safety: keep exploring when scale is degenerate
            if self.stop_rule == "target":
                if mu_hat - best >= self.gamma * alpha_K * sig_eff:
                    used_rule = "target"
                    break
            elif self.stop_rule == "ei":
                if ei_one_draw(best, mu_hat, sig_eff) < self.kappa * sig_eff:
                    used_rule = "ei"
                    break
            elif self.stop_rule == "streak":
                # Distribution-free: count trailing candidates that failed to
                # set a strictly-new running best.
                run_best = float("inf")
                bests = []
                for lv in observed:
                    run_best = min(run_best, lv)
                    bests.append(run_best)
                streak = 0
                for i in range(len(observed) - 1, -1, -1):
                    if observed[i] <= bests[i]:
                        break
                    streak += 1
                if streak >= self.streak_s:
                    used_rule = "streak"
                    break
            # stop_rule == "none": never stop early

        m = len(observed)

        # ── Cross-round update: ONLY sigma_hat (stable scale) + counters ──
        if m >= 2:
            sd = stdev(observed)
            if self.sigma_hat[b] is None:
                self.sigma_hat[b] = sd
            else:
                a = self.sigma_ewma
                self.sigma_hat[b] = (1.0 - a) * self.sigma_hat[b] + a * sd
        self.visit_count[b] += 1
        self.total_candidates += m
        self.total_rounds += 1
        if used_rule != "full":
            self.total_stopped += 1

        # ── Phase 2: winner assembly ──
        if self.per_sample_winner and ps_noise is not None:
            winner = ps_noise
            ps_used = 1
        else:
            winner = best_noise
            ps_used = 0

        # p_raw guard: reinject a purely random candidate with probability p_raw
        p_raw_used = 0
        if self.p_raw > 0.0 and random.random() < self.p_raw:
            winner = [torch.randn_like(tl) for tl in target_latents]
            self.total_p_raw += 1
            p_raw_used = 1

        # ── Record exploration statistics ──
        if self.log_stats:
            mu_hat = self._mu_hat(observed, best)
            self.last_stats = {
                "xm/K_effective": m,
                "xm/uncond": float(is_uncond),
                "xm/best_k": best_k,
                "xm/loss_best": best,
                "xm/loss_worst": max(observed),
                "xm/loss_mean": sum(observed) / m,
                "xm/gap": max(observed) - min(observed),
                "xm/loss_std": stdev(observed) if m > 1 else 0.0,
                "xm/mu_hat": mu_hat,
                "xm/sigma_hat": self.sigma_hat[b] if self.sigma_hat[b] is not None else 0.0,
                "xm/stop_rule": used_rule,
                "xm/bucket": b,
                "xm/first_visit": 1 if first_visit else 0,
                "xm/p_raw": p_raw_used,
                "xm/total_candidates": self.total_candidates,
                "xm/total_rounds": self.total_rounds,
                "xm/total_stopped": self.total_stopped,
                "xm/total_p_raw": self.total_p_raw,
                "xm/per_sample_winner": ps_used,
            }
            self.last_log_line = (
                f"[XMS] step={step} K={m}/{K} rule={used_rule} "
                f"best_k={best_k} best={best:.4f} "
                f"gap={max(observed) - min(observed):.4f}"
            )

        return winner, sigmas, timesteps

    # ── Checkpoint state (only scale + counters — never absolute losses) ──

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "sigma_hat": [None if v is None else float(v) for v in self.sigma_hat],
            "visit_count": list(self.visit_count),
            "total_candidates": self.total_candidates,
            "total_rounds": self.total_rounds,
            "total_stopped": self.total_stopped,
            "total_p_raw": self.total_p_raw,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore cross-round state, tolerating missing keys / wrong types.

        sigma_hat is restored only when its list length equals num_buckets
        (otherwise the current state is kept and a warning is logged);
        visit_count is padded to num_buckets; counters come from state.get().
        """
        if not isinstance(state, dict):
            return
        sh = state.get("sigma_hat", [])
        if isinstance(sh, (list, tuple)) and len(sh) == self.num_buckets:
            self.sigma_hat = [None if v is None else float(v) for v in sh]
        elif sh:
            logger.warning(
                f"ExplorativeSequentialNoiseSelector: sigma_hat has length "
                f"{len(sh)} != num_buckets {self.num_buckets}; keeping current"
            )
        vc = state.get("visit_count", [])
        if isinstance(vc, (list, tuple)):
            self.visit_count = [int(v) for v in vc]
            if len(self.visit_count) < self.num_buckets:
                self.visit_count += [0] * (self.num_buckets - len(self.visit_count))
        self.total_candidates = int(state.get("total_candidates", 0))
        self.total_rounds = int(state.get("total_rounds", 0))
        self.total_stopped = int(state.get("total_stopped", 0))
        self.total_p_raw = int(state.get("total_p_raw", 0))


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
    elif selector_type == "explorative_sequential":
        return ExplorativeSequentialNoiseSelector(ns_cfg)
    elif selector_type == "random":
        return RandomNoiseSelector()
    else:
        logger.warning(
            f"Unknown noise_selector type '{selector_type}', "
            f"falling back to random"
        )
        return RandomNoiseSelector()
