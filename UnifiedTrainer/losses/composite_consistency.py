"""CompositeConsistencyLoss — pixel-space composite supervision for RGBA targets
with PixPerfect-style discriminative pixel space (A), gradient-domain seam loss
(B) and MaskFlow-style area-normalized matte terms.

Reference: UTrainer/tmp/mask_training_improvement.md (design doc), and papers:
 - PixPerfect (arXiv:2512.03247): discriminative pixel space amplifies subtle
   color mismatch between the composited prediction and GT.
 - MaskFlow (arXiv:2608.06929): mask-adaptive loss weighting + gradient-domain
   (screened Poisson) boundary consistency.

What it supervises (all in PIXEL space, decoded from x0_hat once per step):
    comp = a'*T'_rgb + (1-a')*R           (the composited result)
    vs GT composite  = a_T*T_rgb + (1-a_T)*R   (== _O to ~3/255)
Terms:
    data   : weighted L1/L2 of (comp - GT)     [weight 1.0 within loss, see
             data_weight — set 0 to keep telemetry while disabling the term]
    matte  : |a' - a_T_raw|                             (crisp shape reference)
    dice   : soft-Dice(a', a_T_raw)                     (shape overlap, optional)
    fg     : |a'*T'_rgb - a_T*T_rgb|  (premultiplied; suppresses the purple/blue
             RGB garbage the model otherwise free-wheels in the transparent zone)
    disc   : discriminative-space L1  (per-sample polynomial tone map fitted to
             amplify the observed color discrepancy; THE color-shift fix)
    grad   : gradient-domain residual |grad comp - grad GT| in the dilated matte
             region (THE seam fix; MaskFlow SPD's training-time simplification)
    cover  : ALPHA COVERAGE PENALTY (knockout-plate semantics; optional, and the
             only term that supervises no RGB at all)

             B = R - T knocks the target's content region out to transparency;
             X = B + T' must come back fully opaque.  Any transparency left in
             X means T' did not cover all pixels the target asks for:

                 X_alpha(p) = (1 - a_ref(p)) + a'(p)      plate + prediction
                 short(p)   = relu(1 - X_alpha(p)) = relu(a_ref - a')
                 cover      = sum(relu(short - margin)) / sum(a_ref)

             One-sided BY DESIGN: only under-coverage (a' < a_ref) is punished.
             That is exactly the "missing edit region" failure of the transparent
             branch.  Over-coverage / bleed (a' > a_ref) is left to the FM loss
             and to region weighting — adding both directions at once is how the
             earlier composite runs lost to the baseline.

             cover_form='shortfall' (default) is the literal pixel arithmetic of
             "reference - target + prediction": with the knockout plate, X's
             residual transparency is exactly relu(a_T - a').  A prediction that
             MATCHES a feathered edge (a' == a_ref == 0.67) scores 0, so the
             term is satisfiable on soft edges — the data keeps a 1-2px ramp on
             purpose (see md/10 §2.4).

             cover_form='residual' uses a_T*(1-a') — the residual transparency
             of plate+layer under PROPER "over" compositing.  Same thing on
             binary regions, but at a soft edge it demands a'->1 to reach zero,
             i.e. it pushes the matte outward (bloat) exactly where the earlier
             runs learned hard alpha cliffs.  Available for an ablation, not the
             default.

             Why it can work where the comp term did not: it touches alpha only,
             so it cannot pressure RGB high frequencies (the documented blur
             root cause).  FM decides WHAT to paint; this decides WHERE the
             paint must reach.  margin makes it a constraint rather than a
             regression pull (compliant pixels contribute exactly 0), and the
             sum(a_ref) normalisation is MaskFlow-MFL-style area normalisation,
             so small targets are not diluted by the frame.

             NO FEATHERING by default (cover_alpha='raw'): the coverage reference
             is the dataset's own alpha, which already keeps a 1-2px edge, and
             the shortfall form is satisfiable on that ramp (a' == a_ref scores
             0), so blurring the reference buys nothing and only re-introduces
             the mid-alpha / translucency pull the earlier matte experiments
             were bitten by.  cover_alpha='soft' (feathered reference) is kept
             only as an explicit ablation switch; feather_sigma_px must be > 0
             for it to differ from 'raw'.

             `cover_viol` telemetry (margin-free, area-normalised, == mean of
             `short` per unit target alpha, in [0,1]) is reported on EVERY
             decoded step, including when cover_weight == 0 — that is how the
             baseline number is obtained before the term is switched on.

Hard-won guards (each corresponds to an observed failure, see
md/composite_consistency_loss.md):
    * sigma gate        : steps with sigma > sigma_max are skipped — high-noise
                          x0_hat is inherently blurry; supervising it BLURRED
                          the saved adapter.
    * VAE residency     : the VAE is acquired ONCE and held (decode graph must
                          survive backward); vae_load_mode='reload' is rejected.
    * feather + band    : edge targets are feathered and the transition band is
                          down-weighted — pixel-exact edge supervision produced
                          hard alpha cliffs (visible seams).
    * matte vs RAW alpha: matte/dice reference the unfeathered alpha so shape
                          supervision stays crisp while composite/fg targets
                          stay achievable.

Config sketch:
    {"type": "composite_consistency", "weight": 0.03, "params": {
        "sigma_max": 0.2, "metric": "l1",
        "matte_weight": 0.25, "dice_weight": 0.0, "fg_weight": 0.05,
        "feather_sigma_px": 1.0, "edge_band_drop": 0.5,
        "discriminative_space": true, "disc_beta": [20, 40],
        "disc_degree": 5, "grad_weight": 0.05, "every_n_steps": 1}}

Cover-only sketch (everything else off — one new term at a time):
    {"type": "composite_consistency", "weight": 0.03, "params": {
        "data_weight": 0.0, "matte_weight": 0.0, "fg_weight": 0.0,
        "dice_weight": 0.0, "grad_weight": 0.0,
        "cover_weight": 1.0, "cover_margin": 0.1, "cover_norm": "area",
        "cover_alpha": "raw", "cover_form": "shortfall",
        "feather_sigma_px": 0.0,
        "sigma_max": 0.2, "every_n_steps": 1}}
    Run it once with weight=0 first and read `cover_viol` as the baseline.

NOTE: decode_scale was REMOVED (2026-10-10). Half-resolution supervision caused
global blur (laplacian_var 42.8 vs 82.8 baseline) — matching downsampled pixels
only constrains low frequencies. The decode is now always at the full bucket
resolution; a decode_scale key in old configs is ignored with a warning.

Requires: RGBA VAE adapter (qwen_image21), batch_size == 1, and the
dataset/trainer plumbing that exposes extra["pixel_paths"] = {"T": png, "R": png}
(cached bucket-aligned pixel copies).  Missing plumbing -> the loss no-ops with
a one-time warning.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry

# Reuse pfm's decoder block-checkpointing (idempotent via the decoder's
# `_ut_checkpoint_wrapped` flag, so pfm + this loss can share one VAE safely).
from UnifiedTrainer.losses.pfm import PerceptualFlowMatchingLoss as _PFMRef  # noqa: E402

logger = logging.getLogger(__name__)

_PX_CACHE_VERSION = 4

# 5-tap gaussian (sigma ~1px), separable — used to feather the GT alpha so the
# edge TARGETS are soft, achievable transitions instead of pixel-exact cliffs.
_GK = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
_GK = (_GK / _GK.sum()).view(1, 1, 1, 5)


def _feather_alpha(a: torch.Tensor, passes: int) -> torch.Tensor:
    """Blur an alpha map (..., H, W) with `passes` separable 5-tap gaussian passes."""
    if passes <= 0:
        return a
    x = a.unsqueeze(0) if a.dim() == 3 else a
    c = x.shape[1]
    k = _GK.to(x.device, x.dtype)
    k_w = k.view(1, 1, 1, 5).repeat(c, 1, 1, 1)   # along W
    k_h = k.view(1, 1, 5, 1).repeat(c, 1, 1, 1)   # along H
    for _ in range(passes):
        x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"), k_w, groups=c)
        x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="replicate"), k_h, groups=c)
    return x.squeeze(0) if a.dim() == 3 else x


@LossRegistry.register("composite_consistency")
class CompositeConsistencyLoss(BaseLoss):
    """Pixel-space composite consistency loss (see module docstring)."""

    name = "composite_consistency"
    requires_vae = True  # train.py wires the shared VAE via vae_manager

    def __init__(
        self,
        weight: float = 1.0,
        decode_scale: Optional[float] = None,   # REMOVED: full-res supervision only.
        metric: str = "l1",
        data_weight: float = 1.0,
        matte_weight: float = 0.25,
        dice_weight: float = 0.0,
        fg_weight: float = 0.05,
        weight_mode: str = "union",
        feather_sigma_px: float = 1.0,
        edge_band_drop: float = 0.5,
        sigma_max: float = 0.35,
        every_n_steps: int = 1,
        disc_cap: float = 5.0,
        # C. alpha coverage penalty (knockout-plate semantics, see docstring)
        cover_weight: float = 0.0,
        cover_margin: float = 0.1,
        cover_norm: str = "area",
        cover_alpha: str = "raw",
        cover_form: str = "shortfall",
        # A. discriminative pixel space (PixPerfect)
        discriminative_space: bool = False,
        disc_beta: Optional[List[float]] = None,
        disc_degree: int = 5,
        disc_points: int = 32768,
        # B. gradient-domain seam loss (MaskFlow SPD, training-time form)
        grad_weight: float = 0.0,
        keep_vae_on_device: bool = True,
        vae_grad_checkpoint: bool = True,
        vae_dtype: str = "bf16",
        **params: Any,
    ):
        super().__init__(weight=weight, **params)
        if metric not in ("l1", "l2"):
            raise ValueError(f"composite_consistency: metric must be 'l1' or 'l2', got {metric!r}")
        if decode_scale is not None and decode_scale != 1.0:
            logger.warning(
                "composite_consistency: decode_scale is REMOVED (half-res supervision caused "
                "global blur — matching downsampled pixels only constrains low frequencies). "
                f"Ignoring decode_scale={decode_scale}; decoding at full bucket resolution.")
        self.metric = metric
        self.data_weight = float(data_weight)
        self.matte_weight = float(matte_weight)
        self.dice_weight = float(dice_weight)
        self.fg_weight = float(fg_weight)
        if weight_mode not in ("union", "gt", "none"):
            raise ValueError(
                f"composite_consistency: weight_mode must be 'union'|'gt'|'none', got {weight_mode!r}")
        self.weight_mode = weight_mode
        if cover_norm not in ("area", "mean"):
            raise ValueError(
                f"composite_consistency: cover_norm must be 'area'|'mean', got {cover_norm!r}")
        if cover_alpha not in ("soft", "raw"):
            raise ValueError(
                f"composite_consistency: cover_alpha must be 'soft'|'raw', got {cover_alpha!r}")
        if cover_form not in ("shortfall", "residual"):
            raise ValueError(
                "composite_consistency: cover_form must be 'shortfall'|'residual', "
                f"got {cover_form!r}")
        self.cover_weight = float(cover_weight)
        self.cover_margin = float(cover_margin)
        self.cover_norm = cover_norm
        self.cover_alpha = cover_alpha
        self.cover_form = cover_form
        self.grad_weight = float(grad_weight)
        self.discriminative_space = bool(discriminative_space)
        self.disc_beta = tuple(disc_beta) if disc_beta else (20.0, 40.0)
        self.disc_degree = int(disc_degree)
        self.disc_points = int(disc_points)
        self.every_n_steps = max(1, int(every_n_steps))
        self.sigma_max = float(sigma_max)
        # pathological disc-fit cap: skip the term when its value exceeds this
        self.disc_cap = float(disc_cap)
        self.feather_sigma_px = float(feather_sigma_px)
        self.edge_band_drop = float(edge_band_drop)
        self._feather_passes = int(round(max(0.0, self.feather_sigma_px)))
        self.keep_vae_on_device = bool(keep_vae_on_device)
        self.vae_grad_checkpoint = bool(vae_grad_checkpoint)
        self._vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[vae_dtype]
        self._vae: Optional[torch.nn.Module] = None
        self._held_vae: Optional[torch.nn.Module] = None
        self.vae_manager: Optional[Any] = None  # injected by Trainer (requires_vae)
        self._call_count = 0
        self._warned: set = set()
        self.last_components: dict = {}

    # ── VAE plumbing (mirrors losses/pfm.py) ────────────────────────────

    def _ensure_vae(self, adapter, device: torch.device) -> torch.nn.Module:
        if self._vae is not None:
            return self._vae
        vae_path = (getattr(adapter, "config", {}) or {}).get("vae_path")
        if not vae_path:
            raise RuntimeError("composite_consistency: adapter.config['vae_path'] missing")
        vae = adapter.load_vae(vae_path, self._vae_dtype)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        if self.keep_vae_on_device:
            vae.to(device)
            if self.vae_grad_checkpoint:
                _PFMRef._enable_vae_ckpt(vae)
        self._vae = vae
        return vae

    def _acquire_vae(self, adapter, device: torch.device):
        """Acquire the VAE and HOLD it until backward has run.

        The decode graph lives past compute(): accelerator.backward() runs AFTER
        this method returns, and pfm-style block checkpointing re-runs the
        decoder during that backward. If the manager offloads the VAE at release
        (training.vae_on_device=false -> "transfer" mode), the recomputation
        finds CPU weights and raises CheckpointError. So in managed mode we
        acquire ONCE and never release for the rest of the run — identical to
        training.vae_on_device=true; set that in the config to be explicit.
        """
        if self.vae_manager is not None:
            if self.vae_manager.load_mode == "reload":
                raise RuntimeError(
                    "composite_consistency: training.vae_load_mode='reload' reloads the VAE "
                    "from disk on EVERY acquire - incompatible with a per-step decode loss. "
                    "Set training.vae_load_mode='lazy' (or 'eager')."
                )
            if self._held_vae is None:
                vae = self.vae_manager.acquire(device)
                # lazy+on_device manager quirk: acquire() returns the RAM instance
                # WITHOUT pinning (load() is a no-op in lazy mode) - pin it here.
                try:
                    dev = getattr(vae, "device", None) or next(vae.parameters()).device
                except (StopIteration, AttributeError):
                    dev = None
                if dev is None or dev != device:
                    vae.to(device)
                self._held_vae = vae
                if self.vae_grad_checkpoint:
                    _PFMRef._enable_vae_ckpt(self._held_vae)
                self._warn_once("held",
                                "composite_consistency: VAE acquired, pinned on GPU and HELD until "
                                "the run ends (the decode graph must survive backward). Same effect "
                                "as training.vae_on_device=true; set it in the config to be explicit.")
            return self._held_vae, lambda: None
        vae = self._ensure_vae(adapter, device)

        def release() -> None:
            return None

        return vae, release

    # ── cached target pixels ────────────────────────────────────────────

    def _gt_cache_path(self, t_png: str) -> str:
        return os.path.splitext(t_png)[0] + "_comp_px.npz"

    def _load_or_build_gt(self, t_png: str, r_png: str, out_hw: tuple, feather_passes: int = 0) -> tuple:
        """Return (R_rgb, GT_rgb, A_soft, A_raw, Fg) float32 CPU tensors at out_hw.

        A_soft = feathered GT alpha (edge targets become achievable soft
        transitions); A_raw = unfeathered alpha (CRISP reference for matte/dice
        — feathering the matte reference taught mid-alpha everywhere and made
        the model translucent).  GT/Fg derive from A_soft:
            GT = A_soft*T_rgb + (1-A_soft)*R      (== _O to ~3/255)
            Fg = A_soft*T_rgb = GT - (1-A_soft)*R (premultiplied foreground)
        """
        cache_path = self._gt_cache_path(t_png)
        th, tw = int(out_hw[0]), int(out_hw[1])
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as data:
                    if (int(data["ver"]) == _PX_CACHE_VERSION
                            and int(data["fp"]) == int(feather_passes)
                            and (int(data["th"]), int(data["tw"])) == (th, tw)):
                        R = torch.from_numpy(data["R"]).float()
                        GT = torch.from_numpy(data["GT"]).float()
                        A = torch.from_numpy(data["A"]).float()
                        Ar = torch.from_numpy(data["Ar"]).float()
                        Fg = torch.from_numpy(data["F"]).float()
                        return R, GT, A, Ar, Fg
            except Exception as e:  # noqa: BLE001
                logger.warning(f"composite_consistency: corrupt px cache {cache_path}: {e} - rebuilding")

        T_rgba = torch.from_numpy(
            np.asarray(Image.open(t_png).convert("RGBA")).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        R_full = torch.from_numpy(
            np.asarray(Image.open(r_png).convert("RGBA")).astype(np.float32) / 255.0
        )[:, :, :3].permute(2, 0, 1)
        a_raw_full, T_rgb = T_rgba[3:4], T_rgba[:3]
        a_soft_full = _feather_alpha(a_raw_full, feather_passes) if feather_passes > 0 else a_raw_full
        GT_full = T_rgb * a_soft_full + R_full * (1 - a_soft_full)
        F_full = GT_full - R_full * (1 - a_soft_full)      # == a_soft_full * T_rgb

        def rs(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(x.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)

        R, GT, A, Ar, Fg = (rs(t).half() for t in (R_full, GT_full, a_soft_full, a_raw_full, F_full))
        tmp = cache_path + ".tmp.npz"
        np.savez(tmp, R=R.numpy(), GT=GT.numpy(), A=A.numpy(), Ar=Ar.numpy(), F=Fg.numpy(),
                 ver=np.int64(_PX_CACHE_VERSION), fp=np.int64(feather_passes),
                 th=np.int64(th), tw=np.int64(tw))
        os.replace(tmp, cache_path)
        return R.float(), GT.float(), A.float(), Ar.float(), Fg.float()

    # ── A. discriminative pixel space (PixPerfect) ──────────────────────

    def _disc_space_loss(self, comp: torch.Tensor, GT: torch.Tensor, A_soft: torch.Tensor,
                         w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """PixPerfect-style discriminative pixel-space L1.

        Fits a per-sample, per-channel degree-D polynomial map f that sends the
        composited prediction towards the AMPLIFIED target
            y_amp = GT + beta*(comp - GT),   beta ~ U[disc_beta]
        (balanced sampling inside/outside the matte, Moore-Penrose lstsq), then
        penalises |f(comp) - f(GT)| in that amplified space, normalized by beta
        so the term's scale stays comparable to the plain pixel L1.  The map is
        fitted under no_grad — the gradient flows through the EVALUATION only
        (f is a fixed per-step projection once fitted).
        """
        b, c, h, wd = comp.shape
        if b != 1:
            return None
        beta = self.disc_beta[0] + (self.disc_beta[1] - self.disc_beta[0]) * float(torch.rand(1))

        pred_c = comp[0].reshape(c, -1)               # (3, N) WITH grad
        gt_c = GT.reshape(3, -1)                      # (3, N) — npz stores no batch dim
        comp_det = comp.detach()[0].reshape(c, -1)                         # (3, N)
        yamp_c = (GT + beta * (comp.detach() - GT)).clamp(0.0, 1.0)[0].reshape(3, -1)  # (3, N)
        flat_m = (A_soft[0, 0] > 0.02).flatten()      # (N,) per-PIXEL matte mask

        idx_fg = flat_m.nonzero(as_tuple=False).flatten()
        idx_bg = (~flat_m).nonzero(as_tuple=False).flatten()
        n = min(self.disc_points, idx_fg.numel(), idx_bg.numel())
        if n < self.disc_degree + 2:
            return None
        perm = torch.randperm(idx_fg.numel(), device=idx_fg.device)[:n]
        idx_fg = idx_fg[perm]
        perm = torch.randperm(idx_bg.numel(), device=idx_bg.device)[:n]
        idx_bg = idx_bg[perm]
        idx = torch.cat([idx_fg, idx_bg])             # (2n,) pixel indices < N

        total = comp.new_zeros(())
        for ch in range(c):
            # lstsq on CPU: cuSOLVER can silently return garbage coefficients for
            # ill-conditioned Vandermonde systems (observed as loss explosions ~
            # step 1000 on the full-set run). n <= disc_points, X is (2n, D+1) —
            # the CPU solve is ~ms and numerically stable.
            x_fit_cpu = comp_det[ch, idx].float().cpu()
            y_fit_cpu = yamp_c[ch, idx].float().cpu()
            X = torch.stack([x_fit_cpu ** d for d in range(1, self.disc_degree + 1)]
                            + [torch.ones_like(x_fit_cpu)], dim=1)
            coef = torch.linalg.lstsq(X, y_fit_cpu.unsqueeze(1)).solution  # (D+1,1)
            coef = coef.clamp(-1e3, 1e3)
            # move the fitted projection onto the pred tensor's device (CPU fit
            # is for numerical stability; evaluation happens on CUDA)
            coef = coef.to(device=pred_c.device, dtype=pred_c.dtype)

            def poly(x: torch.Tensor) -> torch.Tensor:
                shp = x.shape
                xf = x.reshape(-1)
                Xe = torch.stack([xf ** d for d in range(1, self.disc_degree + 1)]
                                 + [torch.ones_like(xf)], dim=1)
                return (Xe @ coef).reshape(shp)

            disc_ch = (poly(pred_c[ch]) - poly(gt_c[ch])).abs()
            if self.metric == "l2":
                disc_ch = disc_ch.pow(2)
            if w is None:
                total = total + disc_ch.mean()
            else:
                # disc_ch is flat (N,) in row-major pixel order — reshape to the
                # 4-D map before weighting (a 1-D×4-D broadcast would silently
                # misalign on non-square buckets).
                total = total + (disc_ch.reshape(1, 1, h, wd) * w).sum() / w.sum().clamp_min(1e-6)
        return total / beta

    # ── B. gradient-domain seam term (MaskFlow SPD, training-time form) ──

    def _grad_loss(self, comp: torch.Tensor, GT: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        def grads(t):
            gx = F.pad(t[..., :, 1:] - t[..., :, :-1], (0, 1))   # (C,h,w)
            gy = F.pad(t[..., 1:, :] - t[..., :-1, :], (0, 0, 0, 1))  # (C,h,w)
            return gx, gy

        cgx, cgy = grads(comp)
        ggx, ggy = grads(GT)
        res = (cgx - ggx).abs() + (cgy - ggy).abs()
        region = F.max_pool2d((A > 0.02).float(), 5, stride=1, padding=2)
        return (res * region).sum() / region.sum().clamp_min(1.0)

    # ── C. alpha coverage penalty (knockout-plate semantics) ─────────────

    def _cover_terms(self, a_ref: torch.Tensor, a_hat: torch.Tensor) -> tuple:
        """Return (cover_viol, cover_hinge) — see the module docstring.

        Shortfall per pixel is how much of the alpha the target demands the
        prediction failed to deliver.  It is ONE-SIDED — a pixel the prediction
        paints beyond the target contributes 0 (over-coverage is left to the FM
        loss / region weighting on purpose):

            cover_form='shortfall' (default):  relu(1 - X_alpha) == relu(a_ref - a_hat)
                X is built literally:  plate B = R - T knocks the target's
                content region out to transparency (B_alpha = 1 - a_ref), then
                X = B + T' adds the prediction back (X_alpha = 1 - a_ref + a_hat).
                Any transparency left in X is a target pixel the prediction
                failed to cover.  A prediction matching a feathered edge
                (a_ref == a_hat) is fully satisfied, so soft edges are
                achievable and the term cannot push the matte outward.
            cover_form='residual':             relu(a_ref * (1 - a_hat))
                == residual transparency under PROPER "over" compositing of
                plate+layer.  Identical on binary regions; at a soft edge it
                needs a_hat -> 1 to reach zero, i.e. it bloats the matte.

        ``cover_viol`` is margin-free and area-normalised
            sum(short) / sum(a_ref)          in [0, 1]
        i.e. the mean missing alpha per unit target alpha: 0.0 == the prediction
        covered everything the target asks for, 1.0 == nothing covered.  It is
        computed and logged on EVERY decoded step, including when
        cover_weight == 0, so a weight=0 run yields the baseline number.

        ``cover_hinge`` is the actual penalty (only used when cover_weight > 0):
            sum(relu(short - margin)) / sum(a_ref)     cover_norm='area'
            mean(relu(short - margin))                 cover_norm='mean'
        The margin makes this a constraint rather than a regression pull —
        already-covered pixels contribute exactly 0, so the term cannot drag the
        matte shape around.  'area' keeps small targets from being diluted by
        the frame (MaskFlow MFL spirit); the gradient is identical to 'mean' up
        to a constant factor of 1/mean(a_ref).
        """
        if self.cover_form == "shortfall":
            # 底板 B = 原图 − 训练目标  → 编辑区抠空为透明:  B_alpha = 1 − α_T
            plate_alpha = 1.0 - a_ref                          # (B,1,H,W)
            # X = 底板 + 模型预测:                           X_alpha = 1 − α_T + α̂
            x_alpha = plate_alpha + a_hat
            # X 里还残留的透明像素 = 预测没覆盖到的目标像素 → 惩罚
            # relu(1 − X_alpha) == relu(α_T − α̂) —— 与化简形式逐位相等，
            # 这里保留字面构造是为了可读性，也留着 X 这个完整图像的钩子
            #（以后若要给 X 的 RGB 侧加约束或存调试图，直接续用 x_alpha/x_rgb）。
            short = torch.relu(1.0 - x_alpha)
        else:
            short = torch.relu(a_ref * (1.0 - a_hat))
        denom = a_ref.sum().clamp_min(1e-6)
        cover_viol = (short.sum() / denom).detach()
        violation = torch.relu(short - self.cover_margin)
        if self.cover_norm == "area":
            cover_hinge = violation.sum() / denom
        else:
            cover_hinge = violation.mean()
        return cover_viol, cover_hinge

    # ── loss ────────────────────────────────────────────────────────────

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def compute(self, context: LossContext) -> torch.Tensor:
        x0_hat = context.x0_hat
        zero = torch.tensor(0.0, device=x0_hat.device) if x0_hat is not None else torch.tensor(0.0)
        if x0_hat is None:
            self._warn_once("no_x0_hat", "composite_consistency: x0_hat missing - skipping")
            return zero

        self._call_count += 1
        if (self._call_count - 1) % self.every_n_steps != 0:
            return zero

        # sigma gate: high-noise x0_hat is inherently blurry — skip (anti-blur)
        sig = context.sigmas
        s_val = float(sig.detach().flatten()[0]) if sig is not None and sig.numel() > 0 else None
        if s_val is not None and s_val > self.sigma_max:
            self.last_components = {"sigma": s_val, "gated": 1.0}
            return zero

        pp = (context.extra or {}).get("pixel_paths")
        if not pp or not pp.get("T") or not pp.get("R"):
            self._warn_once("no_paths",
                            "composite_consistency: extra['pixel_paths'] missing (old trainer/dataset?) - skipping")
            return zero
        if x0_hat.shape[0] != 1:
            self._warn_once("bs", f"composite_consistency: batch={x0_hat.shape[0]} - using sample 0 (built for bs=1)")

        vae, release = self._acquire_vae(context.adapter, x0_hat.device)
        try:
            # NO torch.no_grad(): the gradient must flow decode -> x0_hat -> LoKR.
            # Full-resolution decode — the half-res downscale variant was REMOVED
            # (it was the global-blur root cause; see module docstring).
            latent = x0_hat[:1]
            pred = context.adapter.decode_latent_differentiable(vae, latent)
            if pred.shape[1] != 4:
                self._warn_once("chan",
                                f"composite_consistency: decode returned {pred.shape[1]} channels (need RGBA) - skipping")
                return zero

            rgba = ((pred.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            pred_rgb, a_hat = rgba[:, :3], rgba[:, 3:4]
            R, GT, A_soft, A_raw, Fg = self._load_or_build_gt(
                pp["T"], pp["R"], pred_rgb.shape[-2:], feather_passes=self._feather_passes)
            R, GT, A_soft, A_raw, Fg = (t.to(pred_rgb) for t in (R, GT, A_soft, A_raw, Fg))

            comp = a_hat * pred_rgb + (1.0 - a_hat) * R
            residual = comp - GT

            # weight map: union of GT matte and predicted matte, transition band
            # (feathered-alpha ring) down-weighted — pixel-exact edge supervision
            # produced hard alpha cliffs (seams).
            if self.weight_mode == "union":
                w = torch.maximum(A_raw, a_hat)
            elif self.weight_mode == "gt":
                w = A_raw
            else:
                w = None
            if w is not None and self.edge_band_drop > 0:
                band = ((A_soft > 0.02) & (A_soft < 0.98)).float()
                band = F.max_pool2d(band, 3, stride=1, padding=1)
                w = w * (1.0 - self.edge_band_drop * band)

            def wmean(t: torch.Tensor) -> torch.Tensor:
                if w is None:
                    return t.mean()
                return (t * w).sum() / w.sum().clamp_min(1e-6)

            data = wmean(residual.abs() if self.metric == "l1" else residual.pow(2))
            total = self.data_weight * data

            # matte: vs RAW alpha (crisp shape reference — v4 fix for the
            # translucent tendency the feathered reference taught)
            matte = wmean((a_hat - A_raw).abs())
            total = total + self.matte_weight * matte

            # soft-Dice on the matte (shape overlap) — also vs the RAW alpha
            if self.dice_weight > 0:
                inter = (a_hat * A_raw).sum()
                dsc = 1.0 - 2.0 * inter / (a_hat.sum() + A_raw.sum()).clamp_min(1e-6)
                total = total + self.dice_weight * dsc
            else:
                dsc = None

            # premultiplied foreground: suppresses alpha leakage AND the raw RGB
            # garbage ("purple/blue noise") in the transparent region
            if self.fg_weight > 0:
                fg_residual = a_hat * pred_rgb - Fg
                fg = fg_residual.abs().mean() if self.metric == "l1" else fg_residual.pow(2).mean()
                total = total + self.fg_weight * fg
            else:
                fg = None

            # C. alpha coverage penalty (knockout-plate semantics).  NO feathering
            # by default: a_ref is the dataset's own alpha (A_raw), which keeps
            # its 1-2px edge — the shortfall form is satisfiable on that ramp, so
            # blurring the reference would only re-teach mid-alpha.
            a_ref_cov = A_soft if self.cover_alpha == "soft" else A_raw
            cover_viol, cover = self._cover_terms(a_ref_cov, a_hat)
            if self.cover_weight > 0:
                total = total + self.cover_weight * cover

            # B. gradient-domain seam term
            if self.grad_weight > 0:
                g = self._grad_loss(comp, GT, A_soft)
                total = total + self.grad_weight * g
            else:
                g = None

            # A. discriminative pixel space (color-shift fix)
            disc = None
            if self.discriminative_space:
                disc = self._disc_space_loss(comp, GT, A_soft, w)
                if disc is not None:
                    # Pathological-fit guard: an ill-conditioned per-sample fit can
                    # produce a huge value that spikes the loss (observed as a loss
                    # explosion ~step 1000 on the full-set run). Cap and skip.
                    if not torch.isfinite(disc) or float(disc) > self.disc_cap:
                        self._warn_once("disc_skip",
                                        f"composite_consistency: pathological disc value "
                                        f"{float(disc) if torch.isfinite(disc) else float('nan'):.3f} - skipping term this step")
                        disc = None
                    else:
                        total = total + disc

            # non-finite guard: never poison the optimizer with NaN/Inf
            if not torch.isfinite(total):
                self._warn_once("nonfinite", "composite_consistency: non-finite total - skipping")
                return zero

            comp_mae = float(residual.detach().abs().mean()) * 127.5
            self.last_components = {
                "mae_255": comp_mae,
                "data": float(data.detach()),
                "matte_mae": float((a_hat - A_raw).detach().abs().mean()),
                "alpha_mean": float(a_hat.detach().mean()),
                "iou_50": float((((a_hat > 0.5) & (A_raw > 0.5)).sum()
                                 / ((a_hat > 0.5) | (A_raw > 0.5)).sum().clamp_min(1)).detach()),
                # coverage: margin-free baseline metric + (when enabled) the penalty
                "cover_viol": float(cover_viol),
                "cover": (float(cover.detach()) if self.cover_weight > 0 else 0.0),
                "cover_ref_area": float(a_ref_cov.detach().mean()),
                "fg_mae_255": (float(fg_residual.detach().abs().mean()) * 127.5) if fg is not None else 0.0,
                "dice": (float(dsc.detach()) if dsc is not None else 0.0),
                "grad": (float(g.detach()) if g is not None else 0.0),
                "disc": (float(disc.detach()) if disc is not None else 0.0),
                "sigma": s_val,
                "gated": 0.0,
            }
            return total
        finally:
            release()

    def requires(self) -> List[str]:
        return ["model_pred", "x0_hat", "sigmas"]
