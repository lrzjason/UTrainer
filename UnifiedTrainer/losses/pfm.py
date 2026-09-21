"""
Perceptual Flow Matching (PFM) loss — arXiv 2607.03524.

Replaces (or complements) the flow-matching velocity MSE with a feature-space
consistency loss between the one-step clean estimate x0_hat and the ground
truth, computed in a frozen self-supervised encoder (DINO-style perceptual
loss):

    x0_hat = xt − σt·v̂,  xt = (1−σt)x0 + σt·ε          (paper Eq.1/4)
    L_pfm = mean_{l in layers} d( phi_l(decode(x0_hat)),
                                  phi_l(decode(x0)) )   (paper Eq.6)

where phi_l are the patch-token features of encoder block l (RoPE ViT, so any
input resolution works). d defaults to RAW L2 (Euclidean) on the final-norm
features — the paper's ablation supervises multiple deeper layers averaged
(single shallow layers blur, the deepest layer collapses); token_norm=True is
a legacy stability escape hatch (per-token L2 + MSE) that erases the feature-
magnitude signal behind PFM's mode-seeking behavior.

Gradient routing (the crux): the VAE is FROZEN but the x0_hat decode is NOT
wrapped in no_grad — autograd therefore propagates the loss through the frozen
decoder activations back into x0_hat -> model_pred -> LoRA/LoKR weights.
requires_grad=False on VAE weights only means the VAE itself never receives
weight gradients (it never trains); it does NOT block input gradients. The
ground-truth branch runs under no_grad (frozen target, no graph).

Per-step pipeline:
    x0_hat (latent, grad) -> adapter.decode_latent_differentiable -> pixels [-1,1]
    x0     (latent)       -> same decode under no_grad
    -> [0,1] -> resize(loss_resolution) -> ImageNet normalize
    -> frozen encoder -> get_intermediate_layers(layers) -> per-token loss.

VAE residency: the VAE is managed centrally by ``engine/vae_manager.py`` —
when train.py wires the manager (it does so whenever any component declares
``requires_vae``), this loss borrows the shared instance via
``self.vae_manager``. With pfm configured, ``training.vae_load_mode`` defaults
to "lazy" (one disk load on first use, then RAM-resident — never a per-step
reload); residency on GPU is then governed by ``training.vae_on_device``
(false = transfer on demand and back to RAM, true = pinned for the whole run;
default false). When no manager is attached (standalone use, tests), the loss
falls back to loading its own frozen copy from adapter.config["vae_path"].
Cost warning: every active step still pays TWO VAE decodes (one with gradient)
plus TWO encoder passes; use apply_prob to thin the loss out while exploring.

Encoder: EUPE (Meta, arXiv 2603.22387) via torch.hub local checkout — clone
https://github.com/facebookresearch/EUPE and point hub_repo_dir at it;
weights_path points at the .pt from the facebook/EUPE-ViT-{T,S,B} HF repos.
Any DINOv2/v3-style torch.hub backbone exposing get_intermediate_layers works
the same way. DINOv2 itself (facebook/dinov2-*, HF transformers format) is a
first-class option: download a snapshot with E:\\hf_models\\dl_dinov2.py and
point model_path at it (encoder names starting with "dinov2"). DINOv2 is a
patch-14 ViT — loss_resolution/input_size must be a multiple of 14 (224/252/
280/448...). RADIO (NVlabs, encoder names starting with "radio") loads from a
local HF dir with its shipped .py sources via model_path.

DINOv3 ConvNeXt (facebook/dinov3-convnext-*-pretrain-lvd1689m, encoder names
starting with "dinov3", e.g. "dinov3_convnext_small") is also first-class:
model_path points at the local HF snapshot and "layers" are STAGE ids — per-
stage channels AND stride differ (stage i: hidden_sizes[i] channels at H/
(4·2^i)), so returned layers are NOT shape-uniform (the per-layer pred/target
comparison doesn't care — pred and target go through the SAME encoder). It is
fully convolutional with NO positional embedding: any resolution divisible by
32 works, well beyond the 224 training size. Constraint: per-stage shapes are
incompatible with the uniform (L, C, h, w) on-disk pfm cache contract, so
dinov3* rejects target_source='original' and precache=true — use the decoded
path (defaults). Resolution: a positive loss_resolution square-resizes both
branches; loss_resolution=0 ("auto", dinov3-only) compares 1:1 at the DECODED
pixel size — no resampling, so the comparison resolution follows whatever the
VAE outputs (512 locally, 768/1024 on a bigger machine) with the same config.
Above the dataset resolution there is no new target information (both sides
are upsampled), but ConvNeXt's receptive fields cover a smaller RELATIVE area
at higher input res, so the same stages encode more local/texture structure
and the loss punishes fine-detail mismatch a 224/256 comparison cannot see.
That is a supervision-geometry shift, not new target information; ViT
encoders (dinov2) cannot follow (pos-emb interpolation + quadratic
attention).

High-frequency residual anchor (HP-L1, optional): perceptual features are
near-invariant to high-frequency PHASE — a zero loss in encoder space still
permits blurry textures, which is why a converged PFM fit can lack fine
detail. The anchor supervises exactly that band: L_freq = |HP(pred_px) −
HP(tgt_px)|₁ on the DECODED pixels, HP(x) = x − gauss_blur(x) (fixed kernel,
fp32), gated to LOW-noise steps (σ ≤ freq_sigma_max — above it x̂0 is
noise-dominated and its HP band is meaningless). Params: freq_weight (0 =
off), freq_sigma_max (0.3), freq_kernel_sigma (1.5), freq_kernel_size (7).
The target-side HP is cached per sample (fp16 CPU, full decode resolution)
next to the feature comparables; requires the on-the-fly decoded path
(original mode / precache carry no target pixels to take a high-pass of).
Pairing: the encoder term says "looks alike", the HP term says "textures
align" — complementary, since the encoder's blind spot is exactly what HP
sees.

Low-σ FM blend (fm_mix_weight, optional): on the low-noise subset the
feature term alone under-constrains x̂0 even though it is closest to clean
there. This blends the EXACT velocity MSE into that regime — main·(1−w·frac)
+ w·frac·FM — so x̂0 is pinned to the data manifold while features carry
semantics and HP carries texture. frac = share of active samples in the
band (exact for batch_size=1, expectation-preserving on mixed batches).
Params: fm_mix_weight (0 = off), fm_sigma_max (0.3).

Config example — pfm REPLACING the MSE (paper-faithful PFM run):
    "losses": [
        {"type": "pfm", "weight": 1.0, "params": {
            "encoder": "eupe_vits16",
            "hub_repo_dir": "/home/waas/EUPE",
            "weights_path": "/home/waas/models/EUPE-ViT-S/EUPE-ViT-S.pt",
            "layers": [6]
        }}
    ]

Config example — DINOv2 encoder (local HF snapshot from dl_dinov2.py):
    "losses": [
        {"type": "pfm", "weight": 1.0, "params": {
            "encoder": "dinov2_base",
            "model_path": "E:/hf_models/dinov2-base",
            "layers": [5, 7, 9, 11],
            "loss_resolution": 224
        }}
    ]

xm baseline keeps the pure flow_matching MSE — do NOT add "pfm" to those
configs; duplicate a config and swap the losses array to start a PFM run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.registry import LossRegistry

logger = logging.getLogger(__name__)

# EUPE image transform (official README): resize -> [0,1] -> ImageNet normalize.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@LossRegistry.register("pfm")
class PerceptualFlowMatchingLoss(BaseLoss):
    """Frozen-encoder perceptual loss on the one-step clean estimate (PFM)."""

    name = "pfm"
    # 声明依赖 VAE：train.py 据此在训练循环前加载共享 VAE（vae_manager），
    # trainer.__init__ 会把 manager 注入 self.vae_manager。
    requires_vae = True

    def __init__(
        self,
        weight: float = 1.0,
        encoder: str = "eupe_vits16",
        hub_repo_dir: Optional[str] = None,
        weights_path: Optional[str] = None,
        model_path: Optional[str] = None,
        layers: Sequence[int] = (5, 7, 9, 11),
        metric: str = "l1",
        token_norm: bool = False,
        encoders: Optional[List[dict]] = None,
        fallback_fm_mse: bool = False,
        fallback_weight: float = 1.0,
        target_source: str = "decoded",
        loss_resolution: int = 256,
        apply_prob: float = 1.0,
        precache: bool = False,
        decode_scale: float = 1.0,
        vae_tile_size: int = 0,
        tone_weight: float = 0.0,
        freq_weight: float = 0.0,
        freq_sigma_max: float = 0.3,
        freq_kernel_sigma: float = 1.5,
        freq_kernel_size: int = 7,
        fm_mix_weight: float = 0.0,
        fm_sigma_max: float = 0.3,
        keep_vae_on_device: bool = True,
        vae_grad_checkpoint: bool = True,
        encoder_dtype: str = "bf16",
        vae_dtype: str = "bf16",
        **params,
    ):
        super().__init__(weight=weight, **params)
        if metric not in ("l1", "l2", "mse", "cosine"):
            raise ValueError(
                f"pfm: unknown metric {metric!r}; expected "
                "'l1', 'l2', 'mse' or 'cosine'"
            )
        if encoder.startswith("eupe_convnext"):
            raise ValueError(
                "pfm: EUPE torch.hub ConvNeXt encoders are not supported "
                "(per-stage feature extraction differs); use "
                "eupe_vitt16/eupe_vits16/eupe_vitb16. For a ConvNeXt "
                "backbone use a DINOv3 ConvNeXt encoder instead "
                "(encoder names starting with 'dinov3_convnext')."
            )
        if not 0.0 < apply_prob <= 1.0:
            raise ValueError(f"pfm: apply_prob must be in (0, 1], got {apply_prob}")
        for name_, value in (("encoder_dtype", encoder_dtype), ("vae_dtype", vae_dtype)):
            if value not in _DTYPE_MAP:
                raise ValueError(
                    f"pfm: {name_} must be one of {sorted(_DTYPE_MAP)}, got {value!r}"
                )

        self.encoder_name = encoder
        self.hub_repo_dir = hub_repo_dir
        self.weights_path = weights_path
        self.model_path = model_path
        self.is_radio = encoder.startswith("radio")
        self.layers = [int(l) for l in layers]
        self.metric = metric
        self.token_norm = bool(token_norm)
        # Official-recipe multi-encoder stack (shared ONE VAE decode pair).
        # Each entry: {net, weight, layers, metric, token_norm, input_size,
        # model_path|hub_repo_dir|weights_path}. net=="vgg_lpips" is the
        # LPIPS-VGG companion (https://github.com/ZhaoChuyang/PFM). When
        # absent, a single entry is synthesized from the legacy flat params
        # so existing configs keep working unchanged.
        if encoders:
            self._encoder_cfgs: List[dict] = []
            for e in encoders:
                cfg_e = dict(e)
                cfg_e["weight"] = float(cfg_e.get("weight", 1.0))
                cfg_e["layers"] = [int(l) for l in cfg_e.get("layers", [])]
                cfg_e["metric"] = str(cfg_e.get("metric", "cosine"))
                if cfg_e["metric"] not in ("l1", "l2", "mse", "cosine"):
                    raise ValueError(
                        f"pfm: unknown entry metric {cfg_e['metric']!r}"
                    )
                cfg_e["token_norm"] = bool(cfg_e.get("token_norm", False))
                cfg_e["input_size"] = int(cfg_e.get("input_size", loss_resolution))
                self._encoder_cfgs.append(cfg_e)
        else:
            self._encoder_cfgs = [{
                "net": encoder,
                "weight": 1.0,
                "layers": self.layers,
                "metric": metric,
                "token_norm": bool(token_norm),
                "input_size": int(loss_resolution),
                "model_path": model_path,
                "hub_repo_dir": hub_repo_dir,
                "weights_path": weights_path,
            }]
        self.fallback_fm_mse = bool(fallback_fm_mse)
        self.fallback_weight = float(fallback_weight)
        if target_source not in ("decoded", "original"):
            raise ValueError(
                f"pfm: target_source must be 'decoded' or 'original', "
                f"got {target_source!r}"
            )
        if target_source == "original" and len(self._encoder_cfgs) != 1:
            raise ValueError(
                "pfm: target_source='original' supports exactly one feature "
                "encoder (the precomputed φ(I_orig) cache contract); use "
                "target_source='decoded' with the multi-encoder stack."
            )
        self.target_source = target_source
        # Target-branch feature cache: decode(x0) -> encode is deterministic
        # per dataset sample (latents come from the immutable cache, the
        # encoder is frozen, loss_resolution fixed) — compute each sample's
        # features once, reuse for every later epoch. Keyed by x0 content.
        self._feat_cache: dict = {}
        self._feat_cache_cap = 4096
        # loss_resolution > 0: fixed square comparison size (resampled).
        # loss_resolution == 0 ("auto", dinov3-only): compare 1:1 at the
        # DECODED pixel size — no resampling anywhere, so whatever the VAE
        # decodes to (512 local, 768/1024 on a bigger box) is what the
        # ConvNeXt encoder sees. Same config, resolution follows the machine.
        self.loss_resolution = int(loss_resolution)
        if self.loss_resolution < 0:
            raise ValueError(
                f"pfm: loss_resolution must be >= 0 (0 = 'auto': 1:1 with "
                f"the decoded pixels), got {loss_resolution!r}"
            )
        if self.loss_resolution == 0:
            _fixed_grid = [
                str(c["net"])
                for c in self._encoder_cfgs
                if not str(c["net"]).startswith("dinov3")
            ]
            if _fixed_grid:
                raise ValueError(
                    f"pfm: loss_resolution=0 (auto 1:1) is dinov3*-only — "
                    f"entries {_fixed_grid} are fixed-grid encoders that "
                    "need a numeric loss_resolution"
                )
        self.apply_prob = float(apply_prob)
        # precache=true: train.py runs precache_targets() before the training
        # loop, persisting every sample's φ(D(x0)) target features to disk.
        # The loss itself consumes them whenever the trainer carries them —
        # this flag only gates the pre-training pass.
        self.precache = bool(precache)
        # DINOv3 ConvNeXt emits per-STAGE features (channels AND stride differ
        # per stage) — incompatible with the uniform (L, C, h, w) on-disk pfm
        # cache contract used by target_source='original' and precache=true.
        if self.target_source == "original" or self.precache:
            _dinov3_entries = [
                str(c["net"])
                for c in self._encoder_cfgs
                if str(c["net"]).startswith("dinov3")
            ]
            if _dinov3_entries:
                raise ValueError(
                    f"pfm: dinov3* ConvNeXt encoders {_dinov3_entries} cannot "
                    "back target_source='original'/precache=true — per-stage "
                    "channels/strides differ, while the pfm cache contract is "
                    "uniform (L, C, h, w). Use target_source='decoded' with "
                    "precache=false (the defaults), or a dinov2/eupe/radio "
                    "encoder when cached targets are needed."
                )
        # decode_scale<1.0 downsamples the LATENT before the differentiable
        # VAE decode (both pred and target sides, and the precache): decode
        # activations scale with pixels², so 0.5 cuts the decode VRAM ~4x
        # and decode time ~4x. The encoders resize to their input sizes
        # (512/224) anyway, so the perceptual features are near-identical.
        self.decode_scale = float(decode_scale)
        if not 0.25 <= self.decode_scale <= 1.0:
            raise ValueError(
                f"pfm: decode_scale must be in [0.25, 1.0], got {decode_scale!r}"
            )
        # vae_tile_size>0: decode in overlapping spatial tiles (pixel edge of
        # one tile, e.g. 512) under per-tile gradient checkpointing — peak
        # decode memory stays at ONE tile regardless of image size, WITHOUT
        # the detail loss that decode_scale's latent downsample causes. The
        # tile split mirrors diffusers' AutoencoderKLQwenImage tiling; the
        # checkpoint+feather execution is ours (see Krea2Adapter.
        # _tiled_decode_differentiable). Pred, target and precache branches
        # all pass the SAME value → identical seams everywhere.
        self.vae_tile_size = int(vae_tile_size or 0)
        if self.vae_tile_size not in (0,) and self.vae_tile_size < 256:
            raise ValueError(
                f"pfm: vae_tile_size must be 0 (off) or >= 256 px, got {vae_tile_size!r}"
            )
        # Tone anchor (2026-09-01-pfm-tone-anchor-design.md): exposure &
        # contrast are ZERO directions of perceptual features — nothing in
        # PFM pins them, and the drift (highlight clipping + crushed
        # shadows, "强曝光") grows from high-σ steps whose x̂0 is mostly
        # noise. L_tone = (m_pred−m_tgt)² + (s_pred−s_tgt)² on Rec.601 luma
        # of the DECODED pixels, same idx mask as the main term. 0 = off,
        # entirely outside the graph (zero overhead).
        self.tone_weight = float(tone_weight)
        # High-frequency residual anchor (HP-L1): perceptual features are
        # near-invariant to high-frequency PHASE — a zero loss in dinov3
        # space still permits blurry textures. L_freq = |HP(pred_px) −
        # HP(tgt_px)|₁ on the DECODED pixels, where HP(x) = x − gauss(x)
        # (fixed kernel, computed in fp32 even from bf16 input — the base
        # signal cancels in the subtraction, so narrow precision would
        # quantize the residual). Gated to LOW-noise steps (σ <=
        # freq_sigma_max): at high σ x̂0 is noise-dominated and its HP band
        # is meaningless. The target-side HP is deterministic per sample →
        # cached (fp16, CPU) alongside the feature comparables. 0 = off,
        # zero overhead anywhere.
        self.freq_weight = float(freq_weight)
        self.freq_sigma_max = float(freq_sigma_max)
        self.freq_kernel_sigma = float(freq_kernel_sigma)
        self.freq_kernel_size = int(freq_kernel_size)
        if self.freq_weight < 0.0:
            raise ValueError(
                f"pfm: freq_weight must be >= 0, got {self.freq_weight!r}"
            )
        if self.freq_weight > 0.0:
            if not 0.0 < self.freq_sigma_max <= 1.0:
                raise ValueError(
                    f"pfm: freq_sigma_max must be in (0, 1], got "
                    f"{self.freq_sigma_max!r}"
                )
            if self.freq_kernel_sigma <= 0.0:
                raise ValueError(
                    f"pfm: freq_kernel_sigma must be > 0, got "
                    f"{self.freq_kernel_sigma!r}"
                )
            if not 3 <= self.freq_kernel_size <= 31 or self.freq_kernel_size % 2 == 0:
                raise ValueError(
                    f"pfm: freq_kernel_size must be an odd int in [3, 31], "
                    f"got {self.freq_kernel_size!r}"
                )
            if self.target_source != "decoded" or self.precache:
                raise ValueError(
                    "pfm: freq_weight>0 requires the on-the-fly decoded "
                    "path (target_source='decoded', precache=false) — the "
                    "original-mode/precache contracts carry no target "
                    "pixels to take a high-pass of"
                )
        self._freq_kernel: Optional[torch.Tensor] = None
        # Low-σ FM blend: on the low-noise subset (σ ≤ fm_sigma_max) the
        # exact velocity MSE is blended IN and the feature term blended OUT
        # by the same weight — main·(1−w·frac) + w·frac·FM. Rationale: PFM's
        # perceptual term under-constrains the low-σ regime even though x̂0
        # is closest to clean there; the FM anchor pins x̂0 to the data
        # manifold while features carry semantics and HP carries texture.
        # frac = fraction of ACTIVE samples in the band → exact for
        # batch_size=1 / uniformly-gated batches, expectation-preserving
        # fractional scaling on mixed batches. 0 = off.
        self.fm_mix_weight = float(fm_mix_weight)
        self.fm_sigma_max = float(fm_sigma_max)
        if not 0.0 <= self.fm_mix_weight <= 1.0:
            raise ValueError(
                f"pfm: fm_mix_weight must be in [0, 1], got {self.fm_mix_weight!r}"
            )
        if self.fm_mix_weight > 0.0 and not 0.0 < self.fm_sigma_max <= 1.0:
            raise ValueError(
                f"pfm: fm_sigma_max must be in (0, 1], got {self.fm_sigma_max!r}"
            )
        # Component telemetry (consumed by the trainer's loss_breakdown merge
        # -> wandb keys loss/pfm/main|tone|freq). Values are AS-APPLIED
        # (weighted), so the three numbers sum exactly to the module loss.
        self.last_components: dict = {}
        self.keep_vae_on_device = bool(keep_vae_on_device)
        self.vae_grad_checkpoint = bool(vae_grad_checkpoint)
        self._encoder_dtype = _DTYPE_MAP[encoder_dtype]
        self._vae_dtype = _DTYPE_MAP[vae_dtype]

        self._encoders = None  # built lazily — heavy checkpoint loads
        self._vae = None      # fallback self-owned VAE (no manager attached)
        self.vae_manager = None  # injected by Trainer.__init__ when wired
        self._device = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def _ensure_encoders(self) -> List[dict]:
        """Load every configured encoder once; returns [{cfg, model}, ...]."""
        if self._encoders is not None:
            return self._encoders
        loaded: List[dict] = []
        for cfg_e in self._encoder_cfgs:
            net = str(cfg_e["net"])
            layers_e = cfg_e["layers"]
            if net == "vgg_lpips":
                from UnifiedTrainer.utils.lpips_vgg import load_lpips_vgg

                # Config-driven weights path: pass through so a custom local
                # VGG backbone (vgg16-397923af.pth) is used without download.
                model = load_lpips_vgg(cfg_e.get("weights_path") or cfg_e.get("model_path"))
            elif net.startswith("radio"):
                from UnifiedTrainer.utils.perceptual_encoder import (
                    load_radio_encoder,
                )

                model = load_radio_encoder(cfg_e.get("model_path"), layers_e)
            elif net.startswith("dinov2"):
                from UnifiedTrainer.utils.perceptual_encoder import (
                    load_dinov2_encoder,
                )

                model = load_dinov2_encoder(cfg_e.get("model_path"), layers_e)
            elif net.startswith("dinov3"):
                from UnifiedTrainer.utils.perceptual_encoder import (
                    load_dinov3_convnext_encoder,
                )

                model = load_dinov3_convnext_encoder(
                    cfg_e.get("model_path"), layers_e
                )
            else:
                hub_repo_dir = cfg_e.get("hub_repo_dir")
                weights_path = cfg_e.get("weights_path")
                if not hub_repo_dir or not weights_path:
                    raise ValueError(
                        f"pfm: entry {net!r} needs 'hub_repo_dir' (local "
                        "checkout of facebookresearch/EUPE) and "
                        "'weights_path' (the EUPE-*.pt file)"
                    )
                model = torch.hub.load(
                    hub_repo_dir,
                    net,
                    source="local",
                    pretrained=True,
                    weights=weights_path,
                )
                depth = getattr(model, "n_blocks", None)
                if depth is not None:
                    bad = [l for l in layers_e if l < 0 or l >= depth]
                    if bad:
                        raise ValueError(
                            f"pfm: layers {bad} out of range for {net} "
                            f"(depth={depth}, 0-indexed block ids)"
                        )
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            loaded.append({"cfg": cfg_e, "model": model})
        if self._device is not None:
            for item in loaded:
                item["model"].to(device=self._device, dtype=self._encoder_dtype)
        self._encoders = loaded
        return loaded

    def _ensure_vae(self, adapter, device: torch.device) -> torch.nn.Module:
        """Fallback loader (no manager attached): self-owned frozen VAE.

        Loaded from adapter.config["vae_path"]; every parameter frozen (the
        VAE never trains — requires_grad=False only skips WEIGHT grads, input
        gradients still flow through the decode, which is the whole point of
        PFM). Pinned on `device` unless keep_vae_on_device=False.
        """
        if self._vae is not None:
            return self._vae
        config = getattr(adapter, "config", None) or {}
        vae_path = config.get("vae_path")
        if not vae_path:
            raise RuntimeError(
                "pfm: adapter.config['vae_path'] is missing — the loss needs a "
                "VAE to decode x0_hat/x0 into pixel space."
            )
        vae = adapter.load_vae(vae_path, self._vae_dtype)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        if self.keep_vae_on_device:
            vae.to(device)
            self._enable_vae_ckpt(vae)
        self._vae = vae
        return vae

    @staticmethod
    def _enable_vae_ckpt(vae: torch.nn.Module) -> None:
        """Opt-in VAE activation checkpointing.

        Diffusers ModelMixins that declare ``_supports_gradient_checkpointing``
        use their native ``enable_gradient_checkpointing()``. VAEs without
        that API (e.g. AutoencoderKLQwenImage — the Krea2 VAE) fall back to
        wrapping the decoder forward in ``torch.utils.checkpoint``: the
        differentiable pfm decode is the dominant VRAM peak, and without a
        checkpointing mechanism every decoder activation is retained for
        backward (this is what pushed 1024px pfm training past 47GB).
        """
        if vae is not None and getattr(
            vae, "_supports_gradient_checkpointing", False
        ):
            vae.enable_gradient_checkpointing()
            return
        if vae is not None and getattr(vae, "decoder", None) is not None:
            PerceptualFlowMatchingLoss._wrap_decoder_checkpoint(vae.decoder)

    @staticmethod
    def _wrap_decoder_checkpoint(decoder: torch.nn.Module) -> None:
        """Checkpoint the VAE decoder at BLOCK granularity (once).

        The QwenImage decoder's ``feat_cache``/``feat_idx`` are shared
        MUTABLE state (``feat_idx[0] += 1`` per conv, list-indexed reads) —
        replaying them in a recompute would index past the end. For
        single-frame inputs (pfm is image-only; T=1) the cache is a no-op
        (every entry None -> plain convs), so it is dropped: identical
        output and the counters stay inert. Multi-frame inputs pass through
        untouched.

        Each heavy block (conv_in / mid_block / up_blocks / conv_out) is
        wrapped individually in torch.utils.checkpoint, so the backward
        recompute materializes ONE block's activations (~a few GB at
        1024px) instead of the whole decode (~30GB — the difference
        between fitting and OOM on 48GB). Trade-off: ~2x decoder compute.
        """
        if getattr(decoder, "_ut_checkpoint_wrapped", False):
            return
        from torch.utils.checkpoint import checkpoint as _ckpt

        _orig_decoder = decoder.forward

        def _decoder_forward(x, **kwargs):
            if x.dim() == 5 and x.shape[2] == 1:
                kwargs.pop("feat_cache", None)
                kwargs.pop("feat_idx", None)
            return _orig_decoder(x, **kwargs)

        decoder.forward = _decoder_forward  # type: ignore[method-assign]

        def _wrap_block(m: torch.nn.Module) -> None:
            if getattr(m, "_ut_ckpt_wrapped", False):
                return
            _orig = m.forward

            def _fwd(*args, **kwargs):
                _x = args[0] if args else kwargs.get("x")
                if isinstance(_x, torch.Tensor) and _x.dim() == 5 and _x.shape[2] == 1:
                    return _ckpt(_orig, *args, use_reentrant=False, **kwargs)
                return _orig(*args, **kwargs)

            m.forward = _fwd  # type: ignore[method-assign]
            m._ut_ckpt_wrapped = True

        for _name in ("conv_in", "mid_block", "conv_out"):
            _m = getattr(decoder, _name, None)
            if _m is not None:
                _wrap_block(_m)
        for _ub in getattr(decoder, "up_blocks", []) or []:
            _wrap_block(_ub)

        decoder._ut_checkpoint_wrapped = True
        logger.info(
            "pfm: VAE decoder block-checkpointed (single-frame, "
            "feat_cache bypassed)"
        )

    def _acquire_vae(self, adapter, device: torch.device):
        """Acquire the VAE on `device`. Returns ``(vae, release)``.

        ``release`` MUST be invoked exactly once. When the differentiable
        decode graph is kept alive (pred side carries gradients), pass it
        through :meth:`_release_after_backward` instead of calling it
        directly: autograd saves the VAE weights BY REFERENCE, so moving the
        module back to RAM before backward reproduces
        ``"weight is on cpu"`` errors inside convolution backward.

        Manager wired (统一管理): borrow the single shared instance owned by
        engine/vae_manager.VAEManager — residency is governed by
        training.vae_on_device through the manager. No manager (standalone
        use, tests): fall back to a self-owned frozen copy.
        """
        if self.vae_manager is not None:
            vae = self.vae_manager.acquire(device)
            if self.vae_grad_checkpoint:
                self._enable_vae_ckpt(vae)

            def release() -> None:
                self.vae_manager.release()

            return vae, release
        # Legacy fallback: self-owned frozen copy.
        vae = self._ensure_vae(adapter, device)
        if self.keep_vae_on_device:

            def release() -> None:
                pass

            return vae, release
        try:
            orig_device = vae.device  # diffusers ModelMixin property
        except AttributeError:
            orig_device = next(vae.parameters()).device
        vae.to(device)

        def release() -> None:
            vae.to(orig_device)

        return vae, release

    @staticmethod
    def _release_after_backward(loss: torch.Tensor, release) -> None:
        """Defer ``release`` until the current backward pass has finished.

        The decode ops live in the autograd graph; the VAE must not move
        until every gradient w.r.t. decoder weights has been computed.

        Implementation: ``queue_callback`` only accepts installation DURING
        a backward pass ("Final callbacks can only be installed during
        backward pass"), so a hook on the loss tensor installs it when its
        gradient is computed; the engine then runs it after the WHOLE
        backward completes. Under no_grad (validation loss) the graph is
        dead and release happens immediately.
        """
        if loss.requires_grad and torch.is_grad_enabled():

            def _hook(_grad) -> None:
                try:
                    torch.autograd.Variable._execution_engine.queue_callback(
                        release
                    )
                except Exception:
                    # Fallback: keep the VAE on-device rather than corrupt
                    # the graph — a small residency leak beats a broken
                    # backward; the next acquire() re-places idempotently.
                    pass

            loss.register_hook(_hook)
            return
        release()

    def to(self, device: torch.device, dtype: torch.dtype) -> "PerceptualFlowMatchingLoss":
        # setup_loss_modules() calls this after the transformer is loaded and
        # before the optimizer is created: build + place the encoders here so
        # the checkpoint load happens once, outside the training loop. The VAE
        # is loaded lazily on first compute (it needs the adapter).
        self._device = device
        for item in self._ensure_encoders():
            item["model"].to(device=device, dtype=self._encoder_dtype)
        return self

    def parameters(self) -> list:
        return []  # encoder AND VAE are frozen; nothing joins the optimizer

    def requires(self) -> list:
        return ["model_pred", "noise", "learning_target", "sigmas"]

    # ── pieces ────────────────────────────────────────────────────────

    def _prepare_pixels(self, pixels_pm1: torch.Tensor) -> torch.Tensor:
        """[-1, 1] pixels -> [0, 1] -> resize -> ImageNet normalize (fp32)."""
        from UnifiedTrainer.utils.perceptual_encoder import prepare_pixels

        return prepare_pixels(pixels_pm1, self.loss_resolution)

    def _encode_entry(self, entry: dict, model, pixels_pm1: torch.Tensor):
        """One encoder entry -> comparable object.

        Feature encoders (radio/eupe/dinov2/dinov3): list of per-layer (B, C,
        h, w) fp32 grids. dinov3 (ConvNeXt) grids are per-STAGE — channels and
        stride differ between layers (that is fine: pred/target pass through
        the SAME encoder, so shapes always match within a compared pair).
        vgg_lpips: the resized [-1,1] pixels tensor itself (the
        LPIPS model consumes pixels directly). Gradient context is inherited
        from the caller — pred-side pixels carry the graph.
        """
        net = str(entry["net"])
        size = int(entry["input_size"])
        if net == "vgg_lpips":
            from UnifiedTrainer.utils.lpips_vgg import lpips_vgg_resize

            return lpips_vgg_resize(pixels_pm1, size)
        if net.startswith("radio"):
            from UnifiedTrainer.utils.perceptual_encoder import (
                encode_radio_layers,
            )

            return encode_radio_layers(
                model, pixels_pm1, entry["layers"], size, self._encoder_dtype
            )
        if net.startswith("dinov2"):
            from UnifiedTrainer.utils.perceptual_encoder import (
                encode_dinov2_layers,
            )

            return encode_dinov2_layers(
                model, pixels_pm1, entry["layers"], size, self._encoder_dtype
            )
        if net.startswith("dinov3"):
            from UnifiedTrainer.utils.perceptual_encoder import (
                encode_dinov3_convnext_layers,
            )

            return encode_dinov3_convnext_layers(
                model, pixels_pm1, entry["layers"], size, self._encoder_dtype
            )
        from UnifiedTrainer.utils.perceptual_encoder import encode_eupe_layers

        return encode_eupe_layers(
            model, pixels_pm1, entry["layers"], size, self._encoder_dtype
        )

    def _distance_entry(self, entry: dict, model, pred_comp, tgt_comp) -> torch.Tensor:
        """Entry distance d(·,·) — official-parity metrics.

        vgg_lpips: LPIPS(pred_px, tgt_px).mean(). Feature encoders: tokens
        (B, N, C) per layer; "cosine" = 1 - cos (the official dino branch),
        "l1"/"l2"/"mse" on raw (or per-token-normalized) features; multiple
        layers are averaged (official: sum over layers / len(selected)).
        """
        if str(entry["net"]) == "vgg_lpips":
            from UnifiedTrainer.utils.lpips_vgg import lpips_vgg_distance

            return lpips_vgg_distance(model, pred_comp, tgt_comp, entry["input_size"])

        metric = entry["metric"]
        token_norm = entry["token_norm"]

        def _one(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            p = p.flatten(2).transpose(1, 2)
            t = t.flatten(2).transpose(1, 2)
            if metric == "cosine":
                p = F.normalize(p.float(), dim=-1)
                t = F.normalize(t.float(), dim=-1)
                return (1.0 - (p * t).sum(-1)).mean()
            if token_norm:
                p = F.normalize(p.float(), dim=-1)
                t = F.normalize(t.float(), dim=-1)
            if metric == "l1":
                return (p - t).abs().mean()
            return ((p - t) ** 2).mean()

        if torch.is_tensor(pred_comp):
            return _one(pred_comp, tgt_comp)
        return torch.stack(
            [_one(p, t) for p, t in zip(pred_comp, tgt_comp)]
        ).mean()

    def _decode_pair(
        self,
        vae: torch.nn.Module,
        adapter,
        x0_hat: torch.Tensor,
        x0: torch.Tensor,
    ):
        """Decode pred (with grad) and target (no grad) to [-1, 1] pixels."""
        with torch.no_grad():
            tgt_pixels = adapter.decode_latent_differentiable(vae, x0)
        pred_pixels = adapter.decode_latent_differentiable(vae, x0_hat)
        return pred_pixels, tgt_pixels

    def _feat_cache_key(self, x0: torch.Tensor) -> str:
        """Content hash of a clean latent — stable per dataset sample."""
        b = x0.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        return hashlib.sha1(b).hexdigest()

    # Rec.601 luma coefficients for the tone anchor ([-1,1] pixels).
    _TONE_Y = (0.299, 0.587, 0.114)

    @staticmethod
    def _tone_stats(pixels_pm1: torch.Tensor) -> torch.Tensor:
        """Per-sample Rec.601 luma (mean, std) on [-1,1] pixels -> (B, 2).

        ``pixels_pm1``: (B, 3, H, W) in [-1, 1] (decode output). Mean pins
        exposure, std pins contrast — the two zero directions of perceptual
        features. Population std (unbiased=False). All fp32; the pred-side
        call stays IN the autograd graph (grad flows luma stats -> decoded
        pixels -> VAE decode -> x̂0 -> model_pred).
        """
        y = (
            pixels_pm1.float()
            * torch.tensor(
                PerceptualFlowMatchingLoss._TONE_Y,
                device=pixels_pm1.device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
        ).sum(dim=1)
        m = y.mean(dim=(-1, -2))
        s = y.std(dim=(-1, -2), unbiased=False)
        return torch.stack([m, s], dim=1)

    def _freq_gaussian_kernel(self, device: torch.device) -> torch.Tensor:
        """Fixed depthwise Gaussian blur kernel (fp32, RGB, built once)."""
        if self._freq_kernel is None or self._freq_kernel.device != device:
            k = self.freq_kernel_size
            half = k // 2
            ax = torch.arange(k, device=device, dtype=torch.float32) - half
            g1 = torch.exp(-(ax * ax) / (2.0 * self.freq_kernel_sigma**2))
            g1 = g1 / g1.sum()
            g2 = torch.outer(g1, g1)
            self._freq_kernel = g2.expand(3, 1, k, k).contiguous()
        return self._freq_kernel

    def _high_pass(self, pixels_pm1: torch.Tensor) -> torch.Tensor:
        """HP(x) = x − gauss_blur(x) on [-1, 1] pixels, in fp32.

        ``pixels_pm1``: (B, 3, H, W). Replicate-padded blur (zero padding
        would darken the border and bias the residual at the frame edge).
        Differentiable — the pred-side call stays in the autograd graph
        (grad flows HP -> decoded pixels -> VAE decode -> x̂0 -> v̂).
        """
        x = pixels_pm1.float()
        p = self.freq_kernel_size // 2
        xp = F.pad(x, (p, p, p, p), mode="replicate")
        blur = F.conv2d(
            xp,
            self._freq_gaussian_kernel(x.device),
            padding=0,
            groups=3,
        )
        return x - blur

    def _target_features(
        self, vae: torch.nn.Module, adapter, x0: torch.Tensor,
        want_hp: bool = False,
    ) -> tuple:
        """Target-branch comparables (+ tone stats, + HP residual), cached.

        Returns ``(feats, tone, hp)``: ``feats`` is a per-SAMPLE list, each
        item being a per-ENTRY list of comparables on the latents' device
        (feature grids for radio/eupe entries, resized [-1,1] pixels for the
        vgg_lpips entry); ``tone`` is a (B, 2) no-grad tensor of per-sample
        Rec.601 luma (mean, std) from the DECODED target pixels (pre-resize,
        so independent of encoder input sizes); ``hp`` is a (B, 3, H, W)
        fp32 no-grad high-pass residual of the decoded target pixels at
        FULL decode resolution (the freq anchor compares it against the
        pred-side HP — downsampling would destroy exactly the band it
        supervises), or None when ``want_hp`` is false. Cache hits skip the
        VAE decode AND the encoder pass entirely; misses are computed (no
        grad) and stored on CPU (HP in fp16). Entries written before the
        freq anchor (2-tuples) or without HP are recomputed when HP is
        requested. First epoch warms the cache, later epochs only pay the
        pred-side branch.
        """
        device = x0.device
        feats: List[List] = [None] * x0.shape[0]  # type: ignore[list-item]
        tone: List = [None] * x0.shape[0]
        hp: List = [None] * x0.shape[0]
        missing: list = []
        for i in range(x0.shape[0]):
            key = self._feat_cache_key(x0[i])
            hit = self._feat_cache.get(key)
            if hit is not None:
                per, (tm, ts) = hit[0], hit[1]
                t_hp = hit[2] if len(hit) > 2 else None
                if want_hp and t_hp is None:
                    # Pre-freq cache entry — recompute so HP gets stored.
                    missing.append((i, key))
                    continue
                feats[i] = [
                    [
                        f.to(device=device, dtype=torch.float32) for f in per_entry
                    ]
                    for per_entry in per
                ]
                tone[i] = (tm, ts)
                if want_hp:
                    hp[i] = t_hp.to(device=device, dtype=torch.float32)
            else:
                missing.append((i, key))
        if missing:
            enc_items = self._ensure_encoders()
            miss_idx = torch.tensor([i for i, _ in missing], device=device)
            x0m = x0[miss_idx]
            if self.decode_scale != 1.0:
                x0m = F.interpolate(
                    x0m,
                    scale_factor=self.decode_scale,
                    mode="bilinear",
                    align_corners=False,
                )
            with torch.no_grad():
                tgt_pixels = adapter.decode_latent_differentiable(
                    vae, x0m, tile_size=self.vae_tile_size
                )
                tone_miss = PerceptualFlowMatchingLoss._tone_stats(tgt_pixels)
                hp_miss = self._high_pass(tgt_pixels) if want_hp else None
                per_entry_batches: List[List] = []
                for item in enc_items:
                    comps = self._encode_entry(item["cfg"], item["model"], tgt_pixels)
                    if torch.is_tensor(comps):
                        per_entry_batches.append([comps])
                    else:
                        per_entry_batches.append(list(comps))
            for j, (i, key) in enumerate(missing):
                per = [
                    [t[j: j + 1].detach().cpu() for t in entry_comps]
                    for entry_comps in per_entry_batches
                ]
                tm = float(tone_miss[j, 0])
                ts = float(tone_miss[j, 1])
                hp_i = (
                    hp_miss[j: j + 1].detach().to(torch.float16).cpu()
                    if want_hp
                    else None
                )
                if len(self._feat_cache) >= self._feat_cache_cap:
                    self._feat_cache.pop(next(iter(self._feat_cache)))
                self._feat_cache[key] = (per, (tm, ts), hp_i)
                feats[i] = [
                    [
                        f.to(device=device, dtype=torch.float32) for f in entry_comps
                    ]
                    for entry_comps in per
                ]
                tone[i] = (tm, ts)
                if want_hp:
                    hp[i] = hp_i.to(device=device, dtype=torch.float32)
        tone_t = (
            torch.tensor(
                [[tone[i][0], tone[i][1]] for i in range(len(tone))],
                device=device,
                dtype=torch.float32,
            )
            if all(v is not None for v in tone)
            else None
        )
        hp_t = (
            torch.cat(hp, dim=0)
            if (want_hp and all(v is not None for v in hp))
            else None
        )
        return feats, tone_t, hp_t

    # ── precache (persisted decoded-mode target features) ────────────

    def precache_targets(
        self,
        datarows: list,
        cache_mgr: Any,
        adapter: Any,
        device: torch.device,
        target_keys: Optional[set] = None,
    ) -> tuple:
        """Persist φ(D(x0)) target features for every cached image sample.

        The decoded-mode target branch is deterministic per cached latent
        (latents are immutable, encoders frozen, input sizes fixed) — the
        in-memory cache merely warms it on the first epoch. This method
        computes the SAME features once, before training, and stores them
        next to the latent npz as ``{basename}_{res}_pfm.npz`` in a
        multi-entry layout::

            target="decoded"           # marker (distinguishes this layout)
            feat_{e} = (L, C, h, w)    # fp16 feature grids, entry e
            pix_{e}  = (C, h, w)       # fp16 resized [-1, 1] pixels, entry e

        The cache itself is NOT recreated: no latent/caption re-encode, no
        per-sample JSON rewrite, no index rewrite — only additive pfm
        feature files. Training then serves the target branch from these
        files (via batch pfm_feats) instead of paying a VAE decode plus
        encoder passes every epoch.

        ``target_keys``: which per-sample target roles to precache (the
        roles the trainer actually consumes as ``learning_target`` — the
        union of ``batch_configs[*].target_config``). When None, ALL image
        target roles are precached (e.g. caches whose samples carry
        alternative target roles like T/D/CF/DC would otherwise write
        unused feature files for every role).

        Files whose marker already says "decoded" are reused; anything else
        (legacy/unreadable files, e.g. original-mode feats) is recomputed
        and overwritten so the file always matches THIS run's semantics.

        Returns (created, reused, failed).
        """
        enc_items = self._ensure_encoders()
        # Belt-and-braces placement: the trainer may not have moved this
        # loss instance yet — make sure every encoder sits on `device`.
        for item in enc_items:
            p0 = next(item["model"].parameters(), None)
            if p0 is not None and p0.device != device:
                item["model"].to(device=device, dtype=self._encoder_dtype)
        self._device = device
        vae, release = self._acquire_vae(adapter, device)
        try:
            created = reused = failed = 0
            t0 = time.time()
            for dr in datarows:
                jp = dr.get("json_path") if isinstance(dr, dict) else None
                if not jp:
                    continue
                try:
                    sample = cache_mgr.load_sample(jp)
                except Exception:
                    continue
                targets = sample.get("targets") or {}
                if target_keys is not None:
                    targets = {
                        k: v for k, v in targets.items() if k in target_keys
                    }
                for t_entry in targets.values():
                    if not isinstance(t_entry, dict):
                        continue
                    if t_entry.get("media") == "video":
                        continue  # pfm features are image-only
                    lp = t_entry.get("latent_path", "")
                    if not lp.endswith(".npz"):
                        continue
                    pfm_path = lp.replace(".npz", "_pfm.npz")
                    if os.path.exists(pfm_path):
                        try:
                            with np.load(pfm_path) as _d:
                                if str(_d["target"].item()) == "decoded":
                                    # Verify the layout matches THIS config's
                                    # encoder entries AND decode_scale — stale
                                    # files (wrong layer count, missing/wrong
                                    # entries, different decode scale) are
                                    # recomputed instead of failing at
                                    # training time.
                                    _scale_ok = float(
                                        _d["scale"].item()
                                    ) == self.decode_scale
                                    _tile_ok = (
                                        int(_d["tile"].item())
                                        if "tile" in _d.files
                                        else 0
                                    ) == self.vae_tile_size
                                    # "layout": aspect policy of the stored
                                    # comparables. Files written before the
                                    # aspect-preserving fix carry "square"
                                    # (or no field) — their token grids are
                                    # forced-square and mismatch the
                                    # preserve-shaped pred branch → recompute.
                                    _asp_ok = str(
                                        _d["layout"].item()
                                        if "layout" in _d.files
                                        else "square"
                                    ) == "preserve"
                                    _layout_ok = True
                                    for e, item in enumerate(enc_items):
                                        fk = f"feat_{e}"
                                        pk = f"pix_{e}"
                                        if fk in _d.files:
                                            if (
                                                _d[fk].ndim != 4
                                                or _d[fk].shape[0]
                                                != len(item["cfg"]["layers"])
                                            ):
                                                _layout_ok = False
                                                break
                                        elif pk in _d.files:
                                            if _d[pk].ndim != 3:
                                                _layout_ok = False
                                                break
                                        else:
                                            _layout_ok = False
                                            break
                                    # tone anchor: tone_weight>0 requires the
                                    # stored target luma scalars; tone-less
                                    # files (older caches) are recomputed.
                                    _tone_ok = (
                                        self.tone_weight <= 0
                                        or (
                                            "tone_mean" in _d.files
                                            and "tone_std" in _d.files
                                        )
                                    )
                                    if (
                                        _layout_ok
                                        and _scale_ok
                                        and _asp_ok
                                        and _tile_ok
                                        and _tone_ok
                                    ):
                                        reused += 1
                                        continue
                        except Exception:
                            pass  # unreadable/legacy file → recompute below
                    try:
                        latent = cache_mgr.load_latent(lp)
                        if latent.ndim == 4 and latent.shape[1] == 1:
                            latent = latent.squeeze(1)
                        out: dict = {
                            "target": "decoded",
                            "scale": self.decode_scale,
                            "tile": self.vae_tile_size,
                            "layout": "preserve",
                        }
                        with torch.no_grad():
                            _lat = latent.unsqueeze(0).to(device)
                            if self.decode_scale != 1.0:
                                _lat = F.interpolate(
                                    _lat,
                                    scale_factor=self.decode_scale,
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            pixels = adapter.decode_latent_differentiable(
                                vae, _lat, tile_size=self.vae_tile_size
                            )
                            # Tone anchor: target luma scalars from the
                            # DECODED pixels (pre-resize), stored only when
                            # the run uses the anchor (tone_weight>0) so
                            # tone-less runs keep reusing tone-less files.
                            if self.tone_weight > 0.0:
                                _tt = self._tone_stats(pixels)  # (1, 2)
                                out["tone_mean"] = float(_tt[0, 0])
                                out["tone_std"] = float(_tt[0, 1])
                            for e, item in enumerate(enc_items):
                                cfg_e = item["cfg"]
                                comps = self._encode_entry(
                                    cfg_e, item["model"], pixels
                                )
                                if torch.is_tensor(comps):
                                    # vgg_lpips: resized [-1, 1] pixels
                                    out[f"pix_{e}"] = (
                                        comps[0]
                                        .detach()
                                        .to(torch.float16)
                                        .cpu()
                                        .numpy()
                                    )
                                else:
                                    # feature encoder: per-layer grids —
                                    # stack (L, B=1, C, h, w), strip the
                                    # batch slot ([:, 0]) keeping ALL layers.
                                    out[f"feat_{e}"] = (
                                        torch.stack(comps, dim=0)[:, 0]
                                        .detach()
                                        .to(torch.float16)
                                        .cpu()
                                        .numpy()
                                    )
                        np.savez(pfm_path, **out)
                        created += 1
                        if created % 50 == 0:
                            logger.info(
                                f"pfm precache: {created} created, {reused} "
                                f"reused, {failed} failed "
                                f"({time.time() - t0:.0f}s elapsed)"
                            )
                    except Exception as e:
                        failed += 1
                        logger.warning(
                            f"pfm precache failed for {lp!r}: {e}",
                            exc_info=True,
                        )
            logger.info(
                f"pfm precache done: created={created}, reused={reused}, "
                f"failed={failed} in {time.time() - t0:.0f}s"
            )
            return created, reused, failed
        finally:
            release()

    # ── loss ──────────────────────────────────────────────────────────

    def _paper_x0_hat(self, context: LossContext) -> torch.Tensor:
        """Paper Eq.4: ``x0_hat = xt − σt·v̂`` with ``xt = (1−σt)x0 + σt·ε``.

        context.x0_hat (built by the trainer via base.compute_x0_hat) is the
        algebraic estimator ``ε − v̂`` — exact at a perfect v̂ but with the
        WRONG error scaling: the paper's estimator scales the velocity error
        by σt, which (a) concentrates perceptual supervision on high-noise
        steps and (b) supervises exactly the one-step quantity that
        consistency sampling evaluates at inference. So PFM rebuilds x̂0 from
        the context primitives (x0, ε, σ, v̂) instead of trusting it.
        """
        v = context.model_pred
        x0 = context.learning_target.detach()
        eps = context.noise.detach()
        sig = context.sigmas.detach().float().reshape(-1)
        if sig.numel() != x0.shape[0]:
            raise ValueError(
                f"pfm: sigma count {sig.numel()} != batch {x0.shape[0]} — "
                "cannot build the Eq.4 x0_hat"
            )
        s = sig.reshape(-1, *([1] * (x0.ndim - 1)))
        xt = (1.0 - s) * x0.float() + s * eps.float()
        return (xt - s * v.float()).to(v.dtype)

    def compute(self, context: LossContext) -> torch.Tensor:
        # Components are recomputed per step; a fully gated-off step (empty
        # idx) leaves the dict empty so no stale keys reach the logs.
        self.last_components = {}
        pred = context.model_pred
        adapter = context.adapter
        if adapter is None or not hasattr(adapter, "decode_latent_differentiable"):
            raise RuntimeError(
                "pfm: the model adapter must implement decode_latent_differentiable() "
                "(gradient-enabled decode); see Krea2Adapter for the reference override."
            )

        # Apply-probability gate — thin the loss along the trajectory while
        # exploring (sigma-range gating removed; every sample participates).
        active = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        if self.apply_prob < 1.0:
            active &= torch.rand(pred.shape[0], device=pred.device) < self.apply_prob
        gate_idx = (~active).nonzero(as_tuple=True)[0]

        x0_hat = self._paper_x0_hat(context)
        # Clamp x̂0 into the data latent range BEFORE decoding. At σ≈1 the
        # paper estimator approaches σ(ε−v̂) — a large-magnitude latent whose
        # decode overflows bf16 activations (inf features → NaN loss, and via
        # backward, poisoned weights). Clamping to ±6σ of the batch's own x0
        # leaves normal samples untouched (clamp passes gradients a.e.) and
        # zeroes them only in the pathological tail.
        _lim = float(6.0 * context.learning_target.detach().float().std())
        _lim = max(_lim, 1.0)
        x0_hat = x0_hat.clamp(-_lim, _lim)

        self._ensure_encoders()
        # Belt-and-braces placement: an encoder may have been built lazily
        # after the loss module was moved, so it can end up on a different
        # device than the latents. The whole pfm graph must be single-device
        # or backward dies on a cross-device conv.
        for item in self._encoders:
            p0 = next(item["model"].parameters(), None)
            if p0 is not None and p0.device != pred.device:
                item["model"].to(device=pred.device, dtype=self._encoder_dtype)
                self._device = pred.device

        idx = active.nonzero(as_tuple=True)[0]
        loss = None
        release = None
        pfm_loss = None
        if idx.numel() > 0:
            x0_hat_b = x0_hat[idx]
            if self.decode_scale != 1.0:
                x0_hat_b = F.interpolate(
                    x0_hat_b,
                    scale_factor=self.decode_scale,
                    mode="bilinear",
                    align_corners=False,
                )
            vae, release = self._acquire_vae(adapter, x0_hat_b.device)
            try:
                # Pred branch: x̂0 changes every step — decode ONCE (grad);
                # every encoder entry consumes the same decoded pixels.
                pred_pixels = adapter.decode_latent_differentiable(
                    vae, x0_hat_b, tile_size=self.vae_tile_size
                )
                pred_comps = [
                    self._encode_entry(item["cfg"], item["model"], pred_pixels)
                    for item in self._encoders
                ]
                # Tone anchor term (computed in the decoded branch where the
                # target stats exist; stays None in original mode).
                tone_loss = None
                freq_loss = None
                # Freq-anchor gate: which active samples sit in the LOW-noise
                # regime (their x̂0 HP band is meaningful — at high σ x̂0 is
                # noise-dominated). Decided upfront so the target branch
                # knows to carry the HP residual.
                if idx.numel() > 0:
                    _sig_b = context.sigmas.detach().float().reshape(-1)[idx]
                    _want_hp = self.freq_weight > 0.0 and bool(
                        (_sig_b <= self.freq_sigma_max).any()
                    )
                else:
                    _sig_b = None
                    _want_hp = False
                if self.target_source == "original":
                    # φ(I_orig) precomputed at CACHE BUILD time (user goal:
                    # "生成的图尽可能像真实照片") — the loss only looks up
                    # features; no VAE decode, no encoder pass, ever.
                    feats_batch = (context.extra or {}).get("pfm_feats")
                    if feats_batch is None:
                        raise RuntimeError(
                            "pfm: target_source='original' but the batch "
                            "carries no pfm_feats — the cache was built "
                            "without feature files. Trigger a cache rebuild "
                            "(recreate_latents or delete the cache dir) so "
                            "{basename}_{res}_pfm.npz files are created."
                        )
                    feats_all = feats_batch.to(
                        device=x0_hat_b.device, dtype=torch.float32
                    )
                    # Strict contract: (L, C, h, w), one grid per requested
                    # layer (the trainer strips the builder's batch slot).
                    entry0 = self._encoder_cfgs[0]
                    if feats_all.dim() != 4 or feats_all.shape[0] != len(
                        entry0["layers"]
                    ):
                        raise RuntimeError(
                            "pfm: pfm_feats must be (L, C, h, w) with L="
                            f"{len(entry0['layers'])}, got "
                            f"{tuple(feats_all.shape)}"
                        )
                    tgt_comps = [
                        [feats_all[l: l + 1] for l in range(len(entry0["layers"]))]
                    ]
                else:
                    # decoded mode: φ(D(x0)). The trainer carries per-sample
                    # precached features (written by precache_targets before
                    # training) in context.extra['pfm_feats'] — use them
                    # directly; otherwise fall back to the on-the-fly path
                    # (in-memory comparables cache, warms on the first epoch).
                    _extra_feats = (context.extra or {}).get("pfm_feats")
                    tgt_comps = None
                    _used_precache = False
                    if isinstance(_extra_feats, dict) and _extra_feats:
                        # Tone anchor: precached files must carry the target
                        # luma scalars; tone-less files (older caches) fall
                        # back to the on-the-fly path which recomputes them.
                        if self.tone_weight > 0 and not (
                            "tone_mean" in _extra_feats
                            and "tone_std" in _extra_feats
                        ):
                            tgt_comps = None
                        else:
                            _used_precache = True
                            tgt_comps = []
                            for e, item in enumerate(self._encoders):
                                cfg_e = item["cfg"]
                                fk = f"feat_{e}"
                                pk = f"pix_{e}"
                                if fk in _extra_feats:
                                    feats_all = _extra_feats[fk].to(
                                        device=x0_hat_b.device, dtype=torch.float32
                                    )
                                    n_layers = len(cfg_e["layers"])
                                    if (
                                        feats_all.dim() != 4
                                        or feats_all.shape[0] != n_layers
                                    ):
                                        raise RuntimeError(
                                            f"pfm: precached feat_{e} must be "
                                            f"(L, C, h, w) with L={n_layers}, got "
                                            f"{tuple(feats_all.shape)} — delete the "
                                            "sample's *_pfm.npz and re-run the "
                                            "precache pass"
                                        )
                                    tgt_comps.append(
                                        [feats_all[l: l + 1] for l in range(n_layers)]
                                    )
                                elif pk in _extra_feats:
                                    tgt_comps.append(
                                        [
                                            _extra_feats[pk]
                                            .to(
                                                device=x0_hat_b.device,
                                                dtype=torch.float32,
                                            )
                                            .unsqueeze(0)
                                        ]
                                    )
                                else:
                                    _used_precache = False
                                    tgt_comps = None
                                    break
                    if tgt_comps is None:
                        tgt_per_sample, tgt_tone, tgt_hp_batch = (
                            self._target_features(
                                vae,
                                adapter,
                                context.learning_target[idx].detach(),
                                want_hp=_want_hp,
                            )
                        )
                        n_entries = len(self._encoders)
                        tgt_comps = [
                            [
                                torch.cat(
                                    [
                                        tgt_per_sample[i][e][l]
                                        for i in range(len(tgt_per_sample))
                                    ],
                                    dim=0,
                                )
                                for l in range(len(tgt_per_sample[0][e]))
                            ]
                            for e in range(n_entries)
                        ]
                    # ── Tone anchor (design doc 2026-09-01) ──
                    # L_tone = (m_pred−m_tgt)² + (s_pred−s_tgt)² on Rec.601
                    # luma of the DECODED pixels, same idx mask as the main
                    # term. Pred stats stay in-graph; target stats are
                    # constants (precached npz scalars or no-grad decode).
                    tone_loss = None
                    if self.tone_weight > 0.0:
                        pred_tone = self._tone_stats(pred_pixels)  # (B, 2)
                        if _used_precache:
                            _tt = torch.tensor(
                                [
                                    [
                                        float(_extra_feats["tone_mean"]),
                                        float(_extra_feats["tone_std"]),
                                    ]
                                ],
                                device=pred_tone.device,
                                dtype=torch.float32,
                            )
                        else:
                            _tt = tgt_tone  # (B, 2) no-grad, aligned with idx
                        if _tt is not None:
                            tone_loss = ((pred_tone - _tt.detach()) ** 2).mean()
                        elif logger.isEnabledFor(logging.WARNING):
                            logger.warning(
                                "pfm: tone_weight>0 but no target luma stats "
                                "available (original-mode cache build?) — "
                                "tone anchor inactive for this step"
                            )
                    # ── High-frequency residual anchor (HP-L1) ──
                    # L_freq = |HP(pred_px) − HP(tgt_px)|₁ over the LOW-σ
                    # subset of the batch. Target HP comes from the
                    # comparables cache (no-grad, FULL decode resolution —
                    # downsampling would destroy exactly the band it
                    # supervises); the pred-side HP stays in the autograd
                    # graph. freq_loss stays None when the batch has no
                    # low-σ sample or no target HP (precached-npz batches).
                    if _want_hp:
                        if _used_precache:
                            if logger.isEnabledFor(logging.WARNING):
                                logger.warning(
                                    "pfm: freq_weight>0 but the batch target "
                                    "came from precached npz features (no "
                                    "target pixels to take a high-pass of) "
                                    "— freq anchor inactive for this step"
                                )
                        elif tgt_hp_batch is not None:
                            _fmask = _sig_b <= self.freq_sigma_max
                            _hp_pred = self._high_pass(pred_pixels[_fmask])
                            freq_loss = (
                                _hp_pred - tgt_hp_batch[_fmask]
                            ).abs().mean()
            except BaseException:
                release()
                raise
            per_entry_losses = []
            for e, item in enumerate(self._encoders):
                cfg_e = item["cfg"]
                if cfg_e["weight"] <= 0.0:
                    continue
                pred_c = pred_comps[e]
                tgt_c = tgt_comps[e]
                # The target cache wraps tensor-comparables (vgg_lpips
                # pixels) in a single-element list; feature entries stay
                # per-layer lists. Normalize the pixel case so both sides
                # are bare tensors.
                if torch.is_tensor(pred_c) and isinstance(tgt_c, list):
                    tgt_c = tgt_c[0]
                d = self._distance_entry(cfg_e, item["model"], pred_c, tgt_c)
                per_entry_losses.append(cfg_e["weight"] * d)
            # ── Low-σ FM blend (fm_mix_weight>0) ──
            # main·(1−w·frac) + w·frac·FM-MSE on the σ ≤ fm_sigma_max
            # subset. See the __init__ comment for the rationale; frac
            # scaling keeps mixed batches expectation-correct.
            _main_scale, _fm_scale, fm_loss = 1.0, 0.0, None
            if self.fm_mix_weight > 0.0 and _sig_b is not None:
                _lmask = _sig_b <= self.fm_sigma_max
                _frac = float(_lmask.float().mean())
                if _frac > 0.0:
                    _v = pred[idx][_lmask]
                    _vt = adapter.compute_target(
                        context.noise[idx][_lmask].detach(),
                        context.learning_target[idx][_lmask].detach(),
                    ).to(_v.dtype)
                    fm_loss = ((_v - _vt) ** 2).mean()
                    _fm_scale = self.fm_mix_weight * _frac
                    _main_scale = 1.0 - _fm_scale

            pfm_main = None
            if per_entry_losses:
                # Official parity: plain weighted sum over encoder entries.
                pfm_main = torch.stack(per_entry_losses).sum()
                pfm_loss = _main_scale * pfm_main
            if tone_loss is not None:
                pfm_loss = (
                    pfm_loss + self.tone_weight * tone_loss
                    if pfm_loss is not None
                    else self.tone_weight * tone_loss
                )
            if fm_loss is not None:
                pfm_loss = (
                    pfm_loss + _fm_scale * fm_loss
                    if pfm_loss is not None
                    else _fm_scale * fm_loss
                )
            if freq_loss is not None:
                pfm_loss = (
                    pfm_loss + self.freq_weight * freq_loss
                    if pfm_loss is not None
                    else self.freq_weight * freq_loss
                )
            loss = pfm_loss
            # Component telemetry — applied (weighted) values so
            # main + fm + tone + freq sum exactly to the logged loss/pfm.
            # Empty when the whole step was gated off.
            self.last_components = {
                "main": (
                    float(_main_scale * pfm_main) if pfm_main is not None else 0.0
                ),
                "fm": (
                    float(_fm_scale * fm_loss) if fm_loss is not None else 0.0
                ),
                "tone": (
                    float(self.tone_weight * tone_loss)
                    if tone_loss is not None
                    else 0.0
                ),
                "freq": (
                    float(self.freq_weight * freq_loss)
                    if freq_loss is not None
                    else 0.0
                ),
            }

        # ── FM-MSE fallback for apply_prob-gated samples ──────────────
        # Samples dropped by apply_prob (< 1.0) still get plain velocity-MSE
        # supervision so every step trains (no wasted decode/backward).
        if self.fallback_fm_mse and gate_idx.numel() > 0:
            v_g = pred[gate_idx]
            vel_target = adapter.compute_target(
                context.noise[gate_idx].detach(),
                context.learning_target[gate_idx].detach(),
            ).to(v_g.dtype)
            fm = ((v_g - vel_target) ** 2).mean()
            loss = fm if loss is None else loss + self.fallback_weight * fm

        if loss is None:
            # Nothing active and no fallback (defensive) — keep the graph
            # connected so pfm can be the sole loss in a pipeline.
            return pred.sum() * 0.0
        if pfm_loss is not None:
            # The decode graph references the VAE weights: release only
            # after backward has consumed them (immediate under no_grad).
            self._release_after_backward(pfm_loss, release)
        return loss
