# =============================================================================
# DERIVED CODE - MOSS timestep sampler + UnifiedTrainer batch bridge.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# `sample_timesteps` re-expresses
#   src/kohakumoss/diffusion/timestep/{moss.py,shift.py}
# (truncated logit-normal pmf over the discrete sigma grid, survival-side bins
# above the median, static shift `s*t/(1+(s-1)*t)`, staged table).  The batch
# bridge is new code for the official `MOSSImageTransformer` forward signature.
# =============================================================================
"""Bridge a UnifiedTrainer batch to ``MOSSImageTransformer.forward``.

The official transformer (vendored from the model repository) is a normal
diffusers-style model - no packed token axis::

    forward(hidden_states [B, 128, H, W], timestep,
            encoder_hidden_states [B, S, 4096], encoder_attention_mask [B, S] bool)

with ``timestep = sigma * 1000`` (the pipeline feeds ``scheduler.timesteps``,
which are the sigma grid scaled by ``num_train_timesteps``).  It loops over the
batch internally and drops padded text tokens, so batching needs no SeqInfo.
"""

from __future__ import annotations

import bisect
import math
from functools import lru_cache
from typing import Any

import torch

__all__ = [
    "prepare_model_input",
    "sample_timesteps",
    "unpack_prediction",
]

STEPS = 1000
#: (mu, s, shift, cutoff, uniform_mix, p_full) - MOSS tech report stage 3 defaults.
DEFAULT_SAMPLER = (-1.09, 0.95, 1.0, 0.95, 0.01, 0.0)
TECH_REPORT_STAGES = (
    (0, (0.0, 1.0, 3.0, None, 0.01, 0.01)),
    (140_000, (-1.45, 1.05, 1.0, None, 0.01, 0.01)),
    (207_000, (-1.45, 1.05, 1.0, 0.95, 0.01, 0.0)),
    (274_000, (-1.09, 0.95, 1.0, 0.95, 0.01, 0.0)),
)


# ── timesteps ───────────────────────────────────────────────────────────


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """``s*t / (1 + (s-1)*t)`` - the reference's static shift."""
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _cdf_sf(x: torch.Tensor, mu: float, s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(P(S <= x), P(S > x)) for S = 1 - sigmoid(Z), computed as two erfc."""
    x = x.clamp(0.0, 1.0)
    u = (torch.log1p(-x) - torch.log(x) - mu) / (s * math.sqrt(2.0))
    cdf, sf = 0.5 * torch.erfc(u), 0.5 * torch.erfc(-u)
    low, high = x <= 0, x >= 1
    cdf = torch.where(low, 0.0, torch.where(high, 1.0, cdf))
    sf = torch.where(low, 1.0, torch.where(high, 0.0, sf))
    return cdf, sf


def _pmf(
    steps: int, mu: float, s: float, cutoff: float | None, mix: float, p_full: float
) -> torch.Tensor:
    """pmf over sigma_j = j/steps, j = 1..steps (float32, sums to 1)."""
    a = 1.0 if cutoff is None else float(cutoff)
    j = torch.arange(steps + 1, dtype=torch.float32)
    cdf, sf = _cdf_sf(a * j / steps, mu, s)
    lower = cdf[1:] <= 0.5
    mass = torch.where(lower, cdf[1:] - cdf[:-1], sf[:-1] - sf[1:])
    mass = mass / cdf[-1].clamp_min(torch.finfo(torch.float32).tiny)
    pmf = (1.0 - mix) * mass + mix / steps
    pmf = pmf * (1.0 - p_full)
    pmf[-1] += p_full
    return pmf / pmf.sum()


@lru_cache(maxsize=8)
def _table(steps: int, params: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """(sigma grid, cdf) for one stage; the grid already carries the shift."""
    mu, s, shift, cutoff, mix, p_full = params
    pmf = _pmf(steps, mu, s, cutoff, mix, p_full)
    grid = torch.arange(1, steps + 1, dtype=torch.float32) / steps
    sigma = _shift_sigma(grid, shift)
    cdf = pmf.cumsum(0)
    cdf[-1] = 1.0
    return sigma, cdf


def _stage_params(schedule: Any, step: int) -> tuple:
    if schedule is None:
        return DEFAULT_SAMPLER
    if schedule == "tech_report":
        starts = [st for st, _ in TECH_REPORT_STAGES]
        return TECH_REPORT_STAGES[bisect.bisect_right(starts, step) - 1][1]
    starts = [int(st) for st, _ in schedule]
    entry = schedule[bisect.bisect_right(starts, step) - 1][1]
    return entry["params"] if isinstance(entry, dict) else entry


def _flux3_params(cfg: dict, latent_height: int | None, latent_width: int | None) -> tuple | None:
    """FLUX.3-style ``flow_shift`` / ``dynamic_shift`` mapped onto the MOSS pmf.

    FLUX.3 samples ``sigma = sigmoid(N(mu, sigmoid_scale))`` with
    ``mu = ln(flow_shift)`` (see ``models/flux3_klein``); MOSS's reference pmf uses
    ``sigma = 1 - sigmoid(Z)``, so the same distribution needs
    ``Z = -N(mu, scale)`` -> ``mu_moss = -ln(flow_shift)`` and ``s = sigmoid_scale``.
    ``dynamic_shift`` interpolates mu over the image token count exactly like
    ``Flux3Adapter``:  mu = base_shift + (max_shift - base_shift) *
    (seq - base_seq) / (max_seq - base_seq), with seq = h * w.

    Returns the 6-tuple the pmf table expects, or None when the mode is not a
    flux3 mode (the caller then uses ``timestep_schedule``).
    """
    mode = cfg.get("timestep_shift_mode")
    if mode not in ("flow_shift", "dynamic_shift"):
        return None
    # KohakuMoss default is the *training* sigma distribution: mu=-1.09, s=0.95,
    # shift=1.0, cutoff=0.95, uniform_mix=0.01 (mean sigma ~0.71).  A config that
    # names the flux3 mode but omits the knob must fall back to that default, not
    # to some arbitrary shift (the old fallback was 16.0 -> mean sigma 0.92, which
    # trains almost only at pure noise and destroys detail).
    if mode == "flow_shift" and cfg.get("flow_shift") is None:
        return None
    scale = float(cfg.get("sigmoid_scale", 1.0))
    if mode == "dynamic_shift":
        base_shift = float(cfg.get("base_shift", 1.5))
        max_shift = float(cfg.get("max_shift", 2.5))
        base_seq = float(cfg.get("base_image_seq_len", 256))
        max_seq = float(cfg.get("max_image_seq_len", 4096))
        seq = float((latent_height or 0) * (latent_width or 0) or base_seq)
        mu_flux = base_shift + (max_shift - base_shift) * (seq - base_seq) / (max_seq - base_seq)
    else:
        mu_flux = math.log(float(cfg["flow_shift"]))
    mu = -mu_flux  # sigma = 1 - sigmoid(mu) mirrors flux3's sigmoid(mu)
    return (mu, scale,
            float(cfg.get("timestep_shift", 1.0)),
            cfg.get("timestep_cutoff", 0.95),
            float(cfg.get("timestep_mix", 0.01)),
            float(cfg.get("timestep_p_full", 0.0)))


def sample_timesteps(
    adapter,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_height: int | None = None,
    latent_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``sigma`` from the MOSS pmf (discrete grid + truncated logit-normal).

    ``timestep_shift_mode`` selects the distribution: omit it (or ``"pmf"``) to use
    ``timestep_schedule`` / the MOSS tech-report defaults; ``"flow_shift"`` and
    ``"dynamic_shift"`` use FLUX.3's ``flow_shift`` / ``sigmoid_scale`` (or
    ``base_shift``/``max_shift``) knobs, mapped with ``mu_moss = -ln(flow_shift)``.

    Returns ``(timesteps, sigmas)``: ``sigmas`` in [0, 1] drive the flow-matching
    target, ``timesteps`` is what the backbone embeds (``sigma * 1000`` here is
    applied in :func:`prepare_model_input`, so both values stay in [0, 1]).
    """
    cfg = getattr(adapter, "config", {}) or {}
    step = int(getattr(adapter, "_global_step", 0) or 0)
    params = _flux3_params(cfg, latent_height, latent_width)
    if params is None:
        params = _stage_params(cfg.get("timestep_schedule"), step)
    steps = int(cfg.get("timestep_steps", STEPS))
    sigma_grid, cdf = _table(steps, tuple(params))
    # Optional low-noise window (edit training): keep only sigma <= cap and
    # renormalise the CDF over the surviving grid.
    cap = cfg.get("timestep_sigma_max")
    if cap is not None:
        keep = sigma_grid <= float(cap)
        if not bool(keep.any()):
            raise ValueError(f"timestep_sigma_max={cap} excludes the whole sigma grid")
        sigma_grid = sigma_grid[keep]
        cdf = cdf[keep]
        cdf = cdf / cdf[-1]
    u = torch.rand(batch_size)
    idx = torch.searchsorted(cdf, u).clamp_max(cdf.numel() - 1)
    sigmas = sigma_grid[idx].to(device=device, dtype=torch.float32)
    return sigmas.to(dtype=dtype), sigmas.to(dtype=dtype)


# ── batch bridge ────────────────────────────────────────────────────────

#: The official backbone embeds ``timestep`` directly as the sinusoid argument
#: and the pipeline feeds ``scheduler.timesteps`` == sigma * num_train_timesteps.
TIME_SCALE = 1000.0


def _as_tensor(value, device, dtype=None):
    """Cache entries arrive as numpy arrays (npz); accept tensor/numpy/list alike."""
    if value is None:
        return None
    if not torch.is_tensor(value):
        import numpy as np

        value = torch.from_numpy(np.asarray(value))
    value = value.to(device=device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value


def _text_tensors(batch: dict, device: torch.device, dtype: torch.dtype):
    """Per-sample caption embeddings -> padded ``(B, S, 4096)`` + bool mask."""
    embeddings = batch.get("embeddings") or []
    if not embeddings:
        raise RuntimeError(
            "MOSS: batch['embeddings'] is empty - caption embeddings were not cached. "
            "Check the cache_builder log for '[embed-cache][build] ... SKIP'."
        )
    embeds: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for entry in embeddings:
        embed = None
        mask = None
        if isinstance(entry, dict):
            embed = entry.get("prompt_embed")
            mask = entry.get("prompt_embeds_mask")
        elif torch.is_tensor(entry) or hasattr(entry, "__array__"):
            embed = entry
        if embed is None:
            raise RuntimeError("MOSS: an embeddings entry has no 'prompt_embed'.")
        embed = _as_tensor(embed, device, dtype)
        if embed.ndim == 3:  # (L, num_layers, dim) -> one vector per token
            embed = embed[:, -1]
        length = int(embed.shape[0])
        if mask is None:
            mask = torch.ones(length, dtype=torch.bool, device=device)
        else:
            mask = _as_tensor(mask, device).to(torch.bool).reshape(-1)[:length]
        embeds.append(embed)
        masks.append(mask)

    max_len = max(int(e.shape[0]) for e in embeds)
    dim = int(embeds[0].shape[-1])
    out = torch.zeros(len(embeds), max_len, dim, device=device, dtype=dtype)
    out_mask = torch.zeros(len(embeds), max_len, device=device, dtype=torch.bool)
    for i, (embed, mask) in enumerate(zip(embeds, masks)):
        n = int(embed.shape[0])
        out[i, :n] = embed
        out_mask[i, :n] = mask
    # The backend rejects a sample with no valid token.
    out_mask[:, 0] = True
    return out, out_mask


def _reference_group_names(adapter) -> list[str]:
    """Reference groups declared by the dataset configs, in declaration order."""
    names: list[str] = []
    data = (getattr(adapter, "config", {}) or {}).get("data", {}) or {}
    for ds in data.get("dataset_configs", []) or []:
        for name in (ds.get("reference_configs") or {}).keys():
            if name not in names:
                names.append(name)
    return names


def _reference_spans(adapter, batch: dict, batch_size: int):
    """``(per_sample_latents, per_sample_frames)`` for the DiT reference spans.

    Frame numbering is Krea-2's convention (and the frozen design in
    `E:\\MOSS_data_pipeline\\idea.txt`): the predicted latent is **frame 0** and
    every reference image is **frame 1, 2, 3, ...** in order.  There is no
    reserved slot: a reference is a reference.  A group declared as
    ``annotation_reference_config`` is only *ordered first*, so it naturally
    lands on frame 1 (the design's "1 = annotated layout"), the next one on 2 and
    the remaining object references on 3+.
    """
    if not getattr(adapter, "reference_conditioning", False):
        return None, None
    latents = batch.get("latents") or {}
    resolved = (batch.get("batch_configs") or [{}])[0]
    ref_key = resolved.get("reference_config")
    declared = _reference_group_names(adapter)
    annotation = (getattr(adapter, "config", {}) or {}).get("annotation_reference_config")

    groups: list[str] = []
    if ref_key and ref_key in latents:
        groups.append(ref_key)
    for name in declared:
        if name in latents and name not in groups:
            groups.append(name)
    # The annotated layout is frame 1, so it is laid out first.
    groups.sort(key=lambda name: (0 if name == annotation else 1, declared.index(name) if name in declared else 99))

    if not groups:
        if not getattr(adapter, "_moss_ref_missing_logged", False):
            adapter._moss_ref_missing_logged = True
            import logging

            logging.getLogger(__name__).warning(
                "MOSS: reference_conditioning is on but the batch carries no reference "
                "latents (reference_config=%r, batch latents=%s, declared groups=%s) - "
                "training runs without the latent reference span.",
                ref_key, sorted(latents.keys()), declared,
            )
        return None, None

    # Krea-2 numbering: target is frame 0, references are 1, 2, 3, ... in order.
    frames: list[int] = []
    tensors: list = []
    for group in groups:
        entry = latents.get(group)
        items = entry if isinstance(entry, (list, tuple)) else [entry]
        for t in [t for t in (items or []) if torch.is_tensor(t)]:
            frames.append(len(frames) + 1)
            tensors.append(t)

    if not tensors:
        return None, None

    per_sample_latents = []
    for i in range(batch_size):
        sample_items = []
        for t in tensors:
            if t.ndim == 4:
                sample_items.append(t[0] if t.shape[0] == 1 else t[min(i, t.shape[0] - 1)])
            else:
                sample_items.append(t)
        per_sample_latents.append(sample_items)
    per_sample_frames = [list(frames) for _ in range(batch_size)]

    if not getattr(adapter, "_moss_ref_logged", False):
        adapter._moss_ref_logged = True
        import logging

        tokens = [int(t.shape[-2] * t.shape[-1]) for t in per_sample_latents[0]]
        logging.getLogger(__name__).info(
            "MOSS reference spans: %d reference image(s) per sample at RoPE frames %s, "
            "tokens=%s (target r=0, references r=1..N like Krea-2; t=0 modulation, "
            "whole image mapped over the canvas)",
            len(per_sample_frames[0]), per_sample_frames[0], tokens,
        )
    return per_sample_latents, per_sample_frames


def prepare_model_input(adapter, batch: dict, noise, sigmas: torch.Tensor) -> dict:
    """UnifiedTrainer batch -> the packed MOSS forward kwargs (Krea-2 style).

    The adapter concatenates the target and the reference latents into one token
    sequence here, before the model::

        hidden_states    (B, N, C)  = [target | ref1 | ref2 | ...]
        target_grids     [(h, w)]   per sample
        reference_grids  [[(rh, rw), ...]]  per sample, in sequence order

    The model then embeds each span with its own ``video_embedder``, prepends the
    text tokens and applies the t = 0 modulation to the reference tail.
    """
    noises = [n for n in (noise if isinstance(noise, list) else [noise]) if n is not None]
    if len(noises) > 1:
        raise ValueError(
            "MOSS trains one target per step: the official transformer takes one 4D "
            f"[B, C, H, W] latent and got {len(noises)} targets. Use a single "
            "target_config (or run the targets as separate steps)."
        )
    target = noises[0]
    if target.ndim != 4 or target.shape[1] != adapter.latent_channels:
        raise ValueError(
            f"MOSS expects target latents [B, {adapter.latent_channels}, H, W] (32 VAE "
            f"channels patchified 2x2), got {tuple(target.shape)}."
        )
    device = target.device
    bs = target.shape[0]
    height, width = int(target.shape[-2]), int(target.shape[-1])

    text, mask = _text_tensors(batch, device, target.dtype)
    ref_latents, ref_frames = _reference_spans(adapter, batch, bs)

    packed = [target.flatten(2).transpose(1, 2)]  # (B, h*w, C)
    reference_grids: list[list[tuple[int, int]]] = [[] for _ in range(bs)]
    if ref_latents is not None:
        counts = {len(entry) for entry in ref_latents}
        if len(counts) > 1:
            raise ValueError(
                "MOSS packs [target | refs] into one rectangular token sequence, so every "
                f"sample in a batch needs the same reference count; got {sorted(counts)}. "
                "The data pipeline groups batches by reference signature - use "
                "batch_size=1 or drop the optional reference."
            )
        n_refs = len(ref_latents[0]) if ref_latents else 0
        for k in range(n_refs):
            per_sample = [ref_latents[i][k] for i in range(bs)]  # (C, rh, rw) each
            shapes = {(int(r.shape[-2]), int(r.shape[-1])) for r in per_sample}
            if len(shapes) > 1:
                raise ValueError(
                    f"MOSS reference {k} has different grids across the batch: {sorted(shapes)}."
                )
            rh, rw = int(per_sample[0].shape[-2]), int(per_sample[0].shape[-1])
            stacked = torch.stack([r.to(device=device, dtype=target.dtype) for r in per_sample], 0)
            packed.append(stacked.flatten(2).transpose(1, 2))  # (B, rh*rw, C)
            for i in range(bs):
                reference_grids[i].append((rh, rw))

    hidden_states = torch.cat(packed, dim=1)
    return {
        "hidden_states": hidden_states,
        "timestep": sigmas.to(device=device, dtype=torch.float32).reshape(-1) * TIME_SCALE,
        "encoder_hidden_states": text,
        "encoder_attention_mask": mask,
        "target_grids": [(height, width) for _ in range(bs)],
        "reference_grids": reference_grids,
    }

def unpack_prediction(adapter, model_pred) -> torch.Tensor:
    """``MOSSImageTransformer`` returns ``(B, 128, H, W)`` (diffusers output dataclass).

    The packed sequence is only the input side; the model already reshapes the
    prediction to the target grid, so the trainer gets the tensor it iterates.
    """
    sample = getattr(model_pred, "sample", None)
    if sample is not None:
        return sample
    if isinstance(model_pred, (tuple, list)) and model_pred:
        return model_pred[0]
    return model_pred
