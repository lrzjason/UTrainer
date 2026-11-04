# =============================================================================
# DERIVED CODE - training forward for the official MOSS-Image transformer.
#
# Original author : OpenMOSS Team - openmoss/MOSS-Image-Base (ModelScope)
# Adapted by      : UnifiedTrainer, 2026-10-20
#
# `training_forward` is the vendored `MOSSImageTransformer.forward` with the
# additions this project needs, following the reference-token design of the
# Krea-2 adapter (`models/krea2/__init__.py`) and the frozen spec in
# `E:\MOSS_data_pipeline\idea.txt`:
#
#   1. **token-level input.** `hidden_states` may be the official 4D spatial
#      latent `[B, C, H, W]`, or a **pre-concatenated token sequence**
#      `[B, N, C]` = `[target | ref1 | ref2 | ...]` built by the adapter. In both
#      cases each span is embedded with the model's own `video_embedder`; the
#      token form exists so the reference concatenation happens *before* the
#      model, like Krea-2, and so the layout can carry per-span shapes.
#   2. **reference spans.** References are clean images appended after the text
#      tokens with their own RoPE frame (`r`): target r = 0, annotated layout
#      r = 1, full-size reference r = 2, object references r = 3+. They take the
#      t = 0 modulation; only the target tokens carry the training sigma.
#   3. **quantization-safe casts.** The official forward casts every activation
#      with `some_linear.weight.dtype`, which breaks under bnb nf4/int8 (uint8
#      packed weights) and torchao fp8. `_lin` casts to the module's compute
#      dtype instead, so quantized training works like on the other adapters.
#
# With float weights, no references and a 4D input the math is identical to the
# vendored forward (pinned by `tests/test_moss_forward.py`).
# =============================================================================
"""Training forward for ``MOSSImageTransformer``: token-level refs + quantization."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from UnifiedTrainer.vendor.openmoss_moss_image.pipeline_moss_image import (
    Transformer2DModelOutput,
    _attention,
    _situ,
    _validate_skip_layers,
)

__all__ = [
    "conditioning_like",
    "grid_rope",
    "install_reference_support",
    "training_forward",
]

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)


def _lin(module, x: torch.Tensor) -> torch.Tensor:
    """Call a linear with an activation dtype its weights can actually consume."""
    dtype = getattr(module, "compute_dtype", None)
    if dtype is None:
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dtype in _FLOAT_DTYPES:
            dtype = weight.dtype
    if dtype is not None and x.dtype != dtype and torch.is_floating_point(x):
        x = x.to(dtype)
    return module(x)


# ── RoPE ────────────────────────────────────────────────────────────────


def grid_rope(
    rope_module,
    ref_height: int,
    ref_width: int,
    frame: int,
    f_value: int,
    device,
    canvas: tuple[int, int] | None = None,
) -> torch.Tensor:
    """RoPE rows for one reference image, laid over the whole canvas grid.

    MOSS's RoPE is four-axis ``(r, f, h, w)``; the official model keeps ``r = 0``
    for every token, i.e. the first axis is *reserved* for a reference index.  A
    reference frame takes ``r = frame`` - the target is 0 and references are
    1, 2, 3, ... in order, exactly Krea-2's `prepare_position_ids`, ``f = f_value``
    (the same slot the target image tokens use, ``text_length``), and its own
    ``h``/``w`` grid scaled over the **whole canvas grid** - there is no per-item
    bbox mapping.  ``round`` is intentional: several reference tokens may land on
    one canvas cell, ``r`` keeps the sources apart.

    Returns ``(1, ref_height*ref_width, head_dim)`` in float32.
    """
    bands = rope_module.bands
    if bands.dtype != torch.float32:
        bands = bands.float()
    canvas_h, canvas_w = canvas if canvas is not None else (ref_height, ref_width)
    rows = torch.arange(ref_height, device=device, dtype=torch.float32)
    cols = torch.arange(ref_width, device=device, dtype=torch.float32)
    if ref_height > 1:
        rows = torch.round(rows * (canvas_h - 1) / (ref_height - 1))
    if ref_width > 1:
        cols = torch.round(cols * (canvas_w - 1) / (ref_width - 1))
    h, w = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack(
        (
            torch.full_like(h, float(frame)),
            torch.full_like(h, float(f_value)),
            h - (canvas_h - 1) / 2,
            w - (canvas_w - 1) / 2,
        ),
        -1,
    )
    coords = coords.flatten(0, 1).float()
    phase = coords.unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


# ── two-span modulation helpers ─────────────────────────────────────────


def _modulate(x, shift, scale, ref_shift, ref_scale, split):
    if ref_shift is None or split >= x.shape[1]:
        return x * (1 + scale[:, None]) + shift[:, None]
    return torch.cat(
        (
            x[:, :split] * (1 + scale[:, None]) + shift[:, None],
            x[:, split:] * (1 + ref_scale[:, None]) + ref_shift[:, None],
        ),
        1,
    )


def _gate(x, gate, ref_gate, split):
    if ref_gate is None or split >= x.shape[1]:
        return x * gate[:, None]
    return torch.cat((x[:, :split] * gate[:, None], x[:, split:] * ref_gate[:, None]), 1)


def _block_forward(block, x, rope, mod_main, mod_ref=None, split=0):
    """``_Block.forward`` with quantization-safe casts and two modulation spans."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod_main.unbind(1)
    if mod_ref is not None:
        r_shift_a, r_scale_a, r_gate_a, r_shift_f, r_scale_f, r_gate_f = mod_ref.unbind(1)
    else:
        r_shift_a = r_scale_a = r_gate_a = r_shift_f = r_scale_f = r_gate_f = None

    heads = block.heads
    y = _modulate(block.norm1(x), shift_a, scale_a, r_shift_a, r_scale_a, split)
    q = block.q_norm(_lin(block.proj["q"], y).unflatten(-1, (heads, -1))).flatten(2)
    k = block.k_norm(_lin(block.proj["k"], y).unflatten(-1, (heads, -1))).flatten(2)
    v = _lin(block.proj["v"], y)
    x = x + _gate(_lin(block.proj["o"], _attention(q, k, v, rope, heads)), gate_a, r_gate_a, split)

    y = _modulate(block.norm2(x), shift_f, scale_f, r_shift_f, r_scale_f, split)
    hidden, gate = _lin(block.proj["ffn_up_gate"], y).chunk(2, -1)
    return x + _gate(_lin(block.proj["ffn_down"], _situ(hidden, gate)), gate_f, r_gate_f, split)


# ── conditioning ────────────────────────────────────────────────────────


def _conditioning(self, sigma: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mod [B,6,D], head [B,2,D])`` for sigma in [0, 1000] - official math."""
    times = torch.as_tensor(sigma, device=next(self.parameters()).device, dtype=torch.float32)
    times = times.reshape(-1).expand(batch)
    half = self.config.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    te0, te2 = self.time_embedding[0], self.time_embedding[2]
    time = _lin(te2, F.silu(_lin(te0, embedding))).float()
    mod = _lin(self.shared_block_modulation, time).view(batch, 6, -1)
    head = _lin(self.video_head_modulation, time).view(batch, 2, -1)
    return mod, head


def conditioning_like(self, sigma: torch.Tensor, batch: int = 1, index: int | None = None):
    """Modulation ``(1, 6, D)`` for the given sigma (used for the t = 0 span)."""
    mod, _ = _conditioning(self, sigma, batch)
    return mod[index if index is not None else 0 : (index or 0) + 1]


# ── layout helpers ─────────────────────────────────────────────────────


def _split_tokens(tokens: torch.Tensor, spans: list[tuple[int, tuple[int, int], int]]):
    """Split one sample's token axis into ``(target, [(ref, (h, w), frame)])``.

    ``spans`` is ``[(n_tokens, (h, w), frame), ...]`` for the references, in
    order; the first ``spans[0].n`` of the trailing rows belong to ref 0, and so
    on.  The target is whatever precedes them.
    """
    n_ref = sum(n for n, _, _ in spans)
    n_target = tokens.shape[0] - n_ref
    if n_target <= 0:
        raise ValueError("MOSS: the token layout leaves no room for the target tokens.")
    target = tokens[:n_target]
    refs = []
    offset = n_target
    for n, hw, frame in spans:
        refs.append((tokens[offset : offset + n], hw, frame))
        offset += n
    return target, refs


# ── the forward ─────────────────────────────────────────────────────────


def training_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask=None,
    target_grids=None,
    reference_grids=None,
    reference_latents=None,
    return_dict=True,
    skip_layers=None,
):
    """Krea-2-style packed forward with quantization-safe casts.

    The adapter does the concatenation, exactly like Krea-2's
    ``prepare_model_input``::

        hidden_states   (B, N, C)   = [ target (h*w) | ref1 (rh*rw) | ref2 ... ]
        target_grids    [(h, w)]    one per sample
        reference_grids [[(rh, rw), ...]]  one list per sample, in sequence order

    Each span is embedded with the model's own ``video_embedder``; the joint
    sequence becomes ``[target | text | refs…]``; the target and text carry the
    training sigma modulation while the reference tail takes ``t = 0`` (Krea-2's
    ``split = text_len + image_len`` rule).  Reference RoPE frames are ``1..N``
    (target frame 0), the same numbering as Krea-2's ``prepare_position_ids``.

    The official 4D entry point (``[B, C, H, W]``) still works - the vendored
    pipeline calls it - and then references may be passed as
    ``reference_latents`` (tensors), which is convenient for direct model use.
    """
    skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
    if hidden_states.ndim not in (3, 4):
        raise ValueError("hidden_states must be [B, C, H, W] (official) or [B, N, C] (packed).")
    channels = self.config.latent_channels
    if hidden_states.ndim == 3 and hidden_states.shape[-1] != channels:
        raise ValueError(
            f"packed hidden_states must be [B, N, {channels}], got {tuple(hidden_states.shape)}."
        )
    if hidden_states.ndim == 4 and hidden_states.shape[1] != channels:
        raise ValueError("hidden_states must be [B, latent_channels, H, W].")
    batch = hidden_states.shape[0]
    if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
        raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
    mask = encoder_attention_mask
    if mask is None:
        mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    if mask.shape != encoder_hidden_states.shape[:2]:
        raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")

    mod, head = _conditioning(self, torch.as_tensor(timestep).reshape(-1), batch)
    mod_ref_zero = self.conditioning_like(torch.zeros(1, device=hidden_states.device), 1, 0)

    outputs = []
    # One sample at a time, exactly like the official forward: the context is
    # trimmed to its true text length (padding tokens never enter the sequence)
    # because `_attention` runs SDPA without a key mask, and the RoPE image slot
    # is `f = text_length`, which differs per sample.
    for index in range(batch):
        context = encoder_hidden_states[index : index + 1, mask[index].bool()]
        if context.shape[1] == 0:
            raise ValueError("Each sample needs at least one valid text token.")

        if hidden_states.ndim == 4:
            height, width = hidden_states.shape[-2:]
            packed = hidden_states[index].flatten(1).transpose(0, 1)
            grids = _ref_grids_from_tensors(reference_latents, index)
        else:
            if target_grids is None:
                raise ValueError(
                    "packed hidden_states needs target_grids=[(h, w), ...] (one per sample) "
                    "and reference_grids=[[(rh, rw), ...], ...]."
                )
            height, width = int(target_grids[index][0]), int(target_grids[index][1])
            packed = hidden_states[index]
            per_sample = (reference_grids or [])[index] if reference_grids is not None else []
            grids = [(int(rh), int(rw)) for rh, rw in per_sample]

        n_target = height * width
        n_ref = sum(rh * rw for rh, rw in grids)
        if packed.shape[0] != n_target + n_ref:
            raise ValueError(
                f"MOSS: packed sequence has {packed.shape[0]} tokens but the layout needs "
                f"{n_target} target + {n_ref} reference tokens."
            )
        target_tokens = packed[:n_target]
        ref_slices = []
        offset = n_target
        for rh, rw in grids:
            ref_slices.append(packed[offset : offset + rh * rw])
            offset += rh * rw

        video = _lin(self.video_embedder, target_tokens[None].contiguous())
        rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
        text = _lin(self.text_embedder.proj, context)
        for text_block in self.text_embedder.blocks:
            text = text_block(text, rope[:, height * width :])
        x = torch.cat((video, text), 1)

        mod_ref = None
        split = x.shape[1]
        if ref_slices:
            tokens = [x]
            ropes = [rope]
            # Krea-2 numbering: reference grids are frames 1, 2, ... (target is 0).
            for k, ((rh, rw), ref_tokens) in enumerate(zip(grids, ref_slices)):
                tokens.append(
                    _lin(self.video_embedder, ref_tokens[None].to(video.dtype).contiguous())
                )
                ropes.append(
                    grid_rope(
                        self.rope, rh, rw, frame=1 + k, f_value=context.shape[1],
                        device=hidden_states.device, canvas=(height, width),
                    ).to(video.dtype)
                )
            x = torch.cat(tokens, 1)
            rope = torch.cat(ropes, 1)
            mod_ref = mod_ref_zero  # references are clean images: t = 0

        for layer_index, layer in enumerate(self.layers):
            if layer_index not in skip_layers:
                x = layer(x, rope, mod[index : index + 1], mod_ref, split)

        img = x[:, : height * width].to(torch.float32)
        img = self.final_norm_video(img)
        img = img * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
        img = _lin(self.final_linear_video, img)
        outputs.append(img.transpose(1, 2).reshape(1, -1, height, width))

    sample = torch.cat(outputs)
    return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def _ref_grids_from_tensors(reference_latents, index):
    """``[(h, w), ...]`` of the references of one sample (official 4D entry point)."""
    if not reference_latents:
        return []
    entry = reference_latents[index] if index < len(reference_latents) else None
    if entry is None:
        return []
    if torch.is_tensor(entry):
        entry = [entry]
    grids = []
    for ref in entry:
        if ref is None:
            continue
        ref = ref[0] if ref.ndim == 4 else ref
        grids.append((int(ref.shape[-2]), int(ref.shape[-1])))
    return grids


def _make_block_entry(block, model):
    """Block entry point: our forward signature + optional gradient checkpointing."""

    @functools.wraps(block.forward)
    def forward(x, rope, mod, mod_ref=None, split=0):
        if getattr(model, "_moss_checkpoint_blocks", False) and torch.is_grad_enabled() and model.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(_block_forward, block, x, rope, mod, mod_ref, split,
                              use_reentrant=False)
        return _block_forward(block, x, rope, mod, mod_ref, split)

    return forward


def install_reference_support(model) -> None:
    """Bind the training forward and the block entry points (idempotent)."""
    import types

    if getattr(model, "_moss_training_forward", False):
        return
    model._moss_checkpoint_blocks = False
    for block in getattr(model, "layers", []):
        block.forward = _make_block_entry(block, model)
    model.conditioning_like = types.MethodType(conditioning_like, model)
    model.forward = types.MethodType(training_forward, model)
    model._moss_training_forward = True
