# =============================================================================
# DERIVED FROM VENDORED SOURCE - upstream is the source of truth for the layout.
#
# Original author : Ostris, LLC ("ostris") - AI Toolkit
# Upstream repo   : https://github.com/ostris/ai-toolkit
# Upstream files  : extensions_built_in/diffusion_models/flux3/src/transformer.py
#                     (its module tree *is* the shipped model.safetensors layout;
#                      `convert_state_dict_on_load` splits the fused int8 qkv)
#                   toolkit/util/comfy_quant_import.py (split_fused_quantized_keys)
# Upstream commit : 12395a103fd0ca499c631dc6f460a8778e11490d (2026-10-15)
# Licence         : MIT - Copyright (c) 2024 Ostris, LLC (see ATTRIBUTION.md)
# Vendored        : 2026-10-16 into UnifiedTrainer
#
# Changes vs upstream: NEW code, not a copy. It writes down the state-dict key
#   mapping that is *implicit* in ai-toolkit's `Flux3Transformer` module tree
#   (whose names are the shipped checkpoint's names) against the diffusers
#   `Flux3Transformer2DModel` tree UTrainer vendors, adds the reverse direction
#   (which nothing upstream provides), and trims
#   `split_fused_quantized_keys` down to the plain-fusion case.
#
#   Cross-checked against diffusers' own converter,
#   `E:\diffusers\src\diffusers\loaders\single_file_utils.py:4017-4077`
#   (`convert_flux3_transformer_checkpoint_to_diffusers`), which is the
#   canonical mapping for this checkpoint - including the `swap_scale_shift` on
#   the output adaLN weight (lines 4046-4060) that a naive key rename silently
#   gets wrong.
#
# Why it is here : UTrainer's flux3_klein adapter can only `from_pretrained()` a
#   diffusers-format directory. A BFL vendor package (`model.safetensors` /
#   `ae.safetensors`) or BFL's single-file ComfyUI repack carries the *native*
#   names. This module is the audit-able, tested statement of that mapping, the
#   reverse direction needed to publish LoRAs in ComfyUI's key space, and the
#   layout detector that tells the two apart.
# =============================================================================
"""Native (BFL / ai-toolkit / ComfyUI) <-> diffusers state-dict keys for FLUX.3.

Key spaces, side by side (native left, UTrainer's vendored diffusers module
right)::

    emb_in.<mode>.weight                      x_embedders.<mode>.weight
    txt_in.weight                             context_embedder.weight
    time_in.in_layer.{weight,bias}            timestep_embedder.linear_1.*
    time_in.out_layer.{weight,bias}           timestep_embedder.linear_2.*
    ref_time_in.in_layer.*                    target_timestep_embedder.linear_1.*   (flow map only)
    ref_time_in.out_layer.*                   target_timestep_embedder.linear_2.*   (flow map only)
    early_stream_modulations.<m>.lin.*        early_stream_modulations.<m>.linear.*
    single_stream_modulations.<m>.lin.*       single_stream_modulations.<m>.linear.*
    content_mode_blocks.<mode>.<i>.*          mode_blocks.<mode>.<i>.*
    txt_mode_blocks.<i>.*                     txt_mode_blocks.<i>.*
    single_blocks.<i>.*                       single_transformer_blocks.<i>.*
    final_layer.image.linear.*                proj_out.*
    final_layer.image.adaLN_modulation.1.*    norm_out.linear.*      (** values swapped, see below **)
    final_layer.image.norm_final.*            (no parameters - non-affine LayerNorm)
    repa_projectors.*                         (training-only heads - dropped)

inside a per-mode / joint block::

    <blk>.pre_norm.*                          <blk>.norm.*           (non-affine)
    <blk>.q_proj.*                            <blk>.attn.to_q.*
    <blk>.k_proj.*                            <blk>.attn.to_k.*
    <blk>.v_proj.*                            <blk>.attn.to_v.*
    <blk>.norm.query_norm.scale               <blk>.attn.norm_q.weight
    <blk>.norm.key_norm.scale                 <blk>.attn.norm_k.weight
    <blk>.attn_out.*                          <blk>.attn.to_out.0.*
    <blk>.mlp_in.*                            <blk>.ff.linear_in.*
    <blk>.mlp_out.*                           <blk>.ff.linear_out.*
    <blk>.mlp_act.gate_fn                     (no parameters)

Two traps this module exists to close:

1. **The output adaLN weight is re-ordered, not just renamed.** The native
   checkpoint's ``final_layer.image.adaLN_modulation.1.weight`` produces
   ``[shift, scale]`` - that is how ComfyUI's modulated norm and ai-toolkit's
   ``LastLayer`` consume it (``x * (1 + scale) + shift`` with ``shift`` taken
   from the **first** half, ``ai-toolkit .../flux3/src/transformer.py:229-240``).
   diffusers' ``Flux3AdaLayerNorm`` expects ``[scale, shift]``, so the halves are
   swapped on conversion. A pure key rename loads the right numbers in the wrong
   order.
2. **The int8 ComfyUI repack fuses q/k/v** per block into one ``.qkv`` entry
   (rows q, k, v); the bf16 and nvfp4 files keep them split.

``content_mode_blocks`` == ``mode_blocks``, ``single_blocks`` ==
``single_transformer_blocks``.

`tests/test_flux3_aitoolkit_keymap.py` verifies the mapping against a real
(tiny) `Flux3Transformer2DModel`: it renames the module's own state dict into the
native spelling and asserts an exact round trip, so a drifted mapping fails the
test instead of silently mis-loading weights.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch

__all__ = [
    "UnsupportedFlux3CheckpointError",
    "NATIVE_TO_DIFFUSERS_GROUPS",
    "DIFFUSERS_TO_NATIVE_GROUPS",
    "NATIVE_TO_DIFFUSERS_SUFFIXES",
    "DIFFUSERS_TO_NATIVE_SUFFIXES",
    "ADALN_OUTPUT_NATIVE_KEY",
    "ADALN_OUTPUT_DIFFUSERS_KEY",
    "FLOW_MAP_KEY_MARKER",
    "detect_layout",
    "is_flow_map_state_dict",
    "swap_scale_shift",
    "native_key_to_diffusers",
    "diffusers_key_to_native",
    "convert_native_to_diffusers",
    "convert_diffusers_to_native",
    "split_fused_qkv",
    "fuse_qkv",
]


class UnsupportedFlux3CheckpointError(RuntimeError):
    """Raised for a FLUX.3 checkpoint UTrainer has no backend for."""


# ── Group-level prefixes (matched at the start of the key) ──────────────────
# Ordered: the first match wins.
NATIVE_TO_DIFFUSERS_GROUPS: Sequence[tuple[str, str]] = (
    ("ref_time_in.in_layer.", "target_timestep_embedder.linear_1."),
    ("ref_time_in.out_layer.", "target_timestep_embedder.linear_2."),
    ("time_in.in_layer.", "timestep_embedder.linear_1."),
    ("time_in.out_layer.", "timestep_embedder.linear_2."),
    ("emb_in.", "x_embedders."),
    ("txt_in.", "context_embedder."),
    ("content_mode_blocks.", "mode_blocks."),
    ("single_blocks.", "single_transformer_blocks."),
)

DIFFUSERS_TO_NATIVE_GROUPS: Sequence[tuple[str, str]] = tuple(
    (dst, src) for src, dst in NATIVE_TO_DIFFUSERS_GROUPS
)

# ── Block-internal suffixes (matched at the end of the key) ─────────────────
# The suffix may be followed by nothing (the QK-norm scales), `.weight` or
# `.bias`. `.lin` -> `.linear` covers the modulation heads (`Modulation.lin` in
# the native tree vs `Flux3Modulation.linear` in diffusers); `.pre_norm` ->
# `.norm` is the (non-affine, parameter-free) block LayerNorm.
NATIVE_TO_DIFFUSERS_SUFFIXES: Sequence[tuple[str, str]] = (
    ("norm.query_norm.scale", "attn.norm_q.weight"),
    ("norm.key_norm.scale", "attn.norm_k.weight"),
    ("q_proj", "attn.to_q"),
    ("k_proj", "attn.to_k"),
    ("v_proj", "attn.to_v"),
    ("attn_out", "attn.to_out.0"),
    ("mlp_in", "ff.linear_in"),
    ("mlp_out", "ff.linear_out"),
    (".lin", ".linear"),
    ("pre_norm", "norm"),
)

DIFFUSERS_TO_NATIVE_SUFFIXES: Sequence[tuple[str, str]] = tuple(
    (dst, src) for src, dst in NATIVE_TO_DIFFUSERS_SUFFIXES
)

# The output projection and the output adaptive-LayerNorm projection: the latter
# is the one tensor whose *values* are re-ordered on conversion.
_OUTPUT_PROJ_NATIVE_KEY = "final_layer.image.linear.weight"
_OUTPUT_PROJ_DIFFUSERS_KEY = "proj_out.weight"
ADALN_OUTPUT_NATIVE_KEY = "final_layer.image.adaLN_modulation.1.weight"
ADALN_OUTPUT_DIFFUSERS_KEY = "norm_out.linear.weight"

#: native keys with no diffusers counterpart that are dropped on conversion.
#: ``repa_projectors.*`` are training-only heads (diffusers' converter drops them
#: too, `single_file_utils.py:4052-4055`).
_DROP_PREFIXES: Sequence[str] = ("repa_projectors.",)
_DROP_SUBSTRINGS: Sequence[str] = (".input_scale", ".norm_final.")

#: markers that say the tensors are quantized in a ComfyUI storage UTrainer has
#: no backend for (UTrainer quantizes with torchao, not comfy_quant/convrot8).
_QUANT_MARKER = ".comfy_quant"

#: the native spelling of the flow-map (step-distilled) extra timestep embedder.
FLOW_MAP_KEY_MARKER = "ref_time_in."


def swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    """``[shift, scale] -> [scale, shift]`` along the output dimension.

    Same op as diffusers' ``loaders.single_file_utils.swap_scale_shift``
    (``single_file_utils.py:1990-1993``); it is its own inverse, so it is used in
    both directions.
    """
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def detect_layout(state_dict: Dict[str, torch.Tensor]) -> str:
    """Classify a FLUX.3 DiT state dict.

    ``"native"`` / ``"native-quantized"``
        shipped, ai-toolkit or ComfyUI spelling (``emb_in.*``); the
        ``-quantized`` variants carry ``*.comfy_quant`` markers (int8 convrot /
        nvfp4) that UTrainer cannot consume.
    ``"diffusers"`` / ``"diffusers-quantized"``
        already UTrainer's spelling (``x_embedders.*``).
    ``"unknown"``
        neither (a LoRA file, a partial dict, ...).
    """
    keys = list(state_dict)
    quantized = any(k.endswith(_QUANT_MARKER) for k in keys)
    if any(k.startswith(("emb_in.", "txt_in.", "content_mode_blocks.")) for k in keys):
        return "native-quantized" if quantized else "native"
    if any(
        k.startswith(("x_embedders.", "context_embedder.", "mode_blocks."))
        for k in keys
    ):
        return "diffusers-quantized" if quantized else "diffusers"
    return "unknown"


def is_flow_map_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Whether the raw state dict belongs to a step-distilled (flow-map) build.

    ai-toolkit keys the check off ``ref_time_in.*`` (``transformer.py:397-401``);
    diffusers renames that module to ``target_timestep_embedder``. Both spellings
    are accepted, so the sniff works before *and* after conversion.

    UTrainer currently trusts the loaded transformer's ``config.flow_map``. When
    the raw state dict is available (single-file loads, hand-made directories)
    this is the direct evidence: a mis-declared ``flow_map`` silently trains the
    distilled model with the base-model objective.
    """
    return any(
        k.startswith(FLOW_MAP_KEY_MARKER)
        or k.startswith("target_timestep_embedder.")
        for k in state_dict
    )


def _apply_groups(key: str, groups: Iterable[tuple[str, str]]) -> str:
    for old, new in groups:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def _replace_suffix(key: str, old: str, new: str) -> str:
    """Replace a trailing module path, keeping any ``.weight`` / ``.bias`` tail."""
    for param in ("", ".weight", ".bias"):
        if key.endswith(old + param):
            return key[: len(key) - len(old + param)] + new + param
    return key


def native_key_to_diffusers(key: str) -> str:
    """Rename one native parameter name to UTrainer's (diffusers) name.

    Returns ``""`` for keys that must be dropped (REPA heads, calibration
    extras, the non-affine output norm).
    """
    if key.startswith(_DROP_PREFIXES) or any(d in key for d in _DROP_SUBSTRINGS):
        return ""
    if key == ADALN_OUTPUT_NATIVE_KEY:
        return ADALN_OUTPUT_DIFFUSERS_KEY
    if key == _OUTPUT_PROJ_NATIVE_KEY:
        return _OUTPUT_PROJ_DIFFUSERS_KEY
    key = _apply_groups(key, NATIVE_TO_DIFFUSERS_GROUPS)
    for old, new in NATIVE_TO_DIFFUSERS_SUFFIXES:
        key = _replace_suffix(key, old, new)
    return key


def diffusers_key_to_native(key: str) -> str:
    """Inverse of :func:`native_key_to_diffusers` (for ComfyUI-style exports)."""
    if key == ADALN_OUTPUT_DIFFUSERS_KEY:
        return ADALN_OUTPUT_NATIVE_KEY
    if key == _OUTPUT_PROJ_DIFFUSERS_KEY:
        return _OUTPUT_PROJ_NATIVE_KEY
    key = _apply_groups(key, DIFFUSERS_TO_NATIVE_GROUPS)
    for old, new in DIFFUSERS_TO_NATIVE_SUFFIXES:
        key = _replace_suffix(key, old, new)
    return key


def split_fused_qkv(
    state_dict: Dict[str, torch.Tensor],
    keys: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Split fused ``<base>.qkv.weight`` rows into ``q_proj/k_proj/v_proj``.

    Trimmed rewrite of ai-toolkit's ``split_fused_quantized_keys``
    (``comfy_quant_import.py:75-135``) for the plain (unquantized) fusion the int8
    repack uses; the int8/nvfp4 scale tensors are deliberately not handled (see
    :func:`convert_native_to_diffusers`). Mutates and returns ``state_dict``.
    """
    todo: List[str] = (
        list(keys)
        if keys is not None
        else [k for k in state_dict if k.endswith(".qkv.weight")]
    )
    for weight_key in todo:
        if not weight_key.endswith(".qkv.weight") or weight_key not in state_dict:
            continue
        base = weight_key[: -len(".qkv.weight")]
        weight = state_dict.pop(weight_key)
        if weight.shape[0] % 3 != 0:
            raise ValueError(
                f"{weight_key}: fused out dim {weight.shape[0]} does not split into 3"
            )
        for suffix, part in zip(
            ("q_proj", "k_proj", "v_proj"), torch.chunk(weight, 3, dim=0)
        ):
            state_dict[f"{base}.{suffix}.weight"] = part.contiguous()
        bias_key = f"{base}.qkv.bias"
        if bias_key in state_dict:
            bias = state_dict.pop(bias_key)
            for suffix, part in zip(
                ("q_proj", "k_proj", "v_proj"), torch.chunk(bias, 3, dim=0)
            ):
                state_dict[f"{base}.{suffix}.bias"] = part.contiguous()
    return state_dict


def fuse_qkv(
    state_dict: Dict[str, torch.Tensor], bases: Sequence[str] | None = None
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`split_fused_qkv` (out-dim concat in q, k, v order)."""
    if bases is None:
        bases = sorted(
            {
                k[: -len(".q_proj.weight")]
                for k in state_dict
                if k.endswith(".q_proj.weight")
            }
        )
    for base in bases:
        keys = [f"{base}.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")]
        if not all(k in state_dict for k in keys):
            continue
        state_dict[f"{base}.qkv.weight"] = torch.cat(
            [state_dict.pop(k) for k in keys], dim=0
        ).contiguous()
    return state_dict


def convert_native_to_diffusers(
    state_dict: Dict[str, torch.Tensor],
    *,
    allow_quantized: bool = False,
    keep_unknown: bool = False,
) -> Dict[str, torch.Tensor]:
    """Rename a native FLUX.3 DiT state dict into UTrainer's diffusers spelling.

    Applies, in order: fused ``qkv`` split, key renames, and the
    ``[shift, scale] -> [scale, shift]`` swap on the output adaLN weight.

    Parameters
    ----------
    allow_quantized
        By default a dict carrying ``*.comfy_quant`` markers raises
        :class:`UnsupportedFlux3CheckpointError`: those tensors are stored
        rotated/int8 (or nvfp4) and must not be treated as plain weights.
        Pass ``True`` only to *inspect* such a dict.
    keep_unknown
        Keep keys the mapping does not recognise instead of raising. Off by
        default: a silently dropped weight is the failure mode this function
        exists to prevent.

    Raises
    ------
    UnsupportedFlux3CheckpointError
        Quantized comfy storage, or unrecognised keys with ``keep_unknown=False``.
    """
    if not allow_quantized and any(k.endswith(_QUANT_MARKER) for k in state_dict):
        raise UnsupportedFlux3CheckpointError(
            "this FLUX.3 checkpoint stores weights in ComfyUI's quantized format "
            "(*.comfy_quant: int8_convrot / nvfp4). UnifiedTrainer has no convrot8 "
            "or nvfp4-import backend; use the plain bf16 file "
            "(flux3_image_klein_bf16.safetensors) or a diffusers-format directory "
            "and let UnifiedTrainer quantize with torchao instead."
        )

    state_dict = split_fused_qkv(dict(state_dict))

    out: Dict[str, torch.Tensor] = {}
    unknown: List[str] = []
    for key, value in state_dict.items():
        if key.endswith(_QUANT_MARKER):
            continue
        new_key = native_key_to_diffusers(key)
        if not new_key:
            continue
        if new_key == key:
            unknown.append(key)
            if keep_unknown:
                out[key] = value
            continue
        if new_key == ADALN_OUTPUT_DIFFUSERS_KEY and value.ndim >= 1:
            value = swap_scale_shift(value)
        out[new_key] = value

    if unknown and not keep_unknown:
        raise UnsupportedFlux3CheckpointError(
            "unmapped FLUX.3 keys (refusing to drop weights silently): "
            + ", ".join(sorted(unknown)[:8])
            + (" ..." if len(unknown) > 8 else "")
        )
    return out


def convert_diffusers_to_native(
    state_dict: Dict[str, torch.Tensor],
    *,
    fuse: bool = False,
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`convert_native_to_diffusers`.

    Used to publish a UTrainer LoRA/checkpoint under ComfyUI's key space.
    ``fuse=True`` additionally re-fuses each block's q/k/v rows into a single
    ``.qkv`` entry.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key == ADALN_OUTPUT_DIFFUSERS_KEY and value.ndim >= 1:
            value = swap_scale_shift(value)
        out[diffusers_key_to_native(key)] = value
    if fuse:
        out = fuse_qkv(out)
    return out
