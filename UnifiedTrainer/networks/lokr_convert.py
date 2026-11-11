"""LoKr checkpoint key conversion: UnifiedTrainer <-> ComfyUI (FLUX.3).

Why this exists
---------------
`LokrNetwork.save_weights` writes keys as
``lycoris_<diffusers module path with dots -> underscores>``.  That is the
SimpleTuner/LyCORIS convention, and ComfyUI registers it in
``comfy.lora.model_lora_keys_unet`` for several architectures -- but **not for
FLUX.3**.  For FLUX.3 ComfyUI only registers::

    lora_unet_<comfy module path with dots -> underscores>
    diffusion_model.<comfy module path>

The ``lycoris_`` form is never added for Flux3 (Krea2 / SD3 / Flux / HiDream /
ACEStep each have a dedicated ``lycoris_`` branch; Flux3 does not).  So every
key of a UnifiedTrainer LoKr file lands in ComfyUI's "lora key not loaded"
warning -- even though shapes and the LoKr math are fully compatible.

Names also differ: ``comfy/ldm/flux3/model.py`` uses the original BFL names,
diffusers (and UnifiedTrainer) uses renamed ones.  The mapping here is the
inverse of diffusers' ``convert_flux3_transformer_checkpoint_to_diffusers``.

Math compatibility
------------------
ComfyUI's ``LoKrAdapter.calculate_weight`` applies ``torch.kron(w1, w2)`` and,
when direct ``lokr_w1``/``lokr_w2`` are present, forces ``alpha = 1.0`` (its
``dim`` stays None).  UnifiedTrainer's full-rank save writes direct
``lokr_w1``/``lokr_w2`` with scale 1.0.  So for full-rank checkpoints this is a
**pure key rename** -- no re-factorization, no numeric change -- with one
exception: the adaLN layer needs a row permutation (see `_adaLN_permute_w1`).

Why names are matched by canonical set, not by un-flattening
------------------------------------------------------------
Flattening ``mode_blocks.image_cond.0.attn.to_out.0`` to
``mode_blocks_image_cond_0_attn_to_out_0`` is lossy -- ``_`` means both a real
underscore and a path separator, so ``_``->``.`` cannot be inverted.
UnifiedTrainer's own loader sidesteps this by iterating its KNOWN layer names.
This module does the same: it generates the canonical FLUX.3 Linear module list
(or accepts an explicit one from a live ``LokrNetwork``) and matches flats
against it.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# in-block leaves: (diffusers suffix, comfy suffix)
_BLOCK_LEAVES = [
    ("attn.to_q", "q_proj"),
    ("attn.to_k", "k_proj"),
    ("attn.to_v", "v_proj"),
    ("attn.to_out.0", "attn_out"),
    ("ff.linear_in", "mlp_in"),
    ("ff.linear_out", "mlp_out"),
]

# top-level single-module renames: (diffusers dotted, comfy dotted)
_TOP_LEVEL = [
    ("x_embedders.image_cond", "emb_in.image_cond"),
    ("x_embedders.image", "emb_in.image"),
    ("context_embedder", "txt_in"),
    ("timestep_embedder.linear_1", "time_in.in_layer"),
    ("timestep_embedder.linear_2", "time_in.out_layer"),
    ("target_timestep_embedder.linear_1", "ref_time_in.in_layer"),
    ("target_timestep_embedder.linear_2", "ref_time_in.out_layer"),
    ("proj_out", "final_layer.image.linear"),
    # diffusers Flux3AdaLayerNorm.linear <- BFL adaLN_modulation.1 (scale/shift swapped)
    ("norm_out.linear", "final_layer.image.adaLN_modulation.1"),
]

# ModuleDict modulations: diffusers ".linear" vs comfy ".lin"
_MODULATION_MODES = ("image", "image_cond", "txt")

# the one layer whose diffusers weight is a row-swapped version of BFL's
_ADALN_DIFFUSERS = "norm_out.linear"
_ADALN_COMFY = "final_layer.image.adaLN_modulation.1"

_SUFFIXES = (".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
             ".lokr_t2", ".lokr_w1", ".lokr_w2", ".alpha")

# qwen-image-2.1 fuses these two into `img_mlp.gate_up`; ComfyUI only exposes
# them through its `lycoris_` aliases (see convert_rename_state_dict)
_QWEN_FUSED_MLP_HALVES = ("_img_mlp_gate_layer", "_img_mlp_proj")


def _flatten(name: str) -> str:
    return name.replace(".", "_")


def build_native_modules(
    num_layers: int = 5,
    num_single_layers: int = 28,
    flow_map: bool = False,
) -> list[str]:
    """Canonical diffusers-side Linear module paths for FLUX.3.

    These are exactly the modules ``lokr_target_modules: null`` (all nn.Linear)
    attaches to, minus the non-Linear ones.  Matches the 271 modules of the
    shipped klein checkpoints.
    """
    mods: list[str] = []

    for dif, _comfy in _TOP_LEVEL:
        if dif.startswith("target_timestep_embedder") and not flow_map:
            continue
        mods.append(dif)

    for mod in _MODULATION_MODES:
        mods.append(f"early_stream_modulations.{mod}.linear")
        mods.append(f"single_stream_modulations.{mod}.linear")

    def block(prefix: str) -> list[str]:
        return [f"{prefix}.{leaf}" for leaf, _ in _BLOCK_LEAVES]

    for i in range(num_layers):
        mods += block(f"mode_blocks.image.{i}")
        mods += block(f"mode_blocks.image_cond.{i}")
        mods += block(f"txt_mode_blocks.{i}")
    for i in range(num_single_layers):
        mods += block(f"single_transformer_blocks.{i}")

    return mods


def native_to_comfy(dif: str) -> Optional[str]:
    """diffusers dotted module -> ComfyUI dotted module (None if none)."""
    for a, b in _TOP_LEVEL:
        if dif == a:
            return b
    for mode in _MODULATION_MODES:
        if dif == f"early_stream_modulations.{mode}.linear":
            return f"early_stream_modulations.{mode}.lin"
        if dif == f"single_stream_modulations.{mode}.linear":
            return f"single_stream_modulations.{mode}.lin"

    for prefix, comfy_prefix in (
        ("mode_blocks.image_cond.", "content_mode_blocks.image_cond."),
        ("mode_blocks.image.", "content_mode_blocks.image."),
        ("txt_mode_blocks.", "txt_mode_blocks."),
        ("single_transformer_blocks.", "single_blocks."),
    ):
        if dif.startswith(prefix):
            rest = dif[len(prefix):]
            if "." not in rest:
                return None
            idx, leaf = rest.split(".", 1)
            for a, b in _BLOCK_LEAVES:
                if leaf == a:
                    return f"{comfy_prefix}{idx}.{b}"
            return None
    return None


def _adaLN_permute_w1(w1):
    """Swap the two halves of ``lokr_w1``'s OUTPUT rows (self-inverse).

    diffusers' ``swap_scale_shift(dim=0)`` turns BFL's ``[shift, scale]`` into
    diffusers' ``[scale, shift]`` on the adaLN output axis.  For ``kron(w1,w2)``
    the output row index is ``i_w1 * out_k + i_w2``, so swapping the halves of
    the output == swapping the halves of w1's rows (valid while out_l is even;
    factor 4 -> 4 rows).  Being an involution, the same helper converts both
    ways.
    """
    import torch

    if w1 is None or not torch.is_tensor(w1) or w1.dim() != 2:
        return w1
    out_l = w1.shape[0]
    if out_l % 2 != 0:
        logger.warning(
            "adaLN lokr_w1 has an odd row count (%d); leaving unpermuted. "
            "Re-factorize that layer to convert it correctly.", out_l,
        )
        return w1
    half = out_l // 2
    return torch.cat([w1[half:], w1[:half]], dim=0)


def _maps(native_modules: Optional[Iterable[str]], **kw):
    """(native_flat -> native_dotted, comfy_flat -> native_dotted)."""
    native = list(native_modules) if native_modules else build_native_modules(**kw)
    n2d = {_flatten(m): m for m in native}
    c2d: dict[str, str] = {}
    for m in native:
        comfy = native_to_comfy(m)
        if comfy is not None:
            c2d[_flatten(comfy)] = m
            c2d[comfy] = m          # diffusion_model.<dotted> form
    return n2d, c2d


def _split(key: str):
    for suf in _SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], suf
    return None, None


def build_moss_native_modules(num_layers: int = 42, num_text_blocks: int = 2) -> list[str]:
    """Canonical MOSS-Image Linear module paths (identical in diffusers and ComfyUI).

    ComfyUI's ``comfy/ldm/moss_image/model.py`` keeps the official MOSS names, so
    the conversion is a pure prefix rename.  269 modules = 17 top-level/text-block
    + 42 video blocks x 6 projections.  ComfyUI registers ``lora_unet_<flat>``
    for every one of them (generic loop for the plain ones, the MOSSImage branch
    for the per-projection keys when the checkpoint stores fused ``proj.qkv``).
    """
    mods = [
        "video_embedder",
        "text_embedder.proj",
        "shared_block_modulation",
        "video_head_modulation",
        "final_linear_video",
        "time_embedding.0",
        "time_embedding.2",
    ]
    for b in range(num_text_blocks):
        for leaf in ("proj", "qkv", "ffn_gate", "ffn_up", "ffn_down"):
            mods.append(f"text_embedder.blocks.{b}.{leaf}")
    for i in range(num_layers):
        for leaf in ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"):
            mods.append(f"layers.{i}.{leaf}")
    return mods


def convert_moss_state_dict(state: dict, to: str = "comfyui",
                            num_layers: int = 42) -> dict:
    """Rename MOSS LoKr keys between UnifiedTrainer-native and ComfyUI form.

    ``to="comfyui"``        ``lycoris_<flat>``  -> ``lora_unet_<flat>``
    ``to="unifiedtrainer"`` ``lora_unet_<flat>`` -> ``lycoris_<flat>``

    Pure prefix rename, because the module names are identical on both sides and
    ComfyUI's LoKR math matches ours for full-rank checkpoints (direct
    ``lokr_w1`` forces ``alpha = 1.0``, which is what we train with).
    """
    native = {_flatten(m) for m in build_moss_native_modules(num_layers=num_layers)}
    out: dict = {}
    unmapped: list[str] = []

    if to == "comfyui":
        for key, tensor in state.items():
            body, suf = _split(key)
            if body is None or not body.startswith("lycoris_"):
                unmapped.append(key)
                continue
            flat = body[len("lycoris_"):]
            if flat not in native:
                unmapped.append(key)
                continue
            out[f"lora_unet_{flat}{suf}"] = tensor
    elif to == "unifiedtrainer":
        for key, tensor in state.items():
            body, suf = _split(key)
            flat = body
            if flat.startswith("lora_unet_"):
                flat = flat[len("lora_unet_"):]
            elif flat.startswith("diffusion_model."):
                flat = flat[len("diffusion_model."):].replace(".", "_")
            if flat not in native:
                unmapped.append(key)
                continue
            out[f"lycoris_{flat}{suf}"] = tensor
    else:
        raise ValueError(f"unknown target {to!r}; expected 'comfyui' or 'unifiedtrainer'")

    if unmapped:
        logger.warning(
            "lokr_convert(moss): %d/%d key(s) had no counterpart and were dropped: %s%s",
            len(unmapped), len(state), unmapped[:4],
            " ..." if len(unmapped) > 4 else "",
        )
    return out


def convert_state_dict(
    state: dict,
    to: str = "comfyui",
    native_modules: Optional[Iterable[str]] = None,
    num_layers: int = 5,
    num_single_layers: int = 28,
    flow_map: bool = False,
) -> dict:
    """Rename LoKr keys between UnifiedTrainer-native and ComfyUI/FLUX.3 form.

    ``to="comfyui"``        native -> ComfyUI (``lora_unet_*``, BFL names)
    ``to="unifiedtrainer"`` ComfyUI -> native (``lycoris_*``, diffusers names)

    Only the adaLN ``lokr_w1`` is transformed; all other tensors are reused.
    """
    import torch

    n2d, c2d = _maps(native_modules, num_layers=num_layers,
                     num_single_layers=num_single_layers, flow_map=flow_map)
    out: dict = {}
    unmapped: list[str] = []

    if to == "comfyui":
        for key, tensor in state.items():
            body, suf = _split(key)
            if body is None or not body.startswith("lycoris_"):
                unmapped.append(key)
                continue
            native = n2d.get(body[len("lycoris_"):])
            if native is None:
                unmapped.append(key)
                continue
            if native == _ADALN_DIFFUSERS and suf == ".lokr_w1":
                tensor = _adaLN_permute_w1(tensor)
            comfy = native_to_comfy(native)
            if comfy is None:  # unknown flux module: never crash on None.flatten
                unmapped.append(key)
                continue
            out[f"lora_unet_{_flatten(comfy)}{suf}"] = tensor

    elif to == "unifiedtrainer":
        for key, tensor in state.items():
            body, suf = _split(key)
            if body is None:
                unmapped.append(key)
                continue
            flat = body
            for pfx in ("lora_unet_", "diffusion_model."):
                if flat.startswith(pfx):
                    flat = flat[len(pfx):]
                    break
            native = c2d.get(flat)
            if native is None:
                unmapped.append(key)
                continue
            if native == _ADALN_DIFFUSERS and suf == ".lokr_w1":
                tensor = _adaLN_permute_w1(tensor)
            out[f"lycoris_{_flatten(native)}{suf}"] = tensor
    else:
        raise ValueError(f"unknown target {to!r}; expected 'comfyui' or 'unifiedtrainer'")

    if unmapped:
        logger.warning(
            "lokr_convert: %d/%d key(s) had no counterpart and were dropped: %s%s",
            len(unmapped), len(state), unmapped[:4],
            " ..." if len(unmapped) > 4 else "",
        )
    del torch
    return out


def convert_rename_state_dict(state: dict, to: str = "comfyui") -> dict:
    """Pure prefix rename: ``lycoris_<path><suf>`` <-> ``lora_unet_<path><suf>``.

    Used for architectures whose diffusers module names are already what ComfyUI
    registers (qwen-image-2.x and friends). The FLUX/Krea name mapping in
    :func:`convert_state_dict` only knows its own module list, so feeding it a
    model with 32 ``transformer_blocks.*`` drops almost every key (690/696 in the
    qwen21 baseline) and writes a ~500 KB "ComfyUI" file instead of the real one.

    Qwen-Image-2.1 exception
    ------------------------
    ``comfy/ldm/qwen_image21/model.py`` fuses the gated MLP into one
    ``img_mlp.gate_up`` Linear (2*hidden rows), while the trainer keeps
    ``img_mlp.gate_layer`` and ``img_mlp.proj`` as separate modules.  ComfyUI
    handles that mismatch only through the ``lycoris_`` aliases registered in
    ``comfy/lora.py`` (``model_lora_keys_unet``: ``gate_up`` -> rows
    ``[0:h]`` for ``.gate_layer`` and ``[h:2h]`` for ``.proj``).  It does **not**
    register a ``lora_unet_*`` alias for those two names, so those keys would be
    silently skipped ("lora key not loaded: ...lokr_w2") -- 64 modules / 192 keys
    / 27.6% of a qwen21 file.  Keep exactly those two module kinds on the
    ``lycoris_`` form; every other key keeps the ``lora_unet_`` form.
    """
    out: dict = {}
    dropped = 0
    if to == "comfyui":
        for key, tensor in state.items():
            body, suf = _split(key)
            if body is None or not body.startswith("lycoris_"):
                dropped += 1
                continue
            flat = _flatten(body[len("lycoris_"):])
            if flat.endswith(_QWEN_FUSED_MLP_HALVES):
                # let ComfyUI slice the fused gate_up layer (see docstring)
                out[f"lycoris_{flat}{suf}"] = tensor
            else:
                out[f"lora_unet_{flat}{suf}"] = tensor
    else:
        for key, tensor in state.items():
            body, suf = _split(key)
            if body is None:
                dropped += 1
                continue
            flat = body
            for pre in ("lycoris_", "lora_unet_"):
                if flat.startswith(pre):
                    flat = flat[len(pre):]
                    break
            out[f"lycoris_{flat}{suf}"] = tensor
    if dropped:
        logger.info("lokr_convert(rename): %d key(s) skipped (unknown prefix)", dropped)
    return out


def convert_file(src, dst, to: str = "comfyui", metadata: Optional[dict] = None,
                 model: str = "flux3", **kw) -> dict:
    """Convert a LoKr safetensors file; returns a summary dict.

    ``model="flux3"`` uses the FLUX.3/Krea2 name mapping (``native_modules`` /
    ``num_layers`` / ``num_single_layers`` / ``flow_map``); ``model="moss"`` uses
    the MOSS-Image mapping (``num_layers`` = 42 video blocks); ``model="rename"``
    is a pure ``lycoris_`` -> ``lora_unet_`` rename for architectures whose module
    names ComfyUI already understands (qwen-image-2.x).
    """
    from safetensors.torch import load_file, save_file

    src, dst = Path(src), Path(dst)
    state = load_file(str(src))
    if model == "moss":
        converted = convert_moss_state_dict(
            state, to=to, num_layers=kw.pop("num_layers", 42))
    elif model == "rename":
        converted = convert_rename_state_dict(state, to=to)
    else:
        converted = convert_state_dict(state, to=to, **kw)

    meta = {"source": "UnifiedTrainer", "network": "lokr", "converted_to": to}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    save_file(converted, str(dst), metadata=meta)

    summary = {"src": str(src), "dst": str(dst), "to": to,
               "in_keys": len(state), "out_keys": len(converted)}
    logger.info("lokr_convert: %s -> %s (%s): %d -> %d keys",
                src.name, dst.name, to, len(state), len(converted))
    return summary


def comfy_filename_for(path) -> Path:
    """``foo_epoch4.safetensors`` -> ``foo_epoch4_comfyui.safetensors``."""
    p = Path(path)
    return p.with_name(f"{p.stem}_comfyui{p.suffix}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert LoKr keys UTrainer <-> ComfyUI")
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--to", default="comfyui", choices=["comfyui", "unifiedtrainer"])
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--num-single-layers", type=int, default=28)
    ap.add_argument("--flow-map", action="store_true")
    ap.add_argument("--model", default="flux3", choices=["flux3", "moss"])
    args = ap.parse_args(argv)

    dst = args.dst
    if dst is None:
        p = Path(args.src)
        dst = str(comfy_filename_for(p) if args.to == "comfyui"
                  else p.with_name(f"{p.stem}_native{p.suffix}"))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = convert_file(args.src, dst, to=args.to, model=args.model,
                           num_layers=args.num_layers,
                           num_single_layers=args.num_single_layers,
                           flow_map=args.flow_map)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
