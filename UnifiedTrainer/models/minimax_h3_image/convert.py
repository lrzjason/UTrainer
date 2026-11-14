"""Original MiniMaxAI/MiniMax-H3-Image checkpoint -> UTrainer-vendored classes.

The released ``transformer/`` + ``pdd_fused_heads.safetensors`` + ``vae/`` checkpoint uses
MiniMax's own module spelling; the vendored transformer/VAE in this package use the
diffusers names they were ported from. This module performs the key mapping in memory so
training loads the released files directly -- no external conversion step, no diffusers
model classes.

Ported from ``diffusers/scripts/convert_minimax_h3_to_diffusers.py`` (transformer key plan,
PDD head split, image-VAE key plan), specialized to the image partition (no audio stream).

The **transformer** conversion is key-only. The vendored blocks keep the release's module
layout (one fused ``attn.qkv_proj``, one ``attn.out_proj``, ``mlp.fc1`` holding
``[gate; value]``), so no tensor is re-ordered and a LoKR factorised on the training side is
the checkpoint's keys verbatim. The **image VAE** is the exception: it keeps the diffusers
layout (``attn.to_q``/``to_k``/``to_v``, ``ff.net.0.proj``), so its conversion still splits the
interleaved ``attn.to_qkv`` and swaps the ``ff.w1`` halves -- the VAE is inference-only and
never carries LoKR weights.
"""
from __future__ import annotations

import json
import os
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch/safetensors are only needed when tensors are read
    import torch

# Original checkpoints keep these in float32; everything else is bfloat16.
FP32_SOURCE_PREFIXES = ("video_patch_proj.", "audio_patch_proj.", "time_embedder.", "final_layer.video_out.")
DROPPED_TRANSFORMER_KEYS = ("rope.inv_freq",)
DROPPED_VAE_KEYS = ("decoder.mask_token",)

# ``AutoencoderKLMiniMaxH3Image`` constructor arguments (baked into the release).
IMAGE_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 24,
    "block_out_channels": [128, 256, 256, 512, 512, 1024],
    "layers_per_block": 2,
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect",
    "decoder_num_layers": 24,
    "decoder_num_attention_heads": 16,
    "decoder_attention_head_dim": 64,
    "decoder_num_register_tokens": 4,
    "decoder_ffn_mult": 4,
    "decoder_rope_theta": 100.0,
    "decoder_rope_dim_ratio": 0.5,
    "decoder_norm_eps": 1e-05,
}

# Transformer config keys of the release that the vendored class spells differently.
_TRANSFORMER_CONFIG_RENAME = {
    "latents_dim": "in_channels",
    "token_refiner_num_layers": "num_refiner_layers",
    "ffn_hidden_size": "ffn_dim",
    "timestep_input_dim": "freq_dim",
    "time_embed_hidden_size": "time_embed_hidden_dim",
    "rope_inv_freq_len": "rope_freq_dim",
}


def checkpoint_root(path: str) -> str:
    """Return the checkpoint root from either the root itself or a component subfolder.

    The config carries ``transformer_path`` / ``vae_path`` pointing at the subfolders while
    the converters expect the root (they append ``transformer/`` / ``vae/``).
    """
    if os.path.isdir(os.path.join(path, "transformer")) or os.path.isdir(os.path.join(path, "vae")):
        return path
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.isdir(os.path.join(parent, "transformer")) or os.path.isdir(os.path.join(parent, "vae")):
        return parent
    return path


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_pdd_config(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, "pdd_config.json")
    if not os.path.isfile(path):
        return {}
    return _read_json(path)


def transformer_config(checkpoint_dir: str) -> dict:
    """Build the vendored transformer's constructor kwargs from the release's config."""
    src = _read_json(os.path.join(checkpoint_dir, "transformer", "config.json"))
    pdd = read_pdd_config(checkpoint_dir)
    cfg: dict[str, Any] = {}
    for key, value in src.items():
        if key.startswith("_") or key in ("audio_latents_dim", "audio_enabled", "adaln_out_features",
                                          "final_adaln_out_features"):
            continue
        cfg[_TRANSFORMER_CONFIG_RENAME.get(key, key)] = value
    cfg["num_pdd_heads"] = int(pdd.get("num_inference_steps", 6))
    return cfg


def vae_config(checkpoint_dir: str) -> dict:
    src = _read_json(os.path.join(checkpoint_dir, "vae", "config.json"))
    cfg = dict(IMAGE_VAE_CONFIG)
    decoder = src.get("vit_decoder_kwargs") or {}
    # The decoder geometry is checkpoint-specific (the video VAE next to it uses a
    # different depth/width), so it always comes from the release's own config.
    cfg["decoder_num_layers"] = int(decoder.get("num_layers", cfg["decoder_num_layers"]))
    cfg["decoder_num_attention_heads"] = int(decoder.get("heads", cfg["decoder_num_attention_heads"]))
    cfg["decoder_attention_head_dim"] = int(decoder.get("dim_head", cfg["decoder_attention_head_dim"]))
    cfg["decoder_rope_theta"] = float(decoder.get("rope_theta", cfg["decoder_rope_theta"]))
    cfg["decoder_rope_dim_ratio"] = float(decoder.get("rope_dim_ratio", cfg["decoder_rope_dim_ratio"]))
    cfg["latents_mean"] = list(src["latents_mean"])
    cfg["latents_std"] = list(src["latents_std"])
    return cfg


# ── tensor transforms ────────────────────────────────────────────────────


def reorder_interleaved_qkv(weight: "torch.Tensor", heads: int, head_dim: int) -> "torch.Tensor":
    """``[head0: q k v, head1: q k v, ...]`` -> ``[q_all; k_all; v_all]``."""
    import torch

    expected = heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(f"fused qkv has {weight.shape[0]} rows, expected {expected}")
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0)


def _split_qkv(weight: "torch.Tensor", heads: int, head_dim: int):
    import torch

    inner = heads * head_dim
    if weight.shape[0] != 3 * inner:
        raise ValueError(f"fused qkv has {weight.shape[0]} rows, expected {3 * inner}")
    q, k, v = weight.split(inner, dim=0)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _convert_vae_key(source_key: str) -> str:
    target = source_key
    if target.startswith("encoder.down."):
        level, rest = target.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target = f"encoder.down_blocks.{level}.{rest}"
    target = target.replace("decoder.x_embedder.", "decoder.proj_in.")
    target = target.replace(".attn.to_out.", ".attn.to_out.0.")
    target = target.replace(".ff.w1.", ".ff.net.0.proj.")
    target = target.replace(".ff.w2.", ".ff.net.2.")
    return target


def _load_safetensors_dir(path: str) -> dict:
    """Read every ``*.safetensors`` shard in a directory into one state dict."""
    from safetensors import safe_open

    state: dict = {}
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    for name in files:
        with safe_open(os.path.join(path, name), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                state[key] = fh.get_tensor(key)
    return state


def _rename_transformer_key(key: str) -> str:
    """Original transformer key -> vendored module path (no tensor transform).

    The block leaves (``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``, ``mlp.fc2``) keep the
    checkpoint's own spelling because the vendored blocks use it too (see the module docstring
    of ``transformer_minimaxh3image.py``); only the stack prefix and the top-level modules are
    renamed. That is what makes the LoKR export a pure rename.
    """
    target = key
    if target.startswith("token_refiner.blocks."):
        target = target.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target.startswith("blocks."):
        target = target.replace("blocks.", "transformer_blocks.", 1)
    target = target.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    target = target.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    target = target.replace("video_patch_proj.", "proj_in.")
    target = target.replace("condition_proj.", "context_embedder.")
    target = target.replace("final_layer.norm.", "norm_out.norm.")
    target = target.replace("final_layer.video_out.", "proj_out.")
    target = target.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    target = target.replace(".attn.q_norm.", ".attn.norm_q.")
    target = target.replace(".attn.k_norm.", ".attn.norm_k.")
    return target


def map_transformer_keys(source_keys, cfg: dict) -> dict[str, list[str]]:
    """``{source_key: [target_key, ...]}`` -- pure string work (no tensors, no torch).

    Dropped keys map to an empty list: ``rope.inv_freq`` (recomputed as a buffer). Every other
    key maps one-to-one: the fused ``attn.qkv_proj`` and the ``[gate; value]`` ``mlp.fc1`` are
    kept as shipped, no re-ordering. The base output head ``final_layer.video_out.*`` is kept as
    ``proj_out`` -- it is the plain flow-matching head, used when ``pdd_step_index`` is None.
    """
    out: dict[str, list[str]] = {}
    for key in source_keys:
        if key in DROPPED_TRANSFORMER_KEYS:
            out[key] = []
            continue
        out[key] = [_rename_transformer_key(key)]
    return out


def map_vae_keys(source_keys, cfg: dict) -> dict[str, list[str]]:
    """``{source_key: [target_key, ...]}`` for the image VAE (string work only)."""
    out: dict[str, list[str]] = {}
    for key in source_keys:
        if key in DROPPED_VAE_KEYS:
            out[key] = []
            continue
        if ".attn.to_qkv." in key:
            prefix, suffix = key.split(".attn.to_qkv.")
            out[key] = [f"{prefix}.attn.to_q.{suffix}", f"{prefix}.attn.to_k.{suffix}",
                        f"{prefix}.attn.to_v.{suffix}"]
            continue
        out[key] = [_convert_vae_key(key)]
    return out


def iter_transformer_state_dicts(checkpoint_dir: str, cfg: dict | None = None):
    """Yield the converted state dict of ONE shard at a time, then the PDD heads.

    ``load_transformer_state_dict`` materializes all 62 GB at once, which does not fit next
    to a 24 GB model on a 90 GB box. Streaming keeps the peak at model + one shard (~6 GB).

    The conversion is key-only (``map_transformer_keys``): no tensor is re-ordered, so each
    shard's tensors are handed over as they are in the release.
    """
    from safetensors import safe_open

    cfg = cfg or transformer_config(checkpoint_dir)
    directory = os.path.join(checkpoint_dir, "transformer")

    for name in sorted(f for f in os.listdir(directory) if f.endswith(".safetensors")):
        with safe_open(os.path.join(directory, name), framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
            key_map = map_transformer_keys(keys, cfg)
            part: dict[str, "torch.Tensor"] = {}
            for key, targets in key_map.items():
                if not targets:
                    continue
                part[targets[0]] = fh.get_tensor(key)
        yield part
        del part

    pdd_meta = _read_json(os.path.join(checkpoint_dir, "model_index.json")).get("_minimax_h3", {})
    pdd_file = pdd_meta.get("pdd_fused_heads")
    if pdd_file:
        heads_part: dict[str, "torch.Tensor"] = {}
        with safe_open(os.path.join(checkpoint_dir, pdd_file), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
                if tensor.shape[0] != cfg["num_pdd_heads"]:
                    raise ValueError(f"{key}: {tensor.shape[0]} heads, config says {cfg['num_pdd_heads']}")
                for step in range(cfg["num_pdd_heads"]):
                    heads_part[f"pdd_proj_out.{step}.{key.rsplit('.', 1)[-1]}"] = tensor[step].contiguous()
        yield heads_part


def load_transformer_state_dict(checkpoint_dir: str):
    """Original ``transformer/`` + PDD heads -> vendored transformer state dict + config."""
    from safetensors import safe_open

    cfg = transformer_config(checkpoint_dir)
    state = _load_safetensors_dir(os.path.join(checkpoint_dir, "transformer"))
    key_map = map_transformer_keys(list(state), cfg)

    out = {}
    for key, targets in key_map.items():
        if not targets:
            continue
        out[targets[0]] = state[key]

    # Per-step PDD heads (stacked in a side file) -> pdd_proj_out.{i}.{weight,bias}
    pdd_meta = _read_json(os.path.join(checkpoint_dir, "model_index.json")).get("_minimax_h3", {})
    pdd_file = pdd_meta.get("pdd_fused_heads")
    if pdd_file:
        with safe_open(os.path.join(checkpoint_dir, pdd_file), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
                if tensor.shape[0] != cfg["num_pdd_heads"]:
                    raise ValueError(f"{key}: {tensor.shape[0]} heads, config says {cfg['num_pdd_heads']}")
                for step in range(cfg["num_pdd_heads"]):
                    out[f"pdd_proj_out.{step}.{key.rsplit('.', 1)[-1]}"] = tensor[step].contiguous()

    return out, cfg


def load_vae_state_dict(checkpoint_dir: str):
    """Original ``vae/`` -> vendored image-VAE state dict + config."""
    import torch
    from safetensors import safe_open  # noqa: F401  (kept for symmetry / clarity)

    cfg = vae_config(checkpoint_dir)
    heads = cfg["decoder_num_attention_heads"]
    head_dim = cfg["decoder_attention_head_dim"]
    state = _load_safetensors_dir(os.path.join(checkpoint_dir, "vae"))
    key_map = map_vae_keys(list(state), cfg)

    out = {}
    for key, targets in key_map.items():
        if not targets:
            continue
        tensor = state[key]
        if ".attn.to_qkv." in key:
            q, k, v = _split_qkv(reorder_interleaved_qkv(tensor, heads, head_dim), heads, head_dim)
            out[targets[0]], out[targets[1]], out[targets[2]] = q, k, v
        elif ".ff.w1." in key:
            gate, up = tensor.chunk(2, dim=0)
            out[targets[0]] = torch.cat([up, gate], dim=0).contiguous()
        else:
            out[targets[0]] = tensor

    return out, cfg
