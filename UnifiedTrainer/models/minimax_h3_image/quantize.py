"""Pre-quantized (nf4) weight cache for the MiniMax-H3-Image DiT.

The DiT is ~31B parameters, so every cold start would otherwise:

1. build the model in bf16 on CPU (62 GB of RAM),
2. stream 62 GB of original weights into it,
3. re-quantize all ~600 Linears to nf4 on the GPU.

This module writes the *result* of step 3 once (packed uint8 nf4 weights + their
quantization statistics, ~16 GB) and reloads it directly onto the GPU, so a restart is
"read 16 GB + H2D copy" instead of "62 GB build + 62 GB read + quantize".

Format (one safetensors file + one JSON manifest next to it)::

    <module path>.weight      uint8 packed nf4, shape (out*in/2, 1)
    <module path>.bias        bf16
    <module path>.qs.<field>  quantization statistics (tensors)
    manifest.json             {"format": ..., "scalars": {...}, "tensors": [...]}

Non-quantized parameters (norms, patch projections that stayed bf16) are stored as-is under
their own name, so a reload never needs the original checkpoint.
"""
from __future__ import annotations

import json
import os
from typing import Any

FORMAT = "minimax_h3_image_nf4_v1"
MANIFEST = "manifest.json"
WEIGHTS = "transformer_nf4.safetensors"


# ── quant-state (de)serialization ────────────────────────────────────────────


def _flatten(obj: Any, prefix: str, tensors: dict, scalars: dict) -> None:
    """QuantState.as_dict() -> flat tensors + scalars (nested state2 handled recursively)."""
    import torch

    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            _flatten(value, name + ".", tensors, scalars)
        elif torch.is_tensor(value):
            tensors[name] = value.detach().to("cpu")
        elif value is None:
            scalars[name] = None
        else:
            scalars[name] = value


def _unflatten(tensors: dict, scalars: dict, prefix: str = "") -> dict:
    """Inverse of :func:`_flatten`, limited to the keys that start with ``prefix``."""
    out: dict[str, Any] = {}
    for key, value in scalars.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            if "." not in rest:
                out[rest] = value
    for key, value in tensors.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            if "." not in rest:
                out[rest] = value
    # nested groups (state2) are reconstructed as sub-dicts
    nested = {key[len(prefix):].split(".", 1)[0] for key in list(scalars) + list(tensors)
              if key.startswith(prefix) and "." in key[len(prefix):]}
    for name in nested:
        out[name] = _unflatten(tensors, scalars, prefix=f"{prefix}{name}.")
    return out


# ── save ─────────────────────────────────────────────────────────────────────


def save_nf4(model, out_dir: str, compute_dtype=None) -> str:
    """Write ``model``'s quantized weights to ``out_dir`` (safetensors + manifest)."""
    import torch
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, Any] = {}
    manifest_tensors: list[str] = []
    n_quantized = 0

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        quant_state = getattr(weight, "quant_state", None) if weight is not None else None
        if quant_state is not None and hasattr(quant_state, "as_dict"):
            n_quantized += 1
            tensors[f"{name}.weight"] = weight.data.detach().to("cpu").contiguous()
            _flatten(quant_state.as_dict(), f"{name}.qs.", tensors, scalars)
            manifest_tensors.append(f"{name}.weight")
            if getattr(module, "bias", None) is not None:
                tensors[f"{name}.bias"] = module.bias.data.detach().to("cpu").contiguous()

    quantized_names = {name for name in (n.rsplit(".weight", 1)[0] for n in tensors) if name}
    for name, param in model.named_parameters():
        if any(name.startswith(q + ".") for q in quantized_names):
            continue
        tensors[name] = param.detach().to("cpu").contiguous()
    for name, buffer in model.named_buffers():
        if name in tensors:
            continue
        tensors[name] = buffer.detach().to("cpu").contiguous()

    weights_path = os.path.join(out_dir, WEIGHTS)
    save_file(tensors, weights_path, metadata={"format": FORMAT})

    manifest = {
        "format": FORMAT,
        "quantized_modules": n_quantized,
        "compute_dtype": str(compute_dtype or "bfloat16"),
        "scalars": scalars,
        "tensor_nbytes": sum(t.numel() * t.element_size() for t in tensors.values()),
    }
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    return weights_path


# ── load ─────────────────────────────────────────────────────────────────────


def is_available(ckpt_dir: str) -> bool:
    return os.path.isfile(os.path.join(ckpt_dir, WEIGHTS)) and os.path.isfile(os.path.join(ckpt_dir, MANIFEST))


def build_meta_model(model_cls, cfg: dict):
    """Build the model on the meta device and turn every nn.Linear into a meta nf4 Linear."""
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    with torch.device("meta"):
        model = model_cls(**cfg)

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, compute_dtype=torch.bfloat16, quant_type="nf4",
                )
                setattr(module, name, new)
            else:
                replace(child)

    replace(model)
    return model


def load_nf4_into(model, ckpt_dir: str, device="cuda") -> dict:
    """Restore a saved nf4 checkpoint into a meta model built by :func:`build_meta_model`."""
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    with open(os.path.join(ckpt_dir, MANIFEST), encoding="utf-8") as fh:
        manifest = json.load(fh)
    tensors = load_file(os.path.join(ckpt_dir, WEIGHTS), device="cpu")
    scalars = manifest["scalars"]

    modules = dict(model.named_modules())
    loaded_params = 0
    for name in list(tensors):
        if name.endswith(".weight") and f"{name[:-7]}.qs.quant_type" in scalars:
            module = modules.get(name[:-7])
            if module is None or not isinstance(module, bnb.nn.Linear4bit):
                raise RuntimeError(f"nf4 checkpoint has weights for unknown module {name[:-7]}")
            stats = _unflatten(tensors, scalars, prefix=f"{name[:-7]}.qs.")
            module.weight = bnb.nn.Params4bit.from_prequantized(
                tensors[name], stats, requires_grad=False, device=device, module=module,
            )
            loaded_params += 1
        elif name.endswith(".bias") and name[:-5] in modules:
            modules[name[:-5]].bias = nn.Parameter(tensors[name].to(device), requires_grad=False)

    for name, param in model.named_parameters():
        if name in tensors:
            continue
        modules_name, _, attr = name.rpartition(".")
        target = modules.get(modules_name)
        if target is None:
            raise RuntimeError(f"cannot place parameter {name}")
        setattr(target, attr, nn.Parameter(tensors[name].to(device), requires_grad=False))
        loaded_params += 1

    for name, buf in model.named_buffers():
        if name in tensors:
            continue
    return {"quantized": loaded_params, "manifest": manifest}


def load_streaming_quantized(model, ckpt_dir: str, cfg: dict, device="cuda") -> int:
    """Fill a meta nf4 model shard by shard, quantizing each layer on the GPU.

    This is the memory-lean path: peak CPU RAM is one checkpoint shard (~5.5 GB) instead of
    a 62 GB bf16 model, and peak VRAM stays at the final nf4 footprint (~16 GB) plus a
    single layer's transient. Every quantized parameter is produced by
    ``Params4bit(...).cuda()``, i.e. bitsandbytes does the quantization once, on the GPU.
    """
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    from . import convert as _convert

    modules = dict(model.named_modules())
    placed = 0
    for part in _convert.iter_transformer_state_dicts(ckpt_dir, cfg):
        for name, tensor in part.items():
            module_path, _, attr = name.rpartition(".")
            module = modules.get(module_path)
            if module is None:
                raise RuntimeError(f"no module for checkpoint tensor {name}")
            if isinstance(module, bnb.nn.Linear4bit):
                param = bnb.nn.Params4bit(tensor, requires_grad=False, quant_type="nf4")
                setattr(module, attr, param.cuda(device=device))
            else:
                setattr(module, attr, nn.Parameter(tensor.to(device), requires_grad=False))
            placed += 1
        del part
    return placed


def save_from_streaming(model, ckpt_dir: str, out_dir: str, cfg: dict, compute_dtype=None) -> str:
    """One-shot helper: meta model -> streaming nf4 -> snapshot on disk."""
    path = save_nf4(model, out_dir, compute_dtype)
    return path


# ── text encoder (Qwen3-VL 33B): meta build + streamed int8 ──────────────────


def build_meta_hf_model(config, model_cls):
    """Instantiate an HF model on the meta device (0 bytes of RAM)."""
    from accelerate import init_empty_weights

    with init_empty_weights():
        return model_cls(config)


def replace_linears_with_int8(model) -> int:
    """Swap every nn.Linear for a meta int8 Linear8bitLt (quantization happens on .cuda())."""
    import bitsandbytes as bnb
    import torch.nn as nn

    count = 0

    def walk(module):
        nonlocal count
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new = bnb.nn.Linear8bitLt(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=0.0,  # LLM.int8() outliers off: frozen weights only
                )
                setattr(module, name, new)
                count += 1
            else:
                walk(child)

    walk(model)
    return count


def stream_hf_shards_into(model, ckpt_dir: str, device="cuda", skip_prefixes=(), quant="int8") -> dict:
    """Place a meta HF model's weights onto ``device`` straight from its safetensors shards.

    Each Linear is quantized (int8 by default) as it is moved, and ``skip_prefixes`` keeps
    unused heads off the device entirely -- for text-embedding extraction the LM head and the
    vision tower are never called, so they are dropped.
    """
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn
    from safetensors import safe_open

    modules = dict(model.named_modules())
    placed = skipped = 0
    for name in sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(ckpt_dir, name), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                if any(key.startswith(prefix) for prefix in skip_prefixes):
                    skipped += 1
                    continue
                tensor = fh.get_tensor(key)
                module_path, _, attr = key.rpartition(".")
                module = modules.get(module_path)
                if module is None:
                    raise RuntimeError(f"no module for text-encoder tensor {key}")
                if quant == "int8" and isinstance(module, bnb.nn.Linear8bitLt):
                    param = bnb.nn.Int8Params(tensor, requires_grad=False, has_fp16_weights=False)
                    setattr(module, attr, param.cuda(device=device))
                elif quant == "nf4" and isinstance(module, bnb.nn.Linear4bit):
                    param = bnb.nn.Params4bit(tensor, requires_grad=False, quant_type="nf4")
                    setattr(module, attr, param.cuda(device=device))
                else:
                    setattr(module, attr, nn.Parameter(tensor.to(device), requires_grad=False))
                placed += 1
                del tensor
    return {"placed": placed, "skipped": skipped}
