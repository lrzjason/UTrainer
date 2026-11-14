"""Pre-quantized weight caches for the MiniMax-H3-Image DiT (nf4 and fp8).

The DiT is ~31B parameters, so every cold start would otherwise:

1. build the model in bf16 on CPU (62 GB of RAM),
2. stream 62 GB of original weights into it,
3. re-quantize all ~600 Linears on the GPU.

This module writes the *result* of step 3 once and reloads it directly onto the GPU, so a
restart is "read 16 GB (nf4) / 33 GB (fp8) + H2D copy" instead of "62 GB build + 62 GB read +
quantize".

nf4 format (one safetensors file + one JSON manifest next to it)::

    <module path>.weight      uint8 packed nf4, shape (out*in/2, 1)
    <module path>.bias        bf16
    <module path>.qs.<field>  quantization statistics (tensors)
    manifest.json             {"format": ..., "scalars": {...}, "tensors": [...]}

fp8 format (same idea, see ``fp8_linear.py``)::

    <module path>.weight_fp8    float8_e4m3fn, [out, in]
    <module path>.weight_scale  float32, [1, out]
    <module path>.bias          bf16
    fp8_manifest.json           {"format": "minimax_h3_image_fp8_v1", "fp8_modules": N, ...}

Non-quantized parameters (norms, and the ``_keep_in_fp32_modules`` projections that stay
float32/bf16) are stored as-is under their own name, so a reload never needs the original
checkpoint.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

FORMAT = "minimax_h3_image_nf4_v1"
MANIFEST = "manifest.json"
WEIGHTS = "transformer_nf4.safetensors"

FP8_FORMAT = "minimax_h3_image_fp8_v1"
FP8_MANIFEST = "fp8_manifest.json"
FP8_WEIGHTS = "transformer_fp8.safetensors"


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


def _quant_state_for_bnb(stats: dict) -> dict:
    """Unpacked snapshot stats -> the dict bitsandbytes' ``QuantState.from_dict`` accepts.

    ``QuantState.as_dict()`` comes in two shapes:

    * **unpacked** -- ``quant_type`` is the plain string ``"nf4"`` and every other non-tensor
      field is a Python scalar;
    * **packed**   -- all non-tensor fields sit JSON-packed inside one uint8 tensor stored
      under a key ending in ``bitsandbytes__nf4``.

    ``QuantState.from_dict`` (bitsandbytes ``functional.py:465-472``) only accepts the
    packed shape: with zero ``quant_state.*`` tensor keys it raises ``There should be
    exactly one 'quant_state' item ... Detected []``. ``save_nf4`` writes the unpacked shape
    on purpose (the manifest stays human-readable and ``quant_type`` is readable without
    torch), so the re-packing happens here, on load. That keeps snapshots written by older
    builds valid -- rebuilding 16 GB is not needed for a key-naming fix.

    The packed payload is byte-identical to what ``as_dict(packed=True)`` would have written,
    because it goes through bitsandbytes' own ``pack_dict_to_tensor``.
    """
    import torch
    from bitsandbytes.functional import QuantState
    from bitsandbytes.utils import pack_dict_to_tensor

    # local copy: QuantState.from_dict() pops/merges keys in place, and callers keep theirs
    stats = dict(stats)
    # Already packed (a snapshot written by a newer build, or a bnb-native file): leave it.
    for key in stats:
        if key.split(".")[-1] in QuantState.valid_qs_type_keys:
            return stats

    quant_type = stats.pop("quant_type", "nf4")
    if torch.is_tensor(quant_type):
        # save_nf4 stored the type as the bytes of the string, prefix included
        quant_type = bytes(quant_type.flatten().to(torch.uint8).tolist()).decode("utf-8")
    quant_type = (str(quant_type).strip() or "nf4")
    if quant_type.startswith("bitsandbytes__"):
        quant_type = quant_type[len("bitsandbytes__"):]

    non_tensor: dict[str, Any] = {}
    tensor_fields: dict[str, Any] = {}
    for key, value in stats.items():
        if not torch.is_tensor(value):
            non_tensor[key] = value
        elif key == "nested_offset":  # as_dict() stores .item(); tolerate a tensor too
            non_tensor[key] = float(value.item())
        else:
            tensor_fields[key] = value

    non_tensor["quant_type"] = quant_type
    if isinstance(non_tensor.get("shape"), (list, tuple)):
        # json round-trip keeps it a list; torch.Size / tensor forms are normalised too
        non_tensor["shape"] = [int(v) for v in non_tensor["shape"]]
    tensor_fields[f"quant_state.bitsandbytes__{quant_type}"] = pack_dict_to_tensor(non_tensor)
    return tensor_fields


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
            qs_dict = dict(quant_state.as_dict())
            qt = qs_dict.get("quant_type")
            if isinstance(qt, str):
                qs_dict["quant_type"] = torch.tensor(
                    list(("" if qt.startswith("bitsandbytes__") else "bitsandbytes__" + qt).encode()),
                    dtype=torch.uint8,
                )
            _flatten(qs_dict, f"{name}.qs.", tensors, scalars)
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
        # bump when the packed layout/quant-state encoding changes; launchers reuse the
        # snapshot only when this matches (see chain_idempotent.py)
        "format_version": 2,
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


def is_fp8_available(ckpt_dir: str) -> bool:
    """True when ``ckpt_dir`` holds an fp8 snapshot (``fp8_manifest.json`` with our format)."""
    manifest = os.path.join(ckpt_dir or "", FP8_MANIFEST)
    if not os.path.isfile(manifest) or not os.path.isfile(os.path.join(ckpt_dir, FP8_WEIGHTS)):
        return False
    try:
        with open(manifest, encoding="utf-8") as fh:
            return json.load(fh).get("format") == FP8_FORMAT
    except (OSError, ValueError):
        return False


def build_meta_model(model_cls, cfg: dict, keep_in_fp32=None):
    """Build the model on the meta device and turn its nf4 Linears into meta nn.Linear4bit.

    ``keep_in_fp32`` defaults to the model class's ``_keep_in_fp32_modules``. The release
    declares ``["proj_in", "time_embedder", "pdd_proj_out", "rope"]`` and keeps them in
    float32; quantizing them (which the first version of this function did to every Linear)
    corrupts the AdaLN modulation and the output head.
    """
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    with torch.device("meta"):
        model = model_cls(**cfg)

    if keep_in_fp32 is None:
        keep_in_fp32 = getattr(model, "_keep_in_fp32_modules", None) or ()
    keep = tuple(keep_in_fp32)
    if keep:
        logger.info("nb4 build: keeping %s in fp32 (checkpoint _keep_in_fp32_modules)", list(keep))

    def replace(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(k in path for k in keep):
                    continue
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, compute_dtype=torch.bfloat16, quant_type="nf4",
                    device="meta",  # without this bnb allocates a full bf16 weight
                )
                setattr(module, name, new)
            else:
                replace(child, path)

    replace(model)
    _materialize_rope(model)
    return model


def _materialize_rope(model) -> None:
    """Recompute the non-persistent ``rope.inv_freq`` of a meta-built model.

    It is a pure function of the config, but ``build_meta_model*`` constructs under
    ``torch.device("meta")`` so it would stay a meta tensor and the first forward would fail
    with a device mismatch. Recomputing is exact; the streamed loaders then only have to move
    it onto the device (`reset_inv_freq(device)`).
    """
    rope = getattr(model, "rope", None)
    if rope is not None and hasattr(rope, "reset_inv_freq"):
        rope.reset_inv_freq()


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
        qs_key = f"{name[:-7]}.qs.quant_type"
        is_quantized = name.endswith(".weight") and (qs_key in scalars or qs_key in tensors)
        if is_quantized:
            module = modules.get(name[:-7])
            if module is None or not isinstance(module, bnb.nn.Linear4bit):
                raise RuntimeError(
                    f"nf4 snapshot {ckpt_dir} has weights for unknown module "
                    f"{name[:-7]!r}. A snapshot written before the checkpoint-layout change "
                    f"uses the old names (attn.to_q/to_k/to_v, ff.net.0.proj) and cannot be "
                    f"resumed -- delete it (or point training.quantize_path elsewhere) and let "
                    f"the run rebuild it from the bf16 shards"
                )
            stats = _quant_state_for_bnb(
                _unflatten(tensors, scalars, prefix=f"{name[:-7]}.qs.")
            )
            module.weight = bnb.nn.Params4bit.from_prequantized(
                tensors[name], stats, requires_grad=False, device=device, module=module,
            )
            loaded_params += 1
        elif name.endswith(".bias") and name[:-5] in modules:
            module = modules[name[:-5]]
            bias = tensors[name]
            if bias.dtype == torch.uint8 or bias.numel() != module.out_features:
                raise RuntimeError(
                    f"nf4 snapshot {ckpt_dir} has a malformed bias for {name[:-5]!r} "
                    f"({tuple(bias.shape)} {bias.dtype}); it was written by the pre-fix "
                    f"streaming loader that quantized 1-D biases. Delete the snapshot (or point "
                    f"training.quantize_path elsewhere) and let the run rebuild it"
                )
            module.bias = nn.Parameter(bias.to(device), requires_grad=False)

    # Everything the quantized path above does not own -- the norms, the projections that
    # stayed float32 (proj_in / time_embedder / pdd_proj_out) and the buffers -- is stored
    # under its own name. The test is "still on meta", not "present in the snapshot": those
    # tensors ARE in the snapshot, so a presence test skips exactly the ones to place and
    # leaves them meta, which only surfaces later as
    # ``NotImplementedError: Cannot copy out of meta tensor`` on the first .to(device).
    placed_plain = 0
    for name, param in list(model.named_parameters()):
        if param.device.type != "meta":
            continue
        if name not in tensors:
            raise RuntimeError(f"parameter {name} is still meta and missing from the snapshot")
        target_name, _, attr = name.rpartition(".")
        target = modules.get(target_name)
        if target is None:
            raise RuntimeError(f"cannot place parameter {name}")
        setattr(target, attr, nn.Parameter(tensors[name].to(device), requires_grad=False))
        placed_plain += 1

    restored_buffers = 0
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        if name not in tensors:
            raise RuntimeError(f"buffer {name} is still meta and missing from the snapshot")
        target_name, _, attr = name.rpartition(".")
        target = modules.get(target_name)
        if target is None:
            raise RuntimeError(f"cannot place buffer {name}")
        setattr(target, attr, tensors[name].to(device))
        restored_buffers += 1

    if placed_plain:
        logger.info("nf4 snapshot: %d non-quantized parameters placed", placed_plain)

    return {"quantized": loaded_params, "buffers": restored_buffers, "manifest": manifest}


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
    shard = 0
    for part in _convert.iter_transformer_state_dicts(ckpt_dir, cfg):
        shard += 1
        logger.info("nf4 streaming: shard %d/13 (%d tensors so far)", shard, placed)
        for name, tensor in part.items():
            module_path, _, attr = name.rpartition(".")
            module = modules.get(module_path)
            if module is None:
                raise RuntimeError(f"no module for checkpoint tensor {name}")
            if isinstance(module, bnb.nn.Linear4bit) and attr == "weight":
                param = bnb.nn.Params4bit(tensor, requires_grad=False, quant_type="nf4")
                setattr(module, attr, param.cuda(device=device))
            else:
                # The bias of a quantized Linear is a plain parameter: wrapping it in
                # Params4bit packs a 1-D [out] bias into uint8 [out/2, 1] and the first
                # forward dies in bnb with "The expanded size of the tensor (917) must match
                # the existing size (2688)".
                setattr(module, attr, nn.Parameter(tensor.to(device), requires_grad=False))
            placed += 1
        del part
    # `rope.inv_freq` is not in the checkpoint (it is recomputed): put it on the device the
    # weights went to, or the first forward mixes a CPU rotation with CUDA activations.
    _materialize_rope_on(model, device)
    return placed


def _materialize_rope_on(model, device) -> None:
    rope = getattr(model, "rope", None)
    if rope is not None and hasattr(rope, "reset_inv_freq"):
        rope.reset_inv_freq(device)


def save_from_streaming(model, ckpt_dir: str, out_dir: str, cfg: dict, compute_dtype=None) -> str:
    """One-shot helper: meta model -> streaming nf4 -> snapshot on disk."""
    path = save_nf4(model, out_dir, compute_dtype)
    return path


# ── fp8 (torch._scaled_mm, see fp8_linear.py) ───────────────────────────────


def build_meta_model_fp8(model_cls, cfg: dict, keep_in_fp32=None):
    """Build the model on the meta device with every quantizable Linear -> meta ``Fp8Linear``.

    The exclusion list is the release's ``_keep_in_fp32_modules``
    (``["proj_in", "time_embedder", "pdd_proj_out", "rope"]``): quantizing those makes the
    AdaLN modulation tens of times too large and the residual stream explodes.
    """
    import torch
    import torch.nn as nn

    from .fp8_linear import Fp8Linear

    with torch.device("meta"):
        model = model_cls(**cfg)

    if keep_in_fp32 is None:
        keep_in_fp32 = getattr(model, "_keep_in_fp32_modules", None) or ()
    keep = tuple(keep_in_fp32)
    if keep:
        logger.info("fp8 build: keeping %s in fp32 (checkpoint _keep_in_fp32_modules)", list(keep))

    def replace(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(k in path for k in keep):
                    continue
                setattr(module, name, Fp8Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, device="meta", dtype=torch.bfloat16,
                ))
            else:
                replace(child, path)

    replace(model)
    _materialize_rope(model)
    return model


def save_fp8(model, out_dir: str) -> str:
    """Write the fp8 snapshot: every parameter/buffer under its own name + a manifest.

    Unlike nf4 there is no packed quant-state to serialise: the fp8 weights and their
    per-output-row float32 scales are ordinary buffers of ``Fp8Linear``. Named parameters
    and buffers are dumped directly (not ``state_dict()``) so the **non-persistent**
    ``rope.inv_freq`` buffer is included -- a meta model rebuilt from the snapshot would
    otherwise still have it on the meta device and fail its first ``.to(device)``.
    """
    from safetensors.torch import save_file

    from .fp8_linear import Fp8Linear

    os.makedirs(out_dir, exist_ok=True)
    tensors = {name: tensor.detach().to("cpu").contiguous()
               for name, tensor in model.named_parameters()}
    tensors.update({name: tensor.detach().to("cpu").contiguous()
                    for name, tensor in model.named_buffers()})
    fp8_modules = sum(1 for m in model.modules() if isinstance(m, Fp8Linear))
    weights_path = os.path.join(out_dir, FP8_WEIGHTS)
    save_file(tensors, weights_path, metadata={"format": FP8_FORMAT})

    manifest = {
        "format": FP8_FORMAT,
        "format_version": 1,
        "fp8_modules": fp8_modules,
        "compute_dtype": "bfloat16",
        "tensor_nbytes": sum(t.numel() * t.element_size() for t in tensors.values()),
    }
    with open(os.path.join(out_dir, FP8_MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    logger.info("fp8 snapshot written: %s (%.1f GB, %d fp8 modules)", out_dir,
                manifest["tensor_nbytes"] / 1024**3, fp8_modules)
    return weights_path


def load_fp8_into(model, ckpt_dir: str, device="cuda") -> dict:
    """Restore an fp8 snapshot into a meta model built by :func:`build_meta_model_fp8`.

    Every tensor is placed by name (``load_state_dict`` cannot be used: a meta tensor is not
    a valid ``copy_`` destination, and ``assign=True`` ignores the non-persistent buffers).
    The test is "still on meta", the same rule the nf4 loader learned the hard way.
    """
    import torch.nn as nn
    from safetensors.torch import load_file

    with open(os.path.join(ckpt_dir, FP8_MANIFEST), encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("format") != FP8_FORMAT:
        raise ValueError(f"{ckpt_dir}: not an fp8 snapshot (format={manifest.get('format')!r})")

    state = load_file(os.path.join(ckpt_dir, FP8_WEIGHTS), device="cpu")
    modules = dict(model.named_modules())
    placed = 0
    for name, tensor in state.items():
        parent_name, _, attr = name.rpartition(".")
        parent = modules.get(parent_name)
        if parent is None:
            raise RuntimeError(f"fp8 snapshot has a tensor for unknown module {parent_name!r}")
        if attr in parent._buffers:
            setattr(parent, attr, tensor)
        elif attr in parent._parameters:
            setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))
        else:
            raise RuntimeError(f"fp8 snapshot tensor {name!r} matches no parameter or buffer")
        placed += 1

    still_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    still_meta += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if still_meta:
        raise RuntimeError(f"fp8 snapshot {ckpt_dir} left {len(still_meta)} tensors on meta, "
                           f"e.g. {still_meta[:4]}")
    model.to(device)
    return {"manifest": manifest, "tensors": placed}


def load_streaming_fp8(model, ckpt_dir: str, cfg: dict, device="cuda") -> int:
    """Fill a meta fp8 model from the ORIGINAL bf16 shards, quantizing each layer on the fly.

    Peak host RAM is one shard (~5.5 GB) instead of a 62 GB bf16 model; peak VRAM ends at the
    fp8 footprint (~33 GB) plus one layer's transient. ``save_fp8`` afterwards skips this pass
    on the next start.
    """
    import torch.nn as nn

    from . import convert as _convert
    from .fp8_linear import Fp8Linear, to_fp8_weight

    modules = dict(model.named_modules())
    placed = 0
    shard = 0
    for part in _convert.iter_transformer_state_dicts(ckpt_dir, cfg):
        shard += 1
        logger.info("fp8 streaming: shard %d (%d tensors so far)", shard, placed)
        for name, tensor in part.items():
            module_path, _, attr = name.rpartition(".")
            module = modules.get(module_path)
            if module is None:
                raise RuntimeError(f"no module for checkpoint tensor {name}")
            if isinstance(module, Fp8Linear) and attr == "weight":
                q, s = to_fp8_weight(tensor)     # [out, in] fp8 + [1, out] float32
                module.weight_fp8 = q.to(device)
                module.weight_scale = s.to(device)
            else:
                setattr(module, attr, nn.Parameter(tensor.to(device), requires_grad=False))
            placed += 1
        del part
    # `rope.inv_freq` is recomputed, not loaded: place it on the weights' device.
    _materialize_rope_on(model, device)
    return placed


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


def replace_linears_with_nf4(model) -> int:
    """Swap every nn.Linear for a meta nf4 Linear4bit (quantization happens on .cuda())."""
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    count = 0

    def walk(module):
        nonlocal count
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
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
    for shard_index, name in enumerate(
        sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")), start=1
    ):
        logger.info("text encoder streaming: %s (%d/%d), %d tensors placed", name, shard_index,
                    len([f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")]), placed)
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
                if quant == "int8" and isinstance(module, bnb.nn.Linear8bitLt) and attr == "weight":
                    param = bnb.nn.Int8Params(tensor, requires_grad=False, has_fp16_weights=False)
                    setattr(module, attr, param.cuda(device=device))
                elif quant == "nf4" and isinstance(module, bnb.nn.Linear4bit) and attr == "weight":
                    param = bnb.nn.Params4bit(tensor, requires_grad=False, quant_type="nf4")
                    setattr(module, attr, param.cuda(device=device))
                else:
                    # biases and every non-quantized tensor: a plain parameter. Wrapping a bias
                    # in Params4bit/Int8Params would quantize a 1-D tensor into a packed uint8
                    # blob and break the module's forward.
                    setattr(module, attr, nn.Parameter(tensor.to(device), requires_grad=False))
                placed += 1
                del tensor
    return {"placed": placed, "skipped": skipped}
