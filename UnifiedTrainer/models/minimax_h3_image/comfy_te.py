"""Load the Comfy NVFP4 text encoder the way ComfyUI does.

Packed weights stay in host RAM (16 GB); every Linear is a lazy module that dequantises its
own matrix onto the GPU inside ``forward``. Norms and the token embedding are the only dense
parameters and live on the GPU with the rest of the module tree. This avoids both failure
modes we hit before: no 66 GB bf16 blow-up and no 33 GB int8 residency.
"""
from __future__ import annotations

import json
import struct

import torch
import torch.nn as nn

from .comfy_layers import Nvfp4Linear

SKIP_PREFIXES = ("model.visual.", "visual.", "lm_head.")

# Which Qwen3-VL decoder layer conditions the transformer. Mirrors
# `UnifiedTrainer.models.minimax_h3_image.TEXT_ENCODER_LAYER`; it is repeated here because this
# module is imported *by* the package `__init__`, so importing the constant back would cycle.
TEXT_ENCODER_LAYER = 50


class ComfySafetensors:
    """Streaming reader: one tensor per call, no bulk materialisation."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            self.header = json.loads(fh.read(n))
        self.base = 8 + n
        self.keys = [k for k in self.header if k != "__metadata__"]

    def _variants(self, key: str):
        """Both spellings of a parameter path (HF ``model.language_model.*`` vs Comfy ``model.*``)."""
        yield key
        if key.startswith("model.language_model."):
            yield "model." + key[len("model.language_model."):]
        elif key.startswith("model."):
            yield "model.language_model." + key[len("model."):]

    def find(self, key: str):
        for candidate in self._variants(key):
            if candidate in self.header:
                return candidate
        return None

    def has(self, key: str) -> bool:
        return self.find(key) is not None

    def shape(self, key: str):
        real = self.find(key) or key
        return tuple(self.header[real]["shape"])

    def get(self, key: str) -> torch.Tensor:
        from .comfy_quant import _f8e4m3_to_float

        key = self.find(key) or key
        meta = self.header[key]
        off0, off1 = meta["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.base + off0)
            raw = fh.read(off1 - off0)
        dt, shape = meta["dtype"], tuple(meta["shape"])
        if dt == "U8":
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
        if dt == "I8":
            return torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(shape)
        if dt == "F32":
            return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        if dt == "BF16":
            return torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16).reshape(shape)
        if dt == "F16":
            return torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape)
        if dt == "F8_E4M3":
            # comfy_kitchen dispatches on the dtype: it wants the fp8 tensor itself, not a
            # decoded float32 copy.
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape).view(
                torch.float8_e4m3fn
            )
        raise ValueError(f"{key}: unsupported dtype {dt}")


def build_lazy_text_encoder(checkpoint_path: str, config, device="cuda",
                            skip_prefixes=SKIP_PREFIXES):
    """Qwen3-VL whose Linear layers decode NVFP4 on the fly (ComfyUI's strategy)."""
    from transformers import Qwen3VLForConditionalGeneration

    # The Comfy checkpoint only carries decoder layers 0..49, and that is enough: the stack is
    # causal, so layer 50's output is identical whether or not layers 51..63 exist. What the
    # conditioning needs is that layer's *pre-norm* output -- taking the last hidden state of a
    # 50-layer stack instead yields the post-norm tensor, which diffusers' image pipeline
    # explicitly rejects:
    #     "The last hidden state of a stack truncated to exactly 50 layers is post-norm and is
    #      not the conditioning MiniMax-H3 expects."
    # `encode_text` therefore reads hidden_states[-2] (pre-norm) when the stack is truncated
    # and hidden_states[50] when it is not.
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "num_hidden_layers", 0) > TEXT_ENCODER_LAYER:
        text_cfg.num_hidden_layers = TEXT_ENCODER_LAYER

    # Real bf16 init on CPU: transformers registers mask/rope buffers during __init__, and a
    # meta build leaves them on meta (which breaks every attention call later).
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3VLForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    reader = ComfySafetensors(checkpoint_path)
    modules = dict(model.named_modules())

    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = modules.get(parent_name)
        if parent is None:
            continue
        if not all(reader.has(f"{name}.{s}") for s in ("weight", "weight_scale", "weight_scale_2")):
            continue
        out_f = reader.shape(name + ".weight")[0]
        in_f = reader.shape(name + ".weight")[1] * 2
        old_weight = getattr(module, "weight", None)
        setattr(parent, attr, Nvfp4Linear(
            out_f, in_f,
            reader.get(name + ".weight"),
            reader.get(name + ".weight_scale"),
            reader.get(name + ".weight_scale_2"),
            bias=None,
        ))
        module.weight = None
        del old_weight
        swapped += 1

    placed = 0
    for name, param in list(model.named_parameters()):
        base = name[:-len(".weight")] if name.endswith(".weight") else name
        if reader.has(base + ".weight_scale"):
            continue  # owned by a lazy layer
        if not reader.has(name):
            continue
        tensor = reader.get(name)
        dtype = torch.bfloat16 if tensor.dtype in (torch.bfloat16, torch.float16) else None
        tensor = tensor.to(device=device, dtype=dtype)
        parent_name, _, attr = name.rpartition(".")
        parent = modules.get(parent_name)
        if parent is None:
            continue
        setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))
        placed += 1

    # Buffers (rope frequencies, mask templates) were created on CPU by the real init; the
    # lazy layers keep their packed tensors as plain attributes, so this move only touches
    # parameters/buffers.
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Tells the trainer's cache-build path not to call .to(device): the packed stack must
    # stay in host RAM and the dense tensors are already on the device.
    model.is_quantized = True
    return model, {"lazy_linears": swapped, "dense_placed": placed}
