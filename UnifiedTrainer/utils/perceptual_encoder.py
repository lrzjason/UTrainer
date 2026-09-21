"""Shared EUPE perceptual-encoder utilities.

Single source of truth for:
  - the frozen EUPE encoder load (torch.hub, local vendored checkout)
  - the canonical image preprocessing ([-1,1] pixels -> [0,1] -> resize ->
    ImageNet normalize) and multi-layer feature extraction

Used by BOTH the pfm loss (pred branch, training time) and the cache
builder (target branch, cache build time) so the two can never drift:
cached φ(I_orig) features and training-time φ(D(x̂0)) features must come
from identical weights + identical preprocessing.

EUPE input transform is the official README's: resize -> [0,1] ->
ImageNet normalize (also `eupe/configs/ssl_default_config.yaml`
rgb_mean/rgb_std). The Qwen VAE's latents_mean/latents_std are 16-channel
LATENT statistics handled in the adapter's encode/decode paths — they
never apply to 3-channel RGB encoder input.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

# ImageNet stats — the EUPE encoder's own training-time normalization
# (eupe/configs/ssl_default_config.yaml: rgb_mean / rgb_std).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP stats — RADIO's input conditioner (input_conditioner.py:
# get_default_conditioner -> OPENAI_CLIP_MEAN/STD, input_scale=1.0 on [0,1]).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_encoder_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"encoder_dtype must be one of {sorted(_DTYPE_MAP)}, got {name!r}"
        )
    return _DTYPE_MAP[name]


def load_eupe_encoder(
    encoder_name: str,
    hub_repo_dir: Optional[str],
    weights_path: Optional[str],
    layers: Sequence[int],
) -> torch.nn.Module:
    """Load the frozen EUPE encoder (CPU; caller places it)."""
    if not hub_repo_dir or not weights_path:
        raise ValueError(
            "EUPE: 'hub_repo_dir' (local checkout of facebookresearch/EUPE) "
            "and 'weights_path' (the EUPE-*.pt file) are both required, e.g. "
            "{'encoder': 'eupe_vits16', 'hub_repo_dir': '.../EUPE', "
            "'weights_path': '.../EUPE-ViT-S.pt'}"
        )
    encoder = torch.hub.load(
        hub_repo_dir,
        encoder_name,
        source="local",
        pretrained=True,
        weights=weights_path,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    depth = getattr(encoder, "n_blocks", None)
    if depth is not None:
        bad = [int(l) for l in layers if l < 0 or l >= depth]
        if bad:
            raise ValueError(
                f"EUPE: layers {bad} out of range for {encoder_name} "
                f"(depth={depth}, 0-indexed block ids)"
            )
    return encoder


def prepare_pixels01(
    pixels_pm1: torch.Tensor,
    resolution: int,
    aspect: str = "square",
) -> torch.Tensor:
    """[-1, 1] pixels -> [0, 1] -> bilinear resize (fp32).

    Shared resize step; encoder-specific normalization is applied on top
    (ImageNet for EUPE, CLIP for RADIO).

    aspect:
      "square"   – legacy behaviour, force (resolution, resolution). Used by
                   the EUPE path (archived runs, square feature caches).
      "preserve" – aspect-preserving: long side -> `resolution`, short side
                   proportional, BOTH sides rounded to multiples of 16 (the
                   RADIO patch size) so the token grid stays exact. Stretching
                   a portrait into a square feeds the encoder a distorted
                   view it was never trained on — features lose calibration
                   exactly where PFM needs them (off-manifold geometry) —
                   and mismatches native-aspect inference.
    """
    x = (pixels_pm1.float() + 1.0) * 0.5
    if aspect == "square":
        size = (resolution, resolution)
    elif aspect == "preserve":
        h, w = int(x.shape[-2]), int(x.shape[-1])
        scale = resolution / max(h, w)
        nh = max(16, round(h * scale / 16) * 16)
        nw = max(16, round(w * scale / 16) * 16)
        size = (nh, nw)
    else:
        raise ValueError(f"aspect must be 'square' or 'preserve', got {aspect!r}")
    return F.interpolate(
        x,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def prepare_pixels(pixels_pm1: torch.Tensor, resolution: Optional[int]) -> torch.Tensor:
    """[-1, 1] pixels -> [0, 1] -> square resize -> ImageNet normalize (fp32).

    ``resolution`` None or 0 skips the resize entirely — pixels enter the
    encoder at their native (decoded) size, 1:1 (the dinov3 'auto' mode).
    """
    x = (pixels_pm1.float() + 1.0) * 0.5
    if resolution:
        x = F.interpolate(
            x,
            size=(int(resolution), int(resolution)),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def prepare_pixels_radio(pixels_pm1: torch.Tensor, resolution: int) -> torch.Tensor:
    """[-1, 1] pixels -> [0, 1] -> aspect-preserving resize -> CLIP normalize.

    Mirrors RADIO's own InputConditioner (get_default_conditioner):
    input_scale=1.0 on [0,1], OPENAI_CLIP_MEAN/STD; the resize itself is
    the caller's job (RADIO prefers 512 long side, patch 16; CPE accepts
    non-square patch grids natively up to 2048).
    """
    x = prepare_pixels01(pixels_pm1, resolution, aspect="preserve")
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def encode_eupe_layers(
    encoder: torch.nn.Module,
    pixels_pm1: torch.Tensor,
    layers: Sequence[int],
    resolution: int,
    encoder_dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Frozen EUPE -> per-block patch-token grids, one (B, C, h, w) per layer.

    NO no_grad here: the pfm pred branch decodes x̂0 WITH gradients and they
    must flow through this encode. Call sites that don't need grads (cache
    build) wrap themselves in torch.no_grad().

    Args:
        pixels_pm1: (B, 3, H, W) in [-1, 1].
    """
    x = prepare_pixels(pixels_pm1, resolution).to(encoder_dtype)
    outputs = encoder.get_intermediate_layers(
        x, n=list(layers), reshape=True, norm=True
    )
    return [out.float() for out in outputs]


def load_radio_encoder(
    model_path: Optional[str],
    layers: Sequence[int],
) -> torch.nn.Module:
    """Load the frozen NVlabs RADIO (HF format) encoder.

    `model_path` points at the model directory (config.json + safetensors +
    the shipped .py sources). We deliberately do NOT go through
    ``transformers.AutoModel.from_pretrained(trust_remote_code=True)``:
    its dynamic-module cache copies only part of the multi-level relative
    import chain (hf_model -> radio_model -> dual_hybrid_vit -> ...) and
    crashes on the missing file. Instead the model dir is registered as a
    synthetic import package so the shipped sources execute in place, then
    weights are loaded from the safetensors directly.
    """
    import importlib
    import json
    import os
    import sys
    import types

    from safetensors.torch import load_file

    if not model_path:
        raise ValueError(
            "RADIO: 'model_path' (local HF dir of C-RADIOv3-B with its "
            "shipped .py sources) is required, e.g. "
            "{'encoder': 'radio_v3b', 'model_path': '.../C-RADIOv3-B'}"
        )
    model_path = os.fspath(model_path)
    if not os.path.isfile(os.path.join(model_path, "hf_model.py")):
        raise FileNotFoundError(
            f"RADIO: {model_path!r} does not contain hf_model.py — point "
            "'model_path' at the model dir with its shipped sources"
        )

    pkg_name = "ut_radio_hf_pkg"
    for stale in [k for k in sys.modules if k == pkg_name or k.startswith(pkg_name + ".")]:
        del sys.modules[stale]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [model_path]
    sys.modules[pkg_name] = pkg
    hf_model_mod = importlib.import_module(f"{pkg_name}.hf_model")

    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = hf_model_mod.RADIOConfig.from_dict(cfg_dict)
    encoder = hf_model_mod.RADIOModel(cfg)

    state = load_file(os.path.join(model_path, "model.safetensors"))
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    # Buffers created at init (e.g. CPE position grids) may legitimately be
    # "missing"; any unexpected key means the wrong checkpoint layout.
    if unexpected:
        raise RuntimeError(
            f"RADIO: unexpected checkpoint keys ({len(unexpected)}), e.g. "
            f"{unexpected[:3]} — wrong safetensors layout for this source tree"
        )
    missing_real = [k for k in missing if k not in dict(encoder.named_buffers())]
    if missing_real:
        raise RuntimeError(
            f"RADIO: {len(missing_real)} parameter keys not in checkpoint, "
            f"e.g. {missing_real[:3]}"
        )

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    depth = len(encoder.radio_model.model.blocks)
    bad = [int(l) for l in layers if l < 0 or l >= depth]
    if bad:
        raise ValueError(
            f"RADIO: layers {bad} out of range (depth={depth}, 0-indexed)"
        )
    return encoder


def encode_radio_layers(
    encoder: torch.nn.Module,
    pixels_pm1: torch.Tensor,
    layers: Sequence[int],
    resolution: int,
    encoder_dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Frozen RADIO -> per-block patch-token grids, one (B, C, h, w) per layer.

    NO no_grad here: the pfm pred branch decodes x̂0 WITH gradients and they
    must flow through this encode. Call sites that don't need grads (cache
    build) wrap themselves in torch.no_grad().

    Pixel prep replicates RADIO's conditioner EXACTLY (CLIP normalize on
    [0,1]; the conditioner's only other op is a dtype cast). We then call
    the inner ViT's forward_intermediates DIRECTLY — going through
    radio_model.forward_intermediates would apply the conditioner a SECOND
    time (double normalization) and re-cast the input to its init dtype
    (fp32), clashing with the bf16 ViT. norm=True keeps the final-norm
    semantics matching the EUPE get_intermediate_layers(..., norm=True)
    contract; output_fmt NCHW.
    """
    from UnifiedTrainer.utils.perceptual_encoder import prepare_pixels_radio

    x = prepare_pixels_radio(pixels_pm1, resolution).to(encoder_dtype)
    items = encoder.model.forward_intermediates(
        x,
        indices=list(layers),
        return_prefix_tokens=False,
        norm=True,
        stop_early=True,
        output_fmt="NCHW",
        intermediates_only=True,
    )
    outputs: List[torch.Tensor] = []
    grid = (x.shape[-2] // 16, x.shape[-1] // 16)  # patch-16 token grid, h≠w for non-square inputs
    for item in items:
        # CPE-patched ViTs may hand back RadioOutput wrappers or raw tensors.
        t = getattr(item, "features", item)
        if t.dim() == 3:  # NLC -> NCHW using the actual input grid (not sqrt(tokens): non-square)
            t = t.transpose(1, 2).reshape(
                t.shape[0], t.shape[2], grid[0], grid[1]
            )
        outputs.append(t.float())
    return outputs


# DINOv2 is a patch-14 ViT (facebook/dinov2-* / dinov2-with-registers-*),
# normalized with the same ImageNet stats as EUPE (prepare_pixels).
DINO_PATCH_SIZE = 14


class _DINOv2HubShim(torch.nn.Module):
    """torch.hub-style ``get_intermediate_layers`` shim over HF ``Dinov2Model``.

    Reproduces the official facebookresearch/dinov2 contract the EUPE encode
    path already relies on:
        get_intermediate_layers(x, n=list_of_block_ids, reshape=True, norm=True)
    returning one patch-token grid per requested block — CLS/register prefix
    stripped, the FINAL LayerNorm applied to every returned block when
    norm=True (torch.hub semantics), reshaped to (B, C, H//14, W//14).
    Built on forward(output_hidden_states=True) so it does not depend on
    whether the installed transformers version ships its own
    intermediate-layers API.
    """

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.patch_size = int(core.config.patch_size)
        self.depth = int(core.config.num_hidden_layers)
        self.embed_dim = int(core.config.hidden_size)
        # 1 CLS token + optional register tokens (dinov2-with-registers-*).
        self.num_prefix_tokens = 1 + int(
            getattr(core.config, "num_register_tokens", 0) or 0
        )

    @property
    def config(self):
        return self.core.config

    def get_intermediate_layers(self, x, n, reshape=True, norm=True):
        out = self.core(pixel_values=x, output_hidden_states=True, return_dict=True)
        # hidden_states[0] = embeddings output; hidden_states[i + 1] = block i
        # output (pre final-layernorm — the shim applies it itself on norm=True).
        hidden = out.hidden_states
        h_grid = x.shape[-2] // self.patch_size
        w_grid = x.shape[-1] // self.patch_size
        outputs = []
        for idx in n:
            t = hidden[int(idx) + 1][:, self.num_prefix_tokens:]
            if norm:
                t = self.core.layernorm(t)
            if reshape:
                t = t.transpose(1, 2).reshape(
                    t.shape[0], t.shape[2], h_grid, w_grid
                )
            outputs.append(t)
        return outputs


def load_dinov2_encoder(
    model_path: Optional[str],
    layers: Sequence[int],
) -> torch.nn.Module:
    """Load the frozen DINOv2 encoder (HF transformers format) from local dir.

    `model_path` points at a local HF snapshot of facebook/dinov2-* or
    facebook/dinov2-with-registers-* (config.json + model.safetensors), as
    downloaded by E:\\hf_models\\dl_dinov2.py. Returns a _DINOv2HubShim so the
    encode path shares the exact get_intermediate_layers contract as EUPE.
    """
    import os

    from transformers import AutoModel

    if not model_path:
        raise ValueError(
            "DINOv2: 'model_path' (local HF dir of a facebook/dinov2-* "
            "snapshot) is required, e.g. {'encoder': 'dinov2_base', "
            "'model_path': 'E:/hf_models/dinov2-base'}"
        )
    model_path = os.fspath(model_path)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"DINOv2: {model_path!r} does not contain config.json — point "
            "'model_path' at the local HF snapshot dir"
        )
    # AutoModel resolves BOTH config model types: "dinov2" -> Dinov2Model
    # and "dinov2_with_registers" -> Dinov2WithRegistersModel (loading a
    # registers snapshot with plain Dinov2Model silently drops
    # embeddings.register_tokens and misaligns the token grid).
    core = AutoModel.from_pretrained(model_path, local_files_only=True)
    encoder = _DINOv2HubShim(core)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    bad = [int(l) for l in layers if l < 0 or l >= encoder.depth]
    if bad:
        raise ValueError(
            f"DINOv2: layers {bad} out of range for depth={encoder.depth} "
            "(0-indexed block ids)"
        )
    return encoder


def encode_dinov2_layers(
    encoder: torch.nn.Module,
    pixels_pm1: torch.Tensor,
    layers: Sequence[int],
    resolution: int,
    encoder_dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Frozen DINOv2 -> per-block patch-token grids, one (B, C, h, w) per layer.

    NO no_grad here: the pfm pred branch decodes x̂0 WITH gradients and they
    must flow through this encode. Call sites that don't need grads (cache
    build) wrap themselves in torch.no_grad().

    Preprocessing: square resize + ImageNet normalize — DINOv2's own image
    processor uses exactly these stats (prepare_pixels). DINOv2 is a
    patch-14 ViT: `resolution` must be a multiple of 14 (224/252/280/448...),
    otherwise the token grid is undefined.
    """
    if resolution % encoder.patch_size != 0:
        raise ValueError(
            f"DINOv2: resolution {resolution} is not a multiple of its patch "
            f"size {encoder.patch_size} — use e.g. 224/252/280/448 for "
            "'input_size'/'loss_resolution'"
        )
    x = prepare_pixels(pixels_pm1, resolution).to(encoder_dtype)
    outputs = encoder.get_intermediate_layers(
        x, n=list(layers), reshape=True, norm=True
    )
    return [out.float() for out in outputs]


# DINOv3 ConvNeXt: fully convolutional (NO positional embedding — the
# official convnext.py has none), so any HxW works; the only structural
# constraint is the total downsample factor: stem stride 4 x three 2x
# stages = 32. Multiples of 32 keep all four stage grids exact.
DINOV3_CONVNEXT_STRIDE = 32


class _DINOv3ConvNextShim(torch.nn.Module):
    """``get_intermediate_layers`` shim over transformers ``DINOv3ConvNextModel``.

    Mirrors the official facebookresearch/dinov3 ConvNeXt contract:
        get_intermediate_layers(x, n=list_of_stage_ids, reshape=True, norm=True)
    -> one NCHW grid per requested STAGE. Unlike the ViT branches, stages
    differ in BOTH channels and stride:
        stage 0: (B,  96, H/4,  W/4)
        stage 1: (B, 192, H/8,  W/8)
        stage 2: (B, 384, H/16, W/16)
        stage 3: (B, 768, H/32, W/32)
    norm=True applies the PER-STAGE norm — Identity for stages 0-2, the
    final LayerNorm for stage 3 — matching the official norms[i] list
    (small == depths [3,3,27,3], dims [96,192,384,768]).
    """

    def __init__(self, core):
        super().__init__()
        self.core = core
        cfg = core.config
        self.depth = int(cfg.num_stages)
        self.embed_dims = [int(c) for c in cfg.hidden_sizes]
        # Total downsample factor (stem 4, three 2x stages) — reused by the
        # encode-time resolution guard, same role as DINOv2's patch_size.
        self.patch_size = DINOV3_CONVNEXT_STRIDE
        self.stage_strides = [4, 8, 16, 32][: self.depth]

    @property
    def config(self):
        return self.core.config

    def _stage_norm(self, idx: int, t: torch.Tensor) -> torch.Tensor:
        if idx == self.depth - 1:
            # Final LayerNorm over channels (official: norms[-1] = LN).
            t = t.permute(0, 2, 3, 1)
            t = self.core.layer_norm(t)
            t = t.permute(0, 3, 1, 2)
        return t  # stages 0..-2: Identity (official norms[0:-1])

    def get_intermediate_layers(self, x, n, reshape=True, norm=True):
        out = self.core(pixel_values=x, output_hidden_states=True, return_dict=True)
        # hidden_states = [input, s0, s1, s2, s3], all NCHW (see
        # DINOv3ConvNextModel.forward — stage outputs stored pre-pooling).
        hidden = out.hidden_states
        outputs = []
        for idx in n:
            t = hidden[int(idx) + 1]
            if norm:
                t = self._stage_norm(int(idx), t)
            outputs.append(t)
        return outputs


def load_dinov3_convnext_encoder(
    model_path: Optional[str],
    layers: Sequence[int],
) -> torch.nn.Module:
    """Load the frozen DINOv3 ConvNeXt encoder (HF transformers format).

    `model_path` points at a local HF snapshot of facebook/
    dinov3-convnext-{tiny,small,base,large}-pretrain-lvd1689m as downloaded
    into E:\\hf_models. Uses the installed transformers' native
    DINOv3ConvNextModel (the snapshot is transformers-format: model_type
    "dinov3_convnext") — loading the official-repo ConvNeXt class instead
    would need a state-dict key remapper, so we don't.

    Returns a _DINOv3ConvNextShim so the encode path shares the exact
    get_intermediate_layers contract as the EUPE/DINOv2 branches, with the
    ConvNeXt twist that requested layers are STAGES (channels AND stride
    differ per stage).
    """
    import os

    try:
        from transformers import DINOv3ConvNextModel
    except ImportError:  # older/newer builds without the top-level export
        from transformers.models.dinov3_convnext.modeling_dinov3_convnext import (
            DINOv3ConvNextModel,
        )

    if not model_path:
        raise ValueError(
            "DINOv3: 'model_path' (local HF dir of a "
            "dinov3-convnext-*-pretrain-lvd1689m snapshot) is required, e.g. "
            "{'encoder': 'dinov3_convnext_small', "
            "'model_path': 'E:/hf_models/dinov3-convnext-small-pretrain-lvd1689m'}"
        )
    model_path = os.fspath(model_path)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"DINOv3: {model_path!r} does not contain config.json — point "
            "'model_path' at the local HF snapshot dir"
        )
    core = DINOv3ConvNextModel.from_pretrained(model_path, local_files_only=True)
    encoder = _DINOv3ConvNextShim(core)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    bad = [int(l) for l in layers if l < 0 or l >= encoder.depth]
    if bad:
        raise ValueError(
            f"DINOv3 ConvNeXt: stages {bad} out of range for depth="
            f"{encoder.depth} (0-indexed stage ids; per-stage channels "
            f"{encoder.embed_dims}, strides {encoder.stage_strides})"
        )
    return encoder


def encode_dinov3_convnext_layers(
    encoder: torch.nn.Module,
    pixels_pm1: torch.Tensor,
    layers: Sequence[int],
    resolution: Optional[int],
    encoder_dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Frozen DINOv3 ConvNeXt -> per-STAGE NCHW grids, one per requested stage.

    NO no_grad here: the pfm pred branch decodes x̂0 WITH gradients and they
    must flow through this encode. Call sites that don't need grads (cache
    build) wrap themselves in torch.no_grad().

    ``resolution`` int > 0: square resize to that size first. ``0``/None
    (pfm 'auto' mode): feed the pixels 1:1 at their decoded size — no
    resampling at all; whatever resolution the VAE decoded to is what the
    encoder sees (local test 512, a bigger remote box 768/1024, same config).

    NOTE the ConvNeXt twist vs the ViT branches: returned layers have
    DIFFERENT (C, h, w) per stage — the pfm loss compares per-layer pairs
    (pred vs target through the SAME encoder), so shapes always match within
    a pair; but the on-disk pfm cache contract (L, C, h, w) is uniform-only,
    which is why dinov3* rejects target_source='original'/precache.

    Preprocessing: ImageNet normalize (the snapshot's
    preprocessor_config.json uses exactly these stats — prepare_pixels).
    Fully convolutional: any resolution divisible by 32 works, including
    resolutions far above the 224 training size (no positional embedding to
    interpolate).
    """
    if resolution:
        if resolution % encoder.patch_size != 0:
            raise ValueError(
                f"DINOv3 ConvNeXt: resolution {resolution} is not a multiple "
                f"of the total stride {encoder.patch_size} (4x4 stem + three "
                "2x stages) — use e.g. 224/256/320/448/512, or 0 for 1:1 "
                "(native decoded size)"
            )
        x = prepare_pixels(pixels_pm1, resolution)
    else:
        # 1:1 mode: no resampling — validate the NATIVE pixel grid instead.
        h, w = int(pixels_pm1.shape[-2]), int(pixels_pm1.shape[-1])
        if h % encoder.patch_size or w % encoder.patch_size:
            raise ValueError(
                f"DINOv3 ConvNeXt: decoded pixels are {h}x{w}; with "
                f"loss_resolution=0 (1:1) both sides must be multiples of "
                f"the total stride {encoder.patch_size} — adjust the dataset "
                "resolution or decode_scale"
            )
        x = prepare_pixels(pixels_pm1, None)
    x = x.to(encoder_dtype)
    outputs = encoder.get_intermediate_layers(
        x, n=list(layers), reshape=True, norm=True
    )
    return [out.float() for out in outputs]


def build_eupe_encode_fn(
    pfm_params: dict,
    device: Optional[torch.device] = None,
):
    """Build a cache-build-time encode closure from a pfm loss's params.

    Returns ``fn(pixels01) -> Tensor (L, B, C, h, w) fp32`` where
    ``pixels01`` is (B, 3, H, W) in [0, 1] (the cache builder's native
    frame format). The encoder is loaded once and stays on `device`.

    Dispatches on the configured encoder family so the φ(I_orig) precompute
    can use whatever the pfm loss itself uses: ``radio*`` (local HF dir),
    ``dinov2*`` (local HF snapshot) or the default EUPE torch.hub checkout.
    ``dinov3*`` (ConvNeXt) is rejected here: its per-stage channels/strides
    differ, so per-stage features cannot stack into the uniform
    (L, B, C, h, w) cache contract — the pfm loss config-validates this
    earlier; this guard is defense in depth.
    """
    encoder_name = str(pfm_params.get("encoder", "eupe_vits16"))
    layers = [int(l) for l in pfm_params.get("layers", (5, 7, 9, 11))]
    resolution = int(pfm_params.get("loss_resolution", 256))
    dtype = resolve_encoder_dtype(str(pfm_params.get("encoder_dtype", "bf16")))
    if encoder_name.startswith("dinov3"):
        raise NotImplementedError(
            f"pfm cache build: encoder {encoder_name!r} (DINOv3 ConvNeXt) "
            "cannot back target_source='original' — per-stage channels/"
            "strides differ and the cache contract is uniform (L, B, C, h, "
            "w). Use target_source='decoded' (default) for dinov3*, or a "
            "dinov2/eupe/radio encoder for cached targets."
        )
    if encoder_name.startswith("radio"):
        encoder = load_radio_encoder(pfm_params.get("model_path"), layers)
        encode_fn = encode_radio_layers
    elif encoder_name.startswith("dinov2"):
        encoder = load_dinov2_encoder(pfm_params.get("model_path"), layers)
        encode_fn = encode_dinov2_layers
    else:
        encoder = load_eupe_encoder(
            encoder_name=encoder_name,
            hub_repo_dir=pfm_params.get("hub_repo_dir"),
            weights_path=pfm_params.get("weights_path"),
            layers=layers,
        )
        encode_fn = encode_eupe_layers
    if device is not None:
        encoder.to(device=device, dtype=dtype)

    def _encode(pixels01: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = encode_fn(
                encoder,
                pixels01.to(device).float() * 2.0 - 1.0,
                layers,
                resolution,
                dtype,
            )
        return torch.stack(feats, dim=0)  # (L, B, C, h, w)

    return _encode
