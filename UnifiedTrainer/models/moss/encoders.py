"""MOSS-Image encoders: Qwen3.5-9B text/vision tower + FLUX.2 VAE.

This is the encoder half of the MOSS-Image adapter (``models/moss/__init__.py``).
It answers three questions the adapter's training path asks:

1. **Text** - what does the DiT consume as conditioning?  A ``(L, 4096)`` tensor:
   the **last hidden state** (after the final zero-centred RMSNorm) of Qwen3.5-9B,
   one vector per token of the *whole* chat-templated prompt.  The reference taps
   exactly one layer - nothing is stacked and nothing is stripped (see
   ``encode_prompt``).
2. **Reference images** - MOSS's own vision tower lives inside the text encoder
   (``Qwen3_5Model.visual``); a reference image becomes ``<|image_pad|>`` tokens
   that the tower fills in.  ``image_token_mask`` marks those positions so the
   cache can tell text from vision tokens.
3. **Latents** - the FLUX.2 VAE's 32 channels, 2x2-patchified to the 128 channels
   the backbone's ``latent_channels`` expects, normalised with the VAE's own
   BatchNorm running statistics.

Attribution
-----------
Derived from **Kohaku-Blueleaf**'s **KohakuMoss**
(https://github.com/KohakuBlueleaf/KohakuMossTrainer, Apache-2.0), read-only
reference at ``ref/KohakuMossTrainer``:

* ``encoders/vae/__init__.py`` - ``patchify`` / ``unpatchify`` and the MOSS latent
  normalisation, i.e. the *order* of operations ``posterior -> patchify ->
  (z - running_mean) / sqrt(running_var + eps)`` and its exact inverse on decode
  (``FluxVae.encode`` / ``FluxVae.decode``, ref lines 24-35 and 187-202).
* ``encoders/text/tokens.py`` + ``text/forward.py`` + ``text/encoder.py`` - the
  prompt layout: the training chat template (``add_generation_prompt``, thinking
  enabled, **no** system prompt), truncation at 4096, single unpadded rows, and
  the conditioning tensor being ``last_hidden_state`` in the final-norm space
  (``HfForward._run`` returns ``out.last_hidden_state``; ``KernelForward`` ends
  with the same norm).  Nothing is stripped from the sequence.
* ``encoders/text/load.py`` - the multimodal ``Qwen3_5Config`` resolves to the
  text model through ``AutoModel`` and the model is used frozen in eval mode.

New in this port (not in the reference):

* HF-checkpoint loading (the reference builds a meta-parameter model and streams
  quantised weights; UTrainer loads ``text_encoder/`` directly).
* The **multimodal reference-image path**: the reference deletes ``model.visual``
  and is text-only, so the ``Picture N: <|vision_start|><|image_pad|>
  <|vision_end|>`` placeholder layout, the processor plumbing and
  ``image_token_mask`` are new (they follow UTrainer's Krea-2 adapter).
* CPU / float32 output contract, numpy-friendly, and the VAE statistics kept in
  float32 next to a bf16 VAE.

Everything here runs under ``torch.no_grad()`` and returns CPU float32 tensors -
the cache stores numpy.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

#: Qwen3.5-9B text width - the backbone's ``text_dim``.
TEXT_DIM = 4096
#: The FLUX.2 VAE's latent width (before MOSS's 2x2 patchify).
VAE_CHANNELS = 32
#: MOSS folds a 2x2 spatial neighbourhood into the channel axis.
LATENT_PATCH = 2
#: What the cache stores / the DiT consumes: 32 * 2 * 2.
LATENT_CHANNELS = VAE_CHANNELS * LATENT_PATCH * LATENT_PATCH
#: The reference truncates the chat-templated prompt at this many tokens
#: (``tokens.MAX_LENGTH``; the model's own ceiling is 4096, see the official
#: pipeline's ``encode_prompt(max_sequence_length=4096)``).
MAX_TEXT_LENGTH = 4096

#: Krea-2-style reference-image placeholder.  ``Picture N:`` is literal caption
#: text; the three special tokens bracket the pads the processor expands.
_IMG_PLACEHOLDER = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

#: Qwen3.5's image-processor constants.  ``preprocessor_config.json`` in a full
#: Qwen3.5 checkout carries exactly these; the MOSS HF directory does not ship
#: that file, so the manual processor below repeats them (verified byte-equal:
#: a 384x384 reference gives identical ``pixel_values`` to
#: ``AutoProcessor.from_pretrained(<full Qwen3.5 checkout>)``).
_IMAGE_MEAN = [0.5, 0.5, 0.5]
_IMAGE_STD = [0.5, 0.5, 0.5]
_IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 16777216}


# ─────────────────────────────────────────────────────────────────────────────
# path resolution - every directory is derived from the ``path`` argument
# ─────────────────────────────────────────────────────────────────────────────


def _as_dir(path: str | os.PathLike) -> Path:
    return Path(path).expanduser()


def _component_dir(path: str | os.PathLike, name: str) -> Path:
    """``<path>/<name>`` when that is a model directory, else ``path`` itself.

    UTrainer passes the component directory (``<model>/text_encoder``,
    ``<model>/vae``); a pipeline root (``<model>``) or an explicit component path
    are equally acceptable.
    """
    p = _as_dir(path)
    sub = p / name
    if sub.is_dir() and (sub / "config.json").is_file():
        return sub
    return p


def _tokenizer_dir(path: str | os.PathLike) -> Path:
    """The directory holding ``tokenizer.json`` / ``tokenizer_config.json``.

    MOSS keeps the tokenizer at ``<model>/tokenizer`` while the text encoder is at
    ``<model>/text_encoder``, so a sibling lookup is part of the contract (the
    reference's ``tokens.tokenizer_dir`` does the same).
    """
    p = _as_dir(path)
    for cand in (p, p / "tokenizer", p.parent / "tokenizer", p / "text_encoder"):
        if (cand / "tokenizer_config.json").is_file() or (cand / "tokenizer.json").is_file():
            return cand
    return p


def _processor_candidates(path: str | os.PathLike) -> List[Path]:
    """Directories that may hold a complete ``preprocessor_config.json``."""
    p = _as_dir(path)
    cands = [p, p / "text_encoder", p.parent, p.parent / "text_encoder", p / "processor"]
    out: List[Path] = []
    for cand in cands:
        if cand.is_dir() and cand not in out:
            out.append(cand)
    return out


def _vision_config(path: str | os.PathLike) -> dict:
    """The multimodal config's ``vision_config`` (empty when there is none)."""
    p = _as_dir(path)
    for cand in (p / "config.json", p / "text_encoder" / "config.json"):
        if cand.is_file():
            try:
                with open(cand, encoding="utf-8") as f:
                    cfg = json.load(f)
            except (OSError, ValueError):
                continue
            vision = cfg.get("vision_config")
            if isinstance(vision, dict):
                return vision
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# small torch helpers
# ─────────────────────────────────────────────────────────────────────────────


def _module_device(module: nn.Module, fallback: Any = None) -> torch.device:
    """The device of ``module``'s first parameter (``fallback`` when it has none)."""
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(fallback if fallback is not None else "cpu")


def _module_dtype(module: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    """The dtype of ``module``'s first parameter (``fallback`` when it has none)."""
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration):
        return fallback


def patchify(z: torch.Tensor) -> torch.Tensor:
    """``[B, C, 2h, 2w] -> [B, 4C, h, w]``; channel = ``c * 4 + i * 2 + j``.

    Same folding as the reference ``encoders/vae/__init__.py::patchify``.
    """
    b, c, hh, ww = z.shape
    z = z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return z.reshape(b, c * 4, hh // 2, ww // 2)


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`patchify` (``[B, 4C, h, w] -> [B, C, 2h, 2w]``)."""
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(b, c // 4, 2 * h, 2 * w)


def _bn(vae: nn.Module) -> nn.Module:
    bn = getattr(vae, "bn", None)
    if bn is None or getattr(bn, "running_mean", None) is None:
        raise ValueError(
            "MOSS expects a FLUX.2 VAE (AutoencoderKLFlux2): the module has no "
            "`bn` running statistics to normalise latents with."
        )
    return bn


def latent_stats(vae: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """The MOSS latent normalisation, ``(mean, std)`` of shape ``(1, 128, 1, 1)``.

    Encode is ``(patchify(z) - mean) / std`` and decode the exact inverse; std is
    ``sqrt(running_var + batch_norm_eps)`` with ``eps = vae.config.batch_norm_eps``
    (``0.0001`` in the MOSS ``vae/config.json``).  Provenance:
    ``ref/KohakuMossTrainer/src/kohakumoss/encoders/vae/__init__.py:123-128``
    (the BN buffers are read before any cast and kept in float32) and
    ``:187-202`` (encode / decode order); the official MOSS pipeline applies the
    same affine inverse at ``pipeline_moss_image.py:432-436``.

    There is **no** scalar ``scaling_factor`` / ``shift_factor`` for this VAE -
    the normalisation is the VAE's BatchNorm, one value per *patchified* channel.
    """
    bn = _bn(vae)
    eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
    mean = bn.running_mean.detach().float().cpu().view(1, -1, 1, 1)
    std = (bn.running_var.detach().float().cpu() + eps).sqrt().view(1, -1, 1, 1)
    if mean.shape[1] != LATENT_CHANNELS:
        logger.warning(
            "MOSS latent stats have %d channels, expected %d",
            int(mean.shape[1]),
            LATENT_CHANNELS,
        )
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# loading
# ─────────────────────────────────────────────────────────────────────────────


def load_text_encoder(
    path: str, dtype: torch.dtype = torch.bfloat16, device: str | None = None
) -> nn.Module:
    """Load Qwen3.5-9B (``Qwen3_5Model``: vision tower + text decoder), frozen.

    ``path`` may be the pipeline root or its ``text_encoder/`` subdirectory.  The
    class is whatever ``AutoModel`` maps ``Qwen3_5Config`` to - with transformers
    5.2 / ``model_type: qwen3_5`` that is ``transformers.models.qwen3_5.
    modeling_qwen3_5.Qwen3_5Model`` (i.e. ``model.visual`` + ``model.
    language_model``), the same class the reference builds (``text/load.py
    ::meta_text``) and the official pipeline loads (``AutoModel.from_pretrained``,
    ``pipeline_moss_image.py:308``).

    The returned module is in ``eval()`` mode with gradients off; ``device`` is
    optional (UTrainer moves components itself).
    """
    from transformers import AutoModel

    model_dir = _component_dir(path, "text_encoder")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"MOSS text encoder: no config.json under {model_dir} - pass the "
            "MOSS-Image pipeline root or its text_encoder/ directory"
        )

    kwargs: dict[str, Any] = {"dtype": dtype}
    try:
        model = AutoModel.from_pretrained(str(model_dir), attn_implementation="sdpa", **kwargs)
    except (TypeError, ValueError) as e:  # older/newer arg spelling
        logger.warning("MOSS text encoder: falling back to the default attention "
                       "implementation (%s)", e)
        model = AutoModel.from_pretrained(str(model_dir), **kwargs)

    model = model.eval().requires_grad_(False)
    if device is not None:
        model = model.to(device)
    logger.info(
        "MOSS text encoder: %s from %s (%s, %.2fB params%s)",
        type(model).__name__,
        model_dir,
        dtype,
        sum(p.numel() for p in model.parameters()) / 1e9,
        ", vision tower present" if hasattr(model, "visual") else ", TEXT-ONLY",
    )
    return model


def load_tokenizer(path: str):
    """Load the Qwen3.5 tokenizer (``Qwen2Tokenizer``) for ``path``.

    ``path`` may be the pipeline root, its ``text_encoder/`` or its ``tokenizer/``
    directory; the chat template ships as ``chat_template.jinja`` next to the
    tokenizer files and is picked up by ``AutoTokenizer``.
    """
    from transformers import AutoTokenizer

    tok_dir = _tokenizer_dir(path)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    logger.info("MOSS tokenizer: %s from %s", type(tokenizer).__name__, tok_dir)
    return tokenizer


def _fast_image_processor_cls():
    """The fast Qwen2-VL image processor when torchvision is available, else the slow one."""
    try:
        from transformers import Qwen2VLImageProcessorFast

        return Qwen2VLImageProcessorFast
    except Exception:  # noqa: BLE001 - optional accelerate/torchvision path
        from transformers import Qwen2VLImageProcessor

        return Qwen2VLImageProcessor


def _build_processor(path: str):
    """Construct a ``Qwen3VLProcessor`` when the model dir ships no preprocessor config.

    The MOSS HF directory has no ``preprocessor_config.json`` (the reference loads
    no image processor at all), so the image processor is built from the text
    encoder's ``vision_config`` (patch 16 / merge 2 / temporal 2) plus Qwen3.5's
    normalisation constants, and the tokenizer is taken from the sibling
    ``tokenizer/`` directory.
    """
    from transformers import Qwen3VLProcessor

    vision = _vision_config(path)
    patch_size = int(vision.get("patch_size", 16))
    merge_size = int(vision.get("spatial_merge_size", 2))
    temporal_patch_size = int(vision.get("temporal_patch_size", 2))

    tokenizer = load_tokenizer(path)
    if tokenizer is None:
        return None

    cls = _fast_image_processor_cls()
    image_processor = cls(
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        image_mean=_IMAGE_MEAN,
        image_std=_IMAGE_STD,
        size=dict(_IMAGE_SIZE),
    )

    video_processor = None
    try:
        from transformers import Qwen3VLVideoProcessor

        video_processor = Qwen3VLVideoProcessor()
    except Exception as e:  # noqa: BLE001 - video processing is unused here
        logger.debug("MOSS processor: no video processor (%s)", e)

    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
    )
    logger.info(
        "MOSS processor: built Qwen3VLProcessor manually (%s, patch=%d merge=%d) - "
        "no preprocessor_config.json under %s",
        type(image_processor).__name__,
        patch_size,
        merge_size,
        _as_dir(path),
    )
    return processor


def load_processor(path: str):
    """Load the multimodal processor, or ``None`` when it cannot be built.

    Preference order:

    1. a real ``preprocessor_config.json`` (UTrainer's ``text_processor_path``
       pointing at a full Qwen3.5 checkout, or any pipeline root that ships one);
    2. a manually constructed ``Qwen3VLProcessor`` from the vision config + the
       sibling tokenizer (MOSS's own directory has no preprocessor config).

    ``None`` is a supported outcome: ``encode_prompt`` then degrades to text-only
    and logs a warning instead of crashing.
    """
    from transformers import AutoProcessor

    for cand in _processor_candidates(path):
        if (cand / "preprocessor_config.json").is_file():
            try:
                processor = AutoProcessor.from_pretrained(str(cand))
                logger.info(
                    "MOSS processor: %s from %s", type(processor).__name__, cand
                )
                return processor
            except Exception as e:  # noqa: BLE001 - fall through to the manual build
                logger.warning("MOSS processor: AutoProcessor failed on %s: %s", cand, e)

    try:
        return _build_processor(path)
    except Exception as e:  # noqa: BLE001 - text-only training must still work
        logger.warning(
            "MOSS processor: could not build one from %s (%s); reference images "
            "will be dropped (text-only encoding)",
            _as_dir(path),
            e,
        )
        return None


def load_vae(
    path: str, dtype: torch.dtype = torch.float32, device: str | None = None
) -> nn.Module:
    """Load the FLUX.2 VAE (``diffusers.AutoencoderKLFlux2``) for MOSS latents.

    Loaded in float32 first - the latent normalisation's BatchNorm statistics are
    read from those buffers and are kept in float32 even when the VAE itself is
    cast to bf16 (the reference ``FluxVae`` does the same: it clones ``mean`` /
    ``std`` as float32 before ``vae.to(dtype)``).  ``bn`` is never called in a
    forward pass; it is a statistics holder.

    ``path`` may be the pipeline root or its ``vae/`` subdirectory.
    """
    from diffusers import AutoencoderKLFlux2

    vae_dir = _component_dir(path, "vae")
    if not (vae_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"MOSS VAE: no config.json under {vae_dir} - pass the MOSS-Image "
            "pipeline root or its vae/ directory"
        )
    try:
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
    except TypeError:  # newer diffusers spells it `dtype`
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_dir), dtype=torch.float32)

    vae = vae.eval().requires_grad_(False)
    vae = vae.to(dtype)
    _restore_bn_dtype(vae)
    if device is not None:
        vae = vae.to(device)
    logger.info(
        "MOSS VAE: %s from %s (%s, %d latent ch, %s)",
        type(vae).__name__,
        vae_dir,
        dtype,
        int(getattr(vae.config, "latent_channels", VAE_CHANNELS)),
        "force_upcast" if getattr(vae.config, "force_upcast", False) else "no force_upcast",
    )
    return vae


def _restore_bn_dtype(vae: nn.Module) -> None:
    """Keep ``vae.bn``'s running statistics in float32 after a dtype cast."""
    bn = getattr(vae, "bn", None)
    if bn is None:
        return
    for name in ("running_mean", "running_var"):
        buf = getattr(bn, name, None)
        if isinstance(buf, torch.Tensor):
            setattr(bn, name, buf.detach().float())


# ─────────────────────────────────────────────────────────────────────────────
# latent encode / decode
# ─────────────────────────────────────────────────────────────────────────────


def encode_latent(vae: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """``(B, 3, H, W)`` in ``[-1, 1]`` -> ``(B, 128, H/16, W/16)`` float32 on CPU.

    The reference ``FluxVae.encode`` (posterior mode, so encoding is
    deterministic), then MOSS's 2x2 patchify and the BatchNorm normalisation::

        z = vae.encode(pixels).latent_dist.mode()      # (B, 32, H/8, W/8)
        z = patchify(z)                                # (B, 128, H/16, W/16)
        z = (z - running_mean) / sqrt(running_var + eps)

    ``mode()`` (not ``sample()``) is what the reference caches and what the
    official pipeline's decode inverts, and it makes the cache reproducible.
    """
    if image_tensor.ndim != 4:
        raise ValueError(
            f"MOSS encode_latent expects (B, 3, H, W), got {tuple(image_tensor.shape)}"
        )
    _, channels, height, width = image_tensor.shape
    if channels != 3:
        raise ValueError(f"MOSS encode_latent expects 3 channels, got {channels}")
    factor = 8 * LATENT_PATCH
    if height % factor or width % factor:
        raise ValueError(
            f"MOSS encode_latent needs H and W divisible by {factor} (VAE /8, then "
            f"2x2 patchify); got {height}x{width}"
        )

    device = _module_device(vae)
    dtype = _module_dtype(vae)
    with torch.no_grad():
        pixels = image_tensor.to(device=device, dtype=dtype)
        posterior = vae.encode(pixels).latent_dist
        z = parse_posterior(posterior)
        packed = patchify(z.float())
        mean, std = latent_stats(vae)
        out = (packed.cpu() - mean) / std
    return out.to(torch.float32).contiguous()


def parse_posterior(posterior: Any) -> torch.Tensor:
    """The deterministic latent of a VAE posterior (``mode``), without sampling."""
    mode = getattr(posterior, "mode", None)
    if callable(mode):
        return mode()
    if isinstance(posterior, torch.Tensor):
        return posterior
    raise TypeError(f"MOSS: cannot read a latent from {type(posterior).__name__}")


def decode_latent(vae: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Cached latent -> pixels ``(B, 3, 16h, 16w)`` in ``[-1, 1]``, float32 on CPU.

    Accepts the cached ``(B, 128, h, w)`` form (undo the BatchNorm, then
    unpatchify - the exact inverse of :func:`encode_latent`) or a raw
    ``(B, 32, h, w)`` VAE-space latent, which is decoded as-is.  The return is a
    tensor, matching every other UTrainer adapter (``engine/trainer.py`` does
    ``(image / 2 + 0.5).clamp(0, 1)`` on it); :func:`pixels_to_pil` converts it.
    """
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    if latent.ndim != 4:
        raise ValueError(f"MOSS decode_latent expects 4D latents, got {tuple(latent.shape)}")

    device = _module_device(vae)
    dtype = _module_dtype(vae)
    with torch.no_grad():
        z = latent.to(device=device, dtype=torch.float32)
        if z.shape[1] == LATENT_CHANNELS:
            mean, std = latent_stats(vae)
            z = z * std.to(z.device) + mean.to(z.device)
            z = unpatchify(z)
        elif z.shape[1] == VAE_CHANNELS:
            pass
        else:
            raise ValueError(
                f"MOSS decode_latent expects {LATENT_CHANNELS} (cached) or "
                f"{VAE_CHANNELS} (raw VAE) channels, got {z.shape[1]}"
            )
        pixels = vae.decode(z.to(dtype=dtype)).sample
    return pixels.detach().float().clamp(-1.0, 1.0).cpu().contiguous()


def pixels_to_pil(pixels: torch.Tensor) -> List[Image.Image]:
    """``(B, 3, H, W)`` / ``(3, H, W)`` in ``[-1, 1]`` -> a list of RGB PIL images."""
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    array = ((pixels.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    array = array.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a, mode="RGB") for a in array]


# ─────────────────────────────────────────────────────────────────────────────
# text / multimodal encoding
# ─────────────────────────────────────────────────────────────────────────────


def _as_image_list(reference_image: Any) -> List[Any]:
    """Normalise ``reference_image`` to a list of PIL images (empty when None)."""
    if reference_image is None:
        return []
    if isinstance(reference_image, (list, tuple)):
        return [im for im in reference_image if im is not None]
    if isinstance(reference_image, torch.Tensor):
        return pixels_to_pil(reference_image)
    return [reference_image]


def chat_text(tokenizer: Any, content: str) -> str:
    """The training-time chat template of the reference (``tokens.chat_text``).

    ``[{"role": "user", "content": ...}]``, ``add_generation_prompt=True``,
    ``enable_thinking=True``, **no** system prompt - which renders to
    ``<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``.
    The whole rendered string is encoded; the reference strips no prefix.
    """
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        # A tokenizer whose template has no `enable_thinking` switch (e.g. Qwen2).
        logger.warning("MOSS: chat template has no enable_thinking switch; using defaults")
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError) as e:
        # No chat template at all: encode the raw caption (the adapter stays usable).
        logger.warning("MOSS: no chat template (%s); encoding the raw prompt", e)
        return content


def _image_token_id(text_encoder: nn.Module, tokenizer: Any) -> Optional[int]:
    """``<|image_pad|>``'s id - the model config first, the tokenizer second."""
    token_id = getattr(getattr(text_encoder, "config", None), "image_token_id", None)
    if token_id is None:
        try:
            token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:  # noqa: BLE001 - tokenizer without the special token
            return None
    if token_id is None or int(token_id) < 0:
        return None
    return int(token_id)


def _reset_rope_deltas(text_encoder: nn.Module) -> None:
    """Drop the mRoPE deltas cached on ``Qwen3_5Model`` by a previous call.

    ``Qwen3_5Model.compute_3d_position_ids`` only rebuilds positions when
    ``pixel_values`` are passed; a text-only call afterwards reuses the *previous*
    image's ``rope_deltas`` and silently shifts every position id.  Resetting per
    call keeps text-only encoding bit-identical to the reference's text-only
    forward and keeps repeated encodes deterministic.
    """
    if hasattr(text_encoder, "rope_deltas"):
        text_encoder.rope_deltas = None


def encode_prompt(
    text_encoder: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    processor: Any = None,
    reference_image: Any = None,
) -> dict:
    """Caption (+ optional reference images) -> the DiT's conditioning tensor.

    Layout (identical to the reference's text path - ``text/tokens.py::chat_text``
    + ``text/forward.py::HfForward`` / ``KernelForward``):

    * the prompt is rendered through the training chat template (thinking on, no
      system prompt) and **nothing is stripped**;
    * the conditioning is the **last hidden state**, i.e.
      ``Qwen3_5TextModel``'s final zero-centred RMSNorm output, shape
      ``(L, 4096)`` - the reference taps no other layer (``KernelForward`` ends
      with ``self.ops.norm(h, self.text.norm)``, which is what HF's
      ``last_hidden_state`` already contains);
    * text-only rows are truncated at :data:`MAX_TEXT_LENGTH` (4096) like
      ``tokens.token_ids``.

    With ``reference_image`` **and** ``processor`` the images are embedded as
    ``Picture N: <|vision_start|><|image_pad|><|vision_end|>`` in the user message,
    the processor expands each placeholder to the tower's token count, and
    ``Qwen3_5Model`` fills those positions from ``model.visual``.  Multimodal rows
    are never truncated (truncation would desync the image pads).  With
    ``reference_image`` but no processor the call degrades to text-only with a
    warning.

    Returns
    -------
    dict
        ``prompt_embed``       - ``(L, 4096)`` CPU float32
        ``prompt_embeds_mask`` - ``(L,)`` bool, all True (rows are unpadded)
        ``image_token_mask``   - ``(L,)`` bool, only when reference images ran
    """
    if text_encoder is None or tokenizer is None:
        raise RuntimeError(
            "MOSS requires a loaded Qwen3.5 text encoder and tokenizer "
            "(load_text_encoder / load_tokenizer)."
        )

    refs = _as_image_list(reference_image)
    if refs and processor is None:
        logger.warning(
            "MOSS: %d reference image(s) given but no processor - encoding text "
            "only. Set text_processor_path to a Qwen3.5 checkout (or keep the "
            "sibling tokenizer/ directory) to enable vision tokens.",
            len(refs),
        )
        refs = []
    if refs and not hasattr(text_encoder, "visual"):
        logger.warning(
            "MOSS: the text encoder has no vision tower (text-only checkpoint) - "
            "dropping %d reference image(s).",
            len(refs),
        )
        refs = []
    multimodal = bool(refs)

    content = "".join(_IMG_PLACEHOLDER.format(i + 1) for i in range(len(refs))) + prompt
    text = chat_text(tokenizer, content)

    target_device = _module_device(text_encoder, device)
    if multimodal:
        try:
            inputs = processor(
                text=[text],
                images=refs,
                return_tensors="pt",
                # transformers >= 5.14 requires M-RoPE token types whenever
                # pixel_values/image_grid_thw are passed; older processors
                # reject the flag (handled below).
                return_mm_token_type_ids=True,
            )
        except TypeError:
            # a Qwen-VL processor that does not know `return_mm_token_type_ids`
            inputs = processor(text=[text], images=refs, return_tensors="pt")
        input_ids = inputs["input_ids"].to(target_device)
        attention_mask = inputs["attention_mask"].to(target_device).bool()
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": inputs["pixel_values"].to(target_device),
            "image_grid_thw": inputs["image_grid_thw"].to(target_device),
        }
        if "mm_token_type_ids" in inputs:
            forward_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(target_device)
    else:
        # No truncation: the full caption is encoded at its own length. Cached
        # embeddings keep per-sample lengths; the collate pads with zeros + mask.
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded["attention_mask"].to(target_device).bool()
        forward_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

    _reset_rope_deltas(text_encoder)
    with torch.no_grad():
        outputs = text_encoder(**forward_kwargs, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state

    if hidden.shape[0] != 1:
        raise RuntimeError(f"MOSS encode_prompt expects one row, got {hidden.shape[0]}")
    if hidden.shape[1] != input_ids.shape[1]:
        # The base model never drops positions; a mismatch means the wrapper
        # trimmed the suffix (Qwen3-VL's ForConditionalGeneration does that).
        logger.warning(
            "MOSS: hidden states %d != input tokens %d - trimming the masks",
            hidden.shape[1],
            input_ids.shape[1],
        )
    length = hidden.shape[1]
    # `dtype` is the adapter contract's compute dtype (UTrainer passes float32);
    # the cache contract is CPU float32 regardless (it is quantised to int8 next).
    embed = hidden[0].detach().to(torch.float32).cpu().contiguous()
    if embed.shape[-1] != TEXT_DIM:
        logger.warning(
            "MOSS: text encoder width %d != text_dim %d", embed.shape[-1], TEXT_DIM
        )

    result: dict[str, Any] = {
        "prompt_embed": embed,
        "prompt_embeds_mask": attention_mask[0, :length].to(torch.bool).cpu(),
    }
    if multimodal:
        token_id = _image_token_id(text_encoder, tokenizer)
        if token_id is None:
            logger.warning("MOSS: no <|image_pad|> id - image_token_mask is empty")
            result["image_token_mask"] = torch.zeros(length, dtype=torch.bool)
        else:
            result["image_token_mask"] = (
                input_ids[0, :length].detach().cpu() == token_id
            )
    return result
