"""MiniMax-H3-Image adapter — LoRA/LoKr training on the image DiT.

MiniMax-H3-Image is MiniMax-H3's transformer without the audio stream: one packed 1-D
sequence holds ``[text | reference rows | target rows]``, attention is full self-attention
over that sequence, and the checkpoint carries one output head per step of its 6-step
Parallel-Decoding-Distillation (PDD) schedule.

Everything the training path needs lives in this package:

* ``transformer_minimaxh3image.py`` — vendored DiT (training copy, offloader hooks),
* ``vae_minimaxh3image.py``          — vendored image VAE (24ch, 16x spatial, ViT decoder),
* ``packing.py``                     — packed-sequence geometry, ported verbatim from the
                                       released sampler,
* ``convert.py``                     — original-checkpoint -> vendored-classes key mapping
                                       (includes the PDD head split).

Training-side conventions, all inherited from the released sampler:

* the transformer predicts a **data-ward** velocity (``v = x0 - x_t``, ``velocity_sign =
  "data_ward"``), and consumes the timestep as ``t = 1 - sigma`` in ``[0, 1]``;
* training uses ONE fixed output head (``training.pdd_head_index``, default 0) and a
  continuous sigma drawn by the shared sampler -- single-prediction training, not the
  released per-step PDD schedule; validation runs a plain N-step Euler loop;
* latents are normalized per channel with the VAE's ``latents_mean`` / ``latents_std``,
  and pixels follow the VAE's ImageNet convention over a ``[0, 1]`` base range.

Scope of this adapter (milestone 1): text-to-image training, LoRA and LoKr, with any
UTrainer loss (flow_matching with weighting/freq/FFT, guide_flow_matching, l2_reg, ...).
Reference-image (edit) conditioning is wired through the same ``token_tags`` /
``image_indices`` layout but is not enabled in the example config yet.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter, sample_sigma_uniform
from UnifiedTrainer.registry import ModelRegistry

from . import convert as _convert
from . import packing as _packing

logger = logging.getLogger(__name__)

# Bucket tiers (32-divisible: vae_scale 16 x spatial patch 2).
RESOLUTION_CONFIG = {
    768: [
        (768, 768),
        (768, 832), (768, 896), (768, 960), (768, 1024),
        (768, 1088), (768, 1152), (768, 1216), (768, 1280), (768, 1344),
        (832, 768), (896, 768), (960, 768), (1024, 768),
        (1088, 768), (1152, 768), (1216, 768), (1280, 768), (1344, 768),
    ],
    1024: [
        (1024, 1024),
        (1024, 1152), (1024, 1280), (1024, 1408), (1024, 1536),
        (1152, 1024), (1280, 1024), (1408, 1024), (1536, 1024),
    ],
    1280: [
        (1280, 1280),
        (1280, 1408), (1280, 1536), (1280, 1664), (1280, 1792),
        (1408, 1280), (1536, 1280), (1664, 1280), (1792, 1280),
    ],
}

# ImageNet pixel normalization of the released VAE (same constants as MiniMax-H3's video VAE).
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# Qwen3-VL decoder layer the conditioner is read from.
TEXT_ENCODER_LAYER = 50


@ModelRegistry.register("minimax_h3_image")
class MiniMaxH3ImageAdapter(BaseModelAdapter):
    """Adapter for the MiniMax-H3-Image DiT (text-to-image LoRA/LoKr training)."""

    name = "minimax_h3_image"
    # Spatial patch of the transformer (the temporal patch is 1: an image is one frame).
    patch_size = 2

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")
        self._num_pdd_heads: Optional[int] = None
        # Single-prediction training: one fixed output head for every forward. The
        # checkpoint still carries all six PDD heads as weights.
        # Optional startup trace on the shared volume: one line per stage, so a lost
        # instance can still be diagnosed from any other box.
        self._trace_path = (config.get("training") or {}).get("startup_trace")
        head = (config.get("training") or {}).get("pdd_head_index", "base")
        # "base" uses the checkpoint's plain final_layer.video_out head (ordinary flow
        # matching); an integer pins one of the six distilled PDD heads; "sigma" selects the
        # head the way the released sampler does -- one head per step of the PDD grid
        # (see :meth:`pdd_head_for_sigma`).
        if isinstance(head, str) and head.lower() in ("sigma", "auto", "per_sigma"):
            self._head_mode = "sigma"
            self._head_index = None
        elif head in (None, "base", "none"):
            self._head_mode = "fixed"
            self._head_index = None
        else:
            self._head_mode = "fixed"
            self._head_index = int(head)

    # ── PDD output heads ───────────────────────────────────────────────

    @staticmethod
    def pdd_head_for_sigma(sigma: float) -> int:
        """Head index of the PDD interval that contains ``sigma``.

        The release distils one head per step of ``pdd_config.json``'s 6-step grid
        [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0]. ComfyUI's ``_pdd_head``
        (``comfy/ldm/minimax/model.py:370``) reads the fused bank as
        ``row_block[0] + dt-weighted mean of the offset rows it spans``; for a single
        grid interval that is ``pdd[0] + pdd[i] - pdd[0] == pdd[i]``, i.e. exactly the
        i-th block of ``pdd_fused_heads.safetensors`` -- which is what this model keeps
        as ``pdd_proj_out[i]``. Interval 0 is the highest-noise one, so a training step at
        sigma ~1 uses head 0 and a step near sigma ~0 uses head 5.
        """
        grid = _packing.DEFAULT_PDD_SIGMAS
        value = float(sigma)
        for k in range(len(grid) - 1):
            if value >= grid[k + 1]:
                return k
        return len(grid) - 2

    def _resolve_head(self, sigma: float) -> Optional[int]:
        """``pdd_step_index`` for one forward (``None`` = the checkpoint's base head)."""
        if self._head_mode == "sigma":
            return self.pdd_head_for_sigma(sigma)
        return self._head_index

    # ── Loading ────────────────────────────────────────────────────────

    def ensure_empty_embedding(self, text_encoder, tokenizer) -> bool:
        """Write ``empty_embedding.<suffix>.npz`` if it is missing (same call as captions)."""
        cache_dir = (self.config.get("data") or {}).get("cache_dir")
        if not cache_dir or text_encoder is None or tokenizer is None:
            return False
        target = os.path.join(cache_dir, f"empty_embedding.{self.empty_embedding_suffix}.npz")
        if os.path.exists(target):
            return True
        try:
            embedding = self.encode_text(
                text_encoder, tokenizer, " ", torch.device("cuda"), torch.float32
            )
            from UnifiedTrainer.data.embedding_cache import EmbeddingCache

            EmbeddingCache.save(target, embedding)
            self._trace(f"empty_embedding: written {target}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._trace(f"empty_embedding: FAILED {type(exc).__name__}: {exc}")
            return False

    def release_text_encoder(self) -> int:
        """Drop the packed NVFP4 text-encoder weights (plain attributes, ~16 GB in RAM).

        The cache builder deletes the module, but the packed stacks are held by the lazy
        linear layers; clearing them explicitly is what actually returns the memory before
        the 62 GB DiT build.
        """
        model = self.__dict__.get("_te_model")
        tokenizer = self.__dict__.get("_te_tokenizer")
        if tokenizer is None and self.__dict__.get("_te_model") is not None:
            try:
                tok_path = self.config.get("tokenizer_path") or self._model_path
                tokenizer = self.load_tokenizer(tok_path)
            except Exception:  # noqa: BLE001
                tokenizer = None
        if model is not None and tokenizer is not None:
            self.ensure_empty_embedding(model, tokenizer)
        freed = 0
        for module in self.__dict__.get("_te_modules", []):
            for attr in ("packed", "block_scale", "codebook", "s_channel", "s_rel"):
                if hasattr(module, attr):
                    setattr(module, attr, None)
                    freed += 1
        self._te_modules = []
        self._te_model = None
        import gc

        gc.collect()
        self._trace(f"text_encoder: released ({freed} packed tensors dropped)")
        return freed

    def _trace(self, message: str) -> None:
        """Append one stage line to the configured startup trace (no-op when unset)."""
        if not self._trace_path:
            return
        try:
            os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)
            with open(self._trace_path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        except OSError:  # tracing must never break training
            pass

    def load_tokenizer(self, path: str):
        """Qwen2 tokenizer shipped next to the text encoder (needed for encode_text(""))."""
        from transformers import AutoTokenizer

        root = _convert.checkpoint_root(path)
        tok_dir = os.path.join(root, "tokenizer")
        if not os.path.isdir(tok_dir):
            tok_dir = path
        self._trace(f"tokenizer: {tok_dir}")
        return AutoTokenizer.from_pretrained(tok_dir)

    def load_processor(self, path: str, tokenizer_path: str = ""):
        """Qwen3-VL processor (needed by the cache builder's empty/suffix embeddings)."""
        from transformers import AutoProcessor

        root = _convert.checkpoint_root(path)
        proc_dir = os.path.join(root, "processor")
        if not os.path.isdir(proc_dir):
            proc_dir = path
        self._trace(f"processor: {proc_dir}")
        try:
            return AutoProcessor.from_pretrained(proc_dir)
        except Exception:  # noqa: BLE001 - processor is optional for text-only captions
            logger.warning("MiniMax-H3-Image: no processor at %s", proc_dir)
            return None

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        """Build the vendored DiT and stream the 62 GB checkpoint into it shard by shard.

        Materializing the whole converted state dict first needs ~62 GB of RAM on top of the
        24 GB model, which does not fit next to a 24 GB GPU box. Streaming keeps the peak at
        model + one 5.5 GB shard.
        """
        from .transformer_minimaxh3image import MiniMaxH3ImageTransformer2DModel
        from . import quantize as _quantize

        path = _convert.checkpoint_root(path)
        cfg = _convert.transformer_config(path)
        training_cfg = self.config.get("training") or {}
        quantize_mode = training_cfg.get("quantize")
        self._trace(f"transformer: start (quantize={quantize_mode})")

        # fp8 (torch._scaled_mm, sm89+): 33 GB of weights, so the whole 31B DiT fits next to
        # 1024² activations on a 48 GB card but NOT at 4 MP.
        #
        # No snapshot directory is invented for fp8: the base is quantized from the bf16 shards
        # on every start (~4 min, one shard in RAM). A snapshot is written ONLY when the config
        # explicitly names ``training.quantize_path`` AND sets ``save_prequantized: true`` --
        # an fp8 run must not leave a 33 GB artifact behind that nobody asked for.
        if quantize_mode in ("fp8", "float8", "fp8_e4m3"):
            cache_dir = training_cfg.get("quantize_path") or None
            if cache_dir and _quantize.is_fp8_available(cache_dir):
                model = _quantize.build_meta_model_fp8(MiniMaxH3ImageTransformer2DModel, cfg)
                info = _quantize.load_fp8_into(model, cache_dir, device="cuda")
                logger.info(
                    "MiniMax-H3-Image: loaded fp8 snapshot %s (%d fp8 modules, %.1f GB)",
                    cache_dir, info["manifest"].get("fp8_modules", -1),
                    info["manifest"].get("tensor_nbytes", 0) / 1024**3,
                )
                self._num_pdd_heads = int(cfg["num_pdd_heads"])
                return model
            logger.info(
                "MiniMax-H3-Image: fp8 base from the original bf16 shards (streaming, one "
                "shard in RAM at a time)%s",
                f"; snapshot dir {cache_dir}" if cache_dir else "; no snapshot dir, nothing "
                "is written to disk",
            )
            model = _quantize.build_meta_model_fp8(MiniMaxH3ImageTransformer2DModel, cfg)
            placed = _quantize.load_streaming_fp8(model, path, cfg, device="cuda")
            logger.info("MiniMax-H3-Image: %d tensors quantized to fp8 on the GPU", placed)
            if cache_dir and training_cfg.get("save_prequantized"):
                try:
                    out = _quantize.save_fp8(model, cache_dir)
                    logger.info("MiniMax-H3-Image: fp8 snapshot written to %s", out)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MiniMax-H3-Image: could not write fp8 snapshot: %s", exc)
            self._num_pdd_heads = int(cfg["num_pdd_heads"])
            return model

        # Pre-quantized nf4 fast path: building the 31B model in bf16 costs 62 GB of RAM and
        # re-quantizing every start costs minutes. ``training.quantize = "prequantized"``
        # loads a previously saved nf4 snapshot (16 GB) straight onto the GPU instead.
        if training_cfg.get("quantize") == "prequantized":
            cache_dir = training_cfg.get("quantize_path") or os.path.join(path, "nf4")
            if _quantize.is_available(cache_dir):
                model = _quantize.build_meta_model(MiniMaxH3ImageTransformer2DModel, cfg)
                info = _quantize.load_nf4_into(model, cache_dir, device="cuda")
                logger.info(
                    "MiniMax-H3-Image: loaded pre-quantized nf4 snapshot (%d quantized modules, %.1f GB)",
                    info["manifest"].get("quantized_modules", -1),
                    info["manifest"].get("tensor_nbytes", 0) / 1024**3,
                )
                self._num_pdd_heads = int(cfg["num_pdd_heads"])
                return model
            logger.warning(
                "MiniMax-H3-Image: no nf4 snapshot at %s -- quantizing from the original "
                "checkpoint (streaming, one shard in RAM at a time)", cache_dir,
            )
            model = _quantize.build_meta_model(MiniMaxH3ImageTransformer2DModel, cfg)
            placed = _quantize.load_streaming_quantized(model, path, cfg, device="cuda")
            logger.info("MiniMax-H3-Image: %d tensors quantized onto the GPU", placed)
            if (self.config.get("training") or {}).get("save_prequantized", True):
                try:
                    out = _quantize.save_nf4(model, cache_dir, dtype)
                    logger.info("MiniMax-H3-Image: nf4 snapshot written to %s", out)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MiniMax-H3-Image: could not write nf4 snapshot: %s", exc)
            self._num_pdd_heads = int(cfg["num_pdd_heads"])
            return model
        # The DiT is ~31B parameters (the AdaLN projections alone are 13B), so building it in
        # the default fp32 would allocate ~124 GB and get OOM-killed. Construct straight in
        # the target dtype: bf16 is 62 GB, and the nf4 pass that follows keeps it there.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = MiniMaxH3ImageTransformer2DModel(**cfg)
        finally:
            torch.set_default_dtype(previous_dtype)
        logger.info(
            "MiniMax-H3-Image transformer built in %s: %.1fB params (%.1f GB)",
            dtype, sum(p.numel() for p in model.parameters()) / 1e9,
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
        )

        assigned: set[str] = set()
        unexpected: list[str] = []
        for part in _convert.iter_transformer_state_dicts(path, cfg):
            result = model.load_state_dict(part, strict=False)
            unexpected.extend(result.unexpected_keys)
            assigned.update(part.keys())
            del part, result

        missing = [k for k in model.state_dict() if k not in assigned]
        if missing:
            raise RuntimeError(
                f"MiniMax-H3-Image transformer: {len(missing)} keys never loaded, e.g. {missing[:6]}"
            )
        if unexpected:
            logger.warning(
                "MiniMax-H3-Image transformer: %d unexpected keys (ignored), e.g. %s",
                len(unexpected), unexpected[:4],
            )
        self._num_pdd_heads = int(cfg["num_pdd_heads"])
        self._last_loaded_transformer = model  # train.py's nf4 branch quantizes this in place
        logger.info(
            "MiniMax-H3-Image transformer loaded: %.2fB params, %d PDD heads, dtype %s",
            sum(p.numel() for p in model.parameters()) / 1e9, self._num_pdd_heads, dtype,
        )
        return model.to(dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from .vae_minimaxh3image import AutoencoderKLMiniMaxH3Image

        self._trace("vae: start")
        state, cfg = _convert.load_vae_state_dict(_convert.checkpoint_root(path))
        model = AutoencoderKLMiniMaxH3Image(**cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"MiniMax-H3-Image VAE: missing keys {missing[:6]}")
        if unexpected:
            logger.warning("MiniMax-H3-Image VAE: %d unexpected keys, e.g. %s", len(unexpected), unexpected[:4])
        return model.to(dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        # Single-prediction training: a plain linear sigma schedule (no PDD grid).
        steps = int((self.config.get("validation") or {}).get("num_inference_steps", 6))
        return {"sigmas": [1.0 - i / steps for i in range(steps + 1)]}

    @property
    def latent_channels(self) -> int:
        return 24

    @property
    def vae_scale_factor(self) -> int:
        return 16

    @property
    def bucket_divisibility(self) -> int:
        return self.vae_scale_factor * self.patch_size

    @property
    def embedding_dim(self) -> int:
        return 5120

    @property
    def empty_embedding_suffix(self) -> str:
        return "minimax_h3_image"

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return True

    @property
    def velocity_sign(self) -> str:
        return "data_ward"

    @property
    def encode_text_accepts_image(self) -> bool:
        return False

    # ── Normalization helpers ──────────────────────────────────────────

    @staticmethod
    def _latents_normalization(vae: nn.Module, device, dtype):
        cfg = getattr(vae, "config", None)
        mean_vals = getattr(cfg, "latents_mean", None) if cfg is not None else None
        std_vals = getattr(cfg, "latents_std", None) if cfg is not None else None
        mean = torch.tensor(list(mean_vals or [0.0] * 24), device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor(list(std_vals or [1.0] * 24), device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean, std

    @staticmethod
    def _pixel_normalization(device, dtype):
        mean = torch.tensor(PIXEL_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor(PIXEL_STD, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean, std

    # ── Encoding ───────────────────────────────────────────────────────

    def encode_image(self, vae: nn.Module, image: torch.Tensor) -> dict:
        """Encode one image to a normalized ``(24, H/16, W/16)`` latent.

        ``image`` is the cache pipeline's convention: ``(1, 3, H, W)`` diffusion-normalized
        to ``[-1, 1]`` (4D ``[0, 1]`` and 5D ``(1, 3, 1, H, W)`` are also accepted).
        """
        if image.ndim == 5:
            if image.shape[2] != 1:
                raise ValueError(f"MiniMax-H3-Image trains on single frames, got T={image.shape[2]}")
            image = image[:, :, 0]
        if image.ndim != 4:
            raise ValueError(f"encode_image expects (1, 3, H, W), got {tuple(image.shape)}")
        if image.shape[-1] / 16 != int(image.shape[-1] / 16) or image.shape[-2] / 16 != int(image.shape[-2] / 16):
            raise ValueError(
                f"image size {tuple(image.shape[-2:])} is not divisible by the VAE's 16x compression"
            )

        device = next(vae.parameters()).device
        # [-1, 1] -> [0, 1] (the pipeline's base range) -> ImageNet-normalized.
        pixels01 = image.to(device=device, dtype=torch.float32)
        if pixels01.min() < 0.0:
            pixels01 = (pixels01.clamp(-1.0, 1.0) + 1.0) / 2.0
        mean, std = self._pixel_normalization(device, torch.float32)
        pixels = (pixels01 - mean) / std

        with torch.no_grad():
            posterior = vae.encode(pixels, return_dict=False)[0]
            latents = posterior.sample()
        # The released recipe rounds the posterior sample through float16.
        latents = latents.to(torch.float16).float()

        lat_mean, lat_std = self._latents_normalization(vae, device, torch.float32)
        latents = (latents - lat_mean) / lat_std
        return {"latent": latents[0]}  # (24, h, w)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        """Load Qwen3-VL with transformers' native 4-bit loader.

        The encoder is 33B. A hand-rolled meta build + per-shard placement was killed by the
        OOM reaper (transformers 5 does not honour ``init_empty_weights`` here), so the load
        goes through ``BitsAndBytesConfig``: transformers streams each shard, quantizes it to
        nf4 on the fly and never materializes the bf16 weights. ``lm_head`` and the vision
        tower are not needed for text embeddings but are kept -- they are only ~2 GB at 4-bit.
        """
        from transformers import AutoConfig, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        root = _convert.checkpoint_root(path)
        te_dir = os.path.join(root, "text_encoder")
        if not os.path.isdir(te_dir):
            te_dir = path

        self._trace("text_encoder: start")
        # Comfy NVFP4-AWQ path: the checkpoint ships no config, so the architecture comes
        # from the original text_encoder dir while the weights come from the Comfy file.
        comfy_file = (self.config.get("training") or {}).get("comfy_text_encoder")
        if comfy_file and (self.config.get("training") or {}).get("text_encoder_quantize") == "comfy_nvfp4":
            from .comfy_te import build_lazy_text_encoder

            self._trace(f"text_encoder: comfy nvfp4 <- {comfy_file}")
            cfg = AutoConfig.from_pretrained(te_dir)
            text_cfg = getattr(cfg, "text_config", cfg)
            if getattr(text_cfg, "rope_scaling", None) is None:
                text_cfg.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}
            model, info = build_lazy_text_encoder(comfy_file, cfg, device="cuda")
            self._te_modules = [m for m in model.modules() if hasattr(m, "packed")]
            self._te_model = model
            self._trace(
                f"text_encoder: done (comfy nvfp4 lazy, linears={info['lazy_linears']}, "
                f"dense={info['dense_placed']}, "
                f"vram={torch.cuda.memory_allocated() / 1024**3:.1f}GB)"
            )
            return model

        # Comfy NVFP4-AWQ path: the checkpoint ships no config, so the architecture comes
        # from the original text_encoder dir while the weights come from the Comfy file.
        comfy_file = (self.config.get("training") or {}).get("comfy_text_encoder")
        if comfy_file and (self.config.get("training") or {}).get("text_encoder_quantize") == "comfy_nvfp4":
            from .comfy_te import build_lazy_text_encoder

            self._trace(f"text_encoder: comfy nvfp4 <- {comfy_file}")
            cfg = AutoConfig.from_pretrained(te_dir)
            text_cfg = getattr(cfg, "text_config", cfg)
            if getattr(text_cfg, "rope_scaling", None) is None:
                text_cfg.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}
            model, info = build_lazy_text_encoder(comfy_file, cfg, device="cuda")
            self._trace(
                f"text_encoder: done (comfy nvfp4 lazy, linears={info['lazy_linears']}, "
                f"dense={info['dense_placed']}, "
                f"vram={torch.cuda.memory_allocated() / 1024**3:.1f}GB)"
            )
            return model

        config = AutoConfig.from_pretrained(te_dir)
        # the released config ships without rope_scaling for the text tower
        text_cfg = getattr(config, "text_config", config)
        if getattr(text_cfg, "rope_scaling", None) is None:
            text_cfg.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}

        training_cfg = self.config.get("training") or {}
        quant = training_cfg.get("text_encoder_quantize", "nf4")
        te_device = training_cfg.get("text_encoder_device", "cuda")
        # Tells the cache builder not to move this encoder to the GPU.
        self._te_on_cpu = te_device == "cpu"
        kwargs = {}
        if quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=False,
            )
        elif quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant.startswith("torchao_"):
            # torchao weight-only quantization: load bf16, then quantize block by block onto
            # the GPU (UnifiedTrainer.utils.quantize). Produces plain Tensors, so no bnb
            # quant_state assertions in the forward path.
            pass

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            te_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=("auto" if quant == "int8" else None),
            max_memory=({0: "29GiB", "cpu": "60GiB"} if quant == "int8" else None),
            **kwargs,
        )
        # Quantize on the GPU move (bnb fills in Params4bit.quant_state there). Loading with
        # device_map="cuda" instead leaves some params with quant_state=None and the first
        # forward trips bnb's `assert quant_state is not None`.
        if te_device == "cpu":
            model = model.to("cpu")
        elif quant.startswith("torchao_"):
            from UnifiedTrainer.utils.quantize import quantize_model as _ao_quantize

            qtype = quant[len("torchao_"):]
            # Verified against the real model tree: the language tower is
            # model.language_model.layers (64 Qwen3VLTextDecoderLayer). quantize_model takes
            # the attribute path of the block LIST, so this is the value it wants.
            _ao_quantize(
                model, qtype=qtype, device=torch.device("cuda"),
                dtype=torch.bfloat16,
                transformer_block_names=["model.language_model.layers"],
                low_vram=True,
            )
            # low_vram puts the quantized blocks back on CPU; int4 is ~17 GB, so it fits.
            moved = model.to("cuda")
            if moved is not None:
                model = moved
        elif quant in ("none", "int8", "nf4"):
            # nf4: quantization happens inside Params4bit.cuda(), so this move is what
            # actually compresses the encoder (and it is why device_map must be off).
            moved = model.to("cuda") if quant != "int8" else model
            if moved is not None:
                model = moved
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._trace(f"text_encoder: done (quant={quant}, "
                    f"vram={torch.cuda.memory_allocated() / 1024**3:.1f}GB)")
        return model

    def encode_text(
        self,
        text_encoder: Optional[nn.Module],
        tokenizer: Optional[Any],
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        reference_image=None,
        processor=None,
        **kwargs,
    ) -> dict:
        """Encode a prompt with Qwen3-VL and read decoder layer 50 (verbatim, no template)."""
        if text_encoder is None or tokenizer is None:
            raise ValueError("MiniMax-H3-Image requires the Qwen3-VL text encoder and tokenizer")

        # The released sampler conditions on the prompt verbatim; an empty prompt (the
        # CFG-unconditional branch of guide_flow_matching, and the cached
        # empty_embedding) would tokenize to zero rows, so give it a single space.
        text = prompt if prompt and prompt.strip() else " "
        try:
            te_device = next(text_encoder.parameters()).device
        except (StopIteration, AttributeError):
            te_device = device
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(te_device)
        length = input_ids.shape[1]
        # Qwen3-VL applies mrope: it requires 3-D position_ids of shape (3, B, L).
        position_ids = torch.arange(length, device=te_device).view(1, 1, -1).expand(3, 1, -1)
        with torch.no_grad():
            # A single-token prompt takes transformers' SDPA fast path, which calls
            # .item() on the padding mask -- with a meta-built encoder some buffers are
            # still meta and that raises. No mask is needed for one token.
            extra = {} if length == 1 else {"attention_mask": torch.ones_like(input_ids)}
            # `.model` is the bare Qwen3-VL text tower: ForConditionalGeneration would run
            # the (skipped, still-meta) lm_head before returning hidden states.
            backbone = getattr(text_encoder, "model", text_encoder)
            outputs = backbone(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                **extra,
            )
        # The conditioning is the **pre-norm** output of decoder layer 50:
        #   * full stack  -> hidden_states[50]      (layers 51.. are never read)
        #   * truncated to exactly 50 layers -> hidden_states[-2], because the last entry is the
        #     post-norm final state (diffusers' `get_qwen3vl_prompt_embeds` raises on that
        #     mistake; it is what this code used to feed, giving an RMSNorm'd embedding whose
        #     per-token RMS is 1.0 and which the transformer largely ignored).
        n_states = len(outputs.hidden_states)
        n_layers = n_states - 1
        if n_layers > TEXT_ENCODER_LAYER:
            layer_idx = TEXT_ENCODER_LAYER
        else:
            layer_idx = n_layers - 1
        hidden = outputs.hidden_states[layer_idx][0].to(device=device, dtype=dtype)
        length = int(hidden.shape[0])
        mask = torch.ones(length, dtype=torch.bool)
        return {
            # cache contract shared with qwen_image21 / krea2
            "prompt_embed": hidden.cpu(),                      # (L, 5120), unnormalized layer 50
            "prompt_embeds_mask": mask,                        # no padding in H3's packed layout
            "prompt_embed_length": length,
            # the builder pops image_token_mask before saving; keep a second copy so the
            # slot information survives into the npz
            "image_token_mask": torch.zeros(length, dtype=torch.bool),
            "prompt_token_tags": torch.ones(length, dtype=torch.long),  # 1 = text modality
        }

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        """Decode a normalized latent back to an ``(1, 3, H, W)`` image in ``[-1, 1]``."""
        if latent.ndim == 4:
            latent = latent.squeeze(0)
        if latent.ndim == 4:  # (C, 1, H, W) -> (C, H, W)
            latent = latent[:, 0]
        try:
            device = next(vae.parameters()).device
        except (StopIteration, AttributeError, TypeError):
            device = latent.device if latent.is_floating_point() else torch.device("cpu")
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        mean, std = self._latents_normalization(vae, device, torch.float32)
        latents = latent.to(device=device, dtype=torch.float32) * std + mean
        with torch.no_grad():
            pixels = vae.decode(latents, return_dict=False)[0]
        pmean, pstd = self._pixel_normalization(device, torch.float32)
        pixels01 = (pixels.float() * pstd + pmean).clamp(0.0, 1.0)
        return pixels01.mul(2.0).sub(1.0)

    # ── Training plumbing ──────────────────────────────────────────────

    def _text_tags(self, batch: dict, embed: dict, length: int, device) -> torch.Tensor:
        tags = embed.get("text_token_tags") if isinstance(embed, dict) else None
        if tags is None:
            return torch.full((length,), _packing.TEXT_TAG, dtype=torch.long, device=device)
        if isinstance(tags, np.ndarray):
            tags = torch.from_numpy(tags)
        return tags.to(device=device, dtype=torch.long)

    def prepare_model_input(
        self, batch: dict, latents: dict, sigmas=None, target_key: Optional[str] = None, **kwargs
    ) -> dict:
        """Pack ``[text | target]`` rows and build the transformer's structural inputs."""
        embeddings = batch.get("embeddings")
        if not embeddings:
            raise ValueError("MiniMax-H3-Image requires batch['embeddings']")

        embed = embeddings[0]
        prompt_embed = embed["prompt_embed"] if isinstance(embed, dict) else embed

        if isinstance(latents, (list, tuple)):
            target = latents[0]
        elif isinstance(latents, dict):
            target = latents[target_key] if target_key and target_key in latents else next(iter(latents.values()))
        else:
            target = latents
        # the cache hands back numpy arrays; every downstream op expects tensors
        import torch as _torch

        if not _torch.is_tensor(target):
            target_entry = target
            target = _torch.as_tensor(target_entry)
            if isinstance(latents, dict):
                latents = {k: _torch.as_tensor(v) for k, v in latents.items()}
            elif isinstance(latents, (list, tuple)):
                latents = [_torch.as_tensor(v) for v in latents]
        # Everything follows the target latent's device unless the caller names one.
        device = kwargs.get("device") or target.device
        import torch as _torch

        if not _torch.is_tensor(prompt_embed):
            prompt_embed = _torch.as_tensor(prompt_embed)  # the cache hands back numpy
        prompt_embed = prompt_embed.to(device=device)
        tags = self._text_tags(batch, embed, prompt_embed.shape[0], device)
        if target.ndim == 5:
            target = target[:, :, 0]
        if target.ndim == 4:
            target = target[0]
        latent_h, latent_w = int(target.shape[-2]), int(target.shape[-1])
        if target.shape[0] != self.latent_channels:
            raise ValueError(f"target latent has {target.shape[0]} channels, expected {self.latent_channels}")

        position_ids, token_tags, image_indices, text_indices, num_ref = _packing.build_packed_sequence(
            tags.cpu(), [], latent_h, latent_w,
        )
        rows = _packing.patchify_latents(target.unsqueeze(0).unsqueeze(2))  # (1, rows, 96)

        if sigmas is None:
            sigmas = batch.get("sigmas", torch.tensor([1.0]))
        # The packed layout is batch-uniform (as in MiniMax-H3's video adapter).
        sigma = float(torch.as_tensor(sigmas).flatten()[0])
        timestep = 1.0 - sigma  # the transformer consumes t = 1 - sigma
        seq_len = position_ids.shape[0]
        row_timesteps = torch.full((seq_len,), timestep, dtype=torch.float32)
        unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)

        self._latent_hw = (latent_h, latent_w)
        self._num_image_rows = int(image_indices.numel())
        return {
            "hidden_states": rows.to(device),
            "encoder_hidden_states": prompt_embed.unsqueeze(0).to(device),
            "timestep": unique_timesteps.to(device),
            "timestep_indices": timestep_indices.to(device),
            "token_tags": token_tags.to(device),
            "position_ids": position_ids.to(device),
            "image_indices": image_indices.to(device),
            "text_indices": text_indices.to(device),
            "pdd_step_index": self._resolve_head(sigma),
            "_latent_hw": (latent_h, latent_w),
            "_num_image_rows": int(image_indices.numel()),
        }

    def unpack_prediction(self, model_pred: torch.Tensor, **kwargs) -> list:
        """``(B, image_rows, 96)`` -> ``(B, 24, h, w)``."""
        # The trainer forwards only the model's own kwargs, so the layout is normally read
        # from the copy stashed during prepare_model_input.
        if hasattr(model_pred, "sample"):  # return_dict=True -> Transformer2DModelOutput
            model_pred = model_pred.sample
        latent_hw = kwargs.get("_latent_hw") or getattr(self, "_latent_hw", None)
        if latent_hw is None:
            raise ValueError("unpack_prediction needs _latent_hw from prepare_model_input")
        latent_h, latent_w = latent_hw
        return [_packing.unpatchify_rows(model_pred, latent_h, latent_w, self.latent_channels)]

    def compute_target(self, noise: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Data-ward target: ``v = x0 - x_t``."""
        return latents - noise

    def sample_timesteps(
        self,
        batch_size,
        device,
        dtype,
        latent_height=None,
        latent_width=None,
    ):
        """Sample (timesteps, sigmas) on the checkpoint's own PDD grid.

        The released checkpoint is a PDD distillation whose sigma grid is
        [1.0, 0.9675, 0.8875, 0.75, 0.5338, 0.232, 0.0] -- heavily biased to high noise.
        Uniform sampling puts most of the mass in the low-noise region the model was least
        shaped for, which is the main reason the adapters barely moved the weights. Intervals
        are drawn with probability proportional to their width and jittered inside, so the
        marginal matches the grid density while still covering every interval.

        ``training.timestep_sampling = "uniform"`` restores the old behaviour.
        """
        mode = (self.config.get("training") or {}).get("timestep_sampling", "pdd")
        if mode != "pdd":
            from UnifiedTrainer.models.base import sample_sigma_uniform

            return sample_sigma_uniform(batch_size, device, dtype)

        grid = torch.tensor(_packing.DEFAULT_PDD_SIGMAS, device=device, dtype=torch.float32)
        lo, hi = grid[1:], grid[:-1]
        widths = (hi - lo).clamp_min(0.0)
        probs = widths / widths.sum()
        idx = torch.multinomial(probs, batch_size, replacement=True)
        u = torch.rand(batch_size, device=device)
        sigmas = lo[idx] + u * (hi[idx] - lo[idx])
        sigmas = sigmas.clamp(1e-4, 1.0)
        # the transformer consumes t unscaled in [0, 1]
        timesteps = 1.0 - sigmas
        return timesteps.to(dtype), sigmas.to(dtype)

    def sample_validation(self, trainer, batch: dict, epoch: int, sample_idx: int):
        """Plain flow-matching Euler loop with the single prediction head.

        ``validation.num_inference_steps`` (default 6) sets the step count; sigma runs
        linearly from 1 to 0 and every step uses the same output head, matching the
        single-prediction objective the model is trained with.
        """
        from PIL import Image as PILImage

        transformer = trainer.transformer
        device = trainer.device
        embeddings = batch.get("embeddings")
        if not embeddings or embeddings[0] is None:
            return None

        latents = batch.get("latents") or {}
        if not latents:
            return None
        first_key = next(iter(latents))
        target = latents[first_key]
        if target.ndim == 5:
            target = target[:, :, 0]
        if target.ndim == 4:
            target = target[0]
        latent_h, latent_w = int(target.shape[-2]), int(target.shape[-1])

        seed = int(getattr(trainer, "val_seed", 42) or 42)
        dtype = next(transformer.parameters()).dtype
        if dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            dtype = torch.bfloat16

        steps = int((self.config.get("validation") or {}).get("num_inference_steps", 6))
        gen = torch.Generator(device=device).manual_seed(seed + sample_idx)
        x = torch.randn(
            (1, self.latent_channels, latent_h, latent_w),
            generator=gen, device=device, dtype=dtype,
        )

        sigmas = torch.linspace(1.0, 0.0, steps + 1).tolist()
        if self._head_mode == "sigma":
            # The checkpoint is a 6-NFE distillation: its heads, and the sigma grid they were
            # distilled on, only mean anything together. Sampling on a linear grid (what the
            # base head allows) puts every step off the grid the release ships, so a
            # validation image would never be comparable to a ComfyUI render of the same
            # LoKR. Use pdd_config.json's grid when the per-sigma heads are selected.
            grid = list(_packing.DEFAULT_PDD_SIGMAS)
            if steps != len(grid) - 1:
                logger.warning(
                    "MiniMax-H3-Image: validation.num_inference_steps=%d but the PDD grid has "
                    "%d steps; sampling on the PDD grid.", steps, len(grid) - 1,
                )
            sigmas = grid
            steps = len(grid) - 1
        transformer.eval()
        for step in range(steps):
            sigma, sigma_next = sigmas[step], sigmas[step + 1]
            inputs = self.prepare_model_input(
                batch, [x], sigmas=torch.tensor([sigma], device=device)
            )
            forward_kwargs = {k: v for k, v in inputs.items() if not k.startswith("_")}
            with torch.no_grad():
                pred = transformer(**forward_kwargs, return_dict=False)[0]
            velocity = self.unpack_prediction(pred, **inputs)[0]
            x = (x.float() + (sigma - sigma_next) * velocity.float()).to(dtype)

        image = self.decode_latent(trainer.vae, x)  # (1, 3, H, W) in [-1, 1]
        array = ((image[0].float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        gen_pil = PILImage.fromarray(array.permute(1, 2, 0).numpy())

        # Trainer contract (engine/trainer.py `gen_pils, ref_pils = imgs`): a pair of LISTS --
        # one generated image plus the cached ground-truth target, so every validation run
        # writes both a sample and its reference next to each other. Returning the bare PIL
        # raised "cannot unpack non-iterable Image object" and silently produced 0 images.
        ref_pils = []
        try:
            paths = batch.get("pixel_paths")
            if isinstance(paths, list) and paths:
                paths = paths[0]
            if isinstance(paths, dict):
                target_path = paths.get("T")
                if target_path and os.path.isfile(target_path):
                    ref_pils = [PILImage.open(target_path).convert("RGB")]
        except Exception as exc:  # noqa: BLE001 - the reference is a convenience, never fatal
            logger.warning("MiniMax-H3-Image: could not load the validation target: %s", exc)
        return [gen_pil], ref_pils

