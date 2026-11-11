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
* the PDD head index of a sample is the index of the sigma band that contains it, so a
  fine-tune stays on the per-step specialization the sampler uses;
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
        self._pdd_sigmas: Optional[tuple] = None
        self._num_pdd_heads: Optional[int] = None

    # ── Loading ────────────────────────────────────────────────────────

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

        # Pre-quantized nf4 fast path: building the 31B model in bf16 costs 62 GB of RAM and
        # re-quantizing every start costs minutes. ``training.quantize = "prequantized"``
        # loads a previously saved nf4 snapshot (16 GB) straight onto the GPU instead.
        if (self.config.get("training") or {}).get("quantize") == "prequantized":
            cache_dir = os.path.join(path, "nf4")
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

        state, cfg = _convert.load_vae_state_dict(_convert.checkpoint_root(path))
        model = AutoencoderKLMiniMaxH3Image(**cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"MiniMax-H3-Image VAE: missing keys {missing[:6]}")
        if unexpected:
            logger.warning("MiniMax-H3-Image VAE: %d unexpected keys, e.g. %s", len(unexpected), unexpected[:4])
        return model.to(dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        # The released sampler uses an explicit sigma grid, not a scheduler class.
        return {"sigmas": self.pdd_sigmas(path)}

    def pdd_sigmas(self, checkpoint_dir: Optional[str] = None) -> tuple:
        """The PDD sigma grid of the checkpoint (``pdd_config.json``)."""
        if self._pdd_sigmas is None:
            pdd = _convert.read_pdd_config(checkpoint_dir or self._model_path) if checkpoint_dir or self._model_path else {}
            sigmas = pdd.get("sigmas") or _packing.DEFAULT_PDD_SIGMAS
            self._pdd_sigmas = tuple(float(s) for s in sigmas)
            self._num_pdd_heads = int(pdd.get("num_inference_steps", len(self._pdd_sigmas) - 1))
        return self._pdd_sigmas

    # ── Architecture specs ─────────────────────────────────────────────

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
        """Load Qwen3-VL in int8 via meta construction + per-shard streaming.

        The encoder is 33B: building it in bf16 costs 66 GB of RAM, which does not leave
        room for anything else. Building on meta and quantizing each Linear to int8 while it
        streams onto the GPU needs ~0 RAM and ~33 GB of VRAM, and the LM head plus the
        vision tower are dropped -- text-embedding extraction never calls them.
        """
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration

        from . import quantize as _quantize

        root = _convert.checkpoint_root(path)
        te_dir = os.path.join(root, "text_encoder")
        if not os.path.isdir(te_dir):
            te_dir = path

        config = AutoConfig.from_pretrained(te_dir, trust_remote_code=True)
        # transformers' Qwen3-VL mrope defaults are missing from the released config.
        mrope = {"rope_type": "default", "mrope_section": [24, 20, 20]}
        text_cfg = getattr(config, "text_config", config)
        if getattr(text_cfg, "rope_scaling", None) is None:
            text_cfg.rope_scaling = mrope
        config._attn_implementation = "sdpa"

        quant = (self.config.get("training") or {}).get("text_encoder_quantize", "int8")
        model = _quantize.build_meta_hf_model(config, Qwen3VLForConditionalGeneration)
        if quant == "int8":
            n = _quantize.replace_linears_with_int8(model)
            logger.info("MiniMax-H3-Image text encoder: %d Linears prepared for int8", n)
        elif quant == "nf4":
            logger.warning("text_encoder_quantize=nf4 is disabled for this model; using bf16")

        info = _quantize.stream_hf_shards_into(
            model, te_dir, device="cuda",
            skip_prefixes=("lm_head.", "model.visual.", "visual."),
            quant="int8" if quant == "int8" else "none",
        )
        logger.info(
            "MiniMax-H3-Image text encoder streamed: %d tensors placed, %d skipped (lm_head/vision)",
            info["placed"], info["skipped"],
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def encode_text(
        self,
        text_encoder: Optional[nn.Module],
        tokenizer: Optional[Any],
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        """Encode a prompt with Qwen3-VL and read decoder layer 50 (verbatim, no template)."""
        if text_encoder is None or tokenizer is None:
            raise ValueError("MiniMax-H3-Image requires the Qwen3-VL text encoder and tokenizer")

        # The released sampler conditions on the prompt verbatim; an empty prompt (the
        # CFG-unconditional branch of guide_flow_matching, and the cached
        # empty_embedding) would tokenize to zero rows, so give it a single space.
        text = prompt if prompt and prompt.strip() else " "
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(device)
        with torch.no_grad():
            outputs = text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(device=device, dtype=dtype)
        return {
            "prompt_embed": hidden.cpu(),  # (L, 5120) — cache convention
            "prompt_embeds_mask": torch.ones(hidden.shape[0], dtype=torch.bool),
            "text_token_tags": torch.full((hidden.shape[0],), _packing.TEXT_TAG, dtype=torch.long),
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

    def _pdd_head_for_sigma(self, sigma: float) -> int:
        grid = self.pdd_sigmas()
        head = len(grid) - 2
        for i in range(len(grid) - 1):
            if grid[i + 1] < sigma <= grid[i]:
                head = i
                break
        if sigma > grid[0]:
            head = 0
        return int(head)

    def prepare_model_input(
        self, batch: dict, latents: dict, sigmas=None, target_key: Optional[str] = None, **kwargs
    ) -> dict:
        """Pack ``[text | target]`` rows and build the transformer's structural inputs."""
        embeddings = batch.get("embeddings")
        if not embeddings:
            raise ValueError("MiniMax-H3-Image requires batch['embeddings']")

        embed = embeddings[0]
        prompt_embed = embed["prompt_embed"] if isinstance(embed, dict) else embed

        target = latents[target_key] if target_key and target_key in latents else next(iter(latents.values()))
        # Everything follows the target latent's device unless the caller names one.
        device = kwargs.get("device") or target.device
        if isinstance(prompt_embed, torch.Tensor):
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
        # The packed layout is batch-uniform (as in MiniMax-H3's video adapter), so
        # every sample in a forward shares this sigma and therefore this PDD head.
        sigma = float(torch.as_tensor(sigmas).flatten()[0])
        timestep = 1.0 - sigma  # the transformer consumes t = 1 - sigma
        seq_len = position_ids.shape[0]
        row_timesteps = torch.full((seq_len,), timestep, dtype=torch.float32)
        unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)

        return {
            "hidden_states": rows.to(device),
            "encoder_hidden_states": prompt_embed.unsqueeze(0).to(device),
            "timestep": unique_timesteps.to(device),
            "timestep_indices": timestep_indices.to(device),
            "token_tags": token_tags.to(device),
            "position_ids": position_ids.to(device),
            "image_indices": image_indices.to(device),
            "text_indices": text_indices.to(device),
            "pdd_step_index": self._pdd_head_for_sigma(sigma),
            "_latent_hw": (latent_h, latent_w),
            "_num_image_rows": int(image_indices.numel()),
        }

    def unpack_prediction(self, model_pred: torch.Tensor, **kwargs) -> list:
        """``(B, image_rows, 96)`` -> ``(B, 24, h, w)``."""
        latent_h, latent_w = kwargs.get("_latent_hw", (None, None))
        if latent_h is None:
            raise ValueError("unpack_prediction needs _latent_hw from prepare_model_input")
        return [_packing.unpatchify_rows(model_pred, latent_h, latent_w, self.latent_channels)]

    def compute_target(self, noise: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Data-ward target: ``v = x0 - x_t``."""
        return latents - noise

    def sample_validation(self, trainer, batch: dict, epoch: int, sample_idx: int):
        """PDD sampling loop for validation images.

        The release samples on a fixed sigma grid and switches output head every step, so
        the loop lives here instead of in the generic scheduler path. Euler step on the
        data-ward velocity (``x0 = x_t + sigma * v``): ``x_next = x_t + (sigma - sigma_next) * v``,
        evaluated in float32 like the released sampler.
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

        dtype = next(transformer.parameters()).dtype
        if dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
            dtype = torch.bfloat16

        gen = torch.Generator(device=device).manual_seed(int(trainer.val_seed) + sample_idx)
        x = torch.randn(
            (self.latent_channels, latent_h, latent_w), generator=gen, device=device, dtype=dtype
        )

        grid = self.pdd_sigmas()
        transformer.eval()
        for step in range(len(grid) - 1):
            sigma = float(grid[step])
            sigma_next = float(grid[step + 1])
            inputs = self.prepare_model_input(
                batch, {first_key: x}, sigmas=torch.tensor([sigma], device=device)
            )
            forward_kwargs = {k: v for k, v in inputs.items() if not k.startswith("_")}
            with torch.no_grad():
                pred = transformer(**forward_kwargs, return_dict=False)[0]
            velocity = self.unpack_prediction(pred, **inputs)[0]
            x = (x.float() + (sigma - sigma_next) * velocity.float()).to(dtype)

        image = self.decode_latent(trainer.vae, x)  # (1, 3, H, W) in [-1, 1]
        array = ((image[0].float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        gen_pil = PILImage.fromarray(array.permute(1, 2, 0).numpy())
        return [gen_pil], []

    def sample_timesteps(self, batch_size: int, device, dtype, **kwargs):
        """Sample sigma on the checkpoint's PDD grid, plus one head index per sample.

        The release distils one head per step, so training draws sigma from the step values
        the heads were trained on (optionally uniform in between, ``training.pdd_sigma_mode``).
        """
        sigmas = self.pdd_sigmas()
        mode = (self.config.get("training") or {}).get("pdd_sigma_mode", "grid")
        values = torch.tensor(list(sigmas[:-1]), dtype=torch.float32, device=device)
        if mode == "uniform":
            s = sample_sigma_uniform(batch_size, device, dtype)
        else:
            idx = torch.randint(0, values.numel(), (batch_size,), device=device)
            s = values[idx].to(dtype)
        # Engine contract: (timesteps, sigmas). The PDD head is derived from sigma in
        # prepare_model_input, so nothing else has to travel with the draw.
        return s, s
