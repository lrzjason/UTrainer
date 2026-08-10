"""Qwen-Image model adapter -text-to-image generation."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from UnifiedTrainer.models.base import BaseModelAdapter
from UnifiedTrainer.registry import ModelRegistry

logger = logging.getLogger(__name__)

RESOLUTION_CONFIG = {
    2048: [
        (1024, 2560), (1152, 2816), (1280, 3072), (1280, 2816),
        (1408, 3008), (1536, 2816), (1664, 2688), (1792, 2368),
        (1920, 2240), (2048, 2560), (2176, 2304), (2048, 2048),
        (2048, 2304),
    ],
}

# Qwen-Image chat-template constants — matches ComfyUI's Qwen-Image template
# (and the Krea-2 template) exactly: the same system instruction, user-opening,
# and assistant suffix.  The full template is:
#   "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
#
# At encoding time the system+user preamble is stripped from the hidden-states
# output (matching ComfyUI's ``encode_token_weights`` prefix stripping), so the
# DiT receives only the user-content tokens.
_PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_TEMPLATE_START_IDX = 34  # token count of _PROMPT_TEMPLATE_PREFIX
_PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS: int = 5  # "<|im_end|>\n<|im_start|>assistant\n"
_DEFAULT_MAX_SEQ_LEN = 512
_IMG_PLACEHOLDER = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


@ModelRegistry.register("qwen_image")
class QwenImageAdapter(BaseModelAdapter):
    """Adapter for Qwen-Image text-to-image model."""

    name = "qwen_image"
    patch_size = 2  # Qwen transformer patches latents in 2x2 blocks

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")
        # Target grid dims cached from prepare_model_input — used by
        # unpack_prediction to invert patchify+pack (krea2-style, no sqrt guesses).
        self._target_grids: list[tuple[int, int]] = []

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from .transformer_qwenimage import BlockSwapQwenImageTransformer2DModel
        return BlockSwapQwenImageTransformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers import AutoencoderKLQwenImage
        return AutoencoderKLQwenImage.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_pretrained(path)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype)
        except Exception:
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        try:
            from transformers import Qwen2Tokenizer
            return Qwen2Tokenizer.from_pretrained(path)
        except Exception:
            return None

    def load_processor(self, path: str, tokenizer_path: str = "") -> Optional[Any]:
        """Load the Qwen2.5-VL processor for multimodal text+image encoding.

        Uses ``AutoProcessor`` when the model directory ships a
        ``preprocessor_config.json``; otherwise constructs a
        ``Qwen2_5_VLProcessor`` manually from a ``Qwen2VLImageProcessor`` and
        the tokenizer loaded from ``tokenizer_path``.
        """
        try:
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(path)
        except Exception:
            pass

        try:
            from transformers import Qwen2_5_VLProcessor
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

            tok_path = tokenizer_path or path
            tokenizer = self.load_tokenizer(tok_path)
            if tokenizer is None:
                logger.warning(
                    f"Cannot construct Qwen2_5_VLProcessor: tokenizer is None from {tok_path}"
                )
                return None

            # Read vision config to get patch_size / merge_size if available.
            import json
            import os

            patch_size = 14
            temporal_patch_size = 2
            merge_size = 2
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    model_config = json.load(f)
                vision_config = model_config.get("vision_config", {})
                patch_size = vision_config.get("patch_size", patch_size)
                temporal_patch_size = vision_config.get(
                    "temporal_patch_size", temporal_patch_size
                )
                merge_size = vision_config.get("spatial_merge_size", merge_size)

            image_processor = Qwen2VLImageProcessor(
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processor = Qwen2_5_VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
            logger.info(
                f"Constructed Qwen2_5_VLProcessor (tokenizer from {tok_path}, "
                f"patch_size={patch_size})"
            )
            return processor
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to construct Qwen2_5_VLProcessor from {path}: {e}")
            return None

    @property
    def latent_channels(self) -> int:
        return 16

    @property
    def vae_scale_factor(self) -> int:
        return 8

    @property
    def embedding_dim(self) -> int:
        return 7680

    @property
    def resolution_config(self) -> dict:
        return RESOLUTION_CONFIG

    @property
    def supports_image_conditioning(self) -> bool:
        return False

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            # AutoencoderKLQwenImage is a video VAE expecting 5D input (B, C, T, H, W).
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(2)
            latent = vae.encode(image_tensor).latent_dist.sample()
            # Squeeze temporal dim back to 4D (B, C, H, W)
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            # Qwen VAE uses per-channel latents_mean/latents_std, not shift_factor/scaling_factor.
            latents_mean = torch.tensor(
                vae.config.latents_mean, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latents_std = torch.tensor(
                vae.config.latents_std, device=latent.device, dtype=latent.dtype
            ).view(1, -1, 1, 1)
            latent = (latent - latents_mean) / latents_std
        return {"latent": latent}

    def encode_text(
        self,
        text_encoder: Optional[nn.Module],
        tokenizer: Optional[Any],
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        reference_image: Optional[Any] = None,
        processor: Optional[Any] = None,
    ) -> dict:
        """Encode a text prompt (and optionally reference images) via Qwen2.5-VL.

        When *reference_image* and *processor* are both provided the reference
        images are embedded as ``Picture N:`` placeholders in the user message
        and forwarded to the Qwen2.5-VL vision encoder.  The text-only path uses
        the standard Qwen-Image chat template.

        Returns
        -------
        dict
            ``prompt_embed``        — ``(seq_len, dim)``
            ``prompt_embeds_mask``  — ``(seq_len,)`` bool
            ``image_token_mask``    — ``(seq_len,)`` bool  *(multimodal only)*
        """
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "Qwen-Image requires a loaded Qwen2.5-VL text encoder and tokenizer."
            )

        is_multimodal = reference_image is not None and processor is not None

        # ── 1. Build prompt text (matches ComfyUI QWEN_IMAGE_TEMPLATE) ────
        # Full chat template: system + user_preamble + {content} + suffix.
        # After encoding the system+user preamble is stripped (see step 4),
        # mirroring ComfyUI's ``encode_token_weights`` prefix stripping.
        prompt_text = _PROMPT_TEMPLATE_PREFIX
        refs: list[Any] = []
        if is_multimodal:
            refs = reference_image if isinstance(reference_image, list) else [reference_image]
            prompt_text += "".join(
                _IMG_PLACEHOLDER.format(i + 1) for i in range(len(refs))
            )
        prompt_text += prompt + _PROMPT_TEMPLATE_SUFFIX

        # ── 2. Tokenize ────────────────────────────────────────────────────
        proc_kwargs: dict[str, Any] = dict(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
        )
        if is_multimodal:
            proc_kwargs.update(images=refs, do_rescale=False)
        else:
            proc_kwargs.update(
                truncation=True,
                max_length=(
                    _DEFAULT_MAX_SEQ_LEN
                    + _PROMPT_TEMPLATE_START_IDX
                    + _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS
                ),
            )

        if processor is not None:
            main_inputs = processor(**proc_kwargs).to(device)
        else:
            # No processor available (text-only encoders / direct calls): fall
            # back to the raw tokenizer.  Reference-image conditioning cannot
            # run without a processor, so it is silently ignored.
            tok_kwargs: dict[str, Any] = dict(
                text=proc_kwargs["text"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tok_kwargs["max_length"] = proc_kwargs.get("max_length")
            main_inputs = tokenizer(**tok_kwargs).to(device)

        input_ids = main_inputs.input_ids
        attention_mask = main_inputs.attention_mask.bool()

        # ── 2b. Trim suffix tokens before the encoder forward ─────────────
        # The suffix ("<|im_end|>\n<|im_start|>assistant\n" — 5 tokens) is
        # included in the prompt text for structural alignment with the
        # chat template, but is stripped before the forward so the output
        # hidden-states length exactly matches the trimmed attention mask.
        valid_lens = attention_mask.sum(dim=1)  # non-pad token count per item
        for b in range(input_ids.shape[0]):
            vl = int(valid_lens[b].item())
            if vl >= _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS:
                attention_mask[b, vl - _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS:vl] = False
        # Also trim input_ids to the max valid length (minus suffix) so the
        # sequence-length tracking matches the mask.
        new_max_len = int(valid_lens.max().item()) - _PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS
        # Keep at least the preamble + 1 content token so the prefix strip in
        # step 4 can never produce an empty tensor.
        new_max_len = max(new_max_len, _PROMPT_TEMPLATE_START_IDX + 1)
        input_ids = input_ids[:, :new_max_len]
        attention_mask = attention_mask[:, :new_max_len]

        # ── 3. Assemble encoder kwargs ─────────────────────────────────────
        # Qwen2.5-VL computes 3D mRoPE position ids internally when
        # ``position_ids`` is not passed, for both the text-only and the
        # vision path — so we deliberately do NOT build them here.
        encoder_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if is_multimodal:
            _pixel = getattr(main_inputs, "pixel_values", None)
            if _pixel is not None:
                encoder_kwargs["pixel_values"] = _pixel.to(dtype=text_encoder.dtype)
            _grid = getattr(main_inputs, "image_grid_thw", None)
            if _grid is not None:
                encoder_kwargs["image_grid_thw"] = _grid

        # ── 4. Run encoder & extract last-layer hidden states ─────────────
        with torch.no_grad():
            outputs = text_encoder(**encoder_kwargs)

        hidden_states = self._extract_text_hidden_states(outputs)
        hidden_states = hidden_states[:, _PROMPT_TEMPLATE_START_IDX:]  # strip preamble

        # Clamp masks to actual hidden-states length (defensive).
        valid_len = hidden_states.shape[1]
        attention_mask = attention_mask[
            :, _PROMPT_TEMPLATE_START_IDX:_PROMPT_TEMPLATE_START_IDX + valid_len
        ]

        # ── 5. Build result ────────────────────────────────────────────────
        result: dict[str, Any] = dict(
            prompt_embed=hidden_states[0].to(dtype),
            prompt_embeds_mask=attention_mask[0].to(torch.bool),
        )

        if is_multimodal:
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            result["image_token_mask"] = (
                (
                    input_ids[
                        0,
                        _PROMPT_TEMPLATE_START_IDX:_PROMPT_TEMPLATE_START_IDX + valid_len,
                    ]
                    == image_pad_id
                )
                if image_pad_id is not None
                else torch.zeros(valid_len, dtype=torch.bool)
            )

        return result

    @staticmethod
    def _extract_text_hidden_states(outputs: Any) -> torch.Tensor:
        """Pull the last-layer text hidden states from a Qwen2.5-VL output.

        ``Qwen2_5_VLForConditionalGeneration.forward`` returns a
        ``Qwen2_5_VLCausalLMOutputWithPast`` whose ``hidden_states`` is a tuple
        of ``(batch, seq_len, dim)`` tensors (one per LM layer).  Accepts a
        tuple (use last layer) or a bare ``(batch, seq_len, dim)`` tensor so
        mocked/test encoders can return either shape.
        """
        hs = getattr(outputs, "hidden_states", None)
        if isinstance(hs, (tuple, list)):
            hs = hs[-1]
        if hs is None:
            hs = getattr(outputs, "last_hidden_state", None)
        if hs is None:
            raise RuntimeError(
                "Text encoder output has neither 'hidden_states' nor "
                "'last_hidden_state' — cannot extract prompt embeddings."
            )
        return hs

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        # Denormalize using latents_mean/latents_std (inverse of encode_image).
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latents_std = torch.tensor(
            vae.config.latents_std, device=latent.device, dtype=latent.dtype
        ).view(1, -1, 1, 1)
        latent = latent * latents_std + latents_mean
        # AutoencoderKLQwenImage expects 5D input (B, C, T, H, W).
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        with torch.no_grad():
            image = vae.decode(latent).sample
        # Squeeze temporal dim back to 4D
        if image.ndim == 5:
            image = image.squeeze(2)
        return image

    def prepare_model_input(
        self, batch: dict, noise: torch.Tensor | list[torch.Tensor], sigmas: torch.Tensor
    ) -> dict:
        target = noise[0] if isinstance(noise, list) else noise
        timesteps = (sigmas * 1000).long()

        device, dtype = target.device, target.dtype

        # Extract per-sample text embeddings from the batch cache (npz dicts).
        encoder_hidden_states = self._extract_encoder_hidden_states(batch, device, dtype)
        encoder_attention_mask = self._extract_encoder_attention_mask(batch, device)

        if encoder_hidden_states is None:
            raise ValueError(
                "Qwen-Image requires encoder_hidden_states in the batch. "
                "Ensure the data pipeline provides pre-computed text embeddings "
                "(cached via encode_text)."
            )
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                f"Qwen-Image expects encoder_hidden_states of shape (B, seq_len, dim), "
                f"got shape {tuple(encoder_hidden_states.shape)} with "
                f"ndim={encoder_hidden_states.ndim}"
            )

        # Text token count per sample — consumed by the transformer's RoPE
        # (pos_embed(img_shapes, txt_seq_lens, ...)).
        txt_seq_lens = encoder_attention_mask.sum(dim=1).long().tolist()

        # Patchify + pack the noisy target latent into the 3D token sequence
        # the transformer expects: (B, C, H, W) -> (B, (H//p)*(W//p), C*p*p).
        # img_in is nn.Linear(in_channels=C*p*p, inner_dim) and the forward
        # docstring states hidden_states is (B, image_seq_len, in_channels).
        p = self.patch_size
        B, _, H, W = target.shape
        target_grids = [(H // p, W // p)]
        # Cache target grids for unpack_prediction (krea2-style, no sqrt guesses).
        self._target_grids = target_grids
        hidden_states = self._patchify_and_pack_latents(target)

        # Image grid shapes for RoPE — per sample (frame=1, H_lat//patch, W_lat//patch).
        # pos_embed expects one (frame, height, width) entry per image with
        # frame*height*width == patched sequence length.  MUST be tuples:
        # QwenEmbedRope.forward does ``video_fhw = video_fhw[0]`` then unpacks
        # ``frame, height, width = fhw`` — a list-of-lists would iterate the
        # inner list's int elements and crash.
        img_shapes = [(1, H // p, W // p)] * B

        return {
            "hidden_states": hidden_states,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_attention_mask,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _patchify_and_pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack (B, C, H, W) latents into (B, (H//p)*(W//p), C*p*p).

        Mirrors krea2's pack convention (models/krea2/__init__.py
        ``_pack_latents``): view (B, C, gh, p, gw, p) -> permute -> flatten
        rows in row-major order.  ``unpack_prediction`` inverts exactly this
        layout.
        """
        p = self.patch_size
        B, C, H, W = latents.shape
        gh, gw = H // p, W // p
        latents = latents.view(B, C, gh, p, gw, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, gh, gw, C, p, p)
        return latents.reshape(B, gh * gw, C * p * p)

    def _extract_encoder_hidden_states(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Extract and stack encoder_hidden_states from cached embeddings.

        The data pipeline stores per-sample embeddings as dicts with
        ``prompt_embed`` of shape (seq_len, dim).  Different captions may have
        different seq_len, so we pad all to the batch-max before stacking.

        Returns:
            (B, max_seq_len, dim) tensor.
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            return None

        prompt_embeds = []
        for emb in embeddings:
            if emb is None:
                return None
            pe = emb["prompt_embed"] if isinstance(emb, dict) else emb
            if isinstance(pe, np.ndarray):
                pe = torch.from_numpy(pe)
            # Cast to target dtype immediately — avoids float32 intermediates
            # accumulating in the padded list.
            pe = pe.to(device=device, dtype=dtype)
            prompt_embeds.append(pe)

        if not prompt_embeds:
            return None

        # Pad all to max sequence length in the batch
        max_len = max(pe.shape[0] for pe in prompt_embeds)
        padded = []
        for pe in prompt_embeds:
            if pe.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - pe.shape[0], *pe.shape[1:],
                    dtype=dtype, device=device,
                )
                pe = torch.cat([pe, pad], dim=0)
            padded.append(pe)

        return torch.stack(padded)

    def _extract_encoder_attention_mask(
        self, batch: dict, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Extract and stack encoder_attention_mask from cached embeddings.

        Pads masks to batch-max sequence length (False for padding positions)
        so the model ignores padded token slots.
        """
        embeddings = batch.get("embeddings")
        if not embeddings:
            return None

        masks = []
        for emb in embeddings:
            if emb is None:
                return None
            if isinstance(emb, dict):
                mask = emb.get("prompt_embeds_mask")
            else:
                mask = None
            if mask is None:
                # If no mask, create all-ones mask
                pe = emb.get("prompt_embed") if isinstance(emb, dict) else emb
                if isinstance(pe, np.ndarray):
                    pe = torch.from_numpy(pe)
                mask = torch.ones(pe.shape[0], dtype=torch.bool)
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            masks.append(mask)

        if not masks:
            return None

        # Pad all masks to max length with False
        max_len = max(m.shape[0] for m in masks)
        padded = []
        for m in masks:
            if m.shape[0] < max_len:
                pad = torch.zeros(max_len - m.shape[0], dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded).to(device=device)

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids=None
    ) -> torch.Tensor | list[torch.Tensor]:
        # Handle Transformer2DModelOutput or tuple (when return_dict=False).
        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample
        elif isinstance(model_pred, (tuple, list)):
            model_pred = model_pred[0]

        # Unpack the 3D packed prediction: (B, seq_len, C*p*p) -> (B, C, H, W)
        # per target, using the grids cached by prepare_model_input.  The
        # transformer preserves token order, so the target tokens are the first
        # sum(gh*gw) positions of the sequence.
        if model_pred.dim() == 3:
            B, seq_len, channels = model_pred.shape
            p = self.patch_size
            C = channels // (p * p)

            if not self._target_grids:
                raise ValueError(
                    f"_target_grids is empty — cannot unpack prediction of seq_len={seq_len}. "
                    f"Call prepare_model_input before unpack_prediction."
                )

            target_seq_len = sum(gh * gw for gh, gw in self._target_grids)
            model_pred = model_pred[:, :target_seq_len, :]

            results = []
            offset = 0
            for gh, gw in self._target_grids:
                chunk_len = gh * gw
                chunk = model_pred[:, offset : offset + chunk_len, :]
                latents = chunk.view(B, gh, gw, C, p, p)
                latents = latents.permute(0, 3, 1, 4, 2, 5)
                latents = latents.reshape(B, C, 1, gh * p, gw * p)
                latents = latents.squeeze(2)
                results.append(latents)
                offset += chunk_len
            return results

        return model_pred

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target
