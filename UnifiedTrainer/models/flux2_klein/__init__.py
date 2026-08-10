"""Flux2 Klein 4B model adapter.

Architecture notes (diffusers ``Flux2KleinPipeline`` parity):
- Text encoder: ``Qwen3ForCausalLM`` (Qwen3-4B, hidden 2560) with hidden-state
  layer taps (9, 18, 27) → ``(seq_len, 3 * 2560 = 7680)`` — matches
  ``embedding_dim`` and the klein-4B transformer's ``joint_attention_dim``.
  The full Qwen3 chat template (system preamble + user content + assistant
  suffix) is encoded and ALL tokens are kept (no preamble stripping, unlike the
  Krea-2 adapter) — matching the reference pipeline.
- VAE: ``AutoencoderKLFlux2`` (16 latent channels, shift+scaling).
- Transformer: ``Flux2Transformer2DModel`` — Flux-2 dual-stream + single-stream
  MMDiT.  Latents are patchified 2×2 *by the pipeline* (the transformer config
  itself reports ``patch_size=1``), so this adapter packs ``(B, C, H, W)``
  latents into ``(B, (H/2)*(W/2), C*4)`` tokens before the forward.
- Position IDs: 4-axis RoPE coordinates ``(T, H, W, L)``.  Text tokens sit at
  ``(0, 0, 0, token_idx)``; target image tokens at ``(0, h, w, 0)``; reference
  image tokens at ``(10 + 10*i, h, w, 0)`` — matching the pipeline's
  ``_prepare_text_ids`` / ``_prepare_latent_ids`` / ``_prepare_image_ids``.
- Timestep: the transformer scales by 1000 internally (``timestep * 1000`` in
  forward), so training sigmas in [0, 1] are passed through unchanged (no
  pre-scaling — the old ``(sigmas * 1000).long()`` shell was wrong).
- Guidance: optional.  The Klein reference pipeline calls ``forward`` with
  ``guidance=None``; a config ``"guidance"`` value is honored when the user's
  variant is guidance-distilled and needs the guidance embedding branch.
"""
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
    1024: [
        (512, 1280), (576, 1408), (640, 1344), (704, 1280),
        (768, 1216), (832, 1152), (896, 1088), (960, 1024),
        (1024, 960), (1088, 896), (1152, 832), (1216, 768),
        (1280, 704), (1344, 640), (1408, 576), (1280, 512),
    ],
}

# Qwen3-4B layer taps — the diffusers Flux2KleinPipeline convention.
# 3 taps × 2560 = 7680, matching ``embedding_dim`` and the klein-4B
# transformer's ``joint_attention_dim``.
DEFAULT_TEXT_ENCODER_SELECT_LAYERS = (9, 18, 27)

# Matches the pipeline's ``tokenizer_max_length``.
_DEFAULT_MAX_SEQ_LEN = 512

# Flux2 latents are turned into 2x2 patches and packed (pipeline-level
# patchification; the transformer config reports patch_size=1).
FLUX2_LATENT_PATCH_SIZE = 2

# T-coordinate separation between target and reference image tokens
# (matches the pipeline's ``_prepare_image_ids`` scale=10).
_REF_T_COORD_SCALE = 10


@ModelRegistry.register("flux2_klein")
class Flux2KleinAdapter(BaseModelAdapter):
    """Adapter for Flux2 Klein 4B model."""

    name = "flux2_klein"

    # Flux2 latents are 2x2-patchified and packed before the transformer, so the
    # framework's bucket divisibility must account for the patch grid as well
    # (vae_scale * patch_size = 8 * 2 = 16).  This deviates from the base
    # docstring's "patch=1" note, which reflects the transformer config's own
    # (unused) patch_size — the pipeline-level patchify is what matters.
    patch_size = FLUX2_LATENT_PATCH_SIZE

    def __init__(self, config: dict):
        self.config = config
        self._model_path = config.get("model_path", "")

        # Text encoder layer taps (Qwen3-4B convention for klein).
        self.text_encoder_select_layers: tuple = tuple(
            config.get("text_encoder_select_layers", DEFAULT_TEXT_ENCODER_SELECT_LAYERS)
        )

        # Chat-template token budget — matches the reference pipeline.
        self.max_sequence_length: int = config.get("max_sequence_length", _DEFAULT_MAX_SEQ_LEN)

        # Guidance embedding.  None = pass no guidance to the transformer
        # (matches the Flux2KleinPipeline, which calls forward(guidance=None)).
        # Set a value (e.g. 3.5) in the user config for guidance-distilled
        # variants whose checkpoints were trained with a guidance embedding.
        self.guidance: Optional[float] = config.get("guidance", None)
        if self.guidance is not None:
            self.guidance = float(self.guidance)

        # Target grid dimensions cached from prepare_model_input — used by
        # unpack_prediction to slice/unpack each target (krea2-style).
        self._target_grids: list[tuple[int, int]] = []

    # ── Model loading ──────────────────────────────────────────────────

    def load_transformer(self, path: str, dtype: torch.dtype) -> nn.Module:
        from .transformer_flux2 import Flux2Transformer2DModel
        return Flux2Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    def load_vae(self, path: str, dtype: torch.dtype) -> nn.Module:
        from diffusers.models import AutoencoderKLFlux2
        return AutoencoderKLFlux2.from_pretrained(path, torch_dtype=dtype)

    def load_scheduler(self, path: str) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_pretrained(path)

    def load_text_encoder(self, path: str, dtype: torch.dtype) -> Optional[nn.Module]:
        """Load the FLUX.2-klein text encoder.

        FLUX.2-klein ships a **Qwen3-4B** causal LM as its text encoder
        (``qwen_3_4b.safetensors`` / ``Qwen3ForCausalLM``), not Qwen2.5-VL.
        ``output_hidden_states`` is enabled at encode time and selected layers
        are stacked into the 7680-dim joint embedding the transformer's
        ``context_embedder`` (``joint_attention_dim=7680``) consumes.
        """
        try:
            from transformers import Qwen3ForCausalLM
            return Qwen3ForCausalLM.from_pretrained(path, torch_dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to load Flux2-Klein text encoder (Qwen3): {e}")
            return None

    def load_tokenizer(self, path: str) -> Optional[Any]:
        """Load the Qwen2 tokenizer family used by FLUX.2-klein."""
        try:
            from transformers import Qwen2TokenizerFast
            return Qwen2TokenizerFast.from_pretrained(path)
        except Exception as e:
            logger.warning(f"Failed to load Flux2-Klein tokenizer (Qwen2TokenizerFast): {e}")
            try:
                from transformers import Qwen2Tokenizer
                return Qwen2Tokenizer.from_pretrained(path)
            except Exception:
                return None

    # ── Architecture specs ─────────────────────────────────────────────

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
        return True

    # ── Encoding ───────────────────────────────────────────────────────

    def encode_image(self, vae: nn.Module, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
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
        """Encode a text prompt via the Qwen3 text encoder.

        Follows the diffusers ``Flux2KleinPipeline._get_qwen3_prompt_embeds``
        convention: apply the Qwen3 chat template (``enable_thinking=False``,
        ``add_generation_prompt=True``), tokenize padded to
        ``max_sequence_length``, and stack the configured hidden-state layer
        taps into ``(seq_len, num_layers * hidden_dim)``.

        ``reference_image`` / ``processor`` are accepted (the cache builder
        passes them unconditionally) but ignored — klein text conditioning is
        text-only; reference images enter via latents in ``prepare_model_input``.

        Returns
        -------
        dict
            ``prompt_embed``       — ``(seq_len, dim)``
            ``prompt_embeds_mask`` — ``(seq_len,)`` bool
        """
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "Flux2 Klein requires a loaded Qwen3 text encoder and tokenizer."
            )

        # ── 1. Build prompt text (Qwen3 chat template) ──────────────────
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers / templates without thinking-mode support.
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # ── 2. Tokenize ─────────────────────────────────────────────────
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # ── 3. Run encoder ──────────────────────────────────────────────
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # ── 4. Extract hidden states ────────────────────────────────────
        if self.text_encoder_select_layers and getattr(outputs, "hidden_states", None):
            # Stack selected layer taps: (B, L, seq, dim) → (B, seq, L*dim).
            out = torch.stack(
                [outputs.hidden_states[i] for i in self.text_encoder_select_layers],
                dim=1,
            )
            hidden = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif getattr(outputs, "hidden_states", None):
            hidden = outputs.hidden_states[-1]
        else:
            raise RuntimeError(
                "Flux2-Klein text encoder output has neither last_hidden_state "
                "nor hidden_states — cannot build prompt embeddings."
            )

        hidden = hidden.to(dtype=dtype, device=device)

        # ── 5. Build result ─────────────────────────────────────────────
        return {
            "prompt_embed": hidden[0],
            "prompt_embeds_mask": attention_mask[0].to(torch.bool),
        }

    def decode_latent(self, vae: nn.Module, latent: torch.Tensor) -> Any:
        latent = latent / vae.config.scaling_factor + vae.config.shift_factor
        with torch.no_grad():
            image = vae.decode(latent).sample
        return image

    # ── Latent packing ─────────────────────────────────────────────────

    def _patchify_and_pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, (H/p)*(W/p), C*p*p) — Flux2 pipeline patchify+pack.

        Mirrors the reference pipeline's ``_patchify_latents`` followed by
        ``_pack_latents`` (2×2 spatial patch folded into the channel axis,
        then the grid flattened in row-major order).
        """
        p = self.patch_size
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // p, p, W // p, p)
        latents = latents.permute(0, 1, 3, 5, 2, 4)  # (B, C, p, p, H//p, W//p)
        latents = latents.reshape(B, C * p * p, H // p, W // p)
        latents = latents.reshape(B, C * p * p, (H // p) * (W // p))
        return latents.permute(0, 2, 1)  # (B, (H//p)*(W//p), C*p*p)

    # ── Position IDs ───────────────────────────────────────────────────

    @staticmethod
    def _prepare_txt_ids(text_seq_len: int, device: torch.device) -> torch.Tensor:
        """Build 4-axis RoPE ids for text tokens: (1, T, 4) with (0, 0, 0, idx).

        Matches the pipeline's ``_prepare_text_ids`` — token index runs along
        the last axis.
        """
        t = torch.arange(1, device=device)
        h = torch.arange(1, device=device)
        w = torch.arange(1, device=device)
        l = torch.arange(text_seq_len, device=device)
        return torch.cartesian_prod(t, h, w, l).unsqueeze(0)

    @staticmethod
    def _prepare_image_ids(
        grids: list[tuple[int, int]],
        t_values: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build 4-axis RoPE ids for image tokens: (sum(H*W), 4) with (T, h, w, 0).

        Target tokens get T=0; reference tokens get positive T coordinates
        (``_REF_T_COORD_SCALE + _REF_T_COORD_SCALE * i``) so their RoPE frame
        index is distinct — exactly the pipeline's ``_prepare_latent_ids`` /
        ``_prepare_image_ids`` convention.
        """
        parts = []
        for (gh, gw), t_val in zip(grids, t_values):
            t = torch.tensor([t_val], device=device)
            h = torch.arange(gh, device=device)
            w = torch.arange(gw, device=device)
            l = torch.arange(1, device=device)
            parts.append(torch.cartesian_prod(t, h, w, l))
        return torch.cat(parts, dim=0)

    # ── Training hooks ─────────────────────────────────────────────────

    def prepare_model_input(
        self,
        batch: dict,
        noise: torch.Tensor | list[torch.Tensor],
        sigmas: torch.Tensor,
    ) -> dict:
        """Assemble the kwargs for ``Flux2Transformer2DModel.forward``.

        Target (noisy) latents are packed into tokens; reference latents are
        appended as extra sequence tokens with a distinct RoPE T-coordinate
        (the Flux2 token-level reference design, not channel concatenation).
        Cached text embeddings are extracted, zero-padded to the batch-max
        sequence length and stacked into ``(B, max_seq, dim)``.
        """
        latents = batch.get("latents", {})

        # Resolve reference from batch_config — user-defined keys, NOT hardcoded.
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else {}
        ref_key = resolved_bc.get("reference_config")
        ref = latents.get(ref_key) if ref_key else None

        p = self.patch_size

        # ── Target (noisy) latents — support single or multiple targets ──
        noises = noise if isinstance(noise, list) else [noise]
        device = noises[0].device
        dtype = noises[0].dtype
        packed_parts: list[torch.Tensor] = []
        target_grids: list[tuple[int, int]] = []
        for n in noises:
            B, C, H, W = n.shape
            target_grids.append((H // p, W // p))
            packed_parts.append(self._patchify_and_pack_latents(n))

        # Cache target grids for unpack_prediction (krea2-style, no sqrt guesses).
        self._target_grids = target_grids

        # ── Reference latents — extra sequence tokens with T>0 RoPE frame ──
        refs = ref if isinstance(ref, list) else ([ref] if ref is not None else [])
        ref_grids: list[tuple[int, int]] = []
        if refs:
            for ref_tensor in refs:
                ref_tensor = ref_tensor.to(device=device, dtype=dtype)
                ref_B, ref_C, ref_H, ref_W = ref_tensor.shape
                ref_grids.append((ref_H // p, ref_W // p))
                packed_parts.append(self._patchify_and_pack_latents(ref_tensor))

        # Token sequence: [targets..., refs...]
        model_input = torch.cat(packed_parts, dim=1)

        # ── Text conditioning from cached embeddings ────────────────────
        encoder_hidden_states = self._extract_encoder_hidden_states(batch, device, dtype)
        encoder_attention_mask = self._extract_encoder_attention_mask(batch, device)
        if encoder_hidden_states is None:
            raise ValueError(
                "Flux2 Klein requires encoder_hidden_states in the batch. "
                "Ensure the data pipeline provides pre-computed text encoder hidden states "
                "(cached via encode_text)."
            )
        text_seq_len = encoder_hidden_states.shape[1]

        batch_size = model_input.shape[0]

        # ── Position IDs ────────────────────────────────────────────────
        txt_ids = self._prepare_txt_ids(text_seq_len, device)  # (1, T, 4)
        img_ids = self._prepare_image_ids(
            target_grids + ref_grids,
            [0] * len(target_grids)
            + [
                _REF_T_COORD_SCALE + _REF_T_COORD_SCALE * i
                for i in range(len(ref_grids))
            ],
            device,
        )  # (S_total, 4)
        # Expand to batch (all batch items share the same grids).
        img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)
        txt_ids = txt_ids.expand(batch_size, -1, -1)

        # Flux2 transformer scales timestep by 1000 internally — pass [0, 1].
        timesteps = sigmas.to(device=device, dtype=dtype)

        result: dict[str, Any] = {
            "hidden_states": model_input,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timesteps,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "return_dict": False,
        }

        # Guidance embedding — only when the user configured it (the reference
        # Klein pipeline runs with guidance=None).
        if self.guidance is not None:
            result["guidance"] = torch.full(
                (batch_size,), self.guidance, device=device, dtype=dtype
            )

        # NOTE: the Flux2 forward does NOT accept an encoder_attention_mask
        # (joint attention over text+image tokens is unmasked; RoPE positions
        # disambiguate tokens).  The mask is still extracted above so padded
        # text slots are handled consistently, but it is deliberately NOT
        # forwarded to the transformer — an unexpected kwarg would TypeError.
        del encoder_attention_mask

        return result

    def unpack_prediction(
        self, model_pred: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor | list[torch.Tensor]:
        # Handle Transformer2DModelOutput or tuple (when return_dict=False).
        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample
        elif isinstance(model_pred, (tuple, list)):
            model_pred = model_pred[0]

        if model_pred.dim() == 3:
            B, seq_len, channels = model_pred.shape
            p = self.patch_size
            C = channels // (p * p)

            # Use cached target grids (set by prepare_model_input).
            # The forward output keeps the SAME token order as the input
            # sequence ([targets..., refs...]) — slice off the target tokens.
            if not self._target_grids:
                raise ValueError(
                    f"_target_grids is empty — cannot unpack prediction of seq_len={seq_len}. "
                    f"Call prepare_model_input before unpack_prediction."
                )
            target_seq_len = sum(gh * gw for gh, gw in self._target_grids)
            model_pred = model_pred[:, :target_seq_len, :]

            # Unpack each target: (B, S, C*p*p) -> (B, C, H, W).
            results = []
            offset = 0
            for gh, gw in self._target_grids:
                chunk_len = gh * gw
                chunk = model_pred[:, offset : offset + chunk_len, :]
                latents = chunk.permute(0, 2, 1).reshape(B, C * p * p, gh, gw)
                latents = latents.reshape(B, C, p, p, gh, gw)
                latents = latents.permute(0, 1, 4, 2, 5, 3)
                latents = latents.reshape(B, C, gh * p, gw * p)
                results.append(latents)
                offset += chunk_len
            return results

        return model_pred

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target

    # ── Helpers ────────────────────────────────────────────────────────

    def _extract_encoder_hidden_states(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Extract and stack prompt embeddings from cached per-sample dicts.

        The data pipeline stores per-sample embeddings as dicts with
        ``prompt_embed`` of shape (seq_len, dim).  Different captions may have
        different seq_len, so all are zero-padded to the batch-max before
        stacking.

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
            # Cast immediately — avoids float32 intermediates piling up.
            pe = pe.to(device=device, dtype=dtype)
            if pe.ndim != 2:
                raise ValueError(
                    f"Flux2 Klein expects cached prompt_embed of shape (seq_len, dim), "
                    f"got shape {tuple(pe.shape)} with ndim={pe.ndim}. "
                    "Recreate the embedding cache for this model."
                )
            prompt_embeds.append(pe)

        if not prompt_embeds:
            return None

        # Pad all to max sequence length in the batch.
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
        """Extract and stack prompt attention masks from cached embeddings.

        Pads masks to batch-max sequence length with False so padded token
        slots stay marked as padding (mirrors krea2's helper; the Flux2
        forward itself does not consume a text mask, but the padded length
        bookkeeping is shared).
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
                # No mask cached → all-ones mask.
                pe = emb.get("prompt_embed") if isinstance(emb, dict) else emb
                if isinstance(pe, np.ndarray):
                    pe = torch.from_numpy(pe)
                mask = torch.ones(pe.shape[0], dtype=torch.bool)
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            masks.append(mask)

        if not masks:
            return None

        # Pad all masks to max length with False.
        max_len = max(m.shape[0] for m in masks)
        padded = []
        for m in masks:
            if m.shape[0] < max_len:
                pad = torch.zeros(max_len - m.shape[0], dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded).to(device=device)
