"""Qwen-Image-Edit model adapter -image editing with reference conditioning."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from UnifiedTrainer.registry import ModelRegistry

# Qwen-Image-Edit shares the Qwen-Image text-encoding pipeline (same
# Qwen2.5-VL text encoder, same chat template, same npz embedding cache
# format).  Inherit the proven implementation instead of duplicating it.
from ..qwen_image import QwenImageAdapter

RESOLUTION_CONFIG = {
    2048: [
        (1024, 2560), (1152, 2816), (1280, 3072), (1280, 2816),
        (1408, 3008), (1536, 2816), (1664, 2688), (1792, 2368),
        (1920, 2240), (2048, 2560), (2176, 2304), (2048, 2048),
        (2048, 2304),
    ],
}


@ModelRegistry.register("qwen_image_edit")
class QwenImageEditAdapter(QwenImageAdapter):
    """Adapter for Qwen-Image-Edit image editing model.

    Inherits from :class:`QwenImageAdapter` — the two models share the
    Qwen2.5-VL text encoder, the Qwen-Image chat template, the VAE, and the
    patchified DiT transformer.

    Reference-image conditioning flows through the multimodal prompt path:
    ``encode_text(reference_image=..., processor=...)`` embeds the reference
    as ``Picture N:`` image tokens in the user message (inherited unchanged).
    The transformer's configured ``in_channels`` is 64 (verified against the
    real Qwen/Qwen-Image-Edit-2509 ``transformer/config.json``), so a
    channel-concat of the reference latent (16+16ch -> token dim 128) is
    architecturally impossible with this repo transformer — the edit path
    therefore does NOT concat reference latents and prepares the noisy target
    exactly like the base text-to-image adapter.
    """

    name = "qwen_image_edit"
    patch_size = 2  # Qwen transformer patches latents in 2x2 blocks

    @property
    def supports_image_conditioning(self) -> bool:
        return True

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

        Inherits the full Qwen-Image chat-template + multimodal path from
        ``QwenImageAdapter``: when *reference_image* and *processor* are both
        provided the reference images are embedded as ``Picture N:``
        placeholders in the user message and forwarded to the Qwen2.5-VL
        vision encoder.  The text-only path uses the standard Qwen-Image
        chat template.

        Returns
        -------
        dict
            ``prompt_embed``        — ``(seq_len, dim)``
            ``prompt_embeds_mask``  — ``(seq_len,)`` bool
            ``image_token_mask``    — ``(seq_len,)`` bool  *(multimodal only)*
        """
        return super().encode_text(
            text_encoder,
            tokenizer,
            prompt,
            device,
            dtype,
            reference_image=reference_image,
            processor=processor,
        )

    def prepare_model_input(
        self, batch: dict, noise: torch.Tensor | list[torch.Tensor], sigmas: torch.Tensor
    ) -> dict:
        """Patchify+pack the noisy target latent only (no reference concat).

        The Qwen-Image-Edit transformer is the standard Qwen-Image transformer
        with ``in_channels=64`` (verified against the real model config), so a
        channel-concatenated reference latent (C=32 -> token dim 128) cannot be
        fed through ``img_in``.  Reference conditioning is therefore carried by
        the multimodal prompt path (reference images -> ``Picture N:`` image
        tokens in the prompt, handled by ``encode_text`` with
        ``reference_image``/``processor``).  Any reference latents in the batch
        are intentionally ignored here.
        """
        return super().prepare_model_input(batch, noise, sigmas)

    def compute_target(self, noise: torch.Tensor, learning_target: torch.Tensor) -> torch.Tensor:
        return noise - learning_target
