"""
EmbeddingCache -caption embedding cache (npz read/write).

Caption embeddings are pre-computed and stored as .npz files containing:
    prompt_embed: [seq_len, dim] -text encoder output
    prompt_embeds_mask: attention mask
    prompt_embed_length: valid sequence length

# Reference: adapted from ai-toolkit cache_text_embeddings config pattern.
# After pre-encoding all prompts, the text encoder is unloaded to CPU
# to free VRAM for the transformer training phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import gc

import numpy as np
import torch
import torch.nn as nn


class EmbeddingCache:
    """Read and write caption embedding .npz files."""

    @staticmethod
    def load(npz_path: str) -> Optional[dict]:
        """Load a caption embedding from .npz file.

        Returns dict with keys: 'prompt_embed', 'prompt_embeds_mask',
        'prompt_embed_length' (or None if file doesn't exist).
        """
        path = Path(npz_path)
        if not path.exists():
            return None

        data = np.load(str(path), allow_pickle=True)
        result = {}
        for key in data.files:
            result[key] = data[key]
        return result

    @staticmethod
    def load_tensor(
        npz_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[dict]:
        """Load embedding and convert to torch tensors on device."""
        result = EmbeddingCache.load(npz_path)
        if result is None:
            return None

        tensor_result = {}
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                tensor_result[key] = torch.from_numpy(val).to(device, dtype)
            else:
                tensor_result[key] = val
        return tensor_result

    @staticmethod
    def save(npz_path: str, embedding: dict) -> Path:
        """Save a caption embedding to .npz file.

        Floating-point arrays are cast to float16 before saving to halve
        disk usage (~50% reduction vs float32).  Text encoder outputs have
        sufficient precision in fp16 for training — the trainer casts to
        bf16 on load anyway.  Boolean arrays (masks) are preserved as-is.

        Uses np.savez (uncompressed) — not savez_compressed — because fp16
        float data is nearly incompressible (only ~7% gain) while compressed
        writes are ~42x slower (337ms vs 8ms per 10MB file).
        """
        path = Path(npz_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert torch tensors to numpy, cast floats to fp16
        save_dict = {}
        for key, val in embedding.items():
            if isinstance(val, torch.Tensor):
                arr = val.cpu().numpy()
            elif isinstance(val, np.ndarray):
                arr = val
            else:
                arr = np.array(val)
            # Cast float32/float64 → float16 for 50% disk reduction
            if arr.dtype in (np.float32, np.float64):
                arr = arr.astype(np.float16)
            save_dict[key] = arr

        np.savez(str(path), **save_dict)
        return path

    @staticmethod
    def exists(npz_path: str) -> bool:
        """Check if an embedding cache file exists."""
        return Path(npz_path).exists()


# ── Batch pre-encoding + TE unload ──────────────────────────────────────────

def cache_text_embeddings(
    text_encoder: nn.Module,
    tokenizer,
    prompts: list[str],
    cache_paths: list[str],
    encode_fn,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-encode all prompts and save to .npz files, then unload the text encoder.

    This mirrors AI Toolkit's ``cache_text_embeddings`` config: run the text
    encoder over the entire dataset *before* training, cache results to disk,
    then move the text encoder to CPU and free VRAM for the transformer.

    Args:
        text_encoder: the text encoder model (moved to device if needed)
        tokenizer: tokenizer instance
        prompts: list of prompt strings
        cache_paths: matching list of output .npz paths
        encode_fn: callable(text_encoder, tokenizer, prompt, device, dtype) -> dict
        device: CUDA device
        dtype: target dtype
    """
    from UnifiedTrainer.utils.flush import flush

    text_encoder.to(device)
    text_encoder.eval()

    for prompt, npz_path in zip(prompts, cache_paths):
        if EmbeddingCache.exists(npz_path):
            continue
        with torch.no_grad():
            embedding = encode_fn(text_encoder, tokenizer, prompt, device, dtype)
        EmbeddingCache.save(npz_path, embedding)

    # Unload text encoder to CPU after caching -frees VRAM for transformer
    unload_text_encoder(text_encoder)
    flush()


def unload_text_encoder(text_encoder: nn.Module) -> None:
    """Move text encoder to CPU and free VRAM.

    Reference: adapted from ai-toolkit cache_text_embeddings + TE unload pattern.
    """
    text_encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
