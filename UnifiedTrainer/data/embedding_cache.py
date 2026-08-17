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
            if key.endswith("_scale"):
                continue
            arr = data[key]
            # Dequantize int8 -> fp32 when a matching per-tensor scale is present
            # (see save(): float arrays are stored as int8 + scale). Old fp16
            # caches load unchanged; the trainer casts to bf16 on load anyway.
            if arr.dtype == np.int8 and (key + "_scale") in data.files:
                scale = data[key + "_scale"]
                arr = arr.astype(np.float32) * scale
            result[key] = arr
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

        Floating-point arrays are quantized **directly to int8** with a
        per-tensor absmax scale (float32 -> int8), giving ~4x smaller disk
        usage than float32. The trainer casts to bf16 on load anyway, and
        text-encoder hidden states tolerate uniform 8-bit quantization for
        conditioning. The per-tensor scale is stored alongside as ``<key>_scale``
        and the loader dequantizes back to fp32. Boolean/int arrays are preserved.

        Uses np.savez (uncompressed) — not savez_compressed — because the
        quantized int8 data is nearly incompressible while compressed writes
        are dramatically slower.
        """
        path = Path(npz_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {}
        for key, val in embedding.items():
            if isinstance(val, torch.Tensor):
                arr = val.cpu().numpy()
            elif isinstance(val, np.ndarray):
                arr = val
            else:
                arr = np.array(val)
            # Quantize float32/float64 -> int8 directly (per-tensor absmax scale)
            # for a ~4x disk reduction vs float32.
            if arr.dtype in (np.float32, np.float64) and arr.size:
                f = arr.astype(np.float32)
                amax = float(np.max(np.abs(f)))
                if amax == 0.0:
                    amax = 1.0
                scale = amax / 127.0
                q = np.clip(np.round(f / scale), -127, 127).astype(np.int8)
                save_dict[key] = q
                save_dict[key + "_scale"] = np.float32(scale)
            else:
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
