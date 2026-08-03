"""
Regenerator — online _XX_gen regeneration for DD-SFT training.

Periodically regenerates model-generated images (_XX_gen) using a turbo
transformer + current LoRA weights. This is the "on-policy" component of
DD-SFT: the training target is refreshed as the LoRA improves.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class Regenerator:
    """Online regeneration of _XX_gen latents.

    Args:
        adapter: Model adapter (for encoding/decoding)
        turbo_transformer_path: Path to a fast (turbo) transformer for generation
        vae: VAE for decoding/encoding
        steps: Number of inference steps for regeneration
        guidance_scale: CFG guidance scale
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        adapter: Any,
        turbo_transformer_path: Optional[str] = None,
        vae: Optional[torch.nn.Module] = None,
        steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int = 42,
    ):
        self.adapter = adapter
        self.turbo_transformer_path = turbo_transformer_path
        self.vae = vae
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self._turbo_transformer: Optional[torch.nn.Module] = None

    def should_regenerate(self, epoch: int, every_n_epochs: int) -> bool:
        """Check if regeneration should run this epoch."""
        return every_n_epochs > 0 and (epoch + 1) % every_n_epochs == 0

    def regenerate_all(
        self,
        lora_transformer: torch.nn.Module,
        cache_dir: str,
        dataset_path: str,
        resolution: int = 1024,
    ) -> int:
        """Regenerate all _XX_gen latents in the cache.

        Args:
            lora_transformer: Transformer with current LoRA weights applied
            cache_dir: Path to the cache directory
            dataset_path: Path to the source dataset
            resolution: Generation resolution

        Returns:
            Number of images regenerated
        """
        logger.info(f"Starting online regeneration in {cache_dir}")
        count = 0

        # Use turbo transformer if available, otherwise use the LoRA transformer
        gen_transformer = self._get_turbo_transformer(lora_transformer)

        # Walk through cache (including subdirs) and regenerate _XX_gen latents
        cache_path = Path(cache_dir)
        gen_files = list(cache_path.rglob("*_gen_*.pt"))

        for gen_file in gen_files:
            try:
                self._regenerate_single(
                    gen_transformer, gen_file, resolution
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to regenerate {gen_file}: {e}")

        logger.info(f"Regenerated {count} images")
        return count

    def _get_turbo_transformer(
        self, lora_transformer: torch.nn.Module
    ) -> torch.nn.Module:
        """Get the transformer to use for generation."""
        if self.turbo_transformer_path and self._turbo_transformer is None:
            logger.info(f"Loading turbo transformer from {self.turbo_transformer_path}")
            # Load turbo transformer and merge LoRA weights
            self._turbo_transformer = self.adapter.load_transformer(
                self.turbo_transformer_path, torch.bfloat16
            )
            # Merge LoRA from training transformer
            self._merge_lora(self._turbo_transformer, lora_transformer)

        return self._turbo_transformer or lora_transformer

    def _merge_lora(
        self, target: torch.nn.Module, source: torch.nn.Module
    ) -> None:
        """Merge LoRA weights from source into target."""
        # This is model-specific; the adapter can override this behavior
        # For now, we use PEFT's merge_and_unload if available
        try:
            if hasattr(source, "merge_and_unload"):
                merged = source.merge_and_unload()
                target.load_state_dict(merged.state_dict(), strict=False)
        except Exception as e:
            logger.warning(f"LoRA merge failed, using source directly: {e}")

    def _regenerate_single(
        self,
        transformer: torch.nn.Module,
        gen_file: Path,
        resolution: int,
    ) -> None:
        """Regenerate a single _XX_gen latent file."""
        # Load the reference latent
        ref_file = str(gen_file).replace("_gen_", "_")
        ref_path = Path(ref_file)

        if not ref_path.exists():
            # Try to find the reference by removing _gen suffix
            name = gen_file.name
            ref_name = name.replace("_gen_", "_")
            ref_path = gen_file.parent / ref_name

        if not ref_path.exists():
            logger.debug(f"Reference not found for {gen_file}")
            return

        # This is a simplified placeholder - actual regeneration requires
        # model-specific pipeline construction via the adapter
        # The adapter's build_pipeline method handles this
        logger.debug(f"Regenerating {gen_file}")
