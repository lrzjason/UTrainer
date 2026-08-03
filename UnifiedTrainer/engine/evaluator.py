"""
Evaluator — validation loop for generating sample images per epoch.

Runs inference with the current LoRA-adapted model to produce validation
images, allowing visual monitoring of training progress.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)


class Evaluator:
    """Validation evaluator that generates sample images.

    Args:
        adapter: Model adapter (provides build_pipeline, decode_latent)
        output_dir: Where to save validation images
        prompts: List of validation prompts
        resolution: Validation image resolution
        num_inference_steps: Steps for validation generation
        guidance_scale: CFG guidance scale
    """

    def __init__(
        self,
        adapter: Any,
        output_dir: str = "output/val",
        prompts: Optional[List[str]] = None,
        resolution: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        seed: int = 42,
    ):
        self.adapter = adapter
        self.output_dir = output_dir
        self.prompts = prompts or ["a beautiful landscape"]
        self.resolution = resolution
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        transformer: torch.nn.Module,
        vae: Optional[torch.nn.Module],
        epoch: int,
        pipeline: Optional[Any] = None,
    ) -> List[str]:
        """Generate validation images for the current epoch.

        Returns list of saved image paths.
        """
        saved_paths = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        for i, prompt in enumerate(self.prompts):
            try:
                image = self._generate(
                    transformer, vae, pipeline, prompt, generator
                )
                if image is not None:
                    path = os.path.join(
                        self.output_dir, f"val_epoch{epoch}_{i}.png"
                    )
                    image.save(path)
                    saved_paths.append(path)
                    logger.info(f"Saved validation image: {path}")
            except Exception as e:
                logger.warning(f"Validation generation failed for prompt {i}: {e}")

        return saved_paths

    def _generate(
        self,
        transformer: torch.nn.Module,
        vae: Optional[torch.nn.Module],
        pipeline: Optional[Any],
        prompt: str,
        generator: torch.Generator,
    ) -> Any:
        """Generate a single validation image."""
        if pipeline is not None:
            # Use full pipeline if available
            result = pipeline(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )
            return result.images[0] if hasattr(result, "images") else result[0]

        # Fallback: manual generation via adapter
        logger.warning("No pipeline available, skipping validation generation")
        return None
