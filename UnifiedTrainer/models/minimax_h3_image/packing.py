"""Packed-sequence geometry for MiniMax-H3-Image training.

Ported from diffusers' ``modular_pipelines/minimax_h3_image/before_denoise.py`` (layout
helpers) and ``modular_pipelines/minimax_h3/before_denoise.py`` (patchify), so the
training-side packing is bit-identical to the released sampler and stays inside UTrainer.

The packed sequence is ``[text rows | reference-image rows | target-image rows]``:

* every row is either a text token (``text_dim`` features) or one ``(1, 2, 2)`` latent
  patch of an image (``24 * 4`` features);
* ``position_ids`` carries the rotary ``(t, h, w)`` coordinate of every row -- text rows
  walk the time axis, each image is one frame on it, and the spatial axes are normalized
  by the square root of the latent area and scaled by 32;
* ``token_tags`` is 0 for image rows and 1 for text rows (vision-block rows inside a
  reference presentation are tagged 0 too);
* ``timestep_indices`` maps every row to the index of its timestep in the distinct
  timestep table the transformer receives.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# The spatial rotary axes are normalized by the square root of the latent area and scaled by 32.
ROPE_SPATIAL_SCALE = 32

# Modality tags of the packed sequence: MiniMax-H3's AdaLN table is indexed by
# ``timestep_index * MINIMAX_H3_MODALITY_NUM + token_tag``, with 0 = video/image, 1 = text,
# 2 = audio. H3-Image has no audio rows, and image rows use the video slot.
IMAGE_TAG = 0
TEXT_TAG = 1
MODALITY_NUM = 3

# The release's PDD sampling grid (``pdd_config.json``): 6 steps, 6 output heads. Used as a
# fallback when the checkpoint does not ship its own grid.
DEFAULT_PDD_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)


def patchify_latents(latents: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """``(B, C, 1, H, W)`` or ``(B, C, H, W)`` latents -> ``(B, num_patches, C*patch)`` rows.

    Frame-major then row-major, exactly like the released sampler's
    ``patchify_video_latents`` for a single frame.
    """
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}."
        )
    latents = latents.reshape(
        batch_size, channels,
        num_frames // patch_t, patch_t,
        height // patch_h, patch_h,
        width // patch_w, patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(batch_size, -1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_rows(
    rows: torch.Tensor, latent_height: int, latent_width: int, channels: int, patch_size=(1, 2, 2)
) -> torch.Tensor:
    """``(B, num_patches, C*patch)`` rows -> ``(B, C, latent_height, latent_width)``.

    Exact inverse of :func:`patchify_latents` for a single latent frame.
    """
    _, patch_h, patch_w = patch_size
    batch_size = rows.shape[0]
    grid_h = latent_height // patch_h
    grid_w = latent_width // patch_w
    expected = grid_h * grid_w
    if rows.shape[1] != expected:
        raise ValueError(
            f"got {rows.shape[1]} rows for a {latent_height}x{latent_width} latent "
            f"({grid_h}x{grid_w} = {expected} patches)"
        )
    patch_t = patch_size[0]
    frames = 1
    grid_t = frames // patch_t
    rows = rows.reshape(batch_size, grid_t, grid_h, grid_w, channels, patch_t, patch_h, patch_w)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(batch_size, channels, latent_height, latent_width).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis, in the release's exact float64 grid."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> torch.Tensor:
    """The ``(h, w)`` rotary coordinates of one latent frame, shape ``(rows, 2)``."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size=(1, 2, 2),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the ``[text | reference images | target image]`` layout.

    Returns ``(position_ids, token_tags, image_indices, text_indices, num_reference_rows)``
    with the exact semantics of the released sampler.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = int(text_token_tags.shape[0])
    num_reference_rows = sum(
        (height // patch_h) * (width // patch_w) for height, width in reference_grids
    )
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
        cursor = rows.stop
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = _frame_position_grid(height, width, patch_h, patch_w)
        rotary_time += 1.0
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    text_indices = torch.arange(num_text_tokens)
    image_indices = torch.arange(num_text_tokens, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = IMAGE_TAG

    return position_ids, token_tags, image_indices, text_indices, num_reference_rows


def get_image_canvas_size(
    height: int, width: int, min_generation_area: int, multiple: int
) -> tuple[int, int]:
    """The canvas H3-Image generates a ``height x width`` image on."""
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )
