# =============================================================================
# VENDORED COPY - upstream is the source of truth; do not edit in place.
#
# Original author : Ostris, LLC ("ostris") - AI Toolkit
# Upstream repo   : https://github.com/ostris/ai-toolkit
# Upstream file   : extensions_built_in/diffusion_models/flux3/src/positional.py
# Upstream commit : 12395a103fd0ca499c631dc6f460a8778e11490d (2026-10-15)
# Licence         : MIT - Copyright (c) 2024 Ostris, LLC (see ATTRIBUTION.md)
# Vendored        : 2026-10-16 into UnifiedTrainer (verbatim, no code changes)
#
# Why it is here  : Reference packers for the 4-axis (t,h,w,l) RoPE ids; independent witness that UTrainer's _prepare_image_ids/_prepare_txt_ids and the reference-image l = -1-i convention match the shipped model recipe.
# =============================================================================

"""Position-id packers for the FLUX.3 joint sequence.

Every token stream fed to the DiT carries a 4-axis position id ``(t, h, w, l)``.
Image tokens use ``h``/``w`` (``t = 0``); the ``l`` axis separates reference
images (i2i conditioning uses ``l = -1, -2, ...``) and carries the sequence
position for text tokens.
"""

from typing import Callable

import torch
from einops import rearrange
from torch import Tensor


def prc_img(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """(C, H, W) latent -> ((H*W, C) tokens, (H*W, 4) ids)."""
    _, h, w = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids


def prc_txt(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """(L, D) text context -> ((L, D) tokens, (L, 4) ids) with l = arange(L)."""
    assert l_coord is None, "l_coord not supported for txt"
    length, _ = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(length, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids


def _batched(fn: Callable) -> Callable:
    def wrapped(
        x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        pairs = [
            fn(
                x[i],
                t_coord[i] if t_coord is not None else None,
                l_coord[i] if l_coord is not None else None,
            )
            for i in range(len(x))
        ]
        xs, ids = zip(*pairs)
        return torch.stack(xs), torch.stack(ids)

    return wrapped


batched_prc_img = _batched(prc_img)
batched_prc_txt = _batched(prc_txt)
