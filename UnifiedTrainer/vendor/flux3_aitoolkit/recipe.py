# =============================================================================
# VENDORED / EXTRACTED COPY - upstream is the source of truth; do not edit in place.
#
# Original author : Ostris, LLC ("ostris") - AI Toolkit
# Upstream repo   : https://github.com/ostris/ai-toolkit
# Upstream files  : extensions_built_in/diffusion_models/flux3/flux3.py
#                   extensions_built_in/diffusion_models/flux3/src/pipeline.py
# Upstream commit : 12395a103fd0ca499c631dc6f460a8778e11490d (2026-10-15)
# Licence         : MIT - Copyright (c) 2024 Ostris, LLC (see ATTRIBUTION.md)
# Vendored        : 2026-10-16 into UnifiedTrainer
#
# Changes vs upstream: this is a *selection* of training-relevant constants and
# helpers pulled out of the two files above (the upstream modules are a full
# BaseModel subclass and an inference pipeline, neither of which is portable).
# Every value is copied unchanged; the docstrings say where each one lives
# upstream and how it lines up with UTrainer's flux3_klein adapter.
#
# Why it is here : these are the numbers the shipped checkpoint was trained
#   with. UTrainer's adapter currently derives the training sigma schedule from
#   a logit-normal (or uniform "sigma") base and ignores the 6.93 static
#   timeshift the model's own sampler uses, and it has no reference-image
#   area-matching policy.
# =============================================================================
"""FLUX.3 klein training recipe constants, extracted from ai-toolkit.

Nothing here is imported at runtime by an adapter yet - it is the reference
value set used to audit (and, if wanted, align) UTrainer's flux3_klein path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sampling / schedule
# ---------------------------------------------------------------------------

# pipeline.py: DEFAULT_ALPHA. flux3.py pins the *training* timestep schedule to
# the same value: `DEFAULT_TIMESTEP_SHIFT = DEFAULT_ALPHA` ->
# CustomFlowMatchEulerDiscreteScheduler(shift=6.93, num_train_timesteps=1000,
# use_dynamic_shifting=False).
DEFAULT_ALPHA = 6.93
DEFAULT_TIMESTEP_SHIFT = DEFAULT_ALPHA
NUM_TRAIN_TIMESTEPS = 1000
USE_DYNAMIC_SHIFTING = False
SCHEDULER_CONFIG = {
    "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
    "use_dynamic_shifting": USE_DYNAMIC_SHIFTING,
}

# pipeline.py: DEFAULT_GUIDANCE (base model / CFG sampling).
DEFAULT_GUIDANCE = 3.5

# pipeline.py: IMAGE_ROPE_SCALE, applied to the h/w axes of every image token.
IMAGE_ROPE_SCALE = 0.5

# pipeline.py: I2I_COND_L; reference i rides at l = -1 - i.
I2I_COND_L = -1


def timeshift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """Flow-matching static timeshift (upstream pipeline.py)."""
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def build_timesteps(n_steps: int, alpha: float) -> torch.Tensor:
    """Timeshifted uniform grid, t=1 (noise) -> t=0 (clean) (pipeline.py)."""
    return timeshift(alpha, torch.linspace(1.0, 0.0, n_steps + 1))


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def get_bucket_divisibility() -> int:
    """flux3.py Flux3Model.get_bucket_divisibility: f8 conv + 2x2 packing."""
    return 16


# ---------------------------------------------------------------------------
# Reference (image_cond) sizing
# ---------------------------------------------------------------------------


def fit_reference(
    img: torch.Tensor, target_pixels: int, match_target_res: bool = True
) -> torch.Tensor:
    """Size a ``(1, C, H, W)`` reference for the DiT (flux3.py ``_fit_reference``).

    References keep their own aspect ratio on their own position grid;
    ``match_target_res`` scales the area to the target's, otherwise the image
    only scales down when it is larger than the target area. Snapped to the
    16-px grid.

    UTrainer difference: the cache builder buckets each reference on its own
    (``BucketSystem.find_bucket_for_image(resolution, image)`` at the
    ``image_configs.<key>.resolution``), so a reference's token count can differ
    from the target's. The upstream default area-matches them.
    """
    sc = get_bucket_divisibility()
    h, w = img.shape[-2], img.shape[-1]
    if match_target_res or h * w > target_pixels:
        ratio = h / w
        new_h = math.sqrt(target_pixels * ratio)
        new_w = new_h / ratio
    else:
        new_h, new_w = float(h), float(w)
    new_h = max(sc, int(round(new_h / sc)) * sc)
    new_w = max(sc, int(round(new_w / sc)) * sc)
    if (new_h, new_w) != (h, w):
        img = F.interpolate(img.float(), size=(new_h, new_w), mode="bilinear")
    return img


def target_pixels_from_latent(h_lat: int, w_lat: int) -> int:
    """flux3.py: ``target_pixels = h_lat * w_lat * 16 * 16``."""
    return h_lat * w_lat * 16 * 16


# ---------------------------------------------------------------------------
# Quantization / LoRA naming
# ---------------------------------------------------------------------------

# flux3.py Flux3Model.get_quantization_exclude_modules: sensitive small modules
# kept in full precision (fnmatch patterns). Worth mirroring in UTrainer's
# torchao quantize_model calls for flux3_klein.
QUANTIZATION_EXCLUDE_MODULES = [
    "emb_in*",
    "txt_in",
    "time_in*",
    "*modulations*",
    "final_layer*",
]

# flux3.py convert_lora_weights_before_{save,load}: UTrainer saves LoRA/LoKr
# under the diffusers module names, ComfyUI expects ``diffusion_model.<native>``.
LORA_SAVE_FROM_PREFIX = "transformer."
LORA_SAVE_TO_PREFIX = "diffusion_model."

# flux3.py Flux3Model.latent_space_version -- the cache-invalidation tag for the
# packed + BatchNorm-normalized latent space.
LATENT_SPACE_VERSION = "flux3_v1"


def convert_lora_weights_before_save(state_dict: dict) -> dict:
    """``transformer.*`` -> ``diffusion_model.*`` (flux3.py, verbatim logic)."""
    return {k.replace("transformer.", "diffusion_model."): v for k, v in state_dict.items()}


def convert_lora_weights_before_load(state_dict: dict) -> dict:
    """``diffusion_model.*`` -> ``transformer.*`` (flux3.py, verbatim logic)."""
    return {k.replace("diffusion_model.", "transformer."): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Checkpoint-variant policy
# ---------------------------------------------------------------------------


def flow_map_supported() -> bool:
    """Whether ai-toolkit can train the step-distilled FLUX.3 checkpoint.

    Upstream: no. ``Flux3Transformer.convert_state_dict_on_load`` raises when it
    sees ``ref_time_in.*`` keys, ``Flux3Model._load_transformer`` raises on a
    package ``dit.flow_map = true``, and the ``*_sd_*`` comfy repack files are
    excluded from the candidate list.

    UTrainer: yes - the vendored diffusers transformer carries
    ``target_timestep_embedder`` and the adapter passes ``target_timestep = 0``.
    """
    return False


def bf16_reference_file(comfy_files) -> Optional[str]:
    """The plain-bf16 member of a comfy candidate list (used for a fallback
    message when a quantized repack cannot be consumed)."""
    for name in comfy_files or ():
        if "bf16" in name:
            return name
    return None
