# =============================================================================
# VENDORED COPY - upstream is the source of truth; do not edit in place.
#
# Original author : Ostris, LLC ("ostris") - AI Toolkit
# Upstream repo   : https://github.com/ostris/ai-toolkit
# Upstream file   : extensions_built_in/diffusion_models/flux3/src/checkpoints.py
# Upstream commit : 12395a103fd0ca499c631dc6f460a8778e11490d (2026-10-15)
# Licence         : MIT - Copyright (c) 2024 Ostris, LLC (see ATTRIBUTION.md)
# Vendored        : 2026-10-16 into UnifiedTrainer (verbatim, no code changes)
#
# Why it is here  : Names/locations of the BFL vendor package and of BFL's single-file ComfyUI repack (bf16 / int8_convrot / nvfp4), plus the note that the repack's _sd_ files are the step-distilled flow-map model.
# =============================================================================

"""Where FLUX.3 image-klein weights come from.

BASE_REPO, the vendor package (model.safetensors + ae.safetensors +
config.json), is the default and the download source. Its files are
byte-identical to the bf16 DiT and VAE of COMFY_REPO, BFL's single-file
ComfyUI repack, and are stored under the repack's names; a repack file already
in the local ComfyUI models folder (int8_convrot = the toolkit's convrot8
storage, attached as-is; bf16; nvfp4) is used instead of the package's.
Nothing is downloaded from the repack unless it is named as `name_or_path`.
The repack's ``_sd_`` files are the step-distilled flow-map model: never
candidates, and a loaded one is rejected by its ``ref_time_in`` keys.
"""

BASE_REPO = "black-forest-labs/flux-3-image-klein-base"
COMFY_REPO = "black-forest-labs/flux-3-image-klein-comfy"

# repo-relative comfy DiT files; the resolver orders them by the requested
# qtype (convrot8 -> int8, nvfp4 -> nvfp4, anything else -> bf16). The text
# encoder stays on the Comfy-Org Qwen3-VL repack (src/text_encoder.py).
COMFY_TRANSFORMER_FILES = [
    "diffusion_models/flux3_image_klein_int8_convrot.safetensors",
    "diffusion_models/flux3_image_klein_bf16.safetensors",
    "diffusion_models/flux3_image_klein_nvfp4.safetensors",
]
# same weights as Comfy-Org's FLUX.2 VAE file of the same name (that one is in
# the diffusers key layout; the loader accepts both)
COMFY_VAE_FILE = "vae/flux2-vae.safetensors"

# vendor package file -> its comfy-layout location (same bytes as the repack)
PACKAGE_TO_COMFY = {
    "model.safetensors": "diffusion_models/flux3_image_klein_bf16.safetensors",
    "ae.safetensors": COMFY_VAE_FILE,
}
