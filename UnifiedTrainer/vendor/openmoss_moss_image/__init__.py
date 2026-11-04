"""Vendored official MOSS-Image model.

The implementation lives in :mod:`pipeline_moss_image` (a verbatim copy of the
file released with the model); this package only re-exports it.

Original author: OpenMOSS Team - openmoss/MOSS-Image-Base (ModelScope).
Licence: Apache-2.0. See ``ATTRIBUTION.md`` next to this file.
"""

from UnifiedTrainer.vendor.openmoss_moss_image.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
    _Block,
    _RoPE,
)

__all__ = [
    "MOSSImagePipeline",
    "MOSSImageScheduler",
    "MOSSImageTransformer",
    "_Block",
    "_RoPE",
]
