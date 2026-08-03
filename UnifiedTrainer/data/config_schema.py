"""
Config schema for the composable 5-layer data config system.

This mirrors T2ITrainer's config structure:
    image_configs    — image types identified by suffix/prefix
    target_configs   — named groups defining what the model predicts (image + from_image)
    reference_configs — named groups defining reference images (sample_type: from_same_name/from_subdir)
    caption_configs  — named groups defining caption sources (ext, image, reference_list, instruction)
    batch_configs    — combinatorial list tying target + caption + reference + per-entry dropout

All five layers are required — configs must specify target_configs, reference_configs,
caption_configs, and batch_configs explicitly.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def find_index_from_right(text: str, suffix: str) -> int:
    """Return the character index where *suffix* begins in *text*,
    searching from the right.  Returns -1 if not found."""
    idx = text.rfind(suffix)
    return idx


def strip_suffix(filename: str, suffix: str, prefix: str = "") -> str:
    """Strip suffix (and optional prefix) from a filename to get the base name.

    e.g. strip_suffix("real_001_t", "_t") -> "real_001"
         strip_suffix("mask_real_001_t", "_t", "mask_") -> "real_001"
    """
    if suffix and len(suffix) > 0:
        idx = find_index_from_right(filename, suffix)
        if idx > 0:
            base = filename[:idx]
        else:
            base = filename
    else:
        base = filename

    if prefix and len(prefix) > 0:
        pidx = find_index_from_right(base, prefix)
        if pidx > 0:
            base = base[:pidx]

    return base


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ImageConfig:
    """A single image type definition."""
    key: str
    suffix: str = ""
    prefix: str = ""

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "ImageConfig":
        return cls(
            key=key,
            suffix=d.get("suffix", ""),
            prefix=d.get("prefix", ""),
        )


@dataclass
class TargetEntry:
    """One entry in a target_configs group."""
    image: str              # key into image_configs

    @classmethod
    def from_dict(cls, d: dict) -> "TargetEntry":
        return cls(
            image=d["image"],
        )


@dataclass
class ReferenceEntry:
    """One entry in a reference_configs group."""
    image: str                          # key into image_configs
    sample_type: str = "from_same_name"  # "from_same_name" or "from_subdir"
    count: int = 1
    suffix: str = ""                    # for from_subdir filtering

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceEntry":
        return cls(
            image=d["image"],
            sample_type=d.get("sample_type", "from_same_name"),
            count=d.get("count", 1),
            suffix=d.get("suffix", ""),
        )


@dataclass
class CaptionConfig:
    """A named caption configuration."""
    key: str
    ext: str = ".txt"
    image: str = ""                     # which image_config the caption belongs to
    reference_list: Optional[dict] = None  # {reference_config, resize, dropout, min_length}
    instruction: str = ""               # static text fallback

    @classmethod
    def from_dict(cls, key: str, caption_config: dict) -> "CaptionConfig":
        return cls(
            key=key,
            ext=caption_config.get("ext", ".txt"),
            image=caption_config["image"] if "image" in caption_config else "",
            reference_list=caption_config.get("reference_list"),
            instruction=caption_config.get("instruction", ""),
        )


@dataclass
class BatchConfig:
    """A single resolved batch_config entry — one training combination."""
    target_config: str                  # key into target_configs
    caption_config: str                 # key into caption_configs
    reference_config: Optional[str] = None  # key into reference_configs
    caption_dropout: float = 0.0
    reference_dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "BatchConfig":
        return cls(
            target_config=d["target_config"],
            caption_config=d.get("caption_config", ""),
            reference_config=d.get("reference_config"),
            caption_dropout=d.get("caption_dropout", 0.0),
            reference_dropout=d.get("reference_dropout", 0.0),
        )


@dataclass
class DatasetConfig:
    """Fully parsed and validated per-dataset configuration.

    Contains all 5 config layers, resolved from the raw dataset_config dict.
    """
    train_data_dir: str
    resolution: int = 1024
    repeats: int = 1
    sample_weight: float = 1.0  # per-dataset sampling weight; 1.0 = proportional
    batch_size: Optional[int] = None  # per-dataset override; None = use global/default
    recreate_cache: bool = False
    recreate_latents: bool = False
    recreate_embeddings: bool = False

    # 5 config layers
    image_configs: Dict[str, ImageConfig] = field(default_factory=dict)
    target_configs: Dict[str, List[TargetEntry]] = field(default_factory=dict)
    reference_configs: Dict[str, List[ReferenceEntry]] = field(default_factory=dict)
    caption_configs: Dict[str, CaptionConfig] = field(default_factory=dict)
    batch_configs: List[BatchConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict, global_data_dir: str = "") -> "DatasetConfig":
        train_data_dir = d.get("train_data_dir", global_data_dir)
        resolution = int(d.get("resolution", 1024))
        repeats = d.get("repeats", 1)
        sample_weight = d.get("sample_weight", 1.0)
        batch_size = d.get("batch_size", None)
        recreate_cache = d.get("recreate_cache", False)
        recreate_latents = d.get("recreate_latents", False)
        recreate_embeddings = d.get("recreate_embeddings", False)

        # ── Parse image_configs ───────────────────────────────────────
        raw_image_configs = d.get("image_configs", {})
        image_configs: Dict[str, ImageConfig] = {}
        for key, cfg in raw_image_configs.items():
            image_configs[key] = ImageConfig.from_dict(key, cfg)

        # ── Parse target_configs (required) ──────────────────────────
        raw_target_configs = d.get("target_configs")
        if not raw_target_configs:
            raise ValueError(
                "target_configs is required. Define at least one named target group "
                "with an 'image' key referencing an image_configs entry.\n"
                "Example: \"target_configs\": {\"T\": [{\"image\": \"T\"}]}"
            )
        target_configs: Dict[str, List[TargetEntry]] = {}
        for key, entries in raw_target_configs.items():
            target_configs[key] = [TargetEntry.from_dict(e) for e in entries]

        # ── Parse reference_configs (required) ───────────────────────
        raw_reference_configs = d.get("reference_configs")
        if not raw_reference_configs:
            raise ValueError(
                "reference_configs is required. Define at least one named reference group "
                "with an 'image' key and 'sample_type' (from_same_name or from_subdir).\n"
                "Example: \"reference_configs\": {\"train_D\": [{\"image\": \"D\", \"sample_type\": \"from_same_name\"}]}"
            )
        reference_configs: Dict[str, List[ReferenceEntry]] = {}
        for key, entries in raw_reference_configs.items():
            reference_configs[key] = [ReferenceEntry.from_dict(e) for e in entries]

        # ── Parse caption_configs (required) ────────────────────────
        raw_caption_configs = d.get("caption_configs")
        if not raw_caption_configs:
            raise ValueError(
                "caption_configs is required. Define at least one named caption group "
                "with 'ext' and 'image' keys.\n"
                "Example: \"caption_configs\": {\"train_T\": {\"ext\": \".txt\", \"image\": \"T\"}}"
            )
        caption_configs: Dict[str, CaptionConfig] = {}
        for key, cfg in raw_caption_configs.items():
            caption_configs[key] = CaptionConfig.from_dict(key, cfg)

        # ── Parse batch_configs (required) ───────────────────────────
        raw_batch_configs = d.get("batch_configs")
        if not raw_batch_configs:
            raise ValueError(
                "batch_configs is required. Define at least one entry tying together "
                "target_config, caption_config, and reference_config.\n"
                "Example: \"batch_configs\": [{\"target_config\": \"T\", \"caption_config\": \"train_T\", \"reference_config\": \"train_D\"}]"
            )
        batch_configs: List[BatchConfig] = []
        for entry in raw_batch_configs:
            batch_configs.append(BatchConfig.from_dict(entry))

        ds_config = cls(
            train_data_dir=train_data_dir,
            resolution=resolution,
            repeats=repeats,
            sample_weight=sample_weight,
            batch_size=batch_size,
            recreate_cache=recreate_cache,
            recreate_latents=recreate_latents,
            recreate_embeddings=recreate_embeddings,
            image_configs=image_configs,
            target_configs=target_configs,
            reference_configs=reference_configs,
            caption_configs=caption_configs,
            batch_configs=batch_configs,
        )

        ds_config.validate()
        return ds_config

    def validate(self) -> None:
        """Cross-reference validation between config layers.

        Raises ValueError if any referenced key is not found.
        """
        # Validate target_configs reference valid image_configs keys
        for t_key, entries in self.target_configs.items():
            for entry in entries:
                if entry.image not in self.image_configs:
                    raise ValueError(
                        f"target_configs['{t_key}'] references image '{entry.image}' "
                        f"which is not in image_configs. Available: {list(self.image_configs.keys())}"
                    )

        # Validate reference_configs reference valid image_configs keys
        for r_key, entries in self.reference_configs.items():
            for entry in entries:
                if entry.image not in self.image_configs:
                    raise ValueError(
                        f"reference_configs['{r_key}'] references image '{entry.image}' "
                        f"which is not in image_configs. Available: {list(self.image_configs.keys())}"
                    )

        # Validate caption_configs reference valid image_configs keys
        for c_key, cfg in self.caption_configs.items():
            if cfg.image and cfg.image not in self.image_configs:
                raise ValueError(
                    f"caption_configs['{c_key}'] references image '{cfg.image}' "
                    f"which is not in image_configs. Available: {list(self.image_configs.keys())}"
                )

        # Validate batch_configs reference valid target/caption/reference keys
        for i, bc in enumerate(self.batch_configs):
            if bc.target_config not in self.target_configs:
                raise ValueError(
                    f"batch_configs[{i}].target_config='{bc.target_config}' "
                    f"not found in target_configs. Available: {list(self.target_configs.keys())}"
                )
            if bc.caption_config and bc.caption_config not in self.caption_configs:
                raise ValueError(
                    f"batch_configs[{i}].caption_config='{bc.caption_config}' "
                    f"not found in caption_configs. Available: {list(self.caption_configs.keys())}"
                )
            if bc.reference_config and bc.reference_config not in self.reference_configs:
                raise ValueError(
                    f"batch_configs[{i}].reference_config='{bc.reference_config}' "
                    f"not found in reference_configs. Available: {list(self.reference_configs.keys())}"
                )

        logger.info(
            f"DatasetConfig validated: "
            f"{len(self.image_configs)} image_types, "
            f"{len(self.target_configs)} target_groups, "
            f"{len(self.reference_configs)} reference_groups, "
            f"{len(self.caption_configs)} caption_groups, "
            f"{len(self.batch_configs)} batch_configs"
        )


def parse_dataset_configs(config: dict) -> List[DatasetConfig]:
    """Parse all dataset_configs from a top-level config dict.

    Looks for config["data"]["dataset_configs"] or config["dataset_configs"].
    Returns a list of fully parsed and validated DatasetConfig objects.
    """
    data_cfg = config.get("data", config)

    raw_datasets = (
        config.get("dataset_configs")
        or data_cfg.get("dataset_configs")
    )

    if not raw_datasets:
        raise ValueError(
            "No dataset_configs found. Provide config[\"data\"][\"dataset_configs\"] "
            "as a list of dataset configuration objects."
        )

    global_data_dir = data_cfg.get("train_data_dir", "")

    result = []
    for ds_dict in raw_datasets:
        result.append(DatasetConfig.from_dict(ds_dict, global_data_dir))

    return result
