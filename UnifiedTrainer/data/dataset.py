"""
UnifiedDataset — T2ITrainer-style cached dataset.

Loads pre-cached latents, embeddings, and metadata from per-sample JSON
files pointed to by explicit index files (train_dataset_default.json /
val_dataset_default.json).

No directory scanning — the index file is the single source of truth
for what samples exist.

T2ITrainer pattern: 1 step = 1 sample + 1 randomly chosen batch_config
from the config file. Batch configs are NOT stored in cache.
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

from UnifiedTrainer.data.cache_manager import CacheManager
from UnifiedTrainer.data.bucket import BucketSystem
from UnifiedTrainer.data.embedding_cache import EmbeddingCache
from UnifiedTrainer.data.config_schema import BatchConfig, DatasetConfig, parse_dataset_configs


class UnifiedDataset(Dataset):
    """T2ITrainer-style cached dataset using explicit index files.

    Args:
        adapter: Model adapter (provides resolution_config, latent_channels, etc.)
        cache_dir: Path to the pre-built cache directory
        config: Dataset config (image_configs, target_configs, batch_configs, etc.)
        split: "train" or "val"
    """

    def __init__(
        self,
        adapter: Any,
        cache_dir: str,
        config: dict,
        split: str = "train",
    ):
        self.adapter = adapter
        self.config = config
        self.split = split
        self.cache = CacheManager(cache_dir)
        self.embedding_cache = EmbeddingCache()
        self.bucket_system = BucketSystem(
            divisibility=adapter.bucket_divisibility,
            resolution_config=adapter.resolution_config,
        )

        # Parse composable config layers
        self.dataset_configs: List[DatasetConfig] = parse_dataset_configs(config)

        # repeats: per-dataset value stored for the sampler.
        # The global self.repeats is kept for backward compat
        # (__len__ fallback) but the sampler reads per-dataset values.
        if split == "val":
            self.repeats = 1
        else:
            self.repeats = 1
            for ds_cfg in self.dataset_configs:
                self.repeats = max(self.repeats, ds_cfg.repeats)

        # Load datarows from index file (T2ITrainer pattern)
        if split == "train":
            self.datarows = self.cache.load_train_index()
        else:
            self.datarows = self.cache.load_val_index()

        if not self.datarows:
            raise FileNotFoundError(
                f"No {'train' if split == 'train' else 'val'} index found in {cache_dir}. "
                f"Run cache build first (set recreate_cache=true in config)."
            )

        logger.info(f"Dataset [{split}]: {len(self.datarows)} datarows "
                     f"(repeats={self.repeats})")

        # Build batch_configs list from config (resolved at training time)
        self.batch_configs: List[dict] = []
        for ds_cfg in self.dataset_configs:
            for bc in ds_cfg.batch_configs:
                self.batch_configs.append({
                    "target_config": bc.target_config,
                    "caption_config": bc.caption_config,
                    "reference_config": bc.reference_config,
                    "caption_dropout": bc.caption_dropout,
                    "reference_dropout": bc.reference_dropout,
                })

        # Load dataset metadata
        self.metadata = self.cache.load_metadata()

    def get_dataset_batch_sizes(self, global_batch_size: int = 1) -> dict:
        """Build dataset_name → batch_size mapping.

        Resolution: per-dataset config.batch_size → global_batch_size → 1.
        """
        result = {}
        for ds_cfg in self.dataset_configs:
            ds_name = ds_cfg.train_data_dir
            ds_bs = ds_cfg.batch_size if ds_cfg.batch_size is not None else global_batch_size
            result[ds_name] = ds_bs
        return result

    def get_dataset_repeats(self) -> dict:
        """Build dataset_name → repeats mapping.

        Each dataset uses its own ``repeats`` value to duplicate samples.
        """
        result = {}
        for ds_cfg in self.dataset_configs:
            ds_name = ds_cfg.train_data_dir
            result[ds_name] = max(1, ds_cfg.repeats)
        return result

    def get_dataset_weights(self) -> dict:
        """Build dataset_name → sample_weight mapping.

        Controls the probability of selecting each dataset's batches
        during the shuffle phase. 1.0 = proportional (default).
        """
        result = {}
        for ds_cfg in self.dataset_configs:
            ds_name = ds_cfg.train_data_dir
            result[ds_name] = ds_cfg.sample_weight
        return result

    def __len__(self) -> int:
        return len(self.datarows) * self.repeats

    def _runtime_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Adapt a unified ``(C, T, H, W)`` cached latent to the adapter's runtime shape.

        The unified media pipeline (D3) stores every latent as ``(C, T, H, W)`` —
        an image is simply ``(C, 1, H, W)``.  Video-capable adapters
        (``supports_video=True``, MiniMax-H3) keep the full 5D path: per-sample
        ``(C, T, H, W)`` stacks to ``(B, C, T, H, W)`` in ``collate_fn``.  Legacy
        image adapters (krea2, ...) keep their 4D ``(B, C, H, W)`` runtime
        contract, so the singleton temporal dim is squeezed here — before
        collate.  P2 video samples only change T; this layer is untouched.
        """
        if getattr(self.adapter, "supports_video", False):
            return latent
        if latent.ndim == 4 and latent.shape[1] == 1:
            return latent.squeeze(1)
        return latent

    def __getitem__(self, idx: int) -> dict:
        base_len = len(self.datarows)
        base_idx = idx % base_len

        datarow = self.datarows[base_idx]
        json_path = datarow["json_path"]

        # Load per-sample JSON
        with open(json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        # T2ITrainer pattern: randomly select 1 batch_config at training time
        if self.batch_configs:
            batch_config = random.choice(self.batch_configs)
        else:
            batch_config = {}

        # Resolve batch_config fields
        if batch_config:
            target_key = batch_config["target_config"]
            caption_key = batch_config.get("caption_config", "")
            reference_key = batch_config.get("reference_config")
            caption_dropout = batch_config.get("caption_dropout", 0.0)
            reference_dropout = batch_config.get("reference_dropout", 0.0)
        else:
            target_key = None
            caption_key = None
            reference_key = None
            caption_dropout = 0.0
            reference_dropout = 0.0

        # Load target latents
        latents = {}
        targets = sample.get("targets", {})
        if target_key and target_key in targets:
            entry = targets[target_key]
            latent_path = entry.get("latent_path")
            if latent_path:
                latents[target_key] = self._runtime_latent(self.cache.load_latent(latent_path))
        else:
            for role, entry in targets.items():
                if isinstance(entry, dict):
                    latent_path = entry.get("latent_path")
                    if latent_path:
                        latents[role] = self._runtime_latent(self.cache.load_latent(latent_path))

        # Load reference latents
        references = sample.get("references", {})
        if reference_key and reference_key in references:
            entry = references[reference_key]
            latent_path = entry.get("latent_path") if isinstance(entry, dict) else None
            if latent_path:
                latents[reference_key] = self._runtime_latent(self.cache.load_latent(latent_path))
        else:
            for role, entry in references.items():
                if isinstance(entry, dict):
                    latent_path = entry.get("latent_path")
                    if latent_path:
                        latents[role] = self._runtime_latent(self.cache.load_latent(latent_path))


        # Load caption embedding
        embedding = None
        captions = sample.get("captions", {})
        if caption_key and caption_key in captions:
            npz_path = captions[caption_key].get("npz_path")
            if npz_path:
                embedding = self.embedding_cache.load(npz_path)
        elif captions:
            first_caption = next(iter(captions.values()))
            npz_path = first_caption.get("npz_path")
            if npz_path:
                embedding = self.embedding_cache.load(npz_path)

        # Apply reference_dropout
        if reference_dropout > 0 and random.random() < reference_dropout:
            ref_keys_to_drop = [k for k in latents if k != target_key and not k.endswith("_from_image")]
            for k in ref_keys_to_drop:
                del latents[k]


        return {
            "group_id": os.path.splitext(os.path.basename(json_path))[0],
            "latents": latents,
            "embedding": embedding,
            "batch_config": {
                "target_config": target_key,
                "caption_config": caption_key,
                "reference_config": reference_key,
                "caption_dropout": caption_dropout,
                "reference_dropout": reference_dropout,
            },
            "image_configs": sample.get("image_configs", {}),
            "bucket": sample.get("bucket", datarow.get("bucket", "")),
        }


def collate_fn(batch: list) -> dict:
    """Default collate function for UnifiedDataset batches.

    Stacks latents by role, keeps embeddings as list (variable length).

    The stacking is shape-agnostic: legacy adapters get per-sample
    ``(C, H, W)`` (temporal dim already squeezed in ``__getitem__``) and stack
    to ``(B, C, H, W)``; video-capable adapters (``supports_video=True``,
    MiniMax-H3) keep per-sample ``(C, T, H, W)`` and stack to
    ``(B, C, T, H, W)``.
    """
    result = {
        "group_ids": [b["group_id"] for b in batch],
        "latents": {},
        "embeddings": [b["embedding"] for b in batch],
        "image_configs": [b["image_configs"] for b in batch],
        "buckets": [b["bucket"] for b in batch],
        "batch_configs": [b["batch_config"] for b in batch],
    }

    # Stack latents by role
    all_roles = set()
    for b in batch:
        all_roles.update(b["latents"].keys())

    for role in all_roles:
        tensors = [b["latents"][role] for b in batch if role in b["latents"]]
        if tensors:
            result["latents"][role] = torch.stack(tensors)

    return result


class BucketBatchSampler(Sampler):
    """Sampler that yields batches of indices, grouped by (dataset, bucket).

    All indices in a batch share the same dataset AND the same bucket
    (i.e., same latent shape). This enables per-dataset batch sizing:
    high-resolution datasets use smaller batch sizes, low-resolution
    datasets use larger ones, maximizing VRAM utilization.

    Features:
        - **Per-dataset batch_size**: each dataset can override the global
          batch_size via dataset_batch_sizes.
        - **Per-dataset repeats**: each dataset's samples are duplicated
          independently based on dataset_repeats (NOT a global max).
        - **Per-dataset sample_weight**: controls the probability of
          selecting each dataset's batches during shuffle. 1.0 = proportional
          (default). When any weight != 1.0, weighted sampling with
          replacement is used — datasets with higher weights appear more
          often. When all weights are 1.0, standard uniform shuffle.

    Args:
        dataset: UnifiedDataset instance (must have .datarows and .repeats).
        batch_size: Global fallback batch size (used when a dataset has no
            per-dataset override).
        dataset_batch_sizes: Optional dict mapping dataset_name to batch_size.
        dataset_repeats: Optional dict mapping dataset_name to repeats.
            If provided, each dataset uses its own repeats value instead
            of the global max. Datasets not in this dict default to 1.
        dataset_weights: Optional dict mapping dataset_name to sample_weight.
            Controls batch selection probability during shuffle.
        drop_last: If True, drop the last incomplete batch per group.
        shuffle: If True, shuffle within and across groups each epoch.
    """

    def __init__(
        self,
        dataset: UnifiedDataset,
        batch_size: int = 1,
        dataset_batch_sizes: Optional[dict] = None,
        dataset_repeats: Optional[dict] = None,
        dataset_weights: Optional[dict] = None,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size  # global fallback
        self.dataset_batch_sizes = dataset_batch_sizes or {}
        self.dataset_repeats = dataset_repeats or {}
        self.dataset_weights = dataset_weights or {}
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Detect whether weighted sampling is active
        self._use_weighted = any(
            w != 1.0 for w in self.dataset_weights.values()
        )

        base_len = len(dataset.datarows)

        # Build (dataset, bucket) -> [indices] map using PER-DATASET repeats
        group_map: dict[tuple, List[int]] = defaultdict(list)
        for i in range(base_len):
            dr = dataset.datarows[i]
            bucket = dr.get("bucket", "unknown")
            ds_name = dr.get("dataset", "default")
            ds_repeats = self.dataset_repeats.get(ds_name, 1)
            for r in range(ds_repeats):
                idx = r * base_len + i
                group_map[(ds_name, bucket)].append(idx)

        self._group_indices: dict[tuple, List[int]] = dict(group_map)

        self._num_batches = 0
        for (ds_name, _bucket), indices in self._group_indices.items():
            bs = self._get_batch_size(ds_name)
            n = len(indices)
            if self.drop_last:
                self._num_batches += n // bs
            else:
                self._num_batches += (n + bs - 1) // bs

    def _get_batch_size(self, dataset_name: str) -> int:
        """Resolve batch size: per-dataset override -> global -> 1."""
        return self.dataset_batch_sizes.get(dataset_name, self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        group_indices = {}
        for key, indices in self._group_indices.items():
            copied = list(indices)
            if self.shuffle:
                random.shuffle(copied)
            group_indices[key] = copied

        # Build all batches from all groups
        batches: List[List[int]] = []
        batch_weights: List[float] = []  # per-batch weight (for weighted mode)
        for (ds_name, _bucket), indices in group_indices.items():
            bs = self._get_batch_size(ds_name)
            w = self.dataset_weights.get(ds_name, 1.0)
            for start in range(0, len(indices), bs):
                batch = indices[start : start + bs]
                if len(batch) == bs or not self.drop_last:
                    batches.append(batch)
                    batch_weights.append(w)

        if self.shuffle:
            if self._use_weighted and batches:
                # Weighted sampling WITH replacement: datasets with higher
                # sample_weight appear more often. Total epoch length stays
                # the same (_num_batches).
                yield from random.choices(
                    batches, weights=batch_weights, k=len(batches)
                )
            else:
                # Uniform shuffle without replacement (default)
                random.shuffle(batches)
                yield from batches
        else:
            yield from batches

    def __len__(self) -> int:
        return self._num_batches
