"""
CacheManager — T2ITrainer-style cache with explicit index files and subdir mirroring.

Cache directory structure:
    {cache_dir}/
    ├── train_dataset_{key}.json        # Top-level train index (list of datarows)
    ├── val_dataset_{key}.json          # Top-level val index (list of datarows)
    ├── metadata_{key}.json             # Dataset-level metadata
    ├── empty_embedding.{suffix}.npz    # Empty-string embedding
    ├── {subdir_name}/                  # Mirrors dataset subdir structure
    │   ├── {basename}.json             # Per-sample metadata (targets, refs, captions)
    │   ├── {basename}_{res}.npz        # Unified 5D latent (C, T, H, W) — image = (C, 1, H, W)
    │   ├── {basename}_{res}.webp       # Resized image (image media only)
    │   └── {basename}_{cap_key}.npz    # Caption embedding
    └── {basename}_{res}.npz            # Root-level (when image is in dataset root)

Datarow format (in index files):
    {"json_path": "...", "bucket": "...", "dataset": "..."}

Latent storage (unified media pipeline, D3):
    Every media latent is saved as a single-array .npz of shape (C, T, H, W);
    an image is (C, 1, H, W) — the temporal dim is unified, there is no 2D/3D
    dual track.  Legacy .pt caches (pre-unified, (C,H,W)) remain loadable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


class CacheManager:
    """T2ITrainer-style cache manager with explicit index files."""

    def __init__(self, cache_dir: str, train_data_dir: str = ""):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_data_dir = train_data_dir

    # ── Subdir-mirrored cache path ──────────────────────────────────

    def get_cache_dir(self, image_path: str) -> Path:
        """Mirror dataset subdir structure into cache.

        T2ITrainer pattern:
        - If image is in dataset root → cache root
        - If image is in subdir → cache root + subdir name

        This prevents basename collisions when multiple subdirs
        contain images with the same filename.
        """
        dir_name = os.path.dirname(image_path)
        if not dir_name or Path(dir_name).resolve() == Path(self.train_data_dir).resolve():
            return self.cache_dir
        subdir_name = os.path.basename(dir_name)
        target_dir = self.cache_dir / subdir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    # ── Per-sample JSON ─────────────────────────────────────────────

    def sample_json_path(self, cache_subdir: Path, basename: str) -> Path:
        """Per-sample JSON path: {cache_subdir}/{basename}.json"""
        return cache_subdir / f"{basename}.json"

    def load_sample(self, json_path: str | Path) -> dict:
        """Load a per-sample JSON metadata."""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_sample(self, json_path: str | Path, data: dict) -> Path:
        """Save a per-sample JSON metadata."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def has_sample(self, json_path: str | Path) -> bool:
        """Check if a per-sample JSON exists."""
        return Path(json_path).exists()

    # ── Latents ─────────────────────────────────────────────────────

    def load_latent(self, path: str | Path) -> torch.Tensor:
        """Load a cached latent tensor.

        Supports both the unified single-array .npz format (``latent`` key,
        shape (C, T, H, W)) and legacy .pt files (torch.save).  npz arrays are
        stored float32 and loaded back as float32.
        """
        path = Path(path)
        if path.suffix.lower() == ".npz":
            with np.load(str(path)) as data:
                arr = data["latent"]
            return torch.from_numpy(arr).float()
        return torch.load(path, map_location="cpu")

    def save_latent(self, path: str | Path, tensor: torch.Tensor) -> Path:
        """Save a latent tensor to cache (legacy .pt format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.cpu(), path)
        return path

    def save_latent_npz(self, path: str | Path, tensor: torch.Tensor) -> Path:
        """Save a latent tensor in the unified (C, T, H, W) .npz format.

        The B dimension is folded into the per-sample dimension — each per-sample
        npz holds one sample's latent (image = (C, 1, H, W), video = (C, T, H, W)).
        Cast to float32 (matches the values the legacy .pt path stored for
        training); the trainer re-casts to the compute dtype on load.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = tensor.detach().cpu().float().numpy()
        np.savez(str(path), latent=arr)
        return path

    # ── Index files (T2ITrainer pattern) ────────────────────────────

    def save_train_index(self, datarows: list, key: str = "default") -> Path:
        """Save the combined train dataset index (list of datarows)."""
        path = self.cache_dir / f"train_dataset_{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(datarows, f, ensure_ascii=False)
        return path

    def load_train_index(self, key: str = "default") -> list:
        """Load the train dataset index."""
        path = self.cache_dir / f"train_dataset_{key}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_val_index(self, datarows: list, key: str = "default") -> Path:
        """Save the combined val dataset index (list of datarows)."""
        path = self.cache_dir / f"val_dataset_{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(datarows, f, ensure_ascii=False)
        return path

    def load_val_index(self, key: str = "default") -> list:
        """Load the val dataset index."""
        path = self.cache_dir / f"val_dataset_{key}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_full_index(self, datarows: list, key: str = "default") -> Path:
        """Save the FULL dataset index (all samples, before val split).

        This file is never modified by val split logic. It serves as the
        source-of-truth for re-splitting without rebuilding the cache.
        """
        path = self.cache_dir / f"full_dataset_{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(datarows, f, ensure_ascii=False)
        return path

    def load_full_index(self, key: str = "default") -> list:
        """Load the full dataset index (all samples, pre-split)."""
        path = self.cache_dir / f"full_dataset_{key}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_index(self, key: str = "default") -> bool:
        """Check if the train index file exists."""
        return (self.cache_dir / f"train_dataset_{key}.json").exists()

    def rebuild_full_index(self, dataset_name: str = "default", key: str = "default") -> list:
        """Reconstruct the full index by scanning existing per-sample JSONs.

        No VAE/text encoder needed — reads per-sample JSON metadata only.
        Scans all subdirs (including cache root) for *.json files that contain
        a 'targets' key (the marker for a valid per-sample cache file).

        Reads the 'dataset' field from each per-sample JSON to preserve
        per-dataset identity (critical for per-dataset batch sizing).

        Returns the rebuilt datarow list and saves it to full_dataset_{key}.json.
        """
        datarows = []

        # Scan cache root + all subdirs for per-sample JSONs
        all_jsons = []
        for p in self.cache_dir.rglob("*.json"):
            # Skip index/metadata files (they live in cache root)
            if p.parent == self.cache_dir:
                if p.name.startswith(("train_dataset_", "val_dataset_",
                                      "full_dataset_", "metadata_")):
                    continue
            all_jsons.append(p)

        for json_path in sorted(all_jsons):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    sample = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Only count valid per-sample cache files (must have 'targets')
            if not isinstance(sample, dict) or "targets" not in sample:
                continue

            # Read dataset identity from per-sample JSON (preserves per-dataset
            # batch sizing after index rebuild). Falls back to dataset_name param
            # for older caches that don't have the 'dataset' field.
            ds_name = sample.get("dataset", dataset_name)

            datarows.append({
                "json_path": str(json_path),
                "bucket": sample.get("bucket", ""),
                "dataset": ds_name,
            })

        if datarows:
            self.save_full_index(datarows, key)

        return datarows

    # ── Dataset metadata ────────────────────────────────────────────

    def load_metadata(self, key: str = "default") -> dict:
        """Load dataset-level metadata."""
        path = self.cache_dir / f"metadata_{key}.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_metadata(self, data: dict, key: str = "default") -> Path:
        """Save dataset-level metadata."""
        path = self.cache_dir / f"metadata_{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    # ── Val split creation ──────────────────────────────────────────

    def create_val_split(
        self,
        train_datarows: list,
        ratio: float,
        seed: int = 42,
        key: str = "default",
        force: bool = False,
    ) -> tuple:
        """Split datarows into train/val sets.

        T2ITrainer pattern: split the datarow LIST, not group IDs.
        Uses sklearn train_test_split for deterministic splitting.

        Args:
            train_datarows: Full list of train datarows to split.
            ratio: fraction for validation (e.g. 0.1 = 10%).
            seed: random seed for deterministic shuffling.
            key: split key.
            force: if True, overwrite existing split.

        Returns:
            (val_datarows, new_train_datarows)
        """
        existing_val = self.load_val_index(key)
        if existing_val and not force:
            return existing_val, train_datarows

        n_total = len(train_datarows)
        if n_total == 0:
            return [], []

        # Single datarow: keep it in both splits
        if n_total == 1:
            self.save_val_index(train_datarows, key)
            return train_datarows, train_datarows

        try:
            from sklearn.model_selection import train_test_split
            train_ratio = 1.0 - ratio
            new_train, val = train_test_split(
                train_datarows, train_size=train_ratio, test_size=ratio,
                random_state=seed,
            )
        except ImportError:
            import random
            rng = random.Random(seed)
            shuffled = train_datarows[:]
            rng.shuffle(shuffled)
            n_val = max(1, min(int(round(n_total * ratio)), n_total - 1))
            val = shuffled[:n_val]
            new_train = shuffled[n_val:]

        self.save_val_index(val, key)
        return val, new_train

    # ── Existence checks ────────────────────────────────────────────

    def exists(self) -> bool:
        """Check if cache has a train index file."""
        return self.has_index()
