"""
BaseDataPipeline — the protocol for data loading backends.

A data pipeline handles: cache management, dataset construction, resolution
management (bucketing), and embedding caching. The shared Trainer calls
build_dataset() and build_dataloader() generically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch.utils.data import Dataset


class BaseDataPipeline(ABC):
    """Abstract base class for data pipeline backends.

    A data pipeline encapsulates everything data-related: where latents are
    cached, how datasets are constructed, how bucketing works, and how
    embeddings are loaded. Model-specific data logic is delegated to the
    adapter (latent shapes, embedding dims, resolution configs).
    """

    @abstractmethod
    def build_dataset(self, config: dict) -> Dataset:
        """Build the training dataset from config.

        Config includes dataset paths, image_configs (suffix mapping),
        resolution, and cache directory.
        """
        ...

    @abstractmethod
    def build_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        sampler: Optional[Any] = None,
    ) -> Any:
        """Build a DataLoader with appropriate sampler and collate."""
        ...

    @abstractmethod
    def cache_exists(self, cache_dir: str) -> bool:
        """Check whether a pre-built cache exists for the given directory."""
        ...

    @abstractmethod
    def build_cache(
        self,
        dataset_path: str,
        cache_dir: str,
        adapter: Any,
        config: dict,
    ) -> str:
        """Pre-compute and cache latents, embeddings, and metadata.

        Returns the path to the cache directory.
        """
        ...
