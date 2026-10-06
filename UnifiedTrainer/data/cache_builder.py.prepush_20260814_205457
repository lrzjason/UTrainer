"""
CacheBuilder — T2ITrainer-style cache builder.

Constructs image pairs, encodes latents & captions, writes per-sample JSON
in subdirs mirroring the dataset structure, and builds explicit index files
(train_dataset_default.json, val_dataset_default.json).

Per-sample JSON schema:
    {
        "targets":     {target_key: {image_path, latent_path, bucket, media, num_frames, ...}},
        "references":  {ref_key: {image_path, latent_path, bucket, media, num_frames, ...}},
        "captions":    {caption_key: {npz_path, text_path}},
        "bucket":      "WxH",
        "mapping_key": "/dataset/subdir/base_name"
    }

Index entry (datarow):
    {"json_path": "/cache/subdir/base.json", "bucket": "WxH", "dataset": "name"}

No batch_configs in cache — resolved at training time from config.

Unified media pipeline (D3):
    Every latent is stored in the unified (C, T, H, W) .npz format via the
    single media dispatch ``_construct_media``:
      media == "video" → load_video_frames (DatasetConfig.video_frames/video_fps)
      otherwise        → load_image_frames (default "image")
    Both branches go through ``adapter.encode_video(vae, frames)`` — image
    samples (T == 1) keep every existing adapter's exact behavior through the
    base default implementation, and the MiniMax-H3 adapter (P1.4) overrides
    the hook for real video encoding.  Per-sample metadata records ``media``
    and ``num_frames``.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from UnifiedTrainer.data.config_schema import (
    DatasetConfig,
    ImageConfig,
    TargetEntry,
    ReferenceEntry,
    CaptionConfig,
    BatchConfig,
    find_index_from_right,
    strip_suffix,
)
from UnifiedTrainer.data.cache_manager import CacheManager
from UnifiedTrainer.data.embedding_cache import EmbeddingCache
from UnifiedTrainer.data.video_utils import load_image_frames, load_video_frames

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
SUPPORTED_VIDEO_TYPES = [".mp4"]


class CacheBuilder:
    """Builds the training cache from raw image data using composable configs.

    T2ITrainer pattern: per-sample JSON in subdirs + explicit index files.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        cache_dir: str,
        adapter: Any,
        dataset_name: str = "default",
    ):
        self.ds_config = dataset_config
        self.dataset_name = dataset_name
        self.cache = CacheManager(
            cache_dir,
            train_data_dir=dataset_config.train_data_dir,
        )
        self.adapter = adapter

    # ── Public API ─────────────────────────────────────────────────────

    def build(
        self,
        vae: Optional[torch.nn.Module] = None,
        text_encoder: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        processor: Optional[Any] = None,
    ) -> List[dict]:
        """Build the full cache (latents + embeddings + per-sample JSON).

        Returns:
            List of datarows for this dataset (to be combined into index).
        """
        if device is None:
            device = torch.device("cpu")

        recreate = self.ds_config.recreate_cache
        recreate_latents = recreate or self.ds_config.recreate_latents
        recreate_embeddings = recreate or self.ds_config.recreate_embeddings

        # ── Phase 1: Scan & construct image pairs ───────────────────
        image_pairs = self._construct_image_pairs()
        logger.info(f"Found {len(image_pairs)} image pairs in {self.ds_config.train_data_dir}")

        # ── Phase 2: Encode targets & references with VAE ────────────
        logger.info("Phase 2/3: Encoding target & reference images with VAE")
        if vae is not None:
            vae = vae.to(device)
            vae.requires_grad_(False)

        subdir_caches: Dict[str, List[str]] = {}

        datarows: List[dict] = []

        pbar = tqdm(
            image_pairs,
            desc="Encoding images",
            unit="pair",
            leave=True,
        )
        for pair in pbar:
            mapping_key = pair["mapping_key"]
            basename = os.path.basename(mapping_key)

            # Determine cache subdir via T2ITrainer get_cache_dir logic
            # Use any image in the pair to find the subdir
            sample_image_path = None
            for k, v in pair.items():
                if k != "mapping_key" and isinstance(v, str):
                    sample_image_path = v
                    break

            if sample_image_path:
                cache_subdir = self.cache.get_cache_dir(sample_image_path)
            else:
                cache_subdir = self.cache.cache_dir

            json_file = self.cache.sample_json_path(cache_subdir, basename)

            pbar.set_postfix_str(basename[-30:])

            # Skip if already cached (unless recreating latents)
            if self.cache.has_sample(json_file) and not recreate_latents:
                # Load existing sample to build datarow.
                # Guard against corrupt files from interrupted cache builds
                # (empty files, truncated JSON, etc.) — treat as "not cached".
                try:
                    sample = self.cache.load_sample(json_file)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"Corrupt cache file detected, will re-create: "
                        f"{os.path.relpath(json_file, self.cache.cache_dir)} ({e})"
                    )
                    # Delete the corrupt file so it gets re-created below
                    try:
                        os.remove(json_file)
                    except OSError:
                        pass
                else:
                    datarows.append({
                        "json_path": str(json_file),
                        "bucket": sample.get("bucket", ""),
                        "dataset": self.dataset_name,
                    })
                    continue

            # Build per-sample metadata
            sample_data: Dict[str, Any] = {
                "mapping_key": mapping_key,
                "bucket": "",
                "dataset": self.dataset_name,
                "targets": {},
                "references": {},
                "captions": {},
            }

            # Encode targets
            for t_key, entries in self.ds_config.target_configs.items():
                target_result = self._encode_target_entries(
                    vae, pair, entries, device, recreate_latents
                )
                if target_result:
                    sample_data["targets"][t_key] = target_result
                    if not sample_data["bucket"]:
                        sample_data["bucket"] = target_result.get("bucket", "")

            # Encode references
            for r_key, entries in self.ds_config.reference_configs.items():
                ref_result = self._encode_reference_entries(
                    vae, pair, entries, subdir_caches, device, recreate_latents
                )
                if ref_result:
                    sample_data["references"][r_key] = ref_result

            # Save per-sample JSON (captions will be added in phase 3)
            self.cache.save_sample(json_file, sample_data)

            datarows.append({
                "json_path": str(json_file),
                "bucket": sample_data.get("bucket", ""),
                "dataset": self.dataset_name,
            })

        pbar.close()

        # ── Phase 3: Encode captions with text encoder ───────────────
        logger.info("Phase 3/3: Encoding captions with text encoder")
        if text_encoder is not None and tokenizer is not None:
            text_encoder = text_encoder.to(device)
            text_encoder.requires_grad_(False)

        caption_pbar = tqdm(
            datarows,
            desc="Encoding captions",
            unit="pair",
            leave=True,
        )
        for datarow in caption_pbar:
            json_file = datarow["json_path"]
            sample = self.cache.load_sample(json_file)

            # Find the original pair for this datarow
            mapping_key = sample.get("mapping_key", "")
            pair = None
            for p in image_pairs:
                if p["mapping_key"] == mapping_key:
                    pair = p
                    break
            if pair is None:
                continue

            caption_pbar.set_postfix_str(os.path.basename(mapping_key)[-30:])

            for c_key, cap_cfg in self.ds_config.caption_configs.items():
                caption_data = self._encode_caption(
                    text_encoder, tokenizer, pair, cap_cfg, c_key, device,
                    recreate, recreate_embeddings,
                    cache_subdir=Path(json_file).parent,
                    processor=processor,
                )
                if caption_data:
                    sample.setdefault("captions", {})[c_key] = caption_data

            self.cache.save_sample(json_file, sample)

        # Save dataset metadata
        self.cache.save_metadata({
            "dataset_dir": self.ds_config.train_data_dir,
            "resolution": self.ds_config.resolution,
            "num_pairs": len(image_pairs),
        })

        # ── Phase 4: Create empty-string embedding for caption_dropout ──
        if text_encoder is not None and tokenizer is not None:
            model_suffix = self.adapter.empty_embedding_suffix
            empty_npz = os.path.join(str(self.cache.cache_dir), f"empty_embedding.{model_suffix}.npz")
            if not os.path.exists(empty_npz) or recreate or recreate_embeddings:
                try:
                    empty_emb = self.adapter.encode_text(
                        text_encoder, tokenizer, "", device, torch.float32,
                        processor=processor,
                    )
                    if empty_emb and "prompt_embed" in empty_emb:
                        EmbeddingCache.save(empty_npz, empty_emb)
                        logger.info(f"Empty-string embedding saved to {empty_npz}")
                except Exception as e:
                    logger.warning(f"Failed to create empty embedding: {e}")

        # ── Phase 5: Create suffix embedding (appended to encoder_hidden_states) ──
        if text_encoder is not None and tokenizer is not None and processor is not None:
            model_suffix = self.adapter.empty_embedding_suffix
            suffix_npz = os.path.join(str(self.cache.cache_dir), f"suffix_embedding.{model_suffix}.npz")
            if not os.path.exists(suffix_npz) or recreate or recreate_embeddings:
                try:
                    if hasattr(self.adapter, "encode_suffix_embedding"):
                        suffix_emb = self.adapter.encode_suffix_embedding(
                            text_encoder, tokenizer, processor, device, torch.float32,
                        )
                        if suffix_emb and "prompt_embed" in suffix_emb:
                            EmbeddingCache.save(suffix_npz, suffix_emb)
                            logger.info(f"Suffix embedding saved to {suffix_npz}")
                except Exception as e:
                    logger.warning(f"Failed to create suffix embedding: {e}")

        logger.info(f"Cache built: {len(datarows)} datarows at {self.cache.cache_dir}")
        return datarows

    # ── Image pair construction ────────────────────────────────────────

    def _file_matches_key(self, filename: str, img_cfg: ImageConfig) -> bool:
        """Whether a media file (basename, extension attached) belongs to an
        image_configs key.

        Media-aware (D3): image keys only match image extensions, video keys only
        match video extensions.  The key's ``suffix`` may be a name marker inside
        the base name (``_t``, ``_s`` — existing krea2 convention) OR the file
        extension itself (``.jpg`` / ``.mp4`` — the MiniMax-H3 image-pair
        convention).  An empty suffix matches every file of the key's media type.
        """
        ext = os.path.splitext(filename)[1].lower()
        if img_cfg.media == "video":
            if ext not in SUPPORTED_VIDEO_TYPES:
                return False
        elif ext not in SUPPORTED_IMAGE_TYPES:
            return False

        if not img_cfg.suffix:
            return True
        if img_cfg.suffix.lower() == ext:
            return True
        base = os.path.splitext(filename)[0]
        return find_index_from_right(base, img_cfg.suffix) > 0

    def _construct_image_pairs(self) -> List[dict]:
        """Scan the dataset directory and construct image pairs by base name.

        Mirrors T2ITrainer's logic:
        1. Scan all image files
        2. Split into image pools by suffix
        3. For each image, strip suffix to get base name → mapping_key
        4. Group by mapping_key to form pairs
        """
        data_dir = self.ds_config.train_data_dir
        if not data_dir or not os.path.isdir(data_dir):
            logger.warning(f"Dataset directory not found: {data_dir}")
            return []

        # Scan all media files recursively: image extensions always, plus video
        # extensions when any image_configs key declares media="video" (D3 —
        # built once in P1, P2 only activates the existing video branch).
        supported_exts = set(SUPPORTED_IMAGE_TYPES)
        if any(
            img_cfg.media == "video"
            for img_cfg in self.ds_config.image_configs.values()
        ):
            supported_exts.update(SUPPORTED_VIDEO_TYPES)
        image_files = [
            f for f in glob.iglob(os.path.join(data_dir, "**"), recursive=True)
            if os.path.isfile(f)
            and os.path.splitext(f)[1].lower() in supported_exts
        ]

        # Build image pools: image_pool[key] = [file_paths]
        image_pool: Dict[str, List[str]] = {key: [] for key in self.ds_config.image_configs}

        for f in image_files:
            base_name = os.path.basename(f)

            for key, img_cfg in self.ds_config.image_configs.items():
                if self._file_matches_key(base_name, img_cfg):
                    image_pool[key].append(f)

        # Build mapping: mapping[key][mapping_key] = [file_paths]
        mapping: Dict[str, Dict[str, List[str]]] = {key: {} for key in self.ds_config.image_configs}

        for key, img_cfg in self.ds_config.image_configs.items():
            for file in image_pool[key]:
                base_name = os.path.basename(file)
                filename, _ = os.path.splitext(base_name)
                filename_without_suffix = strip_suffix(filename, img_cfg.suffix, img_cfg.prefix)
                subdir = os.path.dirname(file)
                mapping_key = f"{subdir}/{filename_without_suffix}"

                if mapping_key not in mapping[key]:
                    mapping[key][mapping_key] = []
                mapping[key][mapping_key].append(file)

        # Construct pairs: iterate over the first image_config type
        image_configs_keys = list(self.ds_config.image_configs.keys())
        if not image_configs_keys:
            return []

        dataset_based_image = image_configs_keys[0]
        exclude_base_keys = [k for k in image_configs_keys if k != dataset_based_image]

        pairs = []
        for mapping_key in mapping[dataset_based_image]:
            for based_image in mapping[dataset_based_image][mapping_key]:
                base_name = os.path.basename(based_image)
                filename, _ = os.path.splitext(base_name)
                pair = {
                    "mapping_key": mapping_key,
                    dataset_based_image: based_image,
                }
                for other_key in exclude_base_keys:
                    if mapping_key in mapping[other_key]:
                        files = mapping[other_key][mapping_key]
                        if len(files) > 1:
                            # Try to find the matching file by filename
                            for pf in files:
                                if filename in os.path.basename(pf):
                                    pair[other_key] = pf
                                    break
                            else:
                                pair[other_key] = files[0]
                        else:
                            pair[other_key] = files[0]
                pairs.append(pair)

        return pairs

    # ── Media encoding (unified 5D cache) ─────────────────────────────

    def _encode_target_entries(
        self,
        vae: Optional[torch.nn.Module],
        pair: dict,
        entries: List[TargetEntry],
        device: torch.device,
        recreate: bool,
    ) -> dict:
        """Encode target media with VAE. Returns single dict (first target)."""
        for entry in entries:
            image_key = entry.image
            if image_key not in pair:
                continue

            image_path = pair[image_key]
            media = self._media_for_key(image_key)
            return self._construct_media(
                vae, image_path, device, recreate, media=media,
                resolution=self._resolution_for_key(image_key),
            )
        return {}

    def _resolution_for_key(self, image_key: str) -> int:
        """Resolve the encode resolution for an image_configs key.

        ``image_configs.<key>.resolution`` overrides the dataset resolution
        (e.g. face crops fixed at 512 while targets encode at 1024/1536).
        """
        img_cfg = self.ds_config.image_configs.get(image_key)
        if img_cfg is not None and img_cfg.resolution:
            return img_cfg.resolution
        return self.ds_config.resolution

    def _encode_reference_entries(
        self,
        vae: Optional[torch.nn.Module],
        pair: dict,
        entries: List[ReferenceEntry],
        subdir_caches: Dict[str, List[str]],
        device: torch.device,
        recreate: bool,
    ) -> Any:
        """Encode reference images with VAE.

        Collects one encoded media dict per entry whose image exists in the
        pair — entries with missing images are skipped (per-sample optional
        references, e.g. a second face ``F2`` that only some samples have).

        Returns:
            - a single dict when exactly one entry was encoded (backward
              compatible with existing caches),
            - a list of dicts for multi-reference groups (one per image),
            - ``{}`` when nothing was encoded.
        """
        encoded: List[dict] = []
        for entry in entries:
            image_key = entry.image

            if entry.sample_type == "from_same_name":
                if image_key not in pair:
                    continue
                image_path = pair[image_key]
                media = self._media_for_key(image_key)
                encoded.append(
                    self._construct_media(
                        vae, image_path, device, recreate, media=media,
                        resolution=self._resolution_for_key(image_key),
                    )
                )

            elif entry.sample_type == "from_subdir":
                if image_key not in pair:
                    continue
                base_path = pair[image_key]
                base_dir = os.path.dirname(base_path)

                if base_dir in subdir_caches:
                    ref_files = subdir_caches[base_dir]
                else:
                    ref_files = [
                        f for f in glob.iglob(os.path.join(base_dir, "**"), recursive=True)
                        if os.path.isfile(f)
                        and os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_TYPES
                    ]
                    subdir_caches[base_dir] = ref_files

                suffix = entry.suffix or self.ds_config.image_configs[image_key].suffix
                filtered = [
                    f for f in ref_files
                    if suffix and suffix in os.path.basename(f)
                    and os.path.abspath(f) != os.path.abspath(base_path)
                ]
                random.shuffle(filtered)

                count = min(entry.count, len(filtered))
                for ref_file in filtered[:count]:
                    media = self._media_for_key(image_key)
                    encoded.append(
                        self._construct_media(
                            vae, ref_file, device, recreate, media=media,
                            resolution=self._resolution_for_key(image_key),
                        )
                    )

        if not encoded:
            return {}
        if len(encoded) == 1:
            return encoded[0]
        return encoded

    def _media_for_key(self, image_key: str) -> str:
        """Resolve the media type ("image" | "video") for an image_configs key."""
        img_cfg = self.ds_config.image_configs.get(image_key)
        return img_cfg.media if img_cfg is not None else "image"

    def _construct_media(
        self,
        vae: Optional[torch.nn.Module],
        media_path: str,
        device: torch.device,
        recreate: bool,
        media: str = "image",
        resolution: Optional[int] = None,
    ) -> dict:
        """Encode one media sample into the unified 5D cache.

        Single media dispatch (D3): ``media == "video"`` → ``load_video_frames``
        (DatasetConfig.video_frames/video_fps); otherwise (default ``"image"``)
        → ``load_image_frames``.  Both branches call ``adapter.encode_video``
        and store the latent as the unified ``(C, T, H, W)`` npz (an image is
        ``(C, 1, H, W)``).  B is folded into the per-sample dimension — one npz
        per sample, matching the existing cache convention.  Per-sample metadata
        records ``media`` and ``num_frames``.

        Image behavior is exactly equivalent to the pre-unified pipeline: the
        same bucket crop is applied, the resized webp is still saved, and the
        latent values are identical (via the base ``encode_video`` default which
        delegates to ``encode_image``).  Existing adapters (krea2, ...) are
        unaware of the media dispatch.

        Args:
            resolution: Encode resolution override (per-image-config);
                defaults to the dataset resolution.
        """
        from PIL import Image

        if media not in ("image", "video"):
            raise ValueError(
                f"Unsupported media {media!r} for {media_path!r}; "
                f"expected 'image' or 'video'."
            )

        if resolution is None:
            resolution = self.ds_config.resolution
        basename = os.path.splitext(os.path.basename(media_path))[0]
        cache_subdir = self.cache.get_cache_dir(media_path)
        cache_dir = str(cache_subdir)

        resized_path = os.path.join(cache_dir, f"{basename}_{resolution}.webp")
        latent_path = os.path.join(cache_dir, f"{basename}_{resolution}.npz")

        # ── Cache hit: skip re-encoding when the sample is already cached ──
        # Images additionally require the resized webp (legacy layout); an
        # incomplete/stale image sample falls through to re-encode.
        if os.path.exists(latent_path) and not recreate:
            if media == "image" and not os.path.exists(resized_path):
                pass  # incomplete → re-encode below
            else:
                try:
                    if media == "image":
                        img = np.array(Image.open(resized_path).convert("RGB"))
                        h, w, _ = img.shape
                        latent_hit = self.cache.load_latent(latent_path)
                        num_frames = int(latent_hit.shape[1]) if latent_hit.ndim >= 4 else 1
                    else:
                        latent_hit = self.cache.load_latent(latent_path)
                        if latent_hit.ndim == 4:
                            h = latent_hit.shape[2] * self.adapter.vae_scale_factor
                            w = latent_hit.shape[3] * self.adapter.vae_scale_factor
                        else:
                            raise ValueError(
                                f"expected (C,T,H,W) latent in {latent_path!r}, "
                                f"got {tuple(latent_hit.shape)}"
                            )
                        # 复用第一次 load 的 latent_hit 算 H/W 与 num_frames，
                        # 不再二次 load（缓存命中时避免双重磁盘 I/O）。
                        num_frames = int(latent_hit.shape[1])
                    return {
                        "image_path": resized_path if media == "image" else None,
                        "original_image_path": media_path,
                        "latent_path": latent_path,
                        "bucket": f"{w}x{h}",
                        "media": media,
                        "num_frames": num_frames,
                    }
                except Exception:
                    pass  # corrupt/missing → re-encode below

        # ── Load frames as unified 5D (B=1, C, T, H, W) float32 [0, 1] ──
        if media == "video":
            frames = load_video_frames(
                media_path,
                num_frames=self.ds_config.video_frames,
                fps=self.ds_config.video_fps,
                resolution=resolution,
                divisibility=self.adapter.bucket_divisibility,
                resolution_config=self.adapter.resolution_config,
            )
        else:
            frames = load_image_frames(
                media_path,
                resolution=resolution,
                divisibility=self.adapter.bucket_divisibility,
                resolution_config=self.adapter.resolution_config,
            )

        h, w = int(frames.shape[3]), int(frames.shape[4])

        # Save the resized image (image media only) — matches legacy layout.
        if media == "image":
            pil_image = Image.fromarray(
                (frames[0, :, 0].clamp(0, 1).permute(1, 2, 0).numpy() * 255)
                .round()
                .astype("uint8")
            )
            pil_image.save(resized_path, "webp")

        # ── Encode via the unified media hook ───────────────────────────
        if vae is not None:
            with torch.no_grad():
                latent_dict = self.adapter.encode_video(vae, frames.to(device))
            latent = latent_dict["latent"]
            # Fold B → canonical (C, T, H, W) npz (B=1 per-sample).
            if latent.ndim == 5:
                if latent.shape[0] != 1:
                    raise ValueError(
                        f"encode_video returned batch size {latent.shape[0]}; "
                        f"expected 1 (B folds into the per-sample dimension)"
                    )
                latent = latent.squeeze(0)
            if latent.ndim != 4:
                raise ValueError(
                    f"encode_video must return (C,T,H,W) or (1,C,T,H,W), "
                    f"got shape {tuple(latent.shape)}"
                )
            self.cache.save_latent_npz(latent_path, latent)
        else:
            # Placeholder latent (no VAE available) — same (C,T,H,W) convention.
            scale = self.adapter.vae_scale_factor
            latent = torch.zeros(
                self.adapter.latent_channels, int(frames.shape[2]), h // scale, w // scale
            )
            self.cache.save_latent_npz(latent_path, latent)

        return {
            "image_path": resized_path if media == "image" else None,
            "original_image_path": media_path,
            "latent_path": latent_path,
            "bucket": f"{w}x{h}",
            "media": media,
            "num_frames": int(frames.shape[2]),
        }

    # ── Caption encoding ──────────────────────────────────────────────

    def _encode_caption(
        self,
        text_encoder: Optional[torch.nn.Module],
        tokenizer: Optional[Any],
        pair: dict,
        cap_cfg: CaptionConfig,
        cap_key: str,
        device: torch.device,
        recreate: bool,
        recreate_embeddings: bool = False,
        cache_subdir: Optional[Path] = None,
        processor: Optional[Any] = None,
    ) -> Optional[dict]:
        """Encode a caption (read text file, encode with text encoder, save npz).

        Args:
            recreate: Force re-encoding everything (from recreate_cache flag).
            recreate_embeddings: Force re-encoding captions only (from recreate_embeddings flag).
        """
        from UnifiedTrainer.data.embedding_cache import EmbeddingCache

        # Determine the image path to find the caption file
        image_key = cap_cfg.image
        if not image_key or image_key not in pair:
            for k, v in pair.items():
                if k != "mapping_key" and isinstance(v, str):
                    image_key = k
                    break

        if image_key not in pair:
            return None

        image_path = pair[image_key]
        folder_path = os.path.dirname(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Caption file: try image name with caption ext first
        text_path = os.path.join(folder_path, f"{filename}{cap_cfg.ext}")
        if not os.path.exists(text_path):
            img_cfg = self.ds_config.image_configs.get(image_key)
            if img_cfg:
                stripped = strip_suffix(filename, img_cfg.suffix, img_cfg.prefix)
                stripped_path = os.path.join(folder_path, f"{stripped}{cap_cfg.ext}")
                if os.path.exists(stripped_path):
                    text_path = stripped_path
        content = ""
        if os.path.exists(text_path):
            try:
                content = open(text_path, encoding="utf-8").read()
            except Exception:
                content = ""
        elif cap_cfg.instruction:
            content = cap_cfg.instruction

        # Build npz paths in cache subdir
        if cache_subdir is None:
            cache_subdir = self.cache.get_cache_dir(image_path)
        npz_path = os.path.join(str(cache_subdir), f"{filename}_{cap_key}.npz")

        caption_data = {
            "text_path": text_path if os.path.exists(text_path) else None,
            "npz_path": npz_path,
            "content": content,
        }

        # Encode with text encoder if available
        force_reencode = recreate or recreate_embeddings
        if text_encoder is not None and tokenizer is not None and content:
            if not os.path.exists(npz_path) or force_reencode:
                ref_images = None
                if cap_cfg.reference_list and processor is not None:
                    ref_images = self._load_reference_images(
                        pair, cap_cfg.reference_list, device
                    )

                # Optional image-conditioned caption encoding hook: only when
                # the adapter declares encode_text_accepts_image (MiniMax-H3
                # P1.4, whose vision-block presentation needs the associated
                # keyframe image).  Existing adapters keep their exact
                # encode_text contract — no extra kwargs are passed to them.
                encode_kwargs: Dict[str, Any] = dict(
                    reference_image=ref_images,
                    processor=processor,
                )
                condition_image = None
                if getattr(self.adapter, "encode_text_accepts_image", False):
                    condition_image = self._load_condition_image(pair, cap_cfg.image)
                    encode_kwargs["condition_image"] = condition_image

                try:
                    embedding = self.adapter.encode_text(
                        text_encoder, tokenizer, content, device, torch.float32,
                        **encode_kwargs,
                    )
                    if embedding and "prompt_embed" in embedding:
                        # Pop metadata before saving (not part of the embedding).
                        image_mask = embedding.pop("image_token_mask", None)
                        EmbeddingCache.save(npz_path, embedding)

                        # NOTE: Dropout npz files are no longer created.
                        # The training loop uses dynamic caption dropout via
                        # _apply_caption_dropout() which replaces embeddings
                        # with the global empty_embedding at runtime.
                        # Pre-computing per-sample dropout variants was dead
                        # weight — each file duplicated the full embedding,
                        # wasting ~50% of cache disk space.
                except Exception as e:
                    # Log the full traceback: a silent warning here is the only
                    # signal when cached embeddings go missing, which otherwise
                    # surfaces much later as "Krea2 requires encoder_hidden_states".
                    logger.warning(
                        f"Failed to encode caption for {cap_key} "
                        f"(image={image_path!r}, text_encoder="
                        f"{type(text_encoder).__name__}, processor="
                        f"{type(processor).__name__}): {e}",
                        exc_info=True,
                    )

        return caption_data

    def _load_condition_image(
        self, pair: dict, image_key: str
    ) -> Optional[Any]:
        """Load the associated source image for image-conditioned caption encoding.

        Only invoked when the adapter declares ``encode_text_accepts_image=True``
        (MiniMax-H3 P1.4 — its caption encoding constructs a vision-block
        presentation from the keyframe image).  Returns a bucket-cropped PIL
        image for ``media == "image"`` sources; ``None`` for video sources
        (video caption conditioning is a P2 concern — the adapter handles it).
        """
        if not image_key or image_key not in pair:
            return None
        img_cfg = self.ds_config.image_configs.get(image_key)
        if img_cfg is not None and img_cfg.media == "video":
            return None

        from PIL import Image

        from UnifiedTrainer.data.bucket import BucketSystem

        try:
            image_path = pair[image_key]
            bucket_system = BucketSystem(
                divisibility=self.adapter.bucket_divisibility,
                resolution_config=self.adapter.resolution_config,
            )
            pil = Image.open(image_path).convert("RGB")
            bucket = bucket_system.find_bucket_for_image(
                self.ds_config.resolution, pil
            )
            return bucket_system.crop_to_bucket(pil, bucket)
        except Exception as e:
            logger.warning(
                f"Failed to load condition image for caption encoding "
                f"({image_key!r}): {e}"
            )
            return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _load_reference_images(
        self, pair: dict, reference_list_config: dict, device: torch.device
    ) -> list:
        """Load reference images for multimodal text encoding.

        Resizes images preserving aspect ratio with total pixel area ≈ resize².
        """
        from PIL import Image

        ref_config_key = reference_list_config.get("reference_config", "")
        resize = int(reference_list_config.get("resize", 384))
        dropout = reference_list_config.get("dropout", 0.0)
        min_length = reference_list_config.get("min_length", 0)

        ref_entries = self.ds_config.reference_configs.get(ref_config_key, [])
        if not ref_entries:
            logger.warning(f"No reference entries found for reference_config '{ref_config_key}'")
            return []

        from UnifiedTrainer.data.bucket import BucketSystem
        bucket_system = BucketSystem(
            divisibility=self.adapter.bucket_divisibility,
            resolution_config=self.adapter.resolution_config,
        )

        image_list = []
        for ref_entry in ref_entries:
            image_key = ref_entry.image
            if image_key not in pair:
                logger.warning(f"Reference image key '{image_key}' not found in pair")
                continue

            image_path = pair[image_key]
            if dropout < random.random() or len(image_list) <= min_length:
                try:
                    pil_image = Image.open(image_path).convert("RGB")
                    bucket = bucket_system.find_bucket_for_image(
                        resize, pil_image
                    )
                    pil_image = bucket_system.crop_to_bucket(pil_image, bucket)
                    image_list.append(pil_image)
                except Exception as e:
                    logger.warning(f"Failed to load reference image {image_path}: {e}")

        return image_list
