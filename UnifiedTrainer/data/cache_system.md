# UnifiedTrainer Cache System — Current Design

## Overview

The cache system pre-encodes images (VAE latents) and captions (text encoder embeddings)
so training never touches the VAE or text encoder. All heavy preprocessing happens once
during cache building, and training reads only lightweight `.pt` / `.npz` files.

## Directory Structure (Current — Flat Layout)

All files live in a single flat directory:

```
{cache_dir}/
├── {group_id}.json                    # Group metadata (dict)
├── {group_id}.json                    # Group metadata (dict)
│   ... (613 group files)
├── {basename}_{resolution}.pt          # VAE latent tensor
├── {basename}_{resolution}.webp        # Resized image
├── {filename}_{caption_key}.npz        # Caption embedding
├── empty_embedding.{model_suffix}.npz  # Empty-string embedding (for caption_dropout)
├── metadata_default.json              # Dataset-level metadata (dict)
├── val_dataset_default.json           # Validation split group IDs (list)
└── dataset_krea2_edit.json            # ← PROBLEM: leaked config file (list)
```

Adjusted:
{cache_dir}/
├── {subdir same name in dataset if exists}
├── --metadata_default.json              # Dataset-level metadata (dict)
├── --{original filename}.json
├── --{original filename}.json
├── --{basename}_{resolution}.pt          # VAE latent tensor
├── --{basename}_{resolution}.webp        # Resized image
├── --{filename}_{caption_key}.npz        # Caption embedding
├── ... (files in same dataset)
├── empty_embedding.{model_suffix}.npz  # Empty-string embedding (for caption_dropout)
├── val_dataset_default.json           # Validation split 
└── train_dataset_default.json         # Training split 

### Problem: No Namespace Separation

Everything is in one flat dir. `list_groups()` discovers group files by globbing `*.json`
and excluding known prefixes (`metadata_*`, `val_*`). Any other JSON file placed in the
cache dir — config dumps, dataset descriptions, user notes — is incorrectly treated as
a group file, causing `AttributeError: 'list' object has no attribute 'get'`.

## File Types

### 1. Group Metadata — `{group_id}.json`

**group_id** is derived from the dataset image path via `_make_group_id()`:

```python
# Input:  "/home/waas/krea2_dataset/anime/real_001_t"
# Output: "_home_waas_krea2_dataset_anime_real_001_t"
```

Schema (always a **dict**):

```json
{
    "mapping_key": "/home/waas/krea2_dataset/anime/real_001_t",
    "bucket": "768x1024",
    "targets": {
        "target": [
            {
                "image_path": "/cache/real_001_t_1024.webp",
                "original_image_path": "/data/real_001_t.png",
                "latent_path": "/cache/real_001_t_1024.pt",
                "bucket": "768x1024"
            }
        ]
    },
    "references": {
        "style_ref": [
            {
                "image_path": "/cache/ref_002_1024.webp",
                "original_image_path": "/data/ref_002.png",
                "latent_path": "/cache/ref_002_1024.pt",
                "bucket": "768x1024"
            }
        ]
    },
    "captions": {
        "caption": {
            "text_path": "/data/real_001.txt",
            "npz_path": "/cache/real_001_t_caption.npz",
            "content": "a beautiful anime girl"
        }
    },
    "batch_configs": [
        {
            "target_config": "target",
            "caption_config": "caption",
            "reference_config": "style_ref",
            "caption_dropout": 0.05,
            "reference_dropout": 0.0
        }
    ]
}
```

### 2. Latent Tensor — `{basename}_{resolution}.pt`

PyTorch tensor saved via `torch.save()`. Shape depends on model adapter:
- Typical: `(latent_channels, H/8, W/8)` — squeezed, no batch dim
- Loaded via `CacheManager.load_latent(path)` → `torch.load(path, map_location="cpu")`

**Naming**: uses the image **basename** (filename without extension), NOT group_id.
This means latent filenames are NOT tied to group structure and can collide if two
different images have the same basename in different subdirectories.

### 3. Resized Image — `{basename}_{resolution}.webp`

WebP format, bucketed to nearest aspect ratio at target resolution.
Used for validation generation display, not for training computation.

### 4. Caption Embedding — `{filename}_{caption_key}.npz`

NumPy compressed archive containing `prompt_embeds` (and optionally other keys
like `pooled_embeds` depending on model adapter).

**Naming**: uses image filename stem + caption config key, e.g. `real_001_t_caption.npz`.

### 5. Empty Embedding — `empty_embedding.{model_suffix}.npz`

Model-specific (e.g. `empty_embedding.npkrea2.npz`). Used as replacement when
`caption_dropout` triggers during training.

### 6. Dataset Metadata — `metadata_default.json`

```json
{
    "dataset_dir": "/home/waas/krea2_dataset",
    "resolution": 1024,
    "num_pairs": 613,
    "num_batch_configs": 1
}
```

### 7. Validation Split — `val_dataset_default.json`

```json
["group_id_1", "group_id_2", ...]
```

A **list** of group IDs held out for validation.

### 8. Leaked Files — e.g. `dataset_krea2_edit.json`

Any JSON file that doesn't match the `metadata_*` or `val_*` prefix gets picked up
by `list_groups()` and treated as a group. This is the root cause of crashes.

## Data Flow

```
CacheBuilder.build()
    │
    ├─ Phase 1: _construct_image_pairs()
    │      Scan dataset dir → match suffixes → group into pairs
    │
    ├─ Phase 2: VAE encode
    │      For each pair:
    │        _encode_target_entries()   → _cache_image() → save .pt + .webp
    │        _encode_reference_entries() → _cache_image() → save .pt + .webp
    │        save_group(group_id, group_data)  → write {group_id}.json
    │
    ├─ Phase 3: Text encode
    │      For each pair:
    │        _encode_caption() → adapter.encode_text() → save .npz
    │        Update group_data["captions"]
    │        save_group(group_id, group_data)
    │      save_metadata()
    │
    └─ Phase 4: Empty embedding
           adapter.encode_text("") → save empty_embedding.{suffix}.npz

Training:
    UnifiedDataset.__getitem__(idx)
        → load_group(group_id)     ← reads {group_id}.json
        → load_latent(path)        ← reads .pt file
        → embedding_cache.load()   ← reads .npz file
        → return {latents, embedding, batch_config, ...}
```

## Key Design Issues for Redesign

1. **Flat directory**: All file types (groups, latents, images, embeddings, metadata,
   config dumps) share one directory. No structural separation.

2. **list_groups() relies on filename prefix exclusion** (`metadata_*`, `val_*`) instead
   of positive identification. Any non-excluded `.json` is assumed to be a group.

3. **Latent/image naming uses basename, not group_id**: If two images in different
   subdirectories share the same filename, their cached latents collide.

4. **No index file**: The system re-scans the directory every time `list_groups()` is
   called. With 613+ files this is slow, especially with the JSON-read-and-validate
   added to `list_groups()`.

5. **Group JSON files mix metadata + resolved batch_configs**: The `batch_configs` array
   is stored per-group, meaning it's duplicated across all 613 groups. Changes to
   batch_config require rebuilding the entire cache.

6. **Val split stored as separate JSON list**: Could be a field in a single index file.

7. **No manifest/versioning**: No way to detect if the cache was built with a different
   model adapter, resolution, or VAE without reading individual group files.

---

## T2ITrainer Reference Design (Proven Working)

T2ITrainer's cache system is fundamentally different: it uses **explicit index files**
and **subdir-mirrored per-sample JSON**, eliminating all directory scanning.

### T2ITrainer Directory Structure

```
{cache_dir}/
├── dataset_{training_name}.json        # ← TOP-LEVEL INDEX (list of datarows)
├── val_dataset_{training_name}.json     # ← TOP-LEVEL VAL INDEX (list of datarows)
├── empty_embedding.{suffix}.npz
├── {subdir_name}/                       # ← MIRRORS DATASET SUBDIR STRUCTURE
│   ├── {basename}.json                 # Per-sample metadata (targets, refs, captions)
│   ├── {basename}_{resolution}.pt      # Latent tensor
│   ├── {basename}_{resolution}.webp    # Resized image
│   └── {basename}_{caption_key}.npz    # Caption embedding
└── {another_subdir}/
    ├── ...
```

When dataset images are in the root data dir (no subdir), files go directly in
`{cache_dir}/` (no subdir wrapper).

### T2ITrainer Two-Level Index System

**Per-dataset subset index** (in `{ds_cache_dir}/`):

```
metadata_{dataset_name}.json       # ← LIST of datarows for this dataset's train split
val_metadata_{dataset_name}.json   # ← LIST of datarows for this dataset's val split
```

**Combined top-level index** (in `{cache_dir}/`):

```
dataset_{training_name}.json       # ← ALL datasets' train datarows combined
val_dataset_{training_name}.json   # ← ALL datasets' val datarows combined
```

### T2ITrainer Datarow Format (LIST item)

Each index file is a JSON **list** of datarow pointers:

```json
[
    {
        "json_path": "/cache/anime/real_001.json",
        "bucket": "krea2_dataset_768x1024",
        "dataset": "krea2_dataset"
    },
    {
        "json_path": "/cache/anime/real_002.json",
        "bucket": "krea2_dataset_1024x1024",
        "dataset": "krea2_dataset"
    }
]
```

The index is just a list of `{json_path, bucket, dataset}` pointers. The actual
sample data lives in the per-sample JSON at `json_path`.

### T2ITrainer Per-Sample JSON Format (the `json_path` target)

```json
{
    "targets": {
        "target": {
            "bucket": "768x1024",
            "image_path": "/cache/anime/real_001_1024.webp",
            "original_image_path": "/data/anime/real_001_t.png",
            "latent_path": "/cache/anime/real_001_1024.pt"
        }
    },
    "references": {
        "style_ref": {
            "bucket": "768x1024",
            "image_path": "/cache/anime/ref_002_1024.webp",
            "original_image_path": "/data/anime/ref_002.png",
            "latent_path": "/cache/anime/ref_002_1024.pt"
        }
    },
    "captions": {
        "caption": {
            "npz_path": "/cache/anime/real_001_t_caption.pt",
            "text_path": "/data/anime/real_001.txt"
        }
    },
    "bucket": "768x1024"
}
```

Note: NO `batch_configs` stored per-sample. Batch configs are resolved at training
time from the config file, not baked into cache.

### T2ITrainer Cache Build Flow

```python
# Per dataset:
for image_pair in image_pairs:
    mk = image_pair["mapping_key"]
    cache_dir = get_cache_dir(mk)       # → {cache_dir}/{subdir_name}/ or {cache_dir}/
    basename = os.path.basename(mk)
    json_file = os.path.join(cache_dir, f"{basename}.json")

    # Build embedding_object with targets, references, captions
    embedding_object = { ... }

    # Save per-sample JSON
    with open(json_file, "w") as f:
        json.dump(embedding_object, f, indent=4)

    # Append datarow pointer to subset index
    cache_datarows.append({
        "json_path": json_file,
        "bucket": f"{dataset_name}_{bucket}",
        "dataset": dataset_name,
    })

# Save per-subset index
with open(subset_metadata_path, "w") as outfile:
    outfile.write(json.dumps(cache_datarows))

# After all datasets: combine into top-level index
dataset_datarows += cache_datarows * repeats
with open(metadata_path, "w") as outfile:
    outfile.write(json.dumps(dataset_datarows))
```

### T2ITrainer Dataset Loading (No Directory Scanning)

```python
# At training startup:
with open(metadata_path, "r") as readfile:
    datarows = json.loads(readfile.read(), strict=False)

# datarows is a LIST — no globbing, no list_groups(), no type guessing
train_dataset = CachedJsonDataset(datarows, ...)

# In __getitem__:
def __getitem__(self, index):
    path_obj = self.datarows[actual_index]
    json_path = path_obj["json_path"]
    with open(json_path, 'r') as f:
        json_obj = json.load(f)
    # Load latents and embeddings from paths in json_obj
    ...
```

### Key Differences: T2ITrainer vs UnifiedTrainer

| Aspect | T2ITrainer (correct) | UnifiedTrainer (broken) |
|--------|---------------------|------------------------|
| **Index** | Explicit `dataset_*.json` list of datarows | Implicit via `list_groups()` globbing |
| **Discovery** | Read index file → get all paths | Scan dir, exclude prefixes, validate each file |
| **Sample JSON** | In subdir mirroring dataset structure | Flat, named by hashed group_id |
| **Batch configs** | NOT in cache (resolved at train time) | Baked into every group JSON (613× duplication) |
| **Val split** | Separate `val_dataset_*.json` index list | Separate `val_dataset_*.json` list of group IDs |
| **File collision** | Impossible (subdirs mirror source) | Likely (flat basename naming) |
| **Corruption safety** | Index only lists known-good files | Any leaked `.json` becomes a fake "group" |

### UnifiedTrainer Should Adopt

1. **Explicit index files**: `train_dataset_default.json` and `val_dataset_default.json`
   as top-level index lists (already in user's Adjusted layout).

2. **Subdir mirroring**: `get_cache_dir(image_path)` mirrors the dataset's subdir
   structure into the cache, preventing basename collisions.

3. **Datarow pointer format**: Each index entry is `{json_path, bucket, dataset}` —
   just a pointer, not the full data.

4. **Remove `list_groups()`**: Dataset reads from index file, not directory scanning.

5. **Remove per-group `batch_configs`**: Resolve at training time from config.

6. **`get_cache_dir()` logic**: If image is in dataset root → cache root; if in subdir
   → cache root + subdir name.
