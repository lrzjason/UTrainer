#!/usr/bin/env python3
"""
UnifiedTrainer - single entry point for all model training.

Usage:
    python train.py --model flux2_klein --config configs/flux2_klein_example.json
    python train.py --model qwen_image --config configs/qwen_image_example.json --resume output/checkpoint.safetensors
    python train.py --model qwen_image --config configs/qwen_image_example.json --resume-full output/checkpoint.safetensors

Resume modes:
    --resume <path>        Load LoRA weights only, restart optimizer/step/epoch
    --resume-full <path>   Full resume: weights + optimizer + scheduler + RNG + step/epoch

The --model flag selects a registered model adapter.
The --config flag points to a JSON config file specifying data, losses, and training params.
"""
import os

# ── CUDA memory fragmentation fix ──────────────────────────────────────
# Must be set BEFORE torch is imported. expandable_segments reduces
# fragmentation by allowing the allocator to grow segments instead of
# allocating new ones, preventing the "large reserved but unallocated"
# scenario that causes OOM even when there should be enough free memory.
# Note: PYTORCH_CUDA_ALLOC_CONF is deprecated in newer PyTorch; use
# PYTORCH_ALLOC_CONF. Set both for backward compatibility.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import logging
import math
import random
import sys

try:  # progress bars for quantization passes
    from tqdm import tqdm as _progress
except ImportError:  # pragma: no cover - tqdm ships with huggingface_hub
    def _progress(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _collect_linears(module, targets, skip_lora=True):
    """Collect every replaceable (parent, attr_name, child) nn.Linear in the tree.

    Collecting first (instead of recursing while replacing) makes the
    replacement loop safe to drive with a progress bar.
    """
    import torch.nn as nn

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (not skip_lora or "lora" not in name.lower()):
            targets.append((module, name, child))
        else:
            _collect_linears(child, targets, skip_lora)


def _replace_linear_with_4bit(module, compute_dtype, quant_type="nf4"):
    """Recursively replace nn.Linear with bnb.nn.Linear4bit for NF4 quantization.

    The weight data stays in full precision on CPU; quantization happens when
    Params4bit.cuda() is called (i.e. when the model is moved to GPU).
    """
    import bitsandbytes as bnb
    import torch.nn as nn

    targets = []
    _collect_linears(module, targets)
    for parent, name, child in _progress(
        targets, desc=f"{quant_type} quantize", unit="linear", leave=False
    ):
        new_module = bnb.nn.Linear4bit(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
        )
        new_module.weight = bnb.nn.Params4bit(
            child.weight.data,
            requires_grad=False,
            quant_type=quant_type,
        )
        if child.bias is not None:
            new_module.bias = nn.Parameter(child.bias.data.clone())
        setattr(parent, name, new_module)


def _replace_linear_with_8bit(module, threshold=0.0):
    """Recursively replace nn.Linear with bnb.nn.Linear8bitLt for int8 quantization.

    threshold=0 disables LLM.int8() outlier decomposition, which avoids the
    torch.argwhere().view(-1) non-contiguity bug in bitsandbytes' int8 ops.
    Outlier handling is not needed for frozen-weight LoRA training.
    """
    import bitsandbytes as bnb
    import torch.nn as nn

    targets = []
    _collect_linears(module, targets)
    for parent, name, child in _progress(
        targets, desc="int8 quantize", unit="linear", leave=False
    ):
        new_module = bnb.nn.Linear8bitLt(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            has_fp16_weights=False,
            threshold=threshold,
        )
        new_module.weight = bnb.nn.Int8Params(
            child.weight.data,
            requires_grad=False,
            has_fp16_weights=False,
        )
        if child.bias is not None:
            new_module.bias = nn.Parameter(child.bias.data.clone())
        setattr(parent, name, new_module)


def _quantize_text_encoder(te, mode, compute_dtype):
    """Apply training.text_encoder_quantize (none|nf4|int8) to a text encoder.

    Used by the cache build so the large H3 Qwen3-VL encoder fits in VRAM
    while encoding; the same bnb helpers power transformer quantization.
    """
    if mode == "nf4":
        _replace_linear_with_4bit(te, compute_dtype=compute_dtype, quant_type="nf4")
    elif mode == "int8":
        _replace_linear_with_8bit(te, threshold=0.0)
    else:
        return
    logger.info(f"Text encoder quantized ({mode})")


def _is_rank0() -> bool:
    """DDP 多进程下仅 rank 0 写 DB/产物/检查点（P5 多卡门控）。

    单进程（reserve 模式 / 非 accelerate）无 RANK 环境变量，恒为 True。
    """
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def main():
    parser = argparse.ArgumentParser(
        description="UnifiedTrainer -modular training for diffusion models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model adapter name (e.g., flux2_klein, qwen_image, longcat). "
             "Not required in --task-id DB mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Not required in --task-id DB mode.",
    )
    # ── ScheduledTrainer DB mode ─────────────────────────────────
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        dest="task_id",
        help="ScheduledTrainer task id. When set, config is materialized "
             "from the orchestrator DB instead of --config.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to orchestrator SQLite DB (required with --task-id).",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=None,
        dest="dry_run_steps",
        help="DB mode only: simulate N training steps (heartbeats + fake "
             "output) without loading any model. For smoke tests.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to LoRA checkpoint (.safetensors) to resume from. "
            "Loads weights ONLY -optimizer/scheduler/step restart fresh."
        ),
    )
    parser.add_argument(
        "--resume-full",
        type=str,
        default=None,
        dest="resume_full",
        help=(
            "Path to LoRA checkpoint (.safetensors) for FULL resume. "
            "Loads weights + optimizer + scheduler + RNG + step/epoch. "
            "Requires the matching _training_state.pt file alongside."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model adapters and exit",
    )
    parser.add_argument(
        "--list-losses",
        action="store_true",
        help="List available loss modules and exit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # ── ScheduledTrainer DB mode: materialize config from DB ────────
    # 放在参数解析后尽早分支；不影响 --model/--config 旧路径。
    db_ctx = None       # orchestrator.db.DB | None
    db_config = None    # materialized config | None
    if args.task_id is not None:
        if not args.db:
            parser.error("--db is required with --task-id")
        from orchestrator.db import DB
        db_ctx = DB(args.db)
        task_row = db_ctx.get_task(args.task_id)
        if task_row is None:
            logger.error(f"Task {args.task_id} not found in {args.db}")
            sys.exit(1)
        db_config = db_ctx.materialize_config(args.task_id)
        args.model = db_config.get("model") or task_row["model"]
        logger.info(
            f"DB mode: task {args.task_id} ({task_row['name']}), "
            f"model={args.model}, config materialized from {args.db}")
        dry_steps = args.dry_run_steps or db_config.get(
            "training", {}).get("dry_run_steps")
        if dry_steps:
            sys.exit(_dry_run(db_ctx, task_row, db_config, int(dry_steps)))
    elif not args.list_models and not args.list_losses:
        # 旧模式校验（list 命令允许缺省）
        if not args.model:
            parser.error("--model is required (unless --task-id/--list-*)")
        if not args.config:
            parser.error("--config is required (unless --task-id/--list-*)")

    # Krea2-specific: force cuDNN SDPA backend (ai-toolkit uses SDPBackend.CUDNN_ATTENTION,
    # 20-40% faster for Krea2's MiDiT attention shapes). Must be set BEFORE diffusers is
    # imported by model adapters at _import_all_adapters() below.
    if args.model == "krea2":
        os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_cudnn")

    # Import after parsing to avoid slow startup for --help
    from UnifiedTrainer.registry import ModelRegistry, LossRegistry

    # Auto-import all registered adapters and losses
    _import_all_adapters()
    _import_all_losses()

    if args.list_models:
        print("Available model adapters:")
        for name in ModelRegistry.names():
            print(f" - {name}")
        return

    if args.list_losses:
        print("Available loss modules:")
        for name in LossRegistry.names():
            print(f" - {name}")
        return

    # Load config
    if db_config is not None:
        config = db_config
        config["model"] = args.model
        logger.info(f"Config materialized from DB (task {args.task_id})")
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["model"] = args.model
        logger.info(f"Config loaded from {args.config}")
    logger.info(f"Model: {args.model}")

    # Get adapter
    if not ModelRegistry.contains(args.model):
        logger.error(f"Model '{args.model}' not found. Available: {ModelRegistry.names()}")
        sys.exit(1)

    adapter_cls = ModelRegistry.get(args.model)
    adapter = adapter_cls(config)

    # Assemble losses
    from UnifiedTrainer.losses.base import BaseLoss

    losses = []
    for loss_cfg in config.get("losses", []):
        loss_type = loss_cfg.get("type", loss_cfg.get("name", ""))
        if not LossRegistry.contains(loss_type):
            logger.warning(f"Loss '{loss_type}' not registered, skipping")
            continue
        loss_cls = LossRegistry.get(loss_type)
        params = loss_cfg.get("params", loss_cfg.get("args", {}))
        weight = loss_cfg.get("weight", 1.0)
        loss_instance = loss_cls(weight=weight, **params)
        losses.append(loss_instance)
        logger.info(f"  Loss: {loss_type} (weight={weight})")

    # ── Load model components via adapter ──────────────────────
    import torch

    # ── CUDA performance optimizations (matching T2ITrainer) ───
    # TF32 (TensorFloat-32) accelerates matmuls 10-15% on Ampere+ (RTX 3090+)
    # by using 19-bit mantissa instead of 23-bit in float32 intermediates.
    # T2ITrainer enables this in ALL Krea2 configs via --allow_tf32.
    torch.backends.cuda.matmul.allow_tf32 = True

    # Let cuDNN auto-tune the fastest algorithm for each convolution shape.
    # Small overhead on first run per shape, but faster thereafter.
    torch.backends.cudnn.benchmark = True

    # Use TF32 precision in float32 matmuls (PyTorch >= 1.12).
    # 'high' uses TF32 for matmul; 'medium' uses TF32 for both matmul+conv.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    weight_dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp8": torch.float8_e4m3fn,
    }
    weight_dtype = weight_dtype_map.get(
        config.get("training", {}).get("weight_dtype", "bf16"), torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VRAM optimization options (all default-enabled for 16GB safety)
    quantize_mode = config.get("training", {}).get("quantize") or "none"  # none|int8|nf4|torchao_float8|torchao_int8|torchao_int4
    # Text encoder quantization for the cache build (the H3 Qwen3-VL encoder
    # is ~15GB bf16; nf4/int8 keeps the cache phase within VRAM).  Same bnb
    # helpers as the transformer path; applied to every text encoder load.
    text_encoder_quantize = config.get("training", {}).get("text_encoder_quantize") or "none"  # none|nf4|int8
    block_swap = config.get("training", {}).get("block_swap", 0)  # 0=disabled
    # offload_base_weights=true (default): BouncingOffloader keeps frozen torchao
    # base weights on CPU (pinned) and streams them H2D per forward/backward.
    # Set false to keep the whole quantized model on GPU — avoids the H2D
    # bottleneck entirely when VRAM is sufficient (~13GB for fp8 krea2) or when
    # the host PCIe link is degraded (slow H2D makes offloaded steps 10-20x slower).
    offload_base_weights = config.get("training", {}).get("offload_base_weights", True)
    offload_vae = config.get("training", {}).get("offload_vae", True)
    offload_text_encoder = config.get("training", {}).get("offload_text_encoder", True)

    # Auto-derive mixed_precision from weight_dtype if not explicitly set.
    # This controls accelerator.autocast() which reduces activation memory
    # by running matmuls/attention in the mixed-precision dtype.
    # Reference: T2ITrainer uses --mixed_precision bf16/fp8.
    weight_dtype_str = config.get("training", {}).get("weight_dtype", "bf16")
    mixed_precision = config.get("training", {}).get("mixed_precision", None)
    if mixed_precision is None:
        # Auto-map: bf16->bf16, fp16->fp16, fp8->fp8, else "no"
        mixed_precision = weight_dtype_str if weight_dtype_str in ("bf16", "fp16", "fp8") else "no"

    # fp8 mixed_precision in Accelerate requires transformer_engine or MS-AMP.
    # Without those libraries, gracefully downgrade to bf16 autocast.
    # NOTE: MXFP8 via torchao does NOT reduce VRAM — it's a speed optimization
    # for Blackwell GPUs. For VRAM savings use quantize="int8" instead.
    if mixed_precision == "fp8":
        fp8_available = False
        # Catch Exception broadly, not just ImportError: transformer_engine
        # may be installed but have a broken native library (OSError on .so
        # load, undefined symbol, ABI mismatch, etc.).
        try:
            import transformer_engine  # noqa: F401
            fp8_available = True
        except Exception:
            pass
        if not fp8_available:
            try:
                import msamp  # noqa: F401
                fp8_available = True
            except Exception:
                pass
        if not fp8_available:
            logger.warning(
                "mixed_precision='fp8' requires working transformer_engine or "
                "MS-AMP (installed but broken libraries don't count). "
                "Falling back to 'bf16'. For VRAM savings on large models, "
                "use quantize='int8' + block_swap instead."
            )
            mixed_precision = "bf16"

    config.setdefault("training", {})["mixed_precision"] = mixed_precision
    logger.info(f"Mixed precision: {mixed_precision} (weight_dtype={weight_dtype_str})")

    model_path = config.get("model_path", "")
    transformer_path = config.get(
        "transformer_path",
        os.path.join(model_path, config.get("transformer_subfolder", "transformer")),
    )
    vae_path = config.get("vae_path", os.path.join(model_path, "vae"))
    text_encoder_path = config.get(
        "text_encoder_path", os.path.join(model_path, "text_encoder")
    )
    tokenizer_path = config.get(
        "tokenizer_path", os.path.join(model_path, "tokenizer")
    )

    # ── Parse dataset configs & check cache status ─────────────
    from UnifiedTrainer.data.config_schema import parse_dataset_configs
    from UnifiedTrainer.data.cache_manager import CacheManager

    data_cfg = config.get("data", config)
    cache_dir = data_cfg.get("cache_dir", "cache")

    dataset_configs = parse_dataset_configs(config)
    logger.info(f"Parsed {len(dataset_configs)} dataset config(s)")
    for i, ds_cfg in enumerate(dataset_configs):
        logger.info(
            f"  Dataset {i}: {len(ds_cfg.image_configs)} image_types, "
            f"{len(ds_cfg.target_configs)} target_groups, "
            f"{len(ds_cfg.reference_configs)} reference_groups, "
            f"{len(ds_cfg.caption_configs)} caption_groups, "
            f"{len(ds_cfg.batch_configs)} batch_configs"
        )

    cache_mgr = CacheManager(cache_dir)
    empty_embedding_path = os.path.join(cache_dir, f"empty_embedding.{adapter.name}.npz")
    suffix_embedding_path = os.path.join(cache_dir, f"suffix_embedding.{adapter.name}.npz")

    # Suffix embedding control (matches training.include_suffix in config)
    include_suffix = config.get("training", {}).get("include_suffix", True)

    # ── Cache validity check (aligned with T2ITrainer: single decision point,
    #     no separate post-build "verify" pass) ───────────────────────────
    def _check_cache_validity():
        """Returns (valid, reason). Valid = cache can be reused as-is."""
        if not cache_mgr.exists():
            return False, "no cache index found"

        # Multi-dataset completeness: verify all configured datasets
        # are present in the existing train index.  Detects:
        #   - New datasets added to the config after initial cache build
        #   - Dataset path changes (old path in index, new path in config)
        try:
            train_rows = cache_mgr.load_train_index()
            if train_rows:
                indexed_datasets = set(
                    r.get("dataset", "") for r in train_rows
                )
                configured_datasets = set(
                    ds_cfg.train_data_dir
                    for ds_cfg in dataset_configs
                    if ds_cfg.train_data_dir
                )
                missing = configured_datasets - indexed_datasets
                if missing:
                    return False, f"new datasets not cached: {missing}"
                stale = indexed_datasets - configured_datasets
                if stale:
                    return False, f"dataset_dir changed (stale={stale})"
        except Exception:
            pass

        # Recreate flags override validity
        if any(ds_cfg.recreate_cache for ds_cfg in dataset_configs):
            return False, "recreate_cache flag set"
        if any(ds_cfg.recreate_latents for ds_cfg in dataset_configs):
            return False, "recreate_latents flag set"
        if any(ds_cfg.recreate_embeddings for ds_cfg in dataset_configs):
            return False, "recreate_embeddings flag set"

        # Empty embedding must exist (needed for caption_dropout)
        if not os.path.exists(empty_embedding_path):
            return False, f"empty embedding missing ({adapter.name})"

        # Spot-check: verify a few random datarows have both latent .pt
        # AND embedding .npz files on disk.  Catches partial cache corruption.
        try:
            train_rows = cache_mgr.load_train_index()
            if train_rows:
                _sample_rows = random.sample(train_rows, min(3, len(train_rows)))
                for row in _sample_rows:
                    jp = row.get("json_path", "")
                    if not jp or not os.path.exists(jp):
                        return False, f"per-sample JSON missing: {jp}"
                    try:
                        with open(jp, "r", encoding="utf-8") as f:
                            sample = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        return False, f"corrupt per-sample JSON: {jp}"
                    # Check latent files
                    for section in ("targets", "references"):
                        for _role, entry in sample.get(section, {}).items():
                            if not isinstance(entry, dict):
                                continue
                            lp = entry.get("latent_path", "")
                            if lp and not os.path.exists(lp):
                                return False, f"latent file missing: {lp}"
                    # Check embedding files
                    for _ckey, cap_entry in sample.get("captions", {}).items():
                        if not isinstance(cap_entry, dict):
                            continue
                        npz = cap_entry.get("npz_path", "")
                        if npz and not os.path.exists(npz):
                            return False, f"embedding .npz missing: {npz}"
        except Exception:
            pass  # if spot-check fails for any reason, rebuild to be safe

        return True, ""

    cache_valid, cache_invalid_reason = _check_cache_validity()
    need_cache_build = not cache_valid

    if not cache_valid:
        logger.warning(f"Cache invalid: {cache_invalid_reason}. Rebuilding...")
        # When dataset paths change, force recreate to overwrite stale
        # per-sample JSONs (mapping_key/original_image_path point to old dir).
        if "dataset_dir changed" in cache_invalid_reason:
            for ds_cfg in dataset_configs:
                ds_cfg.recreate_cache = True
        # For new datasets, DON'T set recreate_cache — existing cache is
        # valid. The builder skips already-cached samples and only builds
        # the missing ones (incremental cache build).

    need_vae = need_cache_build or any(
        ds_cfg.recreate_cache or ds_cfg.recreate_latents for ds_cfg in dataset_configs
    )
    need_text_encoder = need_cache_build or any(
        ds_cfg.recreate_cache or ds_cfg.recreate_embeddings for ds_cfg in dataset_configs
    )
    if not need_text_encoder and not os.path.exists(empty_embedding_path):
        need_text_encoder = True

    # ── Standalone suffix embedding creation (no full cache rebuild) ────
    # When the cache is valid but the suffix embedding file is missing
    # (e.g. upgraded from a pre-suffix codebase), load the text encoder
    # just long enough to create it — no VAE or per-sample re-encoding.
    if not need_cache_build and include_suffix and not os.path.exists(suffix_embedding_path):
        if hasattr(adapter, "encode_suffix_embedding"):
            logger.info("=== Creating suffix embedding (cache valid, suffix missing) ===")
            te = None
            tok = None
            proc = None
            try:
                if text_encoder_path:
                    te = adapter.load_text_encoder(text_encoder_path, weight_dtype)
                    _quantize_text_encoder(te, text_encoder_quantize, weight_dtype)
                if tokenizer_path:
                    tok = adapter.load_tokenizer(tokenizer_path)
                if hasattr(adapter, "load_processor"):
                    proc_path = config.get("processor_path", text_encoder_path or os.path.join(model_path, "processor"))
                    if proc_path:
                        try:
                            proc = adapter.load_processor(proc_path, tokenizer_path=tokenizer_path)
                        except TypeError:
                            proc = adapter.load_processor(proc_path)
                if te is not None and tok is not None and proc is not None:
                    te = te.to(device)
                    te.requires_grad_(False)
                    from UnifiedTrainer.data.embedding_cache import EmbeddingCache
                    suffix_emb = adapter.encode_suffix_embedding(te, tok, proc, device, torch.float32)
                    if suffix_emb and "prompt_embed" in suffix_emb:
                        EmbeddingCache.save(suffix_embedding_path, suffix_emb)
                        logger.info(f"Suffix embedding saved to {suffix_embedding_path}")
            except Exception as e:
                logger.warning(f"Failed to create suffix embedding: {e}")
            finally:
                if te is not None:
                    del te
                te = None
                tok = None
                proc = None
                gc.collect()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache()

    # ── Phase 1: Load cache-only components (VAE + text encoder) ──
    # Only load if we need to build/complete the cache
    if need_cache_build:
        logger.info("=== Phase 1: Cache components (VAE + text encoder) ===")

        vae = None
        if need_vae:
            logger.info(f"Loading VAE from {vae_path}...")
            vae = adapter.load_vae(vae_path, weight_dtype)
        else:
            logger.info("Skipping VAE load (latents preserved)")

        text_encoder = None
        tokenizer = None
        if need_text_encoder:
            if text_encoder_path:
                logger.info(f"Loading text encoder from {text_encoder_path}...")
                text_encoder = adapter.load_text_encoder(text_encoder_path, weight_dtype)
                _quantize_text_encoder(text_encoder, text_encoder_quantize, weight_dtype)
            if tokenizer_path:
                tokenizer = adapter.load_tokenizer(tokenizer_path)
        else:
            logger.info("Skipping text encoder load (embeddings preserved)")

        processor = None
        if need_text_encoder and hasattr(adapter, "load_processor"):
            processor_path = config.get(
                "processor_path", text_encoder_path or os.path.join(model_path, "processor")
            )
            if processor_path:
                logger.info(f"Loading processor from {processor_path}...")
                try:
                    processor = adapter.load_processor(processor_path, tokenizer_path=tokenizer_path)
                except TypeError:
                    processor = adapter.load_processor(processor_path)

        if vae is not None:
            vae = vae.to(device)
            vae.requires_grad_(False)
        if text_encoder is not None:
            text_encoder = text_encoder.to(device)
            text_encoder.requires_grad_(False)
        logger.info("Cache components moved to GPU for cache build")

        logger.info("Building cache...")
        from UnifiedTrainer.data.cache_builder import CacheBuilder
        all_datarows = []
        for ds_idx, ds_cfg in enumerate(dataset_configs):
            ds_name = ds_cfg.train_data_dir or f"dataset_{ds_idx}"
            builder = CacheBuilder(ds_cfg, cache_dir, adapter, dataset_name=ds_name)
            ds_datarows = builder.build(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                device=device,
                processor=processor,
            )
            all_datarows.extend(ds_datarows)
        logger.info("Cache built successfully")

        # Save combined train index (T2ITrainer pattern: single index file)
        cache_mgr.save_train_index(all_datarows)
        # Also save full index (never modified by val split — source for re-splitting)
        cache_mgr.save_full_index(all_datarows)
        logger.info(f"Saved train index: {len(all_datarows)} total datarows")

        # Delete cache components entirely -they'll be reloaded only for validation
        if vae is not None:
            del vae
        if text_encoder is not None:
            del text_encoder
        text_encoder = None
        vae = None
        # accelerator.free_memory() is more thorough than gc.collect() alone:
        # it also clears all Accelerate-tracked tensors and hooks.
        try:
            from accelerate import Accelerator
            Accelerator.clear_envmanager()
        except Exception:
            pass
        gc.collect()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.empty_cache()
        logger.info("VAE + text encoder deleted after cache build (freed VRAM)")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            logger.info("Peak VRAM stats reset (measuring training-only usage)")
    else:
        vae = None
        text_encoder = None
        logger.info(f"Cache valid, reusing {cache_dir}")
        logger.info("Cache components not loaded (will reload for validation)")

    # ── Phase 2: Load training components (transformer + LoRA) ──
    logger.info("=== Phase 2: Training components (transformer + LoRA) ===")

    logger.info(f"Loading transformer from {transformer_path}...")
    if quantize_mode == "nf4":
        transformer = adapter.load_transformer(transformer_path, weight_dtype)
        _replace_linear_with_4bit(transformer, compute_dtype=weight_dtype, quant_type="nf4")
        # Free old bf16 Linear weights BEFORE GPU move - otherwise they
        # survive through the .to(device) call and double VRAM usage.
        gc.collect()
        transformer = transformer.to(device)  # triggers Params4bit quantization
        logger.info(f"Transformer loaded + NF4 quantized on GPU")
    elif quantize_mode == "int8":
        transformer = adapter.load_transformer(transformer_path, weight_dtype)
        _replace_linear_with_8bit(transformer)
        # Must use .cuda() not .to(device): Int8Params overrides .cuda() to trigger
        # int8 quantization, but .to() just moves bf16 data without quantizing.
        # Params4bit (NF4) overrides .to() so .to(device) works for NF4, but
        # Int8Params only overrides .cuda(). Without this, weights stay in bf16
        # (~26GB for a 13B model) instead of int8 (~13GB).
        transformer = transformer.cuda()
        if torch.cuda.is_available():
            logger.info(f"Transformer loaded + int8 quantized on GPU. "
                        f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
    elif quantize_mode.startswith("torchao_"):
        # torchao weight-only quantization (float8/int8/int4)
        # Per-block quantization to avoid VRAM peak of full bf16 model on GPU:
        # each block is moved to GPU one at a time, quantized, frozen, moved back.
        from UnifiedTrainer.utils.quantize import quantize_model as torchao_quantize_model
        transformer = adapter.load_transformer(transformer_path, weight_dtype)
        # Strip "torchao_" prefix to get the qtype name (e.g. "float8", "int8", "int4")
        qtype = quantize_mode[len("torchao_"):]
        logger.info(f"Applying torchao {qtype} quantization (per-block, low VRAM)...")
        # Exclusion patterns matching T2ITrainer: skip embeddings, norms,
        # adaLN modulation tables, timestep/position embeddings, final projection
        torchao_exclude = [
            "*embed*", "*norm*", "*scale_shift_table*", "*mod*",
            "*time_pos_embed*", "*pos_embed*", "*final*", "*lora*",
        ]
        torchao_quantize_model(
            transformer,
            qtype=qtype,
            device=device,
            dtype=weight_dtype,
            exclude_modules=torchao_exclude,
            low_vram=True,
        )
        # Move quantized model to GPU (now fp8, ~half the bf16 size)
        logger.info(f"Moving quantized transformer to GPU...")
        transformer = transformer.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"torchao {qtype} quantization applied. "
                        f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
        else:
            logger.info(f"torchao {qtype} quantization applied.")
    else:
        transformer = adapter.load_transformer(transformer_path, weight_dtype)

    # Freeze ALL base model parameters before applying adapter.
    transformer.requires_grad_(False)

    # ── Network type routing ─────────────────────────────────────────────
    # "lora" (default) → PEFT add_adapter() path (existing, unchanged)
    # "lokr" → LyCORIS library path (Kronecker-product adapter)
    network_type = config.get("training", {}).get("network_type", "lora")
    lycoris_net = None  # Set when network_type == "lokr"

    if network_type == "lokr":
        # ── LoKR via self-contained module (VRAM-optimized) ────────────
        from UnifiedTrainer.networks.lokr_module import LokrConfig, apply_lokr

        training_cfg = config.get("training", {})
        lokr_cfg = LokrConfig(
            rank=training_cfg.get("lora_rank", 4),
            alpha=training_cfg.get("lokr_alpha", 1.0),
            factor=training_cfg.get("lokr_factor", -1),
            model_type=training_cfg.get("lokr_model_type", "krea2"),
            target_modules=training_cfg.get("lokr_target_modules", None),
            # musubi `lokr_full_rank: true` → full-matrix W1/W2, rank/alpha
            # overridden to the 9999 sentinel inside LokrConfig (scale = 1.0).
            full_rank=training_cfg.get("lokr_full_rank", False),
        )

        # Prepare for k-bit training if quantized (same as LoRA path)
        if quantize_mode in ("nf4", "int8"):
            from peft.utils.other import prepare_model_for_kbit_training
            transformer = prepare_model_for_kbit_training(transformer)

        lycoris_net = apply_lokr(transformer, lokr_cfg)
        lycoris_net.to(device)

        logger.info(
            f"LoKR applied (self-contained): rank={lokr_cfg.rank}, "
            f"alpha={lokr_cfg.alpha}, factor={lokr_cfg.factor}, "
            f"model_type={lokr_cfg.model_type}, full_rank={lokr_cfg.full_rank}, "
            f"modules={lycoris_net.num_modules}"
        )

    else:
        # ── Standard LoRA via PEFT (existing code, unchanged) ────────────
        # Prepare for k-bit training if quantized
        if quantize_mode in ("nf4", "int8"):
            from peft.utils.other import prepare_model_for_kbit_training
            transformer = prepare_model_for_kbit_training(transformer)

    # Apply LoRA — use diffusers' add_adapter() like T2ITrainer.
    if network_type != "lokr":
        from peft import LoraConfig

        lora_rank = config.get("training", {}).get("lora_rank", 16)
        lora_alpha = config.get("training", {}).get("lora_alpha", lora_rank)
        lora_target_modules = config.get("training", {}).get(
            "lora_target_modules",
            ["to_q", "to_k", "to_v", "to_out", "to_gate", "ff.gate", "ff.up", "ff.down"],
        )
        # Resolve target_module patterns to explicit full Linear names for ALL
        # quantize modes. PEFT's native pattern matching can resolve to ModuleList
        # parents (e.g. "to_out" matches the ModuleList wrapping Linear+Dropout
        # instead of the inner "to_out.0" Linear) and then crashes with
        # "Target module ModuleList(...) is not supported". Substring matching
        # against nn.Linear names is unambiguous and matches what the torchao
        # path has always done (same target set on cloud runs).
        import torch.nn as nn
        resolved_targets = [
            name
            for name, module in transformer.named_modules()
            if isinstance(module, nn.Linear)
            and any(target in name for target in lora_target_modules)
        ]
        if not resolved_targets:
            raise ValueError(
                f"No Linear modules matched lora_target_modules={lora_target_modules}. "
                "Check the target module names for this model architecture."
            )
        logger.info(f"Resolved {len(resolved_targets)} LoRA targets: "
                    f"{resolved_targets[:3]}... (showing first 3)")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=resolved_targets,
            lora_dropout=0.0,
            bias="none",
        )
        # When using torchao quantization, bypass PEFT's TorchaoLoraLinear dispatcher.
        # PEFT 0.18.1's TorchaoLoraLinear requires a get_apply_tensor_subclass kwarg
        # that diffusers' add_adapter doesn't pass. Regular PEFT Linear works fine —
        # torchao's AffineQuantizedTensor handles F.linear via __torch_function__.
        if quantize_mode.startswith("torchao_"):
            import peft.tuners.lora.model as _peft_lora_model
            _original_dispatch_torchao = _peft_lora_model.dispatch_torchao
            _peft_lora_model.dispatch_torchao = lambda *a, **kw: None
            logger.info("Bypassed PEFT TorchaoLoraLinear dispatcher for torchao quantization")
            try:
                transformer.add_adapter(lora_config)
            finally:
                _peft_lora_model.dispatch_torchao = _original_dispatch_torchao
        else:
            transformer.add_adapter(lora_config)

    # ── BouncingOffloader: bounce frozen base weights to CPU for torchao models ──
    # T2ITrainer pattern: BouncingOffloader attaches after add_adapter() so PEFT
    # LoRA layers exist. Frozen quantized base weights are moved to CPU (pinned)
    # and bounced to GPU per forward/backward. LoRA adapter weights stay permanently
    # on GPU. This matches T2ITrainer's train_krea2_edit.py lines 1736-1742 and
    # saves ~6GB VRAM for torchao_float8 models.
    if quantize_mode.startswith("torchao_") and offload_base_weights:
        from UnifiedTrainer.utils.bouncing_offloader import BouncingOffloader
        bouncing_offloader = BouncingOffloader(device)
        unmanaged = bouncing_offloader.attach(transformer)
        bouncing_offloader.move_unmanaged_to_device(unmanaged)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("BouncingOffloader attached: frozen base weights on CPU, LoRA on GPU")
    elif quantize_mode.startswith("torchao_"):
        logger.info("offload_base_weights=false: keeping quantized base weights on GPU "
                    "(no BouncingOffloader)")

    # Gradient checkpointing — must be called AFTER add_adapter(), matching T2ITrainer.
    # T2ITrainer: add_adapter() (line 1335) → enable_gradient_checkpointing() (line 1338)
    if config.get("training", {}).get("gradient_checkpointing", True):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
            # CRITICAL: torch.utils.checkpoint silently skips when no tensor
            # argument requires grad.  After base-model freezing + LoRA, the
            # patched latents entering img_in/txt_in have requires_grad=False,
            # so hidden_states entering the first transformer block also has
            # requires_grad=False — checkpoint disengages and ALL 28 layers'
            # activations are stored (~50-60 GiB instead of ~4-5 GiB).
            #
            # Register a forward hook on img_in and txt_in to force their
            # outputs to require grad, re-enabling gradient checkpointing.
            def _force_output_grad(module, input, output):
                output.requires_grad_(True)
            for hook_target in ("img_in", "txt_in"):
                if hasattr(transformer, hook_target):
                    getattr(transformer, hook_target).register_forward_hook(
                        _force_output_grad
                    )
            logger.info("Gradient checkpointing enabled (after add_adapter) "
                        "+ img_in/txt_in output grad hooks")
        else:
            logger.warning("Gradient checkpointing requested but "
                           "enable_gradient_checkpointing() not available")

    transformer.train()
    if network_type == "lokr":
        logger.info(f"LoKR applied via LyCORIS: quantize={quantize_mode}")
    else:
        logger.info(f"LoRA applied via add_adapter(): rank={lora_rank}, alpha={lora_alpha}, "
                    f"targets={lora_target_modules}, quantize={quantize_mode}")

    # Diagnostic: verify only adapter params are trainable + checkpointing active
    total_params = sum(p.numel() for p in transformer.parameters())
    if network_type == "lokr" and lycoris_net is not None:
        trainable_params = sum(p.numel() for p in lycoris_net.parameters() if p.requires_grad)
    else:
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    gc_active = getattr(transformer, "gradient_checkpointing", False)
    logger.info(
        f"Model params: {total_params / 1e9:.2f}B total, "
        f"{trainable_params / 1e6:.2f}M trainable ({100 * trainable_params / max(total_params, 1):.2f}%), "
        f"gradient_checkpointing={gc_active}"
    )

    # For quantized models, only move to device (don't cast dtype -would break 4bit params)
    # torchao models with BouncingOffloader: skip .to(device) — the offloader already
    # placed Linear weights on CPU (pinned), LoRA params on GPU, and unmanaged modules
    # (norms/embeddings) on GPU. A .to(device) would undo the bouncing.
    if quantize_mode.startswith("torchao_") and offload_base_weights:
        # Model device state already correct from BouncingOffloader + move_unmanaged_to_device.
        # No .to(device) call needed.
        logger.info("torchao model: skipping .to(device) — BouncingOffloader handles placement")
    elif quantize_mode.startswith("torchao_"):
        # No offloader: move the whole quantized model to GPU. Device-only .to()
        # (no dtype cast) preserves torchao AffineQuantizedTensor params.
        transformer = transformer.to(device=device)
    elif quantize_mode in ("nf4", "int8"):
        transformer = transformer.to(device=device)
    else:
        transformer = transformer.to(device=device, dtype=weight_dtype)
    if torch.cuda.is_available():
        logger.info(f"After PEFT+device move. "
                    f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
    if network_type != "lokr":
        logger.info(
            f"LoRA applied: rank={lora_rank}, alpha={lora_alpha}, "
            f"targets={lora_target_modules}, quantize={quantize_mode}"
        )

    # ── Block swap: offload transformer blocks to CPU during forward/backward ─
    if block_swap > 0 and hasattr(transformer, "enable_block_swap"):
        # torchao-quantized models with BouncingOffloader: block_swap is
        # redundant (weights already on CPU).  Without BouncingOffloader,
        # block_swap can help on very constrained GPUs.
        if quantize_mode.startswith("torchao_") and offload_base_weights:
            logger.info(f"Block swap skipped for torchao {quantize_mode} "
                        "(BouncingOffloader already handles weight placement)")
        else:
            base_model = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
            if hasattr(base_model, "enable_block_swap"):
                base_model.enable_block_swap(block_swap, device)
                logger.info(f"Block swap enabled: {block_swap} blocks")

    # ── Create trainer ──────────────────────────────────────────
    from UnifiedTrainer.engine.trainer import Trainer

    trainer = Trainer(
        config, adapter=adapter, losses=losses,
        transformer=transformer, vae=None,
        lycoris_net=lycoris_net,
    )
    trainer.text_encoder = None  # not needed during training (cached embeddings)
    trainer.tokenizer = tokenizer if need_cache_build else None
    trainer.processor = processor if need_cache_build else None
    trainer.device = device
    logger.info(f"Trainer initialized with {len(losses)} loss modules")

    # ── DB mode: heartbeat callback ──────────────────────────────
    if db_ctx is not None:
        from UnifiedTrainer.engine.callbacks import Callback

        class _DBHeartbeatCallback(Callback):
            """每 log_every step 写 heartbeats 表（orchestrator 判活）。"""

            def __init__(self, db, task_id, every):
                self._db, self._tid, self._every = db, task_id, max(1, every)

            def on_step_end(self, step, loss, trainer):
                if step % self._every != 0:
                    return
                if not _is_rank0():  # P5：非 rank 0 不写心跳
                    return
                lr = None
                try:
                    if trainer.optimizer is not None:
                        lr = trainer.optimizer.param_groups[0].get("lr")
                except Exception:
                    pass
                vram_mb = None
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        vram_mb = int(_t.cuda.memory_allocated() / 1024**2)
                except Exception:
                    pass
                self._db.heartbeat(self._tid, step, loss=loss, lr=lr,
                                   vram_mb=vram_mb)

        hb_every = config.get("training", {}).get("log_every", 10)
        trainer.callbacks.add(
            _DBHeartbeatCallback(db_ctx, args.task_id, hb_every))
        if _is_rank0():
            db_ctx.heartbeat(args.task_id, 0)  # 启动心跳，防误判僵死
        logger.info(f"DB heartbeat enabled: every {hb_every} steps")

        # P2: HookManager 挂载（仅 DB 模式）。trainer 在 step 边界
        # 通过 getattr(self, "hook_manager", None) 消费 hooks 表。
        # P5：DDP 下仅 rank 0 挂载——非 rank 0 不轮询 hooks 表。
        if _is_rank0():
            from UnifiedTrainer.engine.hook_manager import HookManager
            trainer.hook_manager = HookManager(
                db_ctx, args.task_id, config=config)
            logger.info("HookManager attached (DB mode)")

    # Setup: move loss-owned modules (e.g. LISA decoder) to device first,
    # then create optimizer (which collects both LoRA and loss-module params).
    trainer.setup_loss_modules()
    trainer.setup_optimizer()
    trainer.setup_lr_scheduler()
    trainer.setup_ema()

    # ── Setup Accelerator for memory-efficient training ────────────
    # The Accelerator enables:
    #   - accelerator.autocast(): mixed-precision forward (reduces activation memory)
    #   - accelerator.accumulate(): efficient gradient accumulation
    #   - accelerator.backward(): memory-efficient backward pass
    # Reference: T2ITrainer creates Accelerator with mixed_precision and
    # uses accelerator.prepare(transformer, optimizer, lr_scheduler).
    trainer.setup_accelerator()

    # Update device reference (accelerator.prepare may move model)
    if trainer.accelerator is not None:
        device = trainer.accelerator.device
        trainer.device = device

    # Baseline VRAM before training loop
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"VRAM baseline before training: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # ── Setup callbacks (reporter: tensorboard / wandb) ─────────
    reporter_cfg = config.get("reporter", {})
    reporter_type = reporter_cfg.get("type", "tensorboard")  # default: tensorboard
    output_dir = config.get("output", {}).get("dir", "output")
    save_name = config.get("output", {}).get("save_name", "experiment")

    if reporter_type == "tensorboard" and _is_rank0():
        from UnifiedTrainer.engine.callbacks import TensorBoardCallback

        tb_log_dir = reporter_cfg.get("log_dir") or os.path.join(output_dir, "tensorboard")
        tb_port = reporter_cfg.get("port", 6006)
        tb_log_every = reporter_cfg.get("log_every", 1)
        tb_cb = TensorBoardCallback(
            log_dir=tb_log_dir,
            port=tb_port,
            log_every=tb_log_every,
            comment=f"_{save_name}",
        )
        trainer.callbacks.add(tb_cb)
        logger.info(f"TensorBoard reporter enabled: logdir={tb_log_dir}, port={tb_port}")

    elif reporter_type == "wandb" and _is_rank0():
        wandb_cfg = config.get("wandb", {})
        from UnifiedTrainer.engine.callbacks import WandBCallback

        wandb_cb = WandBCallback(
            project=wandb_cfg.get("project", "UnifiedTrainer"),
            config=config,
            run_name=wandb_cfg.get("run_name"),
        )
        trainer.callbacks.add(wandb_cb)
        logger.info(f"WandB reporter enabled: project={wandb_cb.project}")

    elif reporter_type == "none":
        logger.info("Reporter disabled (type=none)")

    else:
        logger.warning(f"Unknown reporter type '{reporter_type}', defaulting to tensorboard")
        from UnifiedTrainer.engine.callbacks import TensorBoardCallback

        tb_log_dir = reporter_cfg.get("log_dir") or os.path.join(output_dir, "tensorboard")
        tb_cb = TensorBoardCallback(log_dir=tb_log_dir, comment=f"_{save_name}")
        trainer.callbacks.add(tb_cb)

    # Legacy: wandb section still works if explicitly enabled (additive)
    wandb_cfg = config.get("wandb", {})
    if (_is_rank0() and wandb_cfg.get("enabled", False)
            and reporter_type != "wandb"):
        from UnifiedTrainer.engine.callbacks import WandBCallback

        wandb_cb = WandBCallback(
            project=wandb_cfg.get("project", "UnifiedTrainer"),
            config=config,
            run_name=wandb_cfg.get("run_name"),
        )
        trainer.callbacks.add(wandb_cb)
        logger.info(f"WandB logging also enabled: project={wandb_cb.project}")

    # Store config on trainer for callback access (hparams logging)
    trainer._config = config

    # Dispatch on_train_start before the loop begins
    trainer.callbacks.on_train_start(trainer)

    # ── Resume logic ────────────────────────────────────────────
    # Two resume modes (CLI flags take priority over config["resume"]):
    #   --resume-full <path>   → weights + optimizer + scheduler + RNG + step/epoch
    #   --resume <path>        → LoRA weights only, fresh optimizer/step
    #
    # Config-level alternative (in JSON):
    #   "resume": {
    #       "checkpoint": "/path/to/lora_epoch9.safetensors",
    #       "full": true                 # false = LoRA-only
    #   }
    #
    # For full resume, the _training_state.pt is derived from the safetensors
    # path:  lora_epoch3.safetensors → lora_epoch3_training_state.pt
    from UnifiedTrainer.engine.checkpoint import CheckpointManager

    # Resolve resume settings: CLI args override config["resume"]
    resume_cfg = config.get("resume", {})
    resume_ckpt = args.resume_full or args.resume or resume_cfg.get("checkpoint")
    is_full_resume = args.resume_full is not None or (
        args.resume is None and resume_cfg.get("full", False)
    )

    if resume_ckpt:
        logger.info(f"Resume checkpoint: {resume_ckpt} (full={is_full_resume})")
        logger.info(f"  resume_cfg={resume_cfg}")
        logger.info(f"  args.resume_full={args.resume_full}, args.resume={args.resume}")

        output_dir_cfg = config.get("output", {}).get("dir", "output")
        save_name_cfg = config.get("output", {}).get("save_name", "lora")
        ckpt = CheckpointManager(output_dir_cfg, save_name_cfg)

        # Step 1: Load adapter weights (both modes)
        if network_type == "lokr" and lycoris_net is not None:
            meta = ckpt.load_lokr(lycoris_net, resume_ckpt)
            logger.info(f"LoKR weights loaded from {resume_ckpt}")
            logger.info(f"  LoKR metadata: step={meta.get('step')}, epoch={meta.get('epoch')}, modules={meta.get('num_modules')}")
        else:
            meta = ckpt.load_lora(trainer.transformer, resume_ckpt)
            logger.info(f"LoRA weights loaded from {resume_ckpt}")
            logger.info(f"  LoRA metadata: step={meta.get('step')}, epoch={meta.get('epoch')}, params={meta.get('num_params')}")

        if is_full_resume:
            # Step 2: Load full training state (optimizer + scheduler + RNG + step/epoch)
            state_path = CheckpointManager.get_training_state_path(resume_ckpt)
            logger.info(f"  Full resume state_path: {state_path}")
            logger.info(f"  Full resume state_path exists: {state_path.exists()}")
            if state_path.exists():
                # Pass lr_scheduler=None: the scheduler will be rebuilt later
                # with correct total_steps. Restoring old scheduler state from
                # a different batch schedule would place the cosine at the wrong
                # position (e.g. step 11058 on a 6000-step schedule → LR ≈ 0).
                state_meta = ckpt.load_training_state(
                    trainer.optimizer, None, str(state_path)
                )
                trainer.step = state_meta["step"]
                # saved epoch = last COMPLETED epoch, so start from the next one
                trainer.epoch = state_meta["epoch"] + 1
                extra_state = state_meta.get("extra_state")
                if extra_state and hasattr(trainer.noise_selector, "load_state_dict"):
                    try:
                        trainer.noise_selector.load_state_dict(extra_state["noise_selector"])
                        logger.info("FULL resume: restored noise_selector state (improved XM baselines)")
                    except Exception as e:
                        logger.warning(f"FULL resume: failed to restore noise_selector state: {e}")
                logger.info(
                    f"FULL resume: restored optimizer, scheduler, RNG, "
                    f"step={trainer.step}, starting at epoch={trainer.epoch}"
                )
            else:
                logger.warning(
                    f"Full resume requested but training state not found "
                    f"at {state_path}. Falling back to LoRA-only resume "
                    f"(fresh optimizer, step/epoch from checkpoint metadata)."
                )
                # Fall through to LoRA-only path below
                is_full_resume = False
        if not is_full_resume and resume_ckpt:
            # LoRA-only: restore global step/epoch from checkpoint metadata for
            # continuous WandB x-axis and progress tracking. Optimizer/scheduler
            # start fresh (expected for LoRA-only resume).
            saved_step = meta.get("step", 0)
            saved_epoch = meta.get("epoch", -1)
            trainer.step = saved_step
            trainer.epoch = saved_epoch + 1 if saved_epoch >= 0 else 0
            logger.info(
                f"LoRA-only resume: weights loaded, "
                f"global step={trainer.step}, epoch={trainer.epoch} "
                f"(optimizer/scheduler restarted from scratch)"
            )

    # ── Create dataset & dataloader ────────────────────────────
    from UnifiedTrainer.data.dataset import UnifiedDataset, collate_fn, BucketBatchSampler
    from torch.utils.data import DataLoader

    # ── Index rebuild + validation split ──────────────────────────
    # T2ITrainer pattern: split the datarow LIST, not group IDs.
    #
    # When rebuild_index=True (config flag), the full index is reconstructed
    # from per-sample JSONs and re-split into train/val regardless of existing
    # val index. Also triggers on cache build or when no val index exists yet.
    #
    # Otherwise: reuse existing train/val indexes to avoid shrinking the
    # train set on every restart.
    val_cfg = config.get("validation", {})
    val_split_ratio = data_cfg.get("val_split_ratio", 0.0)
    rebuild_index = data_cfg.get("rebuild_index", False)
    if val_split_ratio > 0 or val_cfg.get("val_loss", False) or val_cfg.get("generate_images", False):
        effective_ratio = val_split_ratio if val_split_ratio > 0 else 0.1
        val_seed = val_cfg.get("val_seed", 42)

        existing_val = cache_mgr.load_val_index()
        do_split = need_cache_build or rebuild_index or not existing_val

        if not do_split:
            logger.info(
                f"Val split already exists ({len(existing_val)} val samples), reusing"
            )
        else:
            # Rebuild full index from per-sample JSONs, then split
            primary_ds_name = (
                dataset_configs[0].train_data_dir
                if dataset_configs
                else "default"
            )
            full_datarows = cache_mgr.rebuild_full_index(dataset_name=primary_ds_name)
            logger.info(f"Rebuilt full index: {len(full_datarows)} datarows")
            if full_datarows:
                val_datarows, new_train_datarows = cache_mgr.create_val_split(
                    train_datarows=full_datarows,
                    ratio=effective_ratio,
                    seed=val_seed,
                    force=True,
                )
                cache_mgr.save_train_index(new_train_datarows)
                logger.info(
                    f"Val split created: {len(val_datarows)} val / {len(new_train_datarows)} train "
                    f"(ratio={effective_ratio}, seed={val_seed})"
                )
    elif rebuild_index:
        # rebuild_index=True but no validation configured — still rebuild
        # the train index from per-sample JSONs so newly added datasets
        # are included in the sample count.
        primary_ds_name = (
            dataset_configs[0].train_data_dir
            if dataset_configs
            else "default"
        )
        full_datarows = cache_mgr.rebuild_full_index(dataset_name=primary_ds_name)
        if full_datarows:
            cache_mgr.save_train_index(full_datarows)
            logger.info(f"Index rebuilt (no val split): {len(full_datarows)} datarows")

    dataset = UnifiedDataset(
        adapter=adapter,
        cache_dir=cache_dir,
        config=data_cfg,
        split="train",
    )

    # ── Resolve per-dataset batch sizes, repeats, and weights ────
    # Resolution: dataset_config.batch_size → global training.batch_size → 1
    dataset_batch_sizes = dataset.get_dataset_batch_sizes(trainer.batch_size)
    if any(bs != trainer.batch_size for bs in dataset_batch_sizes.values()):
        logger.info(f"Per-dataset batch sizes: {dataset_batch_sizes}")

    # Effective batch size — gates per-sample reference_dropout (unsupported
    # with batch_size > 1, where batches must stay homogeneous w.r.t. refs).
    dataset.batch_size = (
        max(dataset_batch_sizes.values()) if dataset_batch_sizes else trainer.batch_size
    )

    dataset_repeats = dataset.get_dataset_repeats()
    if any(r != 1 for r in dataset_repeats.values()):
        logger.info(f"Per-dataset repeats: {dataset_repeats}")

    dataset_weights = dataset.get_dataset_weights()
    if any(w != 1.0 for w in dataset_weights.values()):
        logger.info(f"Per-dataset sample weights: {dataset_weights}")

    train_sampler = BucketBatchSampler(
        dataset,
        batch_size=trainer.batch_size,
        dataset_batch_sizes=dataset_batch_sizes,
        dataset_repeats=dataset_repeats,
        dataset_weights=dataset_weights,
        drop_last=False,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ── Build validation dataloader if val_loss is enabled ─────────
    val_dataloader = None
    if val_cfg.get("val_loss", False):
        val_dataset = UnifiedDataset(
            adapter=adapter,
            cache_dir=cache_dir,
            config=data_cfg,
            split="val",
        )
        val_dataset.batch_size = max(
            val_dataset.get_dataset_batch_sizes(trainer.batch_size).values()
        )
        if len(val_dataset) > 0:
            val_dataloader = DataLoader(
                val_dataset,
                batch_sampler=BucketBatchSampler(
                    val_dataset,
                    batch_size=trainer.batch_size,
                    dataset_batch_sizes=val_dataset.get_dataset_batch_sizes(trainer.batch_size),
                    dataset_repeats=val_dataset.get_dataset_repeats(),
                    dataset_weights=val_dataset.get_dataset_weights(),
                    drop_last=False,
                    shuffle=False,
                ),
                collate_fn=collate_fn,
                num_workers=data_cfg.get("num_workers", 0),
            )
            logger.info(f"Validation dataset: {len(val_dataset)} samples")
        else:
            logger.warning(
                "val_loss enabled but no validation data found. "
                "Set val_split_ratio > 0 in config to create a val split."
            )

    num_epochs = config.get("training", {}).get("num_epochs", 10)
    # Use actual batch count from the sampler (accounts for per-dataset batch sizes)
    num_batches = len(train_sampler)
    steps_per_epoch = max(1, math.ceil(num_batches / trainer.gradient_accumulation_steps))

    # Full training plan (all epochs, from epoch 0)
    if trainer.max_steps != -1:
        full_total_steps = trainer.max_steps
    else:
        full_total_steps = num_epochs * steps_per_epoch

    # Account for resume: the scheduler should only cover remaining steps.
    # The old schedule (different batch sizes / steps_per_epoch) is incompatible,
    # so the scheduler always starts fresh over the remaining training.
    start_epoch = trainer.epoch
    if trainer.max_steps != -1:
        scheduler_total_steps = max(1, full_total_steps - trainer.step)
    else:
        scheduler_total_steps = max(1, full_total_steps - start_epoch * steps_per_epoch)

    bs_summary = (
        f"per-dataset {dataset_batch_sizes}"
        if any(bs != trainer.batch_size for bs in dataset_batch_sizes.values())
        else f"{trainer.batch_size}"
    )
    if start_epoch > 0:
        logger.info(
            f"Starting training: epoch {start_epoch}–{num_epochs} "
            f"({scheduler_total_steps} remaining steps, {steps_per_epoch} steps/epoch, "
            f"full plan: {full_total_steps} steps) "
            f"(dataset={len(dataset)}, repeats={getattr(dataset, 'repeats', 1)}, batch_size={bs_summary}, batches/epoch={num_batches})"
        )
    else:
        logger.info(
            f"Starting training: {num_epochs} epochs × {steps_per_epoch} steps/epoch = {scheduler_total_steps} total steps "
            f"(dataset={len(dataset)}, repeats={getattr(dataset, 'repeats', 1)}, batch_size={bs_summary}, batches/epoch={num_batches})"
        )

    # ── Rebuild LR scheduler with accurate total_steps ────────────
    # The scheduler was initially created in setup_lr_scheduler() before the
    # dataset length was known. Now that we have the real remaining steps,
    # rebuild it so cosine/linear decay curves are accurate.
    #
    # Key: use scheduler_total_steps (remaining), NOT full_total_steps —
    # EXCEPT after a FULL resume. On full resume the optimizer state is
    # restored and training continues the ORIGINAL schedule; restarting a
    # fresh decay curve over the remaining steps would jump the LR back to
    # peak (e.g. a cosine resumed at epoch 20/40 should be at ~0.5×LR, not
    # 1.0×LR).  Instead, rebuild over the FULL horizon and fast-forward the
    # scheduler to trainer.step so the decay continues from the correct
    # position.
    #
    # LoRA-only resume keeps the old behaviour: optimizer/scheduler are
    # fresh anyway, so a fresh curve over the remaining steps is fine.
    continue_original_schedule = (
        is_full_resume and trainer.step > 0 and trainer.max_steps == -1
    )
    if trainer.lr_scheduler_name not in ("constant",):
        config.setdefault("training", {})["_estimated_total_steps"] = (
            full_total_steps if continue_original_schedule else scheduler_total_steps
        )
        trainer.setup_lr_scheduler()
        if continue_original_schedule:
            # LambdaLR cheap lambda — stepping is O(1); safe to fast-forward.
            # Scheduler steps are 1:1 with optimizer steps (trainer.step),
            # so advancing by trainer.step restores the exact LR position.
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                for _ in range(trainer.step):
                    trainer.lr_scheduler.step()
            logger.info(
                f"LR scheduler fast-forwarded to resumed step {trainer.step} "
                f"(continues original schedule over {full_total_steps} steps; "
                f"current lr={trainer.lr_scheduler.get_last_lr()[0]:.3e})"
            )
        # Re-prepare the new scheduler with accelerator if active
        if trainer.accelerator is not None:
            block_swap = config.get("training", {}).get("block_swap", 0)
            if block_swap > 0:
                trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                    trainer.optimizer, trainer.lr_scheduler
                )
            else:
                trainer.transformer, trainer.optimizer, trainer.lr_scheduler = (
                    trainer.accelerator.prepare(
                        trainer.transformer, trainer.optimizer, trainer.lr_scheduler
                    )
                )
        logger.info(
            f"LR scheduler rebuilt: '{trainer.lr_scheduler_name}' "
            f"with total_steps={full_total_steps if continue_original_schedule else scheduler_total_steps}"
            f"{' (continuing original schedule from step ' + str(trainer.step) + ')' if continue_original_schedule else ''}"
        )

    # NOTE: We intentionally do NOT restore old scheduler state onto the
    # rebuilt scheduler. The old schedule may have different steps_per_epoch
    # (e.g. before per-dataset batch sizing), so the saved last_epoch would
    # place the cosine at the wrong position. Fresh runs and LoRA-only
    # resumes use a fresh curve over the remaining steps; FULL resumes
    # rebuild the original full-horizon curve and fast-forward to
    # trainer.step, which reproduces the correct LR without the old state.

    for epoch in range(trainer.epoch, num_epochs):
        if trainer.max_steps != -1 and trainer.step >= trainer.max_steps:
            break

        result = trainer.train_epoch(epoch, dataloader)
        logger.info(f"Epoch {epoch}: avg_loss={result['loss']:.6f}")

        # ── Per-epoch output directory ───────────────────────────────
        output_dir = config.get("output", {}).get("dir", "output")
        save_name = config.get("output", {}).get("save_name", "lora")
        epoch_dir = os.path.join(output_dir, f"{save_name}-{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # ── Save checkpoint + ComfyUI conversion FIRST ───────────────
        if trainer.should_save_checkpoint(epoch):
            from UnifiedTrainer.engine.checkpoint import CheckpointManager

            ckpt_mgr = CheckpointManager(epoch_dir, save_name)
            if network_type == "lokr" and lycoris_net is not None:
                path = ckpt_mgr.save_lokr(
                    lycoris_net, trainer.step, epoch, config
                )
            else:
                path = ckpt_mgr.save_lora(
                    trainer.transformer, trainer.step, epoch, config
                )
            logger.info(f"Checkpoint saved: {path}")
            # Save full training state for exact resume (--resume-full)
            if trainer.optimizer is not None:
                extra = (
                    {"noise_selector": trainer.noise_selector.state_dict()}
                    if hasattr(trainer.noise_selector, "state_dict")
                    else None
                )
                state_path = ckpt_mgr.save_training_state(
                    optimizer=trainer.optimizer,
                    lr_scheduler=trainer.lr_scheduler,
                    step=trainer.step,
                    epoch=epoch,
                    epoch_idx=epoch,
                    config=config,
                    extra_state=extra,
                )
                logger.info(f"Training state saved: {state_path} (step={trainer.step}, epoch={epoch})")
            else:
                logger.warning(f"Training state NOT saved: trainer.optimizer is None!")
            trainer.callbacks.on_checkpoint(trainer.step, str(path), trainer)

        # ── Validation loss (controlled RNG, no gradient) ─────────────
        val_every = val_cfg.get("val_every_epoch", 1)
        if epoch % val_every == 0 or epoch == num_epochs - 1:
            if val_dataloader is not None:
                val_result = trainer.validate_epoch(epoch, val_dataloader)
                if val_result.get("val_loss") is not None:
                    val_loss_val = val_result['val_loss']
                    logger.info(
                        f"Epoch {epoch}: val_loss={val_loss_val:.6f}"
                    )
                    # Always log per-loss val breakdown to the text log
                    _vb = val_result.get("val_loss_breakdown", {})
                    if _vb:
                        _vb_str = ", ".join(
                            f"{name}={val:.6f}" for name, val in _vb.items()
                        )
                        logger.info(f"Epoch {epoch}: val_loss_breakdown: {_vb_str}")
                    # Log val_loss + per-loss breakdown to active reporters
                    if trainer.callbacks.callbacks:
                        for cb in trainer.callbacks.callbacks:
                            # WandB
                            if hasattr(cb, '_wandb') and cb._wandb:
                                wandb_log = {"val_loss": val_loss_val, "epoch": epoch}
                                for name, val in val_result.get("val_loss_breakdown", {}).items():
                                    wandb_log[f"val_loss/{name}"] = val
                                cb._wandb.log(wandb_log)
                            # TensorBoard
                            if hasattr(cb, '_writer') and cb._writer:
                                cb._writer.add_scalar("val/loss", val_loss_val, trainer.step)
                                for name, val in val_result.get("val_loss_breakdown", {}).items():
                                    cb._writer.add_scalar(f"val/loss_{name}", val, trainer.step)

            # ── Generate validation images into per-epoch dir ──────────
            if val_cfg.get("generate_images", False):
                # Reload VAE on CPU for latent decoding (trainer handles GPU transfer)
                logger.info("Reloading VAE for validation image generation...")
                val_vae = adapter.load_vae(vae_path, weight_dtype).to("cpu")
                trainer.vae = val_vae

                trainer.epoch_output_dir = epoch_dir
                img_paths = trainer.generate_validation_images(epoch, val_dataloader)
                if img_paths:
                    logger.info(
                        f"Epoch {epoch}: generated {len(img_paths)} validation images"
                    )

                # Delete VAE after validation to free memory
                del val_vae
                trainer.vae = None
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("VAE deleted after validation")

    # Save final into root output dir
    from UnifiedTrainer.engine.checkpoint import CheckpointManager

    # P5：DDP 下仅 rank 0 落盘；wait_for_everyone 保证各 rank 训练齐步
    if trainer.accelerator is not None:
        trainer.accelerator.wait_for_everyone()
    final_path = None
    if _is_rank0():
        ckpt_mgr = CheckpointManager(output_dir, save_name)
        if network_type == "lokr" and lycoris_net is not None:
            final_path = ckpt_mgr.save_lokr(
                lycoris_net, trainer.step, num_epochs, config, is_final=True
            )
        else:
            final_path = ckpt_mgr.save_lora(
                trainer.transformer, trainer.step, num_epochs, config,
                is_final=True
            )
        # Save final training state too
        if trainer.optimizer is not None:
            extra = (
                {"noise_selector": trainer.noise_selector.state_dict()}
                if hasattr(trainer.noise_selector, "state_dict")
                else None
            )
            ckpt_mgr.save_training_state(
                optimizer=trainer.optimizer,
                lr_scheduler=trainer.lr_scheduler,
                step=trainer.step,
                epoch=num_epochs,
                epoch_idx=num_epochs,
                config=config,
                is_final=True,
                extra_state=extra,
            )
        # Report peak VRAM for scoring/monitoring
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.info(f"Peak VRAM: {peak_vram_mb:.0f} MB")
    if trainer.accelerator is not None:
        trainer.accelerator.wait_for_everyone()

    logger.info(f"Training complete. Final model: {final_path}")

    # DB mode: 记录产物路径，dispatcher 用它解析后继任务的 $task:<name>.output
    # P5：仅 rank 0 写 DB（多进程并发写同一 kv 无意义）
    if db_ctx is not None and _is_rank0():
        db_ctx.set_config_kv(args.task_id, "_meta.output", str(final_path))
        db_ctx.heartbeat(args.task_id, trainer.step)

    # Dispatch on_train_end (wandb.finish, etc.)
    trainer.callbacks.on_train_end(trainer)


def _dry_run(db_ctx, task_row, config, steps):
    """DB 模式冒烟测试：模拟 N 个 step、写心跳、产出假 output，退出码 0。

    不加载任何模型/数据，仅验证 orchestrator ↔ train.py 的 DB 通道。
    """
    import time

    task_id = task_row["id"]
    # P5：accelerate launch 下每个 rank 都会进入本函数；仅 rank 0
    # 写 DB/假产物，其余直接退出（launcher 会等齐所有进程）。
    if not _is_rank0():
        logger.info("[dry-run] non-rank-0 process, skip dry-run body")
        return 0
    logger.info(f"[dry-run] task {task_id} ({task_row['name']}): "
                f"simulating {steps} steps")

    # P2: dry-run 下也走完整 hook 流程（save/sample/suspend 均为占位实现），
    # suspend 会从 maybe_run 内 raise SystemExit(42) 终止本进程。
    from UnifiedTrainer.engine.hook_manager import HookManager
    hook_mgr = HookManager(db_ctx, task_id, config=config, dry_run=True)

    db_ctx.heartbeat(task_id, 0)
    for step in range(1, steps + 1):
        time.sleep(0.2)
        loss = 1.0 / step
        db_ctx.heartbeat(task_id, step, loss=loss, lr=1e-4, vram_mb=0)
        logger.info(f"[dry-run] step {step}/{steps} loss={loss:.4f}")
        hook_mgr.maybe_run(step, config=config)

    output_dir = config.get("output", {}).get("dir", "output")
    save_name = config.get("output", {}).get("save_name", "lora")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"{save_name}_dryrun.safetensors")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("DRY-RUN placeholder weights\n")
    final_path = os.path.abspath(final_path)
    db_ctx.set_config_kv(task_id, "_meta.output", final_path)
    # P3：记录绑卡信息（dispatcher 并行时注入 CUDA_VISIBLE_DEVICES）
    db_ctx.set_config_kv(task_id, "_meta.cuda_visible_devices",
                         os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    db_ctx.heartbeat(task_id, steps)
    logger.info(f"[dry-run] complete. fake output: {final_path}")
    return 0


def _import_all_adapters():
    """Import all model adapter modules to trigger registration."""
    import importlib
    import pkgutil
    try:
        import UnifiedTrainer.models as models_pkg
        for finder, name, ispkg in pkgutil.iter_modules(models_pkg.__path__):
            if name != "base" and name != "__init__":
                try:
                    importlib.import_module(f"UnifiedTrainer.models.{name}")
                except Exception as e:
                    logger.error(f"Failed to import model adapter '{name}': {e}")
    except Exception as e:
        logger.error(f"Failed to scan model adapters: {e}")


def _import_all_losses():
    """Import all loss modules to trigger registration."""
    import importlib
    import pkgutil
    try:
        import UnifiedTrainer.losses as losses_pkg
        for finder, name, ispkg in pkgutil.iter_modules(losses_pkg.__path__):
            if name != "base" and name != "__init__":
                try:
                    importlib.import_module(f"UnifiedTrainer.losses.{name}")
                except Exception as e:
                    logger.error(f"Failed to import loss '{name}': {e}")
    except Exception as e:
        logger.error(f"Failed to scan loss modules: {e}")


if __name__ == "__main__":
    main()
