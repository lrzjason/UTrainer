"""
Trainer - the shared, model-agnostic training engine.

The Trainer contains the 90% of code that is identical across all models:
accelerator setup, optimizer creation, LoRA configuration, checkpointing,
the training loop, and validation. All model-specific behavior is delegated
to the adapter and composable loss modules.

Target: ~300 lines instead of 3000+ per-model training scripts.
"""
from __future__ import annotations

import gc
import inspect
import logging
import math
import os
import random
from contextlib import nullcontext
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

from UnifiedTrainer.registry import ModelRegistry, LossRegistry
from UnifiedTrainer.losses.base import BaseLoss, LossContext
from UnifiedTrainer.data.config_schema import parse_dataset_configs
from UnifiedTrainer.engine.callbacks import CallbackList, Callback
from UnifiedTrainer.engine.noise_selector import NoiseSelector, build_noise_selector
from UnifiedTrainer.utils.helios import parse_helios_config, apply_helios_corrupt

logger = logging.getLogger(__name__)


class Trainer:
    """Unified training engine.

    Assembly:
        adapter = ModelRegistry.get(config["model"])(config)
        losses  = [LossRegistry.get(l["type"])(**l.get("params", {})) for l in config["losses"]]
        dataset = UnifiedDataset(adapter, config["data"])
        trainer = Trainer(config, adapter, losses)

    Loop (train_epoch):
        for batch in dataloader:
            context = adapter.prepare_model_input(batch, noise, sigmas)
            model_pred = transformer(**context)
            unpacked = adapter.unpack_prediction(model_pred)
            loss_ctx = LossContext(model_pred=unpacked, ...)
            total_loss = sum(loss(loss_ctx) for loss in self.losses)
            total_loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        config: dict,
        adapter: Any = None,
        losses: Optional[List[BaseLoss]] = None,
        transformer: Optional[nn.Module] = None,
        vae_manager: Optional[Any] = None,
        lycoris_net: Optional[nn.Module] = None,
    ):
        self.config = config
        self.adapter = adapter
        self.losses = losses or []
        self.transformer = transformer
        # 统一 VAE 组件：所有可能用到 VAE 的过程（pfm 解码、验证采样、
        # regenerator …）共用 vae_manager 持有的同一个实例。VAE 在训练循环
        # 开始前由 train.py 加载进 RAM；GPU 驻留由 training.vae_on_device
        # 决定（false=需要时转移，true=常驻）。
        self.vae_manager = vae_manager
        # 需要 VAE 的 loss（如 pfm）从 manager 借用同一实例。
        for _loss in self.losses:
            if getattr(_loss, "requires_vae", False):
                _loss.vae_manager = vae_manager
        # 生成验证图像期间的瞬时借用（generate_validation_images 内部
        # acquire/release），供深层 decode 方法读取；训练循环中始终为 None。
        self.vae = None
        self.lycoris_net = lycoris_net  # LyCORIS network (LoKR) or None (LoRA)

        # Training params
        training_cfg = config.get("training", config)
        self.learning_rate: float = training_cfg.get("learning_rate", 1e-4)
        self.batch_size: int = training_cfg.get("batch_size", 1)
        self.max_steps: int = training_cfg.get("max_steps", -1)  # -1 = unlimited
        self.gradient_accumulation_steps: int = training_cfg.get(
            "gradient_accumulation_steps", 1
        )
        self.gradient_checkpointing: bool = training_cfg.get(
            "gradient_checkpointing", True
        )
        self.lora_rank: int = training_cfg.get("lora_rank", 16)
        self.lora_alpha: int = training_cfg.get("lora_alpha", self.lora_rank)
        self.optimizer_name: str = training_cfg.get("optimizer", "adamw")
        self.max_grad_norm: float = training_cfg.get("max_grad_norm", 1.0)
        self.lr_scheduler_name: str = training_cfg.get("lr_scheduler", "constant")
        self.weight_dtype: torch.dtype = training_cfg.get(
            "weight_dtype", torch.bfloat16
        )
        self.mixed_precision: str = training_cfg.get("mixed_precision", "no")

        # Reference: adapted from ai-toolkit config_modules.py (compile_mode)
        # See aitoolkit_acc.md for full technique analysis.
        self.compile: bool = training_cfg.get("compile", False)
        self.compile_mode: str = training_cfg.get(
            "compile_mode", "max-autotune-no-cudagraphs"
        )
        self._compiled = False

        # Validation params
        val_cfg = config.get("validation", {})
        self.val_every_epoch: int = val_cfg.get("val_every_epoch", 1)
        self.val_seed: int = val_cfg.get("val_seed", 42)
        self.val_loss_enabled: bool = val_cfg.get("val_loss", False)
        self.gen_images: bool = val_cfg.get("generate_images", False)
        self.val_num_inference_steps: int = val_cfg.get(
            "num_inference_steps", 20
        )
        self.val_guidance_scale: float = val_cfg.get("guidance_scale", 1.0)
        self.val_resolution: int = val_cfg.get("resolution", 512)
        self.val_num_images: int = val_cfg.get("num_val_images", 4)
        self.noise_scheduler = None  # loaded lazily for validation generation
        self.val_output_dir: str = val_cfg.get(
            "output_dir",
            os.path.join(
                config.get("output", {}).get("dir", "output"), "validation"
            ),
        )
        self.epoch_output_dir: Optional[str] = None  # per-epoch subdir, set by train.py
        # Output save_name prefix (set by train.py); used to prefix validation
        # image filenames so they sort/group with the run's checkpoints.
        self.save_name: str = config.get("output", {}).get("save_name", "lora")

        # CFG: empty embedding for unconditional pass (lazy loaded)
        # Uses model-specific suffix to avoid collisions across models
        self._val_empty_embed: Optional[dict] = None
        cache_dir = config.get("data", {}).get("cache_dir", "")
        model_suffix = adapter.empty_embedding_suffix if adapter else "base"
        self._val_empty_embed_path: str = os.path.join(
            cache_dir, f"empty_embedding.{model_suffix}.npz"
        )

        # Caption dropout: same empty embedding, cached once for training loop reuse
        self._train_empty_embed: Optional[dict] = None

        # Suffix embedding — appended to per-sample encoder_hidden_states at
        # training time to match ComfyUI's full-template output length.
        # Controlled by ``training.include_suffix`` (default True).
        self._include_suffix: bool = training_cfg.get("include_suffix", True)
        self._suffix_embed_path: str = os.path.join(
            cache_dir, f"suffix_embedding.{model_suffix}.npz"
        )
        self._suffix_embed_loaded: bool = False

        # Helios Frame-Aware Corrupt config (perturbs reference latents during training)
        self.helios_config = parse_helios_config(config)
        if self.helios_config.get("enabled", False):
            logger.info(
                "Helios Frame-Aware Corrupt enabled — "
                "reference latents will be perturbed during training"
            )

        # State
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[Any] = None
        self.step: int = 0
        self.epoch: int = 0
        self.accelerator: Optional[Any] = None
        self.callbacks: CallbackList = CallbackList()

        # EMA (Exponential Moving Average) of LoRA weights
        self.ema_decay: float = training_cfg.get("ema_decay", 0.0)  # 0 = disabled
        self.ema = None  # EMAManager instance, created in setup_ema()

        # Per-loss breakdown for logging (updated every step, read by callbacks)
        self.last_loss_breakdown: dict = {}

        # Noise/timestep selection strategy (default: random; explorative: best-of-K)
        self.noise_selector: NoiseSelector = build_noise_selector(config)

    # ── Setup ─────────────────────────────────────────────────────────

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for trainable parameters.

        Supports both PEFT LoRA (params on transformer) and LyCORIS LoKR
        (params on lycoris_net). Also collects trainable parameters from
        loss modules that own auxiliary modules (e.g. LISA depth decoder)
        so they are jointly optimised.

        Reference: T2ITrainer train_flux2klein_lcs_v7.py lines 2002-2020.
        Uses bnb.optim.AdamW8bit with betas=(0.9, 0.99), weight_decay=1e-2.
        """
        if self.lycoris_net is not None:
            params = [p for p in self.lycoris_net.parameters() if p.requires_grad]
        else:
            params = [
                p for p in self.transformer.parameters() if p.requires_grad
            ]
        # Collect parameters from loss modules (e.g. LISA decoder)
        for loss_module in self.losses:
            loss_params = loss_module.parameters()
            if loss_params:
                params.extend(loss_params)
                logger.info(
                    f"Added {len(loss_params)} param tensor(s) from loss "
                    f"'{loss_module.name}' to optimizer"
                )
        # T2ITrainer defaults: betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8
        adam_betas = self.config.get("training", {}).get("adam_betas", (0.9, 0.999))
        adam_weight_decay = self.config.get("training", {}).get("adam_weight_decay", 1e-2)
        adam_epsilon = self.config.get("training", {}).get("adam_epsilon", 1e-8)

        if self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=self.learning_rate,
                betas=adam_betas, weight_decay=adam_weight_decay, eps=adam_epsilon,
            )
        elif self.optimizer_name == "adamw8bit":
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                params, lr=self.learning_rate,
                betas=adam_betas, weight_decay=adam_weight_decay, eps=adam_epsilon,
            )
        elif self.optimizer_name == "adam8bit":
            # Reference: adapted from ai-toolkit/toolkit/optimizers/adam8bit.py
            from UnifiedTrainer.utils.optimizers.adam8bit import Adam8bit
            self.optimizer = Adam8bit(
                params, lr=self.learning_rate, weight_decay=0.0
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params, lr=self.learning_rate
            )
        return self.optimizer

    def setup_loss_modules(self) -> None:
        """Move loss-owned modules (e.g. LISA decoder) to the model's device.

        Called after the transformer is loaded and moved to device, before
        the optimizer is created.
        """
        if not self.transformer or not self.losses:
            return
        # Use self.device (always CUDA) — after BouncingOffloader some params
        # are on CPU, so next(self.transformer.parameters()).device is unreliable.
        device = getattr(self, 'device', torch.device('cuda'))
        # Determine compute dtype (handle quantized models)
        dtype = next(self.transformer.parameters()).dtype
        if dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
            dtype = torch.bfloat16
        for loss_module in self.losses:
            loss_module.to(device, dtype)

    def setup_compile(self) -> None:
        """Wrap the transformer in torch.compile for graph fusion + kernel fusion.

        Config:
            training.compile: true   -enable torch.compile
            training.compile_mode: "max-autotune-no-cudagraphs"  -compile mode

        Reference: adapted from ai-toolkit config_modules.py compile_mode pattern.
        """
        if not self.compile or self._compiled:
            return
        if self.transformer is None:
            return
        self.transformer = torch.compile(
            self.transformer, mode=self.compile_mode
        )
        self._compiled = True

    def setup_lr_scheduler(self) -> Any:
        """Create learning rate scheduler.

        Supported schedulers:
          - constant               → flat LR
          - constant_with_warmup   → linear warmup then flat
          - cosine                 → linear warmup then cosine decay to 0
          - cosine_with_restarts   → warmup then periodic cosine restarts
          - linear                 → linear warmup then linear decay
        """
        from torch.optim.lr_scheduler import LambdaLR
        import math

        training_cfg = self.config.get("training", {})
        warmup_steps = training_cfg.get("lr_warmup_steps", 0)

        # Determine total steps for decay-based schedulers.
        # If max_steps is set (>0) use it, otherwise compute from epochs.
        if self.max_steps and self.max_steps > 0:
            total_steps = self.max_steps
        else:
            num_epochs = training_cfg.get("num_epochs", 10)
            # Estimate dataset length — will be refined after dataloader creation.
            # This is only used for scheduler shape, exact value isn't critical.
            total_steps = training_cfg.get("_estimated_total_steps", num_epochs * 1000)

        if self.lr_scheduler_name == "constant":
            self.lr_scheduler = LambdaLR(self.optimizer, lambda _: 1.0)

        elif self.lr_scheduler_name == "constant_with_warmup":
            def _warmup_fn(step):
                return min(1.0, step / max(1, warmup_steps)) if warmup_steps > 0 else 1.0
            self.lr_scheduler = LambdaLR(self.optimizer, _warmup_fn)

        elif self.lr_scheduler_name == "cosine":
            def _cosine_fn(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            self.lr_scheduler = LambdaLR(self.optimizer, _cosine_fn)

        elif self.lr_scheduler_name == "cosine_with_restarts":
            num_cycles = training_cfg.get("lr_num_cycles", 1)
            def _cosine_restarts_fn(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            self.lr_scheduler = LambdaLR(self.optimizer, _cosine_restarts_fn)

        elif self.lr_scheduler_name == "linear":
            def _linear_fn(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                return 1.0 - progress
            self.lr_scheduler = LambdaLR(self.optimizer, _linear_fn)

        else:
            logger.warning(
                f"Unknown lr_scheduler '{self.lr_scheduler_name}', defaulting to constant"
            )
            self.lr_scheduler = LambdaLR(self.optimizer, lambda _: 1.0)

        return self.lr_scheduler

    def setup_ema(self) -> None:
        """Initialize EMA shadow weights if ema_decay > 0 in config."""
        if self.ema_decay > 0 and self.transformer is not None:
            from UnifiedTrainer.engine.ema import EMAManager
            trainable = [p for p in self.transformer.parameters() if p.requires_grad]
            self.ema = EMAManager(trainable, decay=self.ema_decay)
            logger.info(
                f"EMA enabled: decay={self.ema_decay}, "
                f"tracking {len(trainable)} trainable params"
            )

    def setup_accelerator(self) -> None:
        """Create and configure the Accelerator for memory-efficient training.

        Reference: T2ITrainer creates Accelerator with mixed_precision and
        gradient_accumulation_steps, then calls accelerator.prepare() on the
        model, optimizer, and scheduler.

        Key VRAM optimizations enabled by Accelerator:
          - accelerator.autocast(): mixed-precision forward that reduces
            activation memory (internal matmuls/attention run in bf16/fp8
            instead of implicit fp32).
          - accelerator.accumulate(): efficient gradient accumulation context
            that defers gradient buffers on non-sync steps.
          - accelerator.backward(): memory-efficient backward pass.

        When block_swap > 0, the transformer is NOT passed to
        accelerator.prepare() because the wrapper interferes with custom
        block-swap forward hooks. Only optimizer and scheduler are prepared.
        """
        from accelerate import Accelerator

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )

        block_swap = self.config.get("training", {}).get("block_swap", 0)
        quantize_mode = self.config.get("training", {}).get("quantize", "none")
        _TORCHAO_MODES = {"torchao_float8", "torchao_int8", "torchao_int4"}

        # T2ITrainer: for torchao-quantized models, pass device_placement=[False]
        # to prevent accelerate from calling .to(device) on AffineQuantizedTensor
        # which can crash or create unintended copies. The model is already on
        # the correct device from quantization/PEFT setup above.
        if quantize_mode in _TORCHAO_MODES:
            self.transformer = self.accelerator.prepare(
                self.transformer, device_placement=[False]
            )
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer, self.lr_scheduler
            )
            logger.info(
                "Accelerator ready (torchao model: device_placement=[False], "
                "optimizer + scheduler prepared)"
            )
        elif block_swap > 0:
            # Don't prepare the model when using block swap — prepare() wraps
            # the model and interferes with the custom forward hooks used by
            # ModelOffloader.  Optimizer and scheduler are still prepared.
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer, self.lr_scheduler
            )
            logger.info(
                f"Accelerator ready "
                f"(model NOT prepared due to block_swap={block_swap})"
            )
        else:
            self.transformer, self.optimizer, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.transformer, self.optimizer, self.lr_scheduler
                )
            )
            logger.info(
                "Accelerator ready (model + optimizer + scheduler prepared)"
            )

        # Prepare LyCORIS network separately (owns adapter params for LoKR)
        if self.lycoris_net is not None:
            # DIAG: transformer param devices BEFORE lycoris prepare
            try:
                _g0 = sum(p.numel()*p.element_size() for p in self.transformer.parameters() if p.is_cuda)/1024**3
                _c0 = sum(p.numel()*p.element_size() for p in self.transformer.parameters() if not p.is_cuda)/1024**3
                logger.info(f"[DIAG-PREP] before lycoris prepare: transformer GPU={_g0:.2f}GB CPU={_c0:.2f}GB")
            except Exception as _e:
                logger.warning(f"[DIAG-PREP] before failed: {_e}")
            self.lycoris_net = self.accelerator.prepare(self.lycoris_net)
            logger.info("Accelerator: lycoris_net prepared")
            # DIAG: AFTER lycoris prepare
            try:
                _g1 = sum(p.numel()*p.element_size() for p in self.transformer.parameters() if p.is_cuda)/1024**3
                _c1 = sum(p.numel()*p.element_size() for p in self.transformer.parameters() if not p.is_cuda)/1024**3
                logger.info(f"[DIAG-PREP] after lycoris prepare: transformer GPU={_g1:.2f}GB CPU={_c1:.2f}GB")
            except Exception as _e:
                logger.warning(f"[DIAG-PREP] after failed: {_e}")

    def _get_base_model(self) -> nn.Module:
        """Unwrap PEFT / Accelerate layers to reach the raw diffusers model.

        This is needed for accessing block-swap methods and gradient-
        checkpointing diagnostics that live on the underlying model.
        """
        model = self.transformer
        # Unwrap Accelerate wrapper
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
        # Unwrap PEFT LoraModel wrapper
        if hasattr(model, "base_model"):
            model = model.base_model.model
        return model

    # ── Training Loop ─────────────────────────────────────────────────

    def _ensure_suffix_embed_loaded(self) -> None:
        """Load cached suffix embedding and set it on the adapter (once)."""
        if self._suffix_embed_loaded:
            return
        self._suffix_embed_loaded = True

        if not self._include_suffix:
            logger.info("Suffix embedding disabled (include_suffix=false) — skipping.")
            return

        if not os.path.isfile(self._suffix_embed_path):
            logger.warning(
                f"Suffix embedding not found at {self._suffix_embed_path}. "
                f"Rebuild cache to create it. Training will proceed without suffix."
            )
            return

        from UnifiedTrainer.data.embedding_cache import EmbeddingCache
        try:
            npz = EmbeddingCache.load(self._suffix_embed_path)
            suffix_pe = torch.from_numpy(npz["prompt_embed"])
            suffix_mask = torch.from_numpy(npz["prompt_embeds_mask"])
            if self.adapter is not None and hasattr(self.adapter, "suffix_embed"):
                self.adapter.suffix_embed = suffix_pe
                self.adapter.suffix_mask = suffix_mask
                logger.info(
                    f"Suffix embedding loaded: {suffix_pe.shape[0]} tokens, "
                    f"{suffix_pe.shape[1]} layers, dim={suffix_pe.shape[2]}"
                )
        except Exception as e:
            logger.warning(f"Failed to load suffix embedding: {e}")

    def train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        skip_steps: int = 0,
    ) -> dict:
        """Run one epoch of training.

        Args:
            epoch: epoch number (0-indexed)
            dataloader: training data loader
            skip_steps: number of steps to skip (for resume)

        Returns:
            dict with 'loss', 'steps', 'epoch'
        """
        self.epoch = epoch
        self.transformer.train()
        self.callbacks.on_epoch_start(epoch, self)

        # Lazy-load suffix embedding once on first training step.
        self._ensure_suffix_embed_loaded()

        epoch_loss = 0.0
        steps_this_epoch = 0
        skipped = 0

        # Track previous batch shape for fragmentation-aware cleanup.
        # When per-dataset batch sizing alternates between very different
        # shapes (e.g. [4,16,64,64] → [1,16,168,168]), the CUDA caching
        # allocator holds fragmented blocks that can't satisfy the new
        # allocation, causing OOM even when total free VRAM is sufficient.
        _prev_batch_shape = None

        num_batches = len(dataloader)

        # Effective (optimizer) steps per epoch, accounting for gradient accumulation.
        # The progress bar tracks global optimizer steps, not micro-batches.
        ga_steps = self.gradient_accumulation_steps if self.accelerator is not None else 1
        effective_batches = max(1, num_batches // ga_steps) if ga_steps > 1 else num_batches

        # Build progress description showing global step context
        desc = f"Epoch {epoch}"
        if self.max_steps != -1:
            desc += f" [{self.step}/{self.max_steps}]"

        # Progress bar total: remaining global steps (capped at this epoch's effective steps)
        if self.max_steps != -1:
            remaining = self.max_steps - self.step
            total = min(effective_batches, max(remaining, 1))
        else:
            total = effective_batches

        pbar = tqdm(total=total, desc=desc, leave=True)

        for batch in dataloader:
            # Skip steps for resume
            if skipped < skip_steps:
                skipped += 1
                continue

            if self.max_steps != -1 and self.step >= self.max_steps:
                break

            import time as _time
            _t0 = _time.perf_counter()
            _step_group_ids = batch.get("group_ids", ["?"])

            # ── Resolve batch_config for this sample ──────────────────
            batch_configs = batch.get("batch_configs", [])
            resolved_bc = batch_configs[0] if batch_configs else None

            # ── Move latents to device and cast to compute dtype ──────────
            # When the model is quantized (int8/nf4), parameter dtype is int8/fp4
            # (storage dtype), NOT the compute dtype.  Latents must be cast to
            # the compute dtype (bf16) to avoid "normal_kernel_cuda not
            # implemented for 'Char'" errors when calling torch.randn_like.
            # Reference: T2ITrainer uses accelerator.mixed_precision for the
            # compute dtype, never inspects model parameter dtype.
            # Use self.device (always CUDA) — after BouncingOffloader some params
            # are on CPU, so next(self.transformer.parameters()).device is unreliable.
            device = self.device
            _first_param = next(self.transformer.parameters())
            model_dtype = _first_param.dtype
            # Quantized params report int8/uint8 — use bf16 for latents instead
            if model_dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn,
                               torch.float8_e5m2):
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = model_dtype
            latents = batch.get("latents", {})
            latents = self._move_latents_to_device(latents, device, compute_dtype)
            batch["latents"] = latents
            # Edit mask (optional) rides along with the latents: same
            # device, same compute dtype — it multiplies the velocity MSE
            # in masked_flow_matching.
            if batch.get("loss_mask") is not None:
                batch["loss_mask"] = batch["loss_mask"].to(
                    device=device, dtype=compute_dtype
                )

            # ── Get ALL target latents (multi-target) ──
            target_latents = self._get_target_latents(batch, latents, resolved_bc)
            # Safety: ensure all target latents are floating-point dtype.
            for i, tl in enumerate(target_latents):
                if not torch.is_floating_point(tl):
                    target_latents[i] = tl.to(dtype=compute_dtype)

            # ── Helios Frame-Aware Corrupt ───────────────────────────
            # Perturb reference latents (everything except ALL targets).
            # Target latents stay clean — only references are corrupted.
            if self.helios_config.get("enabled", False):
                target_keys = set(self._resolve_target_keys(resolved_bc) or [])
                if not target_keys and resolved_bc:
                    # Fallback: use the group key directly.
                    target_keys.add(resolved_bc.get("target_config", ""))

                for lk, lv in latents.items():
                    if lk in target_keys:
                        continue
                    if isinstance(lv, torch.Tensor):
                        # Single-reference role — corrupt the tensor.
                        latents[lk] = apply_helios_corrupt(lv, self.helios_config)
                    elif isinstance(lv, list):
                        # Multi-reference role (e.g. D+F1+F2) — corrupt EVERY
                        # reference latent, not just the first frame.
                        latents[lk] = [
                            apply_helios_corrupt(t, self.helios_config)
                            if isinstance(t, torch.Tensor) else t
                            for t in lv
                        ]
                batch["latents"] = latents

            # ── Dynamic caption dropout (T2ITrainer-style: per-step, on-the-fly) ──
            self._apply_caption_dropout(batch, resolved_bc)

            # ── Noise/timestep selection (NoiseSelector strategy) ────────
            # Default: random sampling (identical to original behavior).
            # Explorative: best-of-K noise exploration with no_grad forwards.
            # The selector handles sigma sampling internally; exploration
            # forwards use pre_forward_fn for block_swap preparation.
            quantize_mode = self.config.get("training", {}).get("quantize", "none")
            _TORCHAO_MODES = {"torchao_float8", "torchao_int8", "torchao_int4"}
            base_model = self._get_base_model()

            def _pre_forward():
                if quantize_mode not in _TORCHAO_MODES:
                    if hasattr(base_model, "move_to_device_except_swap_blocks"):
                        base_model.move_to_device_except_swap_blocks(device)
                    if hasattr(base_model, "prepare_block_swap_before_forward"):
                        base_model.prepare_block_swap_before_forward()

            noises, sigmas, timesteps = self.noise_selector.select(
                batch, target_latents, self.adapter, self.transformer,
                self.step, pre_forward_fn=_pre_forward,
            )
            sigmas_b = sigmas.view(-1, *(1,) * (target_latents[0].ndim - 1)) if target_latents[0].ndim >= 2 else sigmas

            # ── DIAG: after noise-selector exploration (K forwards) ──
            try:
                if not getattr(self, "_diag_after_xm", False) and torch.cuda.is_available():
                    self._diag_after_xm = True
                    _alloc = torch.cuda.memory_allocated() / 1024**3
                    _resv = torch.cuda.memory_reserved() / 1024**3
                    _peak = torch.cuda.max_memory_allocated() / 1024**3
                    # base params on GPU?
                    _gpu = sum(p.numel() * p.element_size() for p in self.transformer.parameters() if p.is_cuda) / 1024**3
                    _cpu = sum(p.numel() * p.element_size() for p in self.transformer.parameters() if not p.is_cuda) / 1024**3
                    logger.info(
                        f"[DIAG-AFTER-XM] alloc={_alloc:.2f}G resv={_resv:.2f}G peak={_peak:.2f}G "
                        f"transformer GPU={_gpu:.2f}GB CPU={_cpu:.2f}GB"
                    )
            except Exception as _e2:
                logger.warning(f"[DIAG-AFTER-XM] failed: {_e2}")

            # ── Flow matching interpolation ──────────────────────────
            # One noisy latent per target.
            noisy_latents = [
                (1.0 - sigmas_b) * tl + sigmas_b * n
                for tl, n in zip(target_latents, noises)
            ]
            num_targets = len(noisy_latents)

            _t1 = _time.perf_counter()

            # ── Prepare block swap before training forward ─────────────
            _pre_forward()

            model_input = self.adapter.prepare_model_input(
                batch, noisy_latents, sigmas
            )

            # ── Gradient checkpointing engagement ──────────────────────
            # PEFT's add_adapter() already calls enable_input_require_grads()
            # on the base model, and train.py registers forward hooks on
            # img_in/txt_in to force their outputs to require grad. So
            # torch.utils.checkpoint engages automatically — no need to
            # force requires_grad on hidden_states here.
            # T2ITrainer doesn't do this either; PEFT handles it internally.

            _t2 = _time.perf_counter()

            # ── Training step with accelerator contexts ─────────────────
            # accelerator.accumulate() handles gradient accumulation timing.
            #
            # autocast: Only use when model is NOT already in bf16. When the
            # model is loaded in bf16, autocast(bf16) on top is redundant and
            # causes dtype mismatches in LayerNorm ("input dtype = float,
            # weight dtype = BFloat16") which forces the slow non-fused path.
            # With int8 quantization, the compute happens in bf16 already.
            # Reference: T2ITrainer uses autocast but only for fp16/fp8 mixed
            # precision, not bf16-on-bf16.
            accum_ctx = (
                self.accelerator.accumulate(self.transformer)
                if self.accelerator is not None
                else nullcontext()
            )
            # Skip autocast when mixed_precision is bf16 — the model is already
            # bf16, so autocast adds overhead and causes layer_norm dtype warnings.
            # EXCEPTION — block_swap + nf4/int8: trainer skips accelerator.prepare()
            # for the model in that mode (ModelOffloader hook conflict), so NOTHING
            # wraps the forward in autocast. The quantized base keeps fp32 stragglers
            # (norms/embedders), which then reach SDPA in fp32 and die with
            # "No available kernel" (cuDNN needs half/bfloat16). Re-enable autocast
            # for exactly that path; bf16 matmuls under autocast match what the
            # prepare()-wrapped (non-block-swap) runs already did.
            _model_prepared = not (
                quantize_mode in ("nf4", "int8")
                and self.config.get("training", {}).get("block_swap", 0) > 0
            )
            use_autocast = (
                self.accelerator is not None
                and self.mixed_precision not in ("no", None)
                and (
                    self.mixed_precision != "bf16"
                    or not _model_prepared
                )
            )
            autocast_ctx = (
                self.accelerator.autocast()
                if use_autocast
                else nullcontext()
            )

            with accum_ctx:
                # ── Forward pass ────────────────────────────────────────
                with autocast_ctx:
                    model_pred = self.transformer(**model_input)
                unpacked = self.adapter.unpack_prediction(model_pred)

                # ── Unconditional forward (guidance-corrected losses) ──
                # Losses with needs_uncond_forward=True (guide_flow_matching —
                # the DC-Gen corrected objective, Eq. 10) need the distilled
                # model's own output under the empty prompt:
                # v_eta(z_t, c_hat, t).  Same noisy latents, same sigmas —
                # only the text embedding is swapped for the cached empty one
                # (adapter contract: no image slots -> reference latents drop
                # out, matching the CFG uncond branch).  Gradient flows
                # through BOTH forwards — both are the trained network.
                # Caption dropout already made the conditional pass
                # unconditional when it fired (identical empty embedding), so
                # reuse that prediction instead of paying a second forward.
                uncond_unpacked = None
                if any(
                    getattr(l, "needs_uncond_forward", False)
                    for l in self.losses
                ):
                    if batch.get("_caption_dropped"):
                        uncond_unpacked = unpacked
                    else:
                        uncond_batch = self._build_uncond_batch(
                            batch, device, compute_dtype
                        )
                        model_input_uncond = self.adapter.prepare_model_input(
                            uncond_batch, noisy_latents, sigmas
                        )
                        with autocast_ctx:
                            model_pred_uncond = self.transformer(
                                **model_input_uncond
                            )
                        uncond_unpacked = self.adapter.unpack_prediction(
                            model_pred_uncond
                        )
                        del model_pred_uncond
                    if uncond_unpacked is not None and len(
                        uncond_unpacked
                    ) != len(unpacked):
                        raise RuntimeError(
                            f"guide_flow_matching: uncond prediction count "
                            f"({len(uncond_unpacked)}) != cond prediction "
                            f"count ({len(unpacked)}) — adapter contract "
                            "violation in unpack_prediction."
                        )

                _t3 = _time.perf_counter()

                # ── Build unified loss across ALL targets ──────────
                # Iterate per-target velocity predictions; sum losses then
                # normalise by num_targets so the scale is independent of
                # how many targets are configured.
                total_loss = torch.tensor(
                    0.0, device=target_latents[0].device,
                    dtype=target_latents[0].dtype,
                )
                loss_breakdown = {}

                # Extract reference latent (shared across targets) for loss modules
                reference_latent = self._get_reference_latent(
                    batch, latents, resolved_bc
                )

                # PFM original-pixel target features, keyed by target role:
                # batch["pfm_feats"][key] is (B, L, C, h, w) fp16 (built by
                # the dataset from the cache-build-time φ(I_orig) files).
                pfm_feats_by_key = batch.get("pfm_feats") or {}
                _target_keys_for_feats = list(
                    self._resolve_target_keys(resolved_bc) or []
                )

                for i, (noise_i, unpacked_i, target_i) in enumerate(
                    zip(noises, unpacked, target_latents)
                ):
                    # 按 velocity_sign 分发估算干净 latent（与 losses/flow_matching.py
                    # 同源：standard v = noise - x0 → x0_hat = noise - v；
                    # data_ward v = x0 - noise → x0_hat = noise + v）
                    x0_hat = self.adapter.compute_x0_hat(noise_i, unpacked_i)

                    # Resolve this target's φ(I_orig) features (if any):
                    # preferred = explicit target key; fallback = single-entry
                    # dict (single-target configs).
                    feats_i = None
                    if pfm_feats_by_key:
                        _fk = (
                            _target_keys_for_feats[i]
                            if i < len(_target_keys_for_feats)
                            and _target_keys_for_feats[i] in pfm_feats_by_key
                            else (
                                next(iter(pfm_feats_by_key))
                                if len(pfm_feats_by_key) == 1
                                else None
                            )
                        )
                        if _fk is not None:
                            _fb = pfm_feats_by_key[_fk]
                            if isinstance(_fb, dict):
                                # Precached multi-entry layout: per-key stacked
                                # (B, ...) tensors -> this sample's tensors
                                # (feat_{e}: (L, C, h, w); pix_{e}: (C, h, w)).
                                feats_i = {
                                    k: (v[i] if v.shape[0] > 1 else v[0])
                                    for k, v in _fb.items()
                                }
                            else:
                                _feats = _fb[i] if _fb.shape[0] > 1 else _fb[0]
                                # Dataset features carry the builder's singleton
                                # batch slot: (L, 1, C, h, w) -> (L, C, h, w).
                                feats_i = _feats[:, 0] if _feats.dim() == 5 else _feats

                    loss_ctx = LossContext(
                        model_pred=unpacked_i,
                        noise=noise_i,
                        sigmas=sigmas,
                        learning_target=target_i,
                        x0_hat=x0_hat,
                        reference_latent=reference_latent,
                        loss_mask=batch.get("loss_mask"),
                        adapter=self.adapter,
                        model_pred_uncond=(
                            uncond_unpacked[i]
                            if uncond_unpacked is not None
                            and i < len(uncond_unpacked)
                            else None
                        ),
                        extra=(
                            {"pfm_feats": feats_i} if feats_i is not None else {}
                        ),
                    )

                    for loss_module in self.losses:
                        loss_val = loss_module(loss_ctx)
                        total_loss = total_loss + loss_val
                        loss_breakdown[loss_module.name] = (
                            loss_breakdown.get(loss_module.name, 0.0)
                            + loss_val.item()
                        )
                        # Component telemetry (e.g. pfm main/tone/freq):
                        # merged into the SAME dict so components get the
                        # identical /num_targets normalisation and reach
                        # wandb as loss/<module>/<component>.
                        for _ck, _cv in getattr(
                            loss_module, "last_components", {}
                        ).items():
                            _bk = f"{loss_module.name}/{_ck}"
                            loss_breakdown[_bk] = (
                                loss_breakdown.get(_bk, 0.0) + _cv
                            )

                # Normalise so loss magnitude is independent of target count.
                for k in loss_breakdown:
                    loss_breakdown[k] /= num_targets
                total_loss = total_loss / num_targets

                self.last_loss_breakdown = loss_breakdown

                # Merge explorative noise selector stats (xm/*) for callbacks
                if hasattr(self.noise_selector, "last_stats"):
                    self.last_loss_breakdown.update(self.noise_selector.last_stats)

                # ── Backward ───────────────────────────────────────────
                # Non-finite loss guard: a single bad step (e.g. perceptual
                # losses exploding at high sigma) would otherwise backprop
                # NaN/Inf into the weights — and once the optimizer state is
                # poisoned, EVERY later step is NaN. Skip backward + update
                # but keep the normal per-step bookkeeping; max_grad_norm
                # cannot save this (clip(nan) == nan).
                nonfinite_loss = not torch.isfinite(total_loss)
                # Circuit breaker (two triggers):
                #   1. 20 consecutive non-finite losses — hard divergence.
                #   2. >=10 non-finite within the last 25 steps — chronic
                #      instability. (A purely consecutive counter is too
                #      weak: an objective that NaNs every other sample
                #      resets it forever while producing nothing.)
                # Either way, abort instead of grinding NaN for hours.
                if nonfinite_loss:
                    self._consec_nonfinite = (
                        getattr(self, "_consec_nonfinite", 0) + 1
                    )
                    if not hasattr(self, "_nonfinite_hist"):
                        from collections import deque as _dq

                        self._nonfinite_hist = _dq(maxlen=25)
                    self._nonfinite_hist.append(1)
                else:
                    self._consec_nonfinite = 0
                    if hasattr(self, "_nonfinite_hist"):
                        self._nonfinite_hist.append(0)
                _nan_rate = (
                    sum(self._nonfinite_hist)
                    if hasattr(self, "_nonfinite_hist")
                    else 0
                )
                if self._consec_nonfinite >= 20 or _nan_rate >= 10:
                    raise RuntimeError(
                        f"step {self.step}: non-finite losses chronic "
                        f"(consecutive={self._consec_nonfinite}, "
                        f"rate={_nan_rate}/25) — training is diverged, "
                        "aborting"
                    )
                if nonfinite_loss:
                    logger.warning(
                        f"step {self.step}: non-finite loss "
                        f"({loss_breakdown}) — skipping optimizer step"
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    # Hygiene: the failed forward peaked the allocator's
                    # reserved pool (inf activations). On Windows WDDM a
                    # bloated reserve is never handed back and every later
                    # step stalls for minutes — release it immediately.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if not nonfinite_loss:
                    if self.accelerator is not None:
                        self.accelerator.backward(total_loss)
                    else:
                        total_loss.backward()

                _t4 = _time.perf_counter()

                step_loss_val = total_loss.item()

                # Save diagnostic info before deleting tensors
                _step_latent_shape = [list(tl.shape) for tl in target_latents]

                # ── Clean up intermediate tensors ──────────────────────
                # Match T2ITrainer's aggressive deletion pattern
                # (train_krea2_edit.py L2423-2433 — deletes 15+ tensors
                # inside train_process() BEFORE returning loss). Without
                # this, VRAM pressure from dangling tensor refs causes
                # CUDA allocator fragmentation, making backward explode
                # from 8s → 235s in just one step without cleanup.
                model_input.clear()
                del model_pred, unpacked, noisy_latents, noises, target_latents
                del total_loss, model_input, loss_ctx, x0_hat, reference_latent
                # Also clear any tensors referenced by the adapter
                if hasattr(self.adapter, '_target_grids'):
                    self.adapter._target_grids = None

                # ── Gradient finiteness guard ──────────────────────────
                # A finite loss can still backprop inf/nan grads in bf16
                # (raw-L2 perceptual losses overflow easily through the VAE
                # decode graph). clip_grad_norm_ with a non-finite total norm
                # computes a nan scale factor that poisons EVERY grad, and
                # optimizer.step() then applies them — killing the weights
                # permanently (all later losses nan). Check before clip.
                if not nonfinite_loss:
                    _grad_params = [
                        p
                        for p in self.transformer.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    if _grad_params:
                        _total_norm = torch.norm(
                            torch.stack(
                                [p.grad.detach().float().norm(2) for p in _grad_params]
                            ),
                            2,
                        )
                        if not torch.isfinite(_total_norm):
                            logger.warning(
                                f"step {self.step}: non-finite grad norm "
                                f"({_total_norm.item()}) — skipping optimizer step"
                            )
                            nonfinite_loss = True  # reuse the skip path below
                            self.optimizer.zero_grad(set_to_none=True)

                # ── Gradient clipping + optimizer step ─────────────────
                if nonfinite_loss:
                    pass  # skip update; bookkeeping below still runs
                elif self.accelerator is not None:
                    # With accelerator, step/zero_grad are called every
                    # iteration but accumulate() makes them no-ops on
                    # non-sync steps.
                    if self.accelerator.sync_gradients and self.max_grad_norm > 0:
                        params_to_clip = [
                            p for p in self.transformer.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        if params_to_clip:
                            self.accelerator.clip_grad_norm_(
                                params_to_clip, self.max_grad_norm
                            )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    # set_to_none=True frees gradient tensors instead of
                    # zeroing them, reducing memory pressure.
                    self.optimizer.zero_grad(set_to_none=True)
                    # EMA update (after optimizer step)
                    if self.ema is not None:
                        self.ema.update()
                else:
                    # Manual gradient accumulation (fallback when no accelerator)
                    if (self.step + 1) % self.gradient_accumulation_steps == 0:
                        if hasattr(self.optimizer, "step_hook"):
                            self.optimizer.step_hook()
                        if self.max_grad_norm > 0:
                            params_to_clip = [
                                p for p in self.transformer.parameters()
                                if p.requires_grad and p.grad is not None
                            ]
                            if params_to_clip:
                                torch.nn.utils.clip_grad_norm_(
                                    params_to_clip, self.max_grad_norm
                                )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        # EMA update (after optimizer step)
                        if self.ema is not None:
                            self.ema.update()

            _t5 = _time.perf_counter()

            # ── Memory cleanup ─────────────────────────────────────────
            # When batch shapes vary (per-dataset batch sizing), the CUDA
            # caching allocator fragments: blocks cached from a batch of 4
            # at 512px can't satisfy a batch of 1 at 1536px. Detect shape
            # transitions and flush the allocator to eliminate fragmentation.
            #
            # Cost: ~0.05s for empty_cache() — negligible vs 5-15s/step.
            batch_changed = (
                _prev_batch_shape is not None
                and _step_latent_shape is not None
                and _step_latent_shape != _prev_batch_shape
            )
            if batch_changed or steps_this_epoch % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _prev_batch_shape = (
                list(_step_latent_shape) if _step_latent_shape is not None else None
            )

            # ── Explicitly clear batch GPU tensors ─────────────────────
            # The ``batch`` dict holds GPU latents and other tensors that
            # were moved to device during this step. Delete it now so the
            # memory is freed before the next (differently-shaped) batch.
            batch.clear()

            _t6 = _time.perf_counter()

            # ── Per-step timing log (identify slow samples) ────────────
            _total = _t6 - _t0
            _data_prep = _t1 - _t0
            _block_swap = _t2 - _t1
            _forward = _t3 - _t2
            _backward = _t4 - _t3
            _optim = _t5 - _t4
            _cleanup = _t6 - _t5
            _latent_shape = _step_latent_shape
            # Log slow steps (>15s) or every 50th step
            if _total > 15.0 or steps_this_epoch % 50 == 0:
                _vram_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                _vram_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                logger.info(
                    f"STEP_TIMING step={self.step} total={_total:.1f}s "
                    f"data={_data_prep:.1f}s swap={_block_swap:.1f}s "
                    f"fwd={_forward:.1f}s bwd={_backward:.1f}s "
                    f"optim={_optim:.1f}s cleanup={_cleanup:.1f}s "
                    f"shape={list(_latent_shape)} "
                    f"groups={_step_group_ids} "
                    f"vram={_vram_alloc:.1f}G_alloc/{_vram_reserved:.1f}G_reserved"
                )
                # ── DIAG: one-shot CUDA memory breakdown (first logged step) ──
                if not getattr(self, "_diag_dumped", False) and torch.cuda.is_available():
                    try:
                        self._diag_dumped = True
                        st = torch.cuda.memory_stats()
                        logger.info(
                            f"[DIAG-CUDA] alloc={_vram_alloc:.1f}G reserved={_vram_reserved:.1f}G "
                            f"segments={st.get('num_allocated_segments', '?')} "
                            f"reserved_blocks={st.get('number_of_cuda_malloc_calls', '?')} "
                            f"large_4M={st.get('large_allocated_blocks', '?')}"
                        )
                        # param device 统计（关键：base weights 在 GPU 还是 CPU）
                        gpu_bytes = 0; cpu_bytes = 0
                        for m_name, m in [("transformer", self.transformer)]:
                            for p in m.parameters():
                                if p.is_cuda:
                                    gpu_bytes += p.numel() * p.element_size()
                                else:
                                    cpu_bytes += p.numel() * p.element_size()
                        lyr_gpu = 0; lyr_cpu = 0
                        if self.lycoris_net is not None:
                            for p in self.lycoris_net.parameters():
                                if p.is_cuda:
                                    lyr_gpu += p.numel() * p.element_size()
                                else:
                                    lyr_cpu += p.numel() * p.element_size()
                        logger.info(
                            f"[DIAG-PARAM] transformer GPU={gpu_bytes/1024**3:.2f}GB CPU={cpu_bytes/1024**3:.2f}GB | "
                            f"lycoris GPU={lyr_gpu/1024**3:.2f}GB CPU={lyr_cpu/1024**3:.2f}GB"
                        )
                    except Exception as _de:
                        logger.warning(f"[DIAG] failed: {_de}")

            # Step counting: with accelerator, only increment on sync.
            # Progress bar advances only on effective (optimizer) steps.
            step_advanced = False
            if self.accelerator is not None:
                if self.accelerator.sync_gradients:
                    self.step += 1
                    step_advanced = True
            else:
                self.step += 1
                step_advanced = True

            if step_advanced:
                pbar.update(1)

                # ── ScheduledTrainer hook mount（P2，KI-01 修复）────────
                # 仅在 optimizer 真正 step（step_advanced）后执行 hook：
                # 梯度累积时中间的 micro-batch 不触发，避免重复采样/重复
                # 写 kv/suspend 时机错乱。仅 DB 模式由 train.py 注入
                # self.hook_manager；旧 --config 模式下该属性不存在，
                # getattr 返回 None 跳过，行为与 P2 前一致。
                # suspend hook 会从这里 raise SystemExit(42) 终止进程。
                hook_manager = getattr(self, "hook_manager", None)
                if hook_manager is not None:
                    hook_manager.maybe_run(self.step, trainer=self)

            epoch_loss += step_loss_val
            steps_this_epoch += 1

            lr = self.lr_scheduler.get_last_lr()[0]
            step_str = f"{self.step}/{self.max_steps}" if self.max_steps != -1 else f"{self.step}"
            # Build postfix with total + per-loss breakdown
            postfix = {
                "step": step_str,
                "loss": f"{step_loss_val:.4f}",
                "lr": f"{lr:.2e}",
                "s/step": f"{_total:.1f}",
            }
            for lname, lval in self.last_loss_breakdown.items():
                # xm/* exploration stats are noise for the bar — they are
                # shown by the [XM-IMP] line and logged to wandb instead.
                if lname.startswith("xm/"):
                    continue
                postfix[lname] = f"{lval:.4f}"
            pbar.set_postfix(postfix, refresh=False)

            # ── Per-step Explorative NoiseSelector log ─────────────────
            # Dedicated console line for XM exploration (K, min/max/mean
            # loss, gap, winning candidate). Only emitted when a selector
            # produced a log line this step (RandomNoiseSelector has none).
            _xm_log = getattr(self.noise_selector, "last_log_line", "")
            if _xm_log:
                logger.info(_xm_log)

            # ── Dispatch callback (wandb logging, etc.) ──────────────
            self.callbacks.on_step_end(self.step, step_loss_val, self)

        pbar.close()

        avg_loss = epoch_loss / max(1, steps_this_epoch)
        self.callbacks.on_epoch_end(epoch, avg_loss, self)
        return {
            "loss": avg_loss,
            "steps": steps_this_epoch,
            "epoch": epoch,
        }

    # ── Validation ────────────────────────────────────────────────────

    def _forward_autocast_ctx(self):
        """Autocast context for transformer forwards OUTSIDE the training step
        (validation loss, validation image sampling).

        Mirrors the train-step decision (see train_epoch): on the
        block_swap + nf4/int8 path the model is NOT passed through
        accelerator.prepare(), so nothing else wraps forwards in autocast and
        fp32 stragglers (norms/embedders) reach SDPA in fp32 — cuDNN rejects
        fp32 with "No available kernel". Elsewhere, prepare() (or the plain
        bf16-cast model) already handles dtype, so stay in nullcontext.
        """
        quantize_mode = self.config.get("training", {}).get("quantize", "none")
        if (
            self.accelerator is not None
            and quantize_mode in ("nf4", "int8")
            and self.config.get("training", {}).get("block_swap", 0) > 0
            and self.mixed_precision not in ("no", None)
        ):
            return self.accelerator.autocast()
        return nullcontext()

    def _resolve_val_loss_modules(self):
        """Loss modules that compute val_loss.

        Defaults to the training losses themselves. `validation.loss_types`
        (e.g. ["flow_matching"]) overrides — a pfm-trained run reports its
        val metric in fm-MSE space so the number is directly comparable with
        the fm baseline run. Freshly built modules receive
        `validation.loss_params` (e.g. {"use_weighting": false} to match the
        baseline config's flow_matching params).
        """
        if getattr(self, "_val_loss_modules", None) is not None:
            return self._val_loss_modules
        modules = list(self.losses)
        loss_types = self.config.get("validation", {}).get("loss_types")
        if loss_types:
            loss_params = (
                self.config.get("validation", {}).get("loss_params", {}) or {}
            )
            modules = []
            for t in loss_types:
                existing = next(
                    (
                        l
                        for l in self.losses
                        if getattr(l, "name", "") == t
                    ),
                    None,
                )
                if existing is not None:
                    modules.append(existing)
                    continue
                module = LossRegistry.get(t)(weight=1.0, **loss_params)
                if hasattr(module, "to"):
                    # pfm-style losses build their encoder lazily inside
                    # .to(device, dtype) — must happen before the first use.
                    module.to(self.device, torch.bfloat16)
                if getattr(module, "requires_vae", False):
                    module.vae_manager = self.vae_manager
                modules.append(module)
            logger.info(
                "val_loss modules overridden by validation.loss_types: "
                f"{[getattr(m, 'name', type(m).__name__) for m in modules]}"
            )
        self._val_loss_modules = modules
        return modules

    def validate_epoch(
        self,
        epoch: int,
        val_dataloader: DataLoader,
    ) -> dict:
        """Run validation loss computation with controlled RNG and no gradient.

        Follows the T2ITrainer pattern:
          1. Save RNG state before validation
          2. Set deterministic seed (val_seed)
          3. Run forward passes under torch.no_grad() -no backward, no optimizer step
          4. Compute average validation loss
          5. Restore RNG state after validation

        Args:
            epoch: current epoch number
            val_dataloader: validation data loader

        Returns:
            dict with 'val_loss', 'epoch'
        """
        if not self.val_loss_enabled or val_dataloader is None:
            return {"val_loss": None, "epoch": epoch}

        # Save RNG state (T2ITrainer pattern: before_state + np_seed)
        before_state = torch.random.get_rng_state()
        np_seed = random.randint(0, 2**31 - 1)
        import numpy as np
        np.random.seed(np_seed)  # store current numpy state

        # Freeze RNG with val_seed for reproducible validation
        torch.manual_seed(self.val_seed)
        np.random.seed(self.val_seed)
        torch.backends.cudnn.deterministic = True

        logger.info(f"\n--- Start val_loss (epoch {epoch}, seed={self.val_seed}) ---")

        # Swap in EMA weights for validation (smoother evaluation)
        if self.ema is not None:
            self.ema.apply()

        self.transformer.eval()
        total_loss = 0.0
        num_batches = 0
        # Accumulate per-loss breakdown ACROSS batches (each batch contributes
        # its per-target-averaged value); divided by num_batches at the end so
        # the breakdown is a true mean and sums to avg_val_loss.
        val_breakdown = {}
        val_loss_modules = self._resolve_val_loss_modules()

        try:
            with torch.no_grad():
                pbar = tqdm(
                    val_dataloader,
                    desc=f"Val (epoch {epoch})",
                    leave=False,
                )
                for batch in pbar:
                    # Resolve batch_config
                    batch_configs = batch.get("batch_configs", [])
                    resolved_bc = batch_configs[0] if batch_configs else None

                    # Move latents to device (use self.device — reliable after BouncingOffloader)
                    device = self.device
                    latents = batch.get("latents", {})
                    latents = self._move_latents_to_device(latents, device)
                    batch["latents"] = latents
                    if batch.get("loss_mask") is not None:
                        batch["loss_mask"] = batch["loss_mask"].to(device=device)

                    target_latents = self._get_target_latents(
                        batch, latents, resolved_bc
                    )

                    # Deterministic noise per target
                    gen = torch.Generator(device=target_latents[0].device)
                    gen.manual_seed(self.val_seed + num_batches)
                    noises = [
                        torch.randn(
                            tl.shape, generator=gen,
                            device=tl.device, dtype=tl.dtype,
                        )
                        for tl in target_latents
                    ]
                    # Model-specific timestep sampling (matching training)
                    timesteps, sigmas = self.adapter.sample_timesteps(
                        target_latents[0].shape[0],
                        target_latents[0].device,
                        target_latents[0].dtype,
                        latent_height=target_latents[0].shape[2],
                        latent_width=target_latents[0].shape[3],
                    )
                    # Flow matching interpolation per target
                    sigmas_b = sigmas.view(-1, *(1,) * (target_latents[0].ndim - 1)) if target_latents[0].ndim >= 2 else sigmas
                    noisy_latents = [
                        (1.0 - sigmas_b) * tl + sigmas_b * n
                        for tl, n in zip(target_latents, noises)
                    ]
                    num_targets = len(noisy_latents)

                    # Prepare block swap before forward
                    base_model = self._get_base_model()
                    if hasattr(base_model, "prepare_block_swap_before_forward"):
                        base_model.prepare_block_swap_before_forward()

                    model_input = self.adapter.prepare_model_input(
                        batch, noisy_latents, sigmas
                    )
                    with self._forward_autocast_ctx():
                        model_pred = self.transformer(**model_input)
                    unpacked = self.adapter.unpack_prediction(model_pred)

                    # Unconditional forward for guidance-corrected losses
                    # (guide_flow_matching / DC-Gen Eq. 10).  The whole val
                    # loop already runs under torch.no_grad(), so this second
                    # forward costs activation memory only, no graph.
                    uncond_unpacked = None
                    if any(
                        getattr(l, "needs_uncond_forward", False)
                        for l in val_loss_modules
                    ):
                        uncond_batch = self._build_uncond_batch(
                            batch, device, target_latents[0].dtype
                        )
                        model_input_uncond = self.adapter.prepare_model_input(
                            uncond_batch, noisy_latents, sigmas
                        )
                        with self._forward_autocast_ctx():
                            model_pred_uncond = self.transformer(
                                **model_input_uncond
                            )
                        uncond_unpacked = self.adapter.unpack_prediction(
                            model_pred_uncond
                        )
                        del model_pred_uncond

                    # ── Unified loss across all targets ──
                    reference_latent = self._get_reference_latent(
                        batch, latents, resolved_bc
                    )
                    total = torch.tensor(
                        0.0,
                        device=target_latents[0].device,
                        dtype=target_latents[0].dtype,
                    )
                    batch_breakdown = {}
                    for ti, (noise_i, unpacked_i, target_i) in enumerate(
                        zip(noises, unpacked, target_latents)
                    ):
                        # 按 velocity_sign 分发估算干净 latent（与 losses/flow_matching.py 同源）
                        x0_hat = self.adapter.compute_x0_hat(noise_i, unpacked_i)
                        loss_ctx = LossContext(
                            model_pred=unpacked_i,
                            noise=noise_i,
                            sigmas=sigmas,
                            learning_target=target_i,
                            x0_hat=x0_hat,
                            reference_latent=reference_latent,
                            loss_mask=batch.get("loss_mask"),
                            adapter=self.adapter,
                            model_pred_uncond=(
                                uncond_unpacked[ti]
                                if uncond_unpacked is not None
                                and ti < len(uncond_unpacked)
                                else None
                            ),
                        )
                        for loss_module in val_loss_modules:
                            loss_val = loss_module(loss_ctx)
                            total = total + loss_val
                            batch_breakdown[loss_module.name] = (
                                batch_breakdown.get(loss_module.name, 0.0)
                                + loss_val.item()
                            )
                    # Normalise per target count, then accumulate across batches
                    for k in batch_breakdown:
                        batch_breakdown[k] /= num_targets
                        val_breakdown[k] = val_breakdown.get(k, 0.0) + batch_breakdown[k]
                    total = total / num_targets

                    total_loss += total.item()
                    num_batches += 1
                    pbar.set_postfix({
                        "val_loss": f"{total_loss / max(1, num_batches):.4f}"
                    })

                    del total, model_pred, unpacked, noisy_latents, noises, target_latents

        finally:
            # Restore original weights if EMA was applied
            if self.ema is not None:
                self.ema.restore()

            torch.random.set_rng_state(before_state)
            np.random.seed(np_seed)
            torch.backends.cudnn.deterministic = False

            gc.collect()
            torch.cuda.empty_cache()

        avg_val_loss = total_loss / max(1, num_batches)
        # Average per-loss validation values
        val_loss_breakdown = {
            name: val / max(1, num_batches) for name, val in val_breakdown.items()
        }
        logger.info(f"--- End val_loss: {avg_val_loss:.6f} ---")

        return {
            "val_loss": avg_val_loss,
            "val_loss_breakdown": val_loss_breakdown,
            "epoch": epoch,
            "num_batches": num_batches,
        }

    def _ensure_noise_scheduler(self):
        """Load the noise scheduler for validation generation (lazy init).

        Uses the adapter's load_scheduler when available (respects model-specific
        shift config). Falls back to a default FlowMatchEulerDiscreteScheduler.
        """
        if self.noise_scheduler is not None:
            return

        scheduler_path = self.config.get("scheduler_path", "")
        if scheduler_path:
            try:
                self.noise_scheduler = self.adapter.load_scheduler(scheduler_path)
                logger.info(f"Loaded noise scheduler from {scheduler_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load scheduler from {scheduler_path}: {e}")

        # Try adapter's load_scheduler with empty path (some adapters like Krea2
        # ignore the path and build the scheduler from internal config).
        try:
            self.noise_scheduler = self.adapter.load_scheduler("")
            logger.info("Loaded noise scheduler via adapter.load_scheduler('')")
            return
        except Exception:
            pass

        # Fallback: default flow-matching Euler scheduler with shift=1.0
        # (no dynamic shifting — suboptimal but functional).
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=1.0,
        )
        logger.info("Using default FlowMatchEulerDiscreteScheduler for validation")

    def generate_validation_images(
        self,
        epoch: int,
        val_dataloader: DataLoader,
    ) -> list:
        """Generate validation images from cached val dataset with controlled RNG.

        Instead of using text prompts, this iterates over the cached val dataset
        samples (which have pre-computed embeddings + reference latents) and runs
        the full multi-step denoising loop for each.

        Follows T2ITrainer RNG pattern:
          1. Save RNG state
          2. Set deterministic seed (val_seed)
          3. Run denoising under torch.no_grad()
          4. Restore RNG state

        For each val sample:
         - Uses cached text embeddings + reference latents (not prompts)
         - Starts from deterministic noise (torch.Generator with val_seed)
         - Runs N inference steps via scheduler + transformer
         - Decodes the final latent via adapter.decode_latent
         - Saves both generated image and target reference for comparison

        Args:
            epoch: current epoch number
            val_dataloader: validation data loader (cached samples)

        Returns:
            list of saved image paths
        """
        if not self.gen_images or val_dataloader is None:
            return []

        self._ensure_noise_scheduler()

        # Save RNG state (T2ITrainer pattern)
        before_state = torch.random.get_rng_state()
        np_seed = random.randint(0, 2**31 - 1)
        import numpy as np
        np.random.seed(np_seed)

        # Freeze RNG with val_seed
        torch.manual_seed(self.val_seed)
        np.random.seed(self.val_seed)
        torch.backends.cudnn.deterministic = True

        # Determine output dir (per-epoch subdir if set)
        img_output_dir = self.epoch_output_dir or self.val_output_dir
        os.makedirs(img_output_dir, exist_ok=True)
        saved_paths = []

        logger.info(
            f"\n--- Start val image generation (epoch {epoch}, "
            f"seed={self.val_seed}, steps={self.val_num_inference_steps}) ---"
        )

        # Acquire the shared VAE on GPU for decoding (manager moves it there
        # unless vae_on_device=true keeps it resident; release() restores RAM).
        if self.vae_manager is None:
            logger.warning(
                "generate_validation_images: no vae_manager wired — configure a "
                "VAE-using component or validation.generate_images with a valid "
                "vae_path. Skipping image generation."
            )
            return []
        self.vae = self.vae_manager.acquire(self.device)
        logger.info("VAE acquired on GPU for validation decoding")

        # VRAM snapshot before generation
        if torch.cuda.is_available():
            alloc_before = torch.cuda.memory_allocated() / 1024**3
            peak_before = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(
                f"VRAM before val gen: allocated={alloc_before:.2f} GB, peak={peak_before:.2f} GB"
            )

        # Park block swap for sampling — UNIFORMLY for both transformer
        # classes. Class 1 stores `_offloader` and has disable_block_swap;
        # class 2 (T2ITrainer-compatible) stores `offloader_double` + a
        # `blocks_to_swap` int and has NO disable_block_swap, so a
        # hasattr-gated heal silently skipped: after the last training
        # backward the first `blocks_to_swap` blocks' weights legitimately
        # sit on CPU (the ping-pong only guarantees backward consumed them),
        # and the sampling forward then died on a cuda-vs-cpu matmul.
        # Parking the offloader + zeroing blocks_to_swap makes forward()
        # skip swap logic entirely; the whole-module .to(device) then heals
        # placement (module-level .to is the Params4bit-safe path).
        base_model = self._get_base_model()
        _off = getattr(base_model, "_offloader", None)
        _offd = getattr(base_model, "offloader_double", None)
        _swap_saved = {
            "blocks_to_swap": getattr(base_model, "blocks_to_swap", 0),
            "_offloader": _off,
            "offloader_double": _offd,
        }
        block_swap_was_enabled = (_off is not None) or (_offd is not None)
        if block_swap_was_enabled:
            if hasattr(base_model, "_offloader"):
                base_model._offloader = None
            if hasattr(base_model, "offloader_double"):
                base_model.offloader_double = None
            if hasattr(base_model, "blocks_to_swap"):
                base_model.blocks_to_swap = 0
            base_model.to(self.device)

        # Diagnostic (one-shot): verify the heal actually placed everything.
        if torch.cuda.is_available():
            _cpu = [
                (n, tuple(p.shape), str(p.dtype))
                for n, p in base_model.named_parameters()
                if not p.is_cuda
            ]
            _cpu_extra = [
                (n, tuple(b.shape))
                for n, b in base_model.named_buffers()
                if not b.is_cuda
            ]
            logger.info(
                f"[VAL-GEN] after block-swap park: cpu params={len(_cpu)} "
                f"cpu buffers={len(_cpu_extra)}"
                + (f" e.g. {_cpu[:5]}" if _cpu else "")
                + (f" buf e.g. {_cpu_extra[:3]}" if _cpu_extra else "")
            )

        try:
            with torch.no_grad():
                self.transformer.eval()

                sample_idx = 0
                for batch in val_dataloader:
                    if sample_idx >= self.val_num_images:
                        break

                    # Process each sample in the batch one-by-one.
                    # Validation generation needs batch_size=1 for correct
                    # shape handling (single reference, single embedding).
                    bs = self._batch_size(batch)
                    for sub_idx in range(bs):
                        if sample_idx >= self.val_num_images:
                            break
                        try:
                            sub_batch = self._slice_batch(batch, sub_idx)
                            imgs = self._generate_from_val_sample(
                                sub_batch, epoch, sample_idx
                            )
                            if imgs is not None:
                                gen_pils, ref_pils = imgs
                                # ref_pils may be a tuple (target_pils, ref_conditioning_pils)
                                # or a plain list of target PILs for backward compat.
                                ref_cond_pils = None
                                if isinstance(ref_pils, tuple):
                                    ref_pils, ref_cond_pils = ref_pils

                                # Save each generated image with target index suffix.
                                # Video artifacts are mp4 path strings (or lists
                                # of them) produced by decode_validation_video;
                                # images are PIL objects saved as .png.
                                # Determine target-reference size (for aligning
                                # the generated output resolution to it).
                                _target_size = None
                                for _rp in ref_pils:
                                    if _rp is not None and hasattr(_rp, "size"):
                                        _target_size = _rp.size
                                        break
                                for ti, gen_pil in enumerate(gen_pils):
                                    if isinstance(gen_pil, str):
                                        saved_paths.append(gen_pil)
                                        logger.info(f"Saved validation video: {gen_pil}")
                                        continue
                                    if isinstance(gen_pil, (list, tuple)):
                                        vid_list = [
                                            p for p in gen_pil if isinstance(p, str)
                                        ]
                                        saved_paths.extend(vid_list)
                                        for vp in vid_list:
                                            logger.info(f"Saved validation video: {vp}")
                                        continue
                                    suffix = f"_{ti}" if len(gen_pils) > 1 else ""
                                    gen_path = os.path.join(
                                        img_output_dir,
                                        f"{self.save_name}_val_epoch{epoch}_{sample_idx}{suffix}.png",
                                    )
                                    # Align generated resolution to the target
                                    # reference image size (if they differ).
                                    if _target_size is not None and gen_pil.size != _target_size:
                                        gen_pil = gen_pil.resize(_target_size)
                                        logger.info(
                                            f"Resized val gen {ti} to target size {_target_size}"
                                        )
                                    gen_pil.save(gen_path)
                                    saved_paths.append(gen_path)
                                    logger.info(f"Saved validation image: {gen_path}")
                                # Save reference targets for comparison.
                                for ti, ref_pil in enumerate(ref_pils):
                                    if ref_pil is not None:
                                        suffix = f"_{ti}" if len(ref_pils) > 1 else ""
                                        ref_path = os.path.join(
                                            img_output_dir,
                                            f"{self.save_name}_val_epoch{epoch}_{sample_idx}{suffix}_target.png",
                                        )
                                        ref_pil.save(ref_path)

                                # Save reference conditioning images (e.g. depth maps).
                                if ref_cond_pils is not None:
                                    for ri, ref_cond_pil in enumerate(ref_cond_pils):
                                        if ref_cond_pil is not None:
                                            suffix = f"_{ri}" if len(ref_cond_pils) > 1 else ""
                                            ref_path = os.path.join(
                                                img_output_dir,
                                                f"{self.save_name}_val_epoch{epoch}_{sample_idx}{suffix}_ref.png",
                                            )
                                            ref_cond_pil.save(ref_path)
                                            logger.info(f"Saved reference image: {ref_path}")

                                # Copy the source caption file next to the val
                                # image (if available in the batch). The source
                                # path follows the caption_config ext (could be
                                # .txt / .tags / .i2cf ...), so preserve it.
                                try:
                                    _caption_paths = batch.get("caption_text_paths") or []
                                    _cap_path = None
                                    if isinstance(_caption_paths, list):
                                        if _caption_paths and isinstance(_caption_paths[0], list):
                                            # collated: one list per batch item
                                            _cap_path = _caption_paths[sub_idx][0] if _caption_paths[sub_idx] else None
                                        else:
                                            _cap_path = _caption_paths[0] if _caption_paths else None
                                    if _cap_path and os.path.isfile(_cap_path):
                                        import shutil
                                        _cap_ext = os.path.splitext(_cap_path)[1] or ".txt"
                                        _cp_txt = os.path.join(
                                            img_output_dir,
                                            f"{self.save_name}_val_epoch{epoch}_{sample_idx}_caption{_cap_ext}",
                                        )
                                        shutil.copy2(_cap_path, _cp_txt)
                                        logger.info(f"Saved caption file: {_cp_txt}")
                                except Exception as _e:
                                    logger.warning(f"Failed to copy caption file: {_e}")
                        except Exception as e:
                            import traceback
                            logger.warning(
                                f"Image generation failed for sample {sample_idx}: {e}\n"
                                f"{traceback.format_exc()}"
                            )

                        sample_idx += 1

        finally:
            # VRAM snapshot after generation
            if torch.cuda.is_available():
                alloc_after = torch.cuda.memory_allocated() / 1024**3
                peak_after = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    f"VRAM after val gen: allocated={alloc_after:.2f} GB, peak={peak_after:.2f} GB "
                    f"(delta_peak={peak_after - peak_before:.2f} GB)"
                )

            # Restore RNG state (T2ITrainer pattern)
            torch.random.set_rng_state(before_state)
            np.random.seed(np_seed)
            torch.backends.cudnn.deterministic = False

            # Re-enable block swap for training. Un-park the offloader and
            # blocks_to_swap; the NEXT training step's _pre_forward
            # (move_to_device_except_swap_blocks + prepare_block_swap_before_forward)
            # re-establishes the CPU/GPU ping-pong placement, so no explicit
            # device reshuffle is needed here.
            if block_swap_was_enabled:
                _saved = _swap_saved
                if hasattr(base_model, "_offloader"):
                    base_model._offloader = _saved["_offloader"]
                if hasattr(base_model, "offloader_double"):
                    base_model.offloader_double = _saved["offloader_double"]
                if hasattr(base_model, "blocks_to_swap"):
                    base_model.blocks_to_swap = _saved["blocks_to_swap"]
                # Class 1 keeps its own save/restore bookkeeping consistent.
                if (
                    _saved["_offloader"] is not None
                    and hasattr(base_model, "_saved_offloader")
                ):
                    base_model._saved_offloader = None

            # Release the shared VAE (back to RAM unless vae_on_device=true)
            self.vae_manager.release()
            self.vae = None
            torch.cuda.empty_cache()

            from UnifiedTrainer.utils.flush import flush
            flush()

        logger.info(f"--- End image generation: {len(saved_paths)} images ---")
        return saved_paths

    def _prepare_model_input(
        self,
        batch: dict,
        latent_list: list,
        sigmas: torch.Tensor,
        condition_rows=None,
    ) -> dict:
        """``adapter.prepare_model_input`` with optional pre-built condition rows.

        MiniMax-H3-style adapters build the keyframe-condition rows once per
        generation request (``build_condition_rows``, PR parity — the PR
        freezes the condition for the whole denoising trajectory); the kwarg
        is forwarded only when the adapter supports it, so generic adapters
        are unaffected.
        """
        if hasattr(self.adapter, "build_condition_rows"):
            return self.adapter.prepare_model_input(
                batch, latent_list, sigmas, condition_rows=condition_rows
            )
        return self.adapter.prepare_model_input(batch, latent_list, sigmas)

    def _generate_from_val_sample(
        self,
        batch: dict,
        epoch: int,
        sample_idx: int,
    ) -> Any:
        """Generate a validation image by running the full denoising loop.

        Uses cached embeddings + reference latents from the val dataset batch.
        Runs N inference steps: noise -> denoise -> decode.

        Controlled RNG: torch.Generator(device=device).manual_seed(val_seed + idx)
        for deterministic, reproducible noise (same seed every epoch for progress tracking).

        Returns:
            tuple (generated_PIL_image, reference_PIL_image) or None on failure
        """
        from PIL import Image as PILImage

        # Use self.device (always CUDA) — after BouncingOffloader some params are on CPU
        device = self.device

        # Resolve batch config and move latents to device
        batch_configs = batch.get("batch_configs", [])
        resolved_bc = batch_configs[0] if batch_configs else None

        latents = batch.get("latents", {})
        latents = self._move_latents_to_device(latents, device)
        # The npz cache stores latents as fp32; training re-casts them to the
        # compute dtype on load.  Do the same here so the sampling loop and the
        # VAE decode run in bf16 (fp32 latents would crash the bf16 VAE's conv
        # bias add and slow the loop).
        _first_param = next(self.transformer.parameters())
        if _first_param.dtype in (torch.int8, torch.uint8,
                                  torch.float8_e4m3fn, torch.float8_e5m2):
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = _first_param.dtype
        if compute_dtype not in (torch.float32, torch.float64):
            latents = self._move_latents_to_device(latents, device, compute_dtype)
        batch["latents"] = latents

        # ── Get ALL target latents (single or multi-target) ──
        target_latents = self._get_target_latents(batch, latents, resolved_bc)
        dtype = target_latents[0].dtype

        # ── Controlled RNG — one noise tensor per target ──
        gen = torch.Generator(device=device)
        gen.manual_seed(self.val_seed + sample_idx)

        latent_list = [
            torch.randn(tuple(tl.shape), generator=gen, device=device, dtype=dtype)
            for tl in target_latents
        ]

        # ── Scheduler setup (use first target for mu calculation) ──
        sched_cfg = self.noise_scheduler.config
        mu = None
        if getattr(sched_cfg, "use_dynamic_shifting", False):
            patch_size = getattr(self.adapter, "patch_size", 2)
            # Multi-target: all targets are packed into one sequence, so mu
            # must reflect the TOTAL image sequence length across all targets.
            total_image_seq_len = sum(
                (tl.shape[2] // patch_size) * (tl.shape[3] // patch_size)
                for tl in target_latents
            )
            base_seq = sched_cfg.get("base_image_seq_len", 256)
            max_seq = sched_cfg.get("max_image_seq_len", 4096)
            base_shift = sched_cfg.get("base_shift", 0.5)
            max_shift = sched_cfg.get("max_shift", 1.15)
            m = (max_shift - base_shift) / (max_seq - base_seq)
            b = base_shift - m * base_seq
            mu = total_image_seq_len * m + b

        # Only schedulers that accept the dynamic-shift `mu` parameter
        # (FlowMatch family) receive it; e.g. MiniMaxH3Scheduler.set_timesteps
        # has no `mu` and would raise TypeError on the first validation pass.
        set_timesteps_params = inspect.signature(
            self.noise_scheduler.set_timesteps
        ).parameters
        if "mu" in set_timesteps_params:
            self.noise_scheduler.set_timesteps(
                self.val_num_inference_steps, device=device, mu=mu,
            )
        else:
            self.noise_scheduler.set_timesteps(
                self.val_num_inference_steps, device=device,
            )

        # ── CFG setup ──
        use_cfg = self.val_guidance_scale > 1.0
        uncond_batch = None
        if use_cfg:
            uncond_batch = self._build_uncond_batch(batch, device, dtype)

        # ── Pre-built condition rows (once per request, PR parity) ──
        # MiniMax-H3-style adapters: the keyframe-condition rows are built
        # ONCE before the loop and reused at every denoising step.  Building
        # them inside prepare_model_input per step re-draws the noise-aug
        # component at every sigma (the adapter's condition seed is
        # sigma-derived), so the model would see a wobbling conditioning
        # across the trajectory — a distribution it was never trained on.
        # The uncond batch shares the same latents, so one tensor serves
        # both CFG forwards.
        condition_rows = None
        if hasattr(self.adapter, "build_condition_rows"):
            condition_rows = self.adapter.build_condition_rows(
                batch, device=device, dtype=dtype
            )

        # ── Denoising loop ──
        # Multi-target: latent_list is a list[torch.Tensor]; prepare_model_input
        # packs all targets into one sequence; unpack_prediction returns a list
        # of per-target velocity tensors.
        cfg_label = f" CFG={self.val_guidance_scale}" if use_cfg else ""
        pbar = tqdm(
            self.noise_scheduler.timesteps,
            desc=f"Val gen{cfg_label}",
            leave=False,
        )
        for i, t in enumerate(pbar):
            sigma = self.noise_scheduler.sigmas[i].to(device=device, dtype=dtype)
            sigmas = sigma.expand(latent_list[0].shape[0])

            if use_cfg:
                # Conditional forward — pass list of noise tensors.
                model_input_cond = self._prepare_model_input(
                    batch, latent_list, sigmas, condition_rows
                )
                with self._forward_autocast_ctx():
                    v_pred_cond = self.transformer(**model_input_cond)
                v_conds = self.adapter.unpack_prediction(
                    v_pred_cond
                )  # list[torch.Tensor]

                # Unconditional forward.
                model_input_uncond = self._prepare_model_input(
                    uncond_batch, latent_list, sigmas, condition_rows
                )
                with self._forward_autocast_ctx():
                    v_pred_uncond = self.transformer(**model_input_uncond)
                v_unconds = self.adapter.unpack_prediction(
                    v_pred_uncond
                )  # list[torch.Tensor]

                # Per-target CFG: v_guided[i] = v_uncond[i] + scale * (v_cond[i] - v_uncond[i])
                velocities = [
                    vu + self.val_guidance_scale * (vc - vu)
                    for vc, vu in zip(v_conds, v_unconds)
                ]
            else:
                model_input = self._prepare_model_input(
                    batch, latent_list, sigmas, condition_rows
                )
                with self._forward_autocast_ctx():
                    model_pred = self.transformer(**model_input)
                velocities = self.adapter.unpack_prediction(model_pred)
                # velocities is list[torch.Tensor]

            # Scheduler Euler step — one per target.
            latent_list = [
                self.noise_scheduler.step(v, t, l).prev_sample.to(dtype=dtype, device=device)
                for v, l in zip(velocities, latent_list)
            ]

        # ── Decode each final latent: images -> PIL, videos -> silent mp4 ──
        gen_pils = []
        video_output_dir = self.epoch_output_dir or self.val_output_dir
        multi_target = len(latent_list) > 1
        for ti, latent in enumerate(latent_list):
            if (
                hasattr(self.adapter, "decode_validation_video")
                and latent.ndim == 5
                and latent.shape[2] > 1
            ):
                # Video path (MiniMax-H3 etc.): the adapter decodes to pixels
                # and writes silent mp4s (one per batch item), returning their
                # paths.  gen_pils therefore carries str / list[str] artifacts;
                # the caller saves PIL images (.png) and records video paths
                # verbatim.
                #
                # The mp4 prefix mirrors the PNG identity
                # ({save_name}_val_epoch{epoch}_{sample_idx}[_{ti}]) so
                # per-sample artifacts never collide — validation slices
                # per-sample, so the adapter's batch index b is always 0 and
                # without the prefix samples ≥2 silently overwrite each
                # other's mp4 (and epochs too, when epoch_output_dir is None).
                prefix = f"{self.save_name}_val_epoch{epoch}_{sample_idx}"
                if multi_target:
                    prefix += f"_{ti}"
                vid_paths = self.adapter.decode_validation_video(
                    self.vae, latent, video_output_dir, prefix=prefix
                )
                gen_pils.append(vid_paths[0] if len(vid_paths) == 1 else vid_paths)
            else:
                # Image path (unchanged for krea2 etc.): decode_latent -> PIL.
                image = self.adapter.decode_latent(self.vae, latent)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                if image.ndim == 4:
                    image = image[0]
                gen_pils.append(PILImage.fromarray((image * 255).round().astype("uint8")))

        # Decode reference targets for comparison.  Video references are
        # skipped (None): the generated mp4 already covers the target clip, so
        # decoding a separate target-reference artifact is redundant (avoids a
        # redundant full VAE decode).
        ref_pils = []
        for target_latent in target_latents:
            if (
                hasattr(self.adapter, "decode_validation_video")
                and target_latent.ndim == 5
                and target_latent.shape[2] > 1
            ):
                ref_pils.append(None)
                continue
            try:
                ref_image = self.adapter.decode_latent(self.vae, target_latent)
                ref_image = (ref_image / 2 + 0.5).clamp(0, 1)
                ref_image = ref_image.cpu().permute(0, 2, 3, 1).float().numpy()
                if ref_image.ndim == 4:
                    ref_image = ref_image[0]
                ref_pils.append(PILImage.fromarray((ref_image * 255).round().astype("uint8")))
            except Exception:
                ref_pils.append(None)

        # Decode reference (conditioning) images for visual inspection.
        # These are the input references (e.g. depth maps), not the targets.
        ref_conditioning_pils = self._decode_reference_images(batch, latents, resolved_bc)
        if ref_conditioning_pils is not None:
            ref_pils = (ref_pils, ref_conditioning_pils)

        return gen_pils, ref_pils

    # ── Helpers ───────────────────────────────────────────────────────

    def _decode_reference_images(
        self, batch: dict, latents: dict, resolved_bc: dict = None
    ) -> Optional[list]:
        """Decode reference (conditioning) images from the batch for visual comparison.

        Extracts reference latents (e.g. depth maps) using the resolved
        reference_config key, decodes them to PIL images, and returns them.
        Returns None if no reference latents are found or decoding fails.
        """
        from PIL import Image as PILImage

        ref_latent = self._get_reference_latent(batch, latents, resolved_bc)
        if ref_latent is None:
            return None

        # Reference may be a list (multi-reference) or single tensor
        ref_latents_list = ref_latent if isinstance(ref_latent, list) else [ref_latent]
        ref_conditioning_pils = []
        for ref_lat in ref_latents_list:
            try:
                ref_img = self.adapter.decode_latent(self.vae, ref_lat)
                ref_img = (ref_img / 2 + 0.5).clamp(0, 1)
                ref_img = ref_img.cpu().permute(0, 2, 3, 1).float().numpy()
                if ref_img.ndim == 4:
                    ref_img = ref_img[0]
                ref_conditioning_pils.append(
                    PILImage.fromarray((ref_img * 255).round().astype("uint8"))
                )
            except Exception:
                ref_conditioning_pils.append(None)

        return ref_conditioning_pils if ref_conditioning_pils else None

    def _resolve_target_keys(self, resolved_bc: dict = None) -> list[str] | None:
        """Resolve target image keys from the 5-layer composable config.

        Looks up ``target_configs[batch_config.target_config]`` entries,
        each of which maps to an image key (e.g. ``{"image": "T"}`` →
        ``"T"``). Returns the full list of image keys for multi-target
        datasets, or ``None`` when no config-based resolution is possible
        (caller falls back to legacy single-target logic).

        This is shared by ``_get_target_latents`` (tensor extraction)
        and the Helios corrupt guard (which must skip ALL targets).
        """
        if not resolved_bc or not resolved_bc.get("target_config"):
            return None
        tc_key = resolved_bc["target_config"]
        dataset_configs = self.config.get("data", {}).get("dataset_configs", [])
        for ds_cfg in dataset_configs:
            target_configs = ds_cfg.get("target_configs", {})
            if isinstance(target_configs, dict) and tc_key in target_configs:
                entries = target_configs[tc_key]
                if isinstance(entries, list):
                    return [
                        e.get("image", e) if isinstance(e, dict) else e
                        for e in entries
                    ]
                return [entries]
        return None

    def _get_target_latent(self, batch: dict, latents: dict, resolved_bc: dict = None) -> torch.Tensor:
        """Extract the primary training target latent from the batch.

        When batch_config is available, uses the resolved target_config key
        to find the correct target latent.  Falls back to legacy heuristic
        for backward compatibility.
        """
        # Try resolved target_config key from batch_config
        if resolved_bc and resolved_bc.get("target_config"):
            tc_key = resolved_bc["target_config"]
            if tc_key in latents:
                return latents[tc_key]
            # Try uppercase variant
            if tc_key.upper() in latents:
                return latents[tc_key.upper()]
        # Legacy fallbacks
        if "target" in latents:
            return latents["target"]
        if "T" in latents:
            return latents["T"]
        if "target" in batch:
            return batch["target"]
        # Fallback: first latent
        if latents:
            return next(iter(latents.values()))
        raise ValueError("No target latent found in batch")

    def _get_target_latents(
        self, batch: dict, latents: dict, resolved_bc: dict = None
    ) -> list[torch.Tensor]:
        """Extract ALL target latents from the batch (multi-target support).

        Delegates key resolution to ``_resolve_target_keys`` so the
        config-lookup logic is shared with the Helios corrupt guard.
        Always returns a list — single-target returns a one-element list.
        """
        target_keys = self._resolve_target_keys(resolved_bc)
        if target_keys:
            result = [latents[k] for k in target_keys if k in latents]
            if result:
                return result
        # Fallback — single target via the existing resolution logic.
        return [self._get_target_latent(batch, latents, resolved_bc)]

    @staticmethod
    def _move_latents_to_device(
        latents: dict, device: torch.device, dtype: Optional[torch.dtype] = None
    ) -> dict:
        """Move batch latents to ``device`` (and optionally ``dtype``).

        Single latents are tensors; multi-reference latents are LISTS of
        tensors (one per reference image) and are moved element-wise.
        """
        moved = {}
        for k, v in latents.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device=device, dtype=dtype)
            elif isinstance(v, list):
                moved[k] = [t.to(device=device, dtype=dtype) for t in v]
            else:
                moved[k] = v
        return moved

    def _get_reference_latent(
        self, batch: dict, latents: dict, resolved_bc: dict = None
    ) -> Optional[torch.Tensor]:
        """Extract the reference/condition latent (e.g. depth) from the batch.

        Uses the resolved reference_config key from batch_config.  Falls back
        to any non-target latent when no explicit reference_config is configured.
        Returns None when no reference is present (e.g. unconditional batches).
        """
        # Try resolved reference_config key from batch_config
        if resolved_bc and resolved_bc.get("reference_config"):
            ref_key = resolved_bc["reference_config"]
            if ref_key in latents:
                v = latents[ref_key]
                # Multi-reference roles are lists of tensors — use the
                # primary reference for loss/decode consumers.
                return v[0] if isinstance(v, list) and v else v
        # Fallback: any non-target latent
        if resolved_bc and resolved_bc.get("target_config"):
            tc_key = resolved_bc["target_config"]
            for k, v in latents.items():
                if k == tc_key:
                    continue
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, list) and v:
                    return v[0]
        return None

    def _apply_caption_dropout(self, batch: dict, resolved_bc: dict) -> None:
        """Dynamic per-step caption dropout (matches T2ITrainer behaviour).

        On each training step, with probability ``caption_dropout``, replace
        ALL per-sample embeddings in the batch with the cached unconditional
        (empty-text) embedding.  This is applied *before* the forward pass so
        the dropout decision is fresh every step — unlike the old static
        dataset-level dropout that fixed the embedding at load time.

        The empty embedding is lazy-loaded from the same NPZ file used by
        the validation CFG unconditional pass.
        """
        caption_dropout = (
            resolved_bc.get("caption_dropout", 0.0) if resolved_bc else 0.0
        )
        if caption_dropout <= 0:
            return
        if random.random() >= caption_dropout:
            return

        # Lazy-load and cache the empty embedding (numpy dict — the adapter's
        # _extract_encoder_hidden_states handles numpy→torch conversion).
        if self._train_empty_embed is None:
            if not os.path.isfile(self._val_empty_embed_path):
                logger.warning(
                    f"caption_dropout={caption_dropout:.2f} but empty embedding "
                    f"not found at {self._val_empty_embed_path}. Skipping dropout."
                )
                return
            from UnifiedTrainer.data.embedding_cache import EmbeddingCache
            npz = EmbeddingCache.load(self._val_empty_embed_path)
            self._train_empty_embed = {
                "prompt_embed": npz["prompt_embed"],
                "prompt_embeds_mask": npz["prompt_embeds_mask"],
            }

        # Replace every per-sample embedding with the unconditional one.
        # The adapter's _extract_encoder_hidden_states handles numpy→torch
        # and zero-padding transparently.
        n = len(batch.get("embeddings", []))
        if n > 0:
            batch["embeddings"] = [self._train_empty_embed] * n
            # Flag for downstream consumers (e.g. ExplorativeNoiseSelector
            # uses this to raise K on unconditional batches, which have far
            # higher per-condition multimodality than text-conditioned ones).
            batch["_caption_dropped"] = True

    def _build_uncond_batch(
        self, batch: dict, device: torch.device, dtype: torch.dtype
    ) -> dict:
        """Build an unconditional batch dict for CFG / guidance correction.

        The uncond condition is composed of TWO SEPARATE parts (krea2-style):
          1. the cached global empty-text embedding, swapped into
             ``batch["embeddings"]`` (the mature unconditional text), and
          2. the reference images, which stay in ``batch["latents"]`` untouched
             (shared via the shallow copy).
        The adapter marks the batch with ``_uncond_empty_text`` and synthesizes
        the references' VLM slot layout over the text-only empty embedding, so
        the uncond forward keeps the reference conditioning instead of dropping
        it (the transformer overwrites slot content with the latent rows, so
        only the slot layout matters).

        Used by:
        - validation CFG sampling (``val_guidance_scale > 1``),
        - guide_flow_matching's per-step uncond forward (DC-Gen Eq. 10 —
          the empty-condition pass of the corrected objective).
        """
        import numpy as np

        if self._val_empty_embed is None:
            if not os.path.isfile(self._val_empty_embed_path):
                raise FileNotFoundError(
                    f"Empty embedding not found at {self._val_empty_embed_path}. "
                    f"Cannot use CFG (guidance_scale > 1) without unconditional embeddings."
                )
            from UnifiedTrainer.data.embedding_cache import EmbeddingCache
            npz = EmbeddingCache.load(self._val_empty_embed_path)
            uol = npz.get("user_opener_len", -1)
            uol = int(uol) if uol is not None else -1
            self._val_empty_embed = {
                "prompt_embed": torch.from_numpy(npz["prompt_embed"]).to(
                    dtype=dtype, device=device
                ),
                "prompt_embeds_mask": torch.from_numpy(
                    npz["prompt_embeds_mask"]
                ).to(device=device),
                # Token offset where ti2i reference slots belong in the
                # extracted row (recorded by encode_text at cache time;
                # <=0 on older caches -> adapter falls back to stream head).
                "user_opener_len": uol if uol > 0 else 0,
            }

        # Deep-copy the batch, replacing only the embeddings.
        # Build one unconditional embedding entry per sample in the original batch
        # so that prepare_model_input receives matching B-dim tensors.
        import copy
        uncond_batch = copy.copy(batch)
        orig_embeddings = batch.get("embeddings", [])
        n_samples = len(orig_embeddings) if orig_embeddings else 1
        uncond_batch["embeddings"] = [
            {
                "prompt_embed": self._val_empty_embed["prompt_embed"],
                "prompt_embeds_mask": self._val_empty_embed["prompt_embeds_mask"],
            }
            for _ in range(n_samples)
        ]
        uncond_batch["_uncond_empty_text"] = True
        uncond_batch["_uncond_ref_slot_pos"] = self._val_empty_embed.get(
            "user_opener_len", 0
        )
        return uncond_batch

    @staticmethod
    def _batch_size(batch: dict) -> int:
        """Infer the batch size from the batch dict."""
        latents = batch.get("latents", {})
        for v in latents.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        embeddings = batch.get("embeddings", [])
        if embeddings:
            return len(embeddings)
        return 1

    @staticmethod
    def _slice_batch(batch: dict, idx: int) -> dict:
        """Return a single-sample sub-batch (batch_size=1) from a multi-sample batch."""
        import copy
        sub = copy.copy(batch)

        # Slice latents
        latents = batch.get("latents", {})
        sub["latents"] = {
            k: v[idx:idx+1] if isinstance(v, torch.Tensor) else v
            for k, v in latents.items()
        }

        # Slice embeddings
        embeddings = batch.get("embeddings", [])
        if embeddings and idx < len(embeddings):
            sub["embeddings"] = [embeddings[idx]]

        # Slice batch_configs
        batch_configs = batch.get("batch_configs", [])
        if batch_configs and idx < len(batch_configs):
            sub["batch_configs"] = [batch_configs[idx]]

        return sub

    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if a checkpoint should be saved for this epoch.

        Config options:
            output.save_every_epoch: int (default 1) -save every N epochs
            output.save_start_epoch: int (default 0) -only start saving from this epoch (inclusive)
        """
        save_every = self.config.get("output", {}).get("save_every_epoch", 1)
        save_start = self.config.get("output", {}).get("save_start_epoch", 0)
        return epoch >= save_start and epoch % save_every == 0

    def should_validate(self) -> bool:
        """Check if validation should run at current epoch."""
        val_every = self.config.get("training", {}).get("validate_every_epoch", 1)
        return (self.epoch + 1) % val_every == 0

    def state_dict(self) -> dict:
        """Return trainer state for checkpointing."""
        return {
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load trainer state from checkpoint."""
        self.step = state.get("step", 0)
        self.epoch = state.get("epoch", 0)
