"""
Checkpoint — save and load LoRA checkpoints.

Handles saving LoRA adapter weights, optimizer state, and trainer state
for training resume and final model export.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and load training checkpoints.

    Saves:
       - LoRA adapter weights (safetensors)
       - Optimizer state (optional)
       - Trainer state (step, epoch)
       - Config snapshot
    """

    def __init__(self, output_dir: str, save_name: str = "lora"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_name = save_name

    def save_lora(
        self,
        model: torch.nn.Module,
        step: int,
        epoch: int,
        config: Optional[dict] = None,
        is_final: bool = False,
    ) -> Path:
        """Save LoRA adapter weights.

        Extracts only LoRA parameters (those with 'lora_' in name) and saves
        them in safetensors format.
        """
        suffix = "final" if is_final else f"epoch{epoch}"
        filename = f"{self.save_name}_{suffix}.safetensors"
        path = self.output_dir / filename

        # Extract LoRA state dict via PEFT's native utility (handles
        # buffers like lora_A.alpha, embedding params, and key format correctly).
        # Fallback to manual extraction for non-PEFT models.
        lora_state_dict = {}
        try:
            from peft.utils import get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(model)
            if not lora_state_dict:
                raise ValueError("empty state dict from get_peft_model_state_dict")
        except Exception:
            for name, param in model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    lora_state_dict[name] = param.detach().cpu()

        if not lora_state_dict:
            logger.warning("No LoRA parameters found, saving full state dict")
            lora_state_dict = {
                name: param.detach().cpu()
                for name, param in model.state_dict().items()
                if "lora_" in name
            }

        try:
            from safetensors.torch import save_file as save_safetensors
            save_safetensors(lora_state_dict, str(path))
        except ImportError:
            torch.save(lora_state_dict, str(path).replace(".safetensors", ".pt"))
            path = path.with_suffix(".pt")

        logger.info(f"Saved LoRA checkpoint: {path} ({len(lora_state_dict)} params)")

        # Save metadata
        meta = {"step": step, "epoch": epoch, "num_params": len(lora_state_dict)}
        if config:
            meta["config"] = config
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return path

    # ── ComfyUI key mapping (LOAD side only) ──────────────────────
    # Only the comfy->peft direction is kept: loading legacy ComfyUI
    # checkpoints still works, but nothing writes *_comfyui.safetensors
    # any more (the LoRA export was a plain key rename that ComfyUI could
    # not load, and the LoKr export tripped a torch.kron stride bug).

    # Reverse: ComfyUI → PEFT (order matters: specific patterns first)
    _COMFYUI_REVERSE = [
        (".lora_down.weight", ".lora_A.default.weight"),
        (".lora_up.weight", ".lora_B.default.weight"),
        # Raw format (no .default suffix) — produced by some ComfyUI nodes
        (".lora_A.weight", ".lora_A.default.weight"),
        (".lora_B.weight", ".lora_B.default.weight"),
        ("txtfusion", "text_fusion"),
        # Use dot-delimited matching for module-name segments to avoid
        # corrupting substrings inside other segment names.
        (".gate.", ".to_gate."),
        (".wv.", ".to_v."),
        (".wk.", ".to_k."),
        (".wq.", ".to_q."),
        (".wo.", ".to_out.0."),
        # T2ITrainer ComfyUI format uses mlp. (→ ff. in PEFT)
        (".mlp.", ".ff."),
    ]
    # blocks → transformer_blocks, but NOT refiner_blocks or
    # layerwise_blocks (Krea2 txtfusion sub-structures).  Regex negative
    # lookbehind ensures only the right segments are converted.
    import re as _re
    _BLOCKS_PATTERN = _re.compile(
        r'(?<!refiner_)(?<!layerwise_)(?<![A-Za-z])blocks'
    )
    # H3 comfy keys: top-level 'blocks.N.attn...' only — NOT
    # 'token_refiner.blocks.N...' (refiner blocks keep their own name).
    _H3_BLOCKS_PATTERN = _re.compile(
        r'(?<!refiner\.)(?<![A-Za-z])blocks\.'
    )

    @classmethod
    def _convert_comfyui_to_peft(cls, state_dict: dict) -> dict:
        """Convert ComfyUI-format LoRA keys back to PEFT format.

        MiniMax-H3 comfy keys (fused 'qkv_proj' / 'mlp.fc1-fc2') are
        converted through the H3 path (fused qkv split back into
        to_q/to_k/to_v); everything else uses the generic reverse mapping.
        """
        if cls._is_minimax_h3_comfyui(state_dict):
            return cls._convert_h3_comfyui_to_peft(state_dict)
        converted = {}
        for key, tensor in state_dict.items():
            new_key = key
            # Strip ComfyUI prefix
            if new_key.startswith("diffusion_model."):
                new_key = new_key[len("diffusion_model."):]
            for old, new in cls._COMFYUI_REVERSE:
                new_key = new_key.replace(old, new)
            # blocks → transformer_blocks, but NOT refiner_blocks
            # (handles both top-level blocks.0 and single_blocks.0)
            new_key = cls._BLOCKS_PATTERN.sub("transformer_blocks", new_key)
            # Add PEFT prefix
            if not new_key.startswith("base_model.model."):
                new_key = "base_model.model." + new_key
            converted[new_key] = tensor
        return converted

    # ── MiniMax-H3 ComfyUI -> PEFT conversion (LOAD side) ──────────
    # ComfyUI's MiniMax-H3 (comfy/ldm/minimax/model.py) uses FUSED attention:
    #   blocks.N.attn.qkv_proj  Linear(hidden, inner*3)  (split q,k,v in order)
    #   blocks.N.attn.out_proj, blocks.N.mlp.fc1, blocks.N.mlp.fc2 (swiglu)
    # The diffusers PEFT names (to_q/to_k/to_v, to_out.0, ff.net.0.proj,
    # ff.net.2) must therefore be FUSED/renamed — the krea2-style wq/wk/wv
    # mapping is wrong for H3.  ComfyUI loads `diffusion_model.<exact-key>`
    # lora keys generically (comfy/lora.py model_lora_keys_unet), so the
    # fused qkv_proj pair loads without any H3-specific key map.
    _H3_QKV = ("to_q", "to_k", "to_v")

    @staticmethod
    def _is_minimax_h3_comfyui(state_dict: dict) -> bool:
        """ComfyUI H3 keys use fused qkv_proj / mlp.fc1-fc2 (krea2 comfy uses
        wq/wk/wv/wo + mlp.gate/up/down)."""
        return any("qkv_proj" in k or ".mlp.fc" in k for k in state_dict)

    @staticmethod
    def _split_h3_key(key: str):
        """Normalize a PEFT H3 key to (module_key, suffix).

        Handles both get_peft_model_state_dict output
        (`base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight`)
        and the manual fallback (`transformer_blocks.0.attn.to_q.lora_A.weight`).
        Returns (None, None) for non-weight keys (e.g. alpha buffers).
        """
        k = key
        if k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        for suffix in (".lora_A.default.weight", ".lora_B.default.weight",
                       ".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suffix):
                return k[:-len(suffix)], suffix
        return None, None

    @classmethod
    def _convert_h3_comfyui_to_peft(cls, state_dict: dict) -> dict:
        """Convert ComfyUI fused-qkv H3 keys back to PEFT format.

        Splits qkv_proj lora_down/up back into to_q/to_k/to_v thirds
        (down rows dim 0, up cols dim 1) and reverses all renames.
        Alpha keys are dropped (PEFT recomputes scale from module alpha).
        """
        converted: dict = {}
        fused: dict = {}     # "blocks.N.attn" -> {"down": t, "up": t}
        singles: list = []
        for key, tensor in state_dict.items():
            k = key
            if k.startswith("diffusion_model."):
                k = k[len("diffusion_model."):]
            if k.endswith(".alpha"):
                continue  # PEFT derives scale from the module's own alpha
            if k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight"):
                suffix = ".lora_down.weight" if k.endswith(".lora_down.weight") \
                    else ".lora_up.weight"
                mod_key = k[:-len(suffix)]
                is_down = suffix == ".lora_down.weight"
                if mod_key.endswith(".attn.qkv_proj"):
                    attn_key = mod_key[:-len(".qkv_proj")]
                    fused.setdefault(attn_key, {})["down" if is_down else "up"] = tensor
                else:
                    singles.append((mod_key, tensor, is_down))

        for attn_key, pair in fused.items():
            down, up = pair.get("down"), pair.get("up")
            if (down is None or up is None or down.shape[0] % 3 != 0
                    or up.shape[1] != down.shape[0]):
                logger.warning(
                    f"H3 comfy->peft: skipping malformed fused qkv {attn_key}"
                )
                continue
            r = down.shape[0] // 3
            base = cls._unrename_h3_module(attn_key)
            for i, name in enumerate(cls._H3_QKV):
                mod_key = f"{base}.{name}"
                converted[f"base_model.model.{mod_key}.lora_A.default.weight"] = \
                    down[i * r:(i + 1) * r]
                converted[f"base_model.model.{mod_key}.lora_B.default.weight"] = \
                    up[:, i * r:(i + 1) * r]

        for mod_key, tensor, is_down in singles:
            new_key = "base_model.model." + cls._unrename_h3_module(mod_key)
            converted[f"{new_key}.{'lora_A' if is_down else 'lora_B'}.default.weight"] = \
                tensor

        return converted

    @classmethod
    def _unrename_h3_module(cls, mod_key: str) -> str:
        """ComfyUI H3 module name -> diffusers name."""
        k = mod_key
        if k.endswith(".attn.out_proj"):
            k = k[:-len(".attn.out_proj")] + ".attn.to_out.0"
        elif k.endswith(".mlp.fc1"):
            k = k[:-len(".mlp.fc1")] + ".ff.net.0.proj"
        elif k.endswith(".mlp.fc2"):
            k = k[:-len(".mlp.fc2")] + ".ff.net.2"
        k = cls._H3_BLOCKS_PATTERN.sub("transformer_blocks.", k)
        return k

    @staticmethod
    def _detect_lora_format(state_dict: dict) -> str:
        """Auto-detect whether a LoRA checkpoint is PEFT or ComfyUI format.

        Returns 'comfyui' or 'peft'.
        """
        sample = list(state_dict.keys())[:5]
        for k in sample:
            if "diffusion_model." in k or ".lora_down." in k or ".lora_up." in k:
                return "comfyui"
        return "peft"

    def save_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        step: int,
        epoch: int,
        epoch_idx: int,
        config: Optional[dict] = None,
        is_final: bool = False,
        extra_state: Optional[dict] = None,
    ) -> Path:
        """Save full training state for exact resume.

        Saves optimizer state, scheduler state, RNG states, step, epoch.
        File: {save_name}_epoch{N}_training_state.pt

        Use --resume-full to restore from this.
        """
        suffix = "final" if is_final else f"epoch{epoch_idx}"
        filename = f"{self.save_name}_{suffix}_training_state.pt"
        path = self.output_dir / filename

        # Unwrap AcceleratedOptimizer to reach the raw bnb/torch optimizer.
        # The wrapper's state_dict() may alter the format in ways that
        # bnb's load_state_dict() cannot round-trip correctly.
        raw_opt = optimizer
        if hasattr(raw_opt, "optimizer"):
            raw_opt = raw_opt.optimizer

        state = {
            "step": step,
            "epoch": epoch,
            "optimizer_state_dict": raw_opt.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
            "torch_rng_state": torch.get_rng_state(),
            "config_snapshot": config,
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        import random as _random
        state["python_rng_state"] = _random.getstate()

        if extra_state is not None:
            state["extra_state"] = extra_state

        torch.save(state, str(path))
        logger.info(f"Saved training state: {path}")
        return path

    def load_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        checkpoint_path: str,
    ) -> dict:
        """Load full training state from a _training_state.pt file.

        Restores optimizer, scheduler, RNG states, step, epoch.
        Returns metadata dict with step/epoch.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Training state not found: {path}")

        state = torch.load(str(path), map_location="cpu", weights_only=False)

        # Unwrap AcceleratedOptimizer to reach the raw bnb/torch optimizer
        # so load_state_dict() receives the exact format it originally saved.
        raw_opt = optimizer
        if hasattr(raw_opt, "optimizer"):
            raw_opt = raw_opt.optimizer

        # Move optimizer state tensors to the raw optimizer's device before
        # load_state_dict — bnb's AdamW8bit stores quantized state tensors
        # (uint8) that must be on the correct device for the step() to
        # produce correct updates.  map_location="cpu" above avoids GPU OOM
        # during loading, but we must relocate before handing to the optimizer.
        opt_state = state.get("optimizer_state_dict", {})
        param_state = opt_state.get("state", {})
        if param_state:
            # Resolve target device from the raw optimizer's first parameter
            try:
                first_param = next(iter(raw_opt.param_groups[0]["params"]))
                target_device = first_param.device
            except (StopIteration, IndexError, KeyError):
                target_device = torch.device("cpu")
            for group_state in param_state.values():
                for k in list(group_state.keys()):
                    if isinstance(group_state[k], torch.Tensor):
                        group_state[k] = group_state[k].to(device=target_device)

        raw_opt.load_state_dict(opt_state)
        n_state_entries = len(getattr(raw_opt, "state", {}))
        logger.info(
            f"Restored optimizer state from {path} "
            f"({n_state_entries} state entries)"
        )
        if n_state_entries == 0 and param_state:
            logger.warning(
                "Optimizer state dict had entries but none were loaded — "
                "the optimizer may be starting fresh (loss will spike)!"
            )

        if lr_scheduler and state.get("lr_scheduler_state_dict"):
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
            logger.info(f"Restored LR scheduler state")

        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        if state.get("python_rng_state") is not None:
            import random as _random
            _random.setstate(state["python_rng_state"])

        logger.info(f"Restored RNG states from {path}")
        return {
            "step": state["step"],
            "epoch": state["epoch"],
            "extra_state": state.get("extra_state"),
        }

    @staticmethod
    def get_training_state_path(checkpoint_path: str) -> Path:
        """Derive the _training_state.pt path from a LoRA checkpoint path.

        Given lora_epoch3.safetensors → lora_epoch3_training_state.pt
        Given lora_epoch3_comfyui.safetensors → lora_epoch3_training_state.pt
        """
        path = Path(checkpoint_path)
        stem = path.stem
        if stem.endswith("_comfyui"):
            stem = stem[: -len("_comfyui")]
        return path.parent / f"{stem}_training_state.pt"

    def load_lora(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        strict: bool = False,
    ) -> dict:
        """Load LoRA weights into model. Auto-detects PEFT or ComfyUI format.

        Accepts both:
          - PEFT format:     base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight
          - ComfyUI format:  diffusion_model.blocks.0.attn.wq.lora_down.weight

        Returns metadata dict if available.
        """
        path = Path(checkpoint_path)

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

        # ── Auto-detect format and convert if needed ──────────────
        fmt = self._detect_lora_format(state_dict)
        if fmt == "comfyui":
            logger.info(f"Detected ComfyUI format, converting to PEFT...")
            state_dict = self._convert_comfyui_to_peft(state_dict)
            logger.info(f"Converted {len(state_dict)} keys ComfyUI → PEFT")
        else:
            logger.info(f"Detected PEFT format (no conversion needed)")

        # Load via PEFT's native utility first (handles key format correctly,
        # only touches adapter params, reports LoRA-specific issues only).
        # Falls back to raw load_state_dict for non-PEFT models.
        peft_loaded = False
        try:
            from peft.utils import set_peft_model_state_dict
            set_peft_model_state_dict(model, state_dict)
            peft_loaded = True
            logger.info(f"Loaded LoRA via PEFT set_peft_model_state_dict ({len(state_dict)} keys)")
        except Exception as e:
            logger.debug(f"PEFT load failed ({e}), falling back to load_state_dict")

        if not peft_loaded:
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            # Classify missing keys: LoRA keys missing = real problem,
            # base-model keys missing = expected (we only saved LoRA params).
            lora_missing = [k for k in missing if "lora" in k.lower()]
            base_missing = [k for k in missing if "lora" not in k.lower()]
            lora_unexpected = [k for k in unexpected if "lora" in k.lower()]

            if lora_missing:
                logger.warning(
                    f"Missing LoRA keys: {len(lora_missing)} (THESE ARE PROBLEMATIC!)"
                )
                for k in lora_missing[:10]:
                    logger.warning(f"  missing: {k}")
            if lora_unexpected:
                logger.warning(
                    f"Unexpected LoRA keys: {len(lora_unexpected)} (checkpoint has keys model doesn't)"
                )
                for k in lora_unexpected[:10]:
                    logger.warning(f"  unexpected: {k}")
            if base_missing:
                logger.info(
                    f"Base model keys not in checkpoint: {len(base_missing)} (expected for LoRA-only checkpoint)"
                )

        logger.info(f"Loaded LoRA checkpoint: {path} ({len(state_dict)} params)")

        # Load metadata
        meta_path = path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return meta

    # ── LoKR (LyCORIS) checkpoint support ────────────────────────────

    def save_lokr(
        self,
        lycoris_net,
        step: int,
        epoch: int,
        config: Optional[dict] = None,
        is_final: bool = False,
    ) -> Path:
        """Save LoKR adapter weights via LyCORIS native API.

        Saves in LyCORIS format (lycoris_ prefix, lokr_w1/w2 keys).
        Also saves metadata JSON for resume.
        """
        suffix = "final" if is_final else f"epoch{epoch}"
        filename = f"{self.save_name}_{suffix}.safetensors"
        path = self.output_dir / filename

        # Self-contained LoKR save — safetensors with lycoris-compatible keys
        lycoris_net.save_weights(str(path), dtype=torch.bfloat16)
        logger.info(f"Saved LoKR checkpoint: {path} ({lycoris_net.num_modules} modules)")

        # Metadata
        meta = {"step": step, "epoch": epoch, "network_type": "lokr",
                "num_modules": lycoris_net.num_modules}
        if config:
            meta["config"] = config
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return path

    def load_lokr(self, lycoris_net, checkpoint_path: str) -> dict:
        """Load LoKR weights into an existing LyCORIS network.

        The network must already be created with the same preset/config
        (same target layers, same factor) so parameter shapes match.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"LoKR checkpoint not found: {path}")

        lycoris_net.load_weights(str(path))
        logger.info(f"Loaded LoKR weights from {path}")

        # Load metadata
        meta_path = path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        return meta

    def list_checkpoints(self) -> list:
        """List all saved checkpoints."""
        return sorted(
            p.name
            for p in self.output_dir.glob(f"{self.save_name}_*.safetensors")
        ) + sorted(
            p.name
            for p in self.output_dir.glob(f"{self.save_name}_*.pt")
        )
