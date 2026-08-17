"""
Self-contained LoKR (LoRA with Kronecker Product) implementation.

VRAM-optimized: uses reshaped matmul to compute x @ kron(W1, W2).T
WITHOUT ever materializing the full Kronecker product tensor.

For a [4096, 4096] layer with factor=8:
  - Library approach: torch.kron() → 32MB temporary per layer
  - This approach: two small matmuls → ~4KB temporary per layer

Architecture:
  LokrLayer  — per-layer adapter (params + reshaped forward)
  LokrNetwork — discovers targets, creates LokrLayers, forward hooks
  apply_lokr() — entry point (replaces lycoris-lora library)

Compatible with BouncingOffloader (uses forward hooks, no monkey-patch conflict).
Reference: ai-toolkit/toolkit/models/lokr.py, md/lokr_implementation/new_lokr_implementation_plan.md
"""
from __future__ import annotations

import fnmatch
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# musubi-tuner LoKr full-rank sentinel: its GUI translates `lokr_full_rank: true`
# into network_dim / network_alpha = 9999 (checkpoint metadata: ss_network_dim=9999,
# ss_network_alpha=9999.0). The huge dim forces the full-matrix W2 path
# (lora_dim >= max(out_k, in_n)/2) and scale = alpha/dim = 1.0.
# We reuse the same sentinel so checkpoints stay alpha-compatible with musubi.
FULL_RANK_SENTINEL = 9999


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class LokrConfig:
    """LoKR network configuration.

    Attributes:
        rank: Low-rank dimension for W2 decomposition (and W1 if decompose_both).
        alpha: Scaling factor. Effective scale = alpha / rank; in full-W2 mode
        alpha is forced to rank -> scale = 1.0 (musubi-aligned).
        factor: Factorization factor for splitting dimensions (-1 = auto).
        model_type: Preset selector for target layer patterns.
        target_modules: Override fnmatch patterns; None uses model_type defaults.
        multiplier: Global multiplier for adapter contribution.
        decompose_both: Also decompose W1 into w1_a @ w1_b (saves params).
        full_rank: Force full-matrix W1 and W2 (musubi `lokr_full_rank: true`).
        Disables both low-rank decompositions; scale becomes 1.0. Like musubi,
        rank and alpha are overridden to FULL_RANK_SENTINEL (9999), so the
        saved alpha buffer matches musubi full-rank checkpoints.
    """

    rank: int = 4
    alpha: float = 1.0
    factor: int = -1
    model_type: str = "krea2"
    target_modules: Optional[List[str]] = None
    multiplier: float = 1.0
    decompose_both: bool = False
    full_rank: bool = False

    def __post_init__(self):
        if self.full_rank:
            # musubi-compatible full-rank mode: the configured rank/alpha are
            # ignored, exactly like musubi's GUI overriding `linear`/`linear_alpha`
            # with the 9999 sentinel when `lokr_full_rank` is checked.
            self.rank = FULL_RANK_SENTINEL
            self.alpha = float(FULL_RANK_SENTINEL)


# ── Model-specific target patterns ───────────────────────────────────────

_KREA2_PATTERNS = [
    "*transformer_blocks.*.attn.to_k",
    "*transformer_blocks.*.attn.to_q",
    "*transformer_blocks.*.attn.to_v",
    "*transformer_blocks.*.attn.to_out.0",
    "*transformer_blocks.*.attn.add_k_proj",
    "*transformer_blocks.*.attn.add_q_proj",
    "*transformer_blocks.*.attn.add_v_proj",
    "*transformer_blocks.*.attn.to_add_out",
    "*transformer_blocks.*.ff.gate",
    "*transformer_blocks.*.ff.up",
    "*transformer_blocks.*.ff.down",
    "*text_fusion.*.attn.to_k",
    "*text_fusion.*.attn.to_q",
    "*text_fusion.*.attn.to_v",
    "*text_fusion.*.attn.to_out.0",
]

_QWEN_PATTERNS = [
    "*transformer_blocks.*.attn.to_k",
    "*transformer_blocks.*.attn.to_q",
    "*transformer_blocks.*.attn.to_v",
    "*transformer_blocks.*.attn.to_out.0",
    "*transformer_blocks.*.attn.add_k_proj",
    "*transformer_blocks.*.attn.add_q_proj",
    "*transformer_blocks.*.attn.add_v_proj",
    "*transformer_blocks.*.attn.to_add_out",
    "*transformer_blocks.*.img_mlp.net.2",
    "*transformer_blocks.*.txt_mlp.net.2",
]

_FLUX_PATTERNS = [
    "*transformer_blocks.*.attn.to_k",
    "*transformer_blocks.*.attn.to_q",
    "*transformer_blocks.*.attn.to_v",
    "*transformer_blocks.*.attn.to_out.0",
    "*transformer_blocks.*.ff.net.0.proj",
    "*transformer_blocks.*.ff.net.2",
    "*transformer_blocks.*.ff_context.net.0.proj",
    "*transformer_blocks.*.ff_context.net.2",
    "*single_blocks.*.attn.to_k",
    "*single_blocks.*.attn.to_q",
    "*single_blocks.*.attn.to_v",
    "*single_blocks.*.proj_out",
]

_H3_PATTERNS = [
    "*transformer_blocks.*.attn.to_q",
    "*transformer_blocks.*.attn.to_k",
    "*transformer_blocks.*.attn.to_v",
    "*transformer_blocks.*.attn.to_out.0",
    "*transformer_blocks.*.ff.net.0.proj",
    "*transformer_blocks.*.ff.net.2",
]

_MODEL_PATTERNS = {
    # None → attach to ALL Linear modules (musubi-aligned: KREA2_TARGET_REPLACE_MODULES=None)
    "krea2": None,
    "qwen": _QWEN_PATTERNS,
    "flux": _FLUX_PATTERNS,
    "minimax_h3": _H3_PATTERNS,
}


# ── Core math ────────────────────────────────────────────────────────────

def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    """Decompose dimension into two factors closest to `factor`.

    Returns (m, n) where m * n == dimension and m <= n.
    Examples: factorization(512, 4) → (4, 128)
              factorization(1024, 8) → (8, 128)
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


# ── LokrLayer: per-layer adapter ─────────────────────────────────────────

class LokrLayer(nn.Module):
    """Single LoKR adapter for one nn.Linear layer.

    Computes delta = kron(W1, W2) * scale WITHOUT materializing the full
    Kronecker product. Uses the reshaped-matmul decomposition:

        x @ kron(W1, W2).T
        = reshape(x, [B, in1, in2]) → matmul W2.T → matmul W1 → flatten

    Peak VRAM: O(batch * factor * max_dim) instead of O(out * in).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        factor: int = -1,
        multiplier: float = 1.0,
        decompose_both: bool = False,
        full_rank: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier

        # Factorize dimensions: out = out1 * out2, in = in1 * in2
        self.out1, self.out2 = factorization(out_features, factor)
        self.in1, self.in2 = factorization(in_features, factor)

        # ── W1: "small" factor [out1, in1] ───────────────────────────
        # With factor=-1 (balanced) this is e.g. [64, 64] for 6144-dim layers;
        # with factor=4-8 it shrinks to [4, 4] or [8, 8].
        if not full_rank and decompose_both and rank < max(self.out1, self.in1) / 2:
            # Low-rank decomposition: W1 = w1_a @ w1_b
            self.lokr_w1_a = nn.Parameter(torch.empty(self.out1, rank))
            self.lokr_w1_b = nn.Parameter(torch.empty(rank, self.in1))
            self.use_w1 = False
        else:
            self.lokr_w1 = nn.Parameter(torch.empty(self.out1, self.in1))
            self.use_w1 = True

        # ── W2: "large" factor [out2, in2] ───────────────────────────
        # This is the bigger matrix. Decompose if rank is small enough.
        # full_rank forces the full matrix (musubi `lokr_full_rank`), same as
        # a huge rank satisfying rank >= max(out2, in2)/2.
        if full_rank or rank >= max(self.out2, self.in2) / 2:
            self.lokr_w2 = nn.Parameter(torch.empty(self.out2, self.in2))
            self.use_w2 = True
        else:
            self.lokr_w2_a = nn.Parameter(torch.empty(self.out2, rank))
            self.lokr_w2_b = nn.Parameter(torch.empty(rank, self.in2))
            self.use_w2 = False

        # Scale: alpha / rank (standard LoRA scaling).
        # musubi-aligned: full-matrix W2 mode forces alpha = rank -> scale = 1.0
        # (musubi lokr.py: "if both w1 and w2 are full matrices, use scale = 1").
        alpha_eff = rank if self.use_w2 else alpha
        self.scale = alpha_eff / rank if rank > 0 else 1.0

        # Register alpha as buffer for checkpoint compatibility
        # (full-W2 mode stores alpha = rank, matching musubi checkpoints)
        self.register_buffer("alpha", torch.tensor(alpha_eff))

        self._init_weights()

    def _init_weights(self):
        """Initialize so that initial delta = 0 (zero-init the "output" side)."""
        if self.use_w1:
            nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        if self.use_w2:
            # Zero-init W2 → initial delta is zero
            nn.init.constant_(self.lokr_w2, 0)
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            nn.init.constant_(self.lokr_w2_b, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoKR delta: x @ kron(W1, W2).T * scale * multiplier.

        Uses reshaped matmul — NEVER materializes the full [out, in] kron.

        Args:
            x: Input tensor [batch, in_features] (or [batch, seq, in_features]).
        Returns:
            Delta tensor with same shape as x (last dim = out_features).
        """
        orig_shape = x.shape
        # Flatten to 2D for matmul: [B, in_features]
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)
        batch = x.shape[0]

        # Reshape input: [B, in1, in2]
        x_r = x.view(batch, self.in1, self.in2)

        # Step 1: Apply W2 along the in2 dimension
        # tmp = x_r @ W2.T → [B, in1, out2]
        if self.use_w2:
            tmp = x_r @ self.lokr_w2.T  # [B, in1, out2]
        else:
            # W2 = w2_a @ w2_b, so W2.T = w2_b.T @ w2_a.T
            # x_r @ w2_b.T → [B, in1, rank], then @ w2_a.T → [B, in1, out2]
            tmp = x_r @ self.lokr_w2_b.T  # [B, in1, rank]
            tmp = tmp @ self.lokr_w2_a.T  # [B, in1, out2]

        # Step 2: Apply W1 along the in1 dimension
        # y_r = W1 @ tmp → [B, out1, out2]
        # torch.matmul([out1, in1], [B, in1, out2]) → [B, out1, out2]
        if self.use_w1:
            y_r = torch.matmul(self.lokr_w1, tmp)  # [B, out1, out2]
        else:
            # W1 = w1_a @ w1_b
            # w1_b @ tmp → [B, rank, out2], then w1_a @ that → [B, out1, out2]
            tmp2 = torch.matmul(self.lokr_w1_b, tmp)  # [B, rank, out2]
            y_r = torch.matmul(self.lokr_w1_a, tmp2)  # [B, out1, out2]

        # Flatten back: [B, out_features]
        y = y_r.reshape(batch, self.out_features)

        # Apply scale and multiplier
        y = y * (self.scale * self.multiplier)

        # Restore original shape
        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], self.out_features)

        return y


# ── LokrNetwork: manages all LokrLayers ──────────────────────────────────

class LokrNetwork(nn.Module):
    """Container for all LoKR adapter layers.

    Discovers target Linear modules via fnmatch patterns, creates a LokrLayer
    for each, and registers forward hooks to add the delta to the base output.

    Compatible with BouncingOffloader: hooks run AFTER the base forward
    (which BouncingOffloader patches), so no conflict.
    """

    def __init__(self, config: LokrConfig):
        super().__init__()
        self.config = config
        self.layers: Dict[str, LokrLayer] = nn.ModuleDict()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._target_names: List[str] = []

    @property
    def num_modules(self) -> int:
        return len(self.layers)

    def attach(self, model: nn.Module) -> "LokrNetwork":
        """Discover target layers and attach forward hooks.

        Args:
            model: The frozen transformer model.
        Returns:
            self (for chaining).
        """
        patterns = self.config.target_modules
        if patterns is None:
            patterns = _MODEL_PATTERNS.get(self.config.model_type, _KREA2_PATTERNS)
        # patterns=None (krea2 default) → attach to ALL Linear modules,
        # matching musubi's KREA2_TARGET_REPLACE_MODULES=None behavior.

        # Discover matching Linear modules
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if patterns is not None and not any(fnmatch.fnmatch(name, p) for p in patterns):
                continue

            # Create LokrLayer for this Linear
            layer = LokrLayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                factor=self.config.factor,
                multiplier=self.config.multiplier,
                decompose_both=self.config.decompose_both,
                full_rank=self.config.full_rank,
            )
            # Sanitize name for ModuleDict (replace dots with underscores)
            safe_name = name.replace(".", "_")
            self.layers[safe_name] = layer
            self._target_names.append(name)

            # Register forward hook on the Linear module
            hook = module.register_forward_hook(
                self._make_hook(layer)
            )
            self._hooks.append(hook)

        logger.info(
            f"LokrNetwork attached: {len(self.layers)} modules, "
            f"rank={self.config.rank}, alpha={self.config.alpha}, "
            f"factor={self.config.factor}, model_type={self.config.model_type}, "
            f"full_rank={self.config.full_rank}"
        )
        return self

    def _make_hook(self, layer: LokrLayer):
        """Create a forward hook that adds the LoKR delta to the output."""
        def hook_fn(module, input, output):
            # input[0] is the input tensor to the Linear
            x = input[0]
            delta = layer(x)
            return output + delta.to(output.dtype)
        return hook_fn

    def detach(self):
        """Remove all forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def save_weights(self, path: str, dtype: torch.dtype = torch.bfloat16):
        """Save all LoKR parameters to a safetensors file.

        Key format (ComfyUI-compatible):
            lycoris_{layer_name}.lokr_w1
            lycoris_{layer_name}.lokr_w2
            lycoris_{layer_name}.alpha
        """
        from safetensors.torch import save_file

        state = {}
        for safe_name, layer in self.layers.items():
            prefix = f"lycoris_{safe_name}"
            if layer.use_w1:
                state[f"{prefix}.lokr_w1"] = layer.lokr_w1.data.to(dtype)
            else:
                state[f"{prefix}.lokr_w1_a"] = layer.lokr_w1_a.data.to(dtype)
                state[f"{prefix}.lokr_w1_b"] = layer.lokr_w1_b.data.to(dtype)
            if layer.use_w2:
                state[f"{prefix}.lokr_w2"] = layer.lokr_w2.data.to(dtype)
            else:
                state[f"{prefix}.lokr_w2_a"] = layer.lokr_w2_a.data.to(dtype)
                state[f"{prefix}.lokr_w2_b"] = layer.lokr_w2_b.data.to(dtype)
            state[f"{prefix}.alpha"] = layer.alpha

        save_file(state, path, metadata={"source": "UnifiedTrainer", "network": "lokr"})
        logger.info(f"Saved LoKR weights: {path} ({len(self.layers)} modules, {len(state)} tensors)")

    def load_weights(self, path: str):
        """Load LoKR parameters from a safetensors file."""
        from safetensors.torch import load_file

        state = load_file(path)
        loaded = 0
        for safe_name, layer in self.layers.items():
            prefix = f"lycoris_{safe_name}"
            if layer.use_w1:
                key = f"{prefix}.lokr_w1"
                if key in state:
                    layer.lokr_w1.data.copy_(state[key])
                    loaded += 1
            else:
                for suffix in ("lokr_w1_a", "lokr_w1_b"):
                    key = f"{prefix}.{suffix}"
                    if key in state:
                        getattr(layer, suffix).data.copy_(state[key])
                        loaded += 1
            if layer.use_w2:
                key = f"{prefix}.lokr_w2"
                if key in state:
                    layer.lokr_w2.data.copy_(state[key])
                    loaded += 1
            else:
                for suffix in ("lokr_w2_a", "lokr_w2_b"):
                    key = f"{prefix}.{suffix}"
                    if key in state:
                        getattr(layer, suffix).data.copy_(state[key])
                        loaded += 1

        logger.info(f"Loaded LoKR weights: {path} ({loaded} tensors into {len(self.layers)} modules)")

    def extra_repr(self) -> str:
        return (
            f"modules={len(self.layers)}, rank={self.config.rank}, "
            f"alpha={self.config.alpha}, factor={self.config.factor}, "
            f"full_rank={self.config.full_rank}"
        )


# ── Entry point ──────────────────────────────────────────────────────────

def apply_lokr(model: nn.Module, config: LokrConfig) -> LokrNetwork:
    """Create and attach a LoKR network to the model.

    This is the main entry point, replacing the lycoris-lora library.
    The returned LokrNetwork owns all trainable adapter parameters.

    Args:
        model: The frozen transformer (base weights already on device).
        config: LoKR configuration.

    Returns:
        LokrNetwork instance. Call .parameters() for optimizer,
        .save_weights() / .load_weights() for checkpointing.
    """
    network = LokrNetwork(config)
    network.attach(model)

    num_params = sum(p.numel() for p in network.parameters())
    logger.info(
        f"LoKR applied (self-contained): "
        f"modules={network.num_modules}, params={num_params:,}, "
        f"rank={config.rank}, factor={config.factor}, "
        f"full_rank={config.full_rank}"
    )
    return network
