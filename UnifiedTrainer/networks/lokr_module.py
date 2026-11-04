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

Target selection:
    lokr_target_modules=null (default) → ALL nn.Linear modules get adapters
    (musubi krea2 convention, now the default for every model_type).
    An explicit pattern list restricts targeting via fnmatch; the per-model
    preset lists below (_MODEL_PATTERNS) are reference-only — copy the list
    you want into lokr_target_modules to reproduce the old preset behaviour:
    qwen         -> _QWEN_PATTERNS         (Qwen-Image, models/qwen_image)
    qwen21       -> _QWEN21_PATTERNS       (Qwen-Image 2.1, models/qwen_image21)
    flux         -> _FLUX_PATTERNS
    flux2_klein  -> _FLUX2_KLEIN_PATTERNS
    minimax_h3   -> _H3_PATTERNS
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


# ── MOSS target presets (mirror of KohakuMossTrainer) ───────────────────
# `src/kohakumoss/adapters/build/presets.py` in the reference trainer defines the
# LyCORIS preset expressions below.  `attn-mlp` is that trainer's default and the
# one the recipes use; our old "all nn.Linear" default additionally adapted
# `time_embedding.*`, `shared_block_modulation`, `video_head_modulation`,
# `video_embedder`, `text_embedder.*` and `final_linear_video` - the layers that
# generate every block's (shift, scale, gate) modulation.  Adapting those with a
# full-rank LoKR rewrites the conditioning path itself, which shows up as blurry,
# desaturated samples and garbled text.
MOSS_TARGET_PRESETS: dict[str, list[str]] = {
    # layers.N.proj.(q|k|v|o) + layers.N.proj.(ffn_up_gate|ffn_down) -> 252 modules
    "attn-mlp": [
        "layers.*.proj.q", "layers.*.proj.k", "layers.*.proj.v", "layers.*.proj.o",
        "layers.*.proj.ffn_up_gate", "layers.*.proj.ffn_down",
    ],
    # joint attention only -> 168 modules
    "attn": [
        "layers.*.proj.q", "layers.*.proj.k", "layers.*.proj.v", "layers.*.proj.o",
    ],
    # attn-mlp + the text refiner blocks -> 262 modules
    "full-block": [
        "layers.*.proj.q", "layers.*.proj.k", "layers.*.proj.v", "layers.*.proj.o",
        "layers.*.proj.ffn_up_gate", "layers.*.proj.ffn_down",
        "text_embedder.blocks.*.qkv", "text_embedder.blocks.*.proj",
        "text_embedder.blocks.*.ffn_gate", "text_embedder.blocks.*.ffn_up",
        "text_embedder.blocks.*.ffn_down",
    ],
}


def resolve_target_modules(spec):
    """Expand a MOSS preset name (or list of fnmatch patterns) for ``attach``."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec in ("all", "*"):
            return None
        if spec in MOSS_TARGET_PRESETS:
            return list(MOSS_TARGET_PRESETS[spec])
        return [spec]
    return list(spec)


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class LokrConfig:
    """LoKR network configuration.

    Attributes:
        rank: Low-rank dimension for W2 decomposition (and W1 if decompose_both).
        alpha: Scaling factor. Effective scale = alpha / rank; in full-W2 mode
        alpha is forced to rank -> scale = 1.0 (musubi-aligned).
        factor: Factorization factor for splitting dimensions (-1 = auto).
        model_type: Informational only (musubi-config compatibility).
            No longer selects target modules — `target_modules` owns that.
        target_modules: Optional fnmatch patterns restricting which nn.Linear
            modules get adapters. None (default) = attach to ALL nn.Linear
            modules (musubi krea2 convention). See the module docstring.
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
    # PiSSA-style initialisation (Meng et al., 2404.02948): start the adapter at
    # the dominant Kronecker component of the frozen weight instead of zeros, and
    # subtract it from the base so the total weight is unchanged at step 0.
    pissa_init: bool = False

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

# Flux2 Klein 4B (models/flux2_klein/transformer_flux2.py):
# - double-stream blocks: Flux2Attention (to_q/to_k/to_v/to_out.0 + add_*_proj)
#   and Flux2FeedForward (ff.linear_in/linear_out — NOT ff.net.*)
# - single-stream blocks: Flux2ParallelSelfAttention with FUSED qkv+mlp
#   (attn.to_qkv_mlp_proj) + attn.to_out
# - patch embed / text embed / final proj: x_embedder / context_embedder / proj_out
# Modulation (double_stream_modulation_*/single_stream_modulation .linear) and
# norms are intentionally NOT targeted.
_FLUX2_KLEIN_PATTERNS = [
    "*transformer_blocks.*.attn.to_k",
    "*transformer_blocks.*.attn.to_q",
    "*transformer_blocks.*.attn.to_v",
    "*transformer_blocks.*.attn.to_out.0",
    "*transformer_blocks.*.attn.add_k_proj",
    "*transformer_blocks.*.attn.add_q_proj",
    "*transformer_blocks.*.attn.add_v_proj",
    "*transformer_blocks.*.attn.to_add_out",
    "*transformer_blocks.*.ff.linear_in",
    "*transformer_blocks.*.ff.linear_out",
    "*single_blocks.*.attn.to_qkv_mlp_proj",
    "*single_blocks.*.attn.to_out",
    "*x_embedder",
    "*context_embedder",
    "*proj_out",
]

# FLUX.3 klein (models/flux3_klein/transformer_flux3.py):
# - per-mode blocks (Flux3ModeTransformerBlock): Flux3Attention
#   (to_q/to_k/to_v/to_out.0) + Flux3FeedForward (ff.linear_in/linear_out —
#   same naming as Flux2, NOT ff.net.*)
# - joint blocks (Flux3SingleTransformerBlock): identical sub-module names
# - the blocks live in three ModuleLists/ModuleDicts:
#       txt_mode_blocks.*, mode_blocks.image.*, mode_blocks.image_cond.*,
#       single_transformer_blocks.*
# - patch embed / text embed / final proj: x_embedders.{image,image_cond} /
#   context_embedder / proj_out
# Modulation (early_stream_modulations.*/single_stream_modulations.*.linear),
# timestep embedders and norms are intentionally NOT targeted.
_FLUX3_KLEIN_PATTERNS = [
    "*txt_mode_blocks.*.attn.to_k",
    "*txt_mode_blocks.*.attn.to_q",
    "*txt_mode_blocks.*.attn.to_v",
    "*txt_mode_blocks.*.attn.to_out.0",
    "*txt_mode_blocks.*.ff.linear_in",
    "*txt_mode_blocks.*.ff.linear_out",
    "*mode_blocks.image.*.attn.to_k",
    "*mode_blocks.image.*.attn.to_q",
    "*mode_blocks.image.*.attn.to_v",
    "*mode_blocks.image.*.attn.to_out.0",
    "*mode_blocks.image.*.ff.linear_in",
    "*mode_blocks.image.*.ff.linear_out",
    "*mode_blocks.image_cond.*.attn.to_k",
    "*mode_blocks.image_cond.*.attn.to_q",
    "*mode_blocks.image_cond.*.attn.to_v",
    "*mode_blocks.image_cond.*.attn.to_out.0",
    "*mode_blocks.image_cond.*.ff.linear_in",
    "*mode_blocks.image_cond.*.ff.linear_out",
    "*single_transformer_blocks.*.attn.to_k",
    "*single_transformer_blocks.*.attn.to_q",
    "*single_transformer_blocks.*.attn.to_v",
    "*single_transformer_blocks.*.attn.to_out.0",
    "*single_transformer_blocks.*.ff.linear_in",
    "*single_transformer_blocks.*.ff.linear_out",
    "*x_embedders.*",
    "*context_embedder",
    "*proj_out",
]

# Qwen-Image 2.1 (models/qwen_image21/transformer_qwenimage21.py):
# single-stream blocks only — attn.to_q/k/v/out.0 plus the SwiGLU img_mlp
# (proj/gate_layer/out). Reference-only preset: by default ALL nn.Linear
# modules (incl. img_in / modulation.1 / norm_out.linear / proj_out /
# txt_in.*) are targeted; copy this list into lokr_target_modules to train
# only the in-block subset.
_QWEN21_PATTERNS = [
    "*transformer_blocks.*.attn.to_k",
    "*transformer_blocks.*.attn.to_q",
    "*transformer_blocks.*.attn.to_v",
    "*transformer_blocks.*.attn.to_out.0",
    "*transformer_blocks.*.img_mlp.proj",
    "*transformer_blocks.*.img_mlp.gate_layer",
    "*transformer_blocks.*.img_mlp.out",
]

# Reference-only presets — NOT auto-applied by attach() anymore (lokr_target_modules
# = null now means ALL nn.Linear modules for every model_type, krea2-style).
# Copy the list you want into lokr_target_modules to restrict targeting.
_MODEL_PATTERNS = {
    # krea2 has no preset: its null already meant ALL (musubi convention).
    "krea2": None,
    "qwen": _QWEN_PATTERNS,
    "qwen21": _QWEN21_PATTERNS,
    "flux": _FLUX_PATTERNS,
    "flux2_klein": _FLUX2_KLEIN_PATTERNS,
    "flux3_klein": _FLUX3_KLEIN_PATTERNS,
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

        # Dtype-safe entry: LoKR params are fp32 (trainable-param convention),
        # but the base model may feed bf16 activations — e.g. NF4 runs WITHOUT
        # accelerate's autocast wrapper (the block_swap path skips model
        # prepare). Compute the delta in the param dtype; the caller's hook
        # casts the result back to the base output dtype.
        param_dtype = (
            self.lokr_w1.dtype if self.use_w1 else self.lokr_w1_a.dtype
        )
        if x.dtype != param_dtype:
            x = x.to(param_dtype)

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

def pissa_init_lokr_layer(layer: "LokrLayer", weight: torch.Tensor) -> bool:
    """Initialise a LoKR layer from the dominant Kronecker component of *weight*.

    PiSSA (2404.02948) shows that starting LoRA at "noise & zero" wastes the first
    hundreds of steps re-learning the principal directions of the frozen weight;
    initialising the adapter with the principal components (and subtracting them
    from the base) converges markedly faster.

    For LoKR the update is ``delta = kron(w1, w2) * scale``.  The best single
    Kronecker factor of a matrix is given by the leading singular pair of the
    *reshuffled* matrix (Van Loan & Pitsianis): with
    ``R[(i,j),(p,q)] = W[i*out2+p, j*in2+q]``, the top singular triplet
    ``(u, s, v)`` yields ``kron(w1, w2) ≈ s·u⊗v``.

    The component is subtracted from the frozen weight so the layer's effective
    weight is unchanged at step 0 (PiSSA's residual trick), done in the reshaped
    view so the full Kronecker product is never materialised.
    """
    if not (layer.use_w1 and layer.use_w2):
        # Low-rank factorisations: keep the zero init (documented limitation).
        return False
    with torch.no_grad():
        # 24 GB budget: run the SVD on the CPU.  For an 8192x8192 layer the
        # float32 copy + reshuffled matrix + U factor peak at ~0.8 GB, which we
        # do not want to add to the GPU's steady-state usage (fp8 base ~13.3 GB
        # + activations).  CPU temporaries are free.
        #
        # torchao's Float8Tensor (weight-only quantization) intercepts ``view``/
        # ``reshape`` through __torch_dispatch__ and raises
        # "Only support reshaping from 2D to 3D or from 3D to 2D", so dequantize
        # into a *plain* tensor first, then work on it.
        src = weight
        if hasattr(src, "dequantize"):
            try:
                src = src.dequantize()
            except Exception:  # noqa: BLE001 - not a torchao tensor after all
                pass
        w_cpu = torch.empty(src.shape, dtype=torch.float32)
        w_cpu.copy_(src.detach().to("cpu", torch.float32))
        out1, out2, in1, in2 = layer.out1, layer.out2, layer.in1, layer.in2
        try:
            w4 = w_cpu.view(out1, out2, in1, in2)                # W[i*out2+p, j*in2+q]
            r = w4.permute(0, 2, 1, 3).reshape(out1 * in1, out2 * in2)
            u, s, vh = torch.linalg.svd(r, full_matrices=False)
            s0 = s[0]
            if not torch.isfinite(s0) or s0 <= 0:
                return False
            w1 = (u[:, 0] * s0.sqrt()).reshape(out1, in1)
            w2 = (vh[0, :] * s0.sqrt()).reshape(out2, in2)
            layer.lokr_w1.data.copy_(w1.to(layer.lokr_w1.device, layer.lokr_w1.dtype))
            layer.lokr_w2.data.copy_(w2.to(layer.lokr_w2.device, layer.lokr_w2.dtype))
            del u, s, vh, r, w4, w_cpu

            # PiSSA residual: W <- W - scale * kron(w1, w2).  Skipped for
            # quantization-wrapped weights (torchao Float8Tensor / qbn), whose
            # storage cannot be written elementwise; the adapter still starts at
            # the principal component, just without the "total weight unchanged"
            # property (the dominant component is ~13% of ||W||).
            is_plain_float = (
                type(weight).__module__.split(".")[0] == "torch"
                and weight.is_floating_point()
                and weight.is_contiguous()
            )
            if not is_plain_float:
                logger.info(
                    "pissa_init: base weight is quantized/wrapped "
                    f"({type(weight).__name__}); adapter initialised from the "
                    "principal Kronecker component, residual not subtracted")
                return True
            delta = (
                layer.lokr_w1.data.float().view(out1, 1, in1, 1)
                * layer.lokr_w2.data.float().view(1, out2, 1, in2)
                * layer.scale
            ).reshape(layer.out_features, layer.in_features)
            weight.sub_(delta.to(weight.device, weight.dtype))
            del delta
            return True
        except Exception:  # noqa: BLE001 - never fail a run because of an init
            logger.warning("pissa_init: falling back to zero init (SVD failed)",
                           exc_info=True)
            layer.lokr_w1.data.zero_()
            layer.lokr_w2.data.zero_()
            return False


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
        # patterns=None (the default) → attach to ALL nn.Linear modules,
        # matching musubi's KREA2_TARGET_REPLACE_MODULES=None behavior
        # (krea2 convention, now the default for every model_type).
        # To restrict, set lokr_target_modules to explicit fnmatch patterns;
        # the preset lists below (_MODEL_PATTERNS) are reference-only and can
        # be copied into the config for that purpose.

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
            if getattr(self.config, "pissa_init", False):
                pissa_init_lokr_layer(layer, module.weight.data)
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
        """Load LoKR parameters from a safetensors file.

        Accepts BOTH key layouts:

        * native  -- ``lycoris_<diffusers module, dots->underscores>.<suffix>``
        * ComfyUI -- ``lora_unet_<comfy module, dots->underscores>.<suffix>``
          or ``diffusion_model.<comfy module>.<suffix>`` (the form
          :mod:`UnifiedTrainer.networks.lokr_convert` writes next to every
          checkpoint), so a converted file can be resumed for further training
          without a manual step.  Detection is by key prefix; the translation
          is name-only (plus the adaLN row permutation), so it is lossless.
        """
        from safetensors.torch import load_file

        state = load_file(path)

        if any(k.startswith(("lora_unet_", "diffusion_model.")) for k in state):
            from UnifiedTrainer.networks.lokr_convert import (
                convert_moss_state_dict, convert_state_dict,
            )

            # `_target_names` are the exact dotted module paths this network
            # attached to, so the name match is unambiguous (flattening is
            # lossy and must never be inverted by string surgery).
            targets = list(self._target_names)
            is_moss = any(t.startswith("layers.") for t in targets)
            if is_moss:
                # MOSS keeps the official module names in ComfyUI, so the
                # conversion is a plain prefix rename (see convert_moss_state_dict).
                state = convert_moss_state_dict(state, to="unifiedtrainer")
            else:
                state = convert_state_dict(
                    state, to="unifiedtrainer", native_modules=targets
                )
            logger.info(
                f"LoKR checkpoint detected as ComfyUI "
                f"{'MOSS' if is_moss else 'FLUX.3'} format; "
                f"converted {len(state)} keys back to native names for resume"
            )

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
