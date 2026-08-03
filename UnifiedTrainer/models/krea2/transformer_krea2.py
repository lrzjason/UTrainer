# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import apply_lora_scale, logging
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from UnifiedTrainer.utils.block_swap import ModelOffloader, synchronize_device


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: the effective multiplier is `1 + weight`, matching the Krea 2 checkpoint
    format. The activations are upcast so the normalization runs in float32; the scale weight is kept in float32 by the
    model's `_keep_in_fp32_modules`."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Defensive device guard: BouncingOffloader creates a mixed-device model
        # (Linear weights on CPU, norms on GPU).  If a CPU tensor reaches this
        # norm, the F.rms_norm call crashes with a device-mismatch error.
        # Explicitly move to the weight's device to guarantee correctness.
        if hidden_states.device != self.weight.device:
            hidden_states = hidden_states.to(self.weight.device)
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(hidden_states.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps)
        return hidden_states.to(dtype)


class Krea2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "Krea2Attention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            enable_gqa=attn.num_heads != attn.num_kv_heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)


class Krea2Attention(nn.Module, AttentionModuleMixin):
    """Self-attention with grouped-query projections, q/k RMSNorm, rotary embeddings and a sigmoid output gate."""

    _default_processor_cls = Krea2AttnProcessor
    _available_processors = [Krea2AttnProcessor]

    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int | None = None, eps: float = 1e-5, processor=None
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.use_bias = False

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k in kwargs if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Krea2SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm transformer block (no rotary embeddings, no time modulation) used by the text fusion stage."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    """Fuses the stack of tapped text-encoder hidden states into a single sequence of text features.

    Two `layerwise_blocks` attend across the `num_text_layers` axis independently for every token, a linear `projector`
    collapses that axis, and two `refiner_blocks` attend across the token sequence.
    """

    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(self, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2).contiguous()
        # Reshape 4D -> 3D for bitsandbytes Linear8bitLt (only supports 2D/3D)
        _b, _s, _d, _nl = hidden_states.shape
        hidden_states = hidden_states.reshape(_b * _s, _d, _nl)
        hidden_states = self.projector(hidden_states).squeeze(-1)
        hidden_states = hidden_states.reshape(_b, _s, _d)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # When ``temb`` is a 3-tuple (vec, refvec, split), the block applies
        # per-span timestep modulation: ``vec`` modulates tokens [:split]
        # (generation targets at training timestep) and ``refvec`` modulates
        # tokens [split:] (reference tokens at t=0).  This is the RoPE frame-
        # index design — references are sequence tokens, not channel-concat.
        if isinstance(temb, tuple):
            temb_mod, temb_mod_zero, split = temb
            mod = temb_mod.unflatten(-1, (6, -1)) + self.scale_shift_table
            mod_zero = temb_mod_zero.unflatten(-1, (6, -1)) + self.scale_shift_table
            prescale, preshift, pregate, postscale, postshift, postgate = mod.unbind(-2)
            z_prescale, z_preshift, z_pregate, z_postscale, z_postshift, z_postgate = mod_zero.unbind(-2)

            def _mod_norm(h, scale, shift, z_scale, z_shift):
                return torch.cat(
                    [
                        (1.0 + scale) * self.norm1(h[:, :split]) + shift,
                        (1.0 + z_scale) * self.norm1(h[:, split:]) + z_shift,
                    ],
                    dim=1,
                )

            def _gate(h, g, z_g):
                return torch.cat([g * h[:, :split], z_g * h[:, split:]], dim=1)

            attn_out = self.attn(
                _mod_norm(hidden_states, prescale, preshift, z_prescale, z_preshift),
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + _gate(attn_out, pregate, z_pregate)

            def _mod_norm2(h, scale, shift, z_scale, z_shift):
                return torch.cat(
                    [
                        (1.0 + scale) * self.norm2(h[:, :split]) + shift,
                        (1.0 + z_scale) * self.norm2(h[:, split:]) + z_shift,
                    ],
                    dim=1,
                )

            ff_out = self.ff(
                _mod_norm2(hidden_states, postscale, postshift, z_postscale, z_postshift)
            )
            hidden_states = hidden_states + _gate(ff_out, postgate, z_postgate)
            return hidden_states

        # Single-span path (original): all tokens share the same timestep modulation.
        # temb: (B, 1, 6 * hidden_size), shared across all blocks; each block only learns an additive table.
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding (cos-first, input scaled by 1000) followed by a two-layer MLP.

    Keeps the sequence dimension at size 1 so the per-block modulations broadcast over tokens.
    """

    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2TextProjection(nn.Module):
    """Projects the fused text features into the transformer width."""

    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    """Final adaptive RMSNorm and output projection. Kept as one module (and in `_no_split_modules`) so the learned
    modulation table, norm and projection stay co-located under device-mapped inference."""

    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


# Copied from diffusers.models.transformers.transformer_flux.FluxPosEmbed with FluxPosEmbed->Krea2RotaryPosEmbed
class Krea2RotaryPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, ids.device)
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Krea2Transformer2DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin):
    r"""
    The single-stream MMDiT flow-matching backbone used by the Krea 2 pipeline.

    Text conditioning enters as a stack of hidden states tapped from several layers of a multimodal text encoder. A
    small text-fusion transformer collapses the layer axis and refines the token sequence; the result is concatenated
    with the patchified image latents into a single `[text, image]` sequence processed by the transformer blocks. The
    timestep conditions every block through one shared modulation vector plus per-block learned tables.

    Args:
        in_channels (`int`, defaults to 64):
            Latent channel count after patchification (`vae_channels * patch_size ** 2`).
        num_layers (`int`, defaults to 28):
            Number of transformer blocks.
        attention_head_dim (`int`, defaults to 128):
            Dimension of each attention head; the total hidden size is `attention_head_dim * num_attention_heads`.
        num_attention_heads (`int`, defaults to 48):
            Number of query heads.
        num_key_value_heads (`int`, defaults to 12):
            Number of key/value heads for grouped-query attention.
        intermediate_size (`int`, defaults to 16384):
            Feed-forward hidden size of the SwiGLU MLP inside each block.
        timestep_embed_dim (`int`, defaults to 256):
            Width of the sinusoidal timestep embedding before its MLP.
        text_hidden_dim (`int`, defaults to 2560):
            Hidden size of the text encoder whose hidden states are consumed.
        num_text_layers (`int`, defaults to 12):
            Number of tapped text-encoder hidden states stacked per token.
        text_num_attention_heads (`int`, defaults to 20):
            Number of query heads in the text fusion blocks.
        text_num_key_value_heads (`int`, defaults to 20):
            Number of key/value heads in the text fusion blocks.
        text_intermediate_size (`int`, defaults to 6912):
            Feed-forward hidden size of the SwiGLU MLP inside the text fusion blocks.
        num_layerwise_text_blocks (`int`, defaults to 2):
            Number of text fusion blocks applied across the tapped-layer axis (per token).
        num_refiner_text_blocks (`int`, defaults to 2):
            Number of text fusion blocks applied across the token sequence.
        axes_dims_rope (`tuple[int, int, int]`, defaults to `(32, 48, 48)`):
            Head-dim split across the (t, h, w) rotary position axes.
        rope_theta (`float`, defaults to 1000.0):
            Base used by the rotary position embedding.
        norm_eps (`float`, defaults to 1e-5):
            Epsilon used by all RMSNorm modules.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Krea2TransformerBlock", "Krea2TextFusionBlock", "Krea2FinalLayer"]
    _repeated_blocks = ["Krea2TransformerBlock"]
    _keep_in_fp32_modules = ["norm", "norm1", "norm2", "norm_q", "norm_k"]
    _skip_layerwise_casting_patterns = ["time_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim={attention_head_dim}"
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, eps=norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=in_channels, eps=norm_eps)

        # Block swap state (for VRAM optimization during training)
        self.blocks_to_swap = 0
        self._offloader = None

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        """Enable block swap: offload ``num_blocks`` transformer blocks to CPU.

        The blocks are swapped to GPU on-demand during forward/backward.
        This reduces peak VRAM at the cost of slower training.
        """
        if num_blocks <= 0:
            return
        from UnifiedTrainer.utils.block_swap import ModelOffloader

        self.blocks_to_swap = num_blocks
        num_total = len(self.transformer_blocks)
        self._offloader = ModelOffloader(
            self.transformer_blocks, num_total, num_blocks, device
        )
        logger.info(
            f"Krea2 block swap: {num_blocks}/{num_total} blocks offloaded"
        )

    def prepare_block_swap_before_forward(self):
        """Move first N blocks to GPU and last N blocks to CPU before the forward pass."""
        if self._offloader is not None:
            self._offloader.prepare_block_devices_before_forward(self.transformer_blocks)

    def disable_block_swap(self):
        """Temporarily disable block swap and move all blocks to GPU.

        Call before inference (no gradients) to avoid per-step CPU↔GPU
        transfers.  Use :meth:`restore_block_swap` to restore the offloader.

        ``nn.Module.to(device)`` properly handles ``Params4bit`` (NF4):
        it moves both the quantized data and ``quant_state`` to GPU.
        Manual ``param.cuda()`` does NOT work in-place and must be avoided.
        """
        if self._offloader is None:
            return  # already disabled
        self._saved_offloader = self._offloader
        self._offloader = None
        device = self._saved_offloader.device
        # Move entire model to GPU -Module.to() handles Params4bit properly
        self.to(device)
        synchronize_device(device)

    def restore_block_swap(self):
        """Restore block swap after :meth:`disable_block_swap`."""
        offloader = getattr(self, "_saved_offloader", None)
        if offloader is None:
            return  # nothing to restore
        self._offloader = offloader
        self._saved_offloader = None
        # Reset block positions (first N on GPU, last N on CPU)
        self._offloader.prepare_block_devices_before_forward(self.transformer_blocks)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        reflen: int = 0,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        r"""
        Predict the flow-matching velocity for the image tokens.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_seq_len, in_channels)`):
                Packed (patchified) noisy image latents, possibly followed by
                packed reference latents when ``reflen > 0``.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, num_text_layers, text_hidden_dim)`):
                Stack of tapped text-encoder hidden states per token.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Flow-matching time in `[0, 1]` (1 is pure noise, 0 is clean data).
            position_ids (`torch.Tensor` of shape `(text_seq_len + image_seq_len, 3)`):
                `(t, h, w)` rotary coordinates for the combined sequence. Text rows are all-zero; image rows hold the
                latent-grid coordinates.  Reference tokens use positive ``t`` values (matching the ai-toolkit
                ``index_timestep_zero`` convention).
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Boolean mask marking valid text tokens. Pass `None` when every text token is valid.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that, when it contains a `scale` entry, sets the LoRA scale applied to this
                transformer's adapters for the duration of the forward pass.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.
            reflen (`int`, *optional*, defaults to 0):
                Number of reference image tokens appended at the end of ``hidden_states``.
                When ``>0`` the block receives a ``(vec, refvec, split)`` tuple for
                per-span timestep modulation (generation targets at training timestep,
                references at t=0).

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or a `tuple` whose first element is the velocity
            tensor of shape `(batch_size, image_seq_len - reflen, in_channels)`.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}.")

        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        # ── Per-span t=0 modulation for reference tokens ──
        img_noisylen = image_seq_len - reflen
        if reflen > 0:
            t0 = torch.zeros_like(timestep)
            t0_emb = self.time_embed(t0, dtype=hidden_states.dtype)
            t0_mod = self.time_mod_proj(F.gelu(t0_emb, approximate="tanh"))
            split = text_seq_len + img_noisylen
            blockvec = (temb_mod, t0_mod, split)
        else:
            blockvec = temb_mod

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            # Key-padding masks of shape (B, 1, 1, L): padded text tokens are excluded as attention keys everywhere;
            # their own (garbage) lanes are never read back and are dropped at the output slice.
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Pad combined sequence to a multiple of 256 for stable compiled
        # kernel shapes (matches ai-toolkit / reference behaviour). Padding
        # tokens are masked from attention and sliced off at the output, so
        # their values never matter.
        full_len = hidden_states.shape[1]
        pad_len = (-full_len) % 256
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0.0)
            position_ids = F.pad(position_ids, (0, 0, 0, pad_len))

        image_rotary_emb = self.rotary_emb(position_ids)

        for i, block in enumerate(self.transformer_blocks):
            # Block swap: wait for this block to be on GPU
            if self._offloader is not None:
                self._offloader.wait_for_block(i)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, blockvec, image_rotary_emb, attention_mask
                )
            else:
                hidden_states = block(hidden_states, blockvec, image_rotary_emb, attention_mask)

            # Block swap: submit this block for CPU offload
            if self._offloader is not None:
                self._offloader.submit_move_blocks(self.transformer_blocks, i)

        # Output only the generation-target image tokens (not text, not ref).
        hidden_states = hidden_states[:, text_seq_len : text_seq_len + img_noisylen]
        output = self.final_layer(hidden_states, temb)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class BlockSwapKrea2Transformer2DModel(Krea2Transformer2DModel):
    """Krea2 Transformer with block swap support for VRAM optimization.

    T2ITrainer-compatible subclass that adds ``offloader_double`` attribute
    and ``move_to_device_except_swap_blocks`` for training loop alignment.

    This class MUST be used for Krea2 training (even with ``blocks_to_swap=0``)
    to match T2ITrainer's ``BlockSwapKrea2Transformer2DModel`` behaviour.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            timestep_embed_dim=timestep_embed_dim,
            text_hidden_dim=text_hidden_dim,
            num_text_layers=num_text_layers,
            text_num_attention_heads=text_num_attention_heads,
            text_num_key_value_heads=text_num_key_value_heads,
            text_intermediate_size=text_intermediate_size,
            num_layerwise_text_blocks=num_layerwise_text_blocks,
            num_refiner_text_blocks=num_refiner_text_blocks,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
        )
        self.cpu_offload_checkpointing = False
        self.offloader_double = None
        self.num_double_blocks = len(self.transformer_blocks)

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        """Enable block swap using the T2ITrainer-compatible offloader_double."""
        self.blocks_to_swap = num_blocks
        double_blocks_to_swap = num_blocks

        self.offloader_double = ModelOffloader(
            self.transformer_blocks, self.num_double_blocks, double_blocks_to_swap, device
        )
        logger.info(
            f"Krea2 BlockSwap: Swapping {num_blocks} blocks, "
            f"double blocks: {double_blocks_to_swap}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        """Move model to *device* without moving swap-block weights.

        This avoids a temporary VRAM spike by temporarily removing
        ``transformer_blocks`` from the model before calling ``self.to(device)``.
        """
        if self.blocks_to_swap:
            save_transformer_blocks = self.transformer_blocks
            self.transformer_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.transformer_blocks = save_transformer_blocks

    def prepare_block_swap_before_forward(self):
        """Prepare block devices before the forward pass (T2ITrainer-compatible)."""
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.transformer_blocks)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        reflen: int = 0,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        r"""Forward pass with block swap support (T2ITrainer-compatible).

        Uses ``self.blocks_to_swap`` (int) instead of ``self._offloader is not None``
        to gate block-swap operations, matching T2ITrainer's convention.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}."
            )

        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]
        target_image_seq_len = image_seq_len - reflen

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        # Per-token timestep splitting (index_timestep_zero convention):
        # Reference tokens get t=0 modulation; text + target tokens get real timestep.
        block_temb = temb_mod  # (B, 1, 6*hidden) — shared modulation
        if reflen > 0:
            temb_zero = self.time_embed(
                torch.zeros_like(timestep), dtype=hidden_states.dtype
            )
            temb_mod_zero = self.time_mod_proj(
                F.gelu(temb_zero, approximate="tanh")
            )
            split = text_seq_len + target_image_seq_len
            block_temb = (temb_mod, temb_mod_zero, split)

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat(
                [encoder_attention_mask, image_mask], dim=1
            )[:, None, None, :]

        encoder_hidden_states = self.text_fusion(
            encoder_hidden_states, attention_mask=text_attention_mask
        )
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Pad combined sequence to a multiple of 256 for stable compiled kernel shapes.
        full_len = hidden_states.shape[1]
        pad_len = (-full_len) % 256
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0.0)
            position_ids = F.pad(position_ids, (0, 0, 0, pad_len))

        image_rotary_emb = self.rotary_emb(position_ids)

        for i, block in enumerate(self.transformer_blocks):
            # Block swap: wait for this block to be on GPU
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(i)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, block_temb, image_rotary_emb, attention_mask
                )
            else:
                hidden_states = block(
                    hidden_states, block_temb, image_rotary_emb, attention_mask
                )

            # Block swap: submit this block for CPU offload
            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks(self.transformer_blocks, i)

        # Slice off text prefix, 256-padding, and reference tokens from output
        hidden_states = hidden_states[:, text_seq_len : text_seq_len + target_image_seq_len]
        output = self.final_layer(hidden_states, temb)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
