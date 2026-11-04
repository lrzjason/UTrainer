# =============================================================================
# DERIVED CODE - measurement harness in the style of KohakuMoss's tuner.
#
# Original author : Kohaku-Blueleaf - KohakuMoss (https://github.com/KohakuBlueleaf/KohakuMossTrainer)
# Ported into     : UnifiedTrainer, 2026-10-19
#
# The methodology is theirs (`src/kohakumoss_kernels/tuning/`): a candidate is
# selected only from a measurement on this card, per (op, arch, dtype, shape),
# and the eager implementation is always the third column; the report prints
# the numbers instead of asserting a win. The harness itself is new code.
# =============================================================================
"""Benchmark the kernel candidates against eager PyTorch on this machine.

Usage (from the repo root):

    python -m UnifiedTrainer.kernels.bench attention        # attention op, all adapters' shapes
    python -m UnifiedTrainer.kernels.bench block --tokens 1024 2048   # one Krea-2 block, fwd+bwd
    python -m UnifiedTrainer.kernels.bench ops              # fused block ops vs eager vs compile

Every table reports ``eager`` first: a candidate that does not beat it is not
selected (see ``UnifiedTrainer/kernels/__init__.py``), and ``md/14-kernel-
optimizations-applied.md`` records the verdicts this produced.
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch


# ── timing helper ────────────────────────────────────────────────────────


def timeit(fn, steps: int = 10, warmup: int = 4) -> float:
    """Median CUDA-event time of ``fn`` in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(steps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _fmt(ms: float) -> str:
    return f"{ms:8.3f}"


def _ratio(base: float, cand: float) -> str:
    return f"{base / cand:5.2f}x" if cand > 0 else "  n/a"


# ── attention ────────────────────────────────────────────────────────────

# (label, num_heads, num_kv_heads, head_dim, seq_len) per adapter model, from
# each checkpoint's transformer config.
ATTENTION_SHAPES = [
    ("krea2          (48/12 h, 128 d)", 48, 12, 128),
    ("flux2_klein 4B (24/24 h, 128 d)", 24, 24, 128),
    ("flux3_klein    (24/24 h, 128 d)", 24, 24, 128),
    ("qwen_image     (24/24 h, 128 d)", 24, 24, 128),
    ("qwen_image21   (32/32 h, 128 d)", 32, 32, 128),
    ("minimax_h3     (56/56 h, 128 d)", 56, 56, 128),
]


def bench_attention(seq_lens=(1024, 2048, 4608)) -> list[dict]:
    from UnifiedTrainer.kernels import attention as fused_attention
    from UnifiedTrainer.kernels import eager

    rows: list[dict] = []
    for label, heads, kv_heads, dim in ATTENTION_SHAPES:
        for length in seq_lens:
            b = 1
            q = torch.randn(b, length, heads, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(b, length, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(b, length, kv_heads, dim, device="cuda", dtype=torch.bfloat16)

            def run(fn):
                # Reuse the buffers and drop the grads between steps: the timed
                # region must contain the op, not the allocator.
                qq = q.detach().clone().requires_grad_(True)
                kk = k.detach().clone().requires_grad_(True)
                vv = v.detach().clone().requires_grad_(True)

                def inner():
                    out = fn(qq, kk, vv)
                    out.float().sum().backward()
                    qq.grad = kk.grad = vv.grad = None

                return inner

            # (a) what the adapters do without the hook: enable_gqa when GQA
            def legacy(qq, kk, vv):
                import torch.nn.functional as F

                reps = qq.shape[2] // kk.shape[2]
                if reps > 1:
                    # `enable_gqa=True` path, reproduced to measure the trap
                    return F.scaled_dot_product_attention(
                        qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                        enable_gqa=True,
                    ).transpose(1, 2)

                return F.scaled_dot_product_attention(
                    qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)
                ).transpose(1, 2)

            t_legacy = timeit(run(legacy))
            t_sdpa = timeit(run(eager.sdpa_gqa))
            try:
                t_triton = timeit(run(fused_attention))
            except Exception as exc:  # noqa: BLE001
                print(f"  (triton attention unavailable: {type(exc).__name__}: {exc})")
                t_triton = float("nan")
            rows.append(
                dict(label=label, length=length, legacy=t_legacy, sdpa=t_sdpa, triton=t_triton)
            )
            print(
                f"{label} L={length:5d}  legacy {_fmt(t_legacy)} ms  "
                f"sdpa {_fmt(t_sdpa)} ms ({_ratio(t_legacy, t_sdpa)})  "
                f"triton {_fmt(t_triton)} ms ({_ratio(t_legacy, t_triton)})"
            )
            del q, k, v
            gc.collect()
            torch.cuda.empty_cache()
    return rows


# ── one Krea-2 block ─────────────────────────────────────────────────────


def bench_block(tokens=(1024, 2048), include_compile: bool = True, checkpoint: bool = False) -> list[dict]:
    """Forward+backward of one Krea-2 block, base frozen (the LoRA regime).

    Columns: ``legacy`` is the pre-existing call (``enable_gqa=True``, which
    lands on the fp32 math backend here), ``eager`` is the same block after the
    KV-expand fix with the kernel hook disabled, ``kernels`` enables the hook.
    """
    from UnifiedTrainer import kernels
    from UnifiedTrainer.models.krea2 import transformer_krea2 as tx
    from UnifiedTrainer.models.krea2.transformer_krea2 import (
        Krea2RotaryPosEmbed,
        Krea2TransformerBlock,
    )

    legacy_call = _legacy_krea2_processor_call()
    current_call = tx.Krea2AttnProcessor.__call__

    dim, heads, kv_heads, ffn, eps = 6144, 48, 12, 16384, 1e-5
    block = Krea2TransformerBlock(dim, ffn, heads, kv_heads, eps).to("cuda", torch.bfloat16).eval()
    for p in block.parameters():
        p.requires_grad_(False)
    rope_mod = Krea2RotaryPosEmbed(theta=1000.0, axes_dim=[32, 48, 48]).cuda()

    def make(length):
        length = (length + 255) // 256 * 256
        ids = torch.zeros(length, 3, dtype=torch.long, device="cuda")
        side = int(round(length**0.5))
        ids[:, 1] = torch.arange(length, device="cuda") // side
        ids[:, 2] = torch.arange(length, device="cuda") % side
        cos, sin = rope_mod(ids)
        x = torch.randn(1, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        temb = torch.randn(1, 1, 6 * dim, device="cuda", dtype=torch.bfloat16)
        return x, temb, (cos, sin)

    rows = []
    for length in tokens:
        x, temb, rope = make(length)

        def step(blk):
            def inner():
                if checkpoint:
                    from torch.utils.checkpoint import checkpoint as _ckpt

                    out = _ckpt(blk, x, temb, rope, use_reentrant=False)
                else:
                    out = blk(x, temb, rope)
                out.float().mean().backward()
                x.grad = None

            return inner

        kernels.set_mode("off")
        tx.Krea2AttnProcessor.__call__ = legacy_call
        t_legacy = timeit(step(block), steps=5)
        tx.Krea2AttnProcessor.__call__ = current_call
        t_eager = timeit(step(block), steps=6)
        # the kernel columns also bind the block-op patch and the attention hook
        from UnifiedTrainer.kernels import patch as _patch

        _patch.install()
        kernels.install_attention_backend()
        kernels.set_mode("auto")
        t_kernels = timeit(step(block), steps=6)

        t_compile = t_compile_kernels = float("nan")
        if include_compile:
            _patch.remove()
            kernels.uninstall_attention_backend()
            kernels.set_mode("off")
            compiled = torch.compile(block, dynamic=False)
            step(compiled)()
            torch.cuda.synchronize()
            t_compile = timeit(step(compiled), steps=4, warmup=1)
            _patch.install()
            kernels.install_attention_backend()
            kernels.set_mode("auto")
            t_compile_kernels = timeit(step(compiled), steps=4, warmup=1)
            del compiled
            torch.cuda.empty_cache()
        _patch.remove()
        kernels.uninstall_attention_backend()
        kernels.set_mode("auto")

        rows.append(dict(tokens=x.shape[1], legacy=t_legacy, eager=t_eager, kernels=t_kernels,
                         compile=t_compile, compile_kernels=t_compile_kernels))
        print(
            f"tokens {x.shape[1]:5d}  legacy {_fmt(t_legacy)}  "
            f"eager(kv-expand) {_fmt(t_eager)} ({_ratio(t_legacy, t_eager)})  "
            f"kernels {_fmt(t_kernels)} ({_ratio(t_legacy, t_kernels)})  "
            f"compile {_fmt(t_compile)} ({_ratio(t_legacy, t_compile)})  "
            f"compile+kernels {_fmt(t_compile_kernels)} ({_ratio(t_legacy, t_compile_kernels)})"
        )
        del x, temb, rope
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _legacy_krea2_processor_call():
    """The Krea-2 attention processor as it was before the GQA fix (for A/B)."""
    import torch.nn.functional as F

    def legacy(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            enable_gqa=attn.num_heads != attn.num_kv_heads,
        ).transpose(1, 2)
        return attn.to_out[0](out.flatten(2, 3) * torch.sigmoid(gate))

    return legacy


# ── fused block ops ──────────────────────────────────────────────────────


def bench_ops() -> list[dict]:
    from UnifiedTrainer import kernels
    from UnifiedTrainer.kernels import eager

    torch.manual_seed(0)
    dim, ffn = 6144, 16384
    cases = [
        ("rms_norm_modulate", (1, 2048, dim),
         lambda: (torch.randn(1, 2048, dim, device="cuda", dtype=torch.bfloat16),
                  torch.randn(dim, device="cuda", dtype=torch.bfloat16),
                  torch.randn(1, 1, dim, device="cuda", dtype=torch.bfloat16),
                  torch.randn(1, 1, dim, device="cuda", dtype=torch.bfloat16))),
        ("silu_mul", (1, 2048, ffn),
         lambda: (torch.randn(1, 2048, ffn, device="cuda", dtype=torch.bfloat16),
                  torch.randn(1, 2048, ffn, device="cuda", dtype=torch.bfloat16))),
        ("rope_apply", (1, 2048, 48, 128),
         lambda: (torch.randn(1, 2048, 48, 128, device="cuda", dtype=torch.bfloat16),
                  torch.randn(2048, 128, device="cuda", dtype=torch.float32),
                  torch.randn(2048, 128, device="cuda", dtype=torch.float32))),
        ("add_gated", (1, 2048, dim),
         lambda: (torch.randn(1, 2048, dim, device="cuda", dtype=torch.bfloat16),
                  torch.randn(1, 2048, dim, device="cuda", dtype=torch.bfloat16),
                  torch.randn(1, 2048, dim, device="cuda", dtype=torch.bfloat16))),
    ]
    rows = []
    for name, shape, make in cases:
        args = make()
        eager_fn = getattr(eager, name)
        fused_fn = getattr(kernels, name)
        t_eager = timeit(lambda: eager_fn(*args), steps=20)
        t_fused = timeit(lambda: fused_fn(*args), steps=20)
        compiled_fn = torch.compile(eager_fn, dynamic=False)
        compiled_fn(*args)
        t_compile = timeit(lambda: compiled_fn(*args), steps=10)
        rows.append(dict(op=name, shape=shape, eager=t_eager, triton=t_fused, compile=t_compile))
        print(
            f"{name:20} {str(shape):22} eager {_fmt(t_eager)}  "
            f"triton {_fmt(t_fused)} ({_ratio(t_eager, t_fused)})  "
            f"compile {_fmt(t_compile)} ({_ratio(t_eager, t_compile)})"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", nargs="?", default="attention",
                        choices=["attention", "block", "ops"])
    parser.add_argument("--tokens", nargs="*", type=int, default=[1024, 2048])
    parser.add_argument("--seq-lens", nargs="*", type=int, default=[1024, 2048])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--checkpoint", action="store_true",
                        help="block benchmark with gradient checkpointing, like a real run")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required for these measurements")
        return 2
    from UnifiedTrainer import kernels

    print(f"gpu {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")
    print(f"kernels {kernels.available()}")
    torch.manual_seed(0)
    if args.what == "attention":
        bench_attention(tuple(args.seq_lens))
    elif args.what == "block":
        bench_block(tuple(args.tokens), include_compile=not args.no_compile,
                    checkpoint=args.checkpoint)
    else:
        bench_ops()
    return 0


if __name__ == "__main__":
    sys.exit(main())
