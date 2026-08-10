"""MiniMax-H3 AdaLN modulation surrogate — built inside training, cached on disk.

The 50 per-block AdaLN projections, the final ``norm_out`` modulation and the
timestep MLP of MiniMax-H3 are frozen functions of the timestep ``t``:

    M[t, :] = [W_block . silu(time_embedder(t)) + b_block  for every block]

which amounts to ~13B parameters (~26 GB bf16) that never receive gradients
during LoRA/LoKR training.  ``install_surrogate`` replaces them with a
low-rank function of ``t``,

    M[t, :] ~= u(t) @ V,      u(t) = cheb(t) @ C        (rank R in [32, 128])

fit once at load time by streaming the checkpoint shards block by block
(progress bars; each block's matmuls run on the GPU when CUDA is available,
with only the weight loading / result transfer per layer) and cached on disk
next to the checkpoint as ``adaln_surrogate_r{R}.safetensors`` (+ JSON
sidecar).  The per-forward evaluation is a single ``V @ u(t)`` matmul
(~1 ms on GPU at R=64), so the AdaLN weights can be dropped from VRAM
entirely.

The swap is minimal and lossless by construction: only each
``transformer_blocks[i].adaln_proj.linear`` and ``norm_out.linear`` module is
replaced by a parameter-free cache lookup — the diffusers forward code,
gradient checkpointing, PEFT attachment and quantization all keep working
untouched.  A forward pre-hook reads the packed sequence's distinct
``timestep`` tensor, evaluates the surrogate and fills the per-forward cache
that the lookup modules read.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open
from safetensors.torch import save_file

try:  # progress bars for the multi-minute build passes
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm ships with huggingface_hub
    class tqdm:  # type: ignore[no-redef]
        """Minimal no-op fallback when tqdm is not installed."""

        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable) if self._iterable is not None else iter(())

        def update(self, n=1):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
)

logger = logging.getLogger(__name__)

# Timestep domain of training: t = 1 - sigma with sigma in [1e-5, 1 - 1e-5].
T_MIN, T_MAX = 1e-5, 1.0 - 1e-5
# Chebyshev basis size for the sigma-dependent part u(t).
K_BASIS = 64
# Configurable rank, clamped to [32, 128], default 64 (R_MIN/R_MAX/R_DEFAULT).
R_MIN, R_MAX, R_DEFAULT = 32, 128, 64


def clamp_rank(rank: int) -> int:
    return int(min(R_MAX, max(R_MIN, rank)))


def cheb_polys(t: torch.Tensor, k: int) -> torch.Tensor:
    """Chebyshev polynomials T_0..T_{k-1} of x = 2t - 1, t in [0, 1]."""
    x = 2.0 * t - 1.0
    out = [torch.ones_like(x), x.clone()]
    for _ in range(2, k):
        out.append(2.0 * x * out[-1] - out[-2])
    return torch.stack(out, dim=-1)


class _AdalnLinearLookup(nn.Module):
    """Parameter-free stand-in for an AdaLN projection's ``linear`` module.

    Returns the precomputed modulation rows for the current forward from the
    shared per-forward cache; the timestep embedding argument is ignored (the
    surrogate already accounts for it).  No parameters -> no state_dict
    entries, no PEFT target, no quantization, no gradient path.

    The diffusers AdaLN forwards cast their input with
    ``.to(self.linear.weight.dtype)`` before calling the module, so a plain
    (non-parameter) ``weight`` tensor is kept to mirror that attribute.
    """

    def __init__(self, cache: dict, key: str, weight_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self._cache = cache
        self._key = key
        self.weight = torch.empty(0, dtype=weight_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self._cache[self._key]
        except KeyError:
            raise RuntimeError(
                f"AdaLN lookup {self._key!r}: surrogate cache is empty — the "
                "forward pre-hook did not run for this transformer"
            ) from None


def _build_E(time_proj, time_embedder, t_vals: torch.Tensor, w_dtype: torch.dtype) -> torch.Tensor:
    """silu(time_embedder(time_proj(t))) at the AdaLN projection's dtype.

    Mirrors the model forward: activation at the time embedder's float32
    precision, cast to the (bfloat16) AdaLN projection dtype afterwards.
    Returns float32 with the bf16 rounding baked in (GPU bf16 matmul
    fidelity), so the sampled table matches the real forward.
    """
    temb = time_embedder(time_proj(t_vals).to(time_embedder.linear_1.weight.dtype))
    return F.silu(temb).to(w_dtype).to(torch.float32)


def build_surrogate(
    transformer_dir: str,
    rank: int = R_DEFAULT,
    grid: int = 1024,
    device: str = "auto",
    log: Callable[[str], None] = logger.info,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit M[t, :] ~= u(t) @ V from the checkpoint weights (streamed).

    ``device``: ``"auto"`` (default) runs each block's matmuls on the GPU when
    CUDA is available, ``"cuda"`` forces GPU (falls back to CPU if
    unavailable), ``"cpu"`` forces CPU.  The fp64 Gram-matrix accumulation and
    the eigendecomposition stay on the CPU either way (exactness); only the
    per-block ``E @ W.T`` sampling / reconstruction matmuls are offloaded.

    Returns ``(V fp16 [rows_total, rank], C fp32 [K_BASIS, rank], meta)`` and
    saves ``adaln_surrogate_r{rank}.safetensors`` + JSON sidecar inside
    ``transformer_dir`` (so the cache is keyed per checkpoint variant).
    """
    t_start = time.time()
    with open(os.path.join(transformer_dir, "config.json"), encoding="utf-8") as f:
        tcfg = json.load(f)
    with open(
        os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"),
        encoding="utf-8",
    ) as f:
        wmap = json.load(f)["weight_map"]

    H = int(tcfg["hidden_size"])
    D = int(tcfg["time_embed_dim"])
    n_blocks = int(tcfg["num_layers"])
    rows_block = 6 * 3 * H
    rows_out = 2 * H
    rows_total = n_blocks * rows_block + rows_out

    def load(key: str) -> torch.Tensor:
        shard = wmap[key]
        with safe_open(os.path.join(transformer_dir, shard), framework="pt") as f:
            return f.get_tensor(key)

    # Time embedding chain: exact diffusers modules, real weights.
    time_proj = Timesteps(num_channels=int(tcfg["freq_dim"]), flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = TimestepEmbedding(
        in_channels=int(tcfg["freq_dim"]),
        time_embed_dim=int(tcfg["time_embed_hidden_dim"]),
        out_dim=D,
    )
    for part in ("linear_1", "linear_2"):
        for suf in ("weight", "bias"):
            getattr(getattr(time_embedder, part), suf).data = load(f"time_embedder.{part}.{suf}")

    w_dtype = load("transformer_blocks.0.adaln_proj.linear.weight").dtype

    use_gpu = device in ("cuda", "auto") and torch.cuda.is_available()
    dev = torch.device("cuda" if use_gpu else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log("[adaln_surrogate] device='cuda' requested but CUDA unavailable; "
            "falling back to CPU")
    log(f"[adaln_surrogate] build rank={rank} grid={grid} blocks={n_blocks} H={H} "
        f"adaln_weight_dtype={w_dtype} device={dev} source={transformer_dir}")
    if use_gpu:
        log(f"[adaln_surrogate] streaming blocks one at a time on "
            f"{torch.cuda.get_device_name(0)}")

    def Wb(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a projection's weight/bias as float32 on the build device."""
        w, b = load(f"{key}.weight"), load(f"{key}.bias")
        if use_gpu:
            return w.to(device=dev, dtype=torch.float32), b.to(device=dev, dtype=torch.float32)
        return w.to(torch.float32), b.to(torch.float32)

    def block_key(b: int) -> str:
        return f"transformer_blocks.{b}.adaln_proj.linear" if b < n_blocks else "norm_out.linear"

    # Fitting grid: Chebyshev nodes of the first kind over [T_MIN, T_MAX].
    # Uniform points + high-degree Chebyshev basis suffer Runge-like edge
    # error *off* the grid; the nodes make the lstsq well-conditioned and the
    # off-grid interpolation error exponentially small (the surrogate is
    # continuous in t, so the grid never needs to be a snap/lerp table).
    i = torch.arange(grid, dtype=torch.float32)
    x = torch.cos(torch.pi * (2.0 * i + 1.0) / (2.0 * grid))  # [-1, 1]
    t_grid = ((T_MAX + T_MIN) + (T_MAX - T_MIN) * x).mul(0.5).sort().values
    E = _build_E(time_proj, time_embedder, t_grid, w_dtype)  # [N, D]

    # Held-out timesteps: random draws + the 20-step validation schedule + anchors.
    torch.manual_seed(0)
    t_ho = torch.rand(256, dtype=torch.float32) * (T_MAX - T_MIN) + T_MIN
    try:
        from diffusers import MiniMaxH3Scheduler

        sched = MiniMaxH3Scheduler(shift=12)
        sched.set_timesteps(20)
        t_ho = torch.cat([t_ho, sched.timesteps.float().clamp(T_MIN, T_MAX)])
    except Exception as exc:  # pragma: no cover - scheduler import is optional
        log(f"[adaln_surrogate] scheduler timesteps unavailable: {exc}")
    t_ho = torch.unique(torch.cat([t_ho, torch.tensor([0.999, 0.5, T_MIN, T_MAX])]))
    E_ho = _build_E(time_proj, time_embedder, t_ho, w_dtype)  # [M, D]

    # Pass 1: sample the modulation on the grid, accumulate M M^T (fp64).
    # Each block is streamed one at a time: load -> (GPU matmul if enabled) ->
    # transfer back; the fp64 Gram accumulation stays on CPU for exactness.
    work = tempfile.mkdtemp(prefix="adaln_surrogate_")
    try:
        mm = torch.zeros(grid, grid, dtype=torch.float64)
        E_dev = E.to(dev) if use_gpu else E
        t0 = time.time()
        for b in tqdm(range(n_blocks + 1), desc="[adaln_surrogate] pass1 sample grid", unit="block"):
            key = block_key(b)
            W, bias = Wb(key)
            mod = E_dev @ W.T + bias.unsqueeze(0)  # [N, rows]
            if use_gpu:
                mod = mod.to("cpu")
            fname = "norm_out.pt" if b >= n_blocks else f"{b:02d}.pt"
            torch.save(mod.to(w_dtype), os.path.join(work, fname))
            mm += mod.double() @ mod.double().T
            del W, bias, mod
        log(f"[adaln_surrogate] pass1 done in {time.time() - t0:.1f}s")

        # Spectrum: eigendecomposition of M M^T (fp64 exact for this size).
        evals, evecs = torch.linalg.eigh(mm)  # ascending
        evals = evals.flip(0)
        evecs = evecs.flip(1)
        svals = evals.clamp_min(0.0).sqrt()
        energy = svals.cumsum(0) / svals.sum().clamp_min(1e-30)
        log("[adaln_surrogate] singular values (top 8): "
            + " ".join(f"{svals[i].item():.3e}" for i in range(min(8, svals.numel()))))
        for r in (32, 64, 128):
            if r <= svals.numel():
                log(f"[adaln_surrogate]   energy retained at R={r:3d}: {energy[r - 1].item() * 100:.4f}%")

        # The spectrum is bounded by min(grid, rows), so the effective rank
        # cannot exceed the sampled grid size either.  Directions below 1e-6
        # relative to the top singular value carry no bf16-relevant signal
        # and would blow up V = M^T U S^-1 (fp16 overflow -> NaN), so the
        # numerical rank truncates the requested rank as well.
        reff = int((svals >= svals[0] * 1e-6).sum()) if svals.numel() else 1
        rmax = min(max(R_MIN, min(R_MAX, rank)), grid, max(1, reff))
        if rmax < rank:
            log(f"[adaln_surrogate] rank {rank} truncated to numerical rank {rmax} "
                f"(directions below 1e-6 relative singular value dropped)")
        U = evecs[:, :rmax].to(torch.float32)  # [N, Rmax] orthonormal
        S = svals[:rmax].to(torch.float32)
        u_fit = U * S  # scaled rows: M[t] = u(t) @ V^T
        U_inv = U / S.unsqueeze(0)  # V = M^T U S^-1

        # Chebyshev fit of u(t) over the grid.
        Phi = cheb_polys(t_grid, K_BASIS)  # [N, K]
        C = torch.linalg.lstsq(Phi, u_fit).solution  # [K, Rmax]
        fit_rel = (Phi @ C - u_fit).norm(dim=0) / u_fit.norm(dim=0).clamp_min(1e-12)
        log(f"[adaln_surrogate] Chebyshev u(t) fit: max col rel err = "
            f"{fit_rel.max().item():.3e} (rms {fit_rel.mean().item():.3e})")

        # Pass 2: V = M^T U S^-1 (small matmuls; stays on CPU).
        V = torch.zeros(rows_total, rmax, dtype=torch.float32)
        t0 = time.time()
        for b in tqdm(range(n_blocks + 1), desc="[adaln_surrogate] pass2 build V", unit="block"):
            fname = "norm_out.pt" if b >= n_blocks else f"{b:02d}.pt"
            mod = torch.load(os.path.join(work, fname), weights_only=True).to(torch.float32)
            if b < n_blocks:
                V[b * rows_block:(b + 1) * rows_block] = mod.T @ U_inv
            else:
                V[n_blocks * rows_block:] = mod.T @ U_inv
            del mod
        log(f"[adaln_surrogate] pass2 (V) done in {time.time() - t0:.1f}s "
            f"(max|V|={V.abs().max().item():.3e})")

        # Pass 3: held-out reconstruction error vs exact modulation.
        u_ho = cheb_polys(t_ho, K_BASIS) @ C  # [M, Rmax]
        errors: dict = {}
        ranks = [r for r in (32, 64, 128) if r <= rmax]
        E_ho_dev = E_ho.to(dev) if use_gpu else E_ho
        with tqdm(total=len(ranks) * (n_blocks + 1),
                  desc="[adaln_surrogate] pass3 held-out", unit="block") as pbar:
            for r in ranks:
                rel_all = []
                abs_max = 0.0
                Vr = V[:, :r]
                sur = Vr @ u_ho[:, :r].T  # [rows, M]
                for b in range(n_blocks + 1):
                    key = block_key(b)
                    W, bias = Wb(key)
                    exact = W @ E_ho_dev.T + bias.unsqueeze(1)  # [rows, M]
                    if use_gpu:
                        exact = exact.cpu()
                    row_slice = (slice(b * rows_block, (b + 1) * rows_block)
                                 if b < n_blocks else slice(n_blocks * rows_block, None))
                    rel_all.append(
                        (sur[row_slice] - exact).norm(dim=1)
                        / exact.norm(dim=1).clamp_min(1e-12)
                    )
                    abs_max = max(abs_max, (sur[row_slice] - exact).abs().max().item())
                    pbar.update(1)
                rel = torch.cat(rel_all)
                errors[r] = {
                    "max_rel": rel.max().item(),
                    "mean_rel": rel.mean().item(),
                    "p99_rel": torch.quantile(rel, 0.99).item(),
                    "max_abs": abs_max,
                }
                log(f"[adaln_surrogate]   held-out R={r:3d}: max_rel={errors[r]['max_rel']:.3e} "
                    f"mean_rel={errors[r]['mean_rel']:.3e} p99={errors[r]['p99_rel']:.3e} "
                    f"max_abs={abs_max:.3e}")

        # bf16-table baseline for comparison (the surrogate's error budget).
        rel_bf16 = []
        with tqdm(total=n_blocks + 1, desc="[adaln_surrogate] bf16 baseline", unit="block") as pbar:
            for b in range(n_blocks + 1):
                key = block_key(b)
                W, bias = Wb(key)
                ex = W @ E_ho_dev.T + bias.unsqueeze(1)
                if use_gpu:
                    ex = ex.cpu()
                rel_bf16.append(((ex.to(torch.bfloat16).to(torch.float32) - ex)).norm(dim=1)
                                / ex.norm(dim=1).clamp_min(1e-12))
                pbar.update(1)
        rel_bf16 = torch.cat(rel_bf16)
        log(f"[adaln_surrogate] bf16-table baseline: max_rel={rel_bf16.max().item():.3e} "
            f"mean_rel={rel_bf16.mean().item():.3e}")

        # End-to-end check against the real diffusers modules (block 0).
        adaln = MiniMaxH3AdaLayerNormModulation(time_embed_dim=D, hidden_size=H)
        adaln.linear.weight.data = load("transformer_blocks.0.adaln_proj.linear.weight")
        adaln.linear.bias.data = load("transformer_blocks.0.adaln_proj.linear.bias")
        out_mod = MiniMaxH3AdaLayerNormOut(hidden_size=H, time_embed_dim=D, eps=float(tcfg["norm_eps"]))
        out_mod.linear.weight.data = load("norm_out.linear.weight")
        out_mod.linear.bias.data = load("norm_out.linear.bias")
        t_chk = torch.tensor([0.3, 0.999, 0.72])
        temb = time_embedder(time_proj(t_chk).to(time_embedder.linear_1.weight.dtype))
        real_chunks = adaln(temb)  # 6 x [9, H]
        u_chk = cheb_polys(t_chk, K_BASIS) @ C[:, :rmax]
        sur0 = (V[0:rows_block, :rmax] @ u_chk.T).T  # [3, rows_block]
        maxdiff = 0.0
        for c in range(6):
            sur_c = sur0.view(3, 6, 3, H)[:, c].reshape(-1, H)  # [9, H]
            maxdiff = max(maxdiff, (sur_c - real_chunks[c]).abs().max().item())
        shift_real, scale_real = out_mod.linear(F.silu(temb).to(out_mod.linear.weight.dtype)).chunk(2, dim=-1)
        sur_out = (V[n_blocks * rows_block:, :rmax] @ u_chk.T).T.view(3, 2, H)
        out_maxdiff = max(
            (sur_out[:, 0] - shift_real).abs().max().item(),
            (sur_out[:, 1] - scale_real).abs().max().item(),
        )
        log(f"[adaln_surrogate] real-module check (block 0, R={rmax}): "
            f"adaln max_abs={maxdiff:.3e} norm_out max_abs={out_maxdiff:.3e}")

        # Per-step latency (K = 2 distinct timesteps, the image-pair case).
        u2 = (cheb_polys(torch.tensor([0.3, 0.999]), K_BASIS) @ C[:, :rmax]).to(torch.float16)
        Vr = V[:, :rmax].to(torch.float16)
        latency: dict = {}
        if torch.cuda.is_available():
            Vc, uc = Vr.cuda(), u2.cuda()
            for _ in range(3):
                _ = Vc @ uc.T
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                _ = Vc @ uc.T
            torch.cuda.synchronize()
            latency["gpu_bf16_ms"] = (time.time() - t0) / 20 * 1e3
            del Vc, uc
        Vr32 = V[:, :rmax].to(torch.float32)
        u32 = u2.to(torch.float32)
        for _ in range(2):
            _ = Vr32 @ u32.T
        t0 = time.time()
        for _ in range(10):
            _ = Vr32 @ u32.T
        latency["cpu_fp32_ms"] = (time.time() - t0) / 10 * 1e3
        log(f"[adaln_surrogate] latency K=2 R={rmax}: "
            + ", ".join(f"{k}={v:.2f}ms" for k, v in latency.items()))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Save: V fp16 (below-bf16 table noise), C fp32, sidecar metadata.
    V_r = V[:, :rmax].to(torch.float16).contiguous()
    C_r = C[:, :rmax].to(torch.float32).contiguous()
    out_st = os.path.join(transformer_dir, f"adaln_surrogate_r{rank}.safetensors")
    out_js = os.path.join(transformer_dir, f"adaln_surrogate_r{rank}.json")
    save_file({"V": V_r, "C": C_r}, out_st)
    meta = {
        "rank": rmax,
        "grid": grid,
        "grid_type": "chebyshev_nodes",
        "basis": K_BASIS,
        "hidden_size": H,
        "time_embed_dim": D,
        "num_blocks": n_blocks,
        "rows_block": rows_block,
        "rows_out": rows_out,
        "rows_total": rows_total,
        "t_min": T_MIN,
        "t_max": T_MAX,
        "source": transformer_dir,
        "errors": errors,
        "latency_ms": latency,
        "verify_block0_max_abs": maxdiff,
        "verify_norm_out_max_abs": out_maxdiff,
    }
    with open(out_js, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    freed_gb = (n_blocks * rows_block * D + rows_out * D) * 2 / 1e9
    added_mb = rows_total * rank * 2 / 1e6
    log(f"[adaln_surrogate] saved {out_st}")
    log(f"[adaln_surrogate] AdaLN weights freed ~{freed_gb:.1f} GB (bf16); "
        f"surrogate ~{added_mb:.0f} MB fp16; total {time.time() - t_start:.1f}s")
    return V_r, C_r, meta


def load_surrogate(
    transformer_dir: str, rank: int, log: Callable[[str], None] = logger.info
) -> Optional[Tuple[torch.Tensor, torch.Tensor, dict]]:
    """Load a previously built surrogate, validating it against config.json."""
    out_st = os.path.join(transformer_dir, f"adaln_surrogate_r{rank}.safetensors")
    out_js = os.path.join(transformer_dir, f"adaln_surrogate_r{rank}.json")
    if not (os.path.isfile(out_st) and os.path.isfile(out_js)):
        return None
    try:
        with open(out_js, encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(transformer_dir, "config.json"), encoding="utf-8") as f:
            tcfg = json.load(f)
        if (
            int(meta["num_blocks"]) != int(tcfg["num_layers"])
            or int(meta["hidden_size"]) != int(tcfg["hidden_size"])
        ):
            log("[adaln_surrogate] cached surrogate does not match config; rebuilding")
            return None
        with safe_open(out_st, framework="pt") as f:
            V = f.get_tensor("V")
            C = f.get_tensor("C")
        log(f"[adaln_surrogate] loaded {out_st} "
            f"(V {tuple(V.shape)}, errors R=64 max_rel="
            f"{meta.get('errors', {}).get('64', {}).get('max_rel', 'n/a')})")
        return V, C, meta
    except Exception as exc:  # pragma: no cover - corrupt cache falls back to rebuild
        log(f"[adaln_surrogate] cache load failed ({exc}); rebuilding")
        return None


def _make_pre_hook() -> Callable:
    """Forward pre-hook: evaluate u(t) @ V for the current distinct timesteps."""

    def pre_hook(module: nn.Module, args, kwargs) -> None:
        cache = module._adaln_cache
        timestep = None
        if kwargs is not None:
            timestep = kwargs.get("timestep")
        if timestep is None and args is not None and len(args) > 3:
            timestep = args[3]
        if timestep is None:
            return
        with torch.no_grad():
            t = timestep.detach().float().cpu().clamp(T_MIN, T_MAX)  # [K]
            C = module._adaln_C
            u = (cheb_polys(t, K_BASIS) @ C).to(torch.float16)  # [K, R]
            V = module._adaln_V
            dev = next(module.parameters()).device
            if V.device != dev:
                V = V.to(dev)
                module._adaln_V = V
            mod_all = V @ u.to(dev).T  # [rows_total, K] fp16
            dtype = getattr(module, "_adaln_dtype", None) or next(module.parameters()).dtype
            mod_all = mod_all.to(dtype)  # block-stack precision (bf16 in practice)
            rb = module._adaln_rows_block
            nb = module._adaln_num_blocks
            for b in range(nb):
                cache[f"b{b}"] = mod_all[b * rb:(b + 1) * rb].T.contiguous()
            cache["norm_out"] = mod_all[nb * rb:].T.contiguous()

    return pre_hook


def install_surrogate(
    transformer: nn.Module,
    transformer_dir: str,
    rank: int = R_DEFAULT,
    grid: int = 1024,
    device: str = "auto",
    log: Callable[[str], None] = logger.info,
) -> dict:
    """Build (or load) the surrogate and swap the AdaLN projections for lookups.

    Replaces every ``transformer_blocks[i].adaln_proj.linear`` and
    ``norm_out.linear`` with a parameter-free cache lookup, registers a
    forward pre-hook that evaluates the surrogate per forward, and drops the
    ~13B AdaLN weights from the module.  ``device`` ("auto"/"cuda"/"cpu")
    only affects the one-time build (see ``build_surrogate``); cached builds
    skip it.  Returns the surrogate metadata.
    """
    rank = clamp_rank(rank)
    loaded = load_surrogate(transformer_dir, rank, log=log)
    if loaded is None:
        V, C, meta = build_surrogate(transformer_dir, rank=rank, grid=grid,
                                     device=device, log=log)
    else:
        V, C, meta = loaded

    cache: dict = {}
    transformer._adaln_cache = cache
    transformer._adaln_V = V
    transformer._adaln_C = C
    transformer._adaln_rows_block = int(meta["rows_block"])
    transformer._adaln_num_blocks = int(meta["num_blocks"])
    # Block-stack precision (bf16): ``next(parameters())`` would return the
    # fp32 proj_in module, which is not the AdaLN consumers' dtype.
    transformer._adaln_dtype = next(transformer.transformer_blocks[0].parameters()).dtype
    for b in range(int(meta["num_blocks"])):
        transformer.transformer_blocks[b].adaln_proj.linear = _AdalnLinearLookup(
            cache, f"b{b}", weight_dtype=transformer._adaln_dtype
        )
    transformer.norm_out.linear = _AdalnLinearLookup(cache, "norm_out", weight_dtype=transformer._adaln_dtype)
    # with_kwargs=True: the hook must see the ``timestep`` kwarg the trainer
    # passes (the default pre-hook signature only receives positional args).
    transformer.register_forward_pre_hook(_make_pre_hook(), with_kwargs=True)

    # Params freed = the swapped AdaLN projections + norm_out (counted from
    # metadata, not the module tree — the lookups are parameter-free).  The
    # tiny time-embedder MLP stays: the diffusers forward calls it every
    # step, and its ~16K params are negligible.
    n_dropped = int(meta["num_blocks"]) * int(meta["rows_block"]) * int(meta["time_embed_dim"]) \
        + int(meta["rows_out"]) * int(meta["time_embed_dim"])
    log(f"[adaln_surrogate] installed: {n_dropped / 1e9:.2f}B params replaced by "
        f"{meta['rows_total'] * meta['rank'] * 2 / 1e6:.0f} MB lookup (R={meta['rank']})")
    return meta
