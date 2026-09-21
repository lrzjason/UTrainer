"""Estimate the effective guidance-distillation scale w* of a checkpoint.

Why: the guide_flow_matching loss (DC-Gen Eq. 10, see
md/09-corrected-flow-matching-guide.md) corrects the velocity with the
guidance scale w.  For checkpoints whose transformer takes NO guidance input
(Qwen-Image 2.1, Z-Image-Turbo style), the distillation baked in a single
effective scale w* — using w != w* leaves a residual bias
(w - w*)/(1 + w) * v_theta(^c) in every step's target.

How: with the UNTUNED base model, run the conditional and the empty-prompt
forwards on cached real samples and scan w.  The corrected estimate
    v_hat(w) = (v_eta(c) + w * v_eta(^c)) / (1 + w)
equals the raw velocity (= flow target v_t) exactly at w = w*, so the MSE
curve has its minimum there.  A minimum at w = 0 means the checkpoint is NOT
distilled (plain flow_matching is the matched objective).

Usage (from the UTrainer root):
    python -m UnifiedTrainer.utils.estimate_guidance_scale \
        --config UnifiedTrainer/configs/qwen_image21_guide_fm_lokr.json
    # options: --draws 24  --samples 4  --w-grid 0,0.5,1,1.5,2,3,4,6,8,12

Loads the real transformer (NF4) on GPU; no_grad throughout.  Requires a
built embedding/latent cache (the same one training uses).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_sample(cache_dir: str, adapter_name: str):
    """Resolve (latent_npz, caption_npz, empty_npz) from the cache index."""
    index_path = os.path.join(cache_dir, "train_dataset_default.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"train index not found: {index_path}")
    rows = _load_json(index_path)
    if not rows:
        raise ValueError(f"empty train index: {index_path}")
    json_path = rows[0].get("json_path")
    if not json_path or not os.path.isfile(json_path):
        raise FileNotFoundError(f"sample JSON missing: {json_path}")
    sample = _load_json(json_path)

    targets = sample.get("targets") or {}
    if not targets:
        raise ValueError(f"no targets in {json_path}")
    lat_path = next(iter(targets.values())).get("latent_path")
    captions = sample.get("captions") or {}
    if not captions:
        raise ValueError(f"no captions in {json_path}")
    cap_path = next(iter(captions.values())).get("npz_path")
    empty_path = os.path.join(cache_dir, f"empty_embedding.{adapter_name}.npz")
    for p in (lat_path, cap_path, empty_path):
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f"cache file missing: {p}")
    return lat_path, cap_path, empty_path


def _latent_to_runtime(npz_path: str, device) -> torch.Tensor:
    """Cache stores (C, T, H, W) with B folded -> runtime (B, C, H, W)."""
    npz = np.load(npz_path, allow_pickle=True)
    key = next(
        k for k in npz.files if npz[k].ndim == 4 and npz[k].dtype.kind == "f"
    )
    x = torch.from_numpy(npz[key].astype(np.float32))
    if x.dim() == 4:
        x = x.squeeze(1).unsqueeze(0)
    return x.to(device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="training config JSON (paths + cache_dir)")
    ap.add_argument("--draws", type=int, default=24, help="(noise, sigma) draws per sample")
    ap.add_argument("--samples", type=int, default=1, help="cached samples to average over")
    ap.add_argument("--w-grid", default="0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.getcwd())
    from UnifiedTrainer.train import _import_all_adapters, _replace_linear_with_4bit

    _import_all_adapters()  # adapters register lazily on module import

    cfg = _load_json(args.config)
    cache_dir = cfg.get("data", {}).get("cache_dir", "")
    if not cache_dir:
        print("config has no data.cache_dir", file=sys.stderr)
        return 2

    from UnifiedTrainer.registry import ModelRegistry

    adapter_cls = ModelRegistry.get(cfg["model"])
    adapter = adapter_cls(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    transformer = adapter.load_transformer(cfg["transformer_path"], dtype)
    _replace_linear_with_4bit(transformer, compute_dtype=dtype, quant_type="nf4")
    transformer = transformer.to(device).eval()
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    print(f"[w*] transformer loaded ({adapter.name}, NF4): {n_params:.2f}B params on {device}")

    lat_path, cap_path, empty_path = _first_sample(cache_dir, adapter.name)
    print(f"[w*] sample latent : {lat_path}")
    print(f"[w*] sample caption: {cap_path}")
    print(f"[w*] empty embed   : {empty_path}")

    x0 = _latent_to_runtime(lat_path, device)
    B, C, H, W = x0.shape
    cap_npz = np.load(cap_path, allow_pickle=True)
    empty_npz = np.load(empty_path, allow_pickle=True)

    def make_batch(pe, mask, img_mask=None):
        emb = {"prompt_embed": pe, "prompt_embeds_mask": mask}
        if img_mask is not None:
            emb["img_mask"] = img_mask
        return {
            "latents": {"T": x0},
            "embeddings": [emb],
            "batch_configs": [
                {"target_config": "T", "caption_config": "train_T", "caption_dropout": 0.0}
            ],
        }

    cap_batch = make_batch(
        cap_npz["prompt_embed"], cap_npz["prompt_embeds_mask"],
        cap_npz["img_mask"] if "img_mask" in cap_npz.files else None,
    )
    empty_batch = make_batch(empty_npz["prompt_embed"], empty_npz["prompt_embeds_mask"])

    w_grid = [float(w) for w in args.w_grid.split(",") if w.strip()]
    sse = {w: 0.0 for w in w_grid}
    plain_sse = 0.0
    n_elems = 0
    torch.manual_seed(args.seed)

    with torch.no_grad():
        for k in range(args.draws):
            _, sigmas = adapter.sample_timesteps(
                B, device, x0.dtype, latent_height=H, latent_width=W
            )
            noise = torch.randn_like(x0)
            sigmas_b = sigmas.view(-1, *(1,) * (x0.dim() - 1))
            x_t = (1.0 - sigmas_b) * x0 + sigmas_b * noise
            v_t = noise - x0  # standard velocity target (adapter.velocity_sign)

            v_c = adapter.unpack_prediction(
                transformer(**adapter.prepare_model_input(cap_batch, [x_t], sigmas))
            )[0].float()
            v_u = adapter.unpack_prediction(
                transformer(**adapter.prepare_model_input(empty_batch, [x_t], sigmas))
            )[0].float()

            plain_sse += float(((v_c - v_t) ** 2).sum())
            for w in w_grid:
                v_hat = (v_c + w * v_u) / (1.0 + w)
                sse[w] += float(((v_hat - v_t) ** 2).sum())
            n_elems += v_t.numel()

    mse = {w: sse[w] / n_elems for w in w_grid}
    plain_mse = plain_sse / n_elems
    best_w = min(mse, key=mse.get)
    peak = max(mse.values())

    print(f"\n[w*] corrected-velocity MSE vs flow target ({args.draws} draws):")
    for w in w_grid:
        bar = "#" * int(60 * mse[w] / peak)
        mark = "  <-- argmin" if w == best_w else ""
        print(f"    w={w:5.1f}  mse={mse[w]:.5f}  {bar}{mark}")
    print(f"\n[w*] plain FM (v_cond vs v_t): mse={plain_mse:.5f}")
    print(
        f"[w*] argmin w* = {best_w}  ({100 * (1 - mse[best_w] / plain_mse):.1f}% "
        f"below plain FM)"
    )
    if best_w >= 1.0 and mse[best_w] < plain_mse * 0.95:
        print(f"[w*] => guidance-distilled near w*~{best_w}: configure "
              f"guide_flow_matching with fixed_guidance_scale={best_w} "
              f"(or a narrow range around it).")
    elif mse.get(0.0, plain_mse) <= min(mse[w] for w in w_grid if w > 0):
        print("[w*] => minimum at w=0: no distillation signature; plain "
              "flow_matching is the matched objective.")
    else:
        print("[w*] => weak/flat minimum: inconclusive; treat w* as uncertain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
