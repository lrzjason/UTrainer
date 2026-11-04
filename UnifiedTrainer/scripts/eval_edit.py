"""Objective evaluation for edit runs: does the output edit, and does it keep the rest?

Usage:
    python -m UnifiedTrainer.scripts.eval_edit --run output/moss_edit_r2t_instr

For every ``*_val_epoch*_<i>.png`` triple (gen / _ref / _target) it reports
  1. edit progress  <gen-ref, target-ref> / |target-ref|^2   (1.0 = fully edited)
  2. PSNR(gen, ref) on the *unchanged* pixels                (preservation)
  3. PSNR(gen, target) globally
  4. how much of the change lands inside the R/T difference  (locality)

``val_loss`` is deliberately NOT used: flow-matching MSE averages over sigma in
[0, 1], where the high-sigma term is irreducible, so it barely moves even when the
images go from unrecognisable to a near-exact copy (measured: val_loss 0.39 -> 0.43
while PSNR rose 17.7 -> 28.2 dB).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def evaluate(run_dir: str, diff_threshold: float = 12.0) -> dict:
    rows = defaultdict(list)
    for gen_path in sorted(glob.glob(os.path.join(run_dir, "**", "*_val_epoch*_*.png"), recursive=True)):
        if gen_path.endswith(("_ref.png", "_target.png")):
            continue
        ref_path = gen_path[:-4] + "_ref.png"
        tgt_path = gen_path[:-4] + "_target.png"
        if not (os.path.exists(ref_path) and os.path.exists(tgt_path)):
            continue
        epoch = int(re.search(r"val_epoch(\d+)_", gen_path).group(1))
        gen, ref, tgt = _load(gen_path), _load(ref_path), _load(tgt_path)
        if gen.shape != tgt.shape or ref.shape != tgt.shape:
            continue
        changed = np.abs(tgt - ref).max(axis=2) > diff_threshold
        keep = ~changed
        if changed.sum() < 50:
            continue
        d = (tgt - ref)[changed].reshape(-1, 3)
        e = (gen - ref)[changed].reshape(-1, 3)
        progress = float((d * e).sum() / max(1e-9, (d * d).sum()))
        outside = np.abs(gen - ref).max(axis=2) > diff_threshold
        locality = float((outside & changed).sum() / max(1, outside.sum()))
        rows[epoch].append({
            "progress": progress,
            "psnr_keep": _psnr(gen[keep], tgt[keep]) if keep.any() else float("nan"),
            "psnr_target": _psnr(gen, tgt),
            "psnr_ref": _psnr(gen, ref),
            "locality": locality,
            "edit_area": float(100.0 * changed.mean()),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run output directory")
    ap.add_argument("--diff-threshold", type=float, default=12.0)
    args = ap.parse_args(argv)

    rows = evaluate(args.run, args.diff_threshold)
    if not rows:
        print("no val triples found under", args.run)
        return 1
    print(f"{'epoch':>5} {'progress':>9} {'PSNR keep':>10} {'PSNR tgt':>9} {'PSNR ref':>9} {'locality':>9} {'area%':>6}")
    for epoch in sorted(rows):
        items = rows[epoch]
        mean = {k: float(np.nanmean([r[k] for r in items])) for k in items[0]}
        print(f"{epoch:>5} {mean['progress']:>9.2f} {mean['psnr_keep']:>10.1f} "
              f"{mean['psnr_target']:>9.1f} {mean['psnr_ref']:>9.1f} "
              f"{mean['locality']:>9.2f} {mean['edit_area']:>6.1f}")
    print("\nhealthy edit run: progress -> 1.0, PSNR keep >= 30 dB, locality >= 0.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
