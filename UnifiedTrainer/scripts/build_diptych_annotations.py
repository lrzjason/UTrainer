"""Canonical builder for the diptych (IC-LoRA) edit annotations.

Rules (see md/17-diptych-bbox-rules.md for the reasoning):

  1. edit area  - measured, never hand-placed: blur each panel (gaussian 1.2), take the
                  max-channel left/right difference, threshold 30, ignore a 10 px band
                  around the panel borders, drop connected components < 500 px, drop
                  1-2 px seam strips, 3x3 opening, then the bounding box of what survives.
  2. padding    - grow that box by max(10 px, 1% of the canvas) and clamp it to the right
                  panel.
  3. one item   - if an existing right-panel item already sits on the measured box
                  (IoU >= 0.5) it is *adjusted* to that box, otherwise the box is appended
                  as a new item. Either way exactly one item carries ``operation: "edit"``.
  4. keep all   - every other annotated item is preserved verbatim.
  5. mirrors    - the same object must have the same box on both sides:
                  right_box == left_box + (width // 2, 0). Mismatches are corrected
                  against the left panel (the edited item is exempt by design).
  6. drops      - groups whose diff covers the whole panel (a re-render, not an edit) and
                  the groups rejected during review are skipped.
  7. output     - annotations/<stem>.json  draft layout, abs/xyxy, with ``operation``
                  captions/<stem>.txt     compiled MOSS prompt (abs_centralized/yxyx)
                  bbox/<stem>.jpg         preview: every item numbered, ``EDIT #id`` in red

Run:  python -m UnifiedTrainer.scripts.build_diptych_annotations --data <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

# ── measured edit area ──────────────────────────────────────────────────
BLUR = 1.2
THR = 30.0
MIN_BLOB = 500
BORDER = 10
PAD_FRAC = 0.01
PAD_MIN = 10
MERGE_IOU = 0.5

# ── review decisions (1-based item ids from the previews) ───────────────
DROP = {
    "002_change_d3747c34_0001", "009_change_cab35af1_0001", "012_change_3cea7be3_0001",
    "040_change_f6dafadb_0001", "044_change_aa183cc7_0001", "032_change_8fa9562c_0001",
    "019_change_ec3db370_0001", "015_change_2a98381b_0001",
    "043_change_635b7183_0001", "006_change_7e7972b6_0001", "011_change_379a8647_0001",
    "022_change_da860acc_0001", "031_change_b856ecd2_0001",
}

OVERRIDES = {
    "030_change_e0a65d5d_0001": {"delete": [12], "edit_to": 12},
    "036_change_33b88328_0001": {"delete": [1, 13]},
    "037_change_a4c8a560_0001": {"mirror_edit": {"caption": "A horse's head", "label": "character"},
                                 "insert_at": 12, "edit_to": 13},
    "039_change_1ae24fbd_0001": {"mirror_edit": {"caption": 'Text "OPEN"', "label": "text", "text": "OPEN"},
                                 "insert_at": 11, "edit_to": 12},
    "041_change_30e48fad_0001": {"delete": [5, 6, 7, 9, 17, 18, 19, 21], "edit_to": 13},
    "047_change_4a26915c_0001": {"delete": [11, 12, 23], "merge_edit_into": 13},
    "050_change_ab9d5301_0001": {"delete": [2, 6, 8, 9, 14, 18, 20, 21]},
}

KEEP_RE = re.compile(r",?\s*kept unchanged from the original\.?$", re.I)
COMPILER_DIR = r"E:\MOSS_data_pipeline\ref\MOSS-Image\skills\moss-image-prompt-enhancer\scripts"


# ── helpers ─────────────────────────────────────────────────────────────
def compile_prompt_local():
    if COMPILER_DIR not in sys.path:
        sys.path.insert(0, COMPILER_DIR)
    from compile_prompt import compile_prompt

    return compile_prompt


def strip_keep(caption: str) -> str:
    cap = " ".join(str(caption).split())
    for tail in (", kept unchanged from the original.", " kept unchanged from the original."):
        if cap.lower().endswith(tail):
            return cap[: -len(tail)].rstrip(" ,.")
    return cap.rstrip(" .")


def norm(caption: str) -> str:
    c = KEEP_RE.sub("", " ".join(str(caption).split()))
    c = re.sub(r"\s*\(edited area\)$", "", c, flags=re.I)
    return c.strip().lower()


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def measure_edit_box(im: Image.Image) -> list[int] | None:
    """Outward-expanded box covering every significant left/right difference."""
    arr = np.asarray(im.filter(ImageFilter.GaussianBlur(BLUR))).astype(np.float32)
    ih, iw = arr.shape[:2]
    half = iw // 2
    d = np.abs(arr[:, :half] - arr[:, half:half * 2]).max(axis=2)
    d[:, :BORDER] = 0
    d[:, half - BORDER:] = 0
    mask = d > THR
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    keep = np.zeros_like(mask)
    if n:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        for k in range(n):
            if sizes[k] < MIN_BLOB:
                continue
            ys, xs = np.where(lab == k + 1)
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            if bw <= 2 and bh > 40:
                continue
            if bw < 8 and bh > 0.3 * ih:
                continue
            if min(xs.min(), half - 1 - xs.max()) < 6 and bh > 0.5 * ih:
                continue
            keep |= (lab == k + 1)
    keep = ndimage.binary_opening(keep, np.ones((3, 3)))
    ys, xs = np.where(keep)
    if len(xs) == 0:
        return None
    pad_x = max(PAD_MIN, int(round(PAD_FRAC * iw)))
    pad_y = max(PAD_MIN, int(round(PAD_FRAC * ih)))
    return [max(half, int(xs.min()) + half - pad_x), max(0, int(ys.min()) - pad_y),
            min(iw, int(xs.max()) + 1 + half + pad_x), min(ih, int(ys.max()) + 1 + pad_y)]


def item(it, bbox=None, caption=None, operation=None, label=None, text=None):
    lab = label or it.get("label", "object")
    txt = text if text is not None else it.get("text")
    if lab == "text" and not txt:
        lab = "object"
    out = {"label": lab,
           "caption": caption if caption is not None else " ".join(str(it.get("caption", "")).split()),
           "bbox": [round(float(v), 1) for v in (bbox if bbox is not None else it["bbox"])]}
    if lab == "text" and txt:
        out["text"] = txt
    if operation:
        out["operation"] = operation
    return out


def apply_overrides(items: list[dict], ops: dict, half: int) -> list[dict]:
    edit = next((it for it in items if it.get("operation") == "edit"), None)
    if "mirror_edit" in ops and edit is not None:
        spec = ops["mirror_edit"]
        x1, y1, x2, y2 = edit["bbox"]
        items.insert(ops.get("insert_at", len(items)) - 1,
                     item({"label": spec["label"], "caption": spec["caption"]},
                          bbox=[x1 - half, y1, x2 - half, y2],
                          label=spec["label"], text=spec.get("text")))
    if "merge_edit_into" in ops and edit is not None:
        target = ops["merge_edit_into"] - 1
        if target < len(items):
            items[target].update({"caption": edit["caption"], "operation": "edit",
                                  "label": edit["label"]})
    for idx in sorted(ops.get("delete", []), reverse=True):
        if 0 < idx <= len(items):
            items.pop(idx - 1)
    slot = ops.get("edit_to")
    if slot:
        i = next((i for i, it in enumerate(items) if it.get("operation") == "edit"), None)
        if i is not None:
            items.insert(slot - 1, items.pop(i))
    return items


def align_mirrors(items: list[dict], half: int) -> list[tuple]:
    """right_box must equal left_box + (half, 0); the edit item is exempt."""
    left = [(i, it) for i, it in enumerate(items)
            if it.get("operation") != "edit" and float(it["bbox"][0]) < half]
    right = [(i, it) for i, it in enumerate(items)
             if it.get("operation") != "edit" and float(it["bbox"][0]) >= half]
    used, fixes = set(), []
    for li, l in left:
        tgt = [float(l["bbox"][0]) + half, float(l["bbox"][1]),
               float(l["bbox"][2]) + half, float(l["bbox"][3])]
        cand = [(ri, r) for ri, r in right if ri not in used
                and (norm(r["caption"]) == norm(l["caption"])
                     or (abs(float(r["bbox"][0]) - tgt[0]) < 12
                         and abs(float(r["bbox"][1]) - tgt[1]) < 12))]
        if not cand:
            continue
        ri, r = sorted(cand, key=lambda t: abs(float(t[1]["bbox"][0]) - tgt[0]))[0]
        used.add(ri)
        cur = [float(v) for v in r["bbox"]]
        delta = max(abs(cur[0] - tgt[0]), abs(cur[1] - tgt[1]),
                    abs(cur[2] - tgt[2]), abs(cur[3] - tgt[3]))
        if delta > 0.5:
            r["bbox"] = [round(v, 1) for v in tgt]
            fixes.append((li + 1, ri + 1, round(delta, 1)))
    return fixes


def render(stem, canvas, img_path, items, out_path):
    W, H = canvas
    im = Image.open(img_path).convert("RGB")
    iw, ih = im.size
    sx, sy = iw / W, ih / H
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        small = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = small = ImageFont.load_default()
    for i, it in enumerate(items, 1):
        x1, y1, x2, y2 = it["bbox"]
        bx1, by1, bx2, by2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        if bx2 - bx1 < 3 or by2 - by1 < 3:
            continue
        is_edit = it.get("operation") == "edit"
        keep = ("kept unchanged" in it["caption"].lower()) and not is_edit
        col, wd = ((255, 20, 20), 8) if is_edit else (((0, 215, 255), 2) if keep else ((255, 210, 0), 3))
        dr.rectangle([bx1, by1, bx2, by2], outline=col + (255,), width=wd)
        tag = f"EDIT #{i}" if is_edit else str(i)
        tw = dr.textlength(tag, small if is_edit else font)
        th = 26 if is_edit else 32
        ty = min(ih - th - 4, by1 + 6) if is_edit else max(0, by1 - 32)
        tx = min(iw - tw - 18, bx1 + 8) if is_edit else bx1
        dr.rectangle([tx, ty, tx + tw + 14, ty + th + 2], fill=col + (242,))
        dr.text((tx + 7, ty + 1), tag, fill=(0, 0, 0), font=small if is_edit else font)
    dr.line([(iw // 2, 0), (iw // 2, ih)], fill=(255, 255, 255, 180), width=3)
    im.save(out_path, quality=88)


def caption_items(items):
    return [{"label": it["label"], "caption": it["caption"], "bbox": it["bbox"],
             **({"text": it["text"]} if it.get("text") else {})} for it in items]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build diptych edit annotations from the measured diff")
    ap.add_argument("--data", required=True, help="folder with <stem>.png + <stem>.json drafts")
    ap.add_argument("--out", required=True, help="output folder for annotations/captions/previews")
    args = ap.parse_args(argv)
    compile_prompt = compile_prompt_local()

    ann_dir = os.path.join(args.out, "annotations")
    cap_dir = os.path.join(args.out, "captions")
    viz_dir = os.path.join(args.out, "bbox")
    for d in (ann_dir, cap_dir, viz_dir):
        os.makedirs(d, exist_ok=True)

    kept = dropped = 0
    for p in sorted(glob.glob(os.path.join(args.data, "*.json"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        img_path = os.path.join(args.data, stem + ".png")
        if stem in DROP or not os.path.exists(img_path):
            dropped += 1
            continue
        raw = json.load(open(p, encoding="utf-8"))
        W, H = int(raw["width"]), int(raw["height"])
        half = W // 2
        im = Image.open(img_path)
        iw, ih = im.size
        box_px = measure_edit_box(im)
        if box_px is None:
            dropped += 1
            continue
        box_abs = [round(box_px[0] * W / iw, 1), round(box_px[1] * H / ih, 1),
                   round(box_px[2] * W / iw, 1), round(box_px[3] * H / ih, 1)]

        cands = sorted(((iou([float(v) for v in it["bbox"]], box_abs), it)
                        for it in raw["items"] if float(it["bbox"][0]) >= half),
                       key=lambda t: -t[0])
        merge_item, merge_iou = (cands[0][1], cands[0][0]) if cands and cands[0][0] >= MERGE_IOU else (None, 0.0)

        items = [item(it) for it in raw["items"]]
        if merge_item is not None:
            idx = [i for i, it in enumerate(raw["items"]) if it is merge_item][0]
            cap = strip_keep(merge_item.get("caption", ""))
            items[idx] = item(merge_item, bbox=box_abs, caption=f"{cap} (edited area)", operation="edit")
        else:
            cap = strip_keep(cands[0][1].get("caption", "")) if cands else "the changed region"
            base = cands[0][1] if cands else {"label": "object", "caption": cap}
            items.append(item(base, bbox=box_abs, caption=f"{cap} (edited area)", operation="edit"))

        if stem in OVERRIDES:
            items = apply_overrides(items, OVERRIDES[stem], half)
        fixes = align_mirrors(items, half)

        summary = ("A two-panel image: the left half is the original image, the right half is the "
                   f"same image after one change - {cap[0].lower() + cap[1:]} (edited area). "
                   "Everything else in the two halves is identical.")
        draft = {"width": W, "height": H, "bbox_coord_space": "abs", "bbox_format": "xyxy",
                 "summary": {"canvas": "multi_panel", "caption": summary}, "items": items}
        try:
            prompt = compile_prompt({**draft, "items": caption_items(items)})
        except Exception as exc:  # noqa: BLE001
            print(f"{stem}: compile failed ({exc}) - skipped")
            dropped += 1
            continue

        json.dump(draft, open(os.path.join(ann_dir, stem + ".json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        open(os.path.join(cap_dir, stem + ".txt"), "w", encoding="utf-8").write(prompt + "\n")
        render(stem, (W, H), img_path, items, os.path.join(viz_dir, stem + ".jpg"))
        kept += 1
        edit_id = next((i for i, it in enumerate(items, 1) if it.get("operation") == "edit"), None)
        print(f"{stem[:34]:<36} items={len(items):<3} EDIT #{edit_id:<3} "
              f"iou={merge_iou:4.2f} mirrors_fixed={len(fixes)}")

    print(f"\nkept {kept} | dropped {dropped}")
    print(f"annotations -> {ann_dir}\ncaptions    -> {cap_dir}\npreviews    -> {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
