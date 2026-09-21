"""LPIPS-VGG perceptual distance — official PFM companion encoder.

The official PFM implementation (https://github.com/ZhaoChuyang/PFM,
pfm/models/losses/perceptual.py, branch net=="vgg") stacks VGG-LPIPS with a
DINO-family encoder, both at weight 1.0. LPIPS is a LEARNED perceptual
metric with bounded, gently-behaved activations — in the official recipe it
anchors low-level appearance while DINO anchors semantics.

Official contract (branch net=="vgg" of the file above):
  - model  = LPIPS(net="vgg", pretrained=True)
  - inputs are pixels in [-1, 1] (no [0,1] conversion, NO normalization —
    LPIPS ships its own input scaling layer)
  - resize to 224x224 with plain bilinear (align_corners=False, no
    antialias)
  - distance = model(pred, target).mean()

Only the contract is mirrored here; no code is imported from the reference
tree.
"""
from __future__ import annotations

import os
import shutil

import torch
import torch.nn.functional as F

LPIPS_VGG_INPUT_SIZE = 224


def load_lpips_vgg(weights_path: str | None = None) -> torch.nn.Module:
    """Frozen LPIPS-VGG (CPU / caller's device; cast dtype at call site).

    Args:
        weights_path: Optional path to the torchvision VGG16 state dict
            (``vgg16-397923af.pth``). When None/empty, the default
            ``LPIPS(net="vgg", pretrained=True)`` behavior is used (torchvision
            downloads/loads the backbone from its cache). When a path is given,
            the backbone weights are loaded from that file instead — no
            download, and the config controls the location.
    """
    from lpips import LPIPS

    if not weights_path:
        model = LPIPS(net="vgg", pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # Custom path: build the trunk with pretrained=False (no download), then
    # copy the loaded torchvision features.* weights into lpips' vgg slices,
    # and load the calibrated linear layers from the lpips package weights
    # (same file LPIPS(pretrained=True) uses) so behavior matches the default.
    #
    # NOTE on "no download": lpips still builds its trunk as
    # vgg16(pretrained=not pnet_rand) — i.e. pretrained=True even when we pass
    # pnet_rand=False — so torchvision downloads vgg16-397923af.pth unless the
    # torch hub cache already has it. Seed that cache from our local weights
    # file (byte-identical to the URL's payload) before constructing LPIPS.
    try:
        from torch.hub import get_dir as _hub_dir

        _ckpt_dir = os.path.join(_hub_dir(), "checkpoints")
        os.makedirs(_ckpt_dir, exist_ok=True)
        _cache_file = os.path.join(_ckpt_dir, "vgg16-397923af.pth")
        if not os.path.exists(_cache_file) and os.path.exists(weights_path):
            shutil.copyfile(weights_path, _cache_file)
    except Exception:
        pass  # best-effort: a download fallback is acceptable, not fatal

    model = LPIPS(net="vgg", pretrained=False, pnet_rand=False)
    sd = torch.load(weights_path, map_location="cpu")
    # Map each slice's child modules back to features indices: slice1=0..3,
    # slice2=4..8, slice3=9..15, slice4=16..22, slice5=23..29.
    _SLICE_RANGE = {
        1: range(4),
        2: range(4, 9),
        3: range(9, 16),
        4: range(16, 23),
        5: range(23, 30),
    }
    for sl, idxs in _SLICE_RANGE.items():
        slm = getattr(model.net, f"slice{sl}")
        for j, i in enumerate(idxs):
            child = slm[j]
            if not hasattr(child, "weight"):
                continue  # ReLU/MaxPool — no weights
            child.weight.data.copy_(sd[f"features.{i}.weight"])
            if hasattr(child, "bias") and child.bias is not None:
                child.bias.data.copy_(sd[f"features.{i}.bias"])
    # Load the LPIPS calibrated linear layers (ships inside the lpips package)
    # so the perceptual weights are identical to LPIPS(pretrained=True).
    import inspect
    _lin_path = os.path.abspath(
        os.path.join(inspect.getfile(LPIPS.__init__), "..", "weights", "v0.1", "vgg.pth")
    )
    if os.path.exists(_lin_path):
        model.load_state_dict(torch.load(_lin_path, map_location="cpu"), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def lpips_vgg_resize(pixels_pm1: torch.Tensor, input_size: int = LPIPS_VGG_INPUT_SIZE) -> torch.Tensor:
    """[-1,1] pixels -> aspect-preserving resize.

    The official PFM code resizes to a square 224x224; for non-square
    inputs we keep the geometry instead (long side -> input_size, short
    side proportional, both rounded to /16). VGG-LPIPS is fully
    convolutional and accepts any size; pred and target always share the
    same input shape (same latent -> same decode aspect), so the distance
    stays well-defined. Stretching a portrait square would distort the
    metric the same way it distorts RADIO's semantic features.
    """
    h, w = int(pixels_pm1.shape[-2]), int(pixels_pm1.shape[-1])
    scale = input_size / max(h, w)
    nh = max(16, round(h * scale / 16) * 16)
    nw = max(16, round(w * scale / 16) * 16)
    return F.interpolate(
        pixels_pm1.float(),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )


def lpips_vgg_distance(
    model: torch.nn.Module,
    pred_pm1: torch.Tensor,
    tgt_pm1: torch.Tensor,
    input_size: int = LPIPS_VGG_INPUT_SIZE,
) -> torch.Tensor:
    """LPIPS distance between [-1,1] pixel pairs, scalar mean over batch.

    Gradient context is inherited from the caller: pred carries the graph,
    target is a no-grad tensor (mirrors the official train loop, which
    decodes target pixels under torch.no_grad()).
    """
    a = lpips_vgg_resize(pred_pm1, input_size)
    b = lpips_vgg_resize(tgt_pm1, input_size)
    # The trainer moves every encoder to the shared encoder_dtype (bf16,
    # official parity: pmodel.to(device, dtype=torch.bfloat16)); feed the
    # inputs in the model's dtype.
    dtype = next(model.parameters()).dtype
    return model(a.to(dtype), b.to(dtype)).mean()
