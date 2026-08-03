# Reference: adapted from ai-toolkit/extensions_built_in/diffusion_models/example_model/example_model.py
# See aitoolkit_acc.md for full technique analysis.
"""
Meta-device loading helpers.

Loading a model on the ``meta`` device and then calling
``load_state_dict(..., assign=True)`` avoids the 2x memory peak that occurs
when a model is first allocated on CPU/GPU and then overwritten by the
checkpoint tensors.  Instead, parameters are created on ``meta`` (no storage)
and directly assigned the checkpoint tensors.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def load_with_meta_device(
    model_cls: type,
    *args,
    state_dict: Optional[dict] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> nn.Module:
    """Instantiate *model_cls* on the meta device, then load weights.

    Args:
        model_cls: e.g. ``Flux2KleinTransformer``
        *args, **kwargs: forwarded to the model constructor
        state_dict: checkpoint dict to load (via ``load_state_dict(assign=True)``)
        device: target device after loading ("cpu" or "cuda")
        dtype: target dtype for the model

    Returns:
        The loaded model on *device*.
    """
    with torch.device("meta"):
        model = model_cls(*args, **kwargs)

    if dtype is not None:
        model = model.to_empty(device=device, dtype=dtype)
    else:
        model = model.to_empty(device=device)

    if state_dict is not None:
        model.load_state_dict(state_dict, assign=True)
    else:
        # If no state_dict provided, materialize random weights
        model = model.to_empty(device=device)

    return model


def convert_meta_to_real(
    model: nn.Module,
    state_dict: dict,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Materialize a meta-device model with a state dict using assign=True.

    This is useful when the model was already created (e.g. by a pipeline)
    but still has meta tensors.
    """
    target_kwargs = {"device": device}
    if dtype is not None:
        target_kwargs["dtype"] = dtype
    model.to_empty(**target_kwargs)
    model.load_state_dict(state_dict, assign=True)
    return model
