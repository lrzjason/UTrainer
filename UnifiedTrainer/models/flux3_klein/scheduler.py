"""Inference scheduler for FLUX.3 klein — the official timeshift recipe, verbatim.

The official workflow drives the timestep shift through ComfyUI's
``ModelSamplingFlux`` node titled *"Timestep shift (from alpha)"*::

    alpha = 16            # float input, the 16-step distilled workflow's default
    mu    = log(alpha)    # = 2.77259
    ModelSamplingFlux(max_shift=mu, base_shift=mu, width=1024, height=1024)

With ``base_shift == max_shift`` the interpolation is constant, so ``mu`` (and
therefore ``alpha == exp(mu)``) is the same at every resolution, and the sigma
schedule is the plain shifted grid

    sigma(t) = alpha * t / (1 + (alpha - 1) * t),   t = 1 (noise) .. 0 (clean)

which is also what BFL's reference repo builds (``flux3/sampling.py:
timesteps(steps, alpha)`` with ``Pipeline.generate(..., alpha=16.0)``).

``FlowMatchEulerDiscreteScheduler`` cannot express that exactly: it applies
``shift`` **twice** — once to ``sigma_min`` in ``__init__`` and again to the
interpolation grid in ``set_timesteps`` — and its grid ends at the shifted
``sigma_min`` rather than at ``t = 0``.  At ``alpha = 6.93`` (the model's
reference shift) the last sampling sigma comes out 0.046 instead of 0.267 at
20 steps; at ``alpha = 16`` it is 0.204 instead of 0.457.  The low-noise end of
the trajectory is therefore wrong even when the ``shift`` value is right, so
this subclass emits the recipe grid directly and only borrows diffusers'
``step()``.
"""
from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# The official workflow's default alpha at 1 MP.  ("The model's reference is
# 6.93" is the base 50-step template / ComfyUI's built-in sampling_settings;
# the 16-step distilled workflow ships alpha 16.)
OFFICIAL_ALPHA = 16.0


def flux3_sigmas(alpha: float, num_inference_steps: int) -> torch.Tensor:
    """The official schedule: ``steps + 1`` sigmas, starting at 1.0, ending at 0."""
    if num_inference_steps < 1:
        raise ValueError(
            f"flux3_sigmas: num_inference_steps must be >= 1, got {num_inference_steps}"
        )
    t = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
    sigmas = alpha * t / (1.0 + (alpha - 1.0) * t)
    sigmas[-1] = 0.0  # t == 0 gives exactly 0; keep it exact in float32
    return sigmas


class Flux3FlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """FlowMatch Euler step with the FLUX.3 recipe's sigma schedule.

    Compatible with the diffusers ``set_timesteps`` signature (the trainer calls
    ``set_timesteps(steps, device=device, mu=None)``), but ignores the
    double-shift and the ``sigma_min`` floor described in the module docstring.
    """

    def __init__(
        self,
        alpha: float = OFFICIAL_ALPHA,
        num_train_timesteps: int = 1000,
    ):
        # `shift=alpha` is kept only so `config.shift` / logs report the alpha
        # actually used; the schedule is built by `set_timesteps` below.
        super().__init__(
            num_train_timesteps=num_train_timesteps, shift=float(alpha)
        )
        if alpha <= 0.0:
            raise ValueError(
                f"Flux3FlowMatchEulerDiscreteScheduler: alpha must be > 0, got {alpha}"
            )
        self.alpha = float(alpha)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        """Set the FLUX.3 recipe schedule.

        ``sigmas`` (an explicit schedule, e.g. from a caller that shifted
        already) is honoured verbatim; otherwise ``num_inference_steps`` builds
        the official ``alpha``-shifted grid.  ``mu`` / ``timesteps`` are
        accepted for signature compatibility and ignored.
        """
        if sigmas is not None:
            sig = torch.as_tensor(sigmas, dtype=torch.float32).flatten()
            if sig.numel() < 2:
                raise ValueError("sigmas must contain at least 2 values")
            if float(sig[-1]) != 0.0:
                sig = torch.cat([sig, torch.zeros(1, dtype=torch.float32)])
        else:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps (or sigmas) is required")
            sig = flux3_sigmas(self.alpha, int(num_inference_steps))

        if device is not None:
            sig = sig.to(device)

        self.num_inference_steps = int(sig.numel() - 1)
        self.sigmas = sig
        # One timestep per step: the last entry of `sigmas` is the terminal 0.
        self.timesteps = sig[:-1] * self.config.num_train_timesteps
        self.sigma_max = float(sig[0])
        self.sigma_min = float(sig[-1])
        self._step_index = None
        self._begin_index = None
