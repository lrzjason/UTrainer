"""
VAEManager — the single-owner VAE component for a training run.

Every VAE consumer (perceptual losses such as pfm, validation image sampling,
regenerators, ...) shares ONE instance owned by this manager. When any
configured component declares it needs the VAE, train.py creates the manager.
The VAE is always frozen (eval + requires_grad_(False)): it never trains and
never receives weight gradients. Input gradients still flow THROUGH the
decoder when a caller decodes with autograd enabled (perceptual losses rely
on this) — freezing weights and blocking input grads are independent things.

Load timing is governed by ``training.vae_load_mode`` (explicit config always
wins; the default is resolved by :func:`resolve_default_load_mode` — "lazy"
when a requires_vae loss such as pfm is configured, "reload" otherwise):

    "reload": per-use loading — EVERY acquire reads the VAE fresh
        from disk and every release drops the instance completely. Between
        uses the VAE occupies neither VRAM nor RAM. This matches the
        pre-manager pipeline exactly: the VAE only ever existed transiently
        during validation image generation. Callers may follow release()
        with torch.cuda.empty_cache() to return VRAM immediately
        (trainer.generate_validation_images does).
    "lazy": dynamic loading — construction and :meth:`load` are no-ops; the
        VAE is read from disk on the FIRST acquire and then stays RAM-resident
        (a one-time mid-training stall in exchange for a faster startup).
        Default for runs with a VAE-consuming loss firing every step.
    "eager": the VAE loads into RAM before the training loop and stays
        there. train.py calls :meth:`load` once up front. Zero stalls during
        the run at the cost of standing RAM.

GPU residency is governed by ``training.vae_on_device`` (eager/lazy only):

    False (default): "transfer" — :meth:`acquire` moves the VAE to the GPU on
        demand and :meth:`release` moves it back to RAM after use. Wrap each
        use site with :meth:`device_vae` (or acquire/release pairs).
    True: resident — the VAE is pinned on the GPU; acquire and release become
        no-ops.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_LOAD_MODES = ("eager", "lazy", "reload")


def resolve_default_load_mode(losses_need_vae: bool) -> str:
    """训练配置未显式指定 vae_load_mode 时的默认值。

    - 配了 requires_vae 的 loss（如 pfm）：默认 "lazy" —— 动态加载、首次
      使用后常驻 RAM，避免每步训练都读盘（reload 会显著拖慢 step）。
    - 仅有 val image gen 之类低频消费方：默认 "reload" —— 与引入 manager
      之前的旧管线一致（只有出图时现场加载，用完卸载）。
    显式配置 training.vae_load_mode 永远优先于这里的默认值。
    """
    return "lazy" if losses_need_vae else "reload"


class VAEManager:
    """Own THE VAE instance; place it on GPU on demand or keep it resident."""

    def __init__(
        self,
        adapter,
        vae_path: str,
        dtype: torch.dtype = torch.bfloat16,
        on_device: bool = False,
        load_mode: str = "reload",
    ):
        if load_mode not in _LOAD_MODES:
            raise ValueError(
                f"vae_load_mode must be one of {_LOAD_MODES}, got {load_mode!r}"
            )
        self._adapter = adapter
        self._path = vae_path
        self._dtype = dtype
        self.on_device = bool(on_device)
        self.load_mode = load_mode
        self._vae: Optional[torch.nn.Module] = None

    # ── ownership ─────────────────────────────────────────────────────

    def _load_frozen(self) -> torch.nn.Module:
        """One disk read -> frozen VAE (never trains, no weight grads)."""
        vae = self._adapter.load_vae(self._path, self._dtype)
        vae.eval()
        vae.requires_grad_(False)
        return vae

    @property
    def vae(self) -> torch.nn.Module:
        """The RAM-resident VAE, loaded on first access (eager/lazy modes)."""
        if self._vae is None:
            self._vae = self._load_frozen()
            logger.info(f"VAE loaded into RAM: {self._path}")
        return self._vae

    def load(self, device: Optional[torch.device] = None) -> "VAEManager":
        """Load into RAM now; pin to `device` when resident mode is on.

        Deliberate no-op in lazy/reload modes — loading happens per-acquire
        instead (first acquire for lazy, every acquire for reload).
        """
        if self.load_mode in ("lazy", "reload"):
            logger.info(
                f"VAEManager: {self.load_mode} mode — skipping eager load"
            )
            return self
        self.vae
        if self.on_device and device is not None:
            self._ensure_on_device(device)
            logger.info("VAE pinned on GPU (vae_on_device=true)")
        return self

    def free(self) -> None:
        """Drop the instance entirely (RAM and GPU)."""
        self._vae = None

    # ── placement ─────────────────────────────────────────────────────

    def _vae_device(self) -> Optional[torch.device]:
        if self._vae is None:
            return None
        try:
            return self._vae.device  # diffusers ModelMixin property
        except AttributeError:
            return next(self._vae.parameters()).device

    def _ensure_on_device(self, device: torch.device) -> None:
        if self._vae_device() != device:
            self.vae.to(device)

    def acquire(self, device: torch.device) -> torch.nn.Module:
        """Return the VAE on `device`, loading it first if the mode demands."""
        if self.load_mode == "reload":
            # 每次使用都重新从磁盘加载；上一次的实例已在 release() 卸载，
            # 两次使用之间不占显存也不占内存。
            vae = self._load_frozen()
            logger.info("VAE reloaded from disk (reload mode)")
            vae.to(device)
            self._vae = vae  # tracked so release() can unload it
            return vae
        # eager/lazy: cached RAM-resident instance
        if self.on_device:
            self.load(device)  # idempotent pin (resident mode)
            return self.vae
        self._ensure_on_device(device)
        return self.vae

    def release(self) -> None:
        """Return the VAE to RAM — or unload it completely (reload mode)."""
        if self.load_mode == "reload":
            # 完全卸载：丢弃唯一托管引用，RAM/VRAM 交由 GC 回收（调用方可
            # 追加 torch.cuda.empty_cache() 立即归还显存）。
            self._vae = None
            return
        if self.on_device or self._vae is None:
            return
        dev = self._vae_device()
        if dev is not None and dev.type != "cpu":
            self._vae.to("cpu")

    @contextmanager
    def device_vae(self, device: torch.device):
        """``with mgr.device_vae(dev) as vae: ...`` — acquire + release."""
        vae = self.acquire(device)
        try:
            yield vae
        finally:
            self.release()
