"""
Callbacks - logging, wandb, tensorboard, and progress tracking for training.

Provides a simple callback system that the Trainer invokes at key points:
   - on_epoch_start(epoch)
   - on_step_end(step, loss)
   - on_epoch_end(epoch, avg_loss)
   - on_checkpoint(step)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class. Override methods to hook into training."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        pass

    def on_step_end(self, step: int, loss: float, trainer: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float, trainer: Any) -> None:
        pass

    def on_checkpoint(self, step: int, path: str, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


class LoggingCallback(Callback):
    """Console logging callback."""

    def __init__(self, log_every: int = 50):
        self.log_every = log_every

    def on_step_end(self, step: int, loss: float, trainer: Any) -> None:
        if step % self.log_every == 0:
            lr = trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else 0
            step_str = f"{step}/{trainer.max_steps}" if trainer.max_steps != -1 else f"{step}/inf"
            logger.info(
                f"Step {step_str} | Loss: {loss:.6f} | LR: {lr:.2e}"
            )

    def on_epoch_end(self, epoch: int, avg_loss: float, trainer: Any) -> None:
        step_str = f"{trainer.step}/{trainer.max_steps}" if trainer.max_steps != -1 else f"{trainer.step}/inf"
        logger.info(
            f"Epoch {epoch} complete | Avg Loss: {avg_loss:.6f} | "
            f"Step: {step_str}"
        )


class WandBCallback(Callback):
    """Weights & Biases logging callback.

    Logs:
       - Per-step: loss, learning_rate
       - Per-epoch: epoch_loss
       - Per-checkpoint: artifact upload
       - On train end: finish run

    Config:
        config["wandb"] = {
            "enabled": true,
            "project": "UnifiedTrainer",
            "run_name": "krea2_experiment_1",   # optional
        }
    """

    def __init__(
        self,
        project: str = "UnifiedTrainer",
        config: Optional[dict] = None,
        run_name: Optional[str] = None,
    ):
        self.project = project
        self.config = config or {}
        self.run_name = run_name
        self._wandb = None
        self._initialized = False

    def _init_wandb(self):
        if self._initialized:
            return
        try:
            import wandb
            self._wandb = wandb
            init_kwargs = {"project": self.project, "config": self.config}
            if self.run_name:
                init_kwargs["name"] = self.run_name
            wandb.init(**init_kwargs)
            self._initialized = True
        except ImportError:
            logger.warning("wandb not installed, skipping WandBCallback")

    def on_train_start(self, trainer: Any) -> None:
        self._init_wandb()

    def on_step_end(self, step: int, loss: float, trainer: Any) -> None:
        if self._wandb:
            log_dict = {"loss": loss, "step": step}
            # Log learning rate if scheduler is available
            if hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler:
                lr = trainer.lr_scheduler.get_last_lr()[0]
                log_dict["learning_rate"] = lr
            # Log per-loss breakdown (e.g. flow_matching, lisa_depth)
            breakdown = getattr(trainer, "last_loss_breakdown", {})
            for loss_name, loss_val in breakdown.items():
                log_dict[f"loss/{loss_name}"] = loss_val
            self._wandb.log(log_dict, step=step)

    def on_epoch_end(self, epoch: int, avg_loss: float, trainer: Any) -> None:
        if self._wandb:
            self._wandb.log({"epoch_loss": avg_loss, "epoch": epoch}, step=getattr(trainer, "step", 0))

    def on_checkpoint(self, step: int, path: str, trainer: Any) -> None:
        if self._wandb:
            artifact = self._wandb.Artifact(
                name=f"checkpoint-{step}", type="model"
            )
            artifact.add_file(path)
            self._wandb.log_artifact(artifact)

    def on_train_end(self, trainer: Any) -> None:
        if self._wandb:
            self._wandb.finish()


class TensorBoardCallback(Callback):
    """TensorBoard logging callback.

    Logs:
       - Per-step: loss, learning_rate, per-loss breakdown, VRAM
       - Per-epoch: epoch_loss
       - Scalars grouped under train/ and val/ namespaces

    Config:
        config["reporter"] = {
            "type": "tensorboard",
            "port": 6006,
            "log_dir": null,       # defaults to {output_dir}/tensorboard
            "log_every": 1,        # log every N steps
        }
    """

    def __init__(
        self,
        log_dir: str = "runs",
        port: int = 6006,
        log_every: int = 1,
        comment: str = "",
    ):
        self.log_dir = log_dir
        self.port = port
        self.log_every = max(1, log_every)
        self.comment = comment
        self._writer = None

    def _init_writer(self):
        if self._writer is not None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(
                log_dir=self.log_dir,
                comment=self.comment,
            )
            logger.info(
                f"TensorBoard logging to: {self.log_dir}\n"
                f"  Launch with: tensorboard --logdir {self.log_dir} --port {self.port}"
            )
        except ImportError:
            logger.warning(
                "tensorboard not installed, skipping TensorBoardCallback. "
                "Install with: pip install tensorboard"
            )

    def on_train_start(self, trainer: Any) -> None:
        self._init_writer()
        if self._writer:
            # Log hyperparameters as text
            config = getattr(trainer, "_config", None)
            if config:
                training_cfg = config.get("training", {})
                hparams = {
                    "model": config.get("model", "unknown"),
                    "network_type": training_cfg.get("network_type", "lora"),
                    "learning_rate": training_cfg.get("learning_rate", 0),
                    "batch_size": training_cfg.get("batch_size", 1),
                    "optimizer": training_cfg.get("optimizer", "adamw"),
                    "num_epochs": training_cfg.get("num_epochs", 0),
                    "quantize": training_cfg.get("quantize", "none"),
                }
                self._writer.add_text(
                    "hparams",
                    "\n".join(f"- **{k}**: {v}" for k, v in hparams.items()),
                )

    def on_step_end(self, step: int, loss: float, trainer: Any) -> None:
        if not self._writer or step % self.log_every != 0:
            return
        self._writer.add_scalar("train/loss", loss, step)
        # Learning rate
        if hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler:
            lr = trainer.lr_scheduler.get_last_lr()[0]
            self._writer.add_scalar("train/learning_rate", lr, step)
        # Per-loss breakdown
        breakdown = getattr(trainer, "last_loss_breakdown", {})
        for loss_name, loss_val in breakdown.items():
            self._writer.add_scalar(f"train/loss_{loss_name}", loss_val, step)
        # VRAM usage
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / 1024**3
                self._writer.add_scalar("system/vram_gb", vram_gb, step)
        except Exception:
            pass

    def on_epoch_end(self, epoch: int, avg_loss: float, trainer: Any) -> None:
        if not self._writer:
            return
        step = getattr(trainer, "step", epoch)
        self._writer.add_scalar("train/epoch_loss", avg_loss, step)
        self._writer.add_scalar("train/epoch", epoch, step)

    def on_checkpoint(self, step: int, path: str, trainer: Any) -> None:
        if self._writer:
            self._writer.add_text(
                "checkpoints",
                f"step={step}: `{os.path.basename(path)}`",
                step,
            )

    def on_train_end(self, trainer: Any) -> None:
        if self._writer:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class CallbackList:
    """Manage and dispatch to multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(epoch, trainer)

    def on_step_end(self, step: int, loss: float, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_step_end(step, loss, trainer)

    def on_epoch_end(self, epoch: int, avg_loss: float, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, avg_loss, trainer)

    def on_checkpoint(self, step: int, path: str, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint(step, path, trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)
