"""Training engine: trainer, evaluator, checkpoint, regenerator, callbacks.

P2 改为惰性导出：无 torch 环境（dry-run / orchestrator 联调）下
`UnifiedTrainer.engine.hook_manager` 等子模块可独立导入，
不再因子模块导入而连带 import trainer（torch）。
`from UnifiedTrainer.engine import Trainer` 的旧用法不变。
"""


def __getattr__(name):
    if name == "Trainer":
        from UnifiedTrainer.engine.trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Trainer"]
