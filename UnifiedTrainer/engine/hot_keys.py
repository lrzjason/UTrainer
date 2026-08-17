"""热改配置白名单（P2 hook 机制）。

设计见 md/03-hooks.md：
- 允许：training.learning_rate、各 loss 权重 / 开关类、log/save 间隔；
- 拒绝：batch_size、分辨率、lora_rank、lokr_full_rank 等结构参数（返回明确错误，
  提示需 suspend + 修改 + resume 重启）。

本模块不 import torch，可在无 GPU / 无 torch 环境下使用与测试。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 精确放行的点路径 ──────────────────────────────────────────────
_ALLOWED_EXACT = {
    # 优化器 / 学习率
    "training.learning_rate",
    "training.lr_scheduler",
    "training.warmup_steps",
    "training.max_grad_norm",
    "training.max_steps",
    # 开关类
    "training.include_suffix",
    "training.caption_dropout",
    "training.seed",
    # log / save 间隔
    "training.log_every",
    "training.save_every",
    "training.checkpoint_every",
    "output.checkpoint_every",
    "validation.val_every_epoch",
}

# losses[<i>].weight / .use_weighting / .enabled（loss 权重与开关）
_LOSS_HOT_RE = re.compile(r"^losses\[(\d+)\]\.(weight|use_weighting|enabled)$")

# ── 明确拒绝的结构类参数（给出针对性错误提示）─────────────────────
_REJECT_STRUCTURAL = {
    "training.batch_size",
    "training.gradient_accumulation_steps",
    "training.lora_rank",
    "training.lora_alpha",
    "training.lokr_full_rank",
    "training.resolution",
    "training.optimizer",
    "training.weight_dtype",
    "training.mixed_precision",
    "training.compile",
    "model",
    "model_path",
    "output.dir",
    "output.save_name",
}

_STRUCTURAL_HINT = (
    "结构类参数不支持热改；请先投递 suspend hook 挂起任务，"
    "修改配置后 resume 重启生效"
)


def validate_key(key: str) -> Tuple[bool, Optional[str]]:
    """检查点路径 key 是否允许热改。返回 (ok, error_message)。"""
    if key in _ALLOWED_EXACT:
        return True, None
    if _LOSS_HOT_RE.match(key):
        return True, None
    if key in _REJECT_STRUCTURAL or key.startswith("data."):
        return False, f"'{key}' 被拒绝：{_STRUCTURAL_HINT}"
    # losses[i] 的非白名单字段（如 data_path、lcs_dim）也属结构参数
    if key.startswith("losses["):
        return False, f"'{key}' 被拒绝：loss 仅 weight/use_weighting/enabled 可热改；{_STRUCTURAL_HINT}"
    # 未知 key 默认拒绝（安全优先）
    return False, (
        f"'{key}' 不在热改白名单内；如确需修改，{_STRUCTURAL_HINT}，"
        f"或先将该 key 加入 engine/hot_keys.py 白名单"
    )


def set_dotted(config: dict, dotted_key: str, value: Any) -> bool:
    """把点路径值写进 config dict（losses[i].field 形式也支持）。

    KI-13：返回是否真正写入。losses[i] 越界或该位置不是 dict 时
    返回 False（此前静默跳过导致调用方误以为已生效）。
    """
    m = _LOSS_HOT_RE.match(dotted_key)
    if m:
        idx, field = int(m.group(1)), m.group(2)
        losses = config.get("losses")
        if (isinstance(losses, list) and idx < len(losses)
                and isinstance(losses[idx], dict)):
            losses[idx][field] = value
            return True
        return False
    parts = dotted_key.split(".")
    node = config
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value
    return True


# trainer 属性直改映射（best-effort；不存在的属性跳过）
_TRAINER_ATTR_MAP = {
    "training.learning_rate": "learning_rate",
    "training.max_steps": "max_steps",
    "training.max_grad_norm": "max_grad_norm",
}


def apply_live(trainer: Any, config: Optional[dict], key: str,
               value: Any) -> Tuple[list, list]:
    """把已校验通过的热改应用到运行中的 trainer / config。

    KI-13：返回 (applied, skipped) 两个路径说明列表——set_dotted 越界/
    未命中、losses 下标不存在、trainer 属性缺失等不再静默，skipped 会
    写进 hook.result，调用方可感知部分失败。
    所有动作 best-effort，失败只记日志不抛错。
    """
    applied: list = []
    skipped: list = []
    if config is not None:
        if set_dotted(config, key, value):
            applied.append("config")
        else:
            skipped.append("config (index out of range / not a dict)")

    if trainer is None:
        return applied, skipped

    # 优化器学习率：真正生效点是 optimizer.param_groups
    if key == "training.learning_rate":
        try:
            trainer.learning_rate = value
            if getattr(trainer, "optimizer", None) is not None:
                for g in trainer.optimizer.param_groups:
                    g["lr"] = value
                applied.append("optimizer.param_groups.lr")
        except Exception as e:  # pragma: no cover
            logger.warning(f"hot-patch lr to optimizer failed: {e}")

    m = _LOSS_HOT_RE.match(key)
    if m:
        idx, field = int(m.group(1)), m.group(2)
        try:
            losses = getattr(trainer, "losses", [])
            if idx < len(losses):
                setattr(losses[idx], field, value)
                applied.append(f"losses[{idx}].{field}")
            else:
                skipped.append(
                    f"losses[{idx}].{field} (trainer 只有 {len(losses)} 个 loss)")
        except Exception as e:  # pragma: no cover
            logger.warning(f"hot-patch loss {idx}.{field} failed: {e}")

    attr = _TRAINER_ATTR_MAP.get(key)
    if attr is not None and hasattr(trainer, attr):
        try:
            setattr(trainer, attr, value)
            applied.append(f"trainer.{attr}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"hot-patch trainer.{attr} failed: {e}")

    if key == "training.include_suffix" and hasattr(trainer, "_include_suffix"):
        trainer._include_suffix = bool(value)
        applied.append("trainer._include_suffix")

    return applied, skipped
