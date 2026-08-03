"""Hook 写入侧协议助手（P2）。

hooks 表是唯一通道：CLI / 前端 / inbox 指令文件统一经这里投递；
worker 侧消费逻辑在 UnifiedTrainer/engine/hook_manager.py。

供 cli.py 与后续 FastAPI（P3+）复用，保证"前端与 CLI 零差异"。
"""

from __future__ import annotations

from typing import Any, Optional

from .db import DB

# 已实现的 hook 类型（worker 侧 HookManager 支持）
KNOWN_TYPES = {
    "sample", "sample_from_weights", "save", "restore",
    "patch_config", "suspend",
    # "stop"  # 运行中取消，后续阶段实现
}


class HookError(ValueError):
    pass


def enqueue(db: DB, task_id: int, htype: str,
            payload: Optional[dict] = None) -> int:
    """校验任务与类型后投递 hook，返回 hook id。"""
    if htype not in KNOWN_TYPES:
        raise HookError(
            f"unknown hook type '{htype}'; known: {sorted(KNOWN_TYPES)}")
    task = db.get_task(task_id)
    if task is None:
        raise HookError(f"task not found: {task_id}")
    if task["status"] not in ("running",):
        raise HookError(
            f"task {task_id} status={task['status']}；hook 只对 running "
            f"任务生效（suspend 后请用 resume）")
    return db.enqueue_hook(task_id, htype, payload or {})


def set_config_and_notify(db: DB, task_id: int, key: str,
                          value: Any) -> Optional[int]:
    """写 task_config_kv 的统一入口（KI-04 修复）。

    - 先写 kv（任何状态都允许，非 running 任务下次启动/重物化时生效）；
    - 任务 running 时追加投一条 patch_config hook（payload={key: value}），
      走 enqueue 的统一校验，worker 侧按热改白名单即时生效或拒绝；
    - "_" 前缀的元数据键不注入训练 config，不投 hook。

    返回 hook id；未投 hook 返回 None。
    供 watcher cmd set_config / cli config set / server PATCH config 复用。
    """
    db.set_config_kv(task_id, key, value)
    if key.startswith("_"):
        return None
    task = db.get_task(task_id)
    if task is None:
        raise HookError(f"task not found: {task_id}")
    if task["status"] != "running":
        return None
    return enqueue(db, task_id, "patch_config", {key: value})


def resume_task(db: DB, task_id: int) -> dict:
    """重启 suspended 任务：restart_count+1、生成 wandb run name、
    记录 resumed_from_run、状态置回 pending（dispatcher 自动重派）。

    返回 {"restart_count", "wandb_run_name", "resumed_from_run", "resume_from"}。
    """
    task = db.get_task(task_id)
    if task is None:
        raise HookError(f"task not found: {task_id}")
    if task["status"] != "suspended":
        raise HookError(
            f"task {task_id} status={task['status']}，仅 suspended 任务可 resume")

    kv = db.get_config_kv(task_id)
    ckpt = kv.get("_meta.suspend_checkpoint") or task.get("resume_from")
    old_restart = task.get("restart_count") or 0
    new_restart = old_restart + 1
    # 约定：第 n 次运行（restart_count = n-1）run name = <task>-run-<n>
    # as-built 备注（环境限制，不改行为）：resumed_from_run 存的是 wandb
    # run **name** 而非 run id——本环境 wandb 为惰性 import，orchestrator
    # 侧拿不到 wandb 分配的 run id，只能用自己生成的确定性 name 串联
    # 续跑谱系。若未来 worker 回写真实 run id 到 kv，可切换为 id。
    prev_run = task.get("wandb_run_name") or f"{task['name']}-run-{old_restart + 1}"
    run_name = f"{task['name']}-run-{new_restart + 1}"

    db.set_task_status(
        task_id, "pending",
        restart_count=new_restart,
        wandb_run_name=run_name,
        resume_from=ckpt,
        resume_mode="full",
        error=None,
        finished_at=None,
    )
    # 物化 config 时注入 wandb 配置（真实 wandb.init 惰性消费这些键）
    db.set_config_kv(task_id, "wandb.run_name", run_name)
    db.set_config_kv(task_id, "wandb.resumed_from_run", prev_run)

    return {
        "restart_count": new_restart,
        "wandb_run_name": run_name,
        "resumed_from_run": prev_run,
        "resume_from": ckpt,
    }
