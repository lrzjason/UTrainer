"""HookManager（worker 侧，P2）。

训练循环每个 step 结束调用 ``maybe_run(step, trainer=...)``：
轮询 hooks 表取本 task 的 queued 指令 → ack → 执行 → finish(result)。

六类 hook（md/03-hooks.md）：
- ``sample``               用当前内存中权重采样（占位或真实）
- ``sample_from_weights``  换入指定权重采样后换回（真实路径 best-effort）
- ``save``                 立即落盘权重，不中断训练
- ``restore``              加载权重；可选重置优化器
- ``patch_config``         白名单内热改（engine/hot_keys.py）+ 写 task_config_kv
- ``suspend``              full checkpoint → wandb.finish（惰性）→ SystemExit(42)

所有 torch / wandb import 惰性；``dry_run=True`` 时 save/sample/suspend
均产生占位文件，全流程无需 torch 即可端到端验证（同 P1 --dry-run-steps 思路）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from . import hot_keys
from . import sampling

logger = logging.getLogger(__name__)

EXIT_SUSPENDED = 42  # 与 dispatcher 约定（agent/decisions.md D7）


class HookManager:
    """worker 进程内的 hook 消费者。只在 DB 模式挂载。"""

    def __init__(self, db, task_id: int, config: Optional[dict] = None,
                 workspace: Optional[str] = None, dry_run: bool = False):
        self.db = db
        self.task_id = task_id
        self.config = config if config is not None else {}
        self.dry_run = dry_run
        # db 默认位于 <workspace>/trainer.db
        self.workspace = os.path.abspath(
            workspace) if workspace else os.path.dirname(db.path)

        task = db.get_task(task_id) or {}
        self.task_name = task.get("name", f"task{task_id}")
        self.project_id = task.get("project_id")
        project = db.get_project(self.project_id) if task else None
        self.project_name = (project or {}).get("name", "default")

        out_cfg = self.config.get("output", {})
        self.output_dir = out_cfg.get("dir", "output")
        self.save_name = out_cfg.get("save_name", "lora")

        # KI-19：worker（重）启动时回收本任务 acked 但未 done 的 hooks——
        # 上一次运行在 ack 后崩溃会让 hook 永远卡在 acked，重置回 queued
        # 重新消费（hook 执行本身幂等性由类型语义保证：采样/保存重做一次
        # 无副作用，suspend 重投会再次退出）。
        reclaimed = self.db.requeue_stale_acked_hooks(self.task_id)
        if reclaimed:
            logger.info(f"[hook] reclaimed {reclaimed} stale acked hook(s) "
                        f"for task {self.task_id}")

    # ── 主入口：训练循环 step 边界调用 ────────────────────────────
    def maybe_run(self, step: int, trainer: Any = None,
                  config: Optional[dict] = None) -> int:
        """消费本 task 全部 queued hooks。返回执行条数。

        suspend hook 在保存现场后 raise SystemExit(42) 终止进程。
        """
        if config is not None:
            self.config = config
        hooks = self.db.fetch_queued_hooks(self.task_id)
        for hook in hooks:
            self.db.ack_hook(hook["id"])
            logger.info(
                f"[hook] #{hook['id']} {hook['type']} acked at step {step}")
            try:
                payload = json.loads(hook["payload"] or "{}")
            except json.JSONDecodeError as e:
                self.db.finish_hook(hook["id"], f"bad payload JSON: {e}",
                                    failed=True)
                continue
            try:
                result = self._execute(hook["type"], payload, step, trainer)
                self.db.finish_hook(hook["id"], json.dumps(
                    result, ensure_ascii=False))
                logger.info(f"[hook] #{hook['id']} {hook['type']} done")
            except _SuspendNow as s:
                self.db.finish_hook(hook["id"], json.dumps(
                    {"suspended": True, "checkpoint": s.checkpoint},
                    ensure_ascii=False))
                logger.info(f"[hook] #{hook['id']} suspend -> exit 42")
                raise SystemExit(EXIT_SUSPENDED)
            except Exception as e:
                logger.warning(f"[hook] #{hook['id']} {hook['type']} failed: {e}")
                self.db.finish_hook(hook["id"], str(e), failed=True)
        return len(hooks)

    # ── 分发 ─────────────────────────────────────────────────────
    def _execute(self, htype: str, payload: dict, step: int,
                 trainer: Any) -> dict:
        handler = {
            "sample": self._do_sample,
            "sample_from_weights": self._do_sample_from_weights,
            "save": self._do_save,
            "restore": self._do_restore,
            "patch_config": self._do_patch_config,
            "suspend": self._do_suspend,
        }.get(htype)
        if handler is None:
            raise ValueError(f"unknown hook type: {htype}")
        return handler(payload, step, trainer)

    # ── save ─────────────────────────────────────────────────────
    def _do_save(self, payload: dict, step: int, trainer: Any) -> dict:
        name = payload.get("name") or f"hook_{self.save_name}_step{step}"
        with_optimizer = bool(payload.get("with_optimizer", False))
        os.makedirs(self.output_dir, exist_ok=True)

        if self.dry_run or trainer is None:
            path = os.path.abspath(
                os.path.join(self.output_dir, f"{name}.safetensors"))
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"DRY-RUN placeholder weights (hook save, step {step})\n")
            # KI-19：dry-run 的 result 字段与真实路径对齐
            # （真实路径用 optimizer_state 键返回优化器状态路径）
            result = {"path": path, "dry_run": True}
            if with_optimizer:
                optim_path = path + ".optim"
                with open(optim_path, "w", encoding="utf-8") as f:
                    f.write("DRY-RUN placeholder optimizer state\n")
                result["optimizer_state"] = optim_path
            return result

        from UnifiedTrainer.engine.checkpoint import CheckpointManager  # 惰性
        ckpt = CheckpointManager(self.output_dir, name)
        path = ckpt.save_lora(trainer.transformer, step,
                              getattr(trainer, "epoch", 0), self.config)
        result = {"path": str(path)}
        if with_optimizer and getattr(trainer, "optimizer", None) is not None:
            state_path = ckpt.save_training_state(
                trainer.optimizer, trainer.lr_scheduler, step,
                getattr(trainer, "epoch", 0), getattr(trainer, "epoch", 0),
                self.config)
            result["optimizer_state"] = str(state_path)
        return result

    # ── restore ──────────────────────────────────────────────────
    def _do_restore(self, payload: dict, step: int, trainer: Any) -> dict:
        path = payload.get("path")
        if not path:
            raise ValueError("restore hook requires payload.path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"weights not found: {path}")
        reset_optimizer = bool(payload.get("reset_optimizer", False))

        if self.dry_run or trainer is None:
            return {"restored": os.path.abspath(path),
                    "reset_optimizer": reset_optimizer, "dry_run": True}

        from UnifiedTrainer.engine.checkpoint import CheckpointManager  # 惰性
        ckpt = CheckpointManager(self.output_dir, self.save_name)
        ckpt.load_lora(trainer.transformer, path)
        result = {"restored": os.path.abspath(path)}
        if reset_optimizer:
            trainer.setup_optimizer()
            trainer.setup_lr_scheduler()
            result["reset_optimizer"] = True
        return result

    # ── sample / sample_from_weights ──────────────────────────────
    def _sample_prompts(self, payload: dict) -> list:
        """提示词来源：payload.prompts_path 文件 > DB prompts 表按 tag > 占位。"""
        path = payload.get("prompts_path")
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"prompts_path not found: {path}")
            prompts = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        try:
                            obj = json.loads(line)
                            # KI-19：JSON 行不一定是 dict（可能是 list/
                            # str/num），非 dict 时按原文本行处理
                            if isinstance(obj, dict):
                                prompts.append(obj.get("text", line))
                            else:
                                prompts.append(line)
                            continue
                        except json.JSONDecodeError:
                            pass
                    prompts.append(line)
            if prompts:
                return prompts
        tag = payload.get("tag")
        if tag:
            # KI-19：按本任务所属项目过滤，不再跨项目取提示词
            rows = self.db.prompts_by_tag(self.project_id, tag)
            texts = [r["text"] for r in rows]
            if texts:
                return texts
        return [f"(placeholder prompt, tag={tag or 'none'})"]

    def _do_sample(self, payload: dict, step: int, trainer: Any) -> dict:
        tag = payload.get("tag", "sample")
        out_dir = os.path.join(self.workspace, "samples",
                               self.project_name, self.task_name)
        files = sampling.run_sample(
            output_dir=out_dir, step=step, tag=tag,
            prompts=self._sample_prompts(payload),
            params={"steps": payload.get("steps"),
                    "seed": payload.get("seed")},
            n=int(payload.get("n", 1)),
            model_handle=trainer,
            generate_fn=self._build_generate_fn(trainer),
            dry_run=self.dry_run or trainer is None,
        )
        return {"files": files, "tag": tag, "step": step}

    def _do_sample_from_weights(self, payload: dict, step: int,
                                trainer: Any) -> dict:
        weights = payload.get("weights_path")
        if not weights:
            raise ValueError(
                "sample_from_weights requires payload.weights_path")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"weights not found: {weights}")

        if self.dry_run or trainer is None:
            # dry-run：不真正换权重，占位采样即可
            return self._do_sample(payload, step, trainer) | {
                "weights_path": os.path.abspath(weights), "dry_run": True}

        # 真实路径：换入 → 采样 → 换回（best-effort，惰性 torch）
        from UnifiedTrainer.engine.checkpoint import CheckpointManager
        transformer = trainer.transformer
        backup = self._backup_adapter_params(transformer)
        try:
            ckpt = CheckpointManager(self.output_dir, self.save_name)
            ckpt.load_lora(transformer, weights)
            result = self._do_sample(payload, step, trainer)
        finally:
            transformer.load_state_dict(backup, strict=False)
            del backup
        result["weights_path"] = os.path.abspath(weights)
        return result

    @staticmethod
    def _backup_adapter_params(transformer: Any) -> dict:
        """KI-12：备份"实际将被 load_lora 触及的参数键集合"。

        优先通过 PEFT 的 get_peft_model_state_dict 查询 adapter 实际键
        （与 load_lora 的 PEFT 加载路径键空间一致）；拿不到（非 PEFT
        模型 / peft 不可用）时退化为宽松名称匹配（lora/adapter，大小写
        不敏感）并记录警告——备份范围只多不少，保证换入后完整换回。
        """
        sd = transformer.state_dict()
        keys: set = set()
        try:
            from peft import get_peft_model_state_dict  # 惰性
            keys = {k for k in get_peft_model_state_dict(transformer).keys()
                    if k in sd}
            if keys:
                logger.info(f"[hook] adapter backup via PEFT state dict: "
                            f"{len(keys)} keys")
        except Exception as e:
            logger.warning(f"[hook] PEFT adapter key query failed ({e}); "
                           f"falling back to name-based matching")
        if not keys:
            keys = {k for k in sd
                    if "lora" in k.lower() or "adapter" in k.lower()}
            logger.warning(
                f"[hook] adapter backup fallback: name-matched {len(keys)} "
                f"keys ('lora'/'adapter'); 若适配器参数使用其他命名，"
                f"换回可能不完整")
        return {k: sd[k].detach().clone() for k in keys}

    def _build_generate_fn(self, trainer: Any):
        """真实采样回调。无 trainer（dry-run）时返回 None。

        复用 trainer 的验证图生成管线需要 dataloader/VAE 等上下文，
        本阶段不强行接入（见 progress/003 遗留风险）；返回 None 时
        sampling.run_sample 走占位路径，hook 协议仍可完整验证。
        """
        return None

    # ── patch_config ─────────────────────────────────────────────
    def _do_patch_config(self, payload: dict, step: int,
                         trainer: Any) -> dict:
        if not payload:
            raise ValueError("patch_config requires non-empty payload "
                             "{dot.key: value}")
        applied, rejected = {}, {}
        for key, value in payload.items():
            ok, err = hot_keys.validate_key(key)
            if not ok:
                rejected[key] = err
                continue
            # 先落 DB（worker 运行期 config 以 task_config_kv 为准）
            self.db.set_config_kv(self.task_id, key, value)
            # KI-13：apply_live 返回 (applied, skipped)，越界/未命中
            # 不再静默，如实写进 result
            paths, skipped = hot_keys.apply_live(trainer, self.config,
                                                 key, value)
            entry = {"value": value, "applied_to": paths}
            if skipped:
                entry["skipped"] = skipped
            applied[key] = entry
        if rejected and not applied:
            # 全部被拒：hook 标 failed，错误信息进 result
            raise ValueError("; ".join(rejected.values()))
        result = {"applied": applied}
        if rejected:
            result["rejected"] = rejected
        return result

    # ── suspend ──────────────────────────────────────────────────
    def _do_suspend(self, payload: dict, step: int, trainer: Any) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.dry_run or trainer is None:
            ckpt_path = os.path.abspath(os.path.join(
                self.output_dir, f"{self.save_name}_suspend.safetensors"))
            with open(ckpt_path, "w", encoding="utf-8") as f:
                f.write(f"DRY-RUN placeholder full checkpoint "
                        f"(suspend at step {step})\n")
            with open(ckpt_path + ".training_state", "w",
                      encoding="utf-8") as f:
                f.write("DRY-RUN placeholder optimizer/scheduler/RNG state\n")
        else:
            from UnifiedTrainer.engine.checkpoint import CheckpointManager
            ckpt = CheckpointManager(self.output_dir, self.save_name)
            ckpt_path = str(ckpt.save_lora(
                trainer.transformer, step, getattr(trainer, "epoch", 0),
                self.config))
            if getattr(trainer, "optimizer", None) is not None:
                ckpt.save_training_state(
                    trainer.optimizer, trainer.lr_scheduler, step,
                    getattr(trainer, "epoch", 0), getattr(trainer, "epoch", 0),
                    self.config)

        # resume 时由 orchestrator 读此键写回 tasks.resume_from
        self.db.set_config_kv(self.task_id, "_meta.suspend_checkpoint",
                              ckpt_path)
        self._wandb_finish_lazy()
        raise _SuspendNow(ckpt_path)

    @staticmethod
    def _wandb_finish_lazy() -> None:
        """wandb.finish()；未安装 / 未启用时静默跳过。"""
        try:
            import wandb  # noqa: 惰性导入
            if getattr(wandb, "run", None) is not None:
                wandb.finish()
                logger.info("[hook] wandb.run finished")
        except ImportError:
            pass
        except Exception as e:  # pragma: no cover
            logger.warning(f"[hook] wandb.finish failed (ignored): {e}")


class _SuspendNow(Exception):
    """内部信号：suspend 现场已保存，立即以退出码 42 终止进程。"""

    def __init__(self, checkpoint: str):
        super().__init__(checkpoint)
        self.checkpoint = checkpoint
