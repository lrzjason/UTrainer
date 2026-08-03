# 003 — P2 实施总结（Hook 机制）

日期：2026-07-18 ｜ 状态：**完成，端到端验证通过**

## 交付物

### 新增

| 文件 | 内容 |
|------|------|
| `UnifiedTrainer/engine/hook_manager.py` | worker 侧 HookManager：`maybe_run(step, trainer=)` 轮询 hooks 表 → ack → 执行 → finish(result)。六类 hook：sample / sample_from_weights / save / restore / patch_config / suspend。torch/wandb 全部惰性 import；`dry_run=True` 时 save/sample/suspend 产生占位文件。suspend 保存现场后经内部 `_SuspendNow` 信号 → `SystemExit(42)`。suspend checkpoint 路径写 `task_config_kv["_meta.suspend_checkpoint"]` 供 resume 使用 |
| `UnifiedTrainer/engine/sampling.py` | 采样解耦：`run_sample(output_dir, step, tag, prompts, params, n, model_handle, generate_fn, dry_run)`。真实路径惰性 torch：eval + no_grad + `requires_grad_(False)`，结束恢复 train() 与原 grad 状态；dry-run / 无 generate_fn 时生成 1×1 占位 PNG。输出 `workspace/samples/<project>/<task>/<step>_<tag>.png`，路径写回 hook.result |
| `UnifiedTrainer/engine/hot_keys.py` | 热改白名单：放行 `training.learning_rate` 等 14 个精确键 + `losses[i].weight/use_weighting/enabled`；拒绝结构键（batch_size/lora_rank/data.*/output.dir 等）与未知键，错误信息提示"suspend → 改配置 → resume"。`apply_live()` best-effort 应用到 optimizer.param_groups.lr、loss 实例属性、trainer 属性与 config dict |
| `orchestrator/hooks.py` | 写入侧协议：`enqueue()`（校验类型 + 仅 running 任务可投递）、`resume_task()`（restart_count+1、wandb_run_name=`<task>-run-<n>`、resumed_from_run、resume_from=suspend checkpoint、resume_mode=full、置回 pending；wandb 键同时写 task_config_kv 注入物化 config） |
| `tmp/test_p2_hooks.py` | 端到端验证脚本（31 项断言全过） |

### 修改

| 文件 | 改动 |
|------|------|
| `orchestrator/db.py` | 新增 `list_hooks()`、`prompts_by_tag()` |
| `orchestrator/cli.py` | 新增 `hook <task_id> <type> [--payload JSON] [--set k=v] [--name/--path/--weights/--tag/--prompts/--n/--steps/--seed/--with-optimizer/--reset-optim]`、`hooks [--task-id]`、`resume <task_id>` 三个命令 |
| `UnifiedTrainer/engine/__init__.py` | 改为惰性导出（`__getattr__`），无 torch 环境下 hook_manager 等子模块可独立导入；`from engine import Trainer` 旧用法不变 |
| `UnifiedTrainer/engine/trainer.py` | 训练循环 step 边界（on_step_end 之后）插入 `getattr(self, "hook_manager", None)` 挂载点；旧 --config 模式该属性不存在，分支不执行，行为零变化 |
| `UnifiedTrainer/train.py` | DB 真实路径挂载 `trainer.hook_manager`；`_dry_run()` 循环内每步 `hook_mgr.maybe_run(step, config=config)`（dry_run=True） |

## 验证命令与输出摘录

| 命令 | 结果 |
|------|------|
| `python -m py_compile`（全部 9 个新/改文件） | 0 |
| `python tmp/test_p2_hooks.py` | **31/31 断言通过**，退出码 0 |
| `python tmp/smoke_test.py`（P1 回归） | `=== smoke test PASSED ===` |
| `python UnifiedTrainer/train.py --list-models` | 退出码 0（无 torch 环境为空列表，同 P1 基线） |
| `python UnifiedTrainer/train.py`（缺参） | `error: --model is required`（旧行为不变） |

E2E 关键输出（`python tmp/test_p2_hooks.py`）：

```
hook #1 save                 -> done:    out/snap1.safetensors (+ .optim)
hook #2 patch_config         -> done:    learning_rate=5e-5 applied_to=["config"]
hook #3 patch_config         -> failed:  'training.batch_size' 被拒绝：结构类参数不支持热改；请先投递 suspend hook…
hook #4 sample               -> done:    samples/p2proj/ptask/5_val.png + 5_val_1.png
hook #5 sample_from_weights  -> done:    samples/p2proj/ptask/6_valw.png
hook #6 restore              -> done
hook #7 suspend              -> done -> [hook] #7 suspend -> exit 42
dispatcher: Task 1 suspended (exit 42)

cli resume: restart_count=1 wandb_run_name=ptask-run-2 resumed_from_run=ptask-run-1
            resume_from=.../ptask_lora_suspend.safetensors
→ 任务重新 running → done，restart_count=1，_meta.output 正常写出
```

另验证：suspended 状态下投递 hook 被拒（`hook 只对 running 任务生效`）；
被拒键不写 task_config_kv；hooks 表完整流转 queued→acked→done/failed。

## 设计落地差异（as-built）

- HookManager 落在 `UnifiedTrainer/engine/hook_manager.py`（worker 侧），
  `orchestrator/hooks.py` 只做写入侧协议（enqueue/resume），与 md/03 一致。
- 真实采样回调 `generate_fn` 本阶段恒为 None（见遗留风险 1），dry-run 与
  真实模式当前都走占位路径，但 eval/no_grad/换权重换回的真实代码路径已实现。
- `stop` hook（running 取消）未实现，schema 注释中保留类型位。

## 遗留风险 / 待办

1. **真实采样未接 trainer 验证管线**：`_build_generate_fn` 返回 None。
   接 `trainer.generate_validation_images` 需要 dataloader/VAE/空 embed 上下文，
   建议在有 torch 环境联调时实现（接口已预留：`(prompts, params, out_dir) -> [path]`）。
2. **真实 save/restore/suspend 路径未实测**：CheckpointManager 调用签名按
   源码对接但未经 torch 环境验证；suspend 的真实 checkpoint 需含 optimizer
   状态且 resume 端 `--resume-full` 语义需联调确认。
3. **wandb 连续性只落字段**：run name / resumed_from_run 已写 tasks 表与
   物化 config（`wandb.run_name`、`wandb.resumed_from_run`），但真实
   `wandb.init(name=...)` 接线（WandBCallback 读 run_name 已有；id 续跑
   `resume="allow"` 未做）与 suspend 时 `wandb.finish` 未经真实验证。
4. **sample 提示词 DB 通道**：`prompts_by_tag` 已实现但暂无 prompts 表写入
   入口（CLI/API 待 P3+）；当前提示词来自 `prompts_path` 或占位。
5. **Windows 控制台编码**：worker 中文日志在 GBK 控制台下让
   `subprocess(text=True)` 默认编码炸掉（测试脚本已用
   `encoding="utf-8", errors="replace"` 规避）；P3 server 注意同样问题。

技术决定见 `agent/decisions.md`（D9–D13）。
