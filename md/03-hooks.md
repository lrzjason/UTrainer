# 03 — Hook 机制设计（P2）

> 状态：**已实现**（2026-07-18，见 progress/003-p2-hooks.md）。
> as-built 差异：① HookManager 位于 `UnifiedTrainer/engine/hook_manager.py`，
> `orchestrator/hooks.py` 仅写入侧协议（enqueue/resume，见 D10）；
> ② 真实采样回调 generate_fn 留接口未接 trainer 验证管线（D13）；
> ③ wandb 连续性本阶段只落 DB/config 字段，真实 init/finish 惰性（D11）；
>    且 as-built `resumed_from_run` 存的是上一个 **run name** 字符串而非 run id
>    （下文 §wandb 连续性按此口径）；
> ③b hook 为**六类**（含 sample_from_weights 独立类型），下文标题"六类 hook"；
> ④ 热改白名单为"默认拒绝"策略（D12）；⑤ `stop` hook 未实现。

## 范围

`orchestrator/hooks.py`（worker 侧 HookManager）+ `UnifiedTrainer/engine/sampling.py`（抽离）
+ `engine/hot_keys.py`（热改白名单）+ trainer.py 挂载点。

## 协议

hooks 表为唯一通道：写入方（CLI/前端/inbox）`enqueue_hook`，
worker 每 step 结束 `fetch_queued_hooks → ack → 执行 → finish(result)`。

## 六类 hook

| type | 行为 | payload 要点 |
|------|------|--------------|
| `sample` | 用当前内存中权重采样 | tag / prompts_path / n / steps / seed |
| `sample_from_weights` | 临时换入指定 LoRA 权重采样后换回 | weights_path + 同上 |
| `save` | 立即落盘权重（±优化器）不中断 | name, with_optimizer |
| `restore` | 加载权重；可选重置优化器 | path, reset_optimizer |
| `patch_config` | 白名单内 key 热改 + 写 task_config_kv | {dot.key: value} |
| `suspend` | full checkpoint → wandb.finish → 退出码 42 | — |

as-built 补充（KI-12/13/19，010 轮）：

- `sample_from_weights` 备份集合由"实际 adapter 键"驱动：优先 PEFT
  `get_peft_model_state_dict` 查询，拿不到时退化为宽松名称匹配
  （`lora`/`adapter`，大小写不敏感）并记 warning——不再只认 `lora_` 前缀。
- `patch_config` 的 `apply_live` 返回 `(applied, skipped)`：`losses[i]` 越界/
  未命中不再静默，skipped 明细写进 hook.result。
- worker（重）启动时回收本任务 acked 未 done 的 hooks → 重置回 queued
  （`db.requeue_stale_acked_hooks`），防 worker 崩溃后 hook 卡死。
- sample 提示词 JSONL 行非 dict（数组/字符串）时按原文本处理，不再崩溃；
  DB tag 通道按本任务 project_id 过滤（不跨项目）。
- dry-run `save` 的 result 字段与真实路径对齐（with_optimizer 时同样返回
  `optimizer_state` 键）。

## 采样解耦

- `engine/sampling.py`：`sample(model_handle, prompts, params) -> list[Path]`，
  不依赖训练循环状态；
- 采样期间 `model.eval()` + `no_grad` + `requires_grad_(False)`，结束恢复 `train()`；
- 输出 `workspace/samples/<project>/<task>/`，路径写回 hook.result。

## 热改白名单（engine/hot_keys.py）

- 允许：`training.learning_rate`、各 loss 权重、`use_weighting` 等开关、log/save 间隔；
- 拒绝：batch_size、分辨率、lora_rank、lokr_full_rank 等结构参数（返回错误提示需 suspend+重启）。

## wandb 连续性

- suspend 后重启：`restart_count+1`，run name `<task>-run-<n>`，
  `config["resumed_from_run"]` 指向上一个 run id。

## 决定

- hook 在 step 边界执行，绝不在 forward/backward 中途打断；
- suspend 是唯一释放显存的 hook；其余 hook 保活训练进程。
