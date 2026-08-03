# 04 — 调度与 GPU 准入设计（P3）

> **状态：已实现（2026-07-18，见 progress/004-p3-scheduler-gpu-api.md）**
> as-built 差异：
> - GPU 采集链为 pynvml → nvidia-smi 子进程 → NullProvider（保守拒绝并行）；
>   torch.cuda.mem_get_info 降级未采用（orchestrator 侧无 torch）。
> - 判定结果写 kv `_meta.gpu_decision` / `_meta.gpu_wait_reason`，
>   gpu_snapshots 表结构不变。
> - 增补：cron 任务终态后自动 re-arm；`at` 一次性触发（tasks 表加 at 列）。
> - dispatcher 重构为 worker 池，`--max-parallel` 默认 1 保持串行。

## 范围

`orchestrator/scheduler.py` + `orchestrator/gpu_guard.py` + dispatcher 并行化。

## Scheduler

- 30s tick：扫描 `status='scheduled'` 且 cron 到点的任务 → 转为 pending 交 GPU Guard；
- 支持 `"cron": "0 3 * * *"` 与一次性 `"at": "ISO时间"`；
- 错过触发只保留一个排队实例（防抖堆积）；
- 纯标准库实现 cron 解析（不引入第三方依赖），支持 `分 时 日 月 周` 五段；
- `_meta.last_fire` 去重键统一为**本地时区**字符串（`now.astimezone()`），
  与 cron 字段匹配的本地时间口径一致（KI-05 已修复，D24）；
- at 任务同样受 last_fire 保护：已有 last_fire 的 at 任务不再触发（D24）；
- cron/at 入库前统一经 `scheduler.validate_schedule()` 校验，
  CLI / API / watcher 三入口复用（KI-10 已修复，D24）。

## GPU Guard（核心规则）

> 有任务在跑时，定时/新任务仅当某张卡**空闲 VRAM > 总量 3/4** 且任务
> `allow_parallel=1` 时才并行启动；否则置 `waiting_gpu` 等待补位。

```python
def admit(task, gpus, running_workers, max_parallel=1):
    if running_workers == 0: return Admit(gpu=best_gpu())
    if not task.allow_parallel: return Wait("parallel not allowed")
    if running_workers >= max_parallel: return Wait("max parallel reached")
    for g in gpus:
        if g.free_mb > g.total_mb * 3 / 4: return Admit(gpu=g.index)
    return Wait("insufficient free VRAM")
```

- pynvml 采集，失败时降级为 `torch.cuda.mem_get_info`，再失败则保守拒绝并行；
- nvidia-smi 输出经 `parse_nvidia_smi_csv` 容错解析：`[N/A]` 字段不再崩溃
  （util 记 None，显存字段无法解析的整行跳过）（KI-15 已修复，D24）；
- 采集异常与"无 GPU 信息"区分：采集失败时并行拒绝原因明确写
  `GPU metrics collection failed: ...`（KI-15 已修复，D24）；
- `judge(max_parallel=...)` 用显式 `is None` 判断，`--max-parallel 0`
  （禁止并行）不会被 falsy 绕过；上限单源在 dispatcher（KI-07/KI-08 已修复，D24）；
- 每次判定把快照与结果写 `gpu_snapshots`（前端可查"为什么还在等"）；
- 任一 worker 退出 → 立即重扫 `waiting_gpu` → `pending` 队列补位；
- 并行 worker 通过 `CUDA_VISIBLE_DEVICES=<gpu>` 绑卡。

## 决定

- 并行上限默认 **1**（有意保守，保持 P1 串行行为，D17；早期草稿写 2 已作废），
  可在 orchestrator 配置/`--max-parallel` 改；
- 空机放行不查 VRAM（首任务可能本来就吃满整卡）。
