# 02 — Orchestrator 核心设计：Watcher + Dispatcher + train.py 接入

> 更新：2026-07-18 code review 后回写 as-built（007）。
> 原标"P1 单 worker 串行"；P3 起 dispatcher 为 worker 池（D17），
> `--max-parallel` **默认 1**，默认行为与 P1 串行完全一致。

## 范围

`orchestrator/main.py` / `watcher.py` / `dispatcher.py` / `hooks.py`（写入侧协议）
+ `UnifiedTrainer/train.py` 的 `--task-id/--db` 模式。

## 文件流转

```
workspace/inbox/  ──watcher 1s 轮询──▶  导入 DB  ──▶  原文件移 processing/
                                                         │
              任务结束 ◀──dispatcher──  移 done/ 或 failed/
```

- 文件类型：`project_*.json`（整项目定义）、`task_*.json`（单任务）、
  `cmd_*.json`（指令）。
- **cmd action 集（as-built）**：`cancel` / `hook` / `set_config`。
  cmd `set_config` 走统一入口 `set_config_and_notify`（KI-04 已修复，D23）；
  cmd `hook` 统一走 `hooks.enqueue` 校验（类型白名单 + 仅 running 可投，
  KI-18 已修复，D24），非法投递的 cmd 文件进 failed/。
- 内容 hash 去重（索引 `workspace/import_index.json`，D3），重复投递直接归档 done/；
  导入失败也记 hash，同一坏文件再投递按 duplicate 归档（KI-06 已修复，D24）。
- 归档（as-built）：等同一 source_file 的所有任务进入终态后归档（D4）；
  done 归档时往文件 JSON 追加 `_result` 结果摘要字段（任务 id/状态/起止时间），
  failed 归档时旁写 `<name>.error.log`（含各任务 DB error 字段）（KI-16 已修复，D24）；
  cron 周期任务源文件长期驻留 processing/ 属有意为之——周期任务 re-arm 后
  不是终态，只有熔断或被 cancel 后才归档（D24）。
- 无法识别的文件（非 project_/task_/cmd_ 前缀）连续 3 轮未识别后移入 failed/
  并写 `.note.txt` 说明（KI-06 已修复，D24），不再永久滞留 inbox。

## Dispatcher（worker 池，D17）

- 主循环：收割退出 worker + 准入扫描；每 worker 独立监控线程（心跳判活逻辑不变）；
- 取 `next_runnable_tasks()` 队首 → GPU Guard 准入（P3）→
  `set_task_status(running)` →
  `Popen([sys.executable, "UnifiedTrainer/train.py", "--task-id", id, "--db", db_path])`；
- 心跳判活：`heartbeats` 超过 120s 未更新 → kill + failed；
  从未有心跳的 worker 适用启动宽限：启动后超过 STARTUP_TIMEOUT（默认
  300s，dispatcher.py 常量）仍无任何心跳 → 判僵死 kill + failed（KI-03 已修复，D23）
- 退出码：0→done（解析 `$task:` 引用并放行后继；引用可指向同链更早祖先，
  不限直接前驱，按名字在同项目内解析，KI-14 已修复 D24）、42→suspended、其他→failed；
- 任一 worker 退出后立即重扫 `waiting_gpu` / `pending` 队列补位；
- cron 任务终态后自动 re-arm 回 scheduled（D14）；连续失败熔断：kv
  `_meta.consecutive_failures` 计数（成功清零），达到 3 次（
  dispatcher.MAX_CONSECUTIVE_FAILURES）后不再 re-arm，任务保持 failed 终态，
  原因记 kv `_meta.rearm_blocked`（KI-09 已修复，D24）。

## train.py 接入

```bash
python UnifiedTrainer/train.py --task-id 7 --db workspace/trainer.db [--dry-run-steps N]
```

- 与 `--model/--config` 旧模式并存；DB 模式下：
  1. 启动时 `materialize_config(task_id)` 得到完整 config；
  2. 每 log 间隔写 heartbeat；
  3. 初始化 HookManager（**仅 DB 模式**），每 step 消费 hooks 表；
- **`--dry-run-steps N`**（D1，as-built 补记）：模拟 N 个 step、每步写心跳、
  产出假权重并把路径写 `_meta.output`，退出码 0；也可用 config 键
  `training.dry_run_steps` 触发，CLI 参数优先。仅 DB 模式有效。
- 旧模式行为完全不变（向后兼容）。

## 决定

- worker 用子进程而非线程：崩溃隔离 + 显存彻底释放。
- `$task:<name>.output` 引用在前驱任务 done 时由 dispatcher 解析写入后继的
  `resume_from`，worker 只认具体路径，不感知链。
- 已知问题见 `progress/007-code-review.md` §7（KI-03/06/07/14/16/18）。
