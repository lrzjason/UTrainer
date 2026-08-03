# 002 — P1 实施总结（Orchestrator 核心）

日期：2026-07-18 ｜ 状态：**完成，冒烟测试通过**

## 交付物

### 新增 `orchestrator/` 包（ScheduledTrainer 根下）

| 文件 | 内容 |
|------|------|
| `orchestrator/__init__.py` | 包标记 |
| `orchestrator/schema.sql` | 七表 DDL（projects/tasks/hooks/task_config_kv/prompts/heartbeats/gpu_snapshots），逐字来自 `doc/Improvement_k3.md` §3.2，另加 5 个常用查询索引 |
| `orchestrator/db.py` | sqlite3 + WAL + `busy_timeout=5000`，连接按线程持有；接口与 `md/01` 一致，另加 `get_task/list_tasks/list_projects/successors_of/latest_gpu_snapshots/get_config_kv(raw=)`；所有 SQL 集中于此 |
| `orchestrator/watcher.py` | 1s 轮询 inbox/；识别 `project_*.json / task_*.json / cmd_*.json`；sha256 内容去重（索引存 `workspace/import_index.json`）；导入后移 processing/，cmd 处理完移 done/，失败移 failed/ 附 `.error.log` |
| `orchestrator/dispatcher.py` | 单 worker 串行：`next_runnable_tasks()` 队首 → `Popen(train.py --task-id --db)` → 心跳判活 120s → 退出码 0=done / 42=suspended / 其他=failed；done 时解析后继 `$task:<name>.output` → 写入其 `resume_from`；同一 source_file 全终态后归档 processing/→done/（全 done）或 failed/；worker 退出立即重扫 |
| `orchestrator/main.py` | 常驻入口，`argparse --workspace/--db`，Watcher + Dispatcher 双线程 |
| `orchestrator/cli.py` | `project create/list`、`submit`、`list`、`cancel`、`gpu status`（nvidia-smi 采集入 gpu_snapshots） |

### 修改 `UnifiedTrainer/train.py`（ScheduledTrainer 内副本）

- 新增 `--task-id/--db/--dry-run-steps`；`--model/--config` 改为非必填 + 手动校验
  （旧模式缺参仍报错退出码 2；`--list-models/--list-losses` 现在可不带 dummy 参数）。
- DB 模式分支在参数解析后尽早执行：`materialize_config` → dry-run 判断（config 键
  `training.dry_run_steps` 或 CLI 参数）→ 否则走原有训练流程（config 来源换为 DB，
  后续逻辑零改动）。
- 真实训练路径新增 `_DBHeartbeatCallback`（每 `training.log_every` 步写 heartbeats，
  含 loss/lr/VRAM）；训练结束写 `_meta.output`。
- `_dry_run()`：模拟 N 步、写心跳、产出假权重、退出码 0。

### workspace 目录

`Watcher.__init__` 自动创建 `inbox/ processing/ done/ failed/ samples/ prompts/`。

## 验证命令与退出码

| 命令 | 退出码 |
|------|--------|
| `python -m py_compile orchestrator/*.py UnifiedTrainer/train.py tmp/smoke_test.py` | 0 |
| `python tmp/smoke_test.py`（端到端冒烟） | 0 |
| `python -m orchestrator.cli --workspace tmp/smoke_ws list` | 0（输出见下） |
| `python -m orchestrator.cli --workspace tmp/smoke_ws project create/list` | 0 |
| `python UnifiedTrainer/train.py --list-models` | 0 |
| 外部原版 `python E:\UnifiedTrainer\UnifiedTrainer\train.py --model dummy --config dummy --list-models` | 0（行为与副本逐字节一致：managed Python 无 torch，适配器扫描失败为空列表，属环境问题非回归） |
| `python -m orchestrator.cli ... gpu status` | 1（`nvidia-smi` 在本环境 NVML 初始化失败，CLI 正确报错） |

## 冒烟测试输出摘录（`python tmp/smoke_test.py`）

投递 `project_smoke.json`（t1 → t2 串联，t2.resume_from=`$task:t1.output`，两任务均 dry-run）：

```
id=1 name=t1 status=done  started=12:53:17 finished=12:53:20
id=2 name=t2 status=done  resume_from=E:\...\out1\t1_dryrun.safetensors
            started=12:53:20 finished=12:53:22
=== smoke test PASSED ===
archived -> E:\UnifiedTrainer\ScheduledTrainer\tmp\smoke_ws\done\project_smoke.json
```

orchestrator 日志关键行：

```
[INFO] watcher: Imported project_smoke.json as project: task_ids=[1, 2]
[INFO] dispatcher: Dispatch task 1 (t1)  →  [dry-run] 5 steps  →  Task 1 done
[INFO] dispatcher: Resolved $task:t1.output -> E:\...\t1_dryrun.safetensors for task 2 (t2)
[INFO] dispatcher: Dispatch task 2 (t2)  →  [dry-run] 3 steps  →  Task 2 done
[INFO] dispatcher: Archived project_smoke.json -> ...\done
```

断言全部通过：顺序执行（t2.started ≥ t1.finished）、依赖放行（resume_from 解析为真实
文件且存在）、文件归档 done/、t2 有心跳记录。

`python -m orchestrator.cli --workspace tmp/smoke_ws list` 实际输出：

```
id  project_id  name  model  status  priority  depends_on  resume_from                                       error
1   1           t1    dummy  done    100       None        None                                              None
2   1           t2    dummy  done    100       1           E:\...\smoke_ws\out1\t1_dryrun.safetensors        None
```

另验证：同内容重复投递 → 识别 duplicate 直接归档 done/；`cmd_cancel.json` 正常处理归档。

## 遗留风险 / 待办

1. **真实训练路径未联调**：managed Python 无 torch，`_DBHeartbeatCallback` 与
   `_meta.output` 写入只在 dry-run 下验证过；P2 前应在有 torch 的环境跑一次真实
   小 config。
2. **cancel 竞态**：cmd 取消只作用于 pending/scheduled/waiting_gpu；watcher 与
   dispatcher 并发下任务可能已被置 running。P2 用 `stop` hook 覆盖 running 取消。
3. **失败任务阻塞链**：后继依赖 failed 前驱会永远 pending（`next_runnable_tasks`
   要求前驱 done）。重试/跳过策略待定（P2/P3）。
4. **gpu status 依赖 nvidia-smi**：当前环境 NVML 报错；P3 GPU Guard 需处理
   无 GPU / nvidia-smi 不可用降级。
5. **心跳超时杀进程**（-999 路径）未实测——需要真实长任务或 mock。
6. cmd 仅实现 cancel/hook/set_config 三个 action；sample 等 hook 语义 P2 实现。

技术决定见 `agent/decisions.md`（D1–D8）。
