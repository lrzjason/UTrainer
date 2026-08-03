---
name: unified-trainer-orchestrator
description: 操作 ScheduledTrainer 任务编排器（E:\UnifiedTrainer\ScheduledTrainer）创建/调度/监控扩散模型训练任务。当用户要求提交训练任务、搭串行任务链、设 cron 定时训练、训练中采样/保存/挂起/恢复、查 GPU 排队原因时使用。
---

# UnifiedTrainer Orchestrator 操作手册（Agent 用）

编排器 = SQLite（唯一事实源）+ Watcher/Scheduler/Dispatcher/GPUGuard + HTTP API（127.0.0.1:7860）+ 零构建 SPA 前端（`web/dist/index.html`）。人用手册见 `md/ORCHESTRATOR.md`。

## 架构

```
orchestrator/
├── main.py        → 常驻入口：启动 Watcher + Scheduler + Dispatcher (+ 可选 API)
├── watcher.py     → 1s 轮询 workspace/inbox/，导入 JSON 到 DB
├── scheduler.py   → 30s tick 扫描 scheduled 任务，cron/at 到点置 pending
├── dispatcher.py  → worker 池：spawn train.py 子进程，心跳判活，终态归档
├── gpu_guard.py   → 并行准入判定 + VRAM 采集（pynvml → nvidia-smi → null 降级）
├── hooks.py       → hook 写入侧协议（enqueue/set_config_and_notify/resume_task）
├── server.py      → 标准库 http.server JSON REST + SSE（无 fastapi 依赖）
├── cli.py         → CLI 最小集（project/submit/task/list/cancel/hook/resume/config/gpu）
├── db.py          → SQLite 数据访问层（七表 CRUD）
└── schema.sql     → 七表：projects/tasks/hooks/task_config_kv/prompts/heartbeats/gpu_snapshots
```

## 红线

- **绝不直接写 trainer.db**；一切经 CLI / HTTP API / inbox 文件。
- 热改配置只走 **白名单键**（`UnifiedTrainer/engine/hot_keys.py` 校验；PATCH `/api/tasks/{id}/config` 在 running 时自动投 patch_config hook）。
- hook 只对 **running** 任务生效（suspend 后用 resume）；cancel 只对 pending/scheduled/waiting_gpu。
- 不引用 ScheduledTrainer 以外的代码；不 pip 安装（纯标准库 + torch 生态）。
- 磁盘满会让 sqlite 全面报错，先确认磁盘空间再排查编排器。

## 启动

```bash
cd E:\UnifiedTrainer\ScheduledTrainer
python -m orchestrator.main --workspace workspace --api --port 7860 --max-parallel 2
```

参数：`--workspace`（默认 workspace）、`--db`（默认 <workspace>/trainer.db）、`--tick`（scheduler 间隔，默认 30s）、`--host`（默认 127.0.0.1）、`--port`（默认 7860）、`--max-parallel`（默认 1）。

## workspace 目录

```
workspace/
├── inbox/        ← 投递 JSON（watcher 1s 消费）
├── processing/   ← 已入库、运行中的源文件
├── done/         ← 终态 done 归档
├── failed/       ← 终态 failed 归档
├── samples/      ← 采样图片 <project>/<task>/
├── prompts/      ← 提示词缓存
└── trainer.db    ← SQLite 唯一事实源
```

## inbox 文件格式

- `project_*.json`：整项目定义（含 tasks 数组批量建任务）
- `task_*.json`：单任务 `{"project": name|id, "name":..., "model":..., "config": {...}|"path"}`
- `cmd_*.json`：指令 `{"action": "cancel"|"hook"|"set_config", ...}`

内容 hash 去重（sha256 → import_index.json），重复投递直接归档 done/。

## CLI 参考

```bash
python -m orchestrator.cli project create <name> [--model M] [--desc D]
python -m orchestrator.cli project list
python -m orchestrator.cli submit <json_file>
python -m orchestrator.cli task create --project P --name N [--model M] [--config f.json] [--cron "分 时 日 月 周"] [--at ISO] [--priority 100] [--allow-parallel]
python -m orchestrator.cli list [--project P] [--status S]
python -m orchestrator.cli cancel <task_id>
python -m orchestrator.cli hook <task_id> <type> [--payload JSON] [--name N] [--path P] [--weights W] [--n 4] [--steps 20] [--seed 42] [--set key=value]
python -m orchestrator.cli hooks [--task-id N]
python -m orchestrator.cli resume <task_id>
python -m orchestrator.cli config set <task_id> <key> <value>
python -m orchestrator.cli gpu status
```

hook type 可选：`sample` / `sample_from_weights` / `save` / `restore` / `patch_config` / `suspend`

## HTTP API（/api 前缀，JSON in/out，无鉴权）

```
GET/POST   /api/projects              GET/PATCH /api/projects/{id}
GET/POST   /api/tasks                 GET       /api/tasks/{id}
POST       /api/tasks/{id}/hooks      GET       /api/tasks/{id}/hooks
GET/PATCH  /api/tasks/{id}/config
POST       /api/tasks/{id}/cancel     POST      /api/tasks/{id}/resume
GET/POST   /api/prompts               GET       /api/gpu
GET        /api/tasks/{id}/heartbeats?since=
GET        /api/samples/{task_id}
GET        /api/events                （SSE：每 2s 推 tasks+gpu 快照）
GET        /samples-file/{task_id}/{name}  → 采样图片字节
GET        / 与 /assets/*             → web/dist/ 静态 SPA
```

## 标准操作流

1. **生成项目/任务 JSON**（格式见 `md/ORCHESTRATOR.md` §5；`config` 需含 `model`/`training`/`output`）。
2. **投递**：写文件到 `workspace/inbox/*.json`（watcher 自动消费），或 CLI `task create`，或 API `POST /api/tasks`。
3. **轮询状态**：`GET /api/tasks/{id}`（含 config_kv/_meta），或 `GET /api/events`（SSE），或 `cli list`。
4. **汇报**：终态 done/failed（`error` 字段）+ hook 结果 + 采样文件列表（`/api/samples/{id}`）。

## 常用配方

**串行链**：依次创建任务，后一个 `"depends_on": <前一个 id>`；dispatcher 在前驱 done 后自动释放后继。`resume_from` 支持 `"$task:<name>.output"` 引用前驱产物路径。

**周期任务**：`"cron": "17 3 * * *"`（五段：分 时 日 月 周；避免整点/半点）+ `"allow_parallel": true`（需要并行时）；cron 任务终态后 dispatcher 自动 re-arm（置回 scheduled），同一分钟靠 `_meta.last_fire` 去重。连续失败 3 次熔断不再 re-arm。一次性用 `"at": "2026-07-20 03:00:00"`。

**训练中采样**：`POST /api/tasks/{id}/hooks {"type":"sample","payload":{"n":4,"steps":20,"seed":42}}`；
图片落在 `workspace/samples/<project>/<task>/`，经 `/api/samples/{id}` 列文件、`/samples-file/{id}/{name}` 取字节。

**suspend/resume**：suspend hook → worker 存 checkpoint 并以 exit code 42 退出 → 任务置 suspended →
`POST /api/tasks/{id}/resume`（或 `cli resume`）→ restart_count+1、新 wandb run name（`<task>-run-<n>`）、resume_mode=full 自动续跑。

**热改学习率**：`PATCH /api/tasks/{id}/config {"kv": {"training.learning_rate": 5e-5}}`（running 时即时生效，非 running 下次启动生效）。

## Dispatcher 行为

- 心跳判活：120s 无心跳判僵死（启动宽限 300s）；kill 后任务置 failed。
- 退出码：0=done / 42=suspended / 其他=failed。
- 并行：max_parallel 控制 worker 上限；GPUGuard 准入（空机放行；否则要求 allow_parallel 且某卡 free > 75%）。
- waiting_gpu 原因：`config_kv["_meta.gpu_wait_reason"]`。
- 任一 worker 退出 → 唤醒主循环立即重扫 waiting_gpu 补位。
- 保留策略：每小时裁剪 heartbeats（保留 7 天）/ gpu_snapshots（保留 10000 行）。

## 故障速查

- waiting_gpu 原因看 `config_kv["_meta.gpu_wait_reason"]`
- hook 卡 queued → 先确认任务 running（hook 只对 running 生效）
- API 409 = 状态不允许该操作
- inbox 文件滞留 → 检查文件名前缀（必须 project_/task_/cmd_），无法识别的文件连续 3 次后移入 failed/
- cron 任务不再触发 → 检查是否连续失败 3 次触发熔断
- 完整排查表见 `md/ORCHESTRATOR.md` §8
