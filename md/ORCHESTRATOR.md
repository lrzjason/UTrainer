# ScheduledTrainer Orchestrator — 使用手册

> 基于 `orchestrator/` 实际代码（as-built），对应 P4（2026-07-18）。
> 设计文档见 `md/00`–`md/05`；实现差异见 `progress/` 与 `agent/decisions.md`。

## 1. 架构

```
inbox/*.json ──watcher──► SQLite(trainer.db, db.py 唯一写入口) ◄── CLI / HTTP API / Web 前端
                              ▲ scheduler(cron/at 到点置 pending)
                          dispatcher(worker 池, --max-parallel 默认 1)
                              │ GPU Guard 准入后 spawn
                          UnifiedTrainer/train.py 子进程（心跳 + HookManager 消费 hooks 表）
```

- **Watcher**：轮询 `workspace/inbox/*.json` → 入库 → `processing/` → 终态归档 `done/`/`failed/`。
- **Scheduler**：30s tick（`--tick`）扫 `scheduled` 任务到点置 `pending`；cron 任务终态后自动 re-arm（连续失败 3 次熔断不再 re-arm，记 `_meta.rearm_blocked`）；`at` 一次性。
- **API**（`server.py`）：标准库 `ThreadingHTTPServer`，REST + SSE，静态托管 `web/dist` 前端。

## 2. 项目 / 任务模型

任务状态机：`pending → running → done/failed`；分支：`scheduled`（等 cron/at）、`waiting_gpu`（GPU Guard 拒绝排队，worker 退出自动补位）、`suspended`（待 resume）、`cancelled`。
关键字段：`priority`（小者先派）、`depends_on`（串行链）、`resume_from`/`resume_mode`（weights|full）、`cron`/`at`、`allow_parallel`、`restart_count`/`wandb_run_name`（`<task>-run-<n>`）。
配置 = `config`(JSON) + `task_config_kv` 覆盖层，`materialize_config()` 合成下发；`_meta.*` 键存运行时元数据（gpu_wait_reason、suspend_checkpoint、cuda_visible_devices 等）。

## 3. CLI 参考（全局 `--workspace` `--db`）

```bash
python -m orchestrator.cli project create <name> [--model M] [--desc D]
python -m orchestrator.cli project list
python -m orchestrator.cli submit <json_file>          # 复制进 inbox/
python -m orchestrator.cli task create --project P --name N [--model M] [--config f.json] \
    [--priority 100] [--cron "*/5 * * * *" | --at "2026-07-20 03:00:00"] [--allow-parallel]
python -m orchestrator.cli list [--project P] [--status S]
python -m orchestrator.cli cancel <task_id>            # 仅 pending/scheduled/waiting_gpu
python -m orchestrator.cli hook <task_id> sample [--tag T] [--n 4] [--steps N] [--seed S]
python -m orchestrator.cli hook <task_id> sample_from_weights --weights <path>
python -m orchestrator.cli hook <task_id> save [--name X] [--with-optimizer]
python -m orchestrator.cli hook <task_id> restore --path <ckpt> [--reset-optim]
python -m orchestrator.cli hook <task_id> patch_config --set training.learning_rate=5e-5
python -m orchestrator.cli hook <task_id> suspend
python -m orchestrator.cli hooks [--task-id N]         # queued→acked→done 历史
python -m orchestrator.cli resume <task_id>            # 仅 suspended
python -m orchestrator.cli gpu status                  # nvidia-smi 快照入库
```

主服务：`python -m orchestrator.main [--api] [--port 7860] [--max-parallel N] [--tick S]`；
API 独立：`python -m orchestrator.server --workspace W [--port 7860]`。

## 4. HTTP API（127.0.0.1:7860，JSON，无鉴权）

```
GET/POST /api/projects            GET/PATCH /api/projects/{id}
GET/POST /api/tasks               GET /api/tasks/{id}（含 config_kv）
POST/GET /api/tasks/{id}/hooks    GET/PATCH /api/tasks/{id}/config
POST     /api/tasks/{id}/cancel   （非 pending/scheduled/waiting_gpu → 409）
POST     /api/tasks/{id}/resume   （非 suspended → 409；返回 restart_count/wandb_run_name/resume_from）
GET/POST /api/prompts             GET /api/gpu（{latest, history, waiting}）
GET      /api/tasks/{id}/heartbeats?since=<ts>   （step/loss/lr/vram_mb）
GET      /api/samples/{task_id}   （采样文件列表）
GET      /api/events              SSE 每 2s 一帧 {ts, tasks, gpu}（WebSocket 降级）
```

静态托管（P4）：`GET /` 及非 `/api/` 路径 → `web/dist/`，未命中回退 `index.html`（hash 路由 SPA）；
`GET /samples-file/{task_id}/{name}` → 采样图片字节（防目录穿越）。
前端页面：`#/` 项目卡片、`#/projects/:id` 任务链+定时+提示词库、`#/tasks/:id` loss 曲线+六个 hook 按钮+采样画廊+hook 历史、`#/gpu` 显存+waiting_gpu 拒绝原因、`#/new` 表单↔JSON 双视图。

错误：`{"error":...}`；400 参数 / 404 未找到 / 409 状态冲突。

## 5. 任务文件格式（inbox/*.json）

```json
{"project": "my_proj", "name": "task-a", "model": "flux2_klein",
 "config": {"model":"flux2_klein","training":{},"output":{}},
 "priority": 100, "depends_on": 1, "resume_from": "...", "resume_mode": "weights",
 "cron": "0 3 * * *", "allow_parallel": false}
```
有 cron 或 at → `scheduled`，否则 `pending`；非法 cron/at 入库前拒绝。

## 6. Hook 协议

- 写入侧唯一通道 `hooks.enqueue()`（CLI/API 共用）；worker 侧 `engine/hook_manager.py` 轮询消费，状态 `queued→acked→done/failed`，结果存 `result`。
- **仅 running 任务**可投 hook。类型：`sample` / `sample_from_weights` / `save` / `restore` / `patch_config`（白名单热改，`engine/hot_keys.py` 校验）/ `suspend`。
- `resume`：`restart_count+1`、新 `wandb_run_name`、记录 `resumed_from_run`、`resume_from=_meta.suspend_checkpoint`、`resume_mode=full` → 回 pending 自动重派；写 kv `wandb.run_name`/`wandb.resumed_from_run`。

## 7. GPU Guard 规则

准入顺序：① 空机放行 → ② `allow_parallel=0` 拒绝（parallel not allowed）→ ③ 达 `max_parallel` 拒绝 → ④ 存在某卡 `free > total*3/4` → `Admit(gpu=i)` 并注入 `CUDA_VISIBLE_DEVICES=i`，否则拒绝（insufficient free VRAM (need > 75%)）。
拒绝 → `waiting_gpu` + `_meta.gpu_wait_reason`；每次判定写 `gpu_snapshots` + `_meta.gpu_decision`。
采集链：pynvml → nvidia-smi 子进程 → NullProvider（空机放行、并行保守拒绝）；测试用 FakeProvider。

## 8. 故障排查

| 症状 | 排查 |
|------|------|
| 一直 waiting_gpu | `GET /api/tasks/{id}` 看 `_meta.gpu_wait_reason`；查 max_parallel / allow_parallel / 显存 |
| cron 不触发 | 表达式合法性（入库时已校验）；`_meta.last_fire`；tick 间隔 |
| hook 一直 queued | 任务必须 running；`cli hooks --task-id N` 看状态与 result |
| resume 409 | 仅 suspended；先 suspend |
| 采样画廊空 | sample hook 是否 done；目录 `workspace/samples/<project>/<task>/` |
| API 起不来 | 端口占用 → `--port`；仅绑 127.0.0.1 |
| sqlite "database or disk is full" | **磁盘满会同时阻塞 watcher/dispatcher/API/测试**，先清磁盘 |
| 前端打不开 | 走 `http://127.0.0.1:7860/`（API 需后端），确认 `web/dist/index.html` 存在 |
| Windows 中文乱码 | subprocess 一律 `encoding="utf-8"` |
