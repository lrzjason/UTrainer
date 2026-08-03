# 01 — SQLite 数据层设计（P1）

> 更新：2026-07-18 code review 后回写 as-built（007）。

## 范围

`orchestrator/schema.sql` + `orchestrator/db.py`，所有 SQL 集中在 db.py。

## 表

`projects` / `tasks` / `hooks` / `task_config_kv` / `prompts` / `heartbeats` /
`gpu_snapshots` —— 完整 DDL 见 `doc/Improvement_k3.md` §3.2（含 `tasks.at` 列、
5 个索引、旧库 ALTER 迁移说明，D18）。

## 状态机（8 状态，as-built）

`pending`（可派发）/ `scheduled`（等 cron/at 到点）/ `waiting_gpu`（GPU Guard 拒绝
排队）/ `running` / `suspended`（hook 挂起待 resume）/ `done` / `failed` /
`cancelled`。迁移均通过 `set_task_status` 单点写入（可审计）。

## kv 元数据约定（D2）

key 以 `_` 前缀（`_meta.output` / `_meta.suspend_checkpoint` /
`_meta.gpu_decision` / `_meta.gpu_wait_reason` / `_meta.last_fire` /
`_meta.consecutive_failures` / `_meta.rearm_blocked`（D24，cron 熔断）/
`_meta.gpu_index` / `_meta.cuda_visible_devices` / `wandb.*`）为运行时元数据，
`materialize_config` 跳过不注入训练 config。

项目/任务名字在 `create_project` / `create_task` 统一经 `validate_name()`
校验（拒绝空名、`.`/`..`、路径分隔符 `/` `\`）——名字会拼进 samples/ 等
文件系统路径，防止目录穿越（KI-11 已修复，D24）。

## db.py 接口（as-built 全量）

```python
class DB:
    def __init__(self, path): ...            # sqlite3 + WAL + busy_timeout + row_factory
    # projects
    def create_project(name, description="", default_model=None, tags=()) -> int
    def get_project(id_or_name) -> dict | None
    def list_projects() -> list[dict]
    def update_project(project_id, **fields)
    # tasks
    def create_task(project_id, name, model, config: dict, **kw) -> int
    def get_task(task_id) -> dict | None
    def list_tasks(project_id=None, status=None) -> list[dict]
    def set_task_status(task_id, status, **fields)
    def next_runnable_tasks() -> list[dict]
        # as-built（D14）：status='pending' 且依赖已 done 即返回；
        # 不再过滤 cron —— cron/at 是否到点完全由 scheduler 负责，
        # "pending" 是"已到点/可派发"的唯一语义。结果含 waiting_gpu 供补位扫描。
    def successors_of(task_id) -> list[dict]   # 直接后继（depends_on = task_id）
    def materialize_config(task_id) -> dict    # config_json ⊕ task_config_kv（跳过 _ 前缀键）
    # hooks
    def enqueue_hook(task_id, type_, payload: dict) -> int   # KI-17：type→type_
    def fetch_queued_hooks(task_id) -> list[dict]
    def ack_hook(hook_id); def finish_hook(hook_id, result=None, failed=False)
    def requeue_stale_acked_hooks(task_id) -> int
        # KI-19：worker 启动时把本任务 acked 未 done 的 hooks 重置回 queued
    def list_hooks(task_id=None) -> list[dict]
    # config kv
    def set_config_kv(task_id, key, value)
        # 注意：不产生 patch_config hook、无白名单校验 —— 纪律靠调用方（KI-04）
    def get_config_kv(task_id, raw=False) -> dict
        # raw=False：JSON 解码后的 dict；raw=True：list[dict] 含 updated_at 等原始行
    # prompts
    def add_prompt(project_id, text, tag=None, negative=None, meta=None) -> int
    def list_prompts(project_id=None) -> list[dict]
    def prompts_by_tag(project_id, tag) -> list[dict]
        # KI-19：worker 侧强制传本任务 project_id（不再跨项目取提示词）；
        # None 仍表示不限项目（保留给管理类查询）
    # heartbeats / gpu
    def heartbeat(task_id, step, loss, lr, vram_mb)
    def last_heartbeat(task_id) -> dict | None
    def heartbeats_since(task_id, since_ts) -> list[dict]
    def gpu_snapshot(snapshots: list[dict])
    def latest_gpu_snapshots() -> list[dict]
    def list_gpu_snapshots(limit=...) -> list[dict]
    # retention（KI-17；main.py 每小时调用一次）
    def prune_heartbeats(keep_days=7) -> int      # 删 keep_days 天前心跳
    def prune_gpu_snapshots(keep_rows=10000) -> int  # 每 gpu_index 保留最新 N 行
    # 生命周期
    def close()          # 关闭当前线程连接
    def close_all()      # KI-17：关闭连接注册表内全部跨线程连接
```

## 决定

- WAL 模式 + `busy_timeout=30000`（KI-17：锁等待唯一来源；此前
  connect(timeout=30) 与 busy_timeout=5000 双源矛盾已统一），
  支撑 orchestrator/worker/CLI/API 四方并发。
- `materialize_config` 是唯一配置出口：worker 启动与热改后都走它，杜绝多份配置。
- 连接按线程持有（API 多线程），登记进连接注册表；不跨线程共享 connection。
- 已知问题见 `progress/007-code-review.md` §7（全部已修复清零，见
  `progress/010-ki-final.md`）。
