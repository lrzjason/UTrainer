# Improvement_k3 — ScheduledTrainer 任务编排与高度可控训练系统设计

> 基于 `Improvement.md` 的原始构想扩展而成，已适配 ScheduledTrainer 工作区。
> 目标：把训练器从"简单训练脚本"升级为**由数据库驱动、以 Project 组织任务、
> 带 Web 前端、可热修改配置、可中断/续训、可被 Agent 完全操控**的训练平台。
> 更新日期：2026-07-18（007 code review 后批量回写 as-built，差异依据
> `progress/007-code-review.md`；技术决定见 `agent/decisions.md` D1–D22）

---

## 1. 总体架构

层级模型：**Project → Task（串联链 / 定时 / 独立）→ Hook 指令**

```
┌────────────── Web 前端（零构建静态 SPA，as-built D19） ──────────────┐
│  项目看板 / 任务链列表 / loss 曲线(SVG) / GPU 监控 / hook 操作 / 配置查看   │
└──────────────────────────────┬────────────────────────────────────────┘
                               │ REST + SSE（标准库 ThreadingHTTPServer，as-built D16）
┌──────────────────────────────▼────────────────────────────────────────┐
│                     orchestrator (主监控程序)                            │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Watcher  │ │ Scheduler │ │ Dispatcher │ │ GPU Guard│ │ DB(SQLite)│ │
│  │ 文件监听  │ │ cron/at   │ │ worker 池   │ │ 显存门禁  │ │ 唯一事实源 │ │
│  └──────────┘ └───────────┘ └────────────┘ └──────────┘ └───────────┘ │
│        inbox/ → processing/ → done/（任务文件流转）                       │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │ spawn / monitor / restart
              ┌────────────▼────────────┐
              │  worker = train.py 进程  │
              │  ┌────────────────────┐ │
              │  │ HookManager        │ │ ← 每 step 检查 hook 队列
              │  │  · sample          │ │
              │  │  · sample_from_    │ │
              │  │    weights         │ │
              │  │  · save / restore  │ │
              │  │  · config patch    │ │
              │  │  · suspend (退出)  │ │
              │  └────────────────────┘ │
              └─────────────────────────┘
```

核心原则：

1. **SQLite 是唯一事实源（single source of truth）**。项目、任务、hook、提示词、
   动态配置全部读写数据库；JSON 配置文件只是*导入入口*，导入后即被数据库接管。
2. **Orchestrator 与 Worker 分离**。Worker 崩溃/被 hook 挂起时，Orchestrator 仍存活，
   负责清理、记录状态、按策略重启。
3. **前端与 CLI 平级**。两者都只是 DB + API 的客户端：CLI 直接操作 DB/文件，
   前端通过 HTTP API 操作同一套 DB，功能完全对齐，任何一边的修改另一边立即可见。
4. **一切变更可通过"文件落盘 / CLI / 前端按钮"三种途径发起**，天然兼容 Agent 操作。

---

## 2. 目录结构

代码与文档均在 `E:\UnifiedTrainer\ScheduledTrainer\` 内（agent 交互区，
与外部源代码隔离，详见 `doc/REQUIREMENTS.md`）：

```
E:\UnifiedTrainer\ScheduledTrainer\
├── UnifiedTrainer/                # 训练器源码副本（自外部复制，独立演进）
│   ├── train.py  engine/  models/  losses/  data/  registry.py ...
│
├── orchestrator/                  # 任务编排器
│   ├── main.py                    # 主监控程序入口（常驻进程）
│   ├── watcher.py                 # inbox/ 文件监听
│   ├── scheduler.py               # cron / at 定时任务
│   ├── dispatcher.py              # 任务派发（worker 池）、串行链解析、worker 生命周期
│   ├── gpu_guard.py               # GPU VRAM 检测与并行准入（见 §5.4）
│   ├── server.py                  # 标准库 HTTP：REST + SSE（as-built D16）
│   ├── db.py                      # SQLite 数据访问层（所有 SQL 集中在此）
│   ├── schema.sql                 # 建表语句
│   ├── hooks.py                   # hook 指令的写入侧协议（enqueue/resume）
│   └── cli.py                     # CLI 可执行入口
│
├── web/dist/index.html            # 零构建静态 SPA（as-built D19，见 §8）
│
├── workspace/                     # 编排器工作目录（可配置）
│   ├── inbox/                     # 新任务 / 新指令投递目录（Watcher 监听）
│   ├── processing/                # 已被领取、正在执行的任务文件
│   ├── done/                      # 已完成任务文件（自动归档）
│   ├── failed/                    # 失败任务文件
│   ├── samples/                   # hook 采样输出图片（按 project/task 分目录）
│   └── prompts/                   # 提示词缓存（.txt / .jsonl），路径由主控传入
│
├── doc/                           # 要求文档 + 本总体设计文档
├── md/                            # 每个大功能的设计文档
├── agent/  progress/  skills/  tmp/
```

---

## 3. SQLite 数据管理方案

### 3.1 为什么用 SQLite

- 零部署、单文件、事务安全，训练机本地即可运行；
- 多进程（orchestrator + worker + CLI + API）通过 WAL 模式安全并发读写；
- 所有状态可查询、可审计、可回滚。

### 3.2 核心表设计（`orchestrator/schema.sql`）

```sql
-- 项目：任务的上层组织单位
CREATE TABLE projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',   -- active / archived
    default_model TEXT,                            -- 项目级默认值，任务可覆盖
    tags        TEXT DEFAULT '[]',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 任务定义：JSON config 导入后物化为一行，必属于某个 project
CREATE TABLE tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    name          TEXT NOT NULL,
    model         TEXT NOT NULL,            -- flux2_klein / qwen_image / ...
    config_json   TEXT NOT NULL,            -- 完整训练配置快照
    status        TEXT NOT NULL DEFAULT 'pending',
                  -- pending / scheduled / running / suspended /
                  -- done / failed / cancelled / waiting_gpu
    priority      INTEGER NOT NULL DEFAULT 100,   -- 越小越优先
    depends_on    INTEGER REFERENCES tasks(id),   -- 串行链：前驱任务
    resume_from   TEXT,                     -- 权重路径（可引用前驱产物）
    resume_mode   TEXT DEFAULT 'weights',   -- weights / full（是否带优化器）
    cron          TEXT,                     -- 定时表达式，NULL = 立即/依赖触发
    at            TEXT,                     -- 一次性触发时间（ISO），与 cron 互斥（D18）
    allow_parallel INTEGER NOT NULL DEFAULT 0, -- 定时任务是否允许与在跑任务并行
    wandb_run_name TEXT,
    restart_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    started_at    TEXT, finished_at TEXT,
    source_file   TEXT,                     -- 来自 inbox/ 的原始文件
    error         TEXT
);

-- hook 指令队列：orchestrator/CLI/前端 写入，worker 每 step 消费
CREATE TABLE hooks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER NOT NULL REFERENCES tasks(id),
    type       TEXT NOT NULL,
               -- sample / sample_from_weights / save / restore /
               -- patch_config / suspend / stop
    payload    TEXT NOT NULL DEFAULT '{}',  -- JSON，见 §6
    status     TEXT NOT NULL DEFAULT 'queued',  -- queued / acked / done / failed
    result     TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    acked_at   TEXT, done_at TEXT
);

-- 动态配置：worker 运行期实际生效的 config 以这里为准
CREATE TABLE task_config_kv (
    task_id  INTEGER NOT NULL REFERENCES tasks(id),
    key      TEXT NOT NULL,                 -- 点路径，如 training.learning_rate
    value    TEXT NOT NULL,                 -- JSON 编码
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (task_id, key)
);

-- 采样提示词缓存（项目级共享）
CREATE TABLE prompts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id),
    tag        TEXT,                            -- 分组标签
    text       TEXT NOT NULL,
    negative   TEXT,
    meta       TEXT DEFAULT '{}'                -- 分辨率、步数、seed 等
);

-- 训练进度心跳（worker 定期上报，orchestrator 判活 + 前端画曲线）
CREATE TABLE heartbeats (
    task_id  INTEGER NOT NULL,
    step     INTEGER NOT NULL,
    loss     REAL,
    lr       REAL,
    vram_mb  INTEGER,
    ts       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- GPU 资源快照（orchestrator 周期采集，前端展示 + GPU Guard 决策依据）
CREATE TABLE gpu_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_index   INTEGER NOT NULL DEFAULT 0,
    total_mb    INTEGER NOT NULL,
    used_mb     INTEGER NOT NULL,
    free_mb     INTEGER NOT NULL,
    util_pct    INTEGER,
    ts          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 常用查询索引（as-built，schema.sql:96-100）
CREATE INDEX idx_tasks_status        ON tasks(status);
CREATE INDEX idx_tasks_project       ON tasks(project_id);
CREATE INDEX idx_tasks_depends       ON tasks(depends_on);
CREATE INDEX idx_hooks_task_status   ON hooks(task_id, status);
CREATE INDEX idx_heartbeats_task_ts  ON heartbeats(task_id, ts);
```

> **旧库迁移（as-built，D18）**：`tasks.at` 列为 P3 增补；`DB._init_schema` 用
> `PRAGMA table_info(tasks)` 检测旧库并 `ALTER TABLE tasks ADD COLUMN at TEXT`
> 补齐，不破坏既有数据库（db.py:63-67）。

### 3.3 读写纪律

- **写配置**：设计意图是 CLI / 前端 API / inbox 指令文件三条路径统一写入
  `task_config_kv` 并产生一条 `patch_config` hook；
  **as-built 偏差**：`set_config_kv` 与 cmd `set_config` 目前只写 kv、不产生
  hook 且无白名单校验，纪律靠调用方自觉（已知问题 KI-04，见
  `progress/007-code-review.md` §7）；
- **读配置**：worker 启动时从 DB 物化出完整 config（`config_json` ⊕ `task_config_kv`），
  运行中只接受 hook 带来的 patch；
- **kv 元数据约定（as-built，D2）**：key 以 `_` 前缀（如 `_meta.output`、
  `_meta.suspend_checkpoint`、`_meta.gpu_wait_reason`）为运行时元数据，
  `materialize_config` 跳过不注入训练 config；
- JSON 文件永不回写，避免双写冲突；前端和 CLI 看到的永远是同一份数据。

---

## 4. Project 层级

### 4.1 语义

- **Project** 是一组训练任务的容器，对应一个实验主题
  （如 `klein_lora_v1`、`qwen_style_batch2`）；
- 一个 Project 内可同时存在：
  - **串联任务链**（t1 → t2 → t3，后者可从前者权重续训）；
  - **定时任务**（cron 到点触发，或 at 一次性触发）；
  - **独立任务**（无依赖，按 priority 排队或并行）；
- 项目级共享：提示词库、采样输出目录、默认模型、归档状态；
- 项目不限制任务数，任务不能跨 project 依赖（保持边界清晰）。

### 4.2 项目文件（投递到 inbox/ 即创建整个项目）

```json
{
  "project": {"name": "klein_lora_v1", "default_model": "flux2_klein"},
  "chains": [
    {
      "name": "depth_normal_inpaint",
      "tasks": [
        {"name": "t1_depth",  "config": "configs/depth.json"},
        {"name": "t2_normal", "config": "configs/normal.json",
         "depends_on": "t1_depth", "resume_from": "$task:t1_depth.output"},
        {"name": "t3_inpaint","config": "configs/inpaint.json"}
      ]
    }
  ],
  "scheduled": [
    {"name": "nightly_eval", "config": "configs/eval.json",
     "cron": "0 3 * * *", "allow_parallel": true}
  ]
}
```

`$task:<name>.output` 由 dispatcher 在前驱任务完成后解析为其产物路径；
`depends_on` 缺省即完全独立。

---

## 5. Orchestrator 主监控程序

### 5.1 Watcher：文件驱动编排

- 监听 `workspace/inbox/`（跨平台用 1s 轮询更稳）；
- 识别三类文件（as-built）：
  - **项目/任务文件** `project_*.json` / `task_*.json`：导入为 projects/tasks 行，
    移动到 `processing/`；
  - **指令文件** `cmd_*.json`：支持 action = `cancel` / `hook` / `set_config`，
    处理后移到 `done/`；
- 完成后归档 `done/`、失败移入 `failed/`；
  （设计要求的归档结果摘要 / error.log 附件**未实现**，见 known-issues KI-16）；
- 文件名去重：导入时计算内容 hash，重复投递直接归档。

### 5.2 Scheduler：cron / at 定时

- 任务可带 `"cron": "0 3 * * *"`（周期）或一次性 `"at": "2026-07-20T03:00"`（D18）；
- Scheduler 每 30s（`--tick` 可配）扫描 `status='scheduled'` 且到点的任务，
  置 `pending` 交 Dispatcher 准入判断；
- 定时与依赖可叠加：链头到点启动，后续任务仍按依赖触发；
- 到点但被 GPU Guard 拦截 → 任务置 `waiting_gpu`（见 §5.4）；
- cron 任务终态后由 dispatcher 自动 re-arm 回 `scheduled`；`at` 一次性不 re-arm（D14）；
- 防堆积：任务离开 scheduled 期间不再触发；同一分钟用 kv `_meta.last_fire` 去重，
  错过多次触发只保留一个排队实例（D14）。

### 5.3 Dispatcher：worker 池与生命周期

- **worker 池（as-built，D17）**：P3 起 dispatcher 重构为 worker 池，每 worker 独立
  监控线程，主循环只负责收割+准入；`--max-parallel` **默认 1**，默认行为与 P1
  单 worker 串行完全一致（md/04 早期草稿的默认 2 已作废）；
- 启动 worker：`subprocess.Popen([python, train.py, "--task-id", id, "--db", path])`，
  **不再传 --config 文件**，train.py 从 DB 物化配置；
- 监控心跳表：超过 120s 无心跳 → 判定僵死，kill 并标记 `failed`
  （注意：对"从未有心跳"的 worker 暂不判死，启动宽限缺失，见 KI-03）；
- worker 退出码：`0` → done；`42`（约定）→ hook 挂起 suspended，等待重启；其他 → failed；
- 任一 worker 退出后，立即重扫 `waiting_gpu` / `pending` 队列尝试补位；
- `next_runnable_tasks()` 语义（as-built，D14）：`status=pending` 即"已到点"的唯一
  语义，不再过滤 cron；cron/at 是否到点完全由 scheduler 负责。

### 5.4 GPU Guard：并行准入（定时任务与运行中任务的冲突仲裁）

规则：

> 已有训练任务在运行时，任务到点/到队触发：
> **检测 GPU 可用 VRAM > 总 VRAM 的 3/4（75% 空闲）才允许并行启动**；
> 否则该任务进入 `waiting_gpu`，等在跑任务完成后再启动。

实现：

- 采集链三级降级（as-built，D15）：`pynvml`（惰性 import）→ `nvidia-smi` 子进程
  解析 → NullProvider（无 GPU 信息：空机放行、并行保守拒绝）；
  `GPUGuard(provider=...)` 可注入 FakeProvider 供测试；
- 准入判定顺序（as-built）：① 空机放行 → ② `allow_parallel=0` 拒绝
  （"parallel not allowed"）→ ③ 达 `max_parallel` 拒绝 → ④ 存在某卡
  `free > total*3/4` → Admit(gpu=i)，否则拒绝；
- **waiting_gpu 适用于所有 pending 任务**（as-built，不再仅定时任务）；
- 并行上限可配置，**默认 1**（D17，有意保守；md/04 早期草稿写 2 已作废）；
- 并行启动的 worker 通过 `CUDA_VISIBLE_DEVICES=<gpu>` 绑定到满足条件的卡（D17）；
- 决策记录 as-built 分工（D15）：判定结果（快照+admit+reason）写 kv
  `_meta.gpu_decision`，拒绝原因写 `_meta.gpu_wait_reason`；`gpu_snapshots` 表只在
  采集到非空快照时写入（结构不变），空快照不入库；
- 不满足条件：任务 `waiting_gpu`，Scheduler 不会重复触发同一实例
  （错过的一次 cron 触发只保留一个排队实例，避免堆积）。

### 5.5 HTTP API 服务（前端后端）

- **as-built（D16）**：标准库 `http.server.ThreadingHTTPServer` 实现（环境无
  fastapi/uvicorn 且禁止 pip），路由面与原 FastAPI 设计一致，未来可整体替换；
- 与 orchestrator 同进程启动（`--api`，默认开；**无 `--api-only`**，
  API 独立运行用 `python -m orchestrator.server --workspace W [--port 8100]`）；
- REST：`/api/projects`、`/api/tasks`、`/api/tasks/{id}/hooks`、
  `/api/tasks/{id}/config`、`/api/prompts`、`/api/gpu`；
- **SSE 替代 WebSocket**（D16）：`GET /api/events` 每 2s 全量推 {tasks, gpu} 帧，
  前端据此刷新并驱动 heartbeats 轮询；
- 所有端点内部只调 `db.py` / `hooks.py` 的同一套函数（as-built 不经 dispatcher），
  与 CLI 零差异。

---

## 6. Hook 机制（训练进程内）

### 6.1 HookManager 挂载点

在 `engine/trainer.py` 的训练循环中，**每个 step 结束后**调用（as-built 签名）：

```python
hook_manager.maybe_run(step, trainer=trainer, config=config)
```

- 查询 `hooks` 表中本 task 的 `queued` 指令；
- 训练循环不感知 hook 细节，只提供上下文（模型、优化器、VAE、当前 batch）；
- 已知问题：挂载点当前未受梯度累积的 step_advanced 保护（KI-01）。

### 6.2 采样 hook 与训练解耦

- 采样能力抽成 `engine/sampling.py` 独立模块，输入 = 模型句柄 + 提示词 + 参数；
- 采样 hook 两个独立类型（as-built，原"sample --weights"参数形式已拆分）：
  - `sample`：用**当前训练中权重**（内存直取）；
  - `sample_from_weights`：**任意已训练权重**（payload 指定 weights 路径，
    临时换入 LoRA 权重采样后换回）；
- hook 触发采样时：
  1. `model.eval()` + `torch.no_grad()` + `requires_grad_(False)`（**stop gradient**），
     结束后恢复原始 requires_grad 状态（as-built，优于简单置 True）；
  2. **提示词三级优先级**（as-built）：hook payload 指定 `prompts_path`
     （提示词缓存路径由主控程序传入）> DB `prompts` 表按 `tag` 取 > 内置占位提示词；
     prompts 文件支持 **JSONL 与 txt 双格式**；
  3. 输出到 `workspace/samples/<project>/<task>/<step>_<tag>.png`，路径写回 hook.result，
     前端即刻可见；
  4. 恢复 `model.train()` 继续训练。
- 想暂停更久：`suspend` hook 先存 checkpoint，再以退出码 42 退出，显存彻底释放；
  之后可从容采样/调试，再由 orchestrator 重启。

### 6.3 权重保存 / 恢复 hook

```json
{"type": "save",    "payload": {"name": "manual_snapshot"}}
{"type": "restore", "payload": {"path": "output/manual_snapshot.safetensors",
                                "reset_optimizer": true}}
```

- `save`：立即落盘当前权重（+可选优化器状态），不中断训练；
- `restore`：加载指定权重；`reset_optimizer=true` 时重建 optimizer/scheduler。

### 6.4 配置热修改 hook

```json
{"type": "patch_config", "payload": {"training.learning_rate": 5e-5,
                                     "losses[1].weight": 0.0}}
```

- 白名单制（as-built，D12"默认拒绝"）：精确白名单 +
  `losses[i].weight/use_weighting/enabled` 正则放行；结构类（batch_size、分辨率、
  lora_rank、lokr_full_rank）与一切未知键拒绝并提示需 suspend+重启；
- **拒绝语义（as-built，D12）**：部分拒绝时已放行键照常生效并写 task_config_kv，
  hook 标 done 且 result 含 rejected 段；全部被拒才标 failed；
- **apply_live 热生效点（as-built）**：放行的键立即写入运行中 trainer 的对应对象
  （如 `optimizer.param_groups[*]["lr"]`、loss 模块的权重/开关属性、log/save 间隔），
  下一 step 生效；
- patch 同时写入 `task_config_kv`，重启后依然生效，前端配置面板同步显示。

### 6.5 优雅退出与 wandb 连续性

- `suspend`：保存 full checkpoint（权重+优化器+RNG+step）→ `wandb.finish()` → 退出码 42；
- orchestrator 重启时：`restart_count += 1`，wandb run name 约定
  `<task>-run-<n+1>`（`klein-run-1` → 重启后 `klein-run-2`），并在新 run 的
  `config["resumed_from_run"]` 记录上一个 run（**as-built 存 run name 字符串**，
  原设计要求 run id，见 md/03 备注）；
- train.py 以 full-resume 语义从 DB 记录的 checkpoint 恢复。

---

## 7. CLI 可执行入口

> 本节按 as-built 重写（与 `md/ORCHESTRATOR.md` §3 逐条一致）。全局参数
> `--workspace` `--db`。

```bash
# 常驻主控（含 API 服务）
python -m orchestrator.main --workspace workspace/ --db workspace/trainer.db \
    [--api] [--port 8100] [--max-parallel N] [--tick S]

# 项目管理
python -m orchestrator.cli project create <name> [--model M] [--desc D]
python -m orchestrator.cli project list
# project archive —— 未实现（见 progress/007-code-review.md §7）

# 任务管理（全部围绕 project）
python -m orchestrator.cli submit <json_file>          # 复制进 inbox/，无 --cron/--at flags（用 task create）
python -m orchestrator.cli task create --project P --name N [--model M] [--config f.json] \
    [--priority 100] [--cron "*/5 * * * *" | --at "2026-07-20 03:00:00"] [--allow-parallel]
python -m orchestrator.cli list [--project P] [--status S]
python -m orchestrator.cli cancel <task_id>            # 仅 pending/scheduled/waiting_gpu

# hook（等价于前端按钮 / 往 inbox/ 丢 cmd_*.json）
python -m orchestrator.cli hook <task_id> sample [--tag T] [--n 4] [--steps N] [--seed S]
python -m orchestrator.cli hook <task_id> sample_from_weights --weights <path>
python -m orchestrator.cli hook <task_id> save [--name X] [--with-optimizer]
python -m orchestrator.cli hook <task_id> restore --path <ckpt> [--reset-optim]
python -m orchestrator.cli hook <task_id> patch_config --set training.learning_rate=5e-5
python -m orchestrator.cli hook <task_id> suspend        # 存盘后退出，等重启
python -m orchestrator.cli hooks [--task-id N]           # queued→acked→done 历史
python -m orchestrator.cli resume <task_id>              # 仅 suspended

# 配置热修改 —— `config set` 未实现；用 hook patch_config 代替（见 known-issues KI-04）

# GPU 状态
python -m orchestrator.cli gpu status                  # nvidia-smi 快照入库

# worker 冒烟（DB 模式 dry-run，D1）
python UnifiedTrainer/train.py --task-id <id> --db <db> --dry-run-steps N
```

CLI 只操作数据库 + 文件，不要求 orchestrator 在线；在线时自动拾取。

---

## 8. Web 前端

### 8.1 技术栈（as-built，D19）

- **零构建静态 SPA**：`web/dist/index.html` 单文件，原生 ES module + fetch +
  EventSource，hash 路由，无 CDN 依赖，离线可开
  （环境 node 存在但 **npm 缺失**，无法走 Vite/Vue 构建链；未来补齐 npm 可按原
  Vue 3 + Vite + TS + Pinia 设计重做，API 面不变）；
- 图表：loss/lr 曲线用**内联 SVG** 绘制（ECharts 降级）；
- 通信：REST（操作）+ SSE `GET /api/events`（每 2s 全量帧，驱动任务列表刷新与
  heartbeats 轮询）；
- 由 server.py 静态托管（单端口 8100），未命中路径回退 index.html。

### 8.2 页面结构（as-built 五页面骨架）

```
#/                       → 项目列表（卡片：名称、状态、任务统计）
#/projects/:id           → 项目详情：任务列表（含链/定时）+ 提示词库（列表+新增）
#/tasks/:id              → 任务监控页
    ├─ loss/lr 曲线（内联 SVG，SSE 帧驱动轮询 heartbeats）
    ├─ hook 操作面板      六个按钮：sample / sample_from_weights / save /
    │                     restore / patch_config / suspend（+ resume）
    ├─ 采样画廊           workspace/samples/<project>/<task>/ 缩略图
    └─ hook 历史          每条指令的状态与结果
#/gpu                    → GPU 面板：各卡 VRAM、util、waiting_gpu 队列及拒绝原因
#/new                    → 新建项目/任务（表单 ↔ JSON 双视图，可直接粘贴 JSON）
```

### 8.3 关键交互（as-built 与降级清单）

- **热改配置**：任务页修改白名单内字段 → `PATCH /api/tasks/:id/config`
  → 后端发 `patch_config` hook，worker 下一 step 生效（注：API 侧白名单校验缺失，
  见 KI-04/KI-20）；
- **一键 suspend/resume**：按钮即 hook / resume API，重启后 wandb run name 自动 +1；
- 所有前端操作与 CLI 等价，共用同一 API/DB，状态天然一致；
- **未实现的交互降级**（相对原 Vue 设计）：无 cron 人性化预览、无链式创建向导
  （依赖 id 手填）、无 DAG 图 / 下次触发时间列 / 任务详情抽屉 / 日志 tail /
  wandb 链接、提示词库仅列表+新增（无编辑/删除）、项目卡片无最近活动。

---

## 9. 文档与 Agent 集成

1. **`md/ORCHESTRATOR.md`**（人用）：架构图、项目/任务模型、表结构、CLI 参考、
   API 参考、任务文件格式、hook 协议、GPU Guard 规则、故障排查。
2. **`skills/unified-trainer-orchestrator/SKILL.md`**（Agent 用，符合 skill 格式）：
   - 触发场景（"用户要求启动/修改/监控训练任务"）；
   - 标准操作流：生成项目 JSON → 投递 inbox/ → 轮询状态 → 汇报；
   - 常用配方：串行链模板、cron + allow_parallel 模板、采样 hook 模板、suspend/重启模板；
   - 红线：不直接写 DB 内部表；热改 config 前必查白名单。
3. Agent 由此可以**只通过写文件/调 CLI/调 API** 完成
   "建项目 → 配任务 → 提交 → 监控 → 热改 → 续训"全流程。

---

## 10. 实施路线

| 期 | 内容 | 验收 |
|----|------|------|
| P1 | SQLite schema（含 projects/gpu_snapshots）+ db.py；train.py 支持 `--task-id/--db`；orchestrator 最小版（watcher + dispatcher 单 worker 串行）；project/task 文件导入 | 投递项目文件即自动跑通串行链并归档 |
| P2 | HookManager + sample/sample_from_weights/save/restore/patch_config/suspend **六类** hook；退出码 42 + 自动重启 + wandb run-N 命名 | 运行中 CLI 发 hook 全部生效；suspend 后无损续训 |
| P3 | cron/at scheduler + GPU Guard（3/4 VRAM 准入 + waiting_gpu）+ worker 池（--max-parallel 默认 1）+ HTTP API 全套 REST/SSE（as-built：标准库实现，D16） | 运行中触发定时任务：VRAM 够则并行、不够则排队补位 |
| P4 | Web 前端（as-built：零构建静态 SPA，D19）+ ORCHESTRATOR.md + SKILL.md | 前端与 CLI 功能对齐；Agent 仅凭 skill 文档完成全流程 |

### 对 UnifiedTrainer 副本代码的主要改动点

- `UnifiedTrainer/train.py`：新增 `--task-id/--db` 模式（与旧 `--config` 模式并存，
  向后兼容）与 `--dry-run-steps`（D1）；
- `UnifiedTrainer/engine/trainer.py`：训练循环注入 `hook_manager.maybe_run()`；
  内置采样逻辑抽离到 `engine/sampling.py`；
- `UnifiedTrainer/engine/checkpoint.py`：full checkpoint 补 wandb run 元数据；
- 新增 `orchestrator/`、`web/`，不侵入 `models/` `losses/` `data/`。


---

## 11. MiniMax-H3 训练接入（2026-08，图像对 → 视频，音频延后）

> 实现计划见 `md/minimaxh3_implementation.md`，阶段小结见
> `progress/012-minimaxh3.md`，as-built 设计见 `md/07-minimaxh3-training.md`，
> 决定记录见 `agent/decisions.md` H3-D1–H3-D10。**状态：代码交付完成；
> G2–G7 运行时验收需模型权重，延后未运行。**

### 11.1 目标

- **里程碑 1（图像对训练，优先）**：把 MiniMax-H3 当图像模型训练——源图作
  关键帧条件行、目标图作目标视频行（1 帧视频即图像），配文本 caption。
- **里程碑 2（视频训练）**：激活（不重写）P1 已建成的统一媒体管线视频分支，
  同一条适配器路径扩展到 T>1 帧。
- **音频训练本期不做**（D6 正式决定）。

### 11.2 决策摘要（H3-D1–H3-D10，编号与计划 §3 一致）

| # | 决策 | 要点 |
|---|------|------|
| H3-D1 | 依赖固定到 diffusers `pr-14355` 分支 | 训练/推理共用同一源码；requirements 注释注明 PR #14355；新增 `av`（PyAV 16.1.0），不装 torchaudio |
| H3-D2 | Transformer 直接 import diffusers | `MiniMaxH3Transformer3DModel` 原生 PeftAdapterMixin；LoRA + 梯度检查点起步；block swap 作 P4 增强（OOM 才 vendor） |
| H3-D3 | 数据层一次建成：统一媒体管线 | `media` 字段 + `video_frames/video_fps` 一次入 schema；统一 5D 缓存 (C,T,H,W)，图像 = (C,1,H,W)；cache_builder 单一媒体分发；`data/video_utils.py` 一次实现；里程碑 2 只激活，不改 schema/缓存格式/collate |
| H3-D4 | 新增 `MiniMaxH3Adapter` | 全部抽象方法 + 新可选钩子 `encode_video` / `decode_validation_video` / `velocity_sign`；音频钩子延后 |
| H3-D5 | data-ward 速度约定 | `unpack_prediction` 返回模型原值；loss 按 `velocity_sign` 取反 target（flow_matching 一行） |
| H3-D6 | σ→t=1−σ 时间约定 | 引擎插值逐位等价，无需改；关键帧 t_cond=0.999；每前向最多 2 个去重 timestep；音频行整体省略（空 (B,0,32)） |
| H3-D7 | LoRA 目标模块按 H3 命名 | `["to_q","to_k","to_v","to_out","ff.net.0.proj","ff.net.2"]`（swiglu FF 命名，实测固化）；冻结 proj_in/.../time_embedder（fp32 契约） |
| H3-D8 | 验证生成 | `load_scheduler` 返回 `MiniMaxH3Scheduler(shift=12)`，CFG 关闭（guidance_scale=1）；解码按帧数分发（T==1→PIL，T>1→静音 mp4） |
| H3-D9 | 编排器零代码改动 | 任务即 `train.py --task-id`；61.7GB transformer 用 NF4/int8 或 2×80GB DDP 过 GPU Guard |
| H3-D10 | 验收 = 数值 parity + 过拟合 | 速度符号校验 → 图像对过拟合（里程碑 1）→ 视频过拟合（里程碑 2）→ LoRA 训练验证；运行验收延后 |

### 11.3 数据改动点（统一媒体管线）

- **schema**：`ImageConfig.media`（"image"|"video"，默认 "image"）、
  `DatasetConfig.video_frames=124/video_fps=24`；validate 规则（video 键只能
  被同名媒体 target 引用、17n+5 对齐）一次落地。
- **`data/video_utils.py`**：`load_image_frames` → `(1,C,1,H,W)`；
  `load_video_frames`（PyAV 解码、24fps 抽帧、17n+5 对齐、5–15s 校验）→
  `(1,C,T,H,W)`；`snap_frames` / `video_latent_num_frames` 包装 PR 函数。
- **缓存**：cache_builder 单一媒体分发 → `adapter.encode_video` → 统一
  (C,T,H,W) npz；每样本 JSON 记录 media/num_frames。
- **dataset/collate/bucket**：5D 堆叠与 bucket 在 P1 以图像样本 (C,1,H,W)
  打通硬化；P2 视频只是 T 变大，代码零改动。
- **引擎小改（D6 授权）**：`sigmas.view(-1,1,1,1)` → ndim 通用展开
  `view(-1, *(1,)*(ndim-1))`（trainer.py + noise_selector.py），旧 4D view
  对 5D latent 右对齐成 (1,B,1,1,1)，B>1 必炸；4D 行为逐位一致。

### 11.4 图像对 / 视频配置

- 图像对 smoke：`configs/minimax_h3_image_smoke.json`（S=源图关键帧、
  P=目标图，caption C→S 带 vision block）。
- 生产图像对：`configs/minimax_h3_train.json`（nf4 + adamw8bit +
  gradient_checkpointing + `multi_gpu: "reserve"` + `guidance_scale: 1.0`）。
- 视频 smoke：`configs/minimax_h3_video_smoke.json`（V=.mp4 media video、
  `video_frames=124` → 37 latent 帧；`reference_configs: {"none": []}` =
  纯 t2v 无关键帧）。

### 11.5 音频延后原因（D6）

- PR packed 布局 `[ text | 关键帧 | 音频(A) | 视频(V) ]` 中音频块可整体
  省略：`num_audio_latents=0` → 无 A 块，transformer 内 `audio_proj_in(空)`
  + `index_copy` 空索引均为 no-op（代码级核实）。
- 适配器 `audio_hidden_states` 传空 `(B, 0, 32)`（audio_proj_in 输入维）。
- 不下载 audio_vae/audio_scheduler，不实现音频数据/损失/解码代码——音频
  训练是后续阶段，本期视频 = 无声视频建模。

### 11.6 状态与延后项

- 已交付（无权重验证全绿）：packing 布局、统一媒体管线、适配器 53/53、
  图像对/视频 smoke 配置 validate、LoRA 目标模块固化 + checkpoint 往返、
  --list-models 回归、krea2 44 配置回归。
- **延后（需模型权重）**：G2 图像对缓存、G3 dry-run + 符号校验、
  G4 图像对过拟合、G5 视频缓存、G6 视频过拟合、G7 编排器全链路；
  P4 可选增强（PR parity、量化冒烟、DDP 冒烟）。
