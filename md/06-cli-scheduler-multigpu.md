# 06 — CLI Scheduler CRUD + Multi-GPU Support (Design Plan)

> Status: **implemented (P5a–P5f, 2026-08-03)** — see §9 as-built notes and
> `progress/011-p5-cli-multigpu.md`; verification: `tmp/test_p5.py` (90 checks).
> Scope: `orchestrator/` (cli, db, schema, scheduler, gpu_guard, dispatcher,
> server, watcher) + `UnifiedTrainer/train.py` DDP readiness + `web/dist` parity.
> All constraints from `doc/REQUIREMENTS.md` apply (stdlib-only orchestrator,
> no new pip deps for orchestrator, self-contained, decisions → `agent/`).

---

## 1. Goals

| # | Goal | As-is gap |
|---|------|-----------|
| A | **CLI-first scheduler**: full CRUD for projects, tasks, schedules; table views; pause/resume; logs | CLI today can only *create* tasks/projects, list, cancel, hook, resume. No show/update/delete/pause; schedules are not first-class; no log access; table output only human format. |
| B | **Multi-GPU jobs**: a task/schedule declares how many cards it uses (`gpus`), optionally pinned cards (`gpu_ids`); guard admits N cards; dispatcher launches DDP via `accelerate launch`; occupancy-aware admission | Today: 1 task = exactly 1 GPU (`CUDA_VISIBLE_DEVICES=<i>`), `best_gpu()` picks one card, no occupancy accounting, no multi-process launch. |

Both features share one principle: **SQLite stays the single source of truth**;
CLI, API, inbox files keep calling the same `db.py` / `hooks.py` functions
(global decision 3 in `00-architecture.md`).

---

## 2. Feature A — CLI Scheduler (CRUD)

### 2.1 Target command surface

```
python -m orchestrator.cli <group> <action> ...     [--workspace W] [--db D] [--json]
```

Global: new `--json` flag on every list/show command (machine-readable output;
default stays the human table). All commands keep exit-code discipline:
`0` ok, `1` runtime/not-found, `2` bad usage/validation.

`job` is registered as an **alias** of `task` (both dispatch to the same
handlers; `task` stays canonical in help text) so "add job" workflows work
verbatim.

#### projects
| Command | Behavior |
|---|---|
| `project create <name> [--model M] [--desc D]` | (existing) |
| `project list` | (existing) |
| `project show <id-or-name>` | row + counts of tasks by status |
| `project update <id-or-name> [--desc D] [--model M] [--tags T]` | PATCH via `db.update_project` |
| `project archive <id-or-name>` | `status='archived'`; rejects if it has non-terminal tasks |
| `project delete <id-or-name> --yes` | hard delete only if **all** its tasks are terminal (`done/failed/cancelled`); deletes tasks + their hooks/kv/heartbeats in one transaction |

#### tasks
| Command | Behavior |
|---|---|
| `task create ...` | (existing) **+ `--gpus N` + `--gpu-ids "0,1"` (Feature B)** |
| `task list [--project P] [--status S]` | (existing; add columns `cron`,`at`,`gpus`; `--json`) |
| `task show <id>` | full row + `config_kv` (meta incl.) + materialized config summary; `--json` gives all three blobs |
| `task update <id> [--name] [--priority] [--allow-parallel/--no-allow-parallel] [--depends-on ID] [--resume-from P]` | structural edits; **rejected while `running`** (except priority); re-validates name via `validate_name` |
| `task pause <id>` | only from `scheduled`/`pending`/`waiting_gpu` → new status **`paused`** |
| `task unpause <id>` | `paused` → `scheduled` if cron/at set else `pending` |
| `task delete <id> --yes` | only terminal or `paused`; running tasks must be cancelled/suspended first. Deletes hooks + config_kv + heartbeats of the task, then the row. |
| `task logs <id> [-f] [-n LINES]` | reads `workspace/logs/task_<id>.run_<restart_count>.log` (see 2.3); `-f` tails |
| `task retry <id>` | convenience: terminal `failed` task → `pending` (clears error, keeps restart_count) |

#### schedules (first-class view over `tasks.cron` / `tasks.at`)
Design decision: **no new `schedules` table.** A schedule is a task with
`cron`/`at` set; a separate table would fork the single-source-of-truth and
double the re-arm/dedup logic. Instead a `schedule` command group gives
schedule-centric UX on top of the same columns.

| Command | Behavior |
|---|---|
| `schedule list [--project P]` | all tasks with cron/at: `id, name, cron/at, next_fire, last_fire(_meta.last_fire), status, gpus` |
| `schedule add --project P --name N --cron "0 3 * * *" --config f.json [--gpus N ...]` | sugar over `task create --cron` (status=`scheduled`) |
| `schedule set <task_id> --cron EXPR` / `--at ISO` / `--none` | validate via `scheduler.validate_schedule`; write column; `--none` clears both and moves `scheduled`→`pending`; clears `_meta.last_fire` so an `at` task can be re-armed; re-arms: if task is terminal and now has cron → back to `scheduled` (respect re-arm block `_meta.rearm_blocked`) |
| `schedule pause <task_id>` / `schedule unpause <task_id>` | alias of `task pause/unpause` (schedule semantics) |
| `schedule preview <cron-expr> [--count K]` | prints next K fire times (local tz) — no DB needed |

**Next-fire computation** — new method on `CronExpr` in `scheduler.py`:

```python
def next_after(self, dt: datetime, horizon_days: int = 366) -> Optional[datetime]:
    """First local-time minute > dt matching the expression (minute-walk, capped)."""
```

Minute-walk with a cap is sufficient (5-field cron, no seconds); used by
`schedule list` and API. `_parse_at` tasks show the `at` time directly
(struck through if `_meta.last_fire` already set).

#### prompts
| Command | Behavior |
|---|---|
| `prompt add --project P --tag T --text "..." [--negative ...]` | `db.add_prompt` |
| `prompt list [--project P] [--tag T]` | (db exists, CLI missing) |
| `prompt delete <id> --yes` | new `db.delete_prompt` |

#### gpu
| Command | Behavior |
|---|---|
| `gpu status` | (existing) **+ occupancy column**: which task id currently holds each card (from `_meta.gpu_index` of running tasks) |

### 2.2 Schema / DB changes (Feature A)

- `schema.sql` comment: add `paused` to the task status enum list
  (`pending / scheduled / running / suspended / paused / done / failed /
  cancelled / waiting_gpu`). No DDL needed (status is TEXT).
- New status transitions enforced in code, not DB triggers:
  - `paused` only entered from `scheduled|pending|waiting_gpu`;
  - dispatcher `next_runnable_tasks` unchanged (it only picks
    `pending/waiting_gpu`, so `paused` is naturally skipped);
  - `hooks.resume_task` must reject `paused` (resume is for `suspended` only).
- `db.py` additions:
  - `update_task(task_id, **fields)` — allow-list
    `{name, priority, depends_on, resume_from, resume_mode, cron, at,
    allow_parallel, gpus, gpu_ids}`; name goes through `validate_name`;
    cron/at validated by caller via `validate_schedule`.
  - `delete_task(task_id)` — transaction: delete hooks → config_kv →
    heartbeats → task row; caller enforces status precondition.
  - `delete_prompt(prompt_id)`, `delete_project(project_id)` (project delete
    cascades task-by-task inside one transaction).
  - `task_log_path(task_id, restart_count)` helper is **not** in db (file
    concern, lives in dispatcher/cli).

### 2.3 Task logs (enables `task logs`)

Dispatcher currently lets workers inherit stdout/stderr. Change
`_start_worker`:

```
log_dir = workspace/logs/
log_path = log_dir / f"task_{tid}.run_{restart_count}.log"
proc = Popen(cmd, stdout=open(log_path,'ab'), stderr=STDOUT, ...)
```

- one file per (task, restart) so resume runs don't truncate history;
- CLI `task logs` resolves latest `run_*` file by default;
- retention: pruned together with heartbeats in `main.py` hourly prune
  (delete log files of tasks terminal > 30 days).

### 2.4 API + frontend parity (Feature A)

Same functions, mirrored in `server.py` (keep `BaseHTTPRequestHandler`, no
FastAPI — zero new deps):

- `GET /api/tasks/<id>` (existing, add `log_file` path field),
- `PATCH /api/tasks/<id>` → `db.update_task` (+ schedule validation),
- `DELETE /api/tasks/<id>` → same precondition as CLI,
- `POST /api/tasks/<id>/pause` / `/unpause`,
- `GET /api/schedules` → list view incl. computed `next_fire`,
- `PATCH /api/projects/<id>` (existing) + `DELETE /api/projects/<id>`,
- `DELETE /api/prompts/<id>`,
- `_create_task` body accepts `gpus`, `gpu_ids` (Feature B).

Frontend (`web/dist/index.html`, zero-build SPA): add schedule column +
next_fire to task table, pause/unpause buttons, gpus field in the create
form, GPU occupancy badge. Minimal edits, same file, no build step.

---

## 3. Feature B — Multi-GPU Support

### 3.1 Data model

`tasks` gets two columns (migration in `db._init_schema`, same pattern as the
existing `at` migration):

```sql
ALTER TABLE tasks ADD COLUMN gpus    INTEGER NOT NULL DEFAULT 1;  -- cards required
ALTER TABLE tasks ADD COLUMN gpu_ids TEXT;                        -- optional pin "0,1"
```

- `gpus`: how many cards the job occupies (validated `>= 1`; upper bound
  checked against live GPU count at create time — warn, don't hard-fail,
  since cards may appear later).
- `gpu_ids`: optional comma list of physical indices ("0,2"); if set, its
  length must equal `gpus`. Pinning skips the free-VRAM *selection* but not
  the free-VRAM *threshold* check.
- Both exposed: CLI flags `--gpus`, `--gpu-ids`; inbox `task_*.json` keys
  `gpus`, `gpu_ids`; API create body; `_UPDATABLE_TASK_FIELDS` +
  `update_task` allow-list.
- Recorded decision metadata (kv): `_meta.gpu_index` becomes a **list**
  (single-card tasks store `[i]` — consumers must migrate to list handling;
  keep writing legacy scalar too as `_meta.gpu_index_legacy` for one release
  cycle? **No** — decided: write list only; `train.py` dry-run echo writes
  `_meta.cuda_visible_devices` string which already covers debugging).

### 3.2 Occupancy accounting

New small module-level state in `dispatcher.py` (in-memory, authoritative):

```python
def occupied_gpus(self) -> set[int]:
    # union of gpu lists of all live workers (self._workers[*]["gpus"])
```

Passed into `GPUGuard.judge(task, running_workers, occupied=<set>)`.
The guard also cross-checks persisted `_meta.gpu_index` of DB-`running` tasks
not present in `self._workers` (covers orchestrator restart before heartbeat
timeout reaps zombies).

`max_parallel` keeps its meaning (worker *count*), plus a new capacity rule:
`len(occupied ∪ requested) <= total GPU count` must hold.

### 3.3 GPUGuard multi-card admission

`gpu_guard.py` changes:

- `Admit` carries `gpus: list[int]` (`Admit.gpu` kept as property =
  `gpus[0]` for back-compat logging).
- New selector replacing `best_gpu` for N>1:

```python
def select_gpus(gpus: list, n: int, occupied: set,
                pinned: Optional[list[int]] = None) -> Optional[list[int]]:
    # pinned: verify exactly those cards satisfy FREE_FRACTION
    # auto:   sort candidates (not occupied, free > total*FREE_FRACTION)
    #         by free_mb desc, take first n; None if fewer than n qualify.
```

- `judge()` flow (extends current order):
  1. empty machine (`running_workers == 0` **and** nothing occupied) → admit
     `select_gpus(..., occupied=set())`;
  2. metrics collection failed → Wait (as today);
  3. `allow_parallel == 0` **and** anything else running → Wait (unchanged —
     a `gpus>1` job is inherently parallel-with-itself, that's fine);
  4. `running_workers >= limit` → Wait;
  5. GPU count capacity check → Wait "not enough cards free (need N, M usable)";
  6. `select_gpus` → Admit(list) or Wait (insufficient free VRAM).
- `_meta.gpu_decision` records `gpus: [...]`.

### 3.4 Dispatcher launch modes

**Launch-mode staging** — a task-level config flag gates how K>1 cards are used,
so allocation can ship before the engine is DDP-safe:

```jsonc
"training": { "multi_gpu": "reserve" | "ddp" }   // default "reserve"
```

- `reserve` (default, safe): the task **occupies and binds** K cards
  (`CUDA_VISIBLE_DEVICES` exposes exactly those K) but spawns a single process —
  training runs on the first visible device. This is orchestrator-only, fully
  backward compatible, and already prevents GPU over-subscription.
- `ddp`: spawn `accelerate launch --num_processes=K` so all K cards compute.
  Requires the 3.5 rank gates; rejected at create time when combined with
  `block_swap`/torchao (see 3.5).

`_start_worker(task, gpus: list[int])`:

| Condition | Launch |
|---|---|
| `len(gpus) == 1` | unchanged: `[python, train.py, --task-id, --db]`, `CUDA_VISIBLE_DEVICES="i"` |
| `len(gpus) > 1`, mode `reserve` | unchanged command, `CUDA_VISIBLE_DEVICES="a,b,..."` |
| `len(gpus) > 1`, mode `ddp` | `[python, -m, accelerate.commands.launch, --num_processes, N, --multi_gpu, --main_process_port, <free>, --mixed_precision, bf16, train.py, --task-id, --db]` with `CUDA_VISIBLE_DEVICES="a,b,..."` |

- `--main_process_port` is a per-worker unique port from a stdlib `_free_port()`
  helper (bind socket to port 0, read assigned port, close) — avoids NCCL port
  collisions between concurrent multi-GPU jobs. Orchestrator stays stdlib-only:
  `accelerate` is provided by the worker interpreter, never imported here.
- Mixed precision value taken from materialized config
  (`training.mixed_precision`, default `bf16`); if the config says `no`,
  omit the flag.
- `CUDA_VISIBLE_DEVICES` remaps physical→local indices, so accelerate sees
  devices `0..K-1` regardless of physical ids — no `--gpu_ids` launch arg needed.
- `accelerate launch` is used instead of raw `torchrun` because
  `accelerate>=0.26.1` is already a hard dependency of the trainer; no new
  deps.
- **Process-group kill** (multi-process spawn makes orphan cleanup a real
  risk): spawn with `creationflags=CREATE_NEW_PROCESS_GROUP` (Windows) /
  `start_new_session=True` (POSIX); `_monitor` kill path kills the whole
  group (`taskkill /PID <pid> /T /F` on Windows, `os.killpg(SIGKILL)` on
  POSIX) before the existing `proc.wait(timeout=30)`.
- `self._workers[tid]["gpu"]` → `["gpus"] = list[int]` (rename; `gpu`
  property kept in logs).

### 3.5 Trainer-side DDP readiness (`UnifiedTrainer/`)

`engine/trainer.py` already constructs `Accelerator(...)`, so under
`accelerate launch` it will run DDP automatically — **but** several features
are rank-unsafe and must be guarded:

| Item | Rule |
|---|---|
| DB writes (heartbeats, `_meta.output`, hook ack/finish, exit-42 suspend) | only when `accelerator.is_main_process` (or `int(os.environ.get("RANK","0"))==0` fallback). `train.py` DB-mode block gets a `_is_rank0()` gate. |
| Validation / sampling hooks (`HookManager`) | rank-0 only; non-rank-0 workers skip hook polling but still train. |
| Checkpoint saving | rank-0 only; add `accelerator.wait_for_everyone()` before and after save. |
| W&B init / logging | rank-0 only (env `WANDB_DISABLED=true` injected on non-rank-0 by launch wrapper — set inside train.py before wandb import based on RANK). |
| `torchao` quantized models (`device_placement=False`) | **block at task-create time**: `gpus > 1` + config `quantize` in {torchao_float8, torchao_int8, torchao_int4} → CLI/API error with explanation (quantized base + DDP unsupported in this codebase). |
| `block_swap` / `bouncing_offloader` | same: block `gpus > 1` when `block_swap` enabled (CPU-offload hooks are not DDP-safe). |
| Effective batch size | documented semantics: DDP multiplies global batch by N (`batch_size` stays per-GPU). `task show`/frontend surfaces `effective_bs = bs * gpus`. No auto-scaling of LR (documented; user controls via config). |
| Sampler determinism | per-rank seed = `seed + rank` where the trainer seeds RNGs, to avoid identical augmentations across ranks. |

**Dry-run gate**: `train.py --dry-run-steps` must also work under
`accelerate launch` (it skips model load, so it's cheap); a new
`task check <id>` CLI command runs the dispatcher's exact launch command with
`--dry-run-steps 2` and reports pass/fail — the recommended pre-flight for any
multi-GPU task before first dispatch.

### 3.6 CLI / API / frontend surface for multi-GPU

- `task create --gpus 2 [--gpu-ids 0,1]`; validation: `len(gpu_ids)==gpus`,
  indices within known GPU count (warn if unknown), conflicts with torchao /
  block_swap configs rejected early.
- `task list` / `task show`: `gpus` column; running tasks show assigned cards.
- `schedule list`: `gpus` column.
- `gpu status`: per-card `task_id` occupancy + free/total.
- `schedule add` accepts the same flags.
- Inbox `task_*.json`: `"gpus": 2, "gpu_ids": "0,1"` parsed by watcher
  (`_import_task`), validated via the same shared validator (new
  `orchestrator/validation.py::validate_gpu_request(config, gpus, gpu_ids)`
  reused by CLI, watcher, server — same pattern as `validate_schedule`).
- API: `POST /api/tasks` + `PATCH /api/tasks/<id>` accept both fields;
  `GET /api/gpu` adds `occupancy: {gpu_index: task_id}`.

---

## 4. File change matrix

| File | Feature | Changes |
|---|---|---|
| `orchestrator/schema.sql` | A+B | status comment + `gpus`/`gpu_ids` columns (new DBs) |
| `orchestrator/db.py` | A+B | migrations for `gpus`/`gpu_ids`; `update_task`, `delete_task`, `delete_prompt`, `delete_project`; `create_task` kw pass-through; occupancy query helper |
| `orchestrator/cli.py` | A+B | new subcommands (2.1), `--json`, log tail |
| `orchestrator/scheduler.py` | A | `CronExpr.next_after`; no semantic change to firing |
| `orchestrator/gpu_guard.py` | B | `Admit.gpus`, `select_gpus`, occupied-set param, capacity rule |
| `orchestrator/dispatcher.py` | A+B | worker logs to file; `gpus` per worker; accelerate-launch spawn; process-group kill; occupancy derivation |
| `orchestrator/validation.py` (new) | B | `validate_gpu_request` shared by CLI/watcher/server |
| `orchestrator/watcher.py` | B | parse `gpus`/`gpu_ids` from task/project JSON |
| `orchestrator/server.py` | A+B | routes 2.4 + gpus fields + occupancy in /api/gpu |
| `orchestrator/main.py` | A | log-file retention in hourly prune |
| `UnifiedTrainer/train.py` | B | rank-0 gates (DB, hooks, ckpt, wandb), seed-per-rank, dry-run DDP compat |
| `UnifiedTrainer/engine/trainer.py` | B | `wait_for_everyone()` around checkpoint save; sampler seed |
| `web/dist/index.html` | A+B | table columns, pause/unpause, gpus field, occupancy badge |
| `md/06-*` (this file) | — | mark as-built on completion |
| `agent/decisions.md`, `progress/0NN-*.md` | — | per REQUIREMENTS rules 2 & 3 |

## 5. Implementation phases

| Phase | Deliverable | Depends on |
|---|---|---|
| **P5a** CLI CRUD core | `task show/update/delete/pause/unpause/retry/logs`, `project show/update/archive/delete`, `prompt *`, `--json`, worker log files, DB fns | — |
| **P5b** schedule group | `schedule list/add/set/pause/unpause/preview`, `CronExpr.next_after`, API `/api/schedules` | P5a |
| **P5c** multi-GPU admission | schema migration, `validation.py`, `select_gpus`, occupancy, `gpu status` occupancy, CLI/API flags | — (parallel with P5a) |
| **P5d** multi-GPU reserve mode | multi-card bind (`CUDA_VISIBLE_DEVICES` CSV), `multi_gpu=reserve` default launch, process-group kill, worker log files | P5c |
| **P5e** multi-GPU ddp mode | accelerate-launch spawn + `_free_port`, train.py/trainer.py rank gates, `task check` dry-run gate | P5d |
| **P5f** frontend + docs | `web/dist` parity, as-built doc update, progress summary | P5a–P5e |

Suggested order: P5a → P5c → P5b → P5d → P5e → P5f (admission before
execution; `reserve` ships independently of the DDP engine work).

## 6. Test plan (stdlib `unittest`, files under `ScheduledTrainer/tests/`)

1. **CLI CRUD** (`test_cli_crud.py`): temp workspace + DB; create→show→update→
   pause→unpause→cancel→delete happy paths; delete/pause precondition
   rejections (exit codes 1/2); `--json` shape.
2. **Schedule** (`test_schedule.py`): `next_after` against hand-computed cron
   cases (DST ignored — minute walk in local time, document); `schedule set`
   re-arm from terminal; `--none` clearing.
3. **Multi-GPU admission** (`test_gpu_guard_multi.py`, FakeProvider):
   4 fake GPUs; N=2 picks the two freest; occupied cards excluded; pinned
   cards respected & rejected when busy; capacity rule; empty-machine admit;
   wait reasons recorded in `_meta.gpu_decision`.
4. **Dispatcher spawn** (`test_dispatcher_multigpu.py`): fake train script
   (echo env + exit 0); verify `CUDA_VISIBLE_DEVICES="0,1"`,
   `--num_processes 2` present only when gpus>1; process-group kill leaves no
   orphans (spawn a sleeper child, kill, assert gone).
5. **DDP dry-run smoke**: `task check` on `krea2_100_test.json` with
   `gpus=2` → `--dry-run-steps 2` passes on a 2+ GPU machine (manual gate,
   recorded in progress).
6. **Regression**: existing single-GPU behavior unchanged — all P1–P4
   scenarios in `progress/004-*` rerun green with `gpus=1` default.

## 7. Risks & constraints

- **Windows process trees**: `accelerate launch` children are a real orphan
  risk; process-group kill (3.4) is mandatory, not optional.
- **SQLite write contention under DDP**: rank-0-only DB writes (3.5) — if
  rank>0 ever writes, WAL + busy_timeout will mask it until corruption;
  enforced by gate, covered by test 4.
- **block_swap/torchao + DDP**: explicitly rejected at create time rather
  than failing mid-training; revisit only with a concrete need.
- **`_meta.gpu_index` shape change** (int → list): any consumer (server GPU
  view, frontend) must handle lists; grep all readers during P5c.
- **LR/batch semantics**: DDP changes effective batch size; document only,
  no hidden auto-scaling.
- `gpus` > physical card count: allowed in DB with a warning; such tasks sit
  in `waiting_gpu` with a clear reason until cards exist.

## 8. Out of scope (explicit)

- Cross-node / multi-machine training.
- Model-parallel / pipeline-parallel (only data-parallel via accelerate).
- Per-GPU heterogeneous requirements (e.g. "card 0 needs more VRAM").
- Separate `schedules` table (decision in 2.1).
- GPU MIG / time-slicing partitioning.

---

## 9. As-built（2026-08-03，P5a–P5f 全部实施完成）

实现摘要与验证见 `progress/011-p5-cli-multigpu.md`；实施决策见
`agent/decisions.md` D27。本节只记**与本文设计的偏差**：

1. **`task check <id>` dry-run 预检命令未实现**（§3.5/§5 P5e 交付物）：本环境无
   torch 无法端到端验证；预检仍可用 `python UnifiedTrainer/train.py --task-id N
   --db ... --dry-run-steps 2` 手动执行。
2. **`engine/trainer.py` 未改**（§3.5）：checkpoint `wait_for_everyone()` 与
   seed-per-rank 未动；rank 门控全部集中在 `train.py` 顶层保存点与 `_is_rank0()`
   （`RANK` env 判断，位于 module 级、Accelerator 构造之前，故不用
   `accelerator.is_main_process`）。seed-per-rank 仅文档化。
3. **`GET /api/tasks/<id>` 未附 `log_file` 字段**（§2.4）：路径可从
   workspace/logs 推导，未加。
4. **前端未加 pause/unpause 按钮**（§2.4）：仅做了 GPU ×N / 钉卡 / 定时表卡数列 /
   GPU 页 occupancy 标签 / 新建表单 gpus 输入 / paused 配色；暂停恢复走 CLI/API。
5. **dispatcher 占用不并入 DB 侧 running 任务**（§3.2 cross-check）：
   `occupied_gpus()` 只统计活 worker；DB 侧 `running_task_gpus()` 覆盖展示
   （server /api/gpu occupancy、cli gpu status），不合并进准入，避免重启后误判。
6. **实施期补充拒绝**（§3.5 之外的组合）：`ddp + gpus>1` 且
   `training.mixed_precision='no'` 在创建期直接拒绝（DDP 无梯度同步精度异常，
   保持保守）。
7. **测试位置**：§6 计划写 `tests/`，实际沿用 progress/010 惯例
   `tmp/test_*.py`（gitignored），`tmp/test_p5.py` 90 项断言。
8. **worker 日志 retention 已实现**：§2.3/§4 的 main.py 每小时裁剪落地为
   `dispatcher.prune_worker_logs(log_dir, db, keep_days=30)`（终态超龄删除，
   非终态保留），与 heartbeats/gpu_snapshots 同节奏。
