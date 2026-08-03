# 011 — P5：CLI 全量 CRUD + 多卡支持（实施完成）

> 日期：2026-08-03。范围：`orchestrator/`（cli、db、schema、scheduler、
> gpu_guard、dispatcher、server、watcher、validation、main）+ `UnifiedTrainer/train.py`
> + `web/dist/index.html`。设计依据 `md/06-cli-scheduler-multigpu.md`（已标
> as-built）。决策记录：`agent/decisions.md` D26（设计）/ D27（实施）。
> 验证：`tmp/test_p5.py` 90 项断言全绿 + CLI/API 冒烟。

---

## 1. Feature A — CLI Scheduler（CRUD 命令面）

`cli.py` 重写为完整 CRUD 面（入口 `python -m orchestrator.cli`）：

| 组 | 命令 | 要点 |
|----|------|------|
| `project` | `create / list / show / update / archive / delete` | show 含分状态任务计数；delete 需 `--yes` 且全部任务终态，叶子优先逆 id 序删任务再删项目 |
| `task`（`job` 为别名，同一组处理器） | `create / list / show / update / pause / unpause / retry / delete / logs` | create 增 `--gpus/--gpu-ids`；show 输出 materialized config + `effective_bs = bs*gpus` + `next_fire`；update 运行中仅允许改 priority/allow_parallel；logs 读 `logs/task_<id>.run_<restart>.log`（`-f` 跟随 / `-n` 尾行） |
| `schedule` | `list / add / set / pause / unpause / preview` | **不建 schedules 表**，仍是 `tasks.cron/at` 的一等视图；`set --none` 清定时并 `scheduled→pending`、清 `_meta.last_fire`；终态任务重新获得 cron 自动 re-arm（尊重 `_meta.rearm_blocked` 熔断） |
| `prompt` | `add / list / delete` | 提示词库 CRUD |
| `gpu` | `status` | 增 occupancy 列：每张卡的归属任务 id（`running_task_gpus()` 反查） |
| 全局 | `--json` | 所有 list/show 输出 JSON（`_emit` 统一分发），叶子子命令也支持 |

退出码纪律：`0` 成功 / `1` 未找到或状态前置拒绝 / `2` 用法或校验失败。

**调度计算**（scheduler.py，纯标准库）：`CronExpr.next_after(dt, horizon_days=366)`
本地时区逐分钟走查；`next_fire_of(task, kv)` 对 at 任务在 `_meta.last_fire` 已置
时返回 None（一次性语义）；不可达表达式（如 2 月 30 日）返回 None。

**worker 日志**（dispatcher.py）：`_start_worker` 把 stdout/stderr 落盘
`workspace/logs/task_<id>.run_<restart_count>.log`（追加不截断历史）；
新增 `prune_worker_logs(log_dir, db, keep_days=30)`，main.py 每小时随
heartbeats/gpu_snapshots 一起裁剪（终态超龄删除，非终态保留）。

## 2. Feature B — 多卡支持

**数据模型**（schema.sql + db.py 迁移）：`tasks.gpus INTEGER DEFAULT 1` +
`tasks.gpu_ids TEXT`（钉卡 CSV，可空），沿用 at 列的 ALTER TABLE 迁移模式；
`paused` 状态仅入注释（TEXT 无需 DDL）。

**校验**（新 `validation.py`，CLI/watcher/server 三入口复用）：
`validate_gpu_request(config, gpus, gpu_ids)` 规则——gpus≥1 整数；钉卡长度须等于
gpus 且不重复；`training.multi_gpu ∈ {reserve, ddp}`（默认 reserve）；
**ddp + gpus>1 时拒绝** block_swap>0、torchao 量化（float8/int8/int4）、
`mixed_precision='no'`（后一条为实施期补充，见 D27）。

**准入**（gpu_guard.py）：`Admit` 携带 `gpus` 列表（`.gpu` 属性保留单卡兼容）；
新 `select_gpus(gpus, n, occupied, pinned, empty_machine)`——钉卡验证存在且满足
75% 空闲门槛，自动选卡按 free_mb 取前 n；`judge` 阶梯：空机放行 → 采集失败 →
allow_parallel → max_parallel（worker 数）→ 总卡数容量 `occupied+n ≤ 物理卡数`
→ select。拒绝写 `_meta.gpu_wait_reason` 并置 waiting_gpu；每次判定写
`_meta.gpu_decision`（含 requested/gpus/occupied/provider/快照）。

**launch 分级**（dispatcher.py）：gpus≤1 或 `multi_gpu=reserve` → 单进程
train.py + `CUDA_VISIBLE_DEVICES=<csv>` 绑卡（占 K 卡但不 DDP，向后兼容）；
gpus>1 且 `multi_gpu=ddp` → `accelerate.commands.launch --num_processes=K
--multi_gpu --main_process_port=<_free_port()> [--mixed_precision bf16]`。
ddp 为进程组拉起的强制配套：spawn 用 `CREATE_NEW_PROCESS_GROUP` /
`start_new_session=True`，`_monitor` 杀路径改 `_kill_process_tree()`（Windows
taskkill /T /F、POSIX killpg），不再留 accelerate 孤儿进程。

**train.py rank-0 门控**（DDP 安全）：`_is_rank0()`（`RANK` env，默认 0）——
心跳回调、初始心跳、HookManager 挂载、tensorboard/wandb reporter、最终检查点
保存（前后 `wait_for_everyone()`）、终态 DB 写入、`_dry_run` 主体全部 rank-0
only；非 rank-0 静默跳过。

**占用反查**（db.py）：`running_task_gpus()` 从 running 任务 `_meta.gpu_index`
（兼容旧标量）反查 `{task_id: [gpu,...]}`，供 `gpu status`、`/api/gpu`
occupancy、前端 GPU 页使用。

## 3. API + 前端对齐

server.py 新增：`PATCH/DELETE /api/tasks/<id>`（`_patch_task`/`_delete_task`
白名单 + 状态前置）、`POST /api/tasks/<id>/pause|unpause`、`GET /api/schedules`
（含 next_fire 计算）、`DELETE /api/projects/<id>`（终态检查，叶子优先）、
`DELETE /api/prompts/<id>`、`/api/gpu` 增 `occupancy`、`_create_task` 收
gpus/gpu_ids（validate_gpu_request）。SSE 全量帧附带 schedules。watcher.py
inbox 导入同样过 validate_gpu_request 并落 gpus/gpu_ids。

web/dist/index.html（零构建 SPA）对齐：任务卡/详情页 GPU ×N + 钉卡标注、
定时任务表 卡数 列、GPU 页每卡占用任务 tag（occupancy）、新建表单 GPU 卡数 +
钉卡 输入、`.st-paused` 配色。

## 4. 验证

```
py_compile（10 个模块）                      通过
tmp/test_p5.py    90 项断言全绿（新增）：
  CLI CRUD     project/task/job/schedule/prompt 全命令组 + 退出码 + --json
  schedule     next_after（日常/步进/闰年/不可达）、scan_once 触发/去重/at 一次性
  多卡准入     FakeProvider 空机/钉卡/占用/容量/并行拒绝/waiting_gpu/快照落库
               validate_gpu_request 全部组合拒绝
  dispatcher   桩脚本端到端（gpus=2 派发→done、CUDA_VISIBLE_DEVICES 绑卡、
               _meta.gpu_index 列表、run 日志、logs 命令）、_build_launch 三模式、
               _free_port、_kill_process_tree、prune_worker_logs、running_task_gpus
CLI 冒烟       两轮（任务/调度/prompt/暂停/重试/删除语义）通过
API 冒烟       项目/任务创建（gpus=2）、PATCH、/api/schedules、pause/unpause、
               ddp+混合精度校验拒绝 —— 通过
```

## 5. 与计划的偏差（as-built 记录，详见 md/06 §9）

1. **`task check <id>` dry-run 预检命令未实现**（md/06 §3.5/§5 P5e 交付物）：
   本环境无 torch，无法端到端验证；预检仍可用手动的
   `python UnifiedTrainer/train.py --task-id N --db ... --dry-run-steps 2`。
2. **trainer.py 未改**：计划 §3.5 的 `engine/trainer.py` checkpoint
   `wait_for_everyone()` + seed-per-rank 未动；rank 门控全部集中在 train.py
   顶层保存点与 `_is_rank0()`。seed-per-rank 仅文档化，未实现。
3. **`GET /api/tasks/<id>` 未附 `log_file` 字段**（§2.4）：路径可从
   workspace/logs 推导，价值低，未加。
4. **前端未加 pause/unpause 按钮**（§2.4）：只做了 GPU/定时/占用对齐；
   暂停/恢复走 CLI/API。
5. **dispatcher 占用不并入 DB 侧 running 任务**（§3.2 的 cross-check）：
   `occupied_gpus()` 只统计活 worker；DB 侧占用由 `running_task_gpus()` 覆盖
   展示场景（server/cli），未合并进准入（避免 orchestrator 重启后误判占用）。
6. **测试位置**：计划 §6 写 `tests/`，实际沿用 progress/010 惯例
   `tmp/test_*.py`（gitignored）。
