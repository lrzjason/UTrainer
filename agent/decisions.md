# 技术决定记录（agent/decisions.md）

## D1 — train.py DB 模式增加 `--dry-run-steps N`（P1 冒烟测试）
- 真实模型训练太重，无法在 CI/联调中跑。DB 模式（`--task-id/--db`）下允许
  `--dry-run-steps N`：模拟 N 个 step、每步写 heartbeats、在 output.dir 下
  产出假权重文件并把绝对路径写入 `task_config_kv["_meta.output"]`，退出码 0。
- 也可用 config 键 `training.dry_run_steps` 触发（dispatcher 不改 spawn 参数，
  worker 物化 config 后自行判断）；CLI 参数优先于 config 键。
- 仅 DB 模式可用；`--model/--config` 旧模式传该参数无效（直接忽略，不报错）。

## D2 — 任务产物路径存放在 `task_config_kv["_meta.output"]`
- tasks 表 DDL（doc/Improvement_k3.md §3.2）没有 output 列，P1 不改表结构。
- 约定：key 以 `_` 开头的 kv 为元数据，`materialize_config` 跳过不注入训练
  config。worker（train.py DB 模式）训练完成后写入 `_meta.output`，
  dispatcher 用它解析后继任务的 `$task:<name>.output` 引用。

## D3 — inbox 内容 hash 去重索引入 `workspace/import_index.json`
- 七表 DDL 无 hash 字段；去重状态放 workspace 下的索引文件（JSON 只是导入
  入口，索引文件不属于"永不回写"的业务 JSON）。
- 重复投递直接归档 done/ 并附 `.note.txt` 说明。

## D4 — processing/ → done//failed/ 的归档时机
- project_*.json 可能含多个任务；等**同一 source_file 的所有任务都进入终态**
  （done/failed/cancelled）后才归档：全 done → done/，否则 → failed/。
- 单任务文件（task_*.json）等价于任务结束即归档。

## D5 — argparse 向后兼容处理
- `--model/--config` 由 `required=True` 改为手动校验：DB 模式不需要它们；
  旧模式缺失时 parser.error 退出，行为与原来一致（报错退出码 2）。
- `--list-models/--list-losses` 现在可不带 --model/--config（原来是靠传
  dummy 值绕过的，AGENTS.md 里的旧用法仍然有效）。

## D6 — 心跳判活时间基准
- heartbeats.ts 用 SQLite `datetime('now')`（UTC）。dispatcher 比较时一律按
  UTC 解析，避免本地时区误差。HEARTBEAT_TIMEOUT=120s，轮询 2s。

## D7 — 退出码通信
- worker → dispatcher：0=done、42=suspended（P2 hook 挂起用）、其他=failed；
  dispatcher 内部用 -999 表示心跳超时主动 kill（不落库为退出码，只记 error）。

## D8 — cron 本期不实现
- `next_runnable_tasks()` 直接过滤 `cron IS NULL`；带 cron 的任务导入时置
  `scheduled` 状态，P3 Scheduler 到点后置回 pending。

## D9 — engine/__init__.py 改惰性导出（P2 前置修复）
- 原文件顶层 `from ...trainer import Trainer` 会连带 import torch，导致无
  torch 环境下任何 `UnifiedTrainer.engine.*` 子模块（hook_manager 等）都不可
  导入，dry-run 直接崩。改为 `__getattr__` 惰性导出 Trainer；旧用法
  `from UnifiedTrainer.engine import Trainer` 签名不变。

## D10 — HookManager 放 worker 侧 engine/，orchestrator/hooks.py 只做写入协议
- worker 消费：engine/hook_manager.py（依赖 hot_keys/sampling，需接触
  trainer/CheckpointManager）；orchestrator 侧只提供 enqueue/resume_task
  纯 DB 操作，cli 与未来 FastAPI 共用，保证"前端与 CLI 零差异"。
- hook 只对 status=running 的任务可投递（enqueue 时校验）；suspended 任务
  一律走 resume，杜绝挂起期指令堆积。

## D11 — suspend 现场路径存 `_meta.suspend_checkpoint`，resume 约定 run-N 命名
- suspend hook 保存 full checkpoint 后写 kv 元数据（沿用 D2 的 `_` 前缀约定，
  不注入训练 config）；`cli resume` 读它写回 tasks.resume_from +
  resume_mode=full，restart_count+1，状态置 pending 由 dispatcher 重派。
- wandb 命名约定：第 n 次运行（restart_count=n-1）run name = `<task>-run-<n>`；
  resume 时 prev_run 缺省按同规则回推（首轮未显式存 run name 时为 run-1）。
  run_name / resumed_from_run 同时写 task_config_kv 的 `wandb.*` 键，
  物化 config 时自动注入；真实 wandb.init/finish 调用保持惰性。

## D12 — 热改白名单"默认拒绝"
- hot_keys.validate_key：精确白名单 + `losses[i].weight/use_weighting/enabled`
  正则放行；结构键与一切未知键默认拒绝，错误信息写明"suspend→改配置→resume"
  的替代路径。部分拒绝时：已放行键照常生效并写 task_config_kv，
  hook 标 done 且 result 含 rejected 段；全部被拒才标 failed。

## D13 — dry-run 全程占位，真实采样回调留接口
- sampling.run_sample(generate_fn=None) 时生成 1×1 占位 PNG（合法 PNG 字节，
  查看器可打开）；HookManager._build_generate_fn 本阶段恒返回 None，
  接 trainer 验证管线留待有 torch 环境联调（接口 (prompts, params, out_dir)
  -> [path] 已固定）。suspend/save/restore 真实路径代码已按
  CheckpointManager 现有签名写好，但未在 torch 环境实测。

## D14 — cron 解析纯标准库实现，re-arm 由 dispatcher 负责
- 环境无 croniter 且禁止 pip，五段解析器内置在 orchestrator/scheduler.py
  （* , - / 步进；dow 0/7 均为周日；dom/dow 同受限按 POSIX 取 OR）。
- 周期语义：cron 任务到点置 pending 跑一轮，终态后 dispatcher 自动
  re-arm 置回 scheduled；`at` 任务一次性不 re-arm。
- 防堆积：任务离开 scheduled 期间不再触发；同一分钟用 kv
  `_meta.last_fire` 去重，错过多次触发只保留一个排队实例。
- next_runnable_tasks 不再过滤 cron IS NULL（status=pending 即"已到点"
  的唯一语义），并纳入 waiting_gpu 供补位扫描。

## D15 — GPU 采集三级降级，provider 可注入
- pynvml（惰性 import）→ nvidia-smi 子进程解析 → NullProvider
  （无 GPU 信息：空机放行、并行保守拒绝）。torch.cuda.mem_get_info 降级
  跳过——torch 不存在，且 worker 侧才可能有 torch，orchestrator 侧拿不到。
- GPUGuard(provider=...) 可注入 FakeProvider，无 GPU 环境全链路可测。
- 判定结果（快照+admit+reason）写 kv `_meta.gpu_decision`，拒绝原因写
  `_meta.gpu_wait_reason`；gpu_snapshots 表不改结构。

## D16 — FastAPI 不可用，API 用标准库 http.server + SSE 降级
- 探测：managed Python 无 fastapi/uvicorn/pynvml/torch，禁止 pip 安装。
- orchestrator/server.py 用 ThreadingHTTPServer 实现 md/05 全部 REST
  端点；WebSocket 降级为 SSE（GET /api/events，每 2s 全量推
  tasks+gpu）。路由面与 md/05 保持一致，未来有依赖时可整体替换实现。
- 若环境日后装有 fastapi/uvicorn，应优先重写 server.py 而非并存两实现。

## D17 — dispatcher worker 池，max_parallel 默认 1 保持 P1 串行
- 每 worker 独立监控线程（心跳判活逻辑不变），主循环只负责收割+准入；
  任一 worker 退出 _wake.set() 立即重扫 waiting_gpu 补位。
- Admit(gpu=N) 时 spawn env 注入 CUDA_VISIBLE_DEVICES=N，并写 kv
  `_meta.gpu_index`；worker dry-run 回写 `_meta.cuda_visible_devices`
  供测试断言绑卡生效。
- 默认 max_parallel=1 时非并行任务也会置 waiting_gpu（原因
  "parallel not allowed"），与 P1 串行结果一致、中间状态更可见。

## D18 — tasks 表加 at 列（一次性触发），旧库 ALTER 迁移
- schema.sql 增加 `at TEXT`；DB._init_schema 用 PRAGMA table_info 检测
  旧库并 ALTER TABLE 补齐，不破坏既有数据库。
- 任务 JSON（watcher）与 CLI（task create）、API（POST /api/tasks）均
  支持 cron / at / allow_parallel，入库前校验表达式合法性。

## D19 — P4 前端降级为零构建静态 SPA（无 npm）
- 环境探测：node v24.15.0 存在，但 **npm 不存在**（Git Bash/PATH 中均无），
  无法走 Vite/Vue 构建链 → 降级为 `web/dist/index.html` 单文件 SPA：
  原生 ES module + fetch + EventSource，hash 路由（#/、#/projects/:id、
  #/tasks/:id、#/gpu、#/new），loss 曲线用内联 SVG 绘制，无 CDN 依赖，离线可开。
- 未来若补齐 npm，可按 md/05 原设计重做为 Vue 3 + Vite 工程，API 面不变。

## D20 — server.py 增加静态托管而非独立静态服务器
- 单端口 8100 同时服务 API + 前端：非 /api/ 的 GET → web/dist/，
  未命中回退 index.html（hash 路由）；路径做 abspath+前缀校验防穿越。
- 采样图片经 /samples-file/{task_id}/{name} 提供字节（仅文件名，拒绝路径分隔符）。

## D21 — code review 发现一律不改代码，统一进 known-issues 待修
- 2026-07-18 五路并行 code review（数据层 / orchestrator 核心 / hooks / 调度+GPU /
  API+前端）发现 4 个 high（KI-01 hook 挂载点×梯度累积、KI-02 create_project
  lastrowid、KI-03 心跳判活无启动宽限、KI-04 set_config 不产生 patch_config hook）、
  12 个 med、4 组 low。
- 本轮**不改任何代码**：全部记入 `progress/007-code-review.md` §7 统一 KI 清单
  （编号 KI-01…KI-20，含严重度、文件:行号、修复方向），后续按严重度排期修复，
  避免审查轮与修复轮纠缠、破坏"5 份结论可对照"的基线。

## D22 — 文档批量回写 as-built（007 review 文档同步）
- 把 P1–P4 积累的 as-built 差异一次性回写设计文档，文档以代码为准：
  - `doc/Improvement_k3.md`：§3.2 补 at 列+5 索引+迁移说明；§3.3 kv 纪律偏差；
    §5.2 cron/at/last_fire；§5.3 worker 池（--max-parallel 默认 1）；§5.4 判定顺序、
    kv/gpu_snapshots 分工、waiting_gpu 全 pending；§5.5 标准库+SSE、--api；
    §6.1 maybe_run 签名；§6.2 sample_from_weights/提示词三级优先级/双格式；
    §6.4 拒绝语义+apply_live；§7 CLI 按 ORCHESTRATOR.md 重写；§8 零构建 SPA；
    §10 "五类"→"六类"。
  - `md/01`（接口全量+8 状态+`_` 前缀约定）、`md/02`（worker 池+cmd action+
    --dry-run-steps+归档未实现）、`md/03`（六类+resumed_from_run 存 name）、
    `md/04`（并行默认 1 作废 2）、`md/05`（正文全面 as-built+前端降级清单）。
- `md/ORCHESTRATOR.md` 与 `skills/` 经审查与代码一致，不改。

## D23 — KI-01~04 high 级修复（2026-07-19）
- **KI-01**：trainer.py 的 hook 挂载点移入 `if step_advanced:` 块内——梯度
  累积时只有 optimizer 真正 step 后才执行 `maybe_run`，micro-batch 不再
  触发 hook（重复采样/重复写 kv/suspend 时机错乱消除）。旧 --config 模式
  不注入 `hook_manager` 属性，getattr 返回 None 跳过，行为不变；train.py
  dry-run 循环无梯度累积，保持每步调 maybe_run（语义正确，不改）。
- **KI-02**：create_project 改用 `cur.rowcount == 1` 判断是否真正插入；
  INSERT OR IGNORE 被忽略时 SELECT 取已有 id（lastrowid 陈旧值陷阱消除）。
- **KI-03**：dispatcher 新增 `STARTUP_TIMEOUT = 300.0` 常量与纯函数
  `_hb_stale(hb_age, uptime, ...)`：从未有心跳的 worker 启动超过 300s
  判僵死 kill + failed（-999）；已有心跳维持 120s 间隔判活。
  **取值理由**：300s = 心跳 120s 的 2.5 倍。dry-run worker 首条心跳即时，
  但真实模型（Flux2/Qwen 等 4B+ 级）从磁盘加载 + 量化 + block swap 到
  首个 step 完成常需 2~5 分钟；取 300s 覆盖常规加载又不至于让卡死进程
  占卡过久。超大模型/冷盘场景可调大常量或经 _monitor 参数注入。
- **KI-04**：hooks.py 新增公共函数 `set_config_and_notify(db, task_id,
  key, value)`——统一写 task_config_kv；任务 running 时追加投
  patch_config hook（payload={key: value}，走 enqueue 统一校验 + worker
  热改白名单）；`_` 前缀元数据键不投 hook。watcher cmd set_config、
  新增 CLI `config set <task_id> <key> <value>`（值先 json.loads 后回退
  字符串）、server PATCH /api/tasks/{id}/config 三入口全部复用该函数。
- 验证：smoke_test / test_p2_hooks / test_p3 回归全过；新增
  tmp/test_ki_fixes.py 25 项断言全过（详见 progress/008-ki-high-fixes.md）。

## D24 — KI-05~KI-16 med 级修复（2026-07-19）
- **KI-05（时区统一）**：scheduler `_meta.last_fire` 去重键统一为本地时区
  字符串（`_fire_key() = now.astimezone().strftime(...)`）。cron 字段匹配
  本来就按本地时间，去重键必须同口径；混用 UTC 会在 :30/:45 偏移时区
  失效。采 owner 指示的"统一为本地"而非报告建议的"统一为 UTC"。
- **KI-09（cron 熔断 + at 去重 + 归档语义）**：dispatcher 新增
  `MAX_CONSECUTIVE_FAILURES = 3`，cron 任务终态时 kv
  `_meta.consecutive_failures` 计数（exit 0 清零），达到上限不再 re-arm，
  任务保持 failed 终态并把原因记 kv `_meta.rearm_blocked`；源文件随之
  正常归档 failed/。re-arm 先于归档的顺序**有意保留**——周期 cron 任务
  源文件长期驻留 processing/ 属预期（周期任务永不"完成"），仅在
  _finalize/_archive_source 加注释明确语义，不改行为。at 任务在
  scheduler._due 补 last_fire 去重（已有 last_fire 的 at 任务不再触发）。
- **KI-06（inbox 滞留 + 坏文件反复处理）**：watcher 新增
  `UNRECOGNIZED_LIMIT = 3`——文件名无法识别时连续 3 轮后移 failed/
  并写 `.note.txt`（给写入中文件留宽限）；导入失败也记内容 hash，
  同一坏文件再投递按 duplicate 直接归档 done/，不再每轮重试。
- **KI-10（at 校验下沉 + PATCH 响应）**：scheduler 新增
  `validate_schedule(cron, at)`，CLI / server POST /api/tasks / watcher
  文件导入三入口统一复用，非法 cron/at 一律 400/进 failed 不落库。
  PATCH /api/tasks/{id}/config 保持"kv 可写任意键、热生效由 worker
  白名单裁决"语义，响应改为 `{"config", "hook_results"}` 附每键 hook
  投递结果（hook_id 或 rejected 原因），调用方可感知投递状态。
- **KI-11（路径注入）**：db 层新增 `validate_name()`，create_project /
  create_task 统一拒绝空名、`.`/`..`、路径分隔符——所有创建入口
  （CLI/API/watcher）单点卡住；server samples 路径拼接（`_samples_dir`）
  对库内名字再校验一道，防御历史脏数据。
- **KI-07/KI-08（max_parallel 双源 + falsy 陷阱）**：main.py 删去
  `max(2, ...)` 死参数直接透传，并行上限单源在 dispatcher（内部
  max(1, ...)）；gpu_guard.judge 改显式 `is None` 判断，
  `max_parallel=0`（禁止并行）不再被 falsy 绕过。
- **KI-14（祖先引用）**：`_resolve_successors` 引用名≠直接前驱时按名字
  在同项目内解析更早祖先（须已 done 且有 `_meta.output`），支持
  "c 依赖 b 但 resume_from=$task:a.output" 的多级链间隙场景；
  set_task_status 前重读当前状态，缩小读-改-写窗口。
- **KI-15（nvidia-smi 容错）**：gpu_guard 新增 `parse_nvidia_smi_csv()`，
  util 字段 `[N/A]` 记 None、显存字段不可解析整行跳过（记 warning）；
  cli gpu status 复用同一解析。judge 区分"采集失败"与"无 GPU 信息"：
  采集异常时并行拒绝原因明确写 `GPU metrics collection failed: ...`，
  不再写成误导性的 VRAM 不足；空机放行不受影响。
- **KI-16（归档摘要）**：dispatcher._archive_source——done 归档往文件
  JSON 追加 `_result` 摘要（各任务 id/状态/起止时间/归档时间）；
  failed 归档旁写 `<name>.error.log`（含各任务 DB error 字段）。
- **KI-18（顺手）**：删除 watcher.finish_processing 死代码及
  `_processing` 簿记；cmd hook 统一走 hooks.enqueue 校验。
- **resumed_from_run（环境限制确认）**：wandb 惰性 import 下 orchestrator
  拿不到真实 run id，只能存确定性 run name——不改行为，hooks.py 加
  as-built 注释（md/03 已有对应条目）。
- **范围说明**：KI-12（sample_from_weights 按命名约定备份）、KI-13
  （set_dotted 静默跳过）两项 med 触及 UnifiedTrainer 训练引擎，本轮
  按 owner 清单未动，留待下轮。
- 验证：smoke_test / test_p2_hooks / test_p3（66 项）/ test_ki_fixes
  回归全绿；新增 tmp/test_ki_med.py 57 项断言全过（详见
  progress/009-ki-med-fixes.md）。

## D25（2026-07-19）KI 收尾清零：KI-12/13/17/19/20 + server 响应双发

- **KI-12（adapter 备份键集）**：sample_from_weights 备份改由
  `_backup_adapter_params()` 驱动——优先 PEFT
  `get_peft_model_state_dict` 查实际 adapter 键（与 load_lora 键空间
  同源）；PEFT 不可用时退化为宽松名称匹配（lora/adapter，大小写不敏感）
  并记 warning。原则：备份范围只多不少，换回必须完整。
- **KI-13（热改如实上报）**：set_dotted 返回 bool；apply_live 返回
  (applied, skipped)，越界/未命中从 applied 剔除并写进 hook.result；
  删 `_REJECT_PREFIXES` 死代码。kv 落库语义不变（先落 DB 再热生效）。
- **KI-17（db 四项）**：连接注册表 + close_all；prune_heartbeats(7 天) /
  prune_gpu_snapshots(每 gpu 1 万行) 由 main.py 每小时调用；删
  connect(timeout=30)，busy_timeout=30000 单源（原双源矛盾且 timeout
  为死代码）；enqueue_hook 参数 type→type_。
- **KI-19（hook 细节）**：JSONL 非 dict 行按原文本；worker 启动
  requeue_stale_acked_hooks 回收 acked→queued；提示词 DB 通道按
  project_id 过滤（None 语义保留给管理查询）；dry-run save result 补
  optimizer_state 键与真实路径对齐。
- **KI-20（server）**：_static unquote；_json_body 1MB 上限→400；SSE
  流内异常静默关闭不再追加 500；顺手修复 `_route` 返回值语义
  （_send/_send_file/_sse 返回 True 标记已处理）——此前每个成功响应后
  都追加一个 404，污染 keep-alive 连接（007 清单外新发现，按 KI-18
  先例顺手修复）。
- **retention 运维语义**：裁剪失败仅告警不中断；默认参数（7 天 / 1 万行）
  写死在 main.py，后续如需可调再提 CLI flag。
- 验证：smoke / test_p2_hooks / test_p3（66）/ test_ki_fixes /
  test_ki_med（57）回归全绿；新增 tmp/test_ki_low.py 28 项全过。
  007 known-issues 至此全部清零，详见 progress/010-ki-final.md。

## D26 —（计划）CLI 全量 CRUD + 多卡支持：设计定稿于 md/06-cli-scheduler-multigpu.md

两大需求的实现计划已定稿（未动工），关键设计决定：
- **不建 schedules 表**：调度仍是 tasks.cron/at 列，`schedule` 命令组只是
  UX 层（list/set/clear/preview），复用 validate_schedule，单一事实源不破。
- **新增 paused 状态**（TEXT，无 DDL）：仅从 scheduled/pending/waiting_gpu
  进入；dispatcher 天然跳过；resume 只认 suspended。
- **多卡数据模型**：tasks 增 `gpus INTEGER DEFAULT 1` + `gpu_ids TEXT`
  （可空，钉卡 CSV），沿用 at 列的 ALTER TABLE 迁移模式。
- **`_meta.gpu_index` 由标量改列表**：所有消费方（server GPU 视图、前端）
  需同步改；P5c 期间 grep 全部读者。
- **launch 分级**：config `training.multi_gpu: reserve|ddp`，默认 reserve
  （占 K 卡绑 CUDA_VISIBLE_DEVICES 但单进程跑，orchestrator-only 安全）；
  ddp 才走 `accelerate launch --num_processes=K`，且 block_swap/torchao
  组合在创建期直接拒绝。
- **max_parallel 语义不变**（数任务不数卡），另加总卡数容量检查。
- 进程组杀树（CREATE_NEW_PROCESS_GROUP / start_new_session）为 ddp 模式
  的强制项，防 accelerate 子进程孤儿。
- 分工阶段：P5a CLI CRUD → P5c 多卡准入 → P5b schedule 组 → P5d reserve →
  P5e ddp → P5f 前端/文档。

## D27 —（实施）P5a–P5f 全部落地：实施期决策与偏差（2026-08-03）

按 D26 定稿与 `md/06` 计划完成 P5a–P5f。实施期新决定与偏差：
- **rank 门控实现选型**：不用 `accelerator.is_main_process`——门控点多在
  module 级、Accelerator 构造之前（回调、reporter、最终保存），统一用
  `_is_rank0()`（`int(os.environ.get("RANK", "0")) == 0`，解析失败回退 True）；
  检查点保存前后加 `wait_for_everyone()`，非 rank-0 的 final_path=None 静默跳过。
- **`engine/trainer.py` 不动**：checkpoint `wait_for_everyone()`、seed-per-rank
  原计划落在 trainer.py，实施改为全部集中在 `train.py` 顶层保存点（同一
  Accelerator 实例，语义等价且改动面最小）；seed-per-rank 仅文档化未实现
  （无 torch 环境无法验证，避免盲改）。
- **补充拒绝组合**：ddp + gpus>1 且 `mixed_precision='no'` 创建期直接拒绝
  （DDP 无梯度同步精度会异常，保持保守；D26 只列了 block_swap/torchao）。
- **`task check` dry-run 预检命令未做**：md/06 §3.5 交付物；本环境无 torch，
  无法端到端验证，预检仍可手动 `train.py --dry-run-steps 2`。
- **worker 日志 retention 落地**：新增 `dispatcher.prune_worker_logs(log_dir,
  db, keep_days=30)`（终态超龄删除、非终态保留、任务已删按超龄），main.py
  每小时随 heartbeats/gpu_snapshots 一并裁剪（md/06 §2.3/§4 交付物）。
- **占用展示与准入分离**：`occupied_gpus()` 只统计活 worker；DB 侧
  `running_task_gpus()`（兼容旧标量 `_meta.gpu_index`）供 server /api/gpu 与
  cli gpu status 展示，不合并进准入（orchestrator 重启后旧 running 记录会
  误判占用，等心跳超时回收更稳）。
- **测试惯例**：计划 §6 写 `tests/`，实际沿用 progress/010 惯例
  `tmp/test_*.py`（gitignored）；`tmp/test_p5.py` 90 项断言全绿，CLI/API
  冒烟通过，详见 `progress/011-p5-cli-multigpu.md`。

