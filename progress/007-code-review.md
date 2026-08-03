# 007 — ScheduledTrainer 全量 Code Review 汇总

> 日期：2026-07-18。范围：`orchestrator/`（schema.sql / db.py / watcher / dispatcher /
> scheduler / gpu_guard / hooks / server / cli / main）+ `UnifiedTrainer/train.py /
> engine/trainer.py / engine/hook_manager.py / engine/sampling.py / engine/hot_keys.py`
> + `web/dist/index.html` + 全部设计文档（doc/Improvement_k3.md、md/00–05、
> ORCHESTRATOR.md、SKILL.md）。
> 方法：5 个并行 explore subagent 分组件对照"代码 as-built vs 设计文档 as-designed"，
> 本报告为五份结论的合并与定级。
> 处置原则：**本轮不改任何代码**（D21）。代码 bug 与功能缺口统一记入下方
> known-issues 表待修；文档落后于代码的条目本轮已全部回写（D22）。

---

## 1. 总体结论

核心链路（inbox 文件流转 → DB 入库 → scheduler/dispatcher 派发 → worker DB 模式
训练 → hook 消费 → 心跳判活 → 归档；API/前端只读 DB + 写 hooks）与文档**高度一致**，
协议层（退出码 0/42、心跳 120s、`_meta.*` kv 约定、`$task:` 引用、六类 hook 状态机、
GPU Guard 判定顺序、SSE 降级）实现忠实。主要缺口分三类：

1. **文档滞后**（量大、风险低）：P3/P4 的大量 as-built 决定（D14–D20）未回写进
   doc/Improvement_k3.md 与 md/01–05 正文。本轮已全部同步，见 §8。
2. **功能缺口**（量中）：CLI `config set` / `project archive` / `submit --cron` flags、
   归档结果摘要与 error.log、前端 cron 预览/链式向导/DAG 等未实现。
3. **代码 bug**（量小但有 4 个 high）：训练正确性相关的 hook 挂载点位置、
   create_project lastrowid 陷阱、心跳判活无启动宽限、set_config 不产生
   patch_config hook。全部记入 §7，本轮不改代码。

---

## 2. 数据层（schema.sql / db.py vs doc §3.2、md/01）

**一致**：七表字段（除 at 列）；WAL + busy_timeout；连接按线程持有；
materialize_config 为唯一配置出口。

**文档已同步**：tasks.at 列、5 个索引（schema.sql:96-100）、旧库 ALTER 迁移
（db.py:63-67）；md/01 接口清单补全（list_projects / update_project / get_task /
list_tasks / successors_of / list_hooks / add_prompt / list_prompts / prompts_by_tag /
heartbeats_since / latest_gpu_snapshots / list_gpu_snapshots / close）、
next_runnable_tasks 新语义（cron/at 到点即 pending，不再过滤）、get_config_kv raw
参数、`_` 前缀 kv 元数据约定、8 状态机全口径、set_config_kv 不产生 hook（调用方责任）。

**代码偏离 → known-issues**：KI-02、KI-04、KI-17。

## 3. Orchestrator 核心（watcher / dispatcher / main / cli / train.py vs md/02、doc §4/§5）

**一致**：inbox→processing→done/failed 流转、内容 hash 去重、三类文件识别、
项目文件格式、Popen 启动、退出码 0/42、worker 退出补位、心跳 120s、`$task` 解析、
train.py DB 分支（物化 config + 心跳 + 仅 DB 模式挂 HookManager）、旧模式零影响。

**文档已同步**：worker 池化回写（--max-parallel 默认 1 保持串行，D17）；
--dry-run-steps 补记；cmd action 集 = cancel/hook/set_config；waiting_gpu 适用于
所有 pending 任务；CLI flag 实际为 `--model`；归档结果摘要/error.log 标注未实现（KI-16）。

**代码偏离 → known-issues**：KI-03、KI-06、KI-07、KI-14、KI-16、KI-18。

## 4. Hooks（hook_manager / sampling / hot_keys / orchestrator hooks vs md/03、doc §6）

**一致**：六类 hook 协议与状态机、payload、采样解耦（恢复原始 requires_grad，
优于文档描述）、输出路径、热改白名单、suspend→resume 全链路、惰性 import 彻底、
旧模式不加载。

**文档已同步**：maybe_run 签名 = `maybe_run(step, trainer=, config=)`；"五类"→"六类"
（doc §10 与 md/03）；sample --weights 已拆为独立 sample_from_weights；提示词三级
优先级（payload prompts_path > DB tag > 占位）与 JSONL/txt 双格式；patch_config
部分/全部拒绝语义与 apply_live 热生效点；resumed_from_run 存 run name 而非 run id
的 as-built 备注。

**代码偏离 → known-issues**：KI-01、KI-12、KI-13、KI-19。

## 5. 调度与 GPU（scheduler / gpu_guard vs md/04、doc §5.2/§5.4/§7）

**一致**：cron 五段语义、30s tick 可配（--tick）、at 一次性、last_fire 防堆积、
judge 判定顺序、采集链三级降级、决策记录、补位、re-arm、CUDA_VISIBLE_DEVICES 绑卡、
CLI 入库前校验。

**文档已同步**：§7 定时语法改为 task create --cron/--at；--tick 补记；
gpu_decision / wait_reason 写 kv 而非 gpu_snapshots 的 as-built 分工（空快照不入库）；
并行上限默认 1（有意保守，D17，md/04 早期草稿的 2 作废）。

**代码偏离 → known-issues**：KI-05、KI-08、KI-09、KI-15。

## 6. API / 前端 / 文档（server.py / index.html vs md/05、doc §5.5/§8、ORCHESTRATOR.md、SKILL.md）

**一致**：全部 REST 端点落地；前端 17 处调用、13 个 URL 模式与后端零失配；
五页面骨架；ORCHESTRATOR.md 与 cli.py 逐条吻合（该手册本身就是 as-built）；
SKILL.md 合规。

**文档已同步**：md/05 正文改为 as-built——标准库 ThreadingHTTPServer、SSE 替代
WebSocket、零构建 SPA（npm 缺失，D19）、内联 SVG 曲线替代 ECharts、SSE 帧驱动
heartbeats 轮询、/api/gpu 不含 wait_reason（前端逐任务补查）、端点内部只调
db+hooks（不经 dispatcher）；前端交互降级清单（无 cron 人性化预览、无链式创建
向导、无 DAG/下次触发列/任务详情抽屉/日志 tail/wandb 链接、提示词库仅列表+新增、
项目卡片无最近活动）；doc §5.5 --api-only 改为 --api。

**代码偏离 → known-issues**：KI-10、KI-11、KI-20。

---

## 7. Known-Issues 统一清单（本轮不改代码，D21）

| 编号 | 严重度 | 组件 | 描述 | 位置 | 建议修复方向 |
|------|--------|------|------|------|--------------|
| KI-01 | ~~high~~ ✅ 已修复（008） | hooks | HookManager 挂载点未受 `if step_advanced` 保护，梯度累积时每个 micro-batch 都触发 hook，可能影响训练正确性（重复采样/重复写 kv/ suspend 时机错乱） | UnifiedTrainer/engine/trainer.py:845-851 | **已修复**：挂载点移入 `if step_advanced:` 块内，optimizer 真正 step 后才执行 maybe_run；旧 --config 模式 getattr 跳过不变。验证：test_ki_fixes.py KI-01 静态断言 + P2/P3 回归 |
| KI-02 | ~~high~~ ✅ 已修复（008） | 数据层 | create_project 用 INSERT OR IGNORE 后直接读 lastrowid：名字已存在时返回的是无关行的 id（或 0），调用方拿到错误 project_id | orchestrator/db.py:86-89 | **已修复**：改用 `cur.rowcount == 1` 判断真正插入，被忽略时 SELECT id WHERE name=?。验证：test_ki_fixes.py 重复 create_project 返回相同 id |
| KI-03 | ~~high~~ ✅ 已修复（008） | orchestrator | 心跳判活只处理"有心跳但超时"的 worker；从未上报心跳的 worker（模型加载期挂起）永不判死，无限等待 | orchestrator/dispatcher.py（worker 监控线程） | **已修复**：新增 STARTUP_TIMEOUT=300s 常量 + 纯函数 `_hb_stale`；无心跳且启动超宽限判死 kill+failed，有心跳维持 120s 判活（D23）。验证：判据单测 + _monitor 假 proc 端到端 rc=-999 |
| KI-04 | ~~high~~ ✅ 已修复（008） | 数据层/orchestrator | set_config_kv 与 cmd set_config 均不产生 patch_config hook、无白名单校验，违反 doc §3.3 读写纪律（纪律靠调用方自觉）；running 任务的 kv 直写在 worker 重物化前不可见 | orchestrator/db.py、watcher cmd 处理 | **已修复**：hooks.py 新增 `set_config_and_notify()` 统一入口（写 kv + running 时投 patch_config hook，`_` 前缀键不投），watcher cmd / 新增 CLI `config set` / server PATCH config 三入口复用（D23）。验证：test_ki_fixes.py cmd 文件与 cli 两路径 hook 消费 done、结构键被白名单拒 failed |
| KI-05 | ~~med~~ ✅ 已修复（009） | 调度 | `_meta.last_fire` 比较混用 UTC 与本地时间：时区偏移含 :30/:45 的地区去重失效或漏触发 | orchestrator/scheduler.py:148,170,174,179 | **已修复**：`_fire_key()` 统一用 `now.astimezone()` 本地时区字符串存取/比较（D24）。验证：test_ki_med.py 时区断言 + 同分钟去重 + 跨分钟再触发 |
| KI-06 | ~~med~~ ✅ 已修复（009） | orchestrator | 无法识别/导入失败的文件永久滞留 inbox；失败导入不记 hash，坏文件每轮被反复处理 | orchestrator/watcher.py | **已修复**：未识别文件连续 3 轮（UNRECOGNIZED_LIMIT）后移 failed/ 并写 .note.txt；失败导入记 hash，同一坏文件再投递按 duplicate 直接归档（D24） |
| KI-07 | ~~med~~ ✅ 已修复（009） | orchestrator | `main.py:50` 的 `max(2, max_parallel)` 为误导性死参数；并行上限在 guard 与 dispatcher 双源维护 | orchestrator/main.py:50、gpu_guard.py | **已修复**：main.py 直接透传 args.max_parallel，上限单源在 dispatcher（D24） |
| KI-08 | ~~med~~ ✅ 已修复（009） | 调度 | judge 中 `max_parallel` 用真值判断，`--max-parallel 0`（意图禁止并行）被当 falsy 绕过 | orchestrator/gpu_guard.py | **已修复**：显式 `is None` 判断；judge(max_parallel=0) → Wait "max parallel reached (1/0)"（D24） |
| KI-09 | ~~med~~ ✅ 已修复（009） | 调度 | 失败 cron 任务无限 re-arm 无熔断；re-arm 在归档前执行使 cron 源文件永不归档（dispatcher.py:156-162 顺序）；at 任务无 last_fire 去重保护 | orchestrator/dispatcher.py:156-162、scheduler.py | **已修复**：kv `_meta.consecutive_failures` 计数（成功清零），≥3（MAX_CONSECUTIVE_FAILURES）不再 re-arm 并记 `_meta.rearm_blocked`；cron 源文件驻留 processing/ 语义明确为有意并加注释（不改归档顺序）；at 任务补 last_fire 去重（D24） |
| KI-10 | ~~med~~ ✅ 已修复（009） | API | `at` 字段仅 CLI 入库前校验，API(POST /api/tasks) 与 watcher 入口不校验，非法值直接落库 | orchestrator/server.py、watcher.py | **已修复**：新增 `scheduler.validate_schedule()`，CLI/服务器/watcher 三入口统一复用；PATCH config 保持 kv 任意写语义，响应改 `{"config","hook_results"}` 附每键 hook 投递结果（D24） |
| KI-11 | ~~med~~ ✅ 已修复（009） | API | samples 目录用未校验的 project/task 名拼接路径，存在 `..` 目录穿越注入风险 | orchestrator/server.py:295-298,340-342 | **已修复**：db 层 `validate_name()`（拒绝空名/`.`/`..`/路径分隔符）统一卡住 create_project/create_task 全部入口；server samples 路径拼接再校验一道（D24） |
| KI-12 | ~~med~~ ✅ 已修复（010） | hooks | sample_from_weights 只备份名字含 "lora_" 的参数，依赖命名约定；非 LoRA 命名权重换入后无法完整换回 | UnifiedTrainer/engine/hook_manager.py | **已修复**：新增 `_backup_adapter_params()`——优先 PEFT `get_peft_model_state_dict` 查询实际 adapter 键（与 load_lora 键空间一致），拿不到时退化为宽松名称匹配（lora/adapter，大小写不敏感）并记 warning，保证换回完整。验证：test_ki_low.py KI-12 静态 + fallback 键集断言 |
| KI-13 | ~~med~~ ✅ 已修复（010） | hooks | set_dotted 越界静默跳过，但 hook result 仍报 applied，调用方无法感知部分失败 | UnifiedTrainer/engine/hot_keys.py | **已修复**：set_dotted 返回 bool；apply_live 返回 (applied, skipped)，越界/未命中从 applied 剔除并写进 result.skipped；顺手删 `_REJECT_PREFIXES` 死代码与 hook_manager 未使用的 import torch。验证：test_ki_low.py KI-13（含 patch_config 端到端 result 断言） |
| KI-14 | ~~med~~ ✅ 已修复（009） | orchestrator | `_resolve_successors` 只认直接前驱，多级链间隙场景后继不放行；set_task_status 的"保状态"写法存在读-改-写竞态 | orchestrator/dispatcher.py、db.py | **已修复**：引用名≠直接前驱时按名字在同项目内解析更早祖先（须已 done 且有 _meta.output）；set_task_status 前重读当前状态（D24） |
| KI-15 | ~~med~~ ✅ 已修复（009） | GPU | nvidia-smi 输出 `[N/A]` 字段解析崩溃且拒绝原因误导；`cli gpu status` 不复用 provider 降级链 | orchestrator/gpu_guard.py、cli.py | **已修复**：`parse_nvidia_smi_csv()` 容错解析（util N/A→None，显存不可解析整行跳过），cli 与 provider 共用；采集失败时拒绝原因明确写 "GPU metrics collection failed"（D24） |
| KI-16 | ~~med~~ ✅ 已修复（009） | orchestrator | 归档不写结果摘要/error.log（md/02 原始要求）；失败排查只能看 DB error 字段 | orchestrator/watcher.py | **已修复**：done 归档往文件 JSON 追加 `_result` 摘要；failed 归档旁写 `<name>.error.log`（含各任务 DB error 字段）（D24） |
| KI-17 | ~~low~~ ✅ 已修复（010） | 数据层 | close() 只关当前线程连接；heartbeats 表无 retention 无限增长；connect(timeout=30) 与 PRAGMA busy_timeout=5000 矛盾（30s 为死代码）；enqueue_hook 参数名 `type` 遮蔽内置 | orchestrator/db.py | **已修复**：连接注册表 + `close_all()`；`prune_heartbeats(keep_days=7)` / `prune_gpu_snapshots(keep_rows=10000)` 并由 main.py 每小时调用；删 connect timeout，busy_timeout=30000 单源；参数改名 `type_`。验证：test_ki_low.py KI-17 |
| KI-18 | ~~low~~ ✅ 已修复（009） | orchestrator | Watcher.finish_processing 为死代码且其中 error.log 路径错误；cmd hook 绕过 hooks.enqueue 校验（可对非 running 任务投递） | orchestrator/watcher.py:232-246 | **已修复**：finish_processing 及 `_processing` 簿记已删除；cmd hook 统一走 `hooks.enqueue`（非 running/未知类型被拒，cmd 文件进 failed/）（D24） |
| KI-19 | ~~low~~ ✅ 已修复（010） | hooks | `_sample_prompts` 对 JSON 行直接 `.get`（非 dict 行崩溃）；hot_keys.py:60 `_REJECT_PREFIXES` 死代码；acked 状态无回收（worker 崩溃 hooks 卡 acked）；dry-run save 的 result 字段与真实路径不一致；prompts_by_tag(None, tag) 跨项目取提示词 | engine/hook_manager.py:173、hot_keys.py:60、db.py | **已修复**：JSON 行 isinstance(dict) 校验，非 dict 按原文本；死代码删除（随 KI-13）；`db.requeue_stale_acked_hooks()` + HookManager 启动时回收 acked→queued；dry-run save with_optimizer 同样返回 optimizer_state 键；sample 提示词按本任务 project_id 过滤。验证：test_ki_low.py KI-19 |
| KI-20 | ~~low~~ ✅ 已修复（010） | API | `_static` 不做 URL 解码（%20 等文件名 404）；PATCH config 无 API 侧白名单（与 KI-04 同根）；SSE 异常路径对已开始流的 socket 再发 500 响应头 | orchestrator/server.py | **已修复**：`_static` unquote 后再校验拼接；`_json_body` 1MB 上限（超限 400）；SSE 流内异常记录后静默关闭不再发 500；顺手修复 `_route` 返回值语义（_send/_send_file/_sse 返回 True 标记已处理）——此前每个成功响应后都追加一个 404，污染 keep-alive 连接。验证：test_ki_low.py KI-20（含同连接双请求断言） |

**high 项清单**：KI-01（hook 挂载点×梯度累积）、KI-02（create_project lastrowid）、
KI-03（心跳判活无启动宽限）、KI-04（set_config 不产生 patch_config hook）。
**四项 high 已于 2026-07-19 全部修复并验证**，见 `progress/008-ki-high-fixes.md`
与 `agent/decisions.md` D23。

**med 项清单**：KI-05~KI-16。其中 KI-05~KI-11、KI-14、KI-15、KI-16 已于
2026-07-19 修复并验证（KI-18 两项 low 顺手一并修复），见
`progress/009-ki-med-fixes.md` 与 `agent/decisions.md` D24；
KI-12、KI-13 与 low 项 KI-17、KI-19、KI-20 已于 2026-07-19 全部修复并验证
（另顺手修复 server 响应双发 bug），见 `progress/010-ki-final.md` 与
`agent/decisions.md` D25。**known-issues 至此全部清零。**

---

## 8. 文档同步清单（本轮已改，D22）

| 文件 | 关键改动 |
|------|----------|
| `doc/Improvement_k3.md` | §3.2 tasks DDL 加 at 列+5 索引+迁移说明；§3.3 读写纪律补 as-built 备注；§5.3 补 worker 池/--max-parallel（默认 1，D17）；§5.4 决策记录 kv+gpu_snapshots 分工、waiting_gpu 适用全部 pending、上限默认 1；§5.5 FastAPI→标准库+SSE as-built、--api-only→--api、端点只调 db+hooks；§6.1 maybe_run 签名；§6.2 sample_from_weights 独立类型+提示词三级优先级+JSONL/txt 双格式；§6.4 patch_config 拒绝语义+apply_live；§7 CLI 清单按 ORCHESTRATOR.md 重写（config set / project archive / submit --cron 标未实现）；§8 前端降级为零构建 SPA（D19）；§10 路线表"五类"→"六类"、P3/P4 技术栈注 as-built；架构图注 REST+SSE |
| `md/01-database.md` | 接口清单补全 12 个缺失方法；next_runnable_tasks 新签名与语义；get_config_kv raw；`_` 前缀 kv 约定；8 状态完整列表；set_config_kv 不产生 hook（KI-04） |
| `md/02-orchestrator-core.md` | dispatcher 改 worker 池描述（默认 1 保持串行，D17）；补 --dry-run-steps；cmd action 集 cancel/hook/set_config；归档 error.log/结果摘要标注未实现（KI-16） |
| `md/03-hooks.md` | "五类"→"六类"；resumed_from_run 存 run name 而非 id 的 as-built 备注（并入头部 wandb 条目） |
| `md/04-scheduler-gpu.md` | "决定"中并行上限默认 2 作废，改默认 1（D17） |
| `md/05-api-frontend.md` | 正文全面改 as-built：ThreadingHTTPServer、SSE、零构建 SPA、内联 SVG 曲线、SSE 帧驱动 heartbeats 轮询、/api/gpu 不含 wait_reason、端点只调 db+hooks、前端交互降级清单 |
| `agent/decisions.md` | 追加 D21（审查发现不改代码、统一 known-issues 待修）、D22（文档批量回写 as-built） |

未改：`md/ORCHESTRATOR.md`（本身即 as-built，与代码逐条吻合）、`md/00-architecture.md`、
`skills/`（审查结论为合规）。
