# 009 — KI-05~KI-16 med 级修复总结

> 日期：2026-07-19。范围：`progress/007-code-review.md` §7 的 med 级
> known-issue（按 owner 清单逐项），并顺手修复 KI-18（low，与 KI-06/16
> 同区域）。决策记录：D24（agent/decisions.md）。

---

## 1. 改动文件清单

| 文件 | 改动 |
|------|------|
| `orchestrator/scheduler.py` | KI-05：`_fire_key()` 统一本地时区；KI-09c：at 任务 last_fire 去重；KI-10：新增 `validate_schedule()` |
| `orchestrator/dispatcher.py` | KI-09a/b：`MAX_CONSECUTIVE_FAILURES=3` 熔断 + `_meta.consecutive_failures`/`_meta.rearm_blocked` kv + cron 源文件驻留注释；KI-14：祖先引用解析 + 状态重读；KI-16：`_write_result_summary` / `_write_error_log` |
| `orchestrator/gpu_guard.py` | KI-08：`is None` 显式判断；KI-15：`parse_nvidia_smi_csv()` 容错 + 采集失败 reason 明确 + best_gpu/VRAM 比较 None 安全 |
| `orchestrator/main.py` | KI-07：删 `max(2, ...)` 死参数，直接透传 |
| `orchestrator/db.py` | KI-11：`validate_name()`，create_project/create_task 统一校验 |
| `orchestrator/watcher.py` | KI-06：`UNRECOGNIZED_LIMIT=3` 未识别熔断 + 失败导入记 hash；KI-10：导入前 validate_schedule；KI-18：cmd hook 走 `hooks.enqueue`、删 finish_processing 死代码与 `_processing` 簿记 |
| `orchestrator/server.py` | KI-10：POST /api/tasks 校验 at、PATCH config 响应 `{"config","hook_results"}`；KI-11：`_samples_dir()` 名字安全校验 |
| `orchestrator/cli.py` | KI-15：gpu status 复用 `parse_nvidia_smi_csv` |
| `orchestrator/hooks.py` | resumed_from_run 存 name 的 as-built 注释（环境限制，不改行为） |
| `tmp/test_ki_med.py` | 新增：57 项断言（见 §3） |
| `tmp/test_p3.py` | PATCH config 断言适配新响应契约（+1 项 hook_results 断言，66 项） |
| `md/01-database.md` | validate_name 约定 + 新 `_meta.*` 键 |
| `md/02-orchestrator-core.md` | 归档摘要/error.log 已实现；未识别文件熔断；cmd hook 统一校验；cron 熔断；祖先引用 |
| `md/04-scheduler-gpu.md` | last_fire 本地时区；at 去重；validate_schedule；nvidia-smi 容错；采集失败 reason；max_parallel None 语义 |
| `md/05-api-frontend.md` | PATCH 响应契约；at 校验；name 校验；KI-10/11 已修复标注 |
| `progress/007-code-review.md` | KI-05~11/14/15/16 标 ✅；KI-18 标 ✅；KI-12/13 标注遗留 |
| `agent/decisions.md` | 追加 D24 |

## 2. 修复要点

### KI-05 — last_fire 时区混用
cron 匹配按本地时间（`matches(now.astimezone())`），原去重键却用 UTC
`now.strftime(...)`。修复：`_fire_key()` 统一为 `now.astimezone()` 本地
时区字符串，存取同口径（按 owner 指示取"统一为本地"）。

### KI-09 — cron 熔断 / 归档语义 / at 去重
- 熔断：cron 任务终态时计数 `_meta.consecutive_failures`（exit 0 清零），
  达 `MAX_CONSECUTIVE_FAILURES=3` 不再 re-arm，保持 failed 终态并记
  `_meta.rearm_blocked`；源文件随之正常归档 failed/。
- 归档顺序**有意保留**（re-arm 先于归档检查）：周期 cron 任务源文件
  驻留 processing/ 是预期语义，已加注释，不改行为。
- at 任务：`_due` 补 last_fire 检查，已有 last_fire 的 at 任务不再触发。

### KI-06 — inbox 滞留 / 坏文件反复处理
- 无法识别文件名连续 3 轮（1s/轮）后移 failed/ 并写 `.note.txt`
  （给写入中的文件留宽限）；
- 导入失败的文件也记内容 hash：同一坏文件再投递按 duplicate 直接归档。

### KI-10 — at 校验下沉 / PATCH 响应提示
`scheduler.validate_schedule()` 统一校验 cron+at，CLI / POST /api/tasks /
watcher 导入三入口复用（非法值 400 或进 failed/，不落库）。PATCH config
保持"kv 任意写、worker 白名单裁决热生效"语义，响应改
`{"config": ..., "hook_results": {key: {"hook_id": N|null, "rejected": ...}}}`。

### KI-11 — samples 路径注入
`db.validate_name()` 拒绝空名、`.`/`..`、`/` `\`，在 create_project /
create_task 单点卡住全部入口；server `_samples_dir()` 对库内名字再校验。

### KI-07 / KI-08 — max_parallel
main.py 删 `max(2, ...)` 死参数（上限单源在 dispatcher）；judge 改
`is None` 显式判断，`max_parallel=0` 正确表示禁止并行。

### KI-14 — 祖先引用
`_resolve_successors` 引用名≠直接前驱时按名字在同项目内查更早祖先
（须 done 且有 `_meta.output`）；更新前重读后继当前状态。

### KI-15 — nvidia-smi 容错
`parse_nvidia_smi_csv()`：util `[N/A]`→None；显存字段不可解析整行跳过；
cli 复用同一解析。采集异常时拒绝原因写 `GPU metrics collection failed: ...`
（空机放行不受影响）。

### KI-16 — 归档摘要
done 归档往文件 JSON 追加 `_result`（各任务 id/状态/起止/归档时间）；
failed 归档旁写 `<name>.error.log`（含各任务 DB error 字段）。

### KI-18（顺手）
删 `finish_processing` 死代码与 `_processing` 簿记；cmd hook 统一走
`hooks.enqueue`（非 running / 未知类型被拒 → cmd 文件进 failed/）。

### resumed_from_run（不改行为）
wandb 惰性 import 下只能存确定性 run name，hooks.py 加 as-built 注释；
md/03 已有对应条目。

## 3. 验证证据

回归（全部通过）：
```
python tmp/smoke_test.py     → === smoke test PASSED ===
python tmp/test_ki_fixes.py  → === KI fixes test PASSED ===
python tmp/test_p2_hooks.py  → === P2 E2E PASSED ===
python tmp/test_p3.py        → === 66/66 checks passed === / P3 PASSED
```

新增 `python tmp/test_ki_med.py` → **57/57 PASSED**，分组：
- KI-05：last_fire == `now.astimezone()` 本地格式；同（本地）分钟去重；
  跨分钟再触发（3 项）。
- KI-09b：at 到点触发一次；重置 scheduled 后不再触发（2 项）。
- KI-09a：连续失败 1/2 次正常 re-arm 且计数正确；第 3 次保持 failed +
  rearm_blocked；成功后计数清零并正常 re-arm（7 项）。
- KI-07/08：judge(max_parallel=0) → "max parallel reached (1/0)"；
  None 回退 guard 默认；main.py 无 max(2,...)（3 项）。
- KI-15：`[N/A]` util→None；显存 N/A 行跳过；采集失败 reason 前缀 +
  wait_reason kv 一致；空机放行不受影响（5 项）。
- KI-11：7 个坏名字全拒 + 正常名接受；create_project/create_task 拒绝；
  坏名 project 文件进 failed/ 且带 error.log（11 项）。
- KI-10：validate_schedule 拒/收；坏 at 任务文件进 failed/ 不落库；
  同内容坏文件再投递按 duplicate 进 done/；API PATCH 响应
  config+hook_results；API 坏 at → 400（8 项）。
- KI-06：未识别文件前 2 轮滞留、第 3 轮移 failed/ + note（4 项）。
- KI-18：cmd hook 投非 running 任务被拒进 failed/ 且 DB 无 hook 泄漏；
  未知类型同（4 项）。
- KI-14：c 经祖先 a 解析 resume_from；直接前驱引用不受影响（2 项）。
- KI-16：done 归档含 `_result` 摘要；failed 归档旁写 error.log 且含
  DB error 字段（5 项）。

## 4. 遗留问题

- **KI-12 / KI-13（med，未修）**：sample_from_weights 按 "lora_" 命名约定
  备份（应改为按实际加载键集合）；hot_keys.set_dotted 越界静默跳过但
  result 报 applied。两项均在 UnifiedTrainer/engine 训练引擎侧，按本轮
  owner 清单未动，建议下轮处理。
- **low 项（未修）**：KI-17（close 只关当前线程 / heartbeats 无 retention /
  timeout=30 死代码 / 参数名 type 遮蔽）、KI-19（JSON 行 isinstance /
  _REJECT_PREFIXES 死代码 / acked 回收 / dry-run result 字段 /
  prompts_by_tag(None)）、KI-20（_static URL 解码 / SSE 错误帧）。
  KI-18 已随本轮修复。
- `PATCH config` 响应契约变化（顶层从物化 config 变为
  `{"config","hook_results"}`）：前端未调用该端点，无影响；test_p3 已
  适配。外部脚本若依赖旧契约需同步调整。
- cron 熔断上限 3 为全局常量；如需 per-task 调整可后续做成 kv 配置。
