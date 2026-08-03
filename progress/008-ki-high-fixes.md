# 008 — KI-01~04 high 级修复总结

> 日期：2026-07-19。范围：`progress/007-code-review.md` §7 的 4 项 high 级
> known-issue。决策记录：D23（agent/decisions.md）。本轮同时解除 D21 的
> "不改代码"约束（仅限这 4 项）；med/low 项仍未动。

---

## 1. 改动文件清单

| 文件 | 改动 |
|------|------|
| `UnifiedTrainer/engine/trainer.py` | KI-01：hook 挂载点移入 `if step_advanced:` 块内 |
| `orchestrator/db.py` | KI-02：create_project 用 `rowcount==1` 判断，IGNORE 时 SELECT 取已有 id |
| `orchestrator/dispatcher.py` | KI-03：新增 `STARTUP_TIMEOUT=300.0` 常量、纯函数 `_hb_stale()`，`_monitor` 加启动宽限判死；-999 错误文案更新 |
| `orchestrator/hooks.py` | KI-04：新增公共函数 `set_config_and_notify(db, task_id, key, value)` |
| `orchestrator/watcher.py` | KI-04a：cmd set_config 改走 `set_config_and_notify` |
| `orchestrator/cli.py` | KI-04b/c：新增 `config set <task_id> <key> <value>` 子命令，值先 json.loads 失败回退字符串 |
| `orchestrator/server.py` | KI-04c：PATCH /api/tasks/{id}/config 统一走 `set_config_and_notify` |
| `tmp/test_ki_fixes.py` | 新增：25 项断言覆盖四项修复 |
| `md/02-orchestrator-core.md` | 心跳判活条目补启动宽限说明 |
| `agent/decisions.md` | 追加 D23 |
| `progress/007-code-review.md` | KI-01~04 标记 ✅ 已修复 |

## 2. 修复要点

### KI-01 — hook 挂载点 × 梯度累积
`trainer.py` 训练循环里 `maybe_run` 原先在 epoch 循环尾部无条件执行，梯度
累积时每个 micro-batch 都触发 hook（重复采样/重复写 kv/suspend 时机错乱）。
修复后挂载点移入 `if step_advanced:` 块（`accelerator.sync_gradients` 为真、
optimizer 真正 step 之后），micro-batch 不再触发。旧 `--config` 模式不注入
`hook_manager` 属性，`getattr(self, "hook_manager", None)` 返回 None 跳过，
行为与修复前完全一致。`train.py` dry-run 循环无梯度累积概念，每步调
`maybe_run(step, config=config)` 语义正确，保持不变。

### KI-02 — create_project lastrowid 陷阱
`INSERT OR IGNORE` 被忽略时 `cursor.lastrowid` 是**上一次插入**的陈旧值，
调用方拿到错误 project_id。修复：`cur.rowcount == 1 and cur.lastrowid`
才直接返回，否则 `SELECT id FROM projects WHERE name=?`。

### KI-03 — 心跳判活启动宽限
原 `_monitor` 只在 `hb is not None` 时判超时，模型加载期卡死的 worker
（从未写心跳）永不判死。修复：
- 新常量 `STARTUP_TIMEOUT = 300.0`（取值理由见 D23：约为心跳 120s 的
  2.5 倍，覆盖 4B+ 模型加载 + 量化 + block swap 到首 step 的典型时长，
  又不让卡死进程久占 GPU）；
- 新纯函数 `_hb_stale(hb_age, uptime, startup_timeout, timeout)`：
  `hb_age=None`（无心跳）时按 uptime > 300s 判死；有心跳维持 120s 判活；
- `_monitor` 记录 `started = time.monotonic()`，参数化 timeout 便于测试注入。

### KI-04 — set_config 统一入口
`hooks.py` 新增 `set_config_and_notify(db, task_id, key, value)`：
1. 先写 `task_config_kv`（任何状态允许，非 running 下次启动生效）；
2. `_` 前缀元数据键直接返回（不注入训练 config，不投 hook）；
3. 任务 running 时经 `enqueue()` 投 `patch_config` hook
   （payload=`{key: value}`），走统一校验 + worker 热改白名单。

三个入口全部复用：watcher cmd set_config、新增 CLI
`config set <task_id> <key> <value>`（值 json.loads 优先、回退字符串）、
server PATCH /api/tasks/{id}/config（多键 patch 现按每键一 hook，白名单
粒度更细）。

## 3. 验证证据

回归（全部通过）：
```
python tmp/smoke_test.py     → === smoke test PASSED ===（t1/t2 done，链式解析正常）
python tmp/test_p2_hooks.py  → === P2 E2E PASSED ===（六类 hook + suspend/resume 全绿）
python tmp/test_p3.py        → === 65/65 checks passed === / === P3 test PASSED ===
```

新增 `python tmp/test_ki_fixes.py` → **PASSED**，25 项断言摘录：
- KI-01：`maybe_run` 位于 `if step_advanced:` 块内、pbar.close() 之前，
  缩进正确；dry-run 每步调用保持。
- KI-02：同名重复 `create_project` 返回相同 id（1 == 1），且不等于中间
  插入项目的 id（陈旧 lastrowid 陷阱不复现）。
- KI-03：`_hb_stale` 判据 6 项（宽限内存活/超 300s 判死/自定义宽限/有心跳
  120s 规则/任意长 uptime 判死）；`_monitor` 端到端：假 proc（永不退出、
  永不写心跳）注入 `startup_timeout=0.5` → rc=-999 且收到 kill。
- KI-04：非 running 只写 kv 不投 hook；running dry-run 任务上
  ① inbox `cmd_ki04.json`（set_config）→ kv=5e-5 且 patch_config hook
  被 worker 消费 **done**；② `cli config set`（数字 10 按 JSON 解析为 int、
  `abc-1` 回退字符串）→ hook 消费 **done**；③ 结构键 `output.save_name`
  的 hook 被 worker 白名单拒 **failed**（统一校验生效）；任务最终 done。

## 4. 遗留问题

- med/low KI（KI-05~20）未动，仍按 007 §7 排期；其中 KI-10/20 与 KI-04
  同根的 API 侧白名单已由本次统一入口部分覆盖（server PATCH 已走
  set_config_and_notify），但 `at` 校验下沉、`_static` URL 解码、SSE
  错误帧等仍待修。
- `set_config_and_notify` 对 running 任务投 hook 前不做键白名单预判，
  结构键会写 kv 后被 worker 拒（hook failed）——kv 与热生效状态可能短暂
  不一致，属已知取舍（白名单单一事实源在 worker 侧 hot_keys）。
- STARTUP_TIMEOUT=300s 为全局常量；超大模型冷启动若超时判死，可调大
  常量或后续做成 per-task kv 配置。
