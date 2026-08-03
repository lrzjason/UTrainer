# 004 — P3 实施总结（cron 调度 + GPU Guard + HTTP API）

日期：2026-07-18 ｜ 状态：**完成，端到端验证通过（65/65）**

## 交付物

### 新增

| 文件 | 内容 |
|------|------|
| `orchestrator/scheduler.py` | 纯标准库 cron 解析器（五段 `分 时 日 月 周`，支持 `* , - /` 步进，dow 0/7=周日，dom/dow 同受限取 OR）+ Scheduler：30s tick（可配）扫 `scheduled` 任务到点置 pending；支持一次性 `at`（ISO 时间，naive 按本地时区）；`_meta.last_fire` 按分钟去重，错过触发只保留一个排队实例 |
| `orchestrator/gpu_guard.py` | 采集链 NvmlProvider（pynvml 惰性 import）→ NvidiaSmiProvider（子进程解析）→ NullProvider（无 GPU 信息：空机放行、并行保守拒绝）；FakeProvider 供测试注入。`judge()` 实现 md/04 准入：空机放行 → allow_parallel → max_parallel → 某卡 free > total*3/4；每次判定写 gpu_snapshots + kv `_meta.gpu_decision`（快照+结果）；拒绝置 waiting_gpu 并写 `_meta.gpu_wait_reason` |
| `orchestrator/server.py` | 标准库 `http.server.ThreadingHTTPServer` 实现 md/05 全部 REST 端点（projects/tasks/hooks/config/cancel/resume/prompts/gpu/heartbeats/samples），SSE `GET /api/events` 每 2s 推 tasks+gpu 状态帧（WebSocket 降级）。绑 127.0.0.1:8100，可独立 `python -m orchestrator.server` 启动 |
| `tmp/test_p3.py` | 端到端验证脚本，65 项断言 |

### 修改

| 文件 | 改动 |
|------|------|
| `orchestrator/dispatcher.py` | 重构为 worker 池：每 worker 独立监控线程（心跳判活不变），`max_parallel` 默认 1（P1 串行行为不变）；GPU Guard 准入后才 spawn；Admit(gpu) 时注入 `CUDA_VISIBLE_DEVICES=<gpu>` 并写 `_meta.gpu_index`；任一 worker 退出 `_wake.set()` 立即重扫 waiting_gpu 补位；cron 任务终态后自动 re-arm 置回 scheduled |
| `orchestrator/db.py` | tasks 增加 `at` 列（schema + 旧库 ALTER TABLE 迁移）；`next_runnable_tasks(statuses)` 覆盖 pending+waiting_gpu 且不再过滤 cron（status 即"到点"唯一语义）；新增 `update_project` / `add_prompt` / `list_prompts` / `heartbeats_since` / `list_gpu_snapshots` |
| `orchestrator/schema.sql` | tasks.at 列 |
| `orchestrator/main.py` | 新增 Scheduler 线程、`--api/--port`（起 HTTP API）、`--max-parallel`（默认 1）、`--tick` |
| `orchestrator/cli.py` | 新增 `task create --project --name [--model] [--config] [--cron/--at] [--allow-parallel] [--priority]`，cron/at 入库前校验 |
| `orchestrator/watcher.py` | project/task JSON 支持 `"at"` 字段；cron 或 at 存在即 scheduled |
| `UnifiedTrainer/train.py` | dry-run 结束写 `_meta.cuda_visible_devices`（绑卡可观测性/测试断言用） |

## 验证命令与输出摘录

| 命令 | 结果 |
|------|------|
| `python -m py_compile`（9 个新/改文件） | 0 |
| `python tmp/test_p3.py` | **65/65 断言通过**，退出码 0 |
| `python tmp/smoke_test.py`（P1 回归） | `=== smoke test PASSED ===` |
| `python tmp/test_p2_hooks.py`（P2 回归） | `=== P2 E2E PASSED ===`（31 项全过） |
| `python -m orchestrator.cli task create --cron "*/5 * * * *" --allow-parallel` | `task id=13 ... status=scheduled`；非法 cron `61 * * * *` 被拒退出码 2 |

E2E 关键场景（`python tmp/test_p3.py`）：

```
C. fake VRAM 不足（free 4000/24000）：
   A(40 步) running → B(at=now+3s, allow_parallel) 到点 → scheduler 置 pending
   → guard 拒绝 → B=waiting_gpu, _meta.gpu_wait_reason="insufficient free VRAM (need > 75%)"
   → gpu_snapshots 有判定记录 → A done → B 自动补位 running → done

D. fake VRAM 充足（free 22000/24000 > 75%）：
   C running 期间 D 并行 Admit(gpu=0)，D 带 CUDA_VISIBLE_DEVICES=0 启动
   （worker 回写 _meta.cuda_visible_devices="0" 证实），C/D 均 done

E. API：projects/tasks/hooks/config/prompts/gpu/heartbeats/samples 往返全过；
   hook/resume 非合法状态返回 409；未知路由 404；SSE 首帧含 "tasks" 字段
```

## 设计落地差异（as-built）

- **FastAPI → 标准库降级**（D16）：环境无 fastapi/uvicorn 且禁止 pip 安装，
  用 `http.server.ThreadingHTTPServer` 实现等价 JSON REST；WebSocket 降级为
  SSE（`/api/events`，2s 一帧）。路由与 md/05 对齐，未来环境具备依赖时可
  整体替换 server.py 而不动路由面。
- **GPU 采集降级链**（D15）：pynvml 与 torch 均不存在，实际运行走
  nvidia-smi 子进程（本机无 nvidia-smi 则 NullProvider 保守拒绝并行）；
  测试全程 FakeProvider 注入。
- **cron re-arm**：md/04 未明确周期任务触发后行为，实现为 cron 任务终态后
  自动置回 scheduled 等下一轮；`at` 任务一次性不 re-arm。
- **waiting_gpu 语义扩展**：max_parallel=1（默认串行）时，非 allow_parallel
  的后续任务也会经 guard 判定置 waiting_gpu（原因 "parallel not allowed"），
  worker 退出后补位——与 P1 串行结果一致，仅中间状态更可见。
- **决策记录载体**：gpu_snapshots 表无决策列，判定结果（快照+admit/reason）
  写 kv `_meta.gpu_decision`，拒绝原因另写 `_meta.gpu_wait_reason`。
- **samples 端点**：按 `workspace/samples/<project>/<task>/` 目录扫描返回
  文件列表（沿用 P2 采样输出约定），不含图片字节。

## 遗留风险 / 待办

1. **真实 GPU 路径未实测**：NvmlProvider / NvidiaSmiProvider 代码按文档
   写好但本机无 GPU/nvidia-smi，仅 FakeProvider 路径验证；VRAM>3/4 判定的
   真实误差（其他进程瞬时占用）未评估。
2. **worker 崩溃池状态**：worker 线程异常时 proc 可能残留（_reap 只收
   已完成线程）；真实环境建议加 proc 存活巡检。
3. **cron 任务的源文件归档**：re-arm 后任务非终态，含 cron 任务的
   processing/ 源文件长期不归档（符合"周期任务永不结束"语义，但
   done/failed 目录看不到该文件）。
4. **SSE 无增量**：每帧全量 tasks 列表，任务多时有带宽浪费；前端 P4 可
   改 since 增量或只在版本变化时推。
5. **API 无鉴权**（设计如此，仅绑 127.0.0.1）；PATCH config 目前不过
   热改白名单直接写 kv，白名单校验只在 running 时的 patch_config hook
   侧生效——非 running 任务可写任意键（等价 CLI set_config，记录为已知
   行为）。
6. **Windows 控制台编码**沿用 P2 规避方式（subprocess text 模式注意
   encoding="utf-8"）。

技术决定见 `agent/decisions.md`（D14–D18）。
