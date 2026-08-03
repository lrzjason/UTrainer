# Section 总结 001 — 工作区初始化与设计文档落地

**日期**：2026-07-18
**阶段**：准备（P0）✅

## 已完成

1. **工作区结构确认**：`E:\UnifiedTrainer\ScheduledTrainer\` 已存在，
   含 `UnifiedTrainer/`（训练器源码副本，已就位）、`agent/ doc/ md/ progress/ skills/ tmp/`。
2. **`doc/REQUIREMENTS.md`**：固化 owner 的工作约定——代码隔离、决策留痕（agent/）、
   subagent 总结与 section 总结（progress/）、每大功能一份设计文档（md/）、
   不创建虚拟环境。
3. **`doc/Improvement_k3.md`**：总体设计文档，路径全部迁移到 ScheduledTrainer。
   内容覆盖：Project→Task→Hook 层级、SQLite 七表 DDL、Watcher/Dispatcher/Scheduler/
   GPU Guard（3/4 VRAM 并行准入）、五类 hook、CLI、Vue 3 前端、P1–P4 实施路线。
4. **`md/` 设计文档 6 份**：
   - `00-architecture.md` 索引 + 全局决定
   - `01-database.md` SQLite 数据层（db.py 接口约定）
   - `02-orchestrator-core.md` Watcher + Dispatcher + train.py `--task-id/--db` 接入
   - `03-hooks.md` HookManager 与五类 hook 协议
   - `04-scheduler-gpu.md` cron + GPU Guard 准入规则
   - `05-api-frontend.md` FastAPI + Vue 前端

## 关键决定（详见 md/00）

- SQLite 唯一事实源，JSON 只做导入入口；
- worker = train.py 子进程，退出码 0/42/其他 通信；
- 前端 / CLI / inbox 文件三路径平级；
- 全部代码自包含于 ScheduledTrainer，不建虚拟环境。

## 下一步（P1）

按 `md/01` 和 `md/02` 实施：`orchestrator/schema.sql` + `db.py` + watcher/dispatcher
最小版 + train.py `--task-id/--db` 模式，验收标准为投递项目文件跑通串行链并归档。
