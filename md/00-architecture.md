# 00 — 总体架构（索引）

> 总体设计见 `doc/Improvement_k3.md`。本目录按大功能拆分设计文档，
> 每个功能动工前补全细节、完成后更新为"as-built"状态。

## 功能文档索引

| 文档 | 功能 | 实施期 | 状态 |
|------|------|--------|------|
| `01-database.md` | SQLite 数据层（schema + db.py + 读写纪律） | P1 | 设计完成 |
| `02-orchestrator-core.md` | Watcher + Dispatcher + 文件流转 + train.py 接入 | P1 | 设计完成 |
| `03-hooks.md` | HookManager 与五类 hook（采样/保存/恢复/热改/挂起） | P2 | 设计完成 |
| `04-scheduler-gpu.md` | cron 调度 + GPU Guard 3/4 VRAM 准入 | P3 | 设计完成 |
| `05-api-frontend.md` | FastAPI + Vue 3 前端 | P3/P4 | 设计完成 |
| `06-cli-scheduler-multigpu.md` | CLI 全量 CRUD（task/schedule/project/prompt）+ 多卡任务（gpus/gpu_ids）准入与 DDP 执行 | P5 | 已实施（P5a–P5f，见 progress/011） |
| `07-minimaxh3-training.md` | MiniMax-H3 训练接入：图像对 → 视频，音频延后 | P0–P4 | as-built |

## 关键全局决定

1. SQLite 为唯一事实源；JSON 文件只是导入入口，永不回写。
2. Orchestrator 与 worker（train.py 子进程）分离；worker 以退出码通信
   （0=done，42=suspend 挂起，其他=failed）。
3. 前端、CLI、inbox 文件三条操作路径平级，底层共用 db.py/hooks.py 同一套函数。
4. 层级模型：Project → Task（链/定时/独立）→ Hook 指令。
5. 所有代码在 `E:\UnifiedTrainer\ScheduledTrainer\` 内自包含，不引用外部代码；
   不创建虚拟环境（见 `doc/REQUIREMENTS.md`）。
