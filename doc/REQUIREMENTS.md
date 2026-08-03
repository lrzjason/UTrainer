# 工作约定（REQUIREMENTS）

> 本文档记录 owner 对 ScheduledTrainer 开发过程的硬性要求，所有 agent / subagent 必须遵守。
> 更新日期：2026-07-18

## 目录结构与职责

`E:\UnifiedTrainer\ScheduledTrainer\` 是 **agent 交互区**，与外部源代码严格区分：

| 目录 | 用途 |
|------|------|
| `UnifiedTrainer/` | 训练器源码副本（由 `E:\UnifiedTrainer\UnifiedTrainer` 复制而来，可按需改名/修改） |
| `orchestrator/` （待建） | 任务编排器代码 |
| `web/` （待建） | Vue 3 前端 |
| `doc/` | 要求文档与总体设计文档（本目录），供后续更新和阅读 |
| `md/` | **每个大功能一份对应设计文档** |
| `agent/` | agent 过程中的决定、笔记，可自由创建任意文本记录编写进度 |
| `progress/` | subagent 任务总结 + 随开发推进的 section 总结 |
| `skills/` | 可复用的 skill 文档（**有复用价值才创建**） |
| `tmp/` | 临时代码文件 / 临时文件 |

## 硬性规则

1. **代码隔离**：不引用 `E:\UnifiedTrainer\ScheduledTrainer` 以外的代码；
   需要外部代码时**复制进来**再改。
2. **决策留痕**：开发过程中的所有技术决定记录到 `agent/`。
3. **进度总结**：每个 subagent 完成任务后总结写入 `progress/`；
   随开发进度不断做 section 总结，也写入 `progress/`。
4. **设计文档**：每个大功能动工前/完成时，在 `md/` 保存对应设计文档。
5. **环境**：**不需要也不允许创建/安装虚拟环境**，直接使用系统/managed Python。
6. 可用 subagent 并行完成部分代码实现。
