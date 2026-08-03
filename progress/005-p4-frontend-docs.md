# 005 — P4 实施总结（Web 前端 + 文档）

日期：2026-07-18 ｜ 状态：**完成（P3 回归受磁盘满阻塞待重跑，见 §5）**

## 1. 环境探测与选型

- `node -v` = v24.15.0；**`npm` 不存在** → 无法走 Vite/Vue 构建链 →
  **降级为零构建静态 SPA**（决策 D19，agent/decisions.md）。
- 产物：`web/dist/index.html` 单文件（约 21KB），原生 ES module + fetch +
  EventSource + 内联 SVG loss 曲线；无 CDN/无构建步骤，离线可用。

## 2. 交付物

| 文件 | 内容 |
|------|------|
| `web/dist/index.html`（新增） | hash 路由 SPA：#/ 项目卡片；#/projects/:id 任务链+定时任务+提示词库；#/tasks/:id 实时 loss（SSE 驱动增量拉 heartbeats）+ 六个 hook 按钮 + resume/cancel + 采样画廊 + hook 历史 + 物化配置；#/gpu 显存条+waiting_gpu 拒绝原因+历史快照；#/new 表单↔JSON 双视图新建向导 |
| `orchestrator/server.py`（修改） | 静态托管：非 /api/ GET → web/dist/（未命中回退 index.html，abspath 前缀校验防穿越）；`/samples-file/{task_id}/{name}` 采样图片字节；APIServer 增加 static_root |
| `md/ORCHESTRATOR.md`（新增） | 人用手册（as-built）：架构/模型/CLI/API/任务文件/hook 协议/GPU Guard/故障排查 |
| `skills/unified-trainer-orchestrator/SKILL.md`（新增） | Agent skill：frontmatter + 触发场景 + 标准操作流 + 配方 + 红线 |
| `md/05-api-frontend.md`（修改） | 前端部分标记已实现（含降级说明） |
| `agent/decisions.md`（追加） | D19（零构建 SPA 降级）、D20（server.py 静态托管） |

## 3. 前端 fetch URL 与 server.py 路由核对（逐条一致）

`GET/POST /api/projects`、`GET /api/projects/{id}`、`GET/POST /api/tasks`
（?project_id=）、`GET /api/tasks/{id}`、`GET/POST /api/tasks/{id}/hooks`、
`POST /api/tasks/{id}/cancel|resume`、`GET /api/tasks/{id}/heartbeats?since=`、
`GET/POST /api/prompts`（?project_id=）、`GET /api/gpu`、
`GET /api/samples/{id}`、`/api/events`（SSE）、`/samples-file/{id}/{name}`。
前端热改走 hook 面板 patch_config 按钮，未用 PATCH config 页，无悬空 URL。

## 4. 验证输出

```
GET /            -> 200 (21441B)   GET /index.html -> 200
GET /api/projects-> 200            GET /api/gpu    -> 200
GET /nope        -> 200 (SPA 回退 index.html，符合设计)
SSE /api/events  -> event: status 帧 {ts, tasks, gpu} 正常推送
python tmp/smoke_test.py      -> === smoke test PASSED ===（P1 回归）
python tmp/test_p2_hooks.py   -> === P2 E2E PASSED ===（P2 回归）
python tmp/test_p3.py         -> sqlite3.OperationalError: database or disk is full ✗
```

## 5. 遗留风险 / 阻塞

1. **E: 盘 100% 满**：P4 期间磁盘被写满，P3 回归因 sqlite 无法 commit 失败。
   **非 P4 改动引入**（004 中 65/65 通过；P4 仅给 server.py 增加静态分支，
   py_compile/手工 curl 验证通过；P1/P2 回归本次仍通过）。
   **磁盘清理后需重跑 `python tmp/test_p3.py` 确认。**
2. 前端为降级版（无 ECharts/Naive UI）；loss 曲线为 SVG 折线，样式从简。
   补 npm 后可按 md/05 原设计重做（D19）。
3. 采样画廊与 API 依赖 server 托管；file:// 直开仅静态壳可见。
