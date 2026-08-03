# 05 — API 与前端设计（P3/P4）

> **均已实现（2026-07-18，见 progress/004、005）；2026-07-18 code review 后正文
> 全面回写 as-built（007）。** 原设计（FastAPI/WebSocket/Vue3+Vite+ECharts）因环境
> 约束整体降级，降级决定见 D16/D19/D20；API 路由面与原设计保持一致。

## 范围

`orchestrator/server.py`（标准库 HTTP：REST + SSE + 静态托管）+
`web/dist/index.html`（零构建静态 SPA）。

## 技术栈（as-built）

| 层 | 原设计 | as-built | 原因 |
|----|--------|----------|------|
| API | FastAPI + uvicorn | 标准库 `http.server.ThreadingHTTPServer`（D16） | 环境无 fastapi 且禁止 pip |
| 推送 | WebSocket `/ws` | SSE `GET /api/events`，每 2s 全量推 {ts, tasks, gpu} 帧（D16） | 标准库无 WS |
| 前端 | Vue 3 + Vite + TS + Pinia + Naive UI | 单文件零构建 SPA：原生 ES module + fetch + EventSource + hash 路由（D19） | node 存在但 npm 缺失 |
| 图表 | ECharts | 内联 SVG 手绘 loss/lr 曲线 | 无构建链/无 CDN |
| 托管 | FastAPI 静态目录 | server.py 静态托管 `web/dist/`，未命中回退 index.html（D20） | 单端口 7860 |

## API 面（as-built，全部落地）

```
GET/POST        /api/projects            GET/PATCH /api/projects/{id}
GET/POST        /api/tasks               GET       /api/tasks/{id}（含 config_kv）
POST            /api/tasks/{id}/hooks    GET       /api/tasks/{id}/hooks
GET/PATCH       /api/tasks/{id}/config   POST      /api/tasks/{id}/cancel|resume
GET/POST        /api/prompts             GET       /api/gpu
GET             /api/tasks/{id}/heartbeats?since=
GET             /api/events              （SSE，每 2s 一帧 {ts, tasks, gpu}）
GET             /api/samples/{task_id}   （采样画廊列表）
GET             /samples-file/{task_id}/{name}  （图片字节，防目录穿越）
```

- 端点内部只调 `db.py` / `hooks.py` 的公共函数（**as-built 不经 dispatcher**），
  与 CLI 零差异；
- `POST /api/tasks` 入库前统一校验 cron/at（`scheduler.validate_schedule`，
  非法值 400）；project/task 名字在 db 层统一经 `validate_name` 拒绝
  `..` 与路径分隔符（KI-10/KI-11 已修复，D24）；
- `PATCH /api/tasks/{id}/config` 响应为
  `{"config": <物化 config>, "hook_results": {key: {"hook_id": N|null,
  "rejected": ...}}}`——kv 仍可写任意键（热生效由 worker 白名单裁决），
  响应附带每键 hook 投递结果提示（D24）；
- samples 路径拼接对库内 project/task 名再做一次 `validate_name` 校验
  （防御历史脏数据，KI-11 已修复，D24）；
- `GET /api/gpu` 返回 {latest, history, waiting}；**不含逐任务 wait_reason**——
  前端对 waiting 任务逐一查 `GET /api/tasks/{id}` 取 `_meta.gpu_wait_reason`；
- 无鉴权（本机单机使用），绑定 127.0.0.1，端口默认 7860（与 gradio 默认一致）；
- 错误格式 `{"error":...}`；400 参数 / 404 未找到 / 409 状态冲突。

## 前端（as-built 零构建 SPA）

页面（hash 路由）：

- `#/` 项目列表卡片（名称、状态、任务统计）；
- `#/projects/:id` 任务列表（含链/定时）+ 提示词库（**仅列表+新增**）；
- `#/tasks/:id` 实时监控：内联 SVG loss/lr 曲线（**SSE 帧驱动 heartbeats 轮询**，
  非独立 WS 推送）、六个 hook 按钮（sample / sample_from_weights / save /
  restore / patch_config / suspend，+ resume）、采样画廊、hook 历史；
- `#/gpu` 显存面板 + waiting_gpu 队列及拒绝原因（逐任务补查）；
- `#/new` 新建向导（表单 ↔ JSON 双视图，可直接粘贴 JSON）。

**交互降级清单**（相对原 Vue 设计，未实现）：

- 无 cron 人性化预览（"下次触发：…"）；
- 无链式创建向导（`depends_on` 依赖 id 手填）；
- 无 DAG 图、无"下次触发时间"列、无任务详情抽屉、无日志 tail、无 wandb 链接；
- 提示词库无编辑/删除；项目卡片无最近活动。

## 决定

- 单端口 7860 同时服务 API + 前端（D20）；路径做 abspath+前缀校验防穿越；
- 前端不直接读 DB/文件系统，一切经 API；
- SSE 只做推送，操作一律 REST（幂等、可重试）；
- 未来若补齐依赖：fastapi 优先整体重写 server.py（D16）；npm 补齐后可按原
  Vue 3 + Vite 设计重做前端，API 面不变（D19）。
- KI-20 已修复（010）：`_static` URL 解码（`%20` 等文件名可访问）；
  `_json_body` 1MB 上限（超限 400）；SSE 流内异常记录后静默关闭，不再对已
  流式 socket 追加 500；顺带修复 `_route` 处理器返回值语义——`_send`/
  `_send_file`/`_sse` 返回 True 作为"已处理"标记，修复此前每个成功响应后
  又追加一个 404、污染 keep-alive 连接的 bug。
- 已知问题见 `progress/007-code-review.md` §7（全部已修复清零，见
  `progress/010-ki-final.md`）。
