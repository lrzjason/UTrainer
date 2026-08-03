"""HTTP API（P3）：标准库 http.server 实现的 JSON REST + SSE 推送。

环境约束：managed Python 无 fastapi/uvicorn 且禁止 pip 安装，故用
`http.server.ThreadingHTTPServer` 实现与 md/05 API 面等价的端点；
WebSocket 用 SSE（GET /api/events）降级替代。若未来环境装有
fastapi/uvicorn，可整体替换本模块而保持路由不变。

端点（全部 /api 前缀，JSON in/out，无鉴权，绑 127.0.0.1）：
    GET/POST   /api/projects              GET/PATCH/DELETE /api/projects/{id}
    GET/POST   /api/tasks                 GET/PATCH/DELETE /api/tasks/{id}
    POST       /api/tasks/{id}/hooks      GET       /api/tasks/{id}/hooks
    GET/PATCH  /api/tasks/{id}/config
    POST       /api/tasks/{id}/cancel     POST      /api/tasks/{id}/resume
    POST       /api/tasks/{id}/pause|unpause
    GET        /api/schedules             （cron/at 任务视图，含 next_fire）
    GET/POST   /api/prompts               GET       /api/gpu
    DELETE     /api/prompts/{id}
    GET        /api/tasks/{id}/heartbeats?since=
    GET        /api/samples/{task_id}
    GET        /api/events                （SSE：每 2s 推 tasks+gpu+schedule 快照）

静态托管（P4 新增，零构建 SPA）：
    GET  / 与 /assets/* → web/dist/ 下的静态文件（默认 index.html）
    GET  /samples-file/{task_id}/{name} → workspace/samples/<proj>/<task>/<name>

端点内部只调 db.py / hooks.py 公共函数，与 CLI 零差异。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs, unquote

from .db import DB, validate_name
from . import hooks as hook_proto
from .scheduler import validate_schedule, next_fire_of

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860  # 与 gradio 默认端口一致（T2ITrainer ui_flux_fill 同款）
_SSE_INTERVAL = 2.0
_TERMINAL = {"done", "failed", "cancelled"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "ScheduledTrainer/0.3"
    protocol_version = "HTTP/1.1"

    # ── 基础工具 ────────────────────────────────────────────────
    @property
    def db(self) -> DB:
        return self.server.db  # type: ignore[attr-defined]

    @property
    def workspace(self) -> str:
        return self.server.workspace  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # 静默默认访问日志 → logging
        logger.debug(fmt, *args)

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        # KI-20：请求体大小上限 1MB，防内存放大
        if length > 1024 * 1024:
            raise ValueError("JSON body too large (> 1MB)")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            raise ValueError("invalid JSON body")
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send(self, code: int, obj, headers: Optional[dict] = None) -> bool:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
        # 返回 True 作为"已处理"标记：_dispatch 以此区分路由未命中
        # （此前返回 None，导致每个成功响应后又追加一个 404，污染
        # keep-alive 连接——KI-20 轮顺手修复）
        return True

    def _err(self, code: int, msg: str) -> None:
        self._send(code, {"error": msg})

    # ── 分发 ────────────────────────────────────────────────────
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            body = self._json_body() if method in ("POST", "PATCH") else {}
            result = self._route(method, path, query, body)
            if result is None:
                self._err(404, f"no route: {method} {path}")
        except hook_proto.HookError as e:
            self._err(409, str(e))
        except (ValueError, KeyError) as e:
            self._err(400, str(e))
        except BrokenPipeError:
            pass
        except Exception as e:
            logger.exception(f"API error: {method} {path}")
            self._err(500, f"{type(e).__name__}: {e}")

    def _route(self, method: str, path: str, query: dict, body: dict):
        q1 = lambda k, d=None: (query.get(k, [d])[0])  # noqa: E731

        # ── 静态文件（web/dist SPA + 采样图片）──────────────────
        if method == "GET" and not path.startswith("/api/"):
            m = re.fullmatch(r"/samples-file/(\d+)/([^/]+)", path)
            if m:
                return self._sample_file(int(m.group(1)), unquote(m.group(2)))
            return self._static(path)

        # ── projects ────────────────────────────────────────────
        if path == "/api/projects":
            if method == "GET":
                return self._send(200, {"projects": self.db.list_projects()})
            if method == "POST":
                name = body.get("name")
                if not name:
                    raise ValueError("missing 'name'")
                pid = self.db.create_project(
                    name, description=body.get("description", ""),
                    default_model=body.get("default_model"),
                    tags=body.get("tags", []))
                return self._send(201, self.db.get_project(pid))
        m = re.fullmatch(r"/api/projects/(\d+)", path)
        if m:
            pid = int(m.group(1))
            proj = self.db.get_project(pid)
            if proj is None:
                return self._err(404, f"project not found: {pid}")
            if method == "GET":
                return self._send(200, proj)
            if method == "PATCH":
                fields = {k: body[k] for k in
                          ("description", "status", "default_model", "tags")
                          if k in body}
                self.db.update_project(pid, **fields)
                return self._send(200, self.db.get_project(pid))
            if method == "DELETE":
                # P5：仅全部任务终态时允许；先删任务（叶子优先）再删项目
                tasks = self.db.list_tasks(project_id=pid)
                if any(t["status"] not in _TERMINAL for t in tasks):
                    return self._err(
                        409, f"project {pid} has non-terminal tasks")
                for t in sorted(tasks, key=lambda x: -x["id"]):
                    self.db.delete_task(t["id"])
                self.db.delete_project(pid)
                return self._send(200, {"deleted": pid})

        # ── tasks ───────────────────────────────────────────────
        if path == "/api/tasks":
            if method == "GET":
                project_id = q1("project_id")
                tasks = self.db.list_tasks(
                    project_id=int(project_id) if project_id else None,
                    status=q1("status"))
                return self._send(200, {"tasks": tasks})
            if method == "POST":
                return self._create_task(body)
        m = re.fullmatch(r"/api/tasks/(\d+)", path)
        if m:
            tid = int(m.group(1))
            task = self.db.get_task(tid)
            if task is None:
                return self._err(404, f"task not found: {tid}")
            if method == "GET":
                task["config_kv"] = self.db.get_config_kv(tid)
                return self._send(200, task)
            if method == "PATCH":
                return self._patch_task(tid, body)
            if method == "DELETE":
                return self._delete_task(tid)
        m = re.fullmatch(r"/api/tasks/(\d+)/(hooks|config|cancel|resume|heartbeats|pause|unpause)",
                         path)
        if m:
            tid, sub = int(m.group(1)), m.group(2)
            if self.db.get_task(tid) is None:
                return self._err(404, f"task not found: {tid}")
            if sub == "hooks":
                if method == "GET":
                    return self._send(200, {"hooks": self.db.list_hooks(tid)})
                if method == "POST":
                    hid = hook_proto.enqueue(
                        self.db, tid, body.get("type", ""),
                        body.get("payload") or {})
                    return self._send(201, {"hook_id": hid})
            if sub == "config":
                if method == "GET":
                    return self._send(200, self.db.materialize_config(tid))
                if method == "PATCH":
                    kv = body.get("kv") or {k: v for k, v in body.items()
                                            if k != "kv"}
                    if not kv:
                        raise ValueError("empty config patch")
                    # 语义保持：kv 可写任意键（结构键也先落 kv），热生效由
                    # worker 侧 hot_keys 白名单裁决。KI-04 起统一走
                    # set_config_and_notify；响应附带每键 hook 投递结果，
                    # 调用方可感知 hook 是否被投递/拒绝。
                    hook_results = {}
                    for k, v in kv.items():
                        try:
                            hid = hook_proto.set_config_and_notify(
                                self.db, tid, k, v)
                            hook_results[k] = {"hook_id": hid}
                        except hook_proto.HookError as e:
                            logger.warning(f"patch_config hook rejected: {e}")
                            hook_results[k] = {"hook_id": None,
                                               "rejected": str(e)}
                    return self._send(200, {
                        "config": self.db.materialize_config(tid),
                        "hook_results": hook_results})
            if sub == "cancel" and method == "POST":
                task = self.db.get_task(tid)
                if task["status"] not in ("pending", "scheduled", "waiting_gpu"):
                    return self._err(
                        409, f"task {tid} status={task['status']}, cannot "
                        f"cancel (only pending/scheduled/waiting_gpu)")
                self.db.set_task_status(tid, "cancelled")
                return self._send(200, self.db.get_task(tid))
            if sub == "resume" and method == "POST":
                info = hook_proto.resume_task(self.db, tid)
                return self._send(200, info)
            if sub == "pause" and method == "POST":
                task = self.db.get_task(tid)
                if task["status"] not in ("scheduled", "pending",
                                           "waiting_gpu"):
                    return self._err(
                        409, f"task {tid} status={task['status']}, pause only "
                        f"from scheduled/pending/waiting_gpu")
                self.db.set_task_status(tid, "paused")
                return self._send(200, self.db.get_task(tid))
            if sub == "unpause" and method == "POST":
                task = self.db.get_task(tid)
                if task["status"] != "paused":
                    return self._err(
                        409, f"task {tid} status={task['status']}, "
                        f"unpause only from paused")
                new = "scheduled" if (task.get("cron") or task.get("at")) \
                    else "pending"
                self.db.set_task_status(tid, new)
                return self._send(200, self.db.get_task(tid))
            if sub == "heartbeats" and method == "GET":
                hbs = self.db.heartbeats_since(tid, since=q1("since"))
                return self._send(200, {"heartbeats": hbs})

        # ── schedules（tasks.cron/at 的一等视图，P5）────────────
        if path == "/api/schedules" and method == "GET":
            from .scheduler import next_fire_of
            project_id = q1("project_id")
            pid = int(project_id) if project_id else None
            rows = []
            for t in self.db.list_tasks(project_id=pid):
                if not (t.get("cron") or t.get("at")):
                    continue
                kv = self.db.get_config_kv(t["id"])
                nf = next_fire_of(t, kv)
                rows.append({
                    "id": t["id"], "name": t["name"],
                    "schedule": t.get("cron") or t.get("at"),
                    "next_fire": nf.isoformat() if nf else None,
                    "last_fire": kv.get("_meta.last_fire"),
                    "status": t["status"], "gpus": t.get("gpus", 1)})
            return self._send(200, {"schedules": rows})

        # ── prompts ─────────────────────────────────────────────
        if path == "/api/prompts":
            if method == "GET":
                project_id, tag = q1("project_id"), q1("tag")
                return self._send(200, {"prompts": self.db.list_prompts(
                    project_id=int(project_id) if project_id else None,
                    tag=tag)})
            if method == "POST":
                text = body.get("text")
                if not text:
                    raise ValueError("missing 'text'")
                pid = self.db.add_prompt(
                    body.get("project_id"), text, tag=body.get("tag"),
                    negative=body.get("negative"), meta=body.get("meta"))
                return self._send(201, {"id": pid})
        m = re.fullmatch(r"/api/prompts/(\d+)", path)
        if m and method == "DELETE":
            self.db.delete_prompt(int(m.group(1)))
            return self._send(200, {"deleted": int(m.group(1))})

        # ── gpu ─────────────────────────────────────────────────
        if path == "/api/gpu" and method == "GET":
            return self._send(200, {
                "latest": self.db.latest_gpu_snapshots(),
                "history": self.db.list_gpu_snapshots(
                    limit=int(q1("limit", "50"))),
                "waiting": [t for t in self.db.list_tasks(status="waiting_gpu")],
                "occupancy": self.db.running_task_gpus(),  # P5
            })

        # ── samples ─────────────────────────────────────────────
        m = re.fullmatch(r"/api/samples/(\d+)", path)
        if m and method == "GET":
            return self._samples(int(m.group(1)))

        # ── SSE 状态推送（WebSocket 降级）────────────────────────
        if path == "/api/events" and method == "GET":
            return self._sse()

        return None

    # ── 静态文件服务 ────────────────────────────────────────────
    _MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
             ".css": "text/css", ".json": "application/json",
             ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml",
             ".ico": "image/x-icon"}

    def _send_file(self, abs_path: str) -> None:
        if not os.path.isfile(abs_path):
            return self._err(404, "not found")
        ext = os.path.splitext(abs_path)[1].lower()
        with open(abs_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", self._MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True  # 已处理标记（同 _send）

    def _static(self, path: str) -> None:
        """web/dist 下的静态文件；/ 与未知前端路径回退 index.html（hash 路由单页）。"""
        # KI-20：URL 解码后再拼路径（%20 等文件名的空格不再 404）
        path = unquote(path)
        root = os.path.abspath(self.server.static_root)  # type: ignore[attr-defined]
        rel = path.lstrip("/") or "index.html"
        target = os.path.abspath(os.path.join(root, rel))
        if not target.startswith(root + os.sep) and target != root:
            return self._err(403, "forbidden")
        if not os.path.isfile(target):
            target = os.path.join(root, "index.html")
        return self._send_file(target)

    def _samples_dir(self, task: dict) -> str:
        """workspace/samples/<proj>/<task>，对库内名字做安全校验（KI-11）。

        新数据在 create_project/create_task 已统一拒绝非法名；这里对
        历史/手工落库的数据再防一道，拒绝路径分隔符与 ..。
        """
        proj = self.db.get_project(task["project_id"])
        proj_name = proj["name"] if proj else str(task["project_id"])
        try:
            validate_name(proj_name, "project name")
            validate_name(task["name"], "task name")
        except ValueError as e:
            raise ValueError(f"unsafe samples path: {e}")
        return os.path.join(self.workspace, "samples", proj_name, task["name"])

    def _sample_file(self, task_id: int, name: str) -> None:
        """采样图片字节：/samples-file/<task>/<name> → workspace/samples/<proj>/<task>/<name>。"""
        if name in (".", "..") or os.sep in name or "/" in name or "\\" in name:
            return self._err(400, "bad name")
        task = self.db.get_task(task_id)
        if task is None:
            return self._err(404, f"task not found: {task_id}")
        root = self._samples_dir(task)
        return self._send_file(os.path.abspath(os.path.join(root, name)))

    # ── 子实现 ──────────────────────────────────────────────────
    def _patch_task(self, tid: int, body: dict) -> None:
        """PATCH /api/tasks/{id}：任务字段白名单更新（与 CLI task update 同构）。"""
        from .validation import parse_gpu_ids

        fields = {}
        if "name" in body:
            fields["name"] = body["name"]
        if "priority" in body:
            fields["priority"] = int(body["priority"])
        if "allow_parallel" in body:
            fields["allow_parallel"] = 1 if body["allow_parallel"] else 0
        if "depends_on" in body:
            fields["depends_on"] = body["depends_on"]
        if "resume_from" in body:
            fields["resume_from"] = body["resume_from"]
        if "gpus" in body:
            fields["gpus"] = int(body["gpus"])
        if "gpu_ids" in body:
            ids = parse_gpu_ids(str(body["gpu_ids"]))
            want = int(body.get("gpus", self.db.get_task(tid).get("gpus", 1)))
            if ids is not None and len(ids) != want:
                raise ValueError(f"gpu_ids length {len(ids)} != gpus {want}")
            fields["gpu_ids"] = body["gpu_ids"]
        if "cron" in body or "at" in body:
            cron, at = body.get("cron"), body.get("at")
            validate_schedule(cron, at)
            fields["cron"] = cron
            fields["at"] = at
        if not fields:
            raise ValueError("empty patch")
        self.db.update_task(tid, **fields)
        return self._send(200, self.db.get_task(tid))

    def _delete_task(self, tid: int) -> None:
        """DELETE /api/tasks/{id}：仅终态或 paused 允许（与 CLI 同构）。"""
        task = self.db.get_task(tid)
        if task["status"] not in _TERMINAL and task["status"] != "paused":
            return self._err(
                409, f"task {tid} status={task['status']}, delete only "
                f"from terminal or paused")
        try:
            self.db.delete_task(tid)
        except ValueError as e:
            return self._err(409, str(e))
        return self._send(200, {"deleted": tid})

    def _create_task(self, body: dict) -> None:
        proj_ref = body.get("project_id") or body.get("project")
        if proj_ref is None:
            raise ValueError("missing 'project_id' or 'project'")
        proj = self.db.get_project(proj_ref)
        if proj is None:
            if isinstance(proj_ref, str) and not proj_ref.isdigit():
                pid = self.db.create_project(proj_ref)
            else:
                return self._err(404, f"project not found: {proj_ref}")
        else:
            pid = proj["id"]
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("missing/invalid 'config' (object)")
        name = body.get("name") or f"task-{int(time.time())}"
        model = body.get("model") or config.get("model")
        if not model:
            raise ValueError("missing 'model'")
        cron, at = body.get("cron"), body.get("at")
        # KI-10：cron 与 at 入库前统一校验（此前只校验 cron，at 漏检）
        validate_schedule(cron, at)
        # P5：多卡请求统一校验（gpus/gpu_ids 长度一致 + launch 模式）
        from .validation import validate_gpu_request
        gpus_n, gpu_ids_l = validate_gpu_request(
            config, body.get("gpus"), body.get("gpu_ids"))
        tid = self.db.create_task(
            pid, name, model, config,
            priority=body.get("priority", 100),
            depends_on=body.get("depends_on"),
            resume_from=body.get("resume_from"),
            resume_mode=body.get("resume_mode", "weights"),
            cron=cron, at=at,
            allow_parallel=1 if body.get("allow_parallel") else 0,
            gpus=gpus_n,
            gpu_ids=",".join(map(str, gpu_ids_l)) if gpu_ids_l else None,
            status="scheduled" if (cron or at) else "pending")
        return self._send(201, self.db.get_task(tid))

    def _samples(self, task_id: int) -> None:
        task = self.db.get_task(task_id)
        if task is None:
            return self._err(404, f"task not found: {task_id}")
        root = self._samples_dir(task)
        files = []
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                p = os.path.join(root, name)
                if os.path.isfile(p):
                    files.append({"name": name,
                                  "path": os.path.abspath(p),
                                  "size": os.path.getsize(p)})
        return self._send(200, {"task_id": task_id, "dir": root,
                                "files": files})

    def _sse(self) -> None:
        """每 2s 推一帧 {tasks, gpu, waiting}，直到客户端断开。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        logger.info("SSE client connected")
        try:
            while True:
                # P5：schedules 供前端展示 next_fire / 占用卡数
                sched = []
                for t in self.db.list_tasks():
                    if not (t.get("cron") or t.get("at")):
                        continue
                    kv = self.db.get_config_kv(t["id"])
                    nf = next_fire_of(t, kv)
                    sched.append({
                        "id": t["id"], "name": t["name"],
                        "schedule": t.get("cron") or t.get("at"),
                        "next_fire": nf.isoformat() if nf else None,
                        "last_fire": kv.get("_meta.last_fire"),
                        "status": t["status"], "gpus": t.get("gpus", 1)})
                payload = {
                    "ts": time.time(),
                    "tasks": self.db.list_tasks(),
                    "gpu": self.db.latest_gpu_snapshots(),
                    "schedules": sched,
                }
                frame = ("event: status\n"
                         "data: " + json.dumps(payload, ensure_ascii=False,
                                               default=str) + "\n\n")
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
                time.sleep(_SSE_INTERVAL)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("SSE client disconnected")
        except Exception as e:
            # KI-20：响应头已发出、流已开始，不能再走 _dispatch 的 500
            # 路径（会往流式 socket 再写一个响应头）；记录后静默关闭。
            logger.warning(f"SSE stream closed after error: {e}")
        return True  # 已处理标记：防止 _dispatch 在流结束后追加 404


class APIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, workspace: str, db: DB,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 static_root: Optional[str] = None):
        self.workspace = os.path.abspath(workspace)
        self.db = db
        # 默认 <ScheduledTrainer>/web/dist（orchestrator 包上一级）
        self.static_root = os.path.abspath(
            static_root or os.path.join(os.path.dirname(__file__), os.pardir,
                                        "web", "dist"))
        super().__init__((host, port), _Handler)


def create_server(workspace: str, db: DB, host: str = DEFAULT_HOST,
                  port: int = DEFAULT_PORT) -> APIServer:
    return APIServer(workspace, db, host=host, port=port)


def run_in_thread(server: APIServer):
    import threading
    t = threading.Thread(target=server.serve_forever,
                         kwargs={"poll_interval": 0.5},
                         name="api-server", daemon=True)
    t.start()
    return t


def main() -> None:  # python -m orchestrator.server --workspace ... --db ...
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="ScheduledTrainer API server")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--db", default=None)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()
    workspace = os.path.abspath(args.workspace)
    db = DB(args.db or os.path.join(workspace, "trainer.db"))
    server = create_server(workspace, db, host=args.host, port=args.port)
    logger.info(f"API on http://{args.host}:{args.port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
