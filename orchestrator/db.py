"""SQLite 数据层（P1）。所有 SQL 集中在本文件。

约定见 md/01-database.md：
- WAL 模式；锁等待统一由 PRAGMA busy_timeout=30000 控制（KI-17：
  此前 connect(timeout=30) 与 busy_timeout=5000 双源矛盾，现以
  busy_timeout 为唯一生效来源）；
- 连接按线程持有（每个线程一个 connection），同时登记进连接注册表，
  close() 关当前线程连接，close_all() 统一关闭全部（KI-17）；
- 状态迁移均通过 set_task_status 单点写入；
- materialize_config 是唯一配置出口（config_json ⊕ task_config_kv）。

扩展（记录于 agent/decisions.md）：
- task_config_kv 中 key 以 "_" 开头的条目为元数据（如 _meta.output），
  materialize_config 会跳过，不注入训练 config。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Optional

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# 允许通过 set_task_status(**fields) 一并更新的列（状态迁移审计字段）
_UPDATABLE_TASK_FIELDS = {
    "started_at", "finished_at", "error", "resume_from", "resume_mode",
    "restart_count", "priority", "cron", "source_file", "wandb_run_name",
    "allow_parallel", "gpus", "gpu_ids",
}


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def validate_name(name: Any, kind: str = "name") -> str:
    """项目/任务名安全校验（KI-11）。

    名字会拼进 samples/ 等文件系统路径，必须拒绝目录穿越：
    空名、`.`/`..`、含路径分隔符（/ 或 \\）一律拒绝。
    所有创建入口（create_project / create_task，覆盖 CLI、API、
    watcher 文件导入）统一走这里。
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"invalid {kind}: empty")
    if name in (".", "..") or ".." in name or "/" in name or "\\" in name:
        raise ValueError(
            f"invalid {kind} {name!r}: path separators / '..' not allowed")
    return name


class DB:
    """线程安全的薄封装：每线程一个 sqlite3 连接。"""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        # KI-17：连接注册表，供 close_all() 统一关闭跨线程连接
        self._conns: set = set()
        self._conns_lock = threading.Lock()
        self._init_schema()

    # ── 连接管理 ────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # KI-17：锁等待唯一来源是 PRAGMA busy_timeout=30000；
            # connect() 不再传 timeout（PRAGMA 一旦设置即覆盖 connect
            # 的默认 5s，双源易误导）。
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            with self._conns_lock:
                self._conns.add(conn)
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            ddl = f.read()
        conn = self._conn()
        conn.executescript(ddl)
        # P3/P5 迁移：旧库补列（CREATE TABLE IF NOT EXISTS 不会加列）
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        for col, ddl_col in (("at", "TEXT"),
                             ("gpus", "INTEGER NOT NULL DEFAULT 1"),
                             ("gpu_ids", "TEXT")):
            if col not in cols:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {ddl_col}")
        conn.commit()

    def close(self) -> None:
        """关闭当前线程的连接（KI-17：跨线程连接请用 close_all()）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            with self._conns_lock:
                self._conns.discard(conn)
            self._local.conn = None

    def close_all(self) -> None:
        """关闭注册表内全部连接（KI-17：进程退出/测试清理用）。"""
        with self._conns_lock:
            conns = list(self._conns)
            self._conns.clear()
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    # ── projects ────────────────────────────────────────────────
    def create_project(self, name: str, description: str = "",
                       default_model: Optional[str] = None,
                       tags=()) -> int:
        name = validate_name(name, "project name")
        conn = self._conn()
        cur = conn.execute(
            "INSERT OR IGNORE INTO projects(name, description, default_model, tags)"
            " VALUES (?, ?, ?, ?)",
            (name, description, default_model, json.dumps(list(tags))),
        )
        conn.commit()
        # KI-02 修复：IGNORE 生效时 lastrowid 是上一次插入的陈旧值，
        # 必须用 rowcount 判断是否真正插入；被忽略时 SELECT 取已有 id。
        if cur.rowcount == 1 and cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
        return row["id"]

    def get_project(self, id_or_name) -> Optional[dict]:
        conn = self._conn()
        if isinstance(id_or_name, int) or (isinstance(id_or_name, str) and id_or_name.isdigit()):
            row = conn.execute("SELECT * FROM projects WHERE id=?",
                               (int(id_or_name),)).fetchone()
        else:
            row = conn.execute("SELECT * FROM projects WHERE name=?",
                               (str(id_or_name),)).fetchone()
        return _row_to_dict(row) if row else None

    def list_projects(self) -> list:
        rows = self._conn().execute(
            "SELECT * FROM projects ORDER BY id").fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_project(self, project_id: int, **fields) -> None:
        """P3 API 用：PATCH 项目（description/status/default_model/tags）。"""
        allowed = {"description", "status", "default_model", "tags"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"field not updatable on project: {k}")
            sets.append(f"{k}=?")
            vals.append(json.dumps(v) if k == "tags" else v)
        if not sets:
            return
        vals.append(project_id)
        conn = self._conn()
        conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()

    # ── tasks ───────────────────────────────────────────────────
    def create_task(self, project_id: int, name: str, model: str,
                    config: dict, **kw) -> int:
        """kw: priority / depends_on / resume_from / resume_mode / cron /
        at / allow_parallel / gpus / gpu_ids / wandb_run_name / status /
        source_file"""
        name = validate_name(name, "task name")
        cols = ["project_id", "name", "model", "config_json"]
        vals = [project_id, name, model, json.dumps(config, ensure_ascii=False)]
        for k in ("priority", "depends_on", "resume_from", "resume_mode",
                  "cron", "at", "allow_parallel", "gpus", "gpu_ids",
                  "wandb_run_name", "status", "source_file"):
            if k in kw and kw[k] is not None:
                cols.append(k)
                vals.append(kw[k])
        conn = self._conn()
        cur = conn.execute(
            f"INSERT INTO tasks({', '.join(cols)})"
            f" VALUES ({', '.join('?' * len(cols))})", vals)
        conn.commit()
        return cur.lastrowid

    def get_task(self, task_id: int) -> Optional[dict]:
        row = self._conn().execute("SELECT * FROM tasks WHERE id=?",
                                   (task_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_tasks(self, project_id: Optional[int] = None,
                   status: Optional[str] = None) -> list:
        sql = "SELECT * FROM tasks"
        cond, vals = [], []
        if project_id is not None:
            cond.append("project_id=?")
            vals.append(project_id)
        if status is not None:
            cond.append("status=?")
            vals.append(status)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY priority, id"
        rows = self._conn().execute(sql, vals).fetchall()
        return [_row_to_dict(r) for r in rows]

    def set_task_status(self, task_id: int, status: str, **fields) -> None:
        """状态迁移单点入口。自动维护 started_at/finished_at。"""
        sets = ["status=?"]
        vals: list[Any] = [status]
        if status == "running" and "started_at" not in fields:
            fields["started_at"] = _now()
        if status in ("done", "failed", "cancelled") and "finished_at" not in fields:
            fields["finished_at"] = _now()
        for k, v in fields.items():
            if k not in _UPDATABLE_TASK_FIELDS:
                raise ValueError(f"field not updatable via set_task_status: {k}")
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(task_id)
        conn = self._conn()
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()

    def successors_of(self, task_id: int) -> list:
        rows = self._conn().execute(
            "SELECT * FROM tasks WHERE depends_on=? ORDER BY id",
            (task_id,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def next_runnable_tasks(self, statuses=("pending", "waiting_gpu")) -> list:
        """可派任务：pending / waiting_gpu（GPU Guard 等待补位），依赖已 done。

        P3 起不再按 cron IS NULL 过滤：cron/at 任务由 Scheduler 到点置
        pending，status 即"是否到点"的唯一语义。
        """
        marks = ",".join("?" * len(statuses))
        rows = self._conn().execute(
            f"""
            SELECT t.* FROM tasks t
            LEFT JOIN tasks p ON t.depends_on = p.id
            WHERE t.status IN ({marks})
              AND (t.depends_on IS NULL OR p.status = 'done')
            ORDER BY t.priority, t.id
            """, list(statuses)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_task(self, task_id: int, **fields) -> None:
        """P5：任务字段白名单更新（CLI/API 共用）。

        running 任务仅允许改 priority / allow_parallel；其余结构字段
        （name/depends_on/resume_*/cron/at/gpus/gpu_ids）running 期间拒绝。
        name 走 validate_name；cron/at 由调用方先经 scheduler.validate_schedule。
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id} not found")
        allowed = {"name", "priority", "depends_on", "resume_from",
                   "resume_mode", "cron", "at", "allow_parallel",
                   "gpus", "gpu_ids"}
        structural = allowed - {"priority", "allow_parallel"}
        if task["status"] == "running" and structural.intersection(fields):
            raise ValueError(
                f"task {task_id} is running; structural fields "
                f"{sorted(structural.intersection(fields))} cannot change")
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"field not updatable on task: {k}")
            if k == "name":
                v = validate_name(v, "task name")
            sets.append(f"{k}=?")
            vals.append(v)
        if not sets:
            return
        vals.append(task_id)
        conn = self._conn()
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()

    def delete_task(self, task_id: int, force: bool = False) -> None:
        """P5：硬删除任务（hooks/kv/heartbeats 子行一并删除，单事务）。

        running 任务默认拒绝；force=True 时仍删除（worker 进程交由运维
        处理，日志已记录）。有后继任务（depends_on 指向本任务）时拒绝。
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id} not found")
        if task["status"] == "running" and not force:
            raise ValueError(
                f"task {task_id} is running; cancel it first (or --force)")
        conn = self._conn()
        conn.execute("BEGIN")
        try:
            deps = conn.execute(
                "SELECT id FROM tasks WHERE depends_on=?", (task_id,)).fetchall()
            if deps:
                raise ValueError(
                    f"task {task_id} has dependents: "
                    f"{[d['id'] for d in deps]}; remove depends_on first")
            for table in ("hooks", "task_config_kv", "heartbeats"):
                conn.execute(f"DELETE FROM {table} WHERE task_id=?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def delete_project(self, project_id: int) -> None:
        """P5：删除项目；仍有任务引用时拒绝（先删/迁任务）。"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id FROM tasks WHERE project_id=?", (project_id,)).fetchall()
        if rows:
            raise ValueError(
                f"project {project_id} still has tasks: "
                f"{[r['id'] for r in rows]}; delete them first")
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        conn.commit()

    def delete_prompt(self, prompt_id: int) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM prompts WHERE id=?", (prompt_id,))
        conn.commit()

    def materialize_config(self, task_id: int) -> dict:
        """config_json ⊕ task_config_kv（点路径合并）。

        以 "_" 开头的 kv key 是元数据，不注入 config。
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id} not found")
        config = json.loads(task["config_json"])
        config.setdefault("model", task["model"])
        if task.get("resume_from"):
            resume = config.setdefault("resume", {})
            resume.setdefault("checkpoint", task["resume_from"])
            resume.setdefault("full", task.get("resume_mode") == "full")
        for key, value in self.get_config_kv(task_id, raw=True).items():
            if key.startswith("_"):
                continue
            _set_dotted(config, key, json.loads(value))
        return config

    # ── hooks ───────────────────────────────────────────────────
    def enqueue_hook(self, task_id: int, type_: str, payload: dict) -> int:
        # KI-17：参数名 type_，避免遮蔽内置 type
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO hooks(task_id, type, payload) VALUES (?, ?, ?)",
            (task_id, type_, json.dumps(payload, ensure_ascii=False)))
        conn.commit()
        return cur.lastrowid

    def fetch_queued_hooks(self, task_id: int) -> list:
        rows = self._conn().execute(
            "SELECT * FROM hooks WHERE task_id=? AND status='queued' ORDER BY id",
            (task_id,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def ack_hook(self, hook_id: int) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE hooks SET status='acked', acked_at=datetime('now') WHERE id=?",
            (hook_id,))
        conn.commit()

    def requeue_stale_acked_hooks(self, task_id: int) -> int:
        """KI-19：worker 启动时回收本任务 acked 但未 done 的 hooks。

        worker 在 ack 之后崩溃会留下永远卡 acked 的 hook；下一次（重）启动
        时重置回 queued 重新消费。返回回收条数。
        """
        conn = self._conn()
        cur = conn.execute(
            "UPDATE hooks SET status='queued', acked_at=NULL"
            " WHERE task_id=? AND status='acked'", (task_id,))
        conn.commit()
        return cur.rowcount

    def finish_hook(self, hook_id: int, result: Optional[str] = None,
                    failed: bool = False) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE hooks SET status=?, result=?, done_at=datetime('now') WHERE id=?",
            ("failed" if failed else "done", result, hook_id))
        conn.commit()

    def list_hooks(self, task_id: Optional[int] = None) -> list:
        sql = "SELECT * FROM hooks"
        vals: list[Any] = []
        if task_id is not None:
            sql += " WHERE task_id=?"
            vals.append(task_id)
        sql += " ORDER BY id"
        rows = self._conn().execute(sql, vals).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ── prompts ─────────────────────────────────────────────────
    def add_prompt(self, project_id: Optional[int], text: str,
                   tag: Optional[str] = None, negative: Optional[str] = None,
                   meta: Optional[dict] = None) -> int:
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO prompts(project_id, tag, text, negative, meta)"
            " VALUES (?, ?, ?, ?, ?)",
            (project_id, tag, text, negative,
             json.dumps(meta or {}, ensure_ascii=False)))
        conn.commit()
        return cur.lastrowid

    def list_prompts(self, project_id: Optional[int] = None,
                     tag: Optional[str] = None) -> list:
        sql, vals = "SELECT * FROM prompts", []
        cond = []
        if project_id is not None:
            cond.append("project_id=?")
            vals.append(project_id)
        if tag is not None:
            cond.append("tag=?")
            vals.append(tag)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY id"
        rows = self._conn().execute(sql, vals).fetchall()
        return [_row_to_dict(r) for r in rows]

    def prompts_by_tag(self, project_id: Optional[int], tag: str) -> list:
        """按 tag 取提示词；project_id 为 None 时不限项目。"""
        if project_id is None:
            rows = self._conn().execute(
                "SELECT * FROM prompts WHERE tag=? ORDER BY id", (tag,)).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM prompts WHERE project_id=? AND tag=? ORDER BY id",
                (project_id, tag)).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ── config kv ───────────────────────────────────────────────
    def set_config_kv(self, task_id: int, key: str, value: Any) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO task_config_kv(task_id, key, value, updated_at)"
            " VALUES (?, ?, ?, datetime('now'))"
            " ON CONFLICT(task_id, key) DO UPDATE SET"
            " value=excluded.value, updated_at=excluded.updated_at",
            (task_id, key, json.dumps(value, ensure_ascii=False)))
        conn.commit()

    def get_config_kv(self, task_id: int, raw: bool = False) -> dict:
        rows = self._conn().execute(
            "SELECT key, value FROM task_config_kv WHERE task_id=?",
            (task_id,)).fetchall()
        if raw:
            return {r["key"]: r["value"] for r in rows}
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ── heartbeats / gpu ────────────────────────────────────────
    def heartbeat(self, task_id: int, step: int, loss: Optional[float] = None,
                  lr: Optional[float] = None,
                  vram_mb: Optional[int] = None) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO heartbeats(task_id, step, loss, lr, vram_mb)"
            " VALUES (?, ?, ?, ?, ?)",
            (task_id, step, loss, lr, vram_mb))
        conn.commit()

    def last_heartbeat(self, task_id: int) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM heartbeats WHERE task_id=? ORDER BY ts DESC, rowid DESC"
            " LIMIT 1", (task_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def heartbeats_since(self, task_id: int, since: Optional[str] = None,
                         limit: int = 1000) -> list:
        """心跳序列（按时间升序）；since 为 ts 下界（不含）。"""
        if since:
            rows = self._conn().execute(
                "SELECT * FROM heartbeats WHERE task_id=? AND ts>?"
                " ORDER BY ts, rowid LIMIT ?",
                (task_id, since, limit)).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM heartbeats WHERE task_id=?"
                " ORDER BY ts, rowid LIMIT ?", (task_id, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def gpu_snapshot(self, snapshots: list) -> None:
        conn = self._conn()
        conn.executemany(
            "INSERT INTO gpu_snapshots(gpu_index, total_mb, used_mb, free_mb,"
            " util_pct) VALUES (?, ?, ?, ?, ?)",
            [(s.get("gpu_index", 0), s["total_mb"], s["used_mb"], s["free_mb"],
              s.get("util_pct")) for s in snapshots])
        conn.commit()

    def latest_gpu_snapshots(self) -> list:
        rows = self._conn().execute(
            """
            SELECT g.* FROM gpu_snapshots g
            JOIN (SELECT gpu_index, MAX(id) AS max_id FROM gpu_snapshots
                  GROUP BY gpu_index) m
              ON g.gpu_index = m.gpu_index AND g.id = m.max_id
            ORDER BY g.gpu_index
            """).fetchall()
        return [_row_to_dict(r) for r in rows]

    def running_task_gpus(self) -> dict:
        """P5：DB 视角的卡占用 {task_id: [gpu,...]}（仅 running 任务的
        _meta.gpu_index，兼容旧标量值）。供 gpu status / server occupancy。"""
        rows = self._conn().execute(
            """
            SELECT t.id AS tid, kv.value AS value FROM tasks t
            JOIN task_config_kv kv ON kv.task_id = t.id
             AND kv.key = '_meta.gpu_index'
            WHERE t.status = 'running'
            """).fetchall()
        out = {}
        for r in rows:
            try:
                v = json.loads(r["value"])
            except Exception:
                continue
            if isinstance(v, int):
                v = [v]
            elif not isinstance(v, list):
                continue
            out[r["tid"]] = v
        return out

    def list_gpu_snapshots(self, limit: int = 100) -> list:
        rows = self._conn().execute(
            "SELECT * FROM gpu_snapshots ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ── retention 清理（KI-17；main.py 周期调用）────────────────
    def prune_heartbeats(self, keep_days: int = 7) -> int:
        """删除 keep_days 天前的心跳行，返回删除条数。"""
        conn = self._conn()
        cur = conn.execute(
            "DELETE FROM heartbeats"
            " WHERE ts < datetime('now', ?)",
            (f"-{int(keep_days)} days",))
        conn.commit()
        return cur.rowcount

    def prune_gpu_snapshots(self, keep_rows: int = 10000) -> int:
        """每个 gpu_index 仅保留最新 keep_rows 行快照，返回删除条数。"""
        conn = self._conn()
        cur = conn.execute(
            """
            DELETE FROM gpu_snapshots WHERE id NOT IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY gpu_index ORDER BY id DESC) AS rn
                    FROM gpu_snapshots
                ) WHERE rn <= ?
            )""", (int(keep_rows),))
        conn.commit()
        return cur.rowcount


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _set_dotted(config: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value
