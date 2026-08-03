"""CLI（P1 最小集；P5 起：任务 CRUD + 多卡 + schedule/prompt 命令组）。

用法（在 ScheduledTrainer 根目录下）：
    python -m orchestrator.cli project create <name> [--model M] [--desc D]
    python -m orchestrator.cli project list|show|update|archive|delete ...
    python -m orchestrator.cli submit <json_file>     # 复制进 inbox/
    python -m orchestrator.cli task|job create|list|show|update|pause|unpause|retry|delete|logs
    python -m orchestrator.cli schedule list|add|set|pause|unpause|preview
    python -m orchestrator.cli prompt add|list|delete
    python -m orchestrator.cli list [--project P] [--status S]
    python -m orchestrator.cli cancel <task_id>
    python -m orchestrator.cli hook <task_id> <type> [--payload JSON] [flags]
    python -m orchestrator.cli hooks [--task-id N]
    python -m orchestrator.cli resume <task_id>
    python -m orchestrator.cli config set <task_id> <key> <value>
    python -m orchestrator.cli gpu status

全局参数：--db（默认 workspace/trainer.db）、--workspace（默认 workspace）、
--json（列表/展示命令输出机器可读 JSON）。
退出码约定：0 ok / 1 未找到或运行时错误 / 2 用法或校验错误。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

from .db import DB

_TERMINAL = {"done", "failed", "cancelled"}


def _add_global(p: argparse.ArgumentParser, suppress: bool = False) -> None:
    # 子命令上的默认值用 SUPPRESS，避免覆盖顶层解析到的全局参数
    d = argparse.SUPPRESS if suppress else None
    p.add_argument("--workspace", default=d or "workspace")
    p.add_argument("--db", default=d)
    p.add_argument("--json", action="store_true", default=d or False,
                   help="machine-readable JSON output")


def _open_db(args) -> DB:
    db_path = args.db or os.path.join(os.path.abspath(args.workspace),
                                      "trainer.db")
    return DB(db_path)


def _print_table(rows: list, cols: list) -> None:
    if not rows:
        print("(empty)")
        return
    widths = [max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows))
              for c in cols]
    header = "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w)
                        for c, w in zip(cols, widths)))


def _emit(args, data, cols=None) -> None:
    """--json 时输出 JSON；否则按 cols 表格或键值行打印。"""
    if getattr(args, "json", False):
        print(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    elif cols is not None:
        _print_table(data, cols)
    else:
        for k, v in data.items():
            print(f"{k}: {v}")


# ── project ───────────────────────────────────────────────────
def cmd_project(args) -> int:
    db = _open_db(args)
    act = args.project_action
    if act == "create":
        pid = db.create_project(args.name, description=args.desc or "",
                                default_model=args.model)
        print(f"project id={pid} name={args.name}")
    elif act == "list":
        _emit(args, db.list_projects(),
              ["id", "name", "status", "default_model", "created_at"])
    elif act == "show":
        proj = db.get_project(args.target)
        if proj is None:
            print(f"project not found: {args.target}", file=sys.stderr)
            return 1
        counts = {}
        for t in db.list_tasks(project_id=proj["id"]):
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        if args.json:
            _emit(args, {"project": proj, "task_counts": counts})
        else:
            for k, v in proj.items():
                print(f"{k}: {v}")
            print("task_counts:", json.dumps(counts, ensure_ascii=False))
    elif act == "update":
        proj = db.get_project(args.target)
        if proj is None:
            print(f"project not found: {args.target}", file=sys.stderr)
            return 1
        fields = {}
        if args.desc is not None:
            fields["description"] = args.desc
        if args.model is not None:
            fields["default_model"] = args.model
        if args.tags is not None:
            fields["tags"] = [s.strip() for s in args.tags.split(",")
                              if s.strip()]
        db.update_project(proj["id"], **fields)
        print(f"project {proj['id']} updated: {sorted(fields)}")
    elif act == "archive":
        proj = db.get_project(args.target)
        if proj is None:
            print(f"project not found: {args.target}", file=sys.stderr)
            return 1
        non_terminal = [t for t in db.list_tasks(project_id=proj["id"])
                        if t["status"] not in _TERMINAL]
        if non_terminal:
            print(f"project {proj['id']} has non-terminal tasks: "
                  f"{[t['id'] for t in non_terminal]}", file=sys.stderr)
            return 1
        db.update_project(proj["id"], status="archived")
        print(f"project {proj['id']} archived")
    elif act == "delete":
        proj = db.get_project(args.target)
        if proj is None:
            print(f"project not found: {args.target}", file=sys.stderr)
            return 1
        if not args.yes:
            print("refusing without --yes (deletes all tasks + history)",
                  file=sys.stderr)
            return 2
        tasks = db.list_tasks(project_id=proj["id"])
        non_terminal = [t for t in tasks if t["status"] not in _TERMINAL]
        if non_terminal:
            print(f"project {proj['id']} has non-terminal tasks: "
                  f"{[t['id'] for t in non_terminal]}", file=sys.stderr)
            return 1
        # 先删任务（叶子优先：逆 id 序，避免 depends_on 依赖拦截），再删项目
        for t in sorted(tasks, key=lambda x: -x["id"]):
            db.delete_task(t["id"])
        db.delete_project(proj["id"])
        print(f"project {proj['id']} deleted ({len(tasks)} tasks)")
    return 0


# ── task / job ─────────────────────────────────────────────────
def _create_task_row(db, project_ref, name, model, config,
                     cron=None, at=None, priority=100,
                     allow_parallel=False, gpus=None, gpu_ids=None) -> int:
    """task create / schedule add 共用入库路径（cron/at/多卡统一校验）。"""
    from .scheduler import validate_schedule
    from .validation import validate_gpu_request

    proj = db.get_project(project_ref)
    project_id = proj["id"] if proj else db.create_project(str(project_ref))
    if not model:
        raise ValueError("missing model (and config has no model)")
    validate_schedule(cron, at)
    gpus_n, gpu_ids_l = validate_gpu_request(config, gpus, gpu_ids)
    return db.create_task(
        project_id, name, model, config,
        priority=priority,
        cron=cron, at=at,
        allow_parallel=1 if allow_parallel else 0,
        gpus=gpus_n,
        gpu_ids=",".join(map(str, gpu_ids_l)) if gpu_ids_l else None,
        status="scheduled" if (cron or at) else "pending")


def cmd_task_create(args) -> int:
    """直接建任务（等价 inbox task_*.json，但立即入库）。"""
    db = _open_db(args)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {"model": args.model, "training": {}, "output": {}}
    try:
        tid = _create_task_row(
            db, args.project, args.name, args.model or config.get("model"),
            config,
            cron=args.cron, at=args.at,
            priority=args.priority,
            allow_parallel=args.allow_parallel,
            gpus=args.gpus, gpu_ids=args.gpu_ids)
    except (ValueError, KeyError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2
    task = db.get_task(tid)
    print(f"task id={tid} name={args.name} status={task['status']} "
          f"cron={task['cron']} at={task['at']} gpus={task['gpus']} "
          f"gpu_ids={task['gpu_ids']} "
          f"allow_parallel={task['allow_parallel']}")
    return 0


def cmd_list(args) -> int:
    db = _open_db(args)
    project_id = None
    if args.project:
        proj = db.get_project(args.project)
        if proj is None:
            print(f"project not found: {args.project}", file=sys.stderr)
            return 1
        project_id = proj["id"]
    rows = []
    for t in db.list_tasks(project_id=project_id, status=args.status):
        rows.append({"id": t["id"], "project_id": t["project_id"],
                     "name": t["name"], "model": t["model"],
                     "status": t["status"], "priority": t["priority"],
                     "gpus": t.get("gpus", 1),
                     "cron": t.get("cron") or "",
                     "at": t.get("at") or "",
                     "depends_on": t.get("depends_on") or "",
                     "resume_from": t.get("resume_from") or "",
                     "error": t.get("error") or ""})
    _emit(args, rows, ["id", "project_id", "name", "model", "status",
                       "priority", "gpus", "cron", "at", "depends_on",
                       "resume_from", "error"])
    return 0


def cmd_task_show(args) -> int:
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    from .scheduler import next_fire_of

    kv = db.get_config_kv(args.task_id)
    config = db.materialize_config(args.task_id)
    nf = next_fire_of(task, kv)
    if args.json:
        _emit(args, {"task": task, "config_kv": kv, "config": config,
                     "next_fire": nf.isoformat() if nf else None})
        return 0
    for k, v in task.items():
        if k == "config_json":
            continue
        print(f"{k}: {v}")
    bs = config.get("training", {}).get("batch_size")
    if bs:
        print(f"effective_bs: {int(bs) * int(task.get('gpus') or 1)} "
              f"(batch_size * gpus)")
    print(f"next_fire: {nf.isoformat() if nf else '-'}")
    print("config_kv:", json.dumps(kv, ensure_ascii=False, default=str))
    return 0


def cmd_task_update(args) -> int:
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    fields = {}
    if args.name is not None:
        fields["name"] = args.name
    if args.priority is not None:
        fields["priority"] = args.priority
    if args.allow_parallel is not None:
        fields["allow_parallel"] = 1 if args.allow_parallel else 0
    if args.depends_on is not None:
        fields["depends_on"] = args.depends_on
    if args.resume_from is not None:
        fields["resume_from"] = args.resume_from
    if args.gpus is not None:
        fields["gpus"] = args.gpus
    if args.gpu_ids is not None:
        from .validation import parse_gpu_ids

        ids = parse_gpu_ids(args.gpu_ids)
        want = args.gpus if args.gpus is not None else task.get("gpus", 1)
        if ids is not None and len(ids) != want:
            print(f"gpu_ids length {len(ids)} != gpus {want}",
                  file=sys.stderr)
            return 2
        fields["gpu_ids"] = args.gpu_ids
    if not fields:
        print("nothing to update", file=sys.stderr)
        return 2
    try:
        db.update_task(args.task_id, **fields)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"task {args.task_id} updated: {sorted(fields)}")
    return 0


def cmd_task_pause(args) -> int:
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    if task["status"] not in ("scheduled", "pending", "waiting_gpu"):
        print(f"task {args.task_id} status={task['status']}; pause only "
              f"from scheduled/pending/waiting_gpu", file=sys.stderr)
        return 1
    db.set_task_status(args.task_id, "paused")
    print(f"task {args.task_id} paused")
    return 0


def cmd_task_unpause(args) -> int:
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    if task["status"] != "paused":
        print(f"task {args.task_id} status={task['status']}; "
              f"unpause only from paused", file=sys.stderr)
        return 1
    new = "scheduled" if (task.get("cron") or task.get("at")) else "pending"
    db.set_task_status(args.task_id, new)
    print(f"task {args.task_id} unpaused -> {new}")
    return 0


def cmd_task_retry(args) -> int:
    """failed → pending（清 error，保留 restart_count）。"""
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    if task["status"] != "failed":
        print(f"task {args.task_id} status={task['status']}; "
              f"retry only from failed", file=sys.stderr)
        return 1
    db.set_task_status(args.task_id, "pending", error=None)
    print(f"task {args.task_id} retried -> pending (error cleared, "
          f"restart_count={task.get('restart_count')})")
    return 0


def cmd_task_delete(args) -> int:
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    if not args.yes:
        print("refusing without --yes (deletes hooks/kv/heartbeats)",
              file=sys.stderr)
        return 2
    if task["status"] not in _TERMINAL and task["status"] != "paused":
        print(f"task {args.task_id} status={task['status']}; delete only "
              f"from terminal or paused", file=sys.stderr)
        return 1
    try:
        db.delete_task(args.task_id)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"task {args.task_id} deleted")
    return 0


_RUN_LOG_RE = re.compile(r"\.run_(\d+)\.log$")


def _latest_run_log(log_dir: str, task_id: int) -> Optional[str]:
    try:
        names = os.listdir(log_dir)
    except FileNotFoundError:
        return None
    files = [os.path.join(log_dir, n) for n in names
             if n.startswith(f"task_{task_id}.run_") and n.endswith(".log")]
    if not files:
        return None
    return max(files, key=lambda p: int(_RUN_LOG_RE.search(p).group(1)))


def cmd_task_logs(args) -> int:
    """读取 workspace/logs/task_<id>.run_<restart>.log（默认最新 restart）。"""
    log_dir = os.path.join(os.path.abspath(args.workspace), "logs")
    path = _latest_run_log(log_dir, args.task_id)
    if path is None:
        print(f"no log file for task {args.task_id} "
              f"(expected {log_dir}/task_{args.task_id}.run_*.log)",
              file=sys.stderr)
        return 1
    with open(path, "rb") as f:
        if args.follow:
            f.seek(0, os.SEEK_END)
            print(f"=== tailing {path} (Ctrl-C to stop) ===", file=sys.stderr)
            try:
                while True:
                    data = f.read()
                    if data:
                        sys.stdout.write(
                            data.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        elif args.tail:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))  # 粗定位窗口，再精确到行
            lines = f.read().decode("utf-8", errors="replace").splitlines()
            print("\n".join(lines[-args.tail:]))
        else:
            f.seek(0)
            sys.stdout.write(f.read().decode("utf-8", errors="replace"))
    return 0


_TASK_HANDLERS = {
    "create": cmd_task_create,
    "list": cmd_list,
    "show": cmd_task_show,
    "update": cmd_task_update,
    "pause": cmd_task_pause,
    "unpause": cmd_task_unpause,
    "retry": cmd_task_retry,
    "delete": cmd_task_delete,
    "logs": cmd_task_logs,
}


def cmd_task(args) -> int:
    handler = _TASK_HANDLERS.get(args.task_action)
    return handler(args) if handler else 2


# ── schedule（tasks.cron/at 的一等视图）─────────────────────────
def cmd_schedule(args) -> int:
    db = _open_db(args)
    act = args.schedule_action
    if act == "list":
        return _schedule_list(args, db)
    if act == "add":
        return _schedule_add(args, db)
    if act == "set":
        return _schedule_set(args, db)
    if act == "pause":
        return cmd_task_pause(args)
    if act == "unpause":
        return cmd_task_unpause(args)
    if act == "preview":
        return _schedule_preview(args)
    return 2


def _schedule_list(args, db) -> int:
    project_id = None
    if args.project:
        proj = db.get_project(args.project)
        if proj is None:
            print(f"project not found: {args.project}", file=sys.stderr)
            return 1
        project_id = proj["id"]
    from .scheduler import next_fire_of

    rows = []
    for t in db.list_tasks(project_id=project_id):
        if not (t.get("cron") or t.get("at")):
            continue
        kv = db.get_config_kv(t["id"])
        nf = next_fire_of(t, kv)
        rows.append({"id": t["id"], "name": t["name"],
                     "schedule": t.get("cron") or t.get("at"),
                     "next_fire": nf.isoformat() if nf else "-",
                     "last_fire": kv.get("_meta.last_fire") or "-",
                     "status": t["status"],
                     "gpus": t.get("gpus", 1)})
    _emit(args, rows, ["id", "name", "schedule", "next_fire",
                       "last_fire", "status", "gpus"])
    return 0


def _schedule_add(args, db) -> int:
    if bool(args.cron) == bool(args.at):
        print("need exactly one of --cron EXPR / --at ISO", file=sys.stderr)
        return 2
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {"model": args.model, "training": {}, "output": {}}
    try:
        tid = _create_task_row(
            db, args.project, args.name, args.model or config.get("model"),
            config,
            cron=args.cron, at=args.at,
            priority=args.priority,
            allow_parallel=args.allow_parallel,
            gpus=args.gpus, gpu_ids=args.gpu_ids)
    except (ValueError, KeyError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2
    task = db.get_task(tid)
    print(f"schedule task id={tid} name={args.name} cron={task['cron']} "
          f"at={task['at']} status={task['status']} gpus={task['gpus']}")
    return 0


def _schedule_set(args, db) -> int:
    from .scheduler import validate_schedule

    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    given = [x for x in (args.cron, args.at) if x is not None]
    if args.none:
        given.append(True)
    if len(given) != 1:
        print("give exactly one of --cron EXPR / --at ISO / --none",
              file=sys.stderr)
        return 2
    cron = args.cron if args.cron is not None else None
    at = args.at if args.at is not None else None
    try:
        validate_schedule(cron, at)
    except ValueError as e:
        print(f"invalid schedule: {e}", file=sys.stderr)
        return 2
    try:
        db.update_task(args.task_id, cron=cron, at=at)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    # 清 last_fire：重设后 at 任务可再次触发
    db.set_config_kv(args.task_id, "_meta.last_fire", None)
    cur = db.get_task(args.task_id)
    if cron and cur["status"] in _TERMINAL:
        # 终态任务重新获得 cron → 置回 scheduled（熔断 _meta.rearm_blocked 除外）
        kv = db.get_config_kv(args.task_id)
        if kv.get("_meta.rearm_blocked"):
            print(f"task {args.task_id} re-arm blocked by "
                  f"_meta.rearm_blocked: {kv['_meta.rearm_blocked']}")
        else:
            db.set_task_status(args.task_id, "scheduled")
            print(f"task {args.task_id} re-armed -> scheduled")
    elif not cron and not at and cur["status"] == "scheduled":
        db.set_task_status(args.task_id, "pending")
        print(f"task {args.task_id} schedule cleared -> pending")
    else:
        print(f"task {args.task_id} schedule set: cron={cron!r} at={at!r}")
    return 0


def _schedule_preview(args) -> int:
    from .scheduler import CronExpr

    try:
        expr = CronExpr(args.cron_expr)
    except ValueError as e:
        print(f"invalid cron: {e}", file=sys.stderr)
        return 2
    now = datetime.now()
    rows = []
    for _ in range(args.count):
        nf = expr.next_after(now)
        if nf is None:
            break
        rows.append({"fire": nf.isoformat(timespec="seconds")})
        now = nf + timedelta(minutes=1)
    if not rows:
        print("expression does not match any time within horizon "
              "(check fields)", file=sys.stderr)
        return 1
    _emit(args, rows, ["fire"])
    return 0


# ── prompt ─────────────────────────────────────────────────────
def cmd_prompt(args) -> int:
    db = _open_db(args)
    act = args.prompt_action
    if act == "add":
        proj = db.get_project(args.project)
        project_id = proj["id"] if proj else db.create_project(str(args.project))
        pid = db.add_prompt(project_id, args.text, tag=args.tag,
                            negative=args.negative)
        print(f"prompt id={pid} project={project_id} tag={args.tag}")
    elif act == "list":
        project_id = None
        if args.project:
            proj = db.get_project(args.project)
            if proj is None:
                print(f"project not found: {args.project}", file=sys.stderr)
                return 1
            project_id = proj["id"]
        _emit(args, db.list_prompts(project_id=project_id, tag=args.tag),
              ["id", "project_id", "tag", "text", "negative", "created_at"])
    elif act == "delete":
        if not args.yes:
            print("refusing without --yes", file=sys.stderr)
            return 2
        db.delete_prompt(args.prompt_id)
        print(f"prompt {args.prompt_id} deleted")
    return 0


# ── 其余既有命令 ───────────────────────────────────────────────
def cmd_submit(args) -> int:
    src = os.path.abspath(args.file)
    if not os.path.isfile(src):
        print(f"file not found: {src}", file=sys.stderr)
        return 1
    inbox = os.path.join(os.path.abspath(args.workspace), "inbox")
    os.makedirs(inbox, exist_ok=True)
    dest = os.path.join(inbox, os.path.basename(src))
    shutil.copy2(src, dest)
    print(f"submitted -> {dest}")
    return 0


def cmd_cancel(args) -> int:
    db = _open_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    if task["status"] in ("pending", "scheduled", "waiting_gpu"):
        db.set_task_status(args.task_id, "cancelled")
        print(f"task {args.task_id} cancelled")
    else:
        print(f"task {args.task_id} status={task['status']}, cannot cancel "
              f"(only pending/scheduled/waiting_gpu)", file=sys.stderr)
        return 1
    return 0


def cmd_hook(args) -> int:
    """投递 hook 到 hooks 表（等价于前端按钮 / inbox cmd 文件）。"""
    from . import hooks as hook_proto

    payload = {}
    if args.payload:
        try:
            import json
            payload.update(json.loads(args.payload))
        except Exception as e:
            print(f"invalid --payload JSON: {e}", file=sys.stderr)
            return 2
    # 便捷 flag → payload 键
    flag_map = {
        "name": args.name, "path": args.path, "weights_path": args.weights,
        "tag": args.tag, "prompts_path": args.prompts,
        "n": args.n, "steps": args.steps, "seed": args.seed,
    }
    for k, v in flag_map.items():
        if v is not None:
            payload[k] = v
    if args.with_optimizer:
        payload["with_optimizer"] = True
    if args.reset_optim:
        payload["reset_optimizer"] = True
    for kv in args.set or []:
        if "=" not in kv:
            print(f"invalid --set '{kv}', expect key=value", file=sys.stderr)
            return 2
        key, raw = kv.split("=", 1)
        import json
        try:
            payload[key] = json.loads(raw)
        except json.JSONDecodeError:
            payload[key] = raw

    db = _open_db(args)
    try:
        hid = hook_proto.enqueue(db, args.task_id, args.hook_type, payload)
    except hook_proto.HookError as e:
        print(str(e), file=sys.stderr)
        return 1
    import json
    print(f"hook id={hid} task={args.task_id} type={args.hook_type} "
          f"payload={json.dumps(payload, ensure_ascii=False)}")
    return 0


def cmd_hooks(args) -> int:
    """列出 hooks 表（默认按任务过滤），观察 queued→acked→done 流转。"""
    db = _open_db(args)
    _emit(args, db.list_hooks(task_id=args.task_id),
          ["id", "task_id", "type", "status", "payload", "result",
           "created_at", "done_at"])
    return 0


def cmd_resume(args) -> int:
    """重启 suspended 任务（restart_count+1，wandb run name 续跑）。"""
    from . import hooks as hook_proto

    db = _open_db(args)
    try:
        info = hook_proto.resume_task(db, args.task_id)
    except hook_proto.HookError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"task {args.task_id} resumed -> pending: "
          f"restart_count={info['restart_count']} "
          f"wandb_run_name={info['wandb_run_name']} "
          f"resumed_from_run={info['resumed_from_run']} "
          f"resume_from={info['resume_from']}")
    return 0


def cmd_config_set(args) -> int:
    """config set <task_id> <key> <value>：写 task_config_kv；
    任务 running 时通过统一入口追加 patch_config hook（KI-04）。

    value 先尝试 json.loads（支持数字/布尔/null/数组/对象），失败按字符串。
    """
    import json
    from . import hooks as hook_proto

    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value

    db = _open_db(args)
    if db.get_task(args.task_id) is None:
        print(f"task not found: {args.task_id}", file=sys.stderr)
        return 1
    try:
        hid = hook_proto.set_config_and_notify(db, args.task_id,
                                               args.key, value)
    except hook_proto.HookError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"task {args.task_id} config[{args.key}]={json.dumps(value, ensure_ascii=False)}"
          + (f" patch_config hook id={hid}" if hid else " (kv written, no hook)"))
    return 0


def cmd_gpu(args) -> int:
    """通过 nvidia-smi 采集快照写入 gpu_snapshots 并打印（标准库 subprocess）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("nvidia-smi not found; GPU info unavailable")
        return 1
    if out.returncode != 0:
        print(f"nvidia-smi failed: {out.stderr.strip()}", file=sys.stderr)
        return 1
    # KI-15：与 gpu_guard 共用容错解析（[N/A] 字段不再崩溃）
    from .gpu_guard import parse_nvidia_smi_csv

    snapshots = parse_nvidia_smi_csv(out.stdout)
    db = _open_db(args)
    db.gpu_snapshot(snapshots)
    # P5：occupancy —— running 任务 _meta.gpu_index 反查每张卡的归属
    task_of_gpu = {}
    for tid, gpus in db.running_task_gpus().items():
        for g in gpus:
            task_of_gpu[int(g)] = tid
    rows = [{"gpu_index": s["gpu_index"], "total_mb": s["total_mb"],
             "used_mb": s["used_mb"], "free_mb": s["free_mb"],
             "util_pct": s["util_pct"],
             "task_id": task_of_gpu.get(int(s["gpu_index"]), "")}
            for s in snapshots]
    _emit(args, rows, ["gpu_index", "total_mb", "used_mb", "free_mb",
                       "util_pct", "task_id"])
    return 0


# ── parser 装配 ────────────────────────────────────────────────
def _add_task_group(sub, name: str, help_text: str) -> None:
    """task 命令组（job 是其别名，共用同一组处理器）。"""
    p = sub.add_parser(name, help=help_text)
    _add_global(p, suppress=True)
    tsub = p.add_subparsers(dest="task_action", required=True)

    tc = tsub.add_parser("create")
    _add_global(tc, suppress=True)
    tc.add_argument("--project", required=True, help="项目名或 id")
    tc.add_argument("--name", required=True)
    tc.add_argument("--model", default=None)
    tc.add_argument("--config", default=None, help="训练 config JSON 文件")
    tc.add_argument("--priority", type=int, default=100)
    tc.add_argument("--cron", default=None, help="五段 cron：分 时 日 月 周")
    tc.add_argument("--at", default=None, help="一次性触发 ISO 时间")
    tc.add_argument("--allow-parallel", action="store_true",
                    help="允许与其他任务并行（GPU Guard 准入）")
    tc.add_argument("--gpus", type=int, default=None,
                    help="占用的 GPU 卡数（默认 1）")
    tc.add_argument("--gpu-ids", default=None,
                    help='钉卡索引 CSV，如 "0,1"（长度须等于 --gpus）')

    tl = tsub.add_parser("list")
    _add_global(tl, suppress=True)
    tl.add_argument("--project", default=None)
    tl.add_argument("--status", default=None)

    ts = tsub.add_parser("show")
    _add_global(ts, suppress=True)
    ts.add_argument("task_id", type=int)

    tu = tsub.add_parser("update")
    _add_global(tu, suppress=True)
    tu.add_argument("task_id", type=int)
    tu.add_argument("--name", default=None)
    tu.add_argument("--priority", type=int, default=None)
    tu.add_argument("--allow-parallel", dest="allow_parallel",
                    action="store_true", default=None)
    tu.add_argument("--no-allow-parallel", dest="allow_parallel",
                    action="store_false")
    tu.add_argument("--depends-on", dest="depends_on", type=int, default=None)
    tu.add_argument("--resume-from", dest="resume_from", default=None)
    tu.add_argument("--gpus", type=int, default=None)
    tu.add_argument("--gpu-ids", dest="gpu_ids", default=None)

    for act in ("pause", "unpause", "retry"):
        tp = tsub.add_parser(act)
        _add_global(tp, suppress=True)
        tp.add_argument("task_id", type=int)

    td = tsub.add_parser("delete")
    _add_global(td, suppress=True)
    td.add_argument("task_id", type=int)
    td.add_argument("--yes", action="store_true")

    tlo = tsub.add_parser("logs")
    _add_global(tlo, suppress=True)
    tlo.add_argument("task_id", type=int)
    tlo.add_argument("-n", "--tail", type=int, default=None, metavar="LINES",
                     help="只显示最后 N 行")
    tlo.add_argument("-f", "--follow", action="store_true",
                     help="持续跟随（Ctrl-C 退出）")

    p.set_defaults(func=cmd_task)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="orchestrator.cli")
    _add_global(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("project")
    _add_global(p, suppress=True)
    psub = p.add_subparsers(dest="project_action", required=True)
    pc = psub.add_parser("create")
    _add_global(pc, suppress=True)
    pc.add_argument("name")
    pc.add_argument("--model", default=None)
    pc.add_argument("--desc", default="")
    pls = psub.add_parser("list")
    _add_global(pls, suppress=True)
    for act in ("show", "update", "archive", "delete"):
        pa = psub.add_parser(act)
        _add_global(pa, suppress=True)
        pa.add_argument("target")
        if act == "update":
            pa.add_argument("--desc", default=None)
            pa.add_argument("--model", default=None)
            pa.add_argument("--tags", default=None,
                            help="逗号分隔标签，覆盖旧值")
        if act == "delete":
            pa.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("submit")
    _add_global(p, suppress=True)
    p.add_argument("file")
    p.set_defaults(func=cmd_submit)

    _add_task_group(sub, "task", "任务 CRUD（job 为别名）")
    _add_task_group(sub, "job", "任务 CRUD 别名（等价 task）")

    p = sub.add_parser("list")
    _add_global(p, suppress=True)
    p.add_argument("--project", default=None)
    p.add_argument("--status", default=None)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("cancel")
    _add_global(p, suppress=True)
    p.add_argument("task_id", type=int)
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("schedule")
    _add_global(p, suppress=True)
    ssub = p.add_subparsers(dest="schedule_action", required=True)
    sl = ssub.add_parser("list")
    _add_global(sl, suppress=True)
    sl.add_argument("--project", default=None)
    sa = ssub.add_parser("add")
    _add_global(sa, suppress=True)
    sa.add_argument("--project", required=True)
    sa.add_argument("--name", required=True)
    sa.add_argument("--cron", default=None, help="五段 cron：分 时 日 月 周")
    sa.add_argument("--at", default=None, help="一次性触发 ISO 时间")
    sa.add_argument("--model", default=None)
    sa.add_argument("--config", default=None)
    sa.add_argument("--priority", type=int, default=100)
    sa.add_argument("--allow-parallel", action="store_true")
    sa.add_argument("--gpus", type=int, default=None)
    sa.add_argument("--gpu-ids", default=None)
    ss = ssub.add_parser("set")
    _add_global(ss, suppress=True)
    ss.add_argument("task_id", type=int)
    ss.add_argument("--cron", default=None)
    ss.add_argument("--at", default=None)
    ss.add_argument("--none", action="store_true",
                    help="清除定时，scheduled→pending")
    for act in ("pause", "unpause"):
        sp = ssub.add_parser(act)
        _add_global(sp, suppress=True)
        sp.add_argument("task_id", type=int)
    spv = ssub.add_parser("preview")
    _add_global(spv, suppress=True)
    spv.add_argument("cron_expr", help="五段 cron 表达式")
    spv.add_argument("--count", type=int, default=5)
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("prompt")
    _add_global(p, suppress=True)
    prsub = p.add_subparsers(dest="prompt_action", required=True)
    pr = prsub.add_parser("add")
    _add_global(pr, suppress=True)
    pr.add_argument("--project", required=True)
    pr.add_argument("--tag", default=None)
    pr.add_argument("--text", required=True)
    pr.add_argument("--negative", default=None)
    pl = prsub.add_parser("list")
    _add_global(pl, suppress=True)
    pl.add_argument("--project", default=None)
    pl.add_argument("--tag", default=None)
    pd = prsub.add_parser("delete")
    _add_global(pd, suppress=True)
    pd.add_argument("prompt_id", type=int)
    pd.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("gpu")
    _add_global(p, suppress=True)
    gsub = p.add_subparsers(dest="gpu_action", required=True)
    gs = gsub.add_parser("status")
    _add_global(gs, suppress=True)
    p.set_defaults(func=cmd_gpu)

    p = sub.add_parser("hook", help="投递 hook 指令（sample/save/restore/"
                                    "patch_config/suspend 等）")
    _add_global(p, suppress=True)
    p.add_argument("task_id", type=int)
    p.add_argument("hook_type",
                   choices=["sample", "sample_from_weights", "save",
                            "restore", "patch_config", "suspend"])
    p.add_argument("--payload", default=None, help="完整 JSON payload")
    p.add_argument("--name", default=None, help="save: 快照名")
    p.add_argument("--path", default=None, help="restore: 权重路径")
    p.add_argument("--weights", default=None,
                   help="sample_from_weights: 权重路径")
    p.add_argument("--tag", default=None, help="sample: 标签/提示词分组")
    p.add_argument("--prompts", default=None, help="sample: 提示词文件")
    p.add_argument("--n", type=int, default=None, help="sample: 张数")
    p.add_argument("--steps", type=int, default=None, help="sample: 采样步数")
    p.add_argument("--seed", type=int, default=None, help="sample: 随机种子")
    p.add_argument("--with-optimizer", action="store_true",
                   help="save: 同时保存优化器状态")
    p.add_argument("--reset-optim", action="store_true",
                   help="restore: 重建 optimizer/scheduler")
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                   help="patch_config: 点路径键值（可多次；值按 JSON 解析）")
    p.set_defaults(func=cmd_hook)

    p = sub.add_parser("hooks", help="列出 hook 历史")
    _add_global(p, suppress=True)
    p.add_argument("--task-id", type=int, default=None)
    p.set_defaults(func=cmd_hooks)

    p = sub.add_parser("resume", help="重启 suspended 任务")
    _add_global(p, suppress=True)
    p.add_argument("task_id", type=int)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("config", help="任务配置热改（task_config_kv + "
                                      "running 时投 patch_config hook）")
    _add_global(p, suppress=True)
    csub = p.add_subparsers(dest="config_action", required=True)
    cs = csub.add_parser("set")
    _add_global(cs, suppress=True)
    cs.add_argument("task_id", type=int)
    cs.add_argument("key", help="点路径键，如 training.learning_rate")
    cs.add_argument("value", help="值；先按 JSON 解析，失败按字符串")
    p.set_defaults(func=lambda a: cmd_config_set(a)
                   if a.config_action == "set" else 2)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
