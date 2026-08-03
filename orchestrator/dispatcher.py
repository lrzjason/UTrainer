"""Dispatcher（P3：worker 池 + GPU Guard 准入；P5：多卡）。

P1 行为保持为默认：max_parallel=1 时同一时刻只有一个 worker，等价串行。
P5 起：任务可声明 gpus（卡数）/gpu_ids（钉卡）；GPU Guard 一次准入多张卡，
worker 经 CUDA_VISIBLE_DEVICES=<csv> 绑卡；config training.multi_gpu=ddp
且 gpus>1 时用 accelerate launch 拉起多进程（rank 0 之外由 train.py 门控）。

循环：next_runnable_tasks()（pending + waiting_gpu）→ GPUGuard.judge()
→ Admit：spawn 子进程 train.py --task-id/--db（并行时绑卡），心跳判活
（120s 无心跳 kill 整棵进程树）；→ Wait：任务置 waiting_gpu 并记录原因
（kv `_meta.gpu_wait_reason`）。退出码 0=done / 42=suspended / 其他=failed。

任一 worker 退出 → 唤醒主循环立即重扫 waiting_gpu 补位。
cron 任务进入终态后自动 re-arm（置回 scheduled）等下一轮触发。

任务 done 时：解析后继任务 resume_from 中的 "$task:<name>.output" 引用，
写为前驱任务记录在 task_config_kv["_meta.output"] 的产物路径。

worker stdout/stderr 落盘 workspace/logs/task_<id>.run_<restart>.log（P5），
CLI `task logs` 消费；retention 由 main.py 每小时裁剪。
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .db import DB
from .gpu_guard import GPUGuard

logger = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT = 120.0   # 秒，有心跳但间隔超此值判僵死
STARTUP_TIMEOUT = 300.0     # 秒，启动宽限：超过此值仍无任何心跳判僵死
POLL_INTERVAL = 2.0         # 主循环/子进程轮询间隔
EXIT_SUSPENDED = 42
# KI-09 熔断：cron 任务连续失败达到此次数后不再 re-arm，
# 保持 failed 终态（源文件随之归档 failed/），避免无限重试循环。
MAX_CONSECUTIVE_FAILURES = 3

_REF_RE = re.compile(r"^\$task:(?P<name>.+)\.output$")
_TERMINAL = {"done", "failed", "cancelled"}

_WINDOWS = os.name == "nt"


def _free_port() -> int:
    """绑定端口 0 取系统空闲端口后释放（多卡 ddp 的 main_process_port 用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def prune_worker_logs(log_dir: str, db: DB, keep_days: int = 30) -> int:
    """P5：删除终态超过 keep_days 天的 worker 运行日志（main.py 每小时裁剪）。

    与 heartbeats/gpu_snapshots 同节奏；非终态任务（可能再派发）与
    任务已不存在（已删除）的日志按超龄处理。返回删除文件数。
    """
    if not os.path.isdir(log_dir):
        return 0
    try:
        names = os.listdir(log_dir)
    except OSError:
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for name in names:
        m = re.fullmatch(r"task_(\d+)\.run_(\d+)\.log", name)
        if not m:
            continue
        path = os.path.join(log_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        task = db.get_task(int(m.group(1)))
        terminal = task is None or task["status"] in _TERMINAL
        if terminal and mtime < cutoff:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """整棵进程树杀（P5：accelerate launch 有子进程，直接 kill 会留孤儿）。

    Windows: taskkill /T /F（按 PID 树）；POSIX: killpg(SIGKILL)（进程组）。
    """
    try:
        if _WINDOWS:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=15)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:
        logger.warning(f"process-tree kill failed for pid {proc.pid}: {e}")
        try:
            proc.kill()
        except Exception:
            pass


def _hb_stale(hb_age: Optional[float], uptime: float,
              startup_timeout: float = STARTUP_TIMEOUT,
              timeout: float = HEARTBEAT_TIMEOUT) -> Optional[str]:
    """心跳判活（纯函数，便于单测）。

    hb_age：最近一条心跳距现在的秒数；None 表示从未收到心跳。
    uptime：worker 启动至今的秒数。
    返回判死原因；存活返回 None。
    """
    if hb_age is None:
        # KI-03 修复：模型加载期卡死的 worker 从未写心跳，
        # 超过启动宽限也判死，不再无限等待。
        if uptime > startup_timeout:
            return (f"no heartbeat within startup grace "
                    f"{startup_timeout:.0f}s (uptime {uptime:.0f}s)")
        return None
    if hb_age > timeout:
        return f"heartbeat stale {hb_age:.0f}s"
    return None


class Dispatcher:
    def __init__(self, workspace: str, db: DB,
                 train_script: Optional[str] = None,
                 max_parallel: int = 1,
                 gpu_guard: Optional[GPUGuard] = None):
        self.workspace = os.path.abspath(workspace)
        self.db = db
        # ScheduledTrainer 根目录 = orchestrator 包的上级
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.train_script = train_script or os.path.join(
            self.root, "UnifiedTrainer", "train.py")
        self.max_parallel = max(1, int(max_parallel))
        self.guard = gpu_guard or GPUGuard(db)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        # tid -> {"proc": Popen, "thread": Thread, "gpus": [int],
        #         "log_path": str}
        self._workers: dict[int, dict] = {}
        # P5：worker 日志目录（CLI `task logs` 消费，main.py 每小时裁剪）
        self.log_dir = os.path.join(self.workspace, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    # ── 主循环 ──────────────────────────────────────────────────
    def run_forever(self, stop_flag) -> None:
        logger.info(f"Dispatcher started (max_parallel={self.max_parallel}, "
                    f"gpu_provider={self.guard.provider.name})")
        while not stop_flag.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Dispatcher iteration failed")
            self._wake.wait(POLL_INTERVAL)
            self._wake.clear()

    def occupied_gpus(self) -> set:
        """在跑 worker 占用的卡集合（union，供 guard 准入排除）。"""
        with self._lock:
            out = set()
            for w in self._workers.values():
                out.update(w.get("gpus") or [])
        return out

    def run_once(self) -> bool:
        """收割已结束 worker，并按容量尝试派发。有派发动作返回 True。"""
        self._reap()
        with self._lock:
            running = len(self._workers)
        capacity = self.max_parallel - running
        if capacity <= 0:
            return False
        started = False
        for task in self.db.next_runnable_tasks():
            if capacity <= 0:
                break
            with self._lock:
                if task["id"] in self._workers:
                    continue
                running = len(self._workers)
            decision = self.guard.judge(task, running_workers=running,
                                        max_parallel=self.max_parallel,
                                        occupied=self.occupied_gpus())
            if decision.admit:
                self._start_worker(task, gpus=decision.gpus)
                capacity -= 1
                started = True
        return started

    # ── worker 池管理 ───────────────────────────────────────────
    def _start_worker(self, task: dict, gpus: list = None) -> None:
        tid = task["id"]
        gpus = [int(g) for g in (gpus or [])]
        self.db.set_task_status(tid, "running")
        self.db.set_config_kv(tid, "_meta.gpu_index", gpus)
        cmd, env = self._build_launch(task, gpus)
        logger.info(f"Dispatch task {tid} ({task['name']})"
                    + (f" on GPUs {gpus}" if gpus else " (no binding)"))
        restart = task.get("restart_count") or 0
        log_path = os.path.join(
            self.log_dir, f"task_{tid}.run_{restart}.log")
        # 同一 restart 重复启动（re-dispatch）时追加，不截断历史
        log_fh = open(log_path, "ab")
        kwargs = {}
        if _WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, cwd=self.root, env=env,
                                stdout=log_fh, stderr=subprocess.STDOUT,
                                **kwargs)
        log_fh.close()
        t = threading.Thread(target=self._worker_main, args=(task, proc),
                             name=f"worker-{tid}", daemon=True)
        with self._lock:
            self._workers[tid] = {"proc": proc, "thread": t, "gpus": gpus,
                                  "log_path": log_path}
        t.start()

    def _build_launch(self, task: dict, gpus: list) -> tuple:
        """构造 worker 启动命令与环境。

        gpus<=1 或 training.multi_gpu != 'ddp' → 单进程 train.py；
        gpus>1 且 multi_gpu=ddp → accelerate launch --num_processes=K
        （CUDA_VISIBLE_DEVICES 重映射后 accelerate 看到 0..K-1）。
        """
        tid = task["id"]
        cmd = [sys.executable, self.train_script,
               "--task-id", str(tid), "--db", self.db.path]
        env = dict(os.environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        if len(gpus) > 1:
            try:
                config = self.db.materialize_config(tid)
            except Exception:
                config = {}
            mode = config.get("training", {}).get("multi_gpu", "reserve")
            if mode == "ddp":
                mp = config.get("training", {}).get("mixed_precision", "bf16")
                cmd = [sys.executable, "-m", "accelerate.commands.launch",
                       "--num_processes", str(len(gpus)),
                       "--multi_gpu",
                       "--main_process_port", str(_free_port())]
                if mp not in ("no", "none"):
                    cmd += ["--mixed_precision", str(mp)]
                cmd += [self.train_script, "--task-id", str(tid),
                        "--db", self.db.path]
        return cmd, env

    def _reap(self) -> None:
        with self._lock:
            dead = [tid for tid, w in self._workers.items()
                    if not w["thread"].is_alive()]
            for tid in dead:
                w = self._workers.pop(tid)
                w["thread"].join(timeout=5)
        # 补位由 run_once 下一轮完成（worker 退出已 _wake.set()）

    def _worker_main(self, task: dict, proc: subprocess.Popen) -> None:
        tid = task["id"]
        try:
            exit_code = self._monitor(proc, tid)
            self._finalize(task, exit_code)
        except Exception:
            logger.exception(f"Worker thread for task {tid} failed")
        finally:
            self._wake.set()  # 任一 worker 退出 → 立即重扫 waiting_gpu 补位

    # ── 单任务生命周期（worker 线程内）─────────────────────────
    def _finalize(self, task: dict, exit_code: int) -> None:
        tid = task["id"]
        if exit_code == 0:
            self.db.set_task_status(tid, "done")
            logger.info(f"Task {tid} done")
            self._resolve_successors(task)
        elif exit_code == EXIT_SUSPENDED:
            self.db.set_task_status(tid, "suspended")
            logger.info(f"Task {tid} suspended (exit 42)")
        elif exit_code == -999:  # 心跳超时/启动宽限超时被杀
            self.db.set_task_status(tid, "failed",
                                    error="heartbeat/startup timeout, killed")
            logger.error(f"Task {tid} killed: heartbeat timeout")
        else:
            self.db.set_task_status(tid, "failed",
                                    error=f"worker exit code {exit_code}")
            logger.error(f"Task {tid} failed: exit {exit_code}")

        # cron 任务终态后 re-arm，等下一轮触发（at 任务一次性，不 re-arm）。
        # KI-09 熔断：连续失败计数存 kv `_meta.consecutive_failures`，成功清零；
        # 达到 MAX_CONSECUTIVE_FAILURES 后不再 re-arm，任务保持 failed 终态，
        # 原因记录 kv `_meta.rearm_blocked`（源文件随终态归档 failed/）。
        if task.get("cron") and exit_code != EXIT_SUSPENDED:
            kv = self.db.get_config_kv(tid)
            fails = 0 if exit_code == 0 else \
                int(kv.get("_meta.consecutive_failures") or 0) + 1
            self.db.set_config_kv(tid, "_meta.consecutive_failures", fails)
            cur = self.db.get_task(tid)
            if cur["status"] in _TERMINAL:
                if fails >= MAX_CONSECUTIVE_FAILURES:
                    reason = (f"cron re-arm blocked: {fails} consecutive "
                              f"failures (>= {MAX_CONSECUTIVE_FAILURES})")
                    self.db.set_config_kv(tid, "_meta.rearm_blocked", reason)
                    logger.error(f"Task {tid} {reason}")
                else:
                    self.db.set_task_status(tid, "scheduled")
                    logger.info(f"Task {tid} re-armed (cron={task['cron']!r})")

        self._archive_source(self.db.get_task(tid) or task)

    def _monitor(self, proc: subprocess.Popen, tid: int,
                 startup_timeout: float = STARTUP_TIMEOUT,
                 heartbeat_timeout: float = HEARTBEAT_TIMEOUT) -> int:
        """轮询子进程 + 心跳判活。返回退出码；-999 表示僵死被杀。

        判活两条规则（KI-03）：
        - 已有心跳：最近一条超过 HEARTBEAT_TIMEOUT（120s）判僵死；
        - 从未有心跳：启动超过 STARTUP_TIMEOUT（默认 300s）判僵死
          （覆盖模型加载期卡死的场景）。
        """
        started = time.monotonic()
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            hb = self.db.last_heartbeat(tid)
            hb_age = None
            if hb is not None:
                try:
                    ts = datetime.strptime(hb["ts"], "%Y-%m-%d %H:%M:%S")
                    ts = ts.replace(tzinfo=timezone.utc)
                    hb_age = (datetime.now(timezone.utc) - ts).total_seconds()
                except ValueError:
                    hb_age = None
            reason = _hb_stale(hb_age, time.monotonic() - started,
                               startup_timeout, heartbeat_timeout)
            if reason is not None:
                logger.error(f"Task {tid} {reason}, killing")
                _kill_process_tree(proc)  # P5：整棵进程树（ddp 多进程）
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                return -999
            time.sleep(POLL_INTERVAL)

    # ── $task:<name>.output 引用解析 ────────────────────────────
    def _find_task_by_name(self, project_id: int, name: str) -> Optional[dict]:
        """同项目内按名字查任务（同名取 id 最大者）。"""
        matches = [t for t in self.db.list_tasks(project_id=project_id)
                   if t["name"] == name]
        return matches[-1] if matches else None

    def _resolve_successors(self, done_task: dict) -> None:
        successors = self.db.successors_of(done_task["id"])
        for suc in successors:
            ref = suc["resume_from"]
            if not ref:
                continue
            m = _REF_RE.match(ref)
            if not m:
                continue
            ref_name = m.group("name")
            if ref_name == done_task["name"]:
                src = done_task
            else:
                # KI-14：允许引用同链更早的祖先（如 c 依赖 b，但
                # resume_from="$task:a.output"）；按名字在同项目内解析。
                src = self._find_task_by_name(done_task["project_id"], ref_name)
                if src is None:
                    logger.warning(
                        f"Task {suc['id']} references unknown task "
                        f"{ref_name} in project {done_task['project_id']}")
                    continue
                if src["status"] != "done":
                    logger.warning(
                        f"Task {suc['id']} references {ref_name} (id="
                        f"{src['id']}) which is {src['status']}, not done")
                    continue
            output = self.db.get_config_kv(src["id"]).get("_meta.output")
            if output is None:
                logger.error(
                    f"Task {src['id']} ({ref_name}) done but no "
                    f"_meta.output recorded; cannot resolve resume_from "
                    f"for task {suc['id']}")
                continue
            # 重读当前状态，避免用过期状态做读-改-写
            cur = self.db.get_task(suc["id"])
            self.db.set_task_status(
                suc["id"], cur["status"], resume_from=output)
            logger.info(
                f"Resolved $task:{ref_name}.output -> {output} "
                f"for task {suc['id']} ({suc['name']})")

    # ── 源文件归档 ──────────────────────────────────────────────
    def _archive_source(self, task: dict) -> None:
        """同一 source_file 的任务全部进入终态后，把 processing/ 原件归档。

        归档语义（有意为之，KI-09 注释）：
        - cron 周期任务在 _finalize 中先 re-arm（置回 scheduled），因此
          其源文件长期驻留 processing/ 属预期行为——周期任务永不"完成"，
          只有熔断（连续失败 >= MAX_CONSECUTIVE_FAILURES）或被 cancel 后
          才真正终态并归档；
        - done 归档时往文件 JSON 追加 `_result` 结果摘要字段（KI-16）；
        - failed 归档时旁写 `<name>.error.log`（含各任务 DB error 字段）。
        """
        source = task.get("source_file")
        if not source:
            return
        siblings = [t for t in self.db.list_tasks()
                    if t.get("source_file") == source]
        if not all(t["status"] in _TERMINAL for t in siblings):
            return
        ok = all(t["status"] == "done" for t in siblings)
        processing_path = os.path.join(self.workspace, "processing", source)
        if not os.path.exists(processing_path):
            return
        import shutil
        dest_dir = os.path.join(self.workspace, "done" if ok else "failed")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, source)
        if os.path.exists(dest):
            base, ext = os.path.splitext(source)
            dest = os.path.join(dest_dir, f"{base}.{int(time.time())}{ext}")
        shutil.move(processing_path, dest)
        logger.info(f"Archived {source} -> {dest_dir}")
        if ok:
            self._write_result_summary(dest, siblings)
        else:
            self._write_error_log(dest, siblings)

    @staticmethod
    def _write_result_summary(dest: str, siblings: list) -> None:
        """done 归档：往文件 JSON 追加 `_result` 结果摘要（KI-16）。"""
        summary = {
            "archived_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "tasks": [{
                "id": t["id"], "name": t["name"], "status": t["status"],
                "started_at": t.get("started_at"),
                "finished_at": t.get("finished_at"),
            } for t in siblings],
        }
        try:
            with open(dest, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["_result"] = summary
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"write result summary failed for {dest}: {e}")

    @staticmethod
    def _write_error_log(dest: str, siblings: list) -> None:
        """failed 归档：旁写 <name>.error.log，含各任务 DB error 字段。"""
        lines = []
        for t in siblings:
            lines.append(f"task {t['id']} ({t['name']}): status={t['status']}"
                         f" error={t.get('error') or '(none)'}")
        try:
            with open(dest + ".error.log", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            logger.warning(f"write error.log failed for {dest}: {e}")
