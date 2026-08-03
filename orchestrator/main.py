"""Orchestrator 常驻入口：Watcher + Scheduler + Dispatcher (+ 可选 API)。

用法：
    python -m orchestrator.main --workspace workspace --db workspace/trainer.db
    python -m orchestrator.main --workspace workspace --api --port 7860 \
        --max-parallel 2
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

from .db import DB
from .dispatcher import Dispatcher, prune_worker_logs
from .gpu_guard import GPUGuard
from .scheduler import Scheduler
from .watcher import Watcher

logger = logging.getLogger("orchestrator")


def main() -> None:
    parser = argparse.ArgumentParser(description="ScheduledTrainer orchestrator")
    parser.add_argument("--workspace", default="workspace",
                        help="工作区目录（含 inbox/ 等子目录）")
    parser.add_argument("--db", default=None,
                        help="SQLite 数据库路径，默认 <workspace>/trainer.db")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="并行 worker 上限（默认 1，保持 P1 串行行为）")
    parser.add_argument("--tick", type=float, default=30.0,
                        help="Scheduler 扫描间隔秒（默认 30）")
    parser.add_argument("--api", action="store_true",
                        help="同时启动 HTTP API（默认 127.0.0.1:7860）")
    parser.add_argument("--host", default=None,
                        help="API 绑定地址（默认 127.0.0.1；云端用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=7860,
                        help="API 端口（默认 7860，与 gradio 默认一致）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    workspace = os.path.abspath(args.workspace)
    db_path = os.path.abspath(args.db) if args.db else os.path.join(
        workspace, "trainer.db")

    db = DB(db_path)
    watcher = Watcher(workspace, db)          # 自动创建 workspace 子目录
    # KI-07 修复：并行上限单源在 dispatcher（内部 max(1, ...)），guard 直接
    # 透传，不再在此设置误导性的第二处下限。
    guard = GPUGuard(db, max_parallel=args.max_parallel)
    dispatcher = Dispatcher(workspace, db, max_parallel=args.max_parallel,
                            gpu_guard=guard)
    scheduler = Scheduler(db, tick=args.tick)

    stop = threading.Event()
    threads = [
        threading.Thread(target=watcher.run_forever, args=(stop,),
                         name="watcher", daemon=True),
        threading.Thread(target=scheduler.run_forever, args=(stop,),
                         name="scheduler", daemon=True),
        threading.Thread(target=dispatcher.run_forever, args=(stop,),
                         name="dispatcher", daemon=True),
    ]
    server = None
    if args.api:
        from . import server as server_mod
        api_host = args.host or server_mod.DEFAULT_HOST
        server = server_mod.create_server(workspace, db, host=api_host,
                                          port=args.port)
        server_mod.run_in_thread(server)
        logger.info(f"API listening on http://{api_host}:{args.port}")

    for t in threads:
        t.start()
    logger.info(f"Orchestrator up. workspace={workspace} db={db_path} "
                f"max_parallel={args.max_parallel}")

    # KI-17：每小时裁剪一次 heartbeats / gpu_snapshots，防无限增长
    last_prune = 0.0
    try:
        while any(t.is_alive() for t in threads):
            stop.wait(1.0)
            now = time.monotonic()
            if now - last_prune >= 3600:
                last_prune = now
                try:
                    n_hb = db.prune_heartbeats(keep_days=7)
                    n_gpu = db.prune_gpu_snapshots(keep_rows=10000)
                    n_logs = prune_worker_logs(
                        os.path.join(workspace, "logs"), db)
                    if n_hb or n_gpu or n_logs:
                        logger.info(f"retention prune: heartbeats -{n_hb}, "
                                    f"gpu_snapshots -{n_gpu}, "
                                    f"worker logs -{n_logs}")
                except Exception as e:
                    logger.warning(f"retention prune failed (ignored): {e}")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop.set()
        if server is not None:
            server.shutdown()
        for t in threads:
            t.join(timeout=10)


if __name__ == "__main__":
    main()
