"""Watcher：1s 轮询 workspace/inbox/，导入 project/task/cmd JSON 到 DB。

文件流转（md/02）：
    inbox/ --导入DB--> processing/ --任务结束(dispatcher)--> done/ 或 failed/

- project_*.json：整项目定义（doc/Improvement_k3.md §4.2）
- task_*.json：单任务 {"project": name|id, "name":..., "model":..., "config": {...}|"path"}
- cmd_*.json：指令 {"action": "cancel"|"hook"|"set_config", ...}

内容 hash 去重：sha256 记入 workspace/import_index.json，重复投递直接归档 done/。
JSON 文件永不回写。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from typing import Optional

from .db import DB
from .scheduler import validate_schedule

logger = logging.getLogger(__name__)

POLL_INTERVAL = 1.0  # 秒
# KI-06：文件名无法识别（非 project_/task_/cmd_ 前缀）时连续扫描到此次数后
# 移入 failed/，不再永久滞留 inbox。
UNRECOGNIZED_LIMIT = 3


class Watcher:
    def __init__(self, workspace: str, db: DB):
        self.workspace = os.path.abspath(workspace)
        self.db = db
        for sub in ("inbox", "processing", "done", "failed", "samples", "prompts"):
            os.makedirs(os.path.join(self.workspace, sub), exist_ok=True)
        self.inbox = os.path.join(self.workspace, "inbox")
        self._index_path = os.path.join(self.workspace, "import_index.json")
        self._seen_hashes = self._load_index()
        # 无法识别文件的连续命中计数（KI-06）
        self._unrecognized: dict[str, int] = {}

    # ── 主循环 ──────────────────────────────────────────────────
    def run_forever(self, stop_flag) -> None:
        logger.info(f"Watcher watching {self.inbox}")
        while not stop_flag.is_set():
            try:
                self.scan_once()
            except Exception:
                logger.exception("Watcher scan failed")
            stop_flag.wait(POLL_INTERVAL)

    def scan_once(self) -> None:
        try:
            names = sorted(os.listdir(self.inbox))
        except FileNotFoundError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.inbox, name)
            if not os.path.isfile(path):
                continue
            try:
                self._handle_file(path, name)
            except Exception as e:
                logger.exception(f"Failed to import {name}")
                archived = self._archive(path, name, "failed", error=str(e))
                # KI-06：失败导入也记 hash——同一坏文件再次投递时按
                # duplicate 直接归档，不再每轮反复处理。
                try:
                    with open(archived, "rb") as f:
                        digest = hashlib.sha256(f.read()).hexdigest()
                    self._seen_hashes[digest] = archived
                    self._save_index()
                except OSError:
                    pass

    def _handle_file(self, path: str, name: str) -> None:
        with open(path, "rb") as f:
            content = f.read()
        digest = hashlib.sha256(content).hexdigest()
        if digest in self._seen_hashes:
            logger.info(f"Duplicate content ({name}), archiving to done/")
            self._archive(path, name, "done",
                          note="duplicate of " + self._seen_hashes[digest])
            return

        kind = self._classify(name)
        if kind is None:
            # KI-06：无法识别的文件不永久滞留 inbox——连续 UNRECOGNIZED_LIMIT
            # 轮未识别后移入 failed/ 并记 warning（文件可能正在写入，
            # 给若干轮宽限）。
            count = self._unrecognized.get(name, 0) + 1
            self._unrecognized[name] = count
            if count >= UNRECOGNIZED_LIMIT:
                self._unrecognized.pop(name, None)
                logger.warning(
                    f"Unrecognized file name pattern: {name}; "
                    f"moved to failed/ after {count} scans")
                self._archive(path, name, "failed",
                              note="unrecognized file name pattern "
                                   "(expect project_/task_/cmd_ prefix)")
            else:
                logger.warning(f"Unrecognized file name pattern: {name}, "
                               f"skip ({count}/{UNRECOGNIZED_LIMIT})")
            return
        self._unrecognized.pop(name, None)

        data = json.loads(content.decode("utf-8"))
        task_ids = []
        if kind == "project":
            task_ids = self._import_project(data, source_file=name)
        elif kind == "task":
            task_ids = [self._import_task(data, source_file=name)]
        elif kind == "cmd":
            self._handle_cmd(data)
        # 导入成功后移 processing/（cmd 处理完即 done）
        dest = "done" if kind == "cmd" else "processing"
        archived = self._archive(path, name, dest)
        self._seen_hashes[digest] = archived
        self._save_index()
        logger.info(f"Imported {name} as {kind}: task_ids={task_ids}")

    # ── 分类 ────────────────────────────────────────────────────
    @staticmethod
    def _classify(name: str) -> Optional[str]:
        if name.startswith("project_"):
            return "project"
        if name.startswith("task_"):
            return "task"
        if name.startswith("cmd_"):
            return "cmd"
        return None

    # ── 导入 ────────────────────────────────────────────────────
    def _resolve_config(self, cfg, base_dir: Optional[str] = None) -> dict:
        """config 可以是内联 dict，或 JSON 文件路径（相对项目文件/workspace/绝对）。"""
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, str):
            candidates = [cfg]
            if base_dir:
                candidates.insert(0, os.path.join(base_dir, cfg))
            candidates.append(os.path.join(self.workspace, cfg))
            for c in candidates:
                if os.path.isfile(c):
                    with open(c, "r", encoding="utf-8") as f:
                        return json.load(f)
            raise FileNotFoundError(f"config file not found: {cfg}")
        raise ValueError(f"invalid config value: {type(cfg)}")

    def _import_project(self, data: dict, source_file: str) -> list:
        proj = data.get("project", {})
        name = proj.get("name")
        if not name:
            raise ValueError("project file missing project.name")
        project_id = self.db.create_project(
            name=name,
            description=proj.get("description", ""),
            default_model=proj.get("default_model"),
            tags=proj.get("tags", []),
        )
        default_model = proj.get("default_model")
        task_ids = []
        name_to_id = {}

        def add_task(t: dict, depends_on_name: Optional[str] = None):
            cfg = self._resolve_config(t["config"])
            model = t.get("model") or default_model or cfg.get("model")
            if not model:
                raise ValueError(f"task {t.get('name')} has no model")
            # KI-10：入库前校验 cron / at（与 CLI/API 同一校验函数）
            validate_schedule(t.get("cron"), t.get("at"))
            # P5：多卡请求统一校验（gpus/gpu_ids 长度一致 + launch 模式）
            from .validation import validate_gpu_request
            gpus_n, gpu_ids_l = validate_gpu_request(
                cfg, t.get("gpus"), t.get("gpu_ids"))
            depends_on = name_to_id.get(depends_on_name) if depends_on_name else None
            tid = self.db.create_task(
                project_id, t["name"], model, cfg,
                priority=t.get("priority", 100),
                depends_on=depends_on,
                resume_from=t.get("resume_from"),
                resume_mode=t.get("resume_mode", "weights"),
                cron=t.get("cron"), at=t.get("at"),
                allow_parallel=1 if t.get("allow_parallel") else 0,
                gpus=gpus_n,
                gpu_ids=",".join(map(str, gpu_ids_l)) if gpu_ids_l else None,
                status="scheduled" if (t.get("cron") or t.get("at")) else "pending",
                source_file=source_file,
            )
            name_to_id[t["name"]] = tid
            task_ids.append(tid)
            return tid

        for chain in data.get("chains", []):
            for t in chain.get("tasks", []):
                add_task(t, depends_on_name=t.get("depends_on"))
        for t in data.get("scheduled", []):
            add_task(t, depends_on_name=t.get("depends_on"))
        for t in data.get("tasks", []):
            add_task(t, depends_on_name=t.get("depends_on"))
        return task_ids

    def _import_task(self, data: dict, source_file: str) -> int:
        proj_ref = data.get("project", "default")
        proj = self.db.get_project(proj_ref)
        project_id = proj["id"] if proj else self.db.create_project(str(proj_ref))
        cfg = self._resolve_config(data["config"])
        model = data.get("model") or cfg.get("model")
        if not model:
            raise ValueError("task file missing model")
        # KI-10：入库前校验 cron / at
        validate_schedule(data.get("cron"), data.get("at"))
        # P5：多卡请求统一校验（gpus/gpu_ids 长度一致 + launch 模式）
        from .validation import validate_gpu_request
        gpus_n, gpu_ids_l = validate_gpu_request(
            cfg, data.get("gpus"), data.get("gpu_ids"))
        return self.db.create_task(
            project_id, data.get("name", os.path.splitext(source_file)[0]),
            model, cfg,
            priority=data.get("priority", 100),
            resume_from=data.get("resume_from"),
            resume_mode=data.get("resume_mode", "weights"),
            cron=data.get("cron"), at=data.get("at"),
            allow_parallel=1 if data.get("allow_parallel") else 0,
            gpus=gpus_n,
            gpu_ids=",".join(map(str, gpu_ids_l)) if gpu_ids_l else None,
            status="scheduled" if (data.get("cron") or data.get("at"))
            else "pending",
            source_file=source_file,
        )

    def _handle_cmd(self, data: dict) -> None:
        action = data.get("action")
        if action == "cancel":
            tid = int(data["task_id"])
            task = self.db.get_task(tid)
            if task and task["status"] in ("pending", "scheduled", "waiting_gpu"):
                self.db.set_task_status(tid, "cancelled")
                logger.info(f"Task {tid} cancelled via cmd")
            else:
                logger.warning(f"Cannot cancel task {tid}: status={task and task['status']}")
        elif action == "hook":
            # KI-18：统一走 hooks.enqueue 校验（类型白名单 + 仅 running
            # 任务可投递），不再绕过校验直接写表。
            from . import hooks as hook_proto
            hid = hook_proto.enqueue(self.db, int(data["task_id"]),
                                     data["type"], data.get("payload", {}))
            logger.info(f"cmd hook task={data['task_id']} "
                        f"type={data['type']} hook_id={hid}")
        elif action == "set_config":
            # KI-04 修复：统一走 set_config_and_notify——写 kv 之外，
            # running 任务会追加 patch_config hook（worker 即时生效）。
            from . import hooks as hook_proto
            hid = hook_proto.set_config_and_notify(
                self.db, int(data["task_id"]), data["key"], data["value"])
            logger.info(f"cmd set_config task={data['task_id']} "
                        f"key={data['key']} hook_id={hid}")
        else:
            raise ValueError(f"unknown cmd action: {action}")

    # ── 归档 ────────────────────────────────────────────────────
    def _archive(self, path: str, name: str, dest: str,
                 error: Optional[str] = None, note: Optional[str] = None) -> str:
        dest_dir = os.path.join(self.workspace, dest)
        os.makedirs(dest_dir, exist_ok=True)
        target = os.path.join(dest_dir, name)
        if os.path.exists(target):
            base, ext = os.path.splitext(name)
            target = os.path.join(dest_dir, f"{base}.{int(time.time())}{ext}")
        shutil.move(path, target)
        if error is not None:
            with open(target + ".error.log", "w", encoding="utf-8") as f:
                f.write(error)
        if note is not None:
            with open(target + ".note.txt", "w", encoding="utf-8") as f:
                f.write(note)
        return target

    # ── 去重索引 ────────────────────────────────────────────────
    def _load_index(self) -> dict:
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self) -> None:
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._seen_hashes, f, indent=1)
        os.replace(tmp, self._index_path)
