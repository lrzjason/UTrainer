"""Scheduler（P3）：cron / 一次性 at 触发，纯标准库实现。

- 30s tick（可配）扫描 status='scheduled' 的任务，到点置 pending 交
  Dispatcher / GPU Guard；
- cron 表达式：五段 `分 时 日 月 周`，支持 * , - / 步进；周 0-7（0/7=周日）；
  日/周同时受限时按 POSIX 惯例取 OR；
- 一次性 `"at": "ISO时间"`：到点触发一次，不重复武装；
- 错过触发只保留一个排队实例：任务离开 scheduled（pending/running/…）期间
  不再触发；cron 任务终态后由 dispatcher 重新武装（re-arm），同一分钟内
  靠 `_meta.last_fire` 去重，不会重复放行。

环境约束：无第三方库（无 croniter），解析器为本文件内置实现。
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from .db import DB

logger = logging.getLogger(__name__)

DEFAULT_TICK = 30.0  # 秒

_FIELD_RANGES = [
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("dom", 1, 31),
    ("month", 1, 12),
    ("dow", 0, 7),
]


class CronError(ValueError):
    pass


def _parse_field(spec: str, lo: int, hi: int, name: str) -> set:
    """解析单段 cron 字段为取值集合。"""
    values: set = set()
    if not spec:
        raise CronError(f"empty {name} field")
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise CronError(f"empty item in {name} field: {spec!r}")
        if "/" in part:
            base, step_s = part.split("/", 1)
            try:
                step = int(step_s)
            except ValueError:
                raise CronError(f"bad step in {name}: {part!r}")
            if step <= 0:
                raise CronError(f"step must be > 0 in {name}: {part!r}")
        else:
            base, step = part, 1
        if base in ("*", ""):
            start, end = lo, hi
        elif "-" in base:
            a, b = base.split("-", 1)
            try:
                start, end = int(a), int(b)
            except ValueError:
                raise CronError(f"bad range in {name}: {part!r}")
        else:
            try:
                start = int(base)
            except ValueError:
                raise CronError(f"bad value in {name}: {part!r}")
            # "a/n" 等价 "a-hi/n"；裸 "a" 等价单点
            end = hi if step > 1 else start
        if start < lo or end > hi or start > end:
            raise CronError(
                f"{name} value out of range [{lo},{hi}]: {part!r}")
        values.update(range(start, end + 1, step))
    if name == "dow" and 7 in values:
        values.add(0)  # 周日 0/7 等价
    return values


class CronExpr:
    """五段 cron 表达式。matches(dt) 按本地时间字段判断。"""

    def __init__(self, expr: str):
        parts = expr.split()
        if len(parts) != 5:
            raise CronError(
                f"cron 需要五段 '分 时 日 月 周'，得到 {len(parts)} 段: {expr!r}")
        self.expr = expr
        self.minute = _parse_field(parts[0], 0, 59, "minute")
        self.hour = _parse_field(parts[1], 0, 23, "hour")
        self.dom = _parse_field(parts[2], 1, 31, "dom")
        self.month = _parse_field(parts[3], 1, 12, "month")
        self.dow = _parse_field(parts[4], 0, 7, "dow")
        self._dom_restricted = parts[2] != "*"
        self._dow_restricted = parts[4] != "*"

    def matches(self, dt: datetime) -> bool:
        if dt.minute not in self.minute or dt.hour not in self.hour:
            return False
        if dt.month not in self.month:
            return False
        dom_ok = dt.day in self.dom
        # Python weekday(): Mon=0..Sun=6 → cron dow: Sun=0
        dow_ok = ((dt.weekday() + 1) % 7) in self.dow
        if self._dom_restricted and self._dow_restricted:
            return dom_ok or dow_ok
        return dom_ok and dow_ok

    def next_after(self, dt: datetime,
                   horizon_days: int = 366) -> Optional[datetime]:
        """严格晚于 dt 的下一个命中时刻（本地时区，分钟粒度）。

        供 CLI `schedule list/preview` 与 API /api/schedules 计算 NEXT FIRE。
        按本地时间逐分钟走查（与 matches 的本地时间口径一致，KI-05）；
        DST 跳变由 tz-aware 的 timedelta 运算自然处理。
        不可达表达式（如 2 月 30 日）在 horizon 内无解时返回 None。
        """
        cur = dt.astimezone().replace(second=0, microsecond=0)
        limit = int(horizon_days) * 24 * 60
        for _ in range(limit):
            cur = cur + timedelta(minutes=1)
            if self.matches(cur):
                return cur
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"CronExpr({self.expr!r})"


def validate_schedule(cron: Optional[str], at: Optional[str]) -> None:
    """入库前统一校验 cron / at（非法值抛 CronError/ValueError）。

    供 CLI / server / watcher 三个入口复用，避免非法值落库（KI-10）。
    """
    if cron:
        CronExpr(cron)
    if at:
        _parse_at(at)


def _parse_at(value: str) -> datetime:
    """解析一次性触发时间（ISO 8601；naive 视为本地时间，统一转 UTC）。"""
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise CronError(f"invalid 'at' ISO time: {value!r}")
    if dt.tzinfo is None:
        dt = dt.astimezone()  # naive → 本地时区
    return dt.astimezone(timezone.utc)


def next_fire_of(task: dict, kv: Optional[dict] = None,
                 now: Optional[datetime] = None) -> Optional[datetime]:
    """任务下一次触发时间（返回本地时区 aware datetime，用于展示）。

    - at 一次性任务：已触发过（_meta.last_fire 存在）→ None；否则 at 时间；
    - cron 任务：CronExpr.next_after(now)，不可达表达式 → None；
    - 两者皆无（即时任务）→ None。
    """
    now = now or datetime.now(timezone.utc)
    if task.get("at"):
        if kv and kv.get("_meta.last_fire"):
            return None
        return _parse_at(task["at"]).astimezone()
    if task.get("cron"):
        nf = CronExpr(task["cron"]).next_after(now)
        return nf.astimezone() if nf is not None else None
    return None


class Scheduler:
    def __init__(self, db: DB, tick: float = DEFAULT_TICK):
        self.db = db
        self.tick = tick

    # ── 主循环 ──────────────────────────────────────────────────
    def run_forever(self, stop_flag: threading.Event) -> None:
        logger.info(f"Scheduler started (tick={self.tick}s)")
        while not stop_flag.is_set():
            try:
                self.scan_once()
            except Exception:
                logger.exception("Scheduler scan failed")
            stop_flag.wait(self.tick)

    def scan_once(self, now: Optional[datetime] = None) -> list:
        """扫描 scheduled 任务，到点的置 pending。返回被触发的 task id 列表。"""
        now = now or datetime.now(timezone.utc)
        fired = []
        for task in self.db.list_tasks(status="scheduled"):
            tid = task["id"]
            try:
                if self._due(task, now):
                    self._fire(task, now)
                    fired.append(tid)
            except CronError as e:
                logger.error(f"Task {tid} bad schedule: {e}; marking failed")
                self.db.set_task_status(tid, "failed", error=str(e))
            except Exception:
                logger.exception(f"Scheduler failed on task {tid}")
        return fired

    # ── 判定与触发 ──────────────────────────────────────────────
    @staticmethod
    def _fire_key(now: datetime) -> str:
        """last_fire 去重键：统一用本地时区（KI-05）。

        cron 字段匹配按本地时间（matches(now.astimezone())），去重键
        必须与之一致；若混用 UTC 与本地时间，时区偏移含 :30/:45 的
        地区本地分钟与 UTC 分钟错开，去重会失效或漏触发。
        """
        return now.astimezone().strftime("%Y-%m-%dT%H:%M")

    def _due(self, task: dict, now: datetime) -> bool:
        # at / cron 一次性去重：已有 last_fire 的 at 任务不再触发（KI-09）
        last = self.db.get_config_kv(task["id"]).get("_meta.last_fire")
        if task.get("at"):
            if last is not None:
                return False  # at 一次性任务已触发过
            return _parse_at(task["at"]) <= now
        cron = task.get("cron")
        if not cron:
            return False
        if not CronExpr(cron).matches(now.astimezone()):
            return False
        # 同一（本地）分钟去重：错过/重复 tick 不重复放行
        return last != self._fire_key(now)

    def _fire(self, task: dict, now: datetime) -> None:
        tid = task["id"]
        self.db.set_config_kv(tid, "_meta.last_fire", self._fire_key(now))
        self.db.set_task_status(tid, "pending", error=None,
                                started_at=None, finished_at=None)
        logger.info(f"Task {tid} ({task['name']}) fired -> pending "
                    f"(cron={task.get('cron')!r} at={task.get('at')!r})")
