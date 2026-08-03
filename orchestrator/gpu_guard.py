"""GPU Guard（P3）：并行准入判定 + VRAM 采集（可注入 provider）。

采集链（环境无 pynvml / torch，禁止 pip 安装）：
    NvmlProvider（pynvml，惰性 import）
    → NvidiaSmiProvider（nvidia-smi 子进程解析，复用 P1 cli gpu status 思路）
    → NullProvider（无 GPU 信息：空机放行，并行保守拒绝）

准入规则（md/04）：
    空机放行（不查 VRAM）；否则要求 task.allow_parallel 且某卡
    free_mb > total*3/4；并行上限 max_parallel（默认 2）。
    每次判定把快照写 gpu_snapshots，结果写 kv `_meta.gpu_decision`；
    拒绝时任务置 waiting_gpu 并把原因写 `_meta.gpu_wait_reason`。
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

from .db import DB

logger = logging.getLogger(__name__)

DEFAULT_MAX_PARALLEL = 2
FREE_FRACTION = 0.75


class Admit:
    def __init__(self, gpus: list):
        self.admit, self.gpus, self.reason = True, list(gpus), "admitted"

    @property
    def gpu(self):
        """向后兼容：单卡时返回首卡；多卡调用方请用 .gpus。"""
        return self.gpus[0] if self.gpus else None

    def __repr__(self):
        return f"Admit(gpus={self.gpus})"


class Wait:
    def __init__(self, reason: str):
        self.admit, self.gpus, self.gpu, self.reason = False, [], None, reason

    def __repr__(self):
        return f"Wait({self.reason!r})"


# ── Provider ────────────────────────────────────────────────────
class NvmlProvider:
    name = "pynvml"

    def __init__(self):
        import pynvml  # noqa: F401 — 不存在则 ImportError，调用方降级
        self._nvml = pynvml
        pynvml.nvmlInit()

    def snapshot(self) -> list:
        nvml = self._nvml
        out = []
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            m = nvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = None
            out.append({"gpu_index": i, "total_mb": m.total // (1024 * 1024),
                        "used_mb": m.used // (1024 * 1024),
                        "free_mb": m.free // (1024 * 1024),
                        "util_pct": util})
        return out


def _to_int(s: str) -> Optional[int]:
    """容错整数解析：nvidia-smi 的 `[N/A]` 等字段返回 None（KI-15）。"""
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_nvidia_smi_csv(text: str) -> list:
    """解析 nvidia-smi --format=csv,noheader,nounits 输出。

    KI-15：`[N/A]` 字段不再导致崩溃——util_pct 解析失败记 None；
    total/used/free 任一无法解析的行整行跳过（记 warning），
    避免脏数据进入 gpu_snapshots（NOT NULL 约束）。
    """
    snaps = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, total, used, free, util = parts[:5]
        gpu_index = _to_int(idx)
        total_mb, used_mb, free_mb = _to_int(total), _to_int(used), _to_int(free)
        if gpu_index is None or total_mb is None or used_mb is None \
                or free_mb is None:
            logger.warning(f"nvidia-smi line skipped (unparseable): {line!r}")
            continue
        snaps.append({"gpu_index": gpu_index, "total_mb": total_mb,
                      "used_mb": used_mb, "free_mb": free_mb,
                      "util_pct": _to_int(util)})
    return snaps


class NvidiaSmiProvider:
    name = "nvidia-smi"

    def snapshot(self) -> list:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
        return parse_nvidia_smi_csv(out.stdout)


class NullProvider:
    """拿不到任何 GPU 信息：空机放行，并行保守拒绝。"""
    name = "none"

    def snapshot(self) -> list:
        return []


class FakeProvider:
    """测试注入用。"""

    def __init__(self, gpus: list):
        self.name = "fake"
        self._gpus = gpus

    def set(self, gpus: list) -> None:
        self._gpus = gpus

    def snapshot(self) -> list:
        return [dict(g) for g in self._gpus]


def auto_provider():
    """pynvml → nvidia-smi → none，逐级降级。"""
    try:
        return NvmlProvider()
    except Exception as e:
        logger.info(f"pynvml unavailable ({e}); falling back to nvidia-smi")
    try:
        p = NvidiaSmiProvider()
        p.snapshot()
        return p
    except Exception as e:
        logger.warning(f"nvidia-smi unavailable ({e}); GPU info disabled, "
                       f"parallel admissions will be denied")
    return NullProvider()


def best_gpu(gpus: list) -> Optional[int]:
    """单卡选择（兼容旧调用）；多卡请用 best_gpus/select_gpus。"""
    if not gpus:
        return None
    return max(gpus, key=lambda g: g.get("free_mb") or 0)["gpu_index"]


def best_gpus(gpus: list, n: int, exclude: set = ()) -> list:
    """按 free_mb 从大到小取 n 张卡；exclude 中的卡跳过。可返回少于 n 张。"""
    exclude = set(exclude)
    cands = sorted((g for g in gpus if g["gpu_index"] not in exclude),
                   key=lambda g: g.get("free_mb") or 0, reverse=True)
    return [g["gpu_index"] for g in cands[:n]]


def select_gpus(gpus: list, n: int, occupied: set = (),
                pinned: Optional[list] = None,
                empty_machine: bool = False) -> Optional[list]:
    """P5：为任务挑 n 张卡。返回 [indices]；挑不满返回 None。

    - pinned 指定时：验证这些卡恰好存在且（非空机时）满足 FREE_FRACTION；
    - 自动选择：候选 = 未被占用且（空机跳过 VRAM 门槛 / 否则 free>total*3/4）
      的卡，按 free_mb 取前 n；
    - 空机放行（首个任务可能吃满整卡，沿用 P3 语义）。
    """
    if n <= 0:
        return []
    occupied = set(occupied)
    by_idx = {g["gpu_index"]: g for g in gpus}
    if pinned is not None:
        missing = [i for i in pinned if i not in by_idx]
        if missing:
            logger.warning(f"pinned gpu_ids missing from nvidia-smi: {missing}")
            return None
        if not empty_machine:
            for i in pinned:
                g = by_idx[i]
                if (g.get("free_mb") or 0) <= g["total_mb"] * FREE_FRACTION:
                    return None
        return list(pinned)
    cands = []
    for g in gpus:
        if g["gpu_index"] in occupied:
            continue
        if empty_machine:
            cands.append(g)
        elif (g.get("free_mb") or 0) > g["total_mb"] * FREE_FRACTION:
            cands.append(g)
    cands.sort(key=lambda g: g.get("free_mb") or 0, reverse=True)
    if len(cands) < n:
        return None
    return [g["gpu_index"] for g in cands[:n]]


# ── Guard ───────────────────────────────────────────────────────
class GPUGuard:
    def __init__(self, db: DB, provider=None,
                 max_parallel: int = DEFAULT_MAX_PARALLEL):
        self.db = db
        self.provider = provider if provider is not None else auto_provider()
        self.max_parallel = max_parallel

    def judge(self, task: dict, running_workers: int,
              max_parallel: Optional[int] = None,
              occupied: set = ()) -> "Admit | Wait":
        """判定一次准入；写快照 + 决策 kv；拒绝时置 waiting_gpu。

        P5：任务可声明 gpus（卡数）与 gpu_ids（钉卡）。空机放行（不查
        VRAM）；并行时需 n 张卡各自满足 free>total*3/4。
        返回 Admit(gpus=[...]) 或 Wait(reason)。
        """
        from .validation import parse_gpu_ids
        n = int(task.get("gpus") or 1)
        pinned = parse_gpu_ids(task.get("gpu_ids"))
        occupied = set(occupied)

        # KI-08：显式 None 判断——max_parallel=0（意图禁止并行）不能被
        # 真值判断绕过而落回 self.max_parallel。
        limit = max_parallel if max_parallel is not None else self.max_parallel
        # KI-15：采集失败与"无 GPU 信息"区分——失败时拒绝原因明确标注
        # metrics collection failed，不再写成误导性的 VRAM 不足。
        collect_err: Optional[Exception] = None
        try:
            gpus = self.provider.snapshot()
        except Exception as e:
            collect_err = e
            logger.warning(f"GPU metrics collection failed ({e}); "
                           f"denying parallel")
            gpus = []
        if gpus:
            self.db.gpu_snapshot(gpus)

        empty_machine = running_workers == 0 and not occupied
        if empty_machine:
            if not gpus:
                # 无 GPU 信息（NullProvider 降级）：空机放行不绑卡（沿用 P3）
                decision = Admit([])
            else:
                chosen = select_gpus(gpus, n, empty_machine=True,
                                     pinned=pinned)
                if chosen is None or len(chosen) < n:
                    decision = Wait(
                        f"not enough physical cards "
                        f"({len(chosen or [])}/{n})")
                else:
                    decision = Admit(chosen)
        elif collect_err is not None:
            decision = Wait(f"GPU metrics collection failed: {collect_err}")
        elif not task.get("allow_parallel"):
            decision = Wait("parallel not allowed (allow_parallel=0)")
        elif running_workers >= limit:
            decision = Wait(f"max parallel reached ({running_workers}/{limit})")
        else:
            # 总卡数容量：occupied ∪ requested 不能超过物理卡数
            if len(gpus) and len(occupied) + n > len(gpus):
                decision = Wait(
                    f"not enough cards free (need {n}, occupied "
                    f"{len(occupied)}, total {len(gpus)})")
            else:
                chosen = select_gpus(gpus, n, occupied=occupied,
                                     pinned=pinned)
                if chosen is None:
                    decision = Wait(
                        f"insufficient free VRAM (need {n} card(s) with "
                        f"> {FREE_FRACTION:.0%} free)")
                else:
                    decision = Admit(chosen)

        from datetime import datetime, timezone
        self.db.set_config_kv(task["id"], "_meta.gpu_decision", {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "admit": decision.admit, "gpus": decision.gpus,
            "gpu": decision.gpus[0] if decision.gpus else None,
            "reason": decision.reason, "provider": self.provider.name,
            "running_workers": running_workers, "max_parallel": limit,
            "occupied": sorted(occupied), "requested": n,
            "gpus_snapshot": gpus,
        })
        if decision.admit:
            if task["status"] == "waiting_gpu":
                self.db.set_config_kv(task["id"], "_meta.gpu_wait_reason", "")
        else:
            self.db.set_config_kv(task["id"], "_meta.gpu_wait_reason",
                                  decision.reason)
            if task["status"] in ("pending", "waiting_gpu"):
                self.db.set_task_status(task["id"], "waiting_gpu")
            logger.info(f"Task {task['id']} waits for GPU: {decision.reason}")
        return decision
