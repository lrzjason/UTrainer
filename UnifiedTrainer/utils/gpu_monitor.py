"""GPU monitor — continuous GPU state sampling to a log file and (optionally) wandb.

Standalone background process (deliberately decoupled from training) so it keeps
sampling even if the training job OOMs. Useful to diagnose VRAM peaks/leaks by
correlating a system-level GPU-memory curve with training steps.

Metrics sampled each tick (per GPU):
  memory_used_gb / memory_total_gb / memory_free_gb
  util_gpu_pct, temp_c, power_w, clock_sm_mhz

Usage:
  python -m UnifiedTrainer.utils.gpu_monitor \
      --interval 5 --log /root/gpu_monitor.jsonl \
      --wandb --project UnifiedTrainer --run_name gpu_monitor_0823_32g

Log format (JSONL) — one object per sample:
  {"ts": "...", "gpu": 0, "name": "...", "memory_used_gb": .., ...,
   "cpu_mem_gb": .., "wall": ..}

WandB logging: each tick logs the same metric dict (batched by wandb).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime
from typing import Any, Dict, Optional

# (nvidia-smi field, output key, converter)
# memory.* is reported in MiB by nvidia-smi; divide by 1024 to get GB.
_NVIDIA_FIELDS = [
    ("name", "name", "str"),
    ("memory.used", "memory_used_gb", "gib"),
    ("memory.total", "memory_total_gb", "gib"),
    ("memory.free", "memory_free_gb", "gib"),
    ("utilization.gpu", "util_gpu_pct", "f"),
    ("temperature.gpu", "temp_c", "f"),
    ("power.draw", "power_w", "f"),
    ("clocks.sm", "clock_sm_mhz", "f"),
    ("clocks.mem", "clock_mem_mhz", "f"),
]


def _nvidia_smi_csv(fields: list) -> Optional[str]:
    try:
        import subprocess

        query = ",".join(f for f, _, _ in fields)
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu", query, "--format", "csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except Exception:
        return None


def _convert(value: str, conv: str) -> Any:
    """Convert a raw nvidia-smi csv value using the requested converter."""
    try:
        if conv == "str":
            return value.strip()
        if conv == "gib":
            return float(value) / 1024.0  # MiB -> GiB
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _sample(cpu: bool = True) -> Optional[Dict[str, Any]]:
    """Return one GPU sample dict, or None if nvidia-smi failed."""
    raw = _nvidia_smi_csv(_NVIDIA_FIELDS)
    if raw is None:
        return None

    out: Dict[str, Any] = {"ts": datetime.now().isoformat(timespec="seconds")}
    for i, line in enumerate(raw.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_NVIDIA_FIELDS):
            continue
        gpu = {
            key: _convert(val, conv)
            for (_, key, conv), val in zip(_NVIDIA_FIELDS, parts)
        }
        gpu["gpu"] = i
        gpu["host"] = socket.gethostname()
        if cpu:
            gpu.update(_cpu_mem())
        out.setdefault("gpus", []).append(gpu)

    return out


def _cpu_mem() -> Dict[str, Any]:
    try:
        with open("/proc/meminfo") as f:
            data = {}
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                # value like "120000000 kB" -> strip unit suffix
                data[k] = v.strip().split(" ")[0]
        total_kb = _to_float(data.get("MemTotal", "0"))
        avail_kb = _to_float(data.get("MemAvailable", "0"))
        return {
            "cpu_mem_total_gb": total_kb / 1024 / 1024,
            "cpu_mem_avail_gb": avail_kb / 1024 / 1024,
            "cpu_mem_used_gb": (total_kb - avail_kb) / 1024 / 1024,
        }
    except Exception:
        return {}


def _to_float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _wandb_log(wandb, sample: Dict[str, Any]) -> None:
    for gpu in sample.get("gpus", []):
        prefix = f"gpu{gpu['gpu']}"
        wandb.log({
            f"{prefix}/memory_used_gb": gpu["memory_used_gb"],
            f"{prefix}/memory_total_gb": gpu["memory_total_gb"],
            f"{prefix}/memory_free_gb": gpu["memory_free_gb"],
            f"{prefix}/util_gpu_pct": gpu["util_gpu_pct"],
            f"{prefix}/temp_c": gpu["temp_c"],
            f"{prefix}/power_w": gpu["power_w"],
            f"{prefix}/clock_sm_mhz": gpu["clock_sm_mhz"],
            "system/name": gpu["name"],
            "system/host": gpu["host"],
        })


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous GPU monitor")
    ap.add_argument("--interval", type=float, default=5.0, help="Sample interval seconds")
    ap.add_argument("--log", default="gpu_monitor.jsonl", help="JSONL log path")
    ap.add_argument("--wandb", action="store_true", help="Also log to wandb")
    ap.add_argument("--project", default="UnifiedTrainer")
    ap.add_argument("--run_name", default="gpu_monitor")
    ap.add_argument("--no-cpu", action="store_true", help="Skip CPU mem sampling")
    args = ap.parse_args()

    wandb = None
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.project, name=args.run_name)
            print(f"[gpu_monitor] wandb run started: {args.project}/{args.run_name}")
        except Exception as e:
            print(f"[gpu_monitor] wandb init failed (continuing file-only): {e}")
            wandb = None

    any_ok = False
    with open(args.log, "a", encoding="utf-8") as f:
        while True:
            t0 = time.monotonic()
            sample = _sample(cpu=not args.no_cpu)
            if sample is not None:
                any_ok = True
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()
                if wandb is not None:
                    try:
                        _wandb_log(wandb, sample)
                    except Exception as e:
                        print(f"[gpu_monitor] wandb log error: {e}")
            else:
                print("[gpu_monitor] nvidia-smi returned no data — skipping tick")
            # Sleep to keep the interval stable (interval measured from start of tick).
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, args.interval - elapsed))


if __name__ == "__main__":
    main()
