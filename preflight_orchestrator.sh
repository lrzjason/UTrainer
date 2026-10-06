#!/bin/bash
# preflight_orchestrator.sh — start the ScheduledTrainer orchestrator with the
# verified CUDA13/nvrtc fix env. The dispatcher spawns workers (train.py) with
# its OWN environment, so the orchestrator MUST inherit LD_LIBRARY_PATH or the
# cuDNN Frontend JIT crashes again ("No valid engine configs").
#
# Same verified env as preflight_train.sh: pip cu13 libcudnn first (cuDNN JIT
# loads unversioned libnvrtc names from here), then /opt/nvrtc13 (real 13.0
# copies at libnvrtc.so / .so.12 / .so.13).
#
# Usage (cloud, as root):
#   bash /root/ScheduledTrainer/preflight_orchestrator.sh

cd /root/ScheduledTrainer

CUDNN_LIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib
NVRTC13=/opt/nvrtc13
export LD_LIBRARY_PATH="$CUDNN_LIB:$NVRTC13:$LD_LIBRARY_PATH"

echo "[preflight] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[preflight] nvrtc resolution (fresh subprocess):"
if ! python - <<'PY'
import ctypes


def fresh_version(libname):
    """dlopen in a fresh subprocess = what a spawned worker will see."""
    lib = ctypes.CDLL(libname)
    buf = ctypes.c_char_p()
    lib.nvrtcGetVersion(ctypes.byref(buf))
    return buf.value.decode()


bad = 0
for n in ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13"):
    try:
        print(f"  {n} -> {fresh_version(n)}")
    except Exception as e:
        bad += 1
        print(f"  {n} -> FAILED: {e}")
raise SystemExit(1 if bad else 0)
PY
then
    echo "[preflight] ERROR: nvrtc resolution FAILED — /opt/nvrtc13 contents are suspect."
    echo "[preflight] Ground truth:  ls -la /opt/nvrtc13 && nm -D /opt/nvrtc13/libnvrtc.so | grep nvrtcGetVersion"
    echo "[preflight] Re-copy the real 13.0 file over all three names, e.g.:"
    echo "[preflight]   cp /root/miniconda3/targets/x86_64-linux/lib/libnvrtc.so.13.0.88 /opt/nvrtc13/libnvrtc.so"
    exit 1
fi

# Port conflict check: a stale orchestrator would spawn workers WITHOUT the
# fix env (inherits its own environment) and every training would crash.
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":7860 "; then
    echo "[preflight] ERROR: port 7860 already in use — an old orchestrator is running."
    echo "[preflight] Stop it first:  pkill -f orchestrator.main"
    echo "[preflight] (workers inherit the OLD env; the cuDNN fix would NOT apply)"
    exit 1
fi

exec python -m orchestrator.main --workspace workspace --api --port 7860 --max-parallel 2
