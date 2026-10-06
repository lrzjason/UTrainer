#!/bin/bash
# preflight_train.sh — cloud CUDA-13 launch wrapper for UnifiedTrainer train.py
#
# Guarantees the nvrtc/cuDNN runtime fix is active in THIS process (and its
# children) regardless of which shell/session launched it, then execs train.py.
# Usage (cloud, as root):
#   bash /root/ScheduledTrainer/preflight_train.sh --model krea2 --config /root/ScheduledTrainer/UnifiedTrainer/configs/krea2_train_cloud_xm_100x.json
#
# The two paths below must match the cloud layout; the script skips any that
# are missing instead of failing.

CUDNN_LIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib
NVRTC13=/opt/nvrtc13
TRAIN=/root/ScheduledTrainer/UnifiedTrainer/train.py

# 1) Fix the runtime library resolution order (most important first).
export LD_LIBRARY_PATH="$CUDNN_LIB:$NVRTC13:$LD_LIBRARY_PATH"

echo "== preflight: LD_LIBRARY_PATH =="
echo "$LD_LIBRARY_PATH"
echo

# 2) Sanity: report which cuDNN + nvrtc versions this process will resolve.
echo "== preflight: runtime libraries (fresh subprocess) =="
python - <<'PYEOF'
import ctypes, subprocess, sys, os

def fresh_version(libname):
    code = (
        "import ctypes;"
        f"lib = ctypes.CDLL('{libname}');"
        "m = ctypes.c_int(); n = ctypes.c_int();"
        "lib.nvrtcVersion(ctypes.byref(m), ctypes.byref(n));"
        "print(f'{m.value}.{n.value}')"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=60)
        return out.stdout.strip() or out.stderr.strip()[-200:]
    except Exception as e:
        return f"<error: {e}>"

import ctypes.util
for name in ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13"):
    print(f"    {name:15s} -> {fresh_version(name)}")
try:
    import torch
    print(f"    cudart (torch.version.cuda): {torch.version.cuda}")
    print(f"    cudnn.version(): {torch.backends.cudnn.version()}")
    print(f"    cudnn.lib: {torch.backends.cudnn.lib()}")
except Exception as e:
    print(f"    <torch import failed: {e}>")
PYEOF
echo

# 3) Launch training with the fix applied.
exec python "$TRAIN" "$@"
