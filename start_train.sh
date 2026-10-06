#!/usr/bin/env bash
# UnifiedTrainer Orchestrator — Linux start script
cd "$(dirname "$0")"

# cuDNN Frontend JIT-compiles fused attention via NVRTC at runtime; the nvrtc
# major version must match torch's cudart (cu13 here). Prepend the matching
# nvrtc lib dir so cuDNN loads it first — otherwise Krea2's masked attention
# fails with "No valid engine configs" (nvrtc 12 vs cudart 13 mismatch).
# Guarded no-op when the package/path is absent.
NVRTC_LIB="$(python -c 'import nvidia.cuda_nvrtc,os;print(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__),"lib"))' 2>/dev/null)"
if [ -n "$NVRTC_LIB" ] && [ -d "$NVRTC_LIB" ]; then
    export LD_LIBRARY_PATH="$NVRTC_LIB:$LD_LIBRARY_PATH"
fi

python /root/ScheduledTrainer/UnifiedTrainer/train.py --model krea2 --config /root/ScheduledTrainer/UnifiedTrainer/configs/krea2_train_cloud_xm_100x.json