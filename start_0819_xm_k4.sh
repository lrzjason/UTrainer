#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
NVIDIA_LIBS="$($PY -c 'import nvidia.cuda_nvrtc,os,glob; root=os.path.dirname(os.path.dirname(nvidia.cuda_nvrtc.__file__)); print(":".join(sorted(glob.glob(os.path.join(root,"*","lib")))))' 2>/dev/null)"
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
export WANDB_BASE_URL=https://api.bandw.top/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY /root/UTrainer/UnifiedTrainer/train.py --model krea2 --config /root/UTrainer/UnifiedTrainer/configs/krea2_train_cloud_xm_1024.json
