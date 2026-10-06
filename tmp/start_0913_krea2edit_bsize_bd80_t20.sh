#!/usr/bin/env bash
# UnifiedTrainer — start: 0913_krea2edit_bsize
# krea2 edit training on /home/waas/breasts_size_control (656 _t/_bd pairs),
# LoKR initialized from the 0823_xm_k10 epoch19 weights (weights-only resume:
# fresh optimizer / step=0 / epoch=0), 20 epochs, constant lr 4e-4, XM K=10,
# helios reference corruption enabled.
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
NVIDIA_LIBS="$($PY -c 'import nvidia.cuda_nvrtc,os,glob; root=os.path.dirname(os.path.dirname(nvidia.cuda_nvrtc.__file__)); print(":".join(sorted(glob.glob(os.path.join(root,"*","lib")))))' 2>/dev/null)"
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
export WANDB_BASE_URL=https://api.bandw.top/
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
$PY /root/UTrainer/UnifiedTrainer/train.py --model krea2 --config /root/UTrainer/UnifiedTrainer/configs/0913_krea2edit_bsize_bd80_t20.json
