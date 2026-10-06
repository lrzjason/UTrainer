#!/usr/bin/env bash
# UnifiedTrainer — start: 0917_krea2edit_bsize_lr1e4_xm10
# krea2 edit training on /home/waas/breasts_size_control (656 _t/_bd pairs), on the
# corrected conditions (skin-tone #FADBAE markers, frame guides removed) and the
# repaired instruction captions.
# LoKR WEIGHTS-ONLY initialization from the 0916_krea2edit_bsize_nocolor epoch12
# checkpoint -> fresh optimizer / step=0 / epoch=0, so this is a new run:
#   20 epochs from epoch 0, lr 1e-4, cosine decay to 0 over 12460 steps,
#   XM explorative K=10 (K_cond=10, K_uncond=10), helios reference corruption on,
#   batch mix 4x cap_bd+ref_BD : 1x cap_t+none.
# Output: /home/waas/0917_krea2edit_bsize_lr1e4_xm10_output/0917_krea2edit_bsize_lr1e4_xm10-{0..19}
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
$PY /root/UTrainer/UnifiedTrainer/train.py --model krea2 --config /root/UTrainer/UnifiedTrainer/configs/0917_krea2edit_bsize_lr1e4_xm10.json
