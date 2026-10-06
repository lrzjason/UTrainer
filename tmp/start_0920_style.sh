#!/usr/bin/env bash
# UnifiedTrainer — start: 0920_style  (to be launched AFTER epoch 11 is saved)
#
# What changed vs 0919_bdbc:
#   1. training set: photo_nsfw/爆机少女喵小吉 removed (416 samples, 63% of 655) ->
#      239 samples left (227 train / 12 val at val_split_ratio 0.05).
#   2. every <base>_bd.txt now starts with a style sentence, identical for every
#      sample of its class (this file is the training caption for BOTH interfaces,
#      because cap_bd and cap_bc both read the _bd role's text):
#          二次元动漫:  这是一张二次元动漫图片。
#          写实照片  :  这是一张写实照片。
#      63 anime / 176 realistic.  The captions' own medium wording is NOT part of
#      the signal any more -- every "3D写实/3D渲染" image is a real photo (user
#      confirmed), which the captioner had described as a render.
#   3. WEIGHTS-ONLY resume from 0919_bdbc_epoch11 (full:false), so optimizer/step/
#      epoch all restart: new run, new schedule, new cache (the changed captions
#      force a re-encode of cap_bd/cap_bc; cap_t and all latents are reused).
#   4. lr 1e-4, cosine decay to 0 over 10 epochs, XM K=10 on.
#      (cosine rebuilt over the REMAINING steps for a weights-only resume, so it
#      starts at 1e-4, unlike the full-resume case which continues the old curve.)
#
# Expected wall clock: the dataset is 63% smaller, so ~227 steps/epoch at the
# measured ~31 s/step with XM K=10  ->  ~2.0 h/epoch  ->  ~20 h for 10 epochs.
#
# Output: /home/waas/0920_style_output/0920_style-{0..9}
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
$PY /root/UTrainer/UnifiedTrainer/train.py --model krea2 --config /root/UTrainer/UnifiedTrainer/configs/0920_style.json
