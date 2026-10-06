#!/usr/bin/env bash
# UnifiedTrainer — start: qwen21_transparent_edit
#
# Qwen-Image 2.1 LoKr, transparent-edit (RGBA) dataset.
#   dataset   : /home/waas/qwen21_consist_dataset  (338 pairs, subdirs recursed)
#               _T = RGBA transparent TARGET (learned), _R = RGB reference
#               (_O/_SAM/_preview and .zip are not referenced by the config)
#   network   : LoKr FULL RANK (lokr_full_rank=true), factor 4 -> W1 [4,4] +
#               W2 full out/4 x in/4 matrix (~1/16 of all-Linear params)
#   lr        : 1e-4 CONSTANT (no warmup, no decay), adamw8bit + torchao_float8
#   XM        : explorative noise selector K_cond=K_uncond=5, constant schedule
#   loss      : guide_flow_matching (DC-Gen Eq.10) with fixed_guidance_scale=1.5
#               (measured w* for Qwen-Image 2.1, see md/09 §7) — the correct
#               objective for this guidance-distilled checkpoint
#   validation: val_loss + 3 RGBA val images / epoch, guidance_scale 1.0
#   output    : /home/waas/qwen21_transparent_edit_output/qwen21_transparent_edit-{epoch}
#   cache     : /home/waas/qwen21_cache_transparent_edit (built on first launch;
#               also generates empty_embedding needed by guide_fm + caption dropout)
#
# Launch (foreground, log teed to /root/UTrainer/logs/):
#   /root/UTrainer/start_qwen21_transparent_edit.sh
# Detached:
#   nohup /root/UTrainer/start_qwen21_transparent_edit.sh >/dev/null 2>&1 &
# Resume FULL (weights+optimizer+step/epoch, needs *_training_state.pt):
#   add  --resume-full /home/waas/qwen21_transparent_edit_output/<file>.safetensors
#   to the $PY line below. Weights-only: use --resume instead.
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
NVIDIA_LIBS="$($PY -c 'import nvidia.cuda_nvrtc,os,glob; root=os.path.dirname(os.path.dirname(nvidia.cuda_nvrtc.__file__)); print(":".join(sorted(glob.glob(os.path.join(root,"*","lib")))))' 2>/dev/null)"
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
# api.bandw.top 有 AAAA 记录，但这台机器**没有 IPv6 出口**（无全局 v6 地址、
# 无 v6 默认路由）。wandb 的 Go core 按 RFC 6724 优先试 IPv6 -> 连不上 ->
# 一直等到 90s init_timeout，抛 "Run initialization has timed out"。
# curl 不受影响是因为它有 Happy-Eyeballs 会在几百毫秒内回落到 IPv4。
# Go 的解析器**优先读 /etc/hosts**，所以在这里钉死 IPv4 即可（幂等，可重复跑）。
if ! grep -q 'api\.bandw\.top' /etc/hosts 2>/dev/null; then
    printf '172.67.193.61\tapi.bandw.top\n104.21.20.172\tapi.bandw.top\n' >> /etc/hosts
    echo "[wandb] pinned api.bandw.top -> IPv4 in /etc/hosts (this box has no IPv6 route)"
fi
export WANDB_BASE_URL=https://api.bandw.top/
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="/root/UTrainer/logs/qwen21_transparent_edit_${STAMP}.log"
echo "[qwen21-transparent-edit] log: $LOG"
$PY /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_transparent_edit.json \
    2>&1 | tee "$LOG"
