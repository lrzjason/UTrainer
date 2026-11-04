#!/usr/bin/env bash
# UnifiedTrainer -> start: qwen21_mask_test_fm   (PURE guide_flow_matching baseline)
#
# The reference run for the alpha-coverage-penalty test. Everything outside the
# "losses" block is identical to start_qwen21_mask_test_cover.sh:
#   data    : /home/waas/qwen21_mask_dataset_test (30 groups) -> 28 train steps/epoch
#             cache /home/waas/qwen21_cache_mask_test (latents reused)
#   schedule: 20 epochs, lr 1e-4 constant, LoKr factor=4 full-rank,
#             adamw8bit + torchao_float8 + gradient checkpointing, no xm, from 0
#   loss    : guide_flow_matching ONLY (fixed_guidance_scale 1.5) -- the
#             composite_consistency entry is REMOVED ENTIRELY rather than set to
#             weight 0, so there is no per-step VAE decode, no _comp_px.npz pixel
#             caches are created, and cover_viol is not logged.
#   output  : /home/waas/qwen21_transparent_edit_output/qwen21_mask_test_fm-{0..19}
#
# Launch (foreground, log teed to /root/UTrainer/logs/):
#   bash /root/UTrainer/start_qwen21_mask_test_fm.sh
# Detached:
#   nohup bash /root/UTrainer/start_qwen21_mask_test_fm.sh >/dev/null 2>&1 &
# Resume FULL (weights+optimizer+step/epoch):
#   add  --resume-full /home/waas/qwen21_transparent_edit_output/<file>.safetensors
#   to the $PY line below. Weights-only: use --resume instead.
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
NVIDIA_LIBS="$($PY -c 'import nvidia.cuda_nvrtc,os,glob; root=os.path.dirname(os.path.dirname(nvidia.cuda_nvrtc.__file__)); print(":".join(sorted(glob.glob(os.path.join(root,"*","lib")))))' 2>/dev/null)"
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
# api.bandw.top has AAAA records but this box has no IPv6 route; wandb's Go core
# tries IPv6 first and times out after 90s. Pin IPv4 in /etc/hosts (idempotent).
if ! grep -q 'api\.bandw\.top' /etc/hosts 2>/dev/null; then
    printf '172.67.193.61\tapi.bandw.top\n104.21.20.172\tapi.bandw.top\n' >> /etc/hosts
    echo "[wandb] pinned api.bandw.top -> IPv4 in /etc/hosts"
fi
export WANDB_BASE_URL=https://api.bandw.top/
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p /root/UTrainer/logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="/root/UTrainer/logs/qwen21_mask_test_fm_${STAMP}.log"
echo "[qwen21-mask-test-fm] config: /root/UTrainer/UnifiedTrainer/configs/qwen21_mask_test_fm.json"
echo "[qwen21-mask-test-fm] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_mask_test_fm.json \
    2>&1 | tee "$LOG"
