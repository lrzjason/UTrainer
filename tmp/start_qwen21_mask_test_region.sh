#!/usr/bin/env bash
# UnifiedTrainer -> start: qwen21_mask_test_region
#
# REGION-WEIGHTED FM on the 30-group subset. Everything outside the region
# weighting is identical to start_qwen21_mask_test_fm.sh (the pure
# guide_flow_matching baseline), so the pair isolates ONE mechanism:
#   data    : /home/waas/qwen21_mask_dataset_test (30 groups) -> 28 steps/epoch
#             cache /home/waas/qwen21_cache_mask_test ; adding mask_configs
#             auto-backfills the 30 mask npz files (no rebuild)
#   mask    : the target's OWN alpha channel (_T is RGBA; _encode_mask is
#             alpha-first), bucket-cropped exactly like the target, area-averaged
#             to the latent grid (16x) and binarised at 0.5 -> (1,h,w) 0/1
#   loss    : guide_flow_matching (fixed 1.5) with
#             base_weight=1.0 / edit_weight=3.0  -> w = 1 outside, 4 inside the
#             target region, weighted MEAN by sum(w) so the loss VALUE stays on
#             the plain-mean scale (in-region gradient ~3.5x the uniform one at
#             ~4% coverage). No VAE decode, no second loss term.
#   val     : validation.loss_types + loss_params give a FRESH, UNWEIGHTED
#             guide_flow_matching instance for val_loss, so the val column stays
#             comparable with the fm baseline (the training instance is
#             region-weighted and must not leak into the metric).
#   output  : /home/waas/qwen21_transparent_edit_output/qwen21_mask_test_region-{0..19}
#
# Watch: loss/guide_flow_matching/mask_active (must be 1.0), mask_region_frac
# (~the target's latent-grid area), and val_loss_breakdown vs the baseline.
#
# Launch (foreground, log teed to /root/UTrainer/logs/):
#   bash /root/UTrainer/start_qwen21_mask_test_region.sh
# Detached:
#   nohup bash /root/UTrainer/start_qwen21_mask_test_region.sh >/dev/null 2>&1 &
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
LOG="/root/UTrainer/logs/qwen21_mask_test_region_${STAMP}.log"
echo "[qwen21-mask-test-region] config: /root/UTrainer/UnifiedTrainer/configs/qwen21_mask_test_region.json"
echo "[qwen21-mask-test-region] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_mask_test_region.json \
    2>&1 | tee "$LOG"
