#!/usr/bin/env bash
# UnifiedTrainer -> start: qwen21_mask_test_cover
#
# 30-group subset test of the ALPHA COVERAGE PENALTY (composite_consistency,
# cover-only mode). Same data / cache / schedule as qwen21_mask_test_fullres:
#   data    : /home/waas/qwen21_mask_dataset_test (30 groups) -> 28 train steps/epoch
#   cache   : /home/waas/qwen21_cache_mask_test (latents reused; the 30 px caches
#             are built once on the first pass -- none exist on this box yet,
#             and the key is now fp=0 because feather_sigma_px=0)
#   loss    : guide_flow_matching (fixed 1.5)
#             + composite_consistency COVER-ONLY, weight 0.03:
#               data/matte/fg/dice/grad = 0, cover_weight = 1.0
#               short = relu(alpha_T - alpha_hat)  (plate + prediction, one-sided)
#               L = sum(relu(short - 0.1)) / sum(alpha_T)   NO FEATHERING
#               sigma_max 0.2, every_n_steps 1
#   output  : /home/waas/qwen21_transparent_edit_output/qwen21_mask_test_cover-{0..19}
#
# Baseline variant (telemetry only, no gradient): flip the composite_consistency
# entry's "weight" to 0.0 in the config -- cover_viol is still logged every
# decoded step and gives the number the term has to beat.
#
# Launch (foreground, log teed to /root/UTrainer/logs/):
#   bash /root/UTrainer/start_qwen21_mask_test_cover.sh
# Detached:
#   nohup bash /root/UTrainer/start_qwen21_mask_test_cover.sh >/dev/null 2>&1 &
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
LOG="/root/UTrainer/logs/qwen21_mask_test_cover_${STAMP}.log"
echo "[qwen21-mask-test-cover] config: /root/UTrainer/UnifiedTrainer/configs/qwen21_mask_test_cover.json"
echo "[qwen21-mask-test-cover] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_mask_test_cover.json \
    2>&1 | tee "$LOG"
