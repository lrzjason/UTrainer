#!/usr/bin/env bash
# UnifiedTrainer -> start: qwen21_shape_region_v2   (DATASET REVISION v2)
#
# SHAPE-ONLY region-weighted transparent-edit LoKr, FROM SCRATCH.
#   data    : /home/waas/qwen21_consist_dataset_shape/transparent_edit_shape
#             222 pairs -> 211 train + 11 val per epoch (colour dataset excluded)
#             v2 CONTENT: all 222 _T.txt captions rewritten (transparency
#             statement FIRST, hallucinations / appearance clauses / background
#             spill removed, class-level foreground kept) and all 222 _T.png
#             targets regenerated from the O-R difference (union with the backup
#             alpha, admitted within 32px, fragments stripped BEFORE dilating,
#             12px margin, holes <=256px filled).
#   mask    : each target's OWN alpha channel (_T is RGBA; _encode_mask is
#             alpha-first), bucket-cropped like the target, area-averaged to the
#             latent grid (16x), binarised at 0.5 -> (1,h,w) 0/1
#   loss    : guide_flow_matching (fixed 1.5) with base_weight=1.0 /
#             edit_weight=3.0 -> w = 1 outside, 4 inside the target region,
#             weighted MEAN by sum(w) (loss VALUE stays on the plain-mean scale)
#   val     : validation.loss_types + loss_params -> a FRESH, UNWEIGHTED
#             guide_flow_matching instance, so val_loss is comparable to the
#             unweighted baseline
#   cache   : /home/waas/qwen21_cache_transparent_edit (SHARED). Every T-derived
#             artefact of these 222 samples was ALREADY DELETED (latents, cached
#             images, caption embeddings, comp_px, masks, per-sample JSON, root
#             indexes); the R latents were kept. Expect the startup log to report
#             "Cache invalid: per-sample JSON missing ... Rebuilding" and then an
#             INCREMENTAL rebuild of exactly 222 samples (~a few minutes: VAE
#             encode + text encode + mask), NOT a full re-encode of the colour
#             dataset.
#   schedule: 20 epochs x 211 steps = 4220 steps, lr 1e-4 constant, no xm,
#             LoKr full-rank factor 4, adamw8bit + torchao_float8
#   output  : /home/waas/qwen21_transparent_edit_output/qwen21_shape_region_v2-{0..19}
#
# FIRST-MINUTE CHECK: "Cache invalid: per-sample JSON missing" -> cache build ->
# "Val split created: 11 val / 211 train" -> "Dataset [train]: 211 datarows",
# then mask_active=1.0 in the progress bar.
#
# Launch (foreground, log teed to /root/UTrainer/logs/):
#   bash /root/UTrainer/tmp/start_qwen21_shape_region_v2.sh
# Detached:
#   nohup bash /root/UTrainer/tmp/start_qwen21_shape_region_v2.sh >/dev/null 2>&1 &
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
LOG="/root/UTrainer/logs/qwen21_shape_region_v2_${STAMP}.log"
echo "[qwen21-shape-region-v2] config: /root/UTrainer/UnifiedTrainer/configs/qwen21_transparent_edit_shape_region_v2.json"
echo "[qwen21-shape-region-v2] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_transparent_edit_shape_region_v2.json \
    2>&1 | tee "$LOG"
