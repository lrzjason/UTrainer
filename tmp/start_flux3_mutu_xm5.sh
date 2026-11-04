#!/usr/bin/env bash
# UnifiedTrainer -> start: flux3_mutu_xm5   (FLUX.3 klein BASE, LoRA, XM=5)
#
# FLUX.3 klein (base / non-distilled) LoRA test training on the mutu dataset.
# Recipe follows the established mutu/XM test recipe (start_qwen21_mask_test_fm.sh,
# qwen21_mutu_t2i_lokr_xm_k5_cloud.json):
#   data    : /home/waas/mutu (38 twitter jpg + captions, *_t.jpg / *_t.txt)
#             cache /home/waas/flux3_klein_cache_mutu_t2i (fresh)
#   schedule: 20 epochs, lr 1e-4 constant (warmup 0), LoRA rank16/alpha16 on
#             to_q/to_k/to_v/to_out, bf16, NO gradient checkpointing,
#             grad accumulation 1, max_grad_norm 1
#   sampler : XM explorative noise selector, K_cond=K_uncond=5 (best-of-5)
#   loss    : plain flow_matching (use_weighting false)
#   repeats : 1  -> ~36 train pairs/epoch x 20 epochs ~ 720 steps
#   output  : /home/waas/flux3_klein_mutu_xm5_output/flux3_klein_mutu_xm5-{0..19}
#
# Two FLUX.3-specific things (no other model here does this):
#   * NO scheduler_path -- the FLUX.3 module ships no scheduler component, so the
#     adapter builds the recipe schedule itself (constant shift 6.93). Pointing
#     scheduler_path at the checkpoint would fall back to the wrong shift.
#   * flow_map is auto-detected from transformer/config.json. This checkpoint is
#     the non-distilled BASE, so flow_map=False and no target_timestep is passed.
#
# VRAM: gradient_checkpointing is ON. The earlier attempt had it off and OOMed at
# step 0 -- measured base+adapter 11.6 GB, then the non-checkpointed forward needed
# ~19 GB of activations (30.55 GB of the 31.36 GiB card). With checkpointing the
# blocks are recomputed in backward (~20-30% slower per step). To run without it,
# drop lokr_factor to 8 (346M -> 86.5M trainable) and/or set block_swap.
#
# wandb: ENABLED (reporter.type=wandb, wandb.enabled=true). The two lines the
# other start scripts carry are therefore present here too:
#   * /etc/hosts IPv4 pin for api.bandw.top (this box has no IPv6 route and
#     wandb's Go core tries IPv6 first, stalling ~90 s)
#   * WANDB_BASE_URL pointing at the self-hosted proxy
# Credentials come from ~/.netrc ("machine api.bandw.top"), which already exists
# on this box -- no WANDB_API_KEY is needed.
#
# Launch (foreground, log teed to /root/UTrainer/logs/):
#   bash /root/UTrainer/start_flux3_mutu_xm5.sh
# Detached:
#   nohup bash /root/UTrainer/start_flux3_mutu_xm5.sh >/dev/null 2>&1 &
# Resume FULL (weights+optimizer+step/epoch):
#   add  --resume-full /home/waas/flux3_klein_mutu_xm5_output/<file>.safetensors
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
LOG="/root/UTrainer/logs/flux3_mutu_lokr_r5_fm_dyn15_25_${STAMP}.log"
echo "[flux3-r5-fm-dyn15-25] config: /root/UTrainer/UnifiedTrainer/configs/flux3_klein_mutu_xm5_test.json"
echo "[flux3-r5-fm-dyn15-25] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model flux3_klein \
    --config /root/UTrainer/UnifiedTrainer/configs/flux3_klein_mutu_xm5_test.json \
    2>&1 | tee "$LOG"
