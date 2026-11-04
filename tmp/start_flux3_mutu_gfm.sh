#!/usr/bin/env bash
# UnifiedTrainer -> start: flux3_mutu_lokr_r2_gfm_dyn15_25
#
# FLUX.3 klein (base / non-distilled) LoKr training on the mutu dataset --
# CURRENT TIMESHIFT + GUIDE CFG FLOW MATCHING. Identical to the
# flux3_mutu_lokr_r2_fm_dyn15_25 run except for the loss, so the loss is the
# only variable between the two.
#   data    : /home/waas/mutu (38 twitter jpg + captions, *_t.jpg / *_t.txt)
#             cache /home/waas/flux3_klein_cache_mutu_t2i (REUSED; it already
#             holds empty_embedding.flux3_klein.npz, which this loss REQUIRES --
#             train.py hard-fails if it is absent)
#   schedule: 20 epochs, lr 1e-4 constant (warmup 0), grad accumulation 1,
#             max_grad_norm 1, bf16
#   repeats : 2  -> ~36 train pairs/epoch x 2 = 72 steps/epoch, 1440 total
#   sampler : DYNAMIC high-noise, mu = 1.5 -> 2.5 over 256 -> 8192 image tokens
#             -> mu ~1.98 at 1024x1024, mean sigma ~0.84. UNCHANGED from the
#             fm arm (it was judged 'just right' for flux3).
#   loss    : guide_flow_matching (DC-Gen corrected CFG, arXiv:2509.25180 Eq.10),
#             w ~ U[1,8], v_corrected = (v_cond + w*v_uncond)/(1+w).
#             needs_uncond_forward=True -> a SECOND empty-prompt forward per step,
#             WITH gradients (both are the trained network), so ~2x forward work
#             and two live autograd graphs. ~10% of steps skip it via
#             caption_dropout (0.1) reusing the conditional pass.
#   LoKr    : full-rank, factor 4 (346,157,296 trainable, 6.25% of the 5.54B base)
#   output  : /home/waas/flux3_mutu_lokr_r2_gfm_dyn15_25_output/
#             flux3_mutu_lokr_r2_gfm_dyn15_25_epoch{N}.safetensors
#             (+ a ComfyUI-converted _comfyui.safetensors beside each one)
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
# blocks are recomputed in backward (~20-30% slower per step) and plain FM peaked
# at 22.66 GB. The extra uncond forward here is the new risk: if it OOMs, drop
# lokr_factor to 8 (346M -> 86.5M trainable, frees ~3.6 GB of AdamW state) and/or
# set block_swap.
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
#   bash /root/UTrainer/start_flux3_mutu_gfm.sh
# Detached:
#   nohup bash /root/UTrainer/start_flux3_mutu_gfm.sh >/dev/null 2>&1 &
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
LOG="/root/UTrainer/logs/flux3_mutu_lokr_r2_gfm_dyn15_25_${STAMP}.log"
echo "[flux3-r2-gfm-dyn15-25] config: /root/UTrainer/UnifiedTrainer/configs/flux3_mutu_gfm_dyn15_25_test.json"
echo "[flux3-r2-gfm-dyn15-25] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model flux3_klein \
    --config /root/UTrainer/UnifiedTrainer/configs/flux3_mutu_gfm_dyn15_25_test.json \
    2>&1 | tee "$LOG"
