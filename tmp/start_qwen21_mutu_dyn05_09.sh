#!/usr/bin/env bash
# UnifiedTrainer -> start: qwen21_mutu_lokr_r5_fm_dyn05_09
#
# Qwen-Image 2.1 LoKr training on the mutu dataset -- the qwen21 counterpart of
# the flux3 arm (start_flux3_mutu_xm5.sh / flux3_klein_mutu_xm5_test.json), same
# recipe, but on qwen21's OWN NATIVE timestep schedule (no high-noise boost).
#
#   data    : /home/waas/mutu (38 twitter jpg + captions, *_t.jpg / *_t.txt)
#             cache /home/waas/qwen21_cache_mutu_t2i, which is a symlink to the
#             cache the 0.75->1.25 run already built. REUSED, not rebuilt: the
#             cache holds latents + text embeddings and neither depends on the
#             shift sampling, so switching the schedule costs nothing here.
#   schedule: 20 epochs, lr 1e-4 constant (warmup 0), grad accumulation 1,
#             max_grad_norm 1, bf16
#   repeats : 5  -> ~36 train pairs/epoch x 5 = 180 steps/epoch, 3600 total
#   sampler : NATIVE dynamic shift, mu = 0.5 -> 0.9 over 256 -> 8192 image tokens
#             -> mu 0.694 at 1024x1024, mean sigma 0.639, 24.3% of draws above
#             0.8 (across the mutu buckets 768..1408px mean sigma 0.622..0.674).
#             These are the checkpoint's own scheduler_config.json values,
#             verbatim (use_dynamic_shifting=true, base_shift=0.5, max_shift=0.9),
#             i.e. the same schedule every previously-launched qwen21 run used.
#             HISTORY: the flux3-matching policy 1.5 -> 2.5 (mean sigma 0.842) and
#             its half 0.75 -> 1.25 (0.695) were BOTH trained to completion and
#             judged way too high for qwen21 -- hence back to native. flux3 keeps
#             1.5 -> 2.5, which sits on ITS native reference ln(6.93) = 1.936.
#             XM is OFF (noise_selector.type=random).
#   loss    : PLAIN flow_matching (use_weighting false) -- one forward/step,
#             unlike guide_flow_matching which ran a second uncond forward.
#   LoKr    : full-rank, factor 4 (matches the flux3 arm; the earlier qwen21
#             mutu recipe used factor 8)
#   output  : /home/waas/qwen21_mutu_lokr_r5_fm_dyn05_09_output/
#             qwen21_mutu_lokr_r5_fm_dyn05_09_epoch{N}.safetensors
#             (+ a ComfyUI-converted _comfyui.safetensors beside each one)
#
# WHY QUANTIZED, UNLIKE flux3:
#   flux3 ran quantize=none because its base+adapter was ~11.6 GB. qwen21 is not
#   in that class: the transformer alone is 14.2 GB of bf16 (2 shards, ~7.1B
#   params) and the text encoder is another 17 GB, against a 32 GB card. So this
#   keeps the proven qwen21 mutu memory recipe: quantize=torchao_float8 plus
#   offload_base_weights/offload_vae/offload_text_encoder. bitsandbytes is used
#   by optimizer=adamw8bit, which is why LD_LIBRARY_PATH is set below -- without
#   it bnb fails to load libnvJitLink.so.13.
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
#   bash /root/UTrainer/start_qwen21_mutu_dyn05_09.sh
# Detached:
#   nohup bash /root/UTrainer/start_qwen21_mutu_dyn05_09.sh >/dev/null 2>&1 &
# Resume FULL (weights+optimizer+step/epoch):
#   add  --resume-full /home/waas/qwen21_mutu_lokr_r5_fm_dyn05_09_output/<file>.safetensors
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
LOG="/root/UTrainer/logs/qwen21_mutu_lokr_r5_fm_dyn05_09_${STAMP}.log"
echo "[qwen21-r5-fm-dyn05-09] config: /root/UTrainer/UnifiedTrainer/configs/qwen21_mutu_dyn05_09_test.json"
echo "[qwen21-r5-fm-dyn05-09] log: $LOG"
$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_mutu_dyn05_09_test.json \
    2>&1 | tee "$LOG"
