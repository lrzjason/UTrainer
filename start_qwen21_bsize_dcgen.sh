#!/usr/bin/env bash
# UnifiedTrainer -- qwen_image21 (Qwen-Image 2.1) test run.
#
#   model   : qwen_image21      (unpatched 64ch latent, block-causal, RGBA VAE)
#   dataset : breasts_size_control, same bsize_edit setup as the last krea2 run
#             (0920_style.json), but a FRESH cache -- qwen21 latents are 64-channel
#             and its text embeddings are Qwen3-VL 4096-d, so the krea2 cache is
#             unusable (and `reference_list.resize` must equal the resolution).
#   loss    : guide_flow_matching -- DC-Gen (arXiv:2509.25180) Eq.10 corrected
#             objective for GUIDANCE-DISTILLED checkpoints. Qwen-Image 2.1 is one,
#             so the plain flow-matching target is biased; this loss recovers the
#             raw velocity from two forwards (needs_uncond_forward).
#   schedule: lr 1e-4 CONSTANT over 20 epochs
#   network : LoKr, qwen21 pattern preset
#   quantize: torchao_float8 + offload_base_weights (the krea2-proven combo on this
#             5090; remote train.py carries the torchao/offload prepare() fix)
#
# Output: /home/waas/qwen21_bsize_dcgen_output/qwen21_bsize_dcgen-{0..19}
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
$PY /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config /root/UTrainer/UnifiedTrainer/configs/qwen21_bsize_dcgen.json
