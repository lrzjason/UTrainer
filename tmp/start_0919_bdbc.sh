#!/usr/bin/env bash
# UnifiedTrainer — start: 0919_bdbc
#
# Mixed interface run on /home/waas/breasts_size_control (655 _t/_bd/_bc pairs):
#   2x cap_bd + ref_BD  -> the black-canvas marker condition (position given)
#   2x cap_bc + ref_BC  -> the content-bearing condition (the marker region is
#                          blanked to grey in the target itself, so the model must
#                          synthesise the object into that hole)
# Both interfaces carry the SAME text (the size-free _bd.txt) and the same
# geometry: _bc blanks exactly the region _bd marks.
#
# NEW in this run: losses = region_flow_matching with
#   region_loss = {enable: true, weight: 4.0, modes: ["ref_BD"]}
# i.e. the ~4-9%-of-frame object region is weighted 4x in the latent loss on the
# BD steps ONLY (gated on the batch's resolved reference_config); BC steps stay
# unweighted.  Rationale: an unweighted latent loss over a 3.7% object gives
# almost no signal about the object's area -- measured over 12+ epochs, which is
# why the adapter still needed strength 1.5 at inference.
#
# LoKR WEIGHTS-ONLY init from the 0918 nosize/nohelios epoch-13 checkpoint ->
# fresh optimizer, step 0, epoch 0: same weights as the current chain, new
# objective.  20 epochs, lr 4e-4 constant, XM off (noise_selector random),
# helios off.  Validation is pinned to batch_configs[0] (BD) so val_loss and the
# validation images mean the same thing in every epoch.
#
# Cache: the BC role (_bc latent + webp + cap_bc embedding) is built by train.py
# on startup because the per-sample JSONs were removed; T/BD latents and the
# cap_t/cap_bd embeddings are reused as-is (~30-45 min before step 1).
#
# Output: /home/waas/0919_bdbc_output/0919_bdbc-{0..19}
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
$PY /root/UTrainer/UnifiedTrainer/train.py --model krea2 --config /root/UTrainer/UnifiedTrainer/configs/0919_bdbc.json
