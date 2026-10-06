#!/usr/bin/env bash
# UnifiedTrainer — start: 0930_consis_3loss
#
# 无参数。全部训练配置（含续训）都只来自
#     /home/waas/qwen21_consis/qwen21_consis_lora_3loss.json
# 启动脚本不做任何解析、不读任何参数。
#
#     cd /root/UTrainer && nohup bash start_0930_consis_3loss.sh > /dev/null 2>&1 &
#     tail -f /root/UTrainer/logs/0930_consis_3loss_*.log
#
# 配置里与本次运行相关的：
#   resume      = {"checkpoint": ".../_lokr_epoch3.safetensors", "full": true}
#                 → train.py:1389 读取；删掉该段即从 epoch 0 全新训练
#   data        1 dataset_config / 5 batch_configs / 1 epoch = 1020 步
#               t2i 0.15 | noop 0.15 | hole 0.30 | recol 0.20 | blur 0.20
#               caption_dropout: t2i/noop 0.0，hole/recol/blur 0.1
#   losses      guide_flow_matching(1.5) + masked_flow_matching(edit 3.0)
#               + lcs(dim 6, w 0.5)
#   training    lr 1e-4 constant, 10 epochs, LoKr factor=4 full-rank,
#               adamw8bit + torchao_float8 + gradient checkpointing
#   cache_dir   /home/waas/qwen21_consis/cache/qwen21_consis_v2（已建好）
set -u
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
CONFIG=/home/waas/qwen21_consis/qwen21_consis_lora_3loss.json

NVIDIA_LIBS="$($PY -c 'import nvidia.cuda_nvrtc,os,glob; root=os.path.dirname(os.path.dirname(nvidia.cuda_nvrtc.__file__)); print(":".join(sorted(glob.glob(os.path.join(root,"*","lib")))))' 2>/dev/null)"
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
export WANDB_BASE_URL=https://api.bandw.top/
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG=/root/UTrainer/logs/0930_consis_3loss_$(date +%Y%m%d_%H%M%S).log
mkdir -p /root/UTrainer/logs
echo "config = $CONFIG   log = $LOG"

$PY -u /root/UTrainer/UnifiedTrainer/train.py \
    --model qwen_image21 \
    --config "$CONFIG" 2>&1 | tee "$LOG"
