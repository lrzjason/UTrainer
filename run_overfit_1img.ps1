# 过拟合冒烟：单图 x repeats100 x 20 epochs（Krea-2 NF4 + PFM）
#
# 用法（在 E:\UnifiedTrainer\UTrainer 下）：
#   .\run_overfit_1img.ps1                              # dinov2_base @448
#   .\run_overfit_1img.ps1 -Encoder dinov3              # dinov3_convnext_small @512
#
# 观察：train/val loss 应随训练持续下降并趋近 0（过拟合成立）；
#   wandb 项目 UnifiedTrainer / run overfit_1img_{dinov2|dinov3}；
#   val 用同一张图（cache_manager 单样本分支：train==val）；
#   每个 epoch 存一份 checkpoint 到 E:\images\pfm\overfit_1_out
# 日志：.tmp\overfit_1img_<时间戳>.log
param(
    [ValidateSet("dinov2", "dinov3")]
    [string]$Encoder = "dinov2"
)
Set-Location $PSScriptRoot

# 显存碎片防护（与 smoke_pfm_nf4.ps1 相同）
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$cfgPath = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\overfit_1img_$Encoder.json"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot ".tmp\overfit_1img_$stamp.log"
Write-Host "[overfit] config=$cfgPath log=$log"

# PS5.1: 2>&1 wraps python stderr lines as ErrorRecords; stringify them so
# the log stays clean text (same trick as smoke_pfm_nf4.ps1).
$ErrorActionPreference = "Continue"
python -m UnifiedTrainer.train --model krea2 --config $cfgPath 2>&1 |
    ForEach-Object { "$_" } |
    Tee-Object -FilePath $log
Write-Host "[overfit] done. log: $log"
