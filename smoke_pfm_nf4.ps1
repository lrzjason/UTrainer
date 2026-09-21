# 冒烟测试启动器：本地 NF4 + PFM（Krea-2，5 组 test_5 数据）
#
# 用法（在 E:\UnifiedTrainer\UTrainer 下）：
#   .\smoke_pfm_nf4.ps1                 # 5 步冒烟，复用已有缓存
#   .\smoke_pfm_nf4.ps1 -RebuildCache   # 换了 test_5 里的数据后，删缓存强制重建
#
# 改训练步数/epoch：编辑 configs\local_pfm_nf4_test.json 的
#   training.max_steps / training.num_epochs
param(
    [switch]$RebuildCache
)
Set-Location $PSScriptRoot

# 显存碎片防护：reserved 曾冲到 15.1G/15.9G（expandable segments 缓解 OOM）
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$cacheDir = "E:\images\depth\test_5_cache"
if ($RebuildCache) {
    if (Test-Path $cacheDir) {
        Remove-Item $cacheDir -Recurse -Force
        Write-Host "[smoke] removed cache: $cacheDir"
    }
    # 重建还需要打开 recreate_* 开关：写一个运行时配置副本
    $cfgPath = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\local_pfm_nf4_test.json"
    $raw = Get-Content $cfgPath -Raw
    $raw = $raw -replace '"recreate_cache": false', '"recreate_cache": true'
    $raw = $raw -replace '"recreate_latents": false', '"recreate_latents": true'
    $raw = $raw -replace '"recreate_embeddings": false', '"recreate_embeddings": true'
    $cfgPath = Join-Path $env:TEMP "local_pfm_nf4_test_rebuild.json"
    # 内容纯 ASCII，用 ascii 编码避免 BOM 干扰 json.load
    Out-File -FilePath $cfgPath -InputObject $raw -Encoding ascii
    Write-Host "[smoke] rebuild config: $cfgPath"
}
else {
    $cfgPath = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\local_pfm_nf4_test.json"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot ".tmp\smoke_pfm_$stamp.log"
Write-Host "[smoke] config=$cfgPath log=$log"

# PS5.1: 2>&1 wraps python stderr lines as ErrorRecords; with $EAP=Stop the
# first stderr line (pynvml FutureWarning) would kill the run. Keep Continue
# and stringify records so the log stays clean text.
$ErrorActionPreference = "Continue"
python -m UnifiedTrainer.train --model krea2 --config $cfgPath 2>&1 |
    ForEach-Object { "$_" } |
    Tee-Object -FilePath $log
Write-Host "[smoke] done. log: $log"
