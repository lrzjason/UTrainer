# MOSS-Image t2i LoKr full-rank factor=8 — ds dataset (54 images, .txt captions).
#
# Config: UnifiedTrainer\configs\moss_ds_lokr_fr8.json
# Template: moss_t2i_r1ona.json (proven local moss t2i LoKr setup), factor 4 -> 8,
#           plus vram_limit_gb 15.2 (Windows shared-memory guard, see
#           run_qwen21_lokr_one.ps1 notes).
#
# Usage (from E:\UnifiedTrainer\UTrainer):
#   .\run_moss_ds_lokr_fr8.ps1
param()
Set-Location $PSScriptRoot

$cfgPath = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\moss_ds_lokr_fr8.json"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot ".tmp\moss_ds_lokr_fr8_$stamp.log"
Write-Host "[moss-ds-lokr-fr8] config=$cfgPath log=$log"

$ErrorActionPreference = "Continue"
$argv = @("-m", "UnifiedTrainer.train", "--model", "moss",
          "--config", $cfgPath)

# PS5.1: 2>&1 wraps python stderr lines as ErrorRecords; stringify them so the
# log stays clean text (same trick as run_qwen21_lokr_one.ps1).
python @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
Write-Host "[moss-ds-lokr-fr8] done. log: $log"
