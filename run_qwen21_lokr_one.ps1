# Qwen-Image 2.1 LoKr — single image, 1000 steps.
#
# 1 image (E:\images\test\one, 1 caption) x repeats=100 x 1 batch = 100 steps/epoch,
# 10 epochs = 1000 optimizer steps at lr 1e-4 (constant).
#
# VRAM (measured on this 16 GB RTX 5060 Ti, 15.90 GiB dedicated pool):
#   * the 7.115B transformer is loaded with bitsandbytes NF4 -> ~3.5 GB on GPU.
#     int8 was tried first and OOM'd for real at step 2 (14.75 GiB allocated);
#     block_swap 16 and 24 did NOT lower the transient backward peak because the
#     base weights were already offloaded. NF4 peaks at 5.96 GB and is ~1.4x
#     faster per step (3.7 s vs 5.9 s).
#   * `training.vram_limit_gb: 15.2` hard-caps the caching allocator. On Windows
#     an over-budget allocation would otherwise be silently backed by shared GPU
#     memory (system RAM over PCIe) instead of raising OOM — verified to raise a
#     clean torch.OutOfMemoryError instead.
#
# Usage (from E:\UnifiedTrainer\UTrainer):
#   .\run_qwen21_lokr_one.ps1
#   .\run_qwen21_lokr_one.ps1 -ResumeFull <...>_epoch4.safetensors
param(
    [string]$Resume = "",
    [string]$ResumeFull = ""
)
Set-Location $PSScriptRoot

$cfgPath = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\qwen_image21_lokr_one_1000.json"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot ".tmp\qwen21_lokr_one_$stamp.log"
Write-Host "[qwen21-lokr] config=$cfgPath log=$log"

$ErrorActionPreference = "Continue"
$argv = @("-m", "UnifiedTrainer.train", "--model", "qwen_image21",
          "--config", $cfgPath)
if ($ResumeFull -ne "") {
    # --resume-full restores weights + optimizer + scheduler + RNG + step/epoch,
    # so a resumed run continues the ORIGINAL 1000-step schedule instead of
    # restarting it. Needs the matching _training_state.pt alongside the file.
    $argv += @("--resume-full", $ResumeFull)
} elseif ($Resume -ne "") {
    # --resume loads adapter weights only; optimizer/scheduler/step restart fresh.
    $argv += @("--resume", $Resume)
}

# PS5.1: 2>&1 wraps python stderr lines as ErrorRecords; stringify them so the
# log stays clean text (same trick as smoke_pfm_nf4.ps1 / run_overfit_1img.ps1).
python @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
Write-Host "[qwen21-lokr] done. log: $log"
