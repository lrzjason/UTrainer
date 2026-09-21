# Qwen-Image 2.1 LoKr — DC-Gen corrected flow-matching objective.
#
# Trains a guidance-distilled checkpoint with the CORRECTED objective
# (guide_flow_matching, arXiv:2509.25180 Appendix A Eq. 10): the distilled
# model's own cond + empty-prompt outputs are combined into the raw velocity
# estimate  [v_cond + w * v_uncond] / (1 + w)  before matching the flow
# target. Plain flow_matching on such a checkpoint is biased (its output IS
# the CFG-combined velocity) — DC-Gen Fig. 9 measures CLIP 26.87 -> 27.38
# on MJHQ-30K with the fix.
#
# w is the guidance scale and MUST match the checkpoint's effective
# distillation scale w* (measured 1.5 for Qwen-Image 2.1 — see md/09 §7).
# Re-measure for a different checkpoint:
#   python -m UnifiedTrainer.utils.estimate_guidance_scale --config <cfg>
#
# Cost: TWO gradient-carrying forwards per step (cond + uncond). With NF4 +
# gradient checkpointing on the 16 GB 5060 Ti this still fits (measured peak
# 5.59 GB, ~6 s/step); watch peak.
#
# Usage (from E:\UnifiedTrainer\UTrainer):
#   .\run_qwen21_guide_fm.ps1
#   .\run_qwen21_guide_fm.ps1 -ResumeFull <...>_epoch4.safetensors
param(
    [string]$Resume = "",
    [string]$ResumeFull = ""
)
Set-Location $PSScriptRoot

$cfgPath = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\qwen_image21_guide_fm_lokr.json"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot ".tmp\qwen21_guide_fm_$stamp.log"
Write-Host "[qwen21-guide-fm] config=$cfgPath log=$log"

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
# log stays clean text (same trick as run_qwen21_lokr_one.ps1).
python @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
Write-Host "[qwen21-guide-fm] done. log: $log"
