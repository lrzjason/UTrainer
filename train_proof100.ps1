# proof_100 comparison trainer: FM-MSE vs PFM (local NF4).
#
# Usage (from E:\UnifiedTrainer\UTrainer):
#   .\train_proof100.ps1 -Which fm               # FM-MSE run only (proof_100)
#   .\train_proof100.ps1 -Which pfm              # PFM run only (proof_100)
#   .\train_proof100.ps1 -Which both             # sequential: pfm then fm (default)
#   .\train_proof100.ps1 -Tag proof_10 -Which pfm   # quick code-validation run
#
# First run on a fresh dataset builds the cache (~2-3 min per 100 groups).
# Prerequisite: `wandb login` done once (logs are sent to wandb).
param(
    [ValidateSet("fm", "pfm", "both")]
    [string]$Which = "both",
    [string]$Tag = "proof_100"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# wandb via bandw.top proxy (mirrors api.wandb.ai; SDK reads this env natively)
$env:WANDB_BASE_URL = "https://api.bandw.top/"

# Preflight: wandb logs must reach the cloud; fail loudly before a 4h run.
# Windows wandb stores the key in %USERPROFILE%\_netrc (underscore!) -- the
# dot-prefixed .netrc is the Unix convention; check both.
$hasCred = $env:WANDB_API_KEY `
    -or (Test-Path "$env:USERPROFILE\.netrc") `
    -or (Test-Path "$env:USERPROFILE\_netrc") `
    -or (Test-Path "$env:USERPROFILE\.config\wandb")
if (-not $hasCred) {
    Write-Warning "No wandb credentials found. Run 'wandb login' once (or set WANDB_API_KEY) before training."
    exit 1
}
Write-Host "[wandb] credentials found; base URL: $env:WANDB_BASE_URL"

# VRAM fragmentation guard: expandable segments keep `reserved` from creeping
# up over long runs (smoke test saw 15.1G/15.9G reserved without it).
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$targets = @("pfm", "fm")
if ($Which -ne "both") { $targets = @($Which) }

foreach ($t in $targets) {
    $cfg = "E:\UnifiedTrainer\UTrainer\UnifiedTrainer\configs\${Tag}_$t.json"
    $log = Join-Path $PSScriptRoot ".tmp\${Tag}_${t}_$stamp.log"
    Write-Host "=== [$Tag] training $t ===  log: $log"
    # PS5.1: 2>&1 wraps python stderr lines as ErrorRecords and $EAP=Stop
    # kills the run on the first one (e.g. the pynvml FutureWarning). Keep
    # EAP=Continue for the pipe and stringify every record for clean logs.
    $ErrorActionPreference = "Continue"
    python -m UnifiedTrainer.train --model krea2 --config $cfg 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "run '$t' exited with $LASTEXITCODE -- see $log"
    }
}
Write-Host "=== [$Tag] all done ==="
