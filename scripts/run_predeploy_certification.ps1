[CmdletBinding()]
param(
    [ValidateRange(1, 20)]
    [int]$Repeats = 5,
    [string]$OutputDir = "",
    [switch]$SkipOfficialScorer
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv_yolo26\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing deployment Python environment: $python"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $root "ai_cv\outputs\phase08_predeploy_final"
}

& $python (Join-Path $root "ai_cv\phases\06_robustness_latency\src\run_integrated_deployment.py") `
    --output-dir $OutputDir `
    --repeats $Repeats `
    --warmup-frames 100 `
    --opencv-threads 2 `
    --stereo-workers 2 `
    --latency-target-ms 75
if ($LASTEXITCODE -ne 0) { throw "Integrated deployment runner failed: $LASTEXITCODE" }

if (-not $SkipOfficialScorer) {
    & $python (Join-Path $root "Package_starterkit\package_starterkit\team_kit\evaluation.py") `
        --predictions (Join-Path $OutputDir "conservative_union") `
        --data-dir (Join-Path $root "Practice_Dataset") `
        --output (Join-Path $OutputDir "official_evaluation.json")
    if ($LASTEXITCODE -ne 0) { throw "Official scorer failed: $LASTEXITCODE" }
}

Write-Host "Pre-deploy certification complete: $OutputDir"
