param(
    [switch]$SkipDatasetCheck
)

$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $workspace

$requiredPhases = @(
    "00_scope_interface",
    "01_data_baseline",
    "02_detection_tracking",
    "03_depth_motion",
    "04_ttc_corridor",
    "05_risk_events",
    "06_robustness_latency",
    "07_submission_handoff"
)

$requiredPhaseItems = @("README.md", "TASKS.md", "src", "tests", "verify", "artifacts", "notes")
$errors = New-Object System.Collections.Generic.List[string]

foreach ($phase in $requiredPhases) {
    $phasePath = Join-Path $workspace "phases\$phase"
    foreach ($item in $requiredPhaseItems) {
        $target = Join-Path $phasePath $item
        if (-not (Test-Path -LiteralPath $target)) {
            $errors.Add("Missing: $target")
        }
    }
}

if (-not $SkipDatasetCheck) {
    foreach ($dataset in @("Practice_Dataset", "Hackathon_Dataset_Redacted", "Package_starterkit")) {
        $target = Join-Path $repoRoot $dataset
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            $errors.Add("Missing dataset/starter directory: $target")
        }
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Host "Workspace structure: OK"
Write-Host "Phases checked: $($requiredPhases.Count)"
if ($SkipDatasetCheck) {
    Write-Host "Dataset/starter roots: SKIPPED"
} else {
    Write-Host "Dataset/starter roots: OK"
}
exit 0
