param(
    [string]$OutputRoot = "ai_cv\outputs\benchmarks\phase04_loto"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$SourceRoot = Join-Path $RepositoryRoot (Join-Path $OutputRoot "source")
$ResultsRoot = Join-Path $RepositoryRoot (Join-Path $OutputRoot "results")

Set-Location -LiteralPath $RepositoryRoot

& $PythonExecutable `
    "ai_cv\phases\02_detection_tracking\src\experiment_classical_vertical_slice.py" `
    --practice-root "Practice_Dataset" `
    --starter-root "Package_starterkit\package_starterkit" `
    --output-root $SourceRoot `
    --opencv-threads 6 `
    --stereo-workers 1 `
    --stereo-roi-top 0
if ($LASTEXITCODE -ne 0) {
    throw "Candidate-trace generation failed with exit code $LASTEXITCODE"
}

& $PythonExecutable `
    "ai_cv\phases\02_detection_tracking\src\cross_validate_guarded_ttc.py" `
    --source-root $SourceRoot `
    --practice-root "Practice_Dataset" `
    --starter-root "Package_starterkit\package_starterkit" `
    --output-root $ResultsRoot
if ($LASTEXITCODE -ne 0) {
    throw "Leave-one-trip-out validation failed with exit code $LASTEXITCODE"
}
