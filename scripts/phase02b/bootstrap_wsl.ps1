[CmdletBinding()]
param(
    [string]$DistributionName = "Ubuntu-22.04",
    [string]$InstallLocation = "D:\WSL\Ubuntu-22.04",
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Invoke-WslCapture {
    param([string[]]$Arguments)

    # Windows PowerShell wraps native stderr as ErrorRecord objects. Temporarily
    # use Continue so an expected nonzero WSL probe can be classified below.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & wsl.exe @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    $text = (($output | Out-String) -replace "`0", "").Trim()
    return [pscustomobject]@{
        ExitCode = $exitCode
        Text = $text
    }
}

function Get-InstalledDistributions {
    $result = Invoke-WslCapture -Arguments @("--list", "--quiet")
    if ($result.ExitCode -ne 0) {
        return @()
    }
    return @(
        $result.Text -split "\r?\n" |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
}

function Get-DistributionBasePath {
    param([string]$Name)

    $registryRoot = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss"
    if (-not (Test-Path -LiteralPath $registryRoot)) {
        return $null
    }

    foreach ($key in Get-ChildItem -LiteralPath $registryRoot) {
        $properties = Get-ItemProperty -LiteralPath $key.PSPath
        if ($properties.DistributionName -eq $Name) {
            return [Environment]::ExpandEnvironmentVariables(
                [string]$properties.BasePath
            )
        }
    }
    return $null
}

$wslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    throw "wsl.exe is unavailable. Windows 11 or a supported Windows 10 build is required."
}

$locationPath = [IO.Path]::GetFullPath($InstallLocation)
$locationRoot = [IO.Path]::GetPathRoot($locationPath)
if ($locationRoot -ne "D:\") {
    throw "Phase 2B must keep its WSL VHD on D:. Received: $locationPath"
}

$drive = Get-PSDrive -Name "D" -ErrorAction SilentlyContinue
if ($null -eq $drive) {
    throw "Drive D: is unavailable."
}

$freeGiB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "WSL distribution: $DistributionName"
Write-Host "Requested VHD location: $locationPath"
Write-Host "D: free space: $freeGiB GiB"

$installed = Get-InstalledDistributions
if ($installed -contains $DistributionName) {
    $basePath = Get-DistributionBasePath -Name $DistributionName
    Write-Host "$DistributionName is already installed."
    if ($basePath) {
        $resolvedBase = [IO.Path]::GetFullPath($basePath)
        Write-Host "Registered base path: $resolvedBase"
        if (-not $resolvedBase.StartsWith(
            $locationPath,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            Write-Warning (
                "The distro is not registered below $locationPath. " +
                "Use the verified export/import fallback in " +
                "docs\PHASE_02B_WSL_SETUP.md before installing large assets."
            )
        }
    } else {
        Write-Warning "The registered distro base path could not be resolved."
    }

    $details = Invoke-WslCapture -Arguments @("--list", "--verbose")
    if ($details.Text) {
        Write-Host $details.Text
    }
    Write-Host "No installation action is required."
    exit 0
}

$status = Invoke-WslCapture -Arguments @("--status")
$wslReady = $status.ExitCode -eq 0

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only; no Windows feature or distribution was changed."
    if (-not $wslReady) {
        Write-Host "First administrator pass:"
        Write-Host "  wsl.exe --install --no-distribution"
        Write-Host "Restart Windows if requested, then rerun this script with -Apply."
    } else {
        Write-Host "Administrator commands that would run:"
        Write-Host "  wsl.exe --update"
        Write-Host "  wsl.exe --set-default-version 2"
        Write-Host (
            "  wsl.exe --install --distribution $DistributionName " +
            "--location `"$locationPath`" --no-launch"
        )
    }
    exit 0
}

if (-not (Test-Administrator)) {
    throw "Rerun PowerShell as Administrator before using -Apply."
}

if (-not $wslReady) {
    Write-Host "Enabling WSL without placing a distribution on C: ..."
    & wsl.exe --install --no-distribution
    if ($LASTEXITCODE -ne 0) {
        throw "wsl.exe --install --no-distribution failed with exit code $LASTEXITCODE."
    }
    Write-Host ""
    Write-Host "WSL features were enabled. Restart Windows if requested."
    Write-Host "After restart, rerun this same command to install Ubuntu on D:."
    exit 10
}

Write-Host "Updating the Store-delivered WSL runtime ..."
& wsl.exe --update
if ($LASTEXITCODE -ne 0) {
    throw "wsl.exe --update failed with exit code $LASTEXITCODE."
}

& wsl.exe --set-default-version 2
if ($LASTEXITCODE -ne 0) {
    throw "Could not set WSL 2 as the default."
}

$help = Invoke-WslCapture -Arguments @("--help")
if ($help.Text -notmatch "(?m)--location\b") {
    throw (
        "This WSL build does not support --location. Run wsl.exe --update " +
        "and retry. If it remains unavailable, use the documented " +
        "export/import fallback; this script will not unregister a distro."
    )
}

$locationParent = Split-Path -Parent $locationPath
if (-not (Test-Path -LiteralPath $locationParent)) {
    New-Item -ItemType Directory -Path $locationParent | Out-Null
}
if (Test-Path -LiteralPath $locationPath) {
    $entries = @(Get-ChildItem -Force -LiteralPath $locationPath)
    if ($entries.Count -gt 0) {
        throw "Install location is non-empty and unregistered: $locationPath"
    }
}

Write-Host "Installing $DistributionName below D:\WSL ..."
& wsl.exe --install `
    --distribution $DistributionName `
    --location $locationPath `
    --no-launch
if ($LASTEXITCODE -ne 0) {
    throw "Ubuntu installation failed with exit code $LASTEXITCODE."
}

$installed = Get-InstalledDistributions
if ($installed -notcontains $DistributionName) {
    throw "$DistributionName was not visible after installation."
}

Write-Host ""
Write-Host "Ubuntu was registered successfully."
Write-Host "Launch it once to create the Linux user:"
Write-Host "  wsl.exe --distribution $DistributionName"
Write-Host "Then follow docs\PHASE_02B_WSL_SETUP.md."
