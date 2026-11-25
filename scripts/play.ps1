# ztrx launcher: runs the game for normal play (no scripted presses, no capture).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data",
    [string]$Exe
)

# Resolve the launcher directory in the body: some PowerShell hosts leave
# $PSScriptRoot empty while parameter defaults are being evaluated.
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptPath = $PSCommandPath
    if (-not $scriptPath) {
        $scriptPath = $MyInvocation.MyCommand.Path
    }
    if ($scriptPath) {
        $scriptDir = Split-Path -Parent $scriptPath
    }
}
if (-not $scriptDir) {
    $scriptDir = (Get-Location).Path
}

if (-not $Exe) {
    $candidates = @(
        (Join-Path $scriptDir '..\target\release\victory.exe'),
        (Join-Path $scriptDir '..\target\diag\victory.exe'),
        (Join-Path $scriptDir 'victory.exe'),
        (Join-Path (Get-Location).Path 'target\release\victory.exe')
    )
    $Exe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $Exe) {
        $looked = $candidates -join "`n  "
        throw "victory.exe not found. Looked in:`n  $looked`nPass -Exe <path>, or build with: cargo build --release -p victory"
    }
}

$Exe = [System.IO.Path]::GetFullPath($Exe)
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
    throw "victory.exe not found: $Exe"
}

foreach ($v in "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
& $Exe run $Game --execute 2>&1 | Out-File "$env:TEMP\ztrx.log" -Encoding utf8
Write-Host "Exited with $LASTEXITCODE; log: $env:TEMP\ztrx.log"
