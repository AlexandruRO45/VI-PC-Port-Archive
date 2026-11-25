# ztrx launcher: runs the game for normal play (no scripted presses, no capture).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data",
    [string]$Exe = "$PSScriptRoot\..\target\release\victory.exe",
    # Leave the GPU choice to Windows instead of asking for the high-performance one.
    [switch]$KeepGpuChoice
)
foreach ($v in "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
$Exe = (Resolve-Path $Exe).Path

# Laptops with two GPUs: Windows gives a newly built exe the power-saving one
# (the integrated GPU), where the game runs out of video memory at once
# (E_OUTOFMEMORY). Ask for the high-performance GPU for this exe only, in the
# current user's per-application graphics preference (the same setting as
# Settings > System > Display > Graphics). Nothing system-wide is changed.
if (-not $KeepGpuChoice) {
    $key = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'
    if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
    New-ItemProperty -Path $key -Name $Exe -Value 'GpuPreference=2;' -PropertyType String -Force | Out-Null
}

New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
& $Exe run $Game --execute 2>&1 | Out-File "$env:TEMP\ztrx.log" -Encoding utf8
Write-Host "Exited with $LASTEXITCODE; log: $env:TEMP\ztrx.log"
