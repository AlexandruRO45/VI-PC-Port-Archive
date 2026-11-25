# ztrx launcher: runs the game for normal play (no scripted presses, no capture).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data",
    [string]$Exe = "$PSScriptRoot\..\target\release\victory.exe",
    [string]$LogPath = "$env:TEMP\ztrx.log"
)
try {
    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        throw '-LogPath must name a file.'
    }
    $logParent = Split-Path -LiteralPath $LogPath -ErrorAction Stop
    if (($logParent -and -not (Test-Path -LiteralPath $logParent -PathType Container)) -or
        (Test-Path -LiteralPath $LogPath -PathType Container)) {
        throw "Log path must name a file in an existing directory: $LogPath"
    }

    foreach ($v in "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
    $env:VICTORY_CACHE_DIR = $Cache
    $env:VICTORY_DATA_DIR = $Data
    & $Exe run $Game --execute 2>&1 | Out-File -LiteralPath $LogPath -Encoding utf8 -ErrorAction Stop
} catch {
    Write-Error -ErrorRecord $_ -ErrorAction Continue
    exit 1
}
Write-Host "Exited with $LASTEXITCODE; log: $LogPath"
