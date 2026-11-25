# ztrx launcher: runs the game for normal play.
#
# Scripted input and captures are opt-in, because clearing them by default made
# the launcher unusable for the diagnostics they exist for (issue #19):
#   -Capture   keep VICTORY_CAPTURE, VICTORY_CAPTURE_FRAMES, VICTORY_DUMP_TARGETS
#   -Keys      keep VICTORY_KEYS
# or -KeepScriptedInput for both. Without them the four are cleared, which is
# what "no scripted presses, no capture" means for a normal run.
#
# Exits with victory.exe's own code, so a failed launch is a failed sample in
# batch runs (issue #3).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data",
    [string]$Exe = "$PSScriptRoot\..\target\release\victory.exe",
    [switch]$Capture,
    [switch]$Keys,
    [switch]$KeepScriptedInput
)
$capture = $Capture -or $KeepScriptedInput
$keys = $Keys -or $KeepScriptedInput
if (-not $capture) {
    foreach ($v in "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
}
if (-not $keys) {
    Remove-Item "Env:VICTORY_KEYS" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
& $Exe run $Game --execute 2>&1 | Out-File "$env:TEMP\ztrx.log" -Encoding utf8
# A command that could not be started at all (the executable is not there)
# leaves `$LASTEXITCODE` untouched, so a plain read of it reports the previous
# command's 0. `$?` says whether the call ran.
$code = if ($?) { $LASTEXITCODE } else { 1 }
Write-Host "Exited with $code; log: $env:TEMP\ztrx.log"
# The child's own code: `-File scripts\play.ps1` then reports a failed launch.
exit $code
