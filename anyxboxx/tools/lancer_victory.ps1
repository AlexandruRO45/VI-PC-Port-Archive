<#
.SYNOPSIS
  Builds (if needed) and launches VIctory (third_party/victory) on the user's own game folder.

.PARAMETER GameDir
  Folder of the game package (kept outside this repository). It must contain game_scarlett_master.exe.

.PARAMETER SkipPipelines
  Optional. Passed unchanged to VICTORY_SKIP_PIPELINES when given.

.PARAMETER NoTunedSettings
  Optional. Does not set the tuned variables below (only VICTORY_GAME_DIR is set).

.PARAMETER Dred
  Optional. Turns on device-removed data (VICTORY_D3D12_DRED) to diagnose a device loss.
  Off by default: it notes every command list operation and slows each frame.

.NOTES
  The first cold launch may crash while the pipelines compile; launch again once the cache is filled.
  Settings measured on 2026-11-24 with the submodule's pinned commit (fxceless/victory-gta6).

.EXAMPLE
  .\tools\lancer_victory.ps1 -GameDir D:\Games\MyPackage
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [string]$SkipPipelines,
    [switch]$NoTunedSettings,
    [switch]$Dred
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$victory = Join-Path $repo 'third_party\victory'
$manifest = Join-Path $victory 'Cargo.toml'

if (-not (Test-Path -LiteralPath $manifest)) {
    throw "third_party/victory is empty: run 'git submodule update --init third_party/victory'."
}

$gameExe = Join-Path $GameDir 'game_scarlett_master.exe'
if (-not (Test-Path -LiteralPath $gameExe)) {
    throw "game_scarlett_master.exe not found in '$GameDir'."
}

# The submodule is its own Cargo workspace: its target folder is third_party\victory\target
# unless CARGO_TARGET_DIR is set.
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $victory 'target' }
$exe = Join-Path $target 'diag\victory.exe'

if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host 'Building VIctory (diag profile, diagnostics feature)...'
    & cargo build --profile diag -p victory --features diagnostics --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit code $LASTEXITCODE)." }
}

# No victory.toml is needed: VIctory reads the game folder from VICTORY_GAME_DIR.
$env:VICTORY_GAME_DIR = $GameDir
if (-not $NoTunedSettings) {
    $tuned = [ordered]@{
        # Translated shader loops stop after 256 iterations (32 cuts the hair strand loops short).
        VICTORY_SHADER_LOOP_LIMIT       = '256'
        # Host pipeline wait setting from the fork's pull request #1.
        VICTORY_LATE_WAIT               = '3'
        # Wide back-end option of the fork, on in the measured run.
        VICTORY_WIDE_BACK               = '1'
        # No relay mode on purpose: the run judged best had it off, and the fork's issue #89 reports that it makes
        # the characters' skin disappear and overlaps the subtitles. Set the relay variable by hand to try it.
        # Card buffer sizing: minimum and total, in MB.
        VICTORY_CARD_BUFFERS_MIN_MB     = '1'
        VICTORY_CARD_BUFFERS_MB         = '1024'
    }
    foreach ($name in $tuned.Keys) {
        Set-Item -Path "Env:$name" -Value $tuned[$name]
    }
}
# VIctory turns DRED on when the variable exists, whatever its value: only set it when asked.
if ($Dred) {
    $env:VICTORY_D3D12_DRED = '1'
}
if ($PSBoundParameters.ContainsKey('SkipPipelines') -and $SkipPipelines) {
    $env:VICTORY_SKIP_PIPELINES = $SkipPipelines
}

& $exe run $gameExe --execute
exit $LASTEXITCODE
