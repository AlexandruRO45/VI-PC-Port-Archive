# ztrx updater: pull the current branch, rebuild, and run the game.
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data"
)
$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot ".."
if (Get-Process -Name "victory" -ErrorAction SilentlyContinue) {
    throw "Close the running game before updating victory.exe."
}
Push-Location $root
try {
    git -c pull.rebase=false pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        throw "git -c pull.rebase=false pull --ff-only failed with exit code $LASTEXITCODE."
    }
    cargo build --release -p victory
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
& (Join-Path $PSScriptRoot "play.ps1") -Game $Game -Cache $Cache -Data $Data
