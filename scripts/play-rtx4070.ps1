# scripts\play.ps1 with settings for a 12 GB NVIDIA card (tested on an RTX 4070, 32 GB RAM),
# and the log kept in a file under %LOCALAPPDATA%\victory\logs.
#   scripts\play-rtx4070.ps1 -Exe <path of your own copy's executable>
# -Safe, -RayTracing and -More are passed on to play.ps1; -More wins over the settings below.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$Safe,
    [switch]$RayTracing,
    [hashtable]$More = @{},
    [string]$Log = (Join-Path $env:LOCALAPPDATA ("victory\logs\play-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date)))
)
$settings = @{
    # The console tells the title it is on a 1080p set instead of a 4K one (dxgi.rs display_size):
    # smaller render targets on a card with half the 7900 XTX's memory. Measured on the RTX 4070:
    # the card's memory in use fell from 99 % to 92 % of its budget, and textures let go and made
    # again from about 700 to about 20 a minute.
    VICTORY_DISPLAY_MODE = '1920x1080'
}
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }
& (Join-Path $PSScriptRoot 'play.ps1') -Exe $Exe -Safe:$Safe -RayTracing:$RayTracing -More $settings -Log $Log
