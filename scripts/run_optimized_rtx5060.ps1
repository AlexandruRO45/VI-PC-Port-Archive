<#
.SYNOPSIS
    Optimized launcher and environment tuner for Leonida on NVIDIA RTX 5060 (8 GB VRAM) & Intel CC150 (8c/16t).

.DESCRIPTION
    Detects hardware parameters, checks Resizable BAR, configures memory budgeting and Wave64 emulation,
    and runs the Leonida runtime without triggering the title's 30-second initialization watchdog.

.EXAMPLE
    .\scripts\run_optimized_rtx5060.ps1 -LeonidaDir "Documents\Leonida" -Mode Run
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$LeonidaDir = "Documents\Leonida",

    [ValidateSet("Init", "Run", "DryRun")]
    [string]$Mode = "Run",

    [ValidateRange(0, 3)]
    [int]$TextureMipDrop = 1,

    [int]$ArenaMb = 256,

    [switch]$SkipBarCheck
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   LEONIDA HARDWARE OPTIMIZER: RTX 5060 + INTEL CC150   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. CPU Configuration (Intel CC150: 8 Cores / 16 Threads @ 3.5 GHz locked)
$cpuCores = [Environment]::ProcessorCount
Write-Host "[CPU] Detected $cpuCores logical processors." -ForegroundColor Green
if ($cpuCores -ge 16) {
    Write-Host "      Optimal parallel job pool: 16 threads." -ForegroundColor Gray
}

# 2. GPU Detection & Resizable BAR Check
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi -and -not $SkipBarCheck) {
    try {
        $smiOutput = & $nvidiaSmi.Source "-q" "-d" "MEMORY" 2>$null
        $bar1Line = ($smiOutput | Select-String "BAR1 Memory Usage" -Context 0, 4) -join "`n"
        if ($bar1Line -match "Total\s*:\s*(\d+)\s*MiB") {
            $bar1Mb = [int]$matches[1]
            if ($bar1Mb -le 512) {
                Write-Host "[GPU] Resizable BAR is currently DISABLED (BAR1 = $bar1Mb MiB)." -ForegroundColor Yellow
                Write-Host "      Recommended: Enable 'Above 4G Decoding' and 'Re-Size BAR Support' in BIOS." -ForegroundColor Yellow
                Write-Host "      This reduces PCIe 3.0 x8 staging bottlenecks for 8 GB VRAM." -ForegroundColor Yellow
            } else {
                Write-Host "[GPU] Resizable BAR is ENABLED (BAR1 = $bar1Mb MiB)." -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "[GPU] Note: could not query BAR1 size via nvidia-smi." -ForegroundColor Gray
    }
}

# 3. Memory & VRAM Safeguards (8 GB VRAM over PCIe 3.0 x8)
Write-Host "[VRAM] Configuring 8 GB VRAM safeguards:" -ForegroundColor Green
Write-Host "       - LEONIDA_TEXTURE_MIP_DROP = $TextureMipDrop (saves ~70% texture VRAM)" -ForegroundColor Gray
Write-Host "       - LEONIDA_ARENA_MB = $ArenaMb MiB (matches BAR boundary)" -ForegroundColor Gray
Write-Host "       - LEONIDA_W64_EMULATE = 1 (emulates 64-lane waves on 32-lane Blackwell SMs)" -ForegroundColor Gray

$env:LEONIDA_D3D12 = "vulkan"
$env:LEONIDA_TEXTURE_MIP_DROP = [string]$TextureMipDrop
$env:LEONIDA_ARENA_MB = [string]$ArenaMb
$env:LEONIDA_W64_EMULATE = "1"

# 4. Watchdog Protection: Avoid unbuffered stderr/stdout flooding
# Fixed 3.5 GHz CPU clock without Turbo can trip the title's 30s watchdog if flooded by synchronous disk I/O.
$env:LEONIDA_TRACE_BUFFERED = "1"

# 5. Locate Runtime Executable
$rtExe = Join-Path $LeonidaDir "gta6rt.exe"
if (-not (Test-Path $rtExe)) {
    # Check local build folder
    $localBuild = Join-Path $PSScriptRoot "..\runtime\build\gta6rt.exe"
    if (Test-Path $localBuild) {
        $rtExe = (Resolve-Path $localBuild).Path
    } else {
        Write-Host "ERROR: Could not find gta6rt.exe at $rtExe or in runtime\build." -ForegroundColor Red
        Write-Host "       Build the runtime first with scripts\build_runtime.ps1" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "[EXEC] Target runtime: $rtExe" -ForegroundColor Cyan
$argFlag = switch ($Mode) {
    "Init"   { "--init" }
    "Run"    { "--run" }
    "DryRun" { "--dry-run" }
}

Write-Host "[EXEC] Launching with mode: $argFlag" -ForegroundColor Cyan
Write-Host ""

& $rtExe $argFlag
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[STATUS] Execution finished successfully (code 0)." -ForegroundColor Green
} else {
    Write-Host "[STATUS] Runtime exited with code: $exitCode." -ForegroundColor Yellow
}
exit $exitCode
