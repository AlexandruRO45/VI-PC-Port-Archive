<#
.SYNOPSIS
    Optimized launcher and environment tuner for Leonida on NVIDIA RTX 5060 (8 GB VRAM) & Intel CC150 (8c/16t).

.DESCRIPTION
    Detects hardware parameters, checks Resizable BAR, configures memory budgeting,
    Wave64 emulation, multimedia timer resolution, thread concurrency, and process priority,
    then runs the Leonida runtime (gta6rt.exe) with safety guards against watchdog timeouts.

.EXAMPLE
    .\scripts\run_optimized_rtx5060.ps1 -Image "game.exe" -Mode Run
    .\scripts\run_optimized_rtx5060.ps1 -Image "game.exe" -Affinity PhysicalOnly
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Image = "",

    [Parameter(Position = 1)]
    [string]$LeonidaDir = "Documents\Leonida",

    [ValidateSet("Init", "Run", "DryRun")]
    [string]$Mode = "Run",

    [ValidateRange(0, 3)]
    [int]$TextureMipDrop = 1,

    [int]$ArenaMb = 256,

    [ValidateSet("All", "PhysicalOnly")]
    [string]$Affinity = "All",

    [ValidateSet("Normal", "AboveNormal", "High")]
    [string]$Priority = "High",

    [switch]$SkipBarCheck,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# WinMM timeBeginPeriod / timeEndPeriod for 1 ms timer precision
$timePeriodCode = @"
using System;
using System.Runtime.InteropServices;
public static class TimePeriod {
    [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")]
    public static extern uint TimeBeginPeriod(uint uMilliseconds);
    [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")]
    public static extern uint TimeEndPeriod(uint uMilliseconds);
}
"@
if (-not ([System.Management.Automation.PSTypeName]'TimePeriod').Type) {
    Add-Type -TypeDefinition $timePeriodCode -Language CSharp
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   LEONIDA HARDWARE OPTIMIZER: RTX 5060 + INTEL CC150   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. CPU Configuration & Tuning (Intel CC150: 8 Cores / 16 Threads @ 3.5 GHz locked)
$cpuInfo = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
$cpuName = if ($cpuInfo) { $cpuInfo.Name.Trim() } else { "Unknown CPU" }
$cpuCores = [Environment]::ProcessorCount

Write-Host "[CPU] Detected: $cpuName ($cpuCores logical processors)." -ForegroundColor Green

$isIntel = $cpuName -match "Intel"
$isCC150 = $cpuName -match "CC150"

if ($isCC150) {
    Write-Host "      Profile: Intel CC150 (8C/16T Coffee Lake @ 3.50 GHz locked frequency)." -ForegroundColor Cyan
    Write-Host "      Watchdog mitigation: Active (pinned base clock requires unthrottled worker threads)." -ForegroundColor Gray
} elseif ($isIntel) {
    Write-Host "      Profile: Intel Architecture detected." -ForegroundColor Cyan
}

# Sizing host worker thread pools to 16 threads
$env:OMP_NUM_THREADS = [string]$cpuCores
$env:VK_LOADER_THREAD_MAX = [string]$cpuCores
$env:MKL_NUM_THREADS = [string]$cpuCores
$env:LEONIDA_THREAD_POOL_SIZE = [string]$cpuCores

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
    $localBuild = Join-Path $PSScriptRoot "..\runtime\build\RelWithDebInfo\gta6rt.exe"
    if (-not (Test-Path $localBuild)) {
        $localBuild = Join-Path $PSScriptRoot "..\runtime\build\gta6rt.exe"
    }
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
if ($Image) {
    Write-Host "[EXEC] Image target: $Image" -ForegroundColor Cyan
}

$callArgs = @($argFlag)
if ($Image) {
    $callArgs += $Image
}
if ($ExtraArgs) {
    $callArgs += $ExtraArgs
}

# Calculate affinity mask
# Intel CC150 has 8 cores / 16 threads.
# All = 0xFFFF (all 16 logical threads)
# PhysicalOnly = 0x5555 (even logical threads 0, 2, 4, 6, 8, 10, 12, 14 - physical cores only)
$affinityMask = [IntPtr]0xFFFF
if ($Affinity -eq "PhysicalOnly") {
    $affinityMask = [IntPtr]0x5555
    Write-Host "[EXEC] CPU Affinity: Physical cores only (mask 0x5555, 8 cores)." -ForegroundColor Cyan
} else {
    Write-Host "[EXEC] CPU Affinity: All logical threads (mask 0xFFFF, 16 threads)." -ForegroundColor Cyan
}

Write-Host "[EXEC] Process Priority: $Priority" -ForegroundColor Cyan
Write-Host "[TIMER] Enabling 1 ms multimedia timer period (winmm.dll)." -ForegroundColor Green
[TimePeriod]::TimeBeginPeriod(1) | Out-Null

$exitCode = 0
try {
    $escapedArgs = ($callArgs | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '(\\+)"', '$1$1\"' -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $rtExe
    $psi.Arguments = $escapedArgs
    $psi.UseShellExecute = $false

    $proc = [System.Diagnostics.Process]::Start($psi)
    if ($proc) {
        try {
            $proc.ProcessorAffinity = $affinityMask
            $proc.PriorityClass = switch ($Priority) {
                "AboveNormal" { [System.Diagnostics.ProcessPriorityClass]::AboveNormal }
                "High"        { [System.Diagnostics.ProcessPriorityClass]::High }
                Default       { [System.Diagnostics.ProcessPriorityClass]::Normal }
            }
        } catch {
            # Non-fatal if OS restricts priority adjustment
        }
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
    } else {
        $exitCode = 1
    }
} finally {
    [TimePeriod]::TimeEndPeriod(1) | Out-Null
    Write-Host "[TIMER] Restored multimedia timer period." -ForegroundColor Gray
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[STATUS] Execution finished successfully (code 0)." -ForegroundColor Green
} else {
    Write-Host "[STATUS] Runtime exited with code: $exitCode." -ForegroundColor Yellow
}
exit $exitCode
