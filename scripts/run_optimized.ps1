<#
.SYNOPSIS
    Optimized launcher and environment tuner for Leonida on PC hardware.

.DESCRIPTION
    Dynamically inspects system hardware (GPU VRAM, Resizable BAR, CPU core topology),
    configures memory budgets, Wave64 emulation, thread concurrency, timer resolution,
    and process priority, then launches the Leonida runtime (gta6rt.exe) with safety guards
    against watchdog timeouts.

.EXAMPLE
    .\scripts\run_optimized.ps1
    .\scripts\run_optimized.ps1 -Image "game.exe" -Mode Run
    .\scripts\run_optimized.ps1 -Image "game.exe" -Affinity PhysicalOnly
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Image = "",

    [Parameter(Position = 1)]
    [string]$LeonidaDir = "Documents\Leonida",

    [ValidateSet("Init", "Run", "DryRun")]
    [string]$Mode = "Run",

    [int]$TextureMipDrop = -1,

    [int]$ArenaMb = -1,

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
public static class TimePeriodRunner {
    [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")]
    public static extern uint TimeBeginPeriod(uint uMilliseconds);
    [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")]
    public static extern uint TimeEndPeriod(uint uMilliseconds);
}
"@
if (-not ([System.Management.Automation.PSTypeName]'TimePeriodRunner').Type) {
    Add-Type -TypeDefinition $timePeriodCode -Language CSharp
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "            LEONIDA HARDWARE OPTIMIZER & RUNNER           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. CPU Configuration & Affinity Safeguards
$cpuInfo = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
$cpuName = if ($cpuInfo) { $cpuInfo.Name.Trim() } else { "Unknown CPU" }
$logicalCores = [Environment]::ProcessorCount

Write-Host "[CPU] Detected: $cpuName ($logicalCores logical processors)." -ForegroundColor Green

$isIntel = $cpuName -match "Intel"
$isCC150 = $cpuName -match "CC150"

if ($isCC150) {
    Write-Host "      Profile: Intel CC150 (8C/16T Coffee Lake @ 3.50 GHz locked frequency)." -ForegroundColor Cyan
    Write-Host "      Watchdog mitigation: Active (pinned base clock requires unthrottled worker threads)." -ForegroundColor Gray
} elseif ($isIntel) {
    Write-Host "      Profile: Intel Architecture detected." -ForegroundColor Cyan
}

# Sizing host worker thread pools
$env:OMP_NUM_THREADS = [string]$logicalCores
$env:VK_LOADER_THREAD_MAX = [string]$logicalCores
$env:MKL_NUM_THREADS = [string]$logicalCores
$env:LEONIDA_THREAD_POOL_SIZE = [string]$logicalCores

# 2. GPU Detection & Resizable BAR Check
$is8GbOrLess = $false
$bar1Mb = 0
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue

if ($nvidiaSmi -and -not $SkipBarCheck) {
    try {
        $smiMem = & $nvidiaSmi.Source "--query-gpu=memory.total" "--format=csv,noheader,nounits" 2>$null
        if ($smiMem -match "^\s*(\d+)") {
            $totalVramMb = [int]$matches[1]
            Write-Host "[GPU] Total VRAM: $totalVramMb MiB." -ForegroundColor Green
            if ($totalVramMb -le 8500) {
                $is8GbOrLess = $true
            }
        }
        $smiOutput = & $nvidiaSmi.Source "-q" "-d" "MEMORY" 2>$null
        $bar1Line = ($smiOutput | Select-String "BAR1 Memory Usage" -Context 0, 4) -join "`n"
        if ($bar1Line -match "Total\s*:\s*(\d+)\s*MiB") {
            $bar1Mb = [int]$matches[1]
            if ($bar1Mb -le 512) {
                Write-Host "[GPU] Resizable BAR: DISABLED (BAR1 = $bar1Mb MiB)." -ForegroundColor Yellow
                Write-Host "      Notice: Enable 'Above 4G Decoding' and 'Re-Size BAR Support' in BIOS" -ForegroundColor Yellow
                Write-Host "      to eliminate PCIe staging stalls for upload heaps in VRAM." -ForegroundColor Yellow
            } else {
                Write-Host "[GPU] Resizable BAR: ENABLED (BAR1 = $bar1Mb MiB)." -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "[GPU] Could not query GPU memory properties via nvidia-smi." -ForegroundColor Gray
    }
}

# 3. Memory & VRAM Budgeting
if ($TextureMipDrop -lt 0) {
    if ($is8GbOrLess) {
        $TextureMipDrop = 1
        Write-Host "[VRAM] Auto-configured TextureMipDrop = 1 for <=8 GB VRAM budget." -ForegroundColor Yellow
    } else {
        $TextureMipDrop = 0
    }
}

if ($ArenaMb -lt 0) {
    if ($is8GbOrLess) {
        $ArenaMb = 256
    } else {
        $ArenaMb = 512
    }
}

Write-Host "[TUNING] Applying environment parameters:" -ForegroundColor Cyan
Write-Host "         LEONIDA_D3D12 = vulkan" -ForegroundColor Gray
Write-Host "         LEONIDA_TEXTURE_MIP_DROP = $TextureMipDrop" -ForegroundColor Gray
Write-Host "         LEONIDA_ARENA_MB = $ArenaMb" -ForegroundColor Gray
Write-Host "         LEONIDA_W64_EMULATE = 1" -ForegroundColor Gray
Write-Host "         LEONIDA_TRACE_BUFFERED = 1" -ForegroundColor Gray

$env:LEONIDA_D3D12 = "vulkan"
$env:LEONIDA_TEXTURE_MIP_DROP = [string]$TextureMipDrop
$env:LEONIDA_ARENA_MB = [string]$ArenaMb
$env:LEONIDA_W64_EMULATE = "1"
$env:LEONIDA_TRACE_BUFFERED = "1"

if (-not $Image) {
    Write-Host ""
    Write-Host "[CONFIG] Environment configured successfully." -ForegroundColor Green
    Write-Host "         To launch an executable, pass: .\scripts\run_optimized.ps1 -Image <game.exe>" -ForegroundColor Cyan
    exit 0
}

# 4. Locate Runtime Executable
$rtExe = Join-Path $LeonidaDir "gta6rt.exe"
if (-not (Test-Path $rtExe)) {
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

Write-Host "[EXEC] Mode: $argFlag" -ForegroundColor Cyan
Write-Host "[EXEC] Image target: $Image" -ForegroundColor Cyan

$callArgs = @($argFlag, $Image)
if ($ExtraArgs) {
    $callArgs += $ExtraArgs
}

# Calculate affinity mask
$affinityMask = [IntPtr]0xFFFF
if ($Affinity -eq "PhysicalOnly") {
    $affinityMask = [IntPtr]0x5555
    Write-Host "[EXEC] CPU Affinity: Physical cores only (mask 0x5555)." -ForegroundColor Cyan
} else {
    Write-Host "[EXEC] CPU Affinity: All logical threads (mask 0xFFFF)." -ForegroundColor Cyan
}

Write-Host "[EXEC] Process Priority: $Priority" -ForegroundColor Cyan
Write-Host "[TIMER] Enabling 1 ms multimedia timer period (winmm.dll)." -ForegroundColor Green
[TimePeriodRunner]::TimeBeginPeriod(1) | Out-Null

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
    [TimePeriodRunner]::TimeEndPeriod(1) | Out-Null
    Write-Host "[TIMER] Restored multimedia timer period." -ForegroundColor Gray
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[STATUS] Execution finished successfully (code 0)." -ForegroundColor Green
} else {
    Write-Host "[STATUS] Runtime exited with code: $exitCode." -ForegroundColor Yellow
}
exit $exitCode
