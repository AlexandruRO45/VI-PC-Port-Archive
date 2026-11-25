# Run from a reviewed checkout. Outputs and title-derived data stay outside it.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$Cache,
    [Parameter(Mandatory = $true)][string]$Data,
    [string]$Exe = "$PSScriptRoot\..\target\diag\victory.exe",
    [ValidateRange(10, 3600)][int]$Seconds = 120,
    [string]$Keys = '',
    [string]$Scene = 'unspecified',
    [string]$NvidiaSmi = '',
    [ValidateSet('cold', 'warm', 'unknown')][string]$CacheState = 'unknown',
    [switch]$Profile
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
foreach ($path in @($Output, $Cache, $Data)) {
    $absolute = [IO.Path]::GetFullPath($path)
    if ($absolute.TrimEnd('\') -eq $repo -or $absolute.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Benchmark outputs, caches and saves must be outside the repository: $path"
    }
    $ancestor = $absolute
    while ($ancestor) {
        if (Test-Path -LiteralPath (Join-Path $ancestor '.victory-root')) { throw "Source tree destination: $path" }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
}
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$Game = (Resolve-Path -LiteralPath $Game).Path
$Output = [IO.Path]::GetFullPath($Output)
$Cache = [IO.Path]::GetFullPath($Cache)
$Data = [IO.Path]::GetFullPath($Data)
if (Test-Path -LiteralPath $Output) { throw 'Choose a new output directory for each run.' }
foreach ($path in @($Output, $Cache, $Data)) {
    $drive = [IO.Path]::GetPathRoot($path)
    if (([IO.DriveInfo]::new($drive)).AvailableFreeSpace -lt 16GB) { throw "Less than 16 GiB free on $drive" }
}
New-Item -ItemType Directory -Path $Output | Out-Null
foreach ($path in @($Cache, $Data)) { New-Item -ItemType Directory -Force -Path $path | Out-Null }
$saved = @{}
$names = @('VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_KEYS', 'VICTORY_PROFILE_DIR', 'VICTORY_LOG')
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$process = $null
$gpuMonitor = $null
try {
    $env:VICTORY_CACHE_DIR = $Cache
    $env:VICTORY_DATA_DIR = $Data
    $env:VICTORY_KEYS = $Keys
    $env:VICTORY_LOG = 'info'
    $env:VICTORY_PROFILE_DIR = if ($Profile) { Join-Path $Output 'profile' } else { $null }
    $metadata = [ordered]@{
        schema_version = 1; commit = (& git -C $repo rev-parse HEAD); dirty = [bool](& git -C $repo status --porcelain)
        exe_sha256 = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
        exe = $Exe; scene = $Scene; cache_state = $CacheState; cache = $Cache; data = $Data
        seconds_requested = $Seconds; started_utc = [DateTime]::UtcNow.ToString('o')
        environment = @(Get-ChildItem Env:VICTORY* | Sort-Object Name | Select-Object Name, Value)
        os = [Environment]::OSVersion.VersionString
        cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfLogicalProcessors)
        gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
        ram_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        free_bytes = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($Cache))).AvailableFreeSpace
        nvidia_smi = $NvidiaSmi
    }
    $metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Output 'metadata.json') -Encoding utf8
    # Start-Process on Windows flattens arguments. Explicit quotes preserve spaces.
    if ($Game.Contains('"')) { throw 'A game path cannot contain a quote.' }
    $process = Start-Process -FilePath $Exe -ArgumentList @('run', ('"' + $Game + '"'), '--execute') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Output 'stdout.log') -RedirectStandardError (Join-Path $Output 'stderr.log')
    if ($NvidiaSmi) {
        $monitorExe = (Resolve-Path -LiteralPath $NvidiaSmi).Path
        # Device-wide utilization and VRAM use, not per-process GPU execution time.
        $gpuMonitor = Start-Process -FilePath $monitorExe -ArgumentList @('--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv', '-l', '1') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Output 'gpu.csv') -RedirectStandardError (Join-Path $Output 'gpu-monitor.log')
    }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $samples = [Collections.Generic.List[object]]::new()
    while ($watch.Elapsed.TotalSeconds -lt $Seconds -and -not $process.HasExited) {
        $process.Refresh()
        $samples.Add([pscustomobject]@{
            elapsed_s = $watch.Elapsed.TotalSeconds.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
            cpu_total_s = $process.TotalProcessorTime.TotalSeconds.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
            working_set_bytes = $process.WorkingSet64
            private_bytes = $process.PrivateMemorySize64
        })
        Start-Sleep -Milliseconds 1000
    }
    $stoppedAtDeadline = -not $process.HasExited
    if ($stoppedAtDeadline) { Stop-Process -Id $process.Id }
    $process.WaitForExit()
    $samples | Export-Csv -LiteralPath (Join-Path $Output 'process.csv') -NoTypeInformation
    [ordered]@{ elapsed_s = $watch.Elapsed.TotalSeconds; stopped_at_deadline = $stoppedAtDeadline; exit_code = $process.ExitCode } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'result.json') -Encoding utf8
} finally {
    if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id }
    if ($gpuMonitor -and -not $gpuMonitor.HasExited) { Stop-Process -Id $gpuMonitor.Id }
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
