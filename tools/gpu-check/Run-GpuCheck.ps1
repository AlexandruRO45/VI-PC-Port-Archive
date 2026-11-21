#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('hardware', 'warp')]
    [string]$Backend = 'hardware',
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
if (-not $ReportPath) { $ReportPath = Join-Path $PSScriptRoot "gpu-report-$Backend.md" }
$binary = (Get-Item -LiteralPath (Join-Path $PSScriptRoot 'offscreen_check.exe')).FullName
$revision = (Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'SOURCE_COMMIT.txt')).Trim()
if ($revision -notmatch '^[0-9a-f]{40}$') { throw 'The package is missing a valid source revision.' }
$windows = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$report = [System.Collections.Generic.List[string]]::new()
$report.Add('## Xbox2Windows GPU report')
$report.Add('')
$report.Add("Source commit: $revision")
$report.Add("Windows build: $($windows.CurrentBuildNumber).$($windows.UBR)")
$report.Add("Backend: $Backend")
$report.Add('')
$failed = $false
foreach ($test in 'dual-uav', 'texture-mips', 'bvh-waves', 'bvh-wave64', 'bvh-shuffle', 'bvh-shuffle-16', 'bvh-shuffle-32', 'bvh-shuffle-64', 'bvh-differential', 'bvh-packed', 'bvh-atomics', 'bvh-numeric', 'bvh-root-roundtrip', 'dxr-update') {
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = $binary
    $process.StartInfo.Arguments = "$Backend --$test"
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $timedOut = -not $process.WaitForExit(60000)
        if ($timedOut) {
            $process.Kill()
            $process.WaitForExit()
        }
        $details = ($stdout.Result.Trim() + [Environment]::NewLine + $stderr.Result.Trim()).Trim()
        if ($timedOut) { $status = 'FAIL'; $details += "`nTimed out after 60 seconds." }
        elseif ($process.ExitCode -eq 0) { $status = 'PASS' }
        elseif ($process.ExitCode -eq 77) { $status = 'SKIP' }
        else { $status = 'FAIL'; $details += "`nProcess exit code: $($process.ExitCode)" }
    } catch {
        $status = 'FAIL'
        $details = $_.Exception.Message
    } finally {
        $process.Dispose()
    }
    if ($status -eq 'FAIL') { $failed = $true }
    $caseName = $test.Replace('-', '_')
    $report.Add("### ${caseName}_${Backend} - $status")
    $report.Add('')
    $report.Add('```text')
    $report.Add($details)
    $report.Add('```')
    $report.Add('')
}
$report | Write-Output
[IO.File]::WriteAllLines($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ReportPath), $report)
Write-Output "Saved report: $ReportPath"
if ($failed) { exit 1 }
exit 0
