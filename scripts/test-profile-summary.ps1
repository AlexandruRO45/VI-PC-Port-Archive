# Synthetic cadence data only; no title files or GPU required.
$ErrorActionPreference = 'Stop'
$temp = Join-Path ([IO.Path]::GetTempPath()) ('ztrx-profile-test-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    $csv = Join-Path $temp 'synthetic.csv'
    $rows = [Collections.Generic.List[object]]::new()
    $elapsed = 0L
    for ($i = 0; $i -le 1000; $i++) {
        $interval = if ($i -eq 0) { 0L } elseif ($i -le 990) { 10000000L } elseif ($i -lt 1000) { 20000000L } else { 100000000L }
        $elapsed += $interval
        $rows.Add([pscustomobject]@{ schema_version = 1; kind = 'present'; sequence = $i + 1; elapsed_ns = $elapsed; interval_ns = $interval; lost_records = 0 })
    }
    $rows | Export-Csv -LiteralPath $csv -NoTypeInformation
    $summary = (& "$PSScriptRoot/summarize-profile.ps1" -InputCsv $csv) | ConvertFrom-Json
    if ($summary.intervals -ne 1000 -or [Math]::Abs($summary.mean_frame_ms - 10.18) -gt 0.00001 -or
        $summary.median_frame_ms -ne 10 -or $summary.low_0_1_percent_fps -ne 10 -or [Math]::Abs($summary.low_1_percent_fps - (1000.0 / 28)) -gt 0.00001 -or
        $summary.p99_frame_ms -ne 10 -or $summary.p99_9_frame_ms -ne 20) { throw 'Incorrect percentile or tail-mean calculation.' }
    $rows[20].lost_records = 1
    $rows | Export-Csv -LiteralPath $csv -NoTypeInformation
    $rejected = $false
    try { & "$PSScriptRoot/summarize-profile.ps1" -InputCsv $csv | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Loss must invalidate the summary.' }
    $rows[20].lost_records = 0
    $rows[20].sequence = 22
    $rows | Export-Csv -LiteralPath $csv -NoTypeInformation
    $rejected = $false
    try { & "$PSScriptRoot/summarize-profile.ps1" -InputCsv $csv | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Sequence gaps must invalidate the summary.' }
    $rows[20].sequence = 21
    $rows | Export-Csv -LiteralPath $csv -NoTypeInformation
    Add-Content -LiteralPath $csv -Value '1,present,1002'
    $rejected = $false
    try { & "$PSScriptRoot/summarize-profile.ps1" -InputCsv $csv | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'A partial final row must invalidate the summary.' }
    Write-Output 'Profile summary: synthetic cadence, percentiles, loss, gaps and partial rows passed.'
} finally {
    # Only remove the two explicitly created paths, without recursive deletion.
    if (Test-Path -LiteralPath $csv) { Remove-Item -LiteralPath $csv }
    Remove-Item -LiteralPath $temp
}
