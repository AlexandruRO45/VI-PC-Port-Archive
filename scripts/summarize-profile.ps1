[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InputCsv,
    [ValidateRange(0, 3600000)][double]$StartSeconds = 0,
    [ValidateRange(0, 3600000)][double]$EndSeconds = 3600000
)
$ErrorActionPreference = 'Stop'
$rows = @(Import-Csv -LiteralPath $InputCsv)
if (@($rows | Where-Object { $_.schema_version -ne '1' }).Count) { throw 'Unsupported schema.' }
$presents = @($rows | Where-Object { $_.kind -eq 'present' })
for ($i = 0; $i -lt $presents.Count; $i++) {
    foreach ($name in @('sequence', 'elapsed_ns', 'interval_ns', 'lost_records')) {
        if ($presents[$i].$name -notmatch '^\d+$') { throw "Incomplete or invalid record: $name" }
    }
    if ([long]$presents[$i].lost_records -ne 0) { throw 'Lost records: percentiles would be biased.' }
    if ($i -gt 0 -and [long]$presents[$i].sequence -ne ([long]$presents[$i - 1].sequence + 1)) { throw 'Sequence gap: incomplete profile.' }
}
# Both ends of an interval must be inside the selected window.
$times = @($presents | Where-Object {
    [long]$_.sequence -gt 1 -and [double]$_.interval_ns -gt 0 -and
    (([double]$_.elapsed_ns - [double]$_.interval_ns) / 1e9) -ge $StartSeconds -and
    ([double]$_.elapsed_ns / 1e9) -le $EndSeconds
} | ForEach-Object { [double]$_.interval_ns / 1e6 } | Sort-Object -Descending)
if ($times.Count -lt 2) { throw 'At least two complete presentation intervals are required.' }
function Mean($values) { ($values | Measure-Object -Average).Average }
function TailFps($fraction) {
    $count = [Math]::Max(1, [Math]::Ceiling($times.Count * $fraction))
    1000.0 / (Mean @($times | Select-Object -First $count))
}
[ordered]@{
    schema_version = 1
    metric = 'successful application Present cadence; not scan-out or GPU time'
    intervals = $times.Count
    start_seconds = $StartSeconds
    end_seconds = $EndSeconds
    mean_frame_ms = Mean $times
    fps = 1000.0 / (Mean $times)
    low_1_percent_fps = TailFps 0.01
    low_0_1_percent_fps = TailFps 0.001
    low_0_1_percent_small_sample = ($times.Count -lt 1000)
    p99_frame_ms = $times[[Math]::Max(0, $times.Count - [Math]::Ceiling($times.Count * 0.99))]
    p99_9_frame_ms = $times[[Math]::Max(0, $times.Count - [Math]::Ceiling($times.Count * 0.999))]
} | ConvertTo-Json
