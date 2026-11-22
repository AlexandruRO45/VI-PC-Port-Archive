# Carves every .rpf in the game folder into <Out>\<archive path with _>, smallest first (research tool; the
# build itself doesn't need it). Skips archives that already have an index.json. Summary: <Out>\coverage.json
param([Parameter(Mandatory)][string]$Game, [Parameter(Mandatory)][string]$Out)
$root = Split-Path $PSScriptRoot -Parent
$tool = Join-Path $root "tools\rpf9_carve.py"
# -Game may be relative, end in a separator or hold an 8.3 short name (C:\Users\RUNNER~1). Get-ChildItem -Name gives
# each archive's path relative to -Game, so names never depend on cutting a prefix off FullName, which Windows
# returns with long names.
$Game = (Resolve-Path -LiteralPath $Game).ProviderPath
$archives = foreach ($rel in Get-ChildItem -LiteralPath $Game -Recurse -File -Filter *.rpf -Name) {
    [pscustomobject]@{ rel = $rel; file = Get-Item -LiteralPath (Join-Path $Game $rel) }
}
$rows = @()
foreach ($a in $archives | Sort-Object { $_.file.Length }) {
    $rel, $f = $a.rel, $a.file
    $dest = Join-Path $Out ([IO.Path]::GetFileNameWithoutExtension(($rel -replace "[\\/]", "_")))
    if (-not (Test-Path (Join-Path $dest "index.json"))) {
        Write-Host ("{0:HH:mm:ss} carving {1} ({2:N1} GiB)" -f (Get-Date), $rel, ($f.Length / 1GB))
        python -I $tool $f.FullName $dest | Select-Object -Last 1
    }
    $s = (Get-Content (Join-Path $dest "index.json") -Raw | ConvertFrom-Json).summary
    $rows += [pscustomobject]@{ archive = $rel; files = $s.files; bink2 = $s.bink2_files; span = $s.span
                               readable = [math]::Round(($s.compressed_bytes) / $s.span, 4)
                               unread = [math]::Round($s.unread_bytes / $s.span, 4); inflated = $s.inflated_bytes }
}
$rows | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Out "coverage.json")
$rows | Format-Table archive, files, bink2, @{n = "GiB"; e = { [math]::Round($_.span / 1GB, 2) } }, readable, unread -AutoSize | Out-String -Width 200
