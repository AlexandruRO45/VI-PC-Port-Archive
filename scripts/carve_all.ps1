# Carves every .rpf in the game folder into <Out>\<archive path with _>, smallest first (research tool; the
# build itself doesn't need it). Skips archives that already have an index.json. Summary: <Out>\coverage.json
param([Parameter(Mandatory)][string]$Game, [Parameter(Mandatory)][string]$Out)
$root = Split-Path $PSScriptRoot -Parent
$tool = Join-Path $root "tools\rpf9_carve.py"
# -Game may be relative or end in a separator; archive names are cut from the resolved, separator-ended prefix.
# Resolve-Path keeps 8.3 short names (C:\Users\RUNNER~1\...) while Get-ChildItem returns long ones, so the prefix
# is taken from Get-Item, which gives the same long form as the files below it.
$Game = (Get-Item -LiteralPath (Resolve-Path -LiteralPath $Game).ProviderPath).FullName
$prefix = $Game.TrimEnd('\') + '\'
$rows = @()
foreach ($f in Get-ChildItem $Game -Recurse -Filter *.rpf | Sort-Object Length) {
    $rel = $f.FullName.Substring($prefix.Length)
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
