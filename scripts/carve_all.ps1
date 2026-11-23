# Carves every .rpf in the game folder into <Out>\<archive path with _>, smallest first (research tool; the
# build itself doesn't need it). Skips archives that already have an index.json. Summary: <Out>\coverage.json
param([Parameter(Mandatory)][string]$Game, [Parameter(Mandatory)][string]$Out)
$root = Split-Path $PSScriptRoot -Parent
$tool = Join-Path $root "tools\rpf9_carve.py"

# The carver needs Python 3.12+. `python` can be missing, the App Installer stub, or an older install first on PATH
# (Python 2.7 puts C:\Python27 there and stops at -I with "Unknown option"). Resolved once, at the first archive that
# needs carving, so an empty or fully cached -Game needs no Python at all: a `python` that reports 3.12+, else the py
# launcher's versions in Build.bat's order. `py -3` alone is only the last try: it follows PY_PYTHON3, which can name
# an older Python than one the launcher also has.
function Get-PythonVersion([string]$Exe, [string[]]$Prefix) {
    # No spaces in the probe, so it reaches a .cmd wrapper (pyenv-win, a venv shim) without cmd.exe re-quoting it;
    # and the answer is the last all-digit line, so text a wrapper or cmd's AutoRun prints first doesn't hide it.
    try {
        $v = & $Exe @Prefix -c "import sys;print(sys.version_info[0]*100+sys.version_info[1])" 2>$null |
            ForEach-Object { "$_".Trim() } | Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1
    } catch {
        return $null
    }
    if ($LASTEXITCODE -eq 0 -and $v) { return [int]$v }
    return $null
}
function Resolve-Python {
    $found = Get-PythonVersion "python" @()
    if ($null -ne $found -and $found -ge 312) { return [pscustomobject]@{ exe = "python"; prefix = @() } }
    if (Get-Command py -CommandType Application -ErrorAction SilentlyContinue) {
        foreach ($version in "-3.14", "-3.13", "-3.12", "-3") {
            $got = Get-PythonVersion "py" @($version)
            if ($null -ne $got -and $got -ge 312) { return [pscustomobject]@{ exe = "py"; prefix = @($version) } }
        }
    }
    $seen = if ($null -ne $found) { "python on PATH is {0}.{1}" -f [math]::Floor($found / 100), ($found % 100) }
            else { "python on PATH is missing or did not answer" }
    throw ("Python 3.12 or newer is needed, but $seen and the py launcher has no Python 3.12+. " +
           "Install it from https://www.python.org/downloads/ and run this again.")
}
$python = $null
# -Game may be relative or end in a separator; archive names are cut from a separator-ended prefix.
$Game = (Resolve-Path -LiteralPath $Game).ProviderPath
# Resolve-Path preserves 8.3 short components (e.g. RUNNER~1), while Get-ChildItem emits canonical
# FullName paths. Deriving the prefix from Resolve-Path then cuts the wrong number of characters and
# re-carves already-caved archives; use the canonical form for both sides of the Substring.
$prefix = (Get-Item -LiteralPath $Game).FullName.TrimEnd('\') + '\'
$destinations = [System.Collections.Generic.Dictionary[string, string]]::new([StringComparer]::OrdinalIgnoreCase)
# Validate every destination before accepting any cached index or starting a carve.
$archives = foreach ($f in Get-ChildItem -LiteralPath $Game -Recurse -Filter *.rpf | Sort-Object Length) {
    $rel = $f.FullName.Substring($prefix.Length)
    $name = [IO.Path]::GetFileNameWithoutExtension(($rel -replace "[\\/]", "_"))
    if ($destinations.ContainsKey($name)) {
        throw ("Archives '{0}' and '{1}' both map to '{2}'." -f $destinations[$name], $rel, $name)
    }
    $destinations.Add($name, $rel)
    [pscustomobject]@{ file = $f; relative = $rel; destination = Join-Path $Out $name }
}

$rows = @()
foreach ($archive in $archives) {
    $f = $archive.file
    $rel = $archive.relative
    $dest = $archive.destination
    $index = Join-Path $dest "index.json"
    if (-not (Test-Path -LiteralPath $index)) {
        if ($null -eq $python) { $python = Resolve-Python }
        Write-Host ("{0:HH:mm:ss} carving {1} ({2:N1} GiB)" -f (Get-Date), $rel, ($f.Length / 1GB))
        $pythonPrefix = $python.prefix
        & $python.exe @pythonPrefix -I $tool $f.FullName $dest | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { throw "Carving $rel failed (exit $LASTEXITCODE)" }
    }
    try {
        $s = (Get-Content -LiteralPath $index -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop).summary
        foreach ($field in 'files', 'bink2_files', 'span', 'compressed_bytes', 'unread_bytes', 'inflated_bytes') {
            $value = $s.$field
            if ((($value -isnot [int]) -and ($value -isnot [long])) -or $value -lt 0) {
                throw "summary.$field must be a non-negative integer."
            }
        }
        if ($s.span -eq 0 -and ($s.compressed_bytes -ne 0 -or $s.unread_bytes -ne 0)) {
            throw 'A zero summary.span requires zero compressed and unread bytes.'
        }
        # Stored streams and unread gaps occupy disjoint parts of the span. Subtract instead of
        # adding the counters so a total just above Int64.MaxValue cannot lose precision.
        if ($s.compressed_bytes -gt $s.span -or $s.unread_bytes -gt ($s.span - $s.compressed_bytes)) {
            throw 'summary.compressed_bytes and summary.unread_bytes must fit within summary.span.'
        }
        if ($s.bink2_files -gt $s.files) {
            throw 'summary.bink2_files must not exceed summary.files.'
        }
    } catch {
        throw "Invalid carve index '$index': $($_.Exception.Message)"
    }
    $denominator = [math]::Max(1L, $s.span)
    $rows += [pscustomobject]@{ archive = $rel; files = $s.files; bink2 = $s.bink2_files; span = $s.span
                               readable = [math]::Round($s.compressed_bytes / $denominator, 4)
                               unread = [math]::Round($s.unread_bytes / $denominator, 4); inflated = $s.inflated_bytes }
}
$rows | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $Out "coverage.json")
$rows | Format-Table archive, files, bink2, @{n = "GiB"; e = { [math]::Round($_.span / 1GB, 2) } }, readable, unread -AutoSize | Out-String -Width 200
