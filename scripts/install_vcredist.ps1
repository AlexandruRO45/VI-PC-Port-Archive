# Downloads VisualCppRedist_AIO_x86_x64.exe (v0.107.0), verifies its pinned SHA-256,
# and runs the installer with Windows UAC Administrator elevation (-Verb RunAs).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_vcredist.ps1
param(
    [string]$Url = "https://github.com/abbodi1406/vcredist/releases/download/v0.107.0/VisualCppRedist_AIO_x86_x64.exe",
    [string]$ExpectedSha256 = "1990ebb978d80b7a2abdd76fa6562f5969e39b775dc4a846ca9398a354d27b6b",
    [string]$InstallerArgs = "/ai /gm2"
)
$ErrorActionPreference = "Stop"

if (-not ($ExpectedSha256 -match '^[0-9a-fA-F]{64}$')) {
    throw "ExpectedSha256 must be a valid 64-character hex SHA-256 digest."
}

$tmp = Join-Path $env:TEMP ("leonida-vcredist-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    $exe = Join-Path $tmp "VisualCppRedist_AIO_x86_x64.exe"
    Write-Host "Downloading $Url ..."
    Invoke-WebRequest -Uri $Url -OutFile $exe -UseBasicParsing

    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
    $expected = $ExpectedSha256.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "SHA-256 mismatch for VisualCppRedist_AIO_x86_x64.exe: got $actual, expected $expected"
    }
    Write-Host "[OK] SHA-256 verified ($actual)."
    Write-Host "Requesting Administrator permission (UAC) to run VisualCppRedist_AIO_x86_x64.exe ..."
    $p = Start-Process -FilePath $exe -ArgumentList $InstallerArgs -Verb RunAs -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "VisualCppRedist_AIO_x86_x64.exe exited with code $($p.ExitCode)."
    }
    Write-Host "[OK] Visual C++ Redistributable AIO installed."
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
