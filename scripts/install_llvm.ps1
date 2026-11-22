# Automatic Windows installer for LLVM llvm-mc (AMDGPU target, from llvm/llvm-project).
# Downloads the prebuilt llvm-mc-windows-x64.zip (verified with SHA-256) from GitHub Releases,
# or falls back to building from https://github.com/llvm/llvm-project via scripts/build_llvm_mc.ps1.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_llvm.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_llvm.ps1 -FromSource
param(
    [string]$Version = "21.1.8",
    [string]$Dest = (Join-Path $env:USERPROFILE ".leonida\llvm-mc"),
    [string]$Repo = "FelipeMzero/Leonida",
    [switch]$FromSource,
    [switch]$NoEnv
)
$ErrorActionPreference = "Stop"

function Test-LlvmMc([string]$ExePath) {
    if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) { return $false }
    $probe = Join-Path $env:TEMP ("leonida-probe-" + [Guid]::NewGuid().ToString("N") + ".txt")
    try {
        [IO.File]::WriteAllText($probe, "0x00 0x00 0x81 0xbf`n")
        $out = & $ExePath --disassemble -triple=amdgcn -mcpu=gfx1013 $probe 2>$null
        return ($LASTEXITCODE -eq 0 -and ($out -match "s_endpgm"))
    } catch {
        return $false
    } finally {
        if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue }
    }
}

$finalExe = Join-Path $Dest "llvm-mc.exe"
if ((Test-LlvmMc $finalExe) -and -not $FromSource) {
    Write-Host "[OK] llvm-mc (AMDGPU) already installed at $finalExe"
    if (-not $NoEnv) {
        [Environment]::SetEnvironmentVariable("LLVM_MC", $finalExe, "User")
    }
    exit 0
}

$installed = $false
if (-not $FromSource) {
    $asset = "llvm-mc-windows-x64.zip"
    $base = "https://github.com/$Repo/releases/download/llvm-$Version"
    $tmp = Join-Path $env:TEMP ("leonida-llvm-dl-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        $zipPath = Join-Path $tmp $asset
        $sumsPath = Join-Path $tmp "SHA256SUMS.txt"
        Write-Host "Downloading $base/$asset ..."
        Invoke-WebRequest -Uri "$base/$asset" -OutFile $zipPath -UseBasicParsing
        Invoke-WebRequest -Uri "$base/SHA256SUMS.txt" -OutFile $sumsPath -UseBasicParsing

        $expectedLine = Get-Content -LiteralPath $sumsPath | Where-Object { $_ -match $asset } | Select-Object -First 1
        if ($expectedLine) {
            $expectedHash = ($expectedLine -split '\s+')[0].ToLowerInvariant()
            $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
            if ($actualHash -ne $expectedHash) {
                throw "SHA-256 mismatch for $asset (got $actualHash, expected $expectedHash)"
            }
            Write-Host "[OK] SHA-256 verified."
        }

        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Expand-Archive -LiteralPath $zipPath -DestinationPath $Dest -Force
        if (Test-LlvmMc $finalExe) {
            $installed = $true
        }
    } catch {
        Write-Warning "Prebuilt download unavailable ($($_.Exception.Message)). Falling back to source build..."
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $installed) {
    $builder = Join-Path $PSScriptRoot "build_llvm_mc.ps1"
    if (-not (Test-Path -LiteralPath $builder -PathType Leaf)) {
        throw "Could not download prebuilt llvm-mc and $builder was not found."
    }
    & $builder -Version $Version -Dest $Dest -NoEnv:$NoEnv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    exit 0
}

if (-not $NoEnv) {
    [Environment]::SetEnvironmentVariable("LLVM_MC", $finalExe, "User")
    $env:LLVM_MC = $finalExe
    Write-Host "LLVM_MC set to $finalExe for the current user."
}
Write-Host "[OK] Installed $finalExe"
