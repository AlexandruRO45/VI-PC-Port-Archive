# Automatic Windows installer for LLVM llvm-mc (AMDGPU target, from llvm/llvm-project).
# Respects any existing working LLVM_MC (including WSL command lines).
# Defaults to building natively from https://github.com/llvm/llvm-project via scripts/build_llvm_mc.ps1.
# Optional prebuilt download requires BOTH -PrebuiltUrl and -ExpectedSha256 (fails closed on mismatch).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_llvm.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_llvm.ps1 -PrebuiltUrl URL -ExpectedSha256 SHA256
param(
    [string]$Version = "21.1.8",
    [string]$Dest = (Join-Path $env:USERPROFILE ".leonida\llvm-mc"),
    [string]$PrebuiltUrl = "",
    [string]$ExpectedSha256 = "",
    [switch]$FromSource,
    [switch]$NoEnv
)
$ErrorActionPreference = "Stop"

function Test-LlvmMcCommand([string]$CommandLine) {
    $trimmed = $CommandLine.Trim()
    if (-not $trimmed) { return $false }
    $probe = Join-Path $env:TEMP ("leonida-probe-" + [Guid]::NewGuid().ToString("N") + ".txt")
    try {
        [IO.File]::WriteAllText($probe, "0x00 0x00 0x81 0xbf`n")
        if (Test-Path -LiteralPath $trimmed.Trim('"') -PathType Leaf) {
            $exe = $trimmed.Trim('"')
            $out = & $exe --disassemble -triple=amdgcn -mcpu=gfx1013 $probe 2>$null
            return ($LASTEXITCODE -eq 0 -and ($out -match "s_endpgm"))
        }
        $parts = @($trimmed -split '\s+')
        $head = $parts[0].Trim('"')
        $rest = if ($parts.Length -gt 1) { $parts[1..($parts.Length - 1)] } else { @() }
        $out = & $head @rest --disassemble -triple=amdgcn -mcpu=gfx1013 $probe 2>$null
        return ($LASTEXITCODE -eq 0 -and ($out -match "s_endpgm"))
    } catch {
        return $false
    } finally {
        if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue }
    }
}

# 1. Never replace an already working LLVM_MC (including WSL command lines).
if ($env:LLVM_MC -and (Test-LlvmMcCommand $env:LLVM_MC)) {
    Write-Host "[OK] Existing LLVM_MC is already working: $env:LLVM_MC"
    exit 0
}

$finalExe = Join-Path $Dest "llvm-mc.exe"
if ((Test-LlvmMcCommand $finalExe) -and -not $FromSource) {
    Write-Host "[OK] llvm-mc (AMDGPU) already installed at $finalExe"
    if (-not $NoEnv -and -not $env:LLVM_MC) {
        [Environment]::SetEnvironmentVariable("LLVM_MC", $finalExe, "User")
    }
    exit 0
}

# 2. Optional pinned prebuilt download (fails closed if ExpectedSha256 is missing or mismatches).
if ($PrebuiltUrl -and -not $FromSource) {
    if (-not ($ExpectedSha256 -match '^[0-9a-fA-F]{64}$')) {
        throw "-ExpectedSha256 must be a valid 64-character hex SHA-256 digest when using -PrebuiltUrl."
    }
    $tmp = Join-Path $env:TEMP ("leonida-llvm-dl-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        $zipPath = Join-Path $tmp "llvm-mc.zip"
        Write-Host "Downloading $PrebuiltUrl ..."
        Invoke-WebRequest -Uri $PrebuiltUrl -OutFile $zipPath -UseBasicParsing
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
        if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
            throw "SHA-256 mismatch: got $actualHash, expected $($ExpectedSha256.ToLowerInvariant())"
        }
        Write-Host "[OK] SHA-256 verified."
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Expand-Archive -LiteralPath $zipPath -DestinationPath $Dest -Force
        if (-not (Test-LlvmMcCommand $finalExe)) {
            throw "Extracted $finalExe failed the gfx1013 s_endpgm probe."
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not $NoEnv) {
        [Environment]::SetEnvironmentVariable("LLVM_MC", $finalExe, "User")
        $env:LLVM_MC = $finalExe
        Write-Host "LLVM_MC set to $finalExe for the current user."
    }
    Write-Host "[OK] Installed $finalExe"
    exit 0
}

# 3. Default: build natively from official https://github.com/llvm/llvm-project source.
$builder = Join-Path $PSScriptRoot "build_llvm_mc.ps1"
if (-not (Test-Path -LiteralPath $builder -PathType Leaf)) {
    throw "Missing $builder."
}
& $builder -Version $Version -Dest $Dest -NoEnv:$NoEnv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
