# Builds Leonida Launcher and puts Leonida.exe in the repository root, next to Build.bat.
#   powershell -ExecutionPolicy Bypass -File launcher-wpf\build.ps1            # framework-dependent, ~2 MB
#   powershell -ExecutionPolicy Bypass -File launcher-wpf\build.ps1 -SelfContained   # no .NET runtime needed, ~70 MB
# Needs the .NET 8 SDK (the build needs it anyway, for rpf9dump). Leonida.exe is not committed (.gitignore).
param([switch]$SelfContained, [switch]$Test)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $PSScriptRoot 'src\Leonida.Launcher\Leonida.Launcher.csproj'
$stage = Join-Path $PSScriptRoot 'dist'

if ($Test) {
    dotnet run --project (Join-Path $PSScriptRoot 'tests\Leonida.Core.Tests\Leonida.Core.Tests.csproj') -c Release
    if ($LASTEXITCODE -ne 0) { throw "Leonida.Core tests failed" }
}

$sc = if ($SelfContained) { 'true' } else { 'false' }
dotnet publish $project -c Release -r win-x64 --self-contained $sc -o $stage `
    -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:DebugType=none
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }

$exe = Join-Path $root 'Leonida.exe'
try {
    Copy-Item (Join-Path $stage 'Leonida.exe') $exe -Force
} catch {
    throw "Could not replace $exe - close Leonida Launcher and run this again."
}
Write-Host ""
Write-Host "Leonida.exe -> $exe" -ForegroundColor Green
