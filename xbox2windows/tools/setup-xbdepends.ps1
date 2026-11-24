$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\dev.ps1"
$revision = 'b1e77b6ac10f95b07296025fda171ca2ddc42dbe'
$root = Join-Path $PSScriptRoot '..\out\tools\xbdepends'
$src = Join-Path $root $revision
$build = Join-Path $root 'build'
$url = "https://raw.githubusercontent.com/microsoft/Xbox-GDK-Samples/$revision/Samples/Tools/xbdepends"
New-Item -ItemType Directory -Force -Path $src | Out-Null
foreach ($name in 'CMakeLists.txt', 'xbdepends.cpp', 'KnownDLLs.h', 'GameOSAPIs.h') {
    $file = Join-Path $src $name
    if (-not (Test-Path -LiteralPath $file)) {
        Invoke-WebRequest -UseBasicParsing "$url/$name" -OutFile "$file.tmp"
        Move-Item -LiteralPath "$file.tmp" -Destination $file -Force
    }
}
cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'xbdepends configuration failed.' }
cmake --build $build
if ($LASTEXITCODE -ne 0) { throw 'xbdepends build failed.' }
