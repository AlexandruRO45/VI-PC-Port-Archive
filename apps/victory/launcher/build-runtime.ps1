param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$workPath = Join-Path $env:TEMP "VIctory-QmlRuntime-$PID"
New-Item -ItemType Directory -Force -Path $workPath | Out-Null
$environment = Join-Path $env:LOCALAPPDATA "victory\launcher\python313"
$python = Join-Path $environment "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    py install --target=$environment 3.13
    if ($LASTEXITCODE -ne 0) { throw "Downloading the portable Python 3.13 runtime failed." }
}

& $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Installing the PySide6 launcher requirements failed." }
& $python -m pip install "PyInstaller>=6.17,<7"
if ($LASTEXITCODE -ne 0) { throw "Installing the launcher packaging tool failed." }

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--name", "VictoryQmlRuntime",
    "--distpath", $outputPath,
    "--workpath", $workPath,
    "--specpath", $workPath,
    "--collect-all", "PySide6",
    "--collect-all", "shiboken6",
    (Join-Path $PSScriptRoot "main.py")
)
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Building the bundled Python 3.13 / PySide6 launcher failed." }

$runtimeExe = Join-Path $outputPath "VictoryQmlRuntime\VictoryQmlRuntime.exe"
if (-not (Test-Path -LiteralPath $runtimeExe)) {
    throw "PyInstaller completed without producing $runtimeExe."
}
Write-Output "Runtime ready: $runtimeExe"
