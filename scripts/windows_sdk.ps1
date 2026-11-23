# Dot-sourced by build_runtime.ps1 and build_llvm_mc.ps1: stop before a build when no Windows 10/11 SDK is installed.
# vcvars64.bat then sets up the compiler without one, and the build fails late: CMake's compiler check stops with
# "RC Pass 1: command ... failed" (no rc.exe), and the cl.exe fallback with a missing windows.h.

# The SDK versions that have the headers, the x64 import libraries and rc.exe, newest first. -Roots replaces the
# Windows Kits\10 folders from the registry (both views) and the default install path, for tests.
function Get-WindowsSdkVersions([string[]]$Roots) {
    if (-not $Roots) {
        $Roots = @(
            foreach ($key in "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots",
                             "HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots") {
                (Get-ItemProperty -LiteralPath $key -Name KitsRoot10 -ErrorAction SilentlyContinue).KitsRoot10
            }
            Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10"
        )
    }
    $found = foreach ($root in @($Roots | Where-Object { $_ })) {
        $include = Join-Path $root "Include"
        if (-not (Test-Path -LiteralPath $include)) { continue }
        foreach ($dir in Get-ChildItem -LiteralPath $include -Directory -Filter "10.*" -ErrorAction SilentlyContinue) {
            $v = $dir.Name
            if (($v -match '^\d+\.\d+\.\d+\.\d+$') -and
                (Test-Path -LiteralPath (Join-Path $dir.FullName "um\Windows.h")) -and
                (Test-Path -LiteralPath (Join-Path $root "Lib\$v\um\x64\kernel32.lib")) -and
                (Test-Path -LiteralPath (Join-Path $root "bin\$v\x64\rc.exe"))) {
                [version]$v
            }
        }
    }
    @($found | Sort-Object -Unique -Descending)
}

function Assert-WindowsSdk([string[]]$Roots) {
    if (-not (Get-WindowsSdkVersions $Roots)) {
        throw ("No Windows 10/11 SDK was found (Windows Kits\10 with um\Windows.h, the x64 libraries and rc.exe of " +
            "one version). Add 'Windows 11 SDK (10.0.26100)' in the Visual Studio Installer (Individual components), " +
            "or run: winget install Microsoft.WindowsSDK.10.0.26100")
    }
}
