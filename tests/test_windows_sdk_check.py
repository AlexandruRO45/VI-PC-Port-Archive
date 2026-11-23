"""scripts/windows_sdk.ps1 stops build_runtime.ps1 and build_llvm_mc.ps1 when no Windows 10/11 SDK is installed.

vcvars64.bat sets up the compiler without an SDK, so the build used to fail late: CMake's compiler check with
"RC Pass 1: command ... failed" (no rc.exe), and build_llvm_mc.ps1 only after cloning LLVM.

    python -I tests/test_windows_sdk_check.py
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "windows_sdk.ps1"
DOT_SOURCE = '. (Join-Path $PSScriptRoot "windows_sdk.ps1")'


class BuildScriptsCheckFirst(unittest.TestCase):
    def test_both_build_scripts_check_before_they_build(self):
        # Where each script starts its long work: the LLVM clone, or the batch file that runs CMake or cl.exe.
        for name, work in (("build_llvm_mc.ps1", "& git clone"), ("build_runtime.ps1", "$bat = Join-Path")):
            with self.subTest(script=name):
                text = (ROOT / "scripts" / name).read_text(encoding="utf-8-sig")
                self.assertIn(DOT_SOURCE, text)
                check = text.index("\nAssert-WindowsSdk", text.index(DOT_SOURCE))
                self.assertLess(check, text.index(work))

    def test_the_helper_is_ascii(self):
        # Windows PowerShell 5.1 reads a script without a BOM in the ANSI code page.
        HELPER.read_bytes().decode("ascii")


def make_sdk(root: Path, version: str, parts=("include", "lib", "rc")):
    if "include" in parts:
        (root / "Include" / version / "um").mkdir(parents=True)
        (root / "Include" / version / "um" / "Windows.h").write_bytes(b"")
    if "lib" in parts:
        (root / "Lib" / version / "um" / "x64").mkdir(parents=True)
        (root / "Lib" / version / "um" / "x64" / "kernel32.lib").write_bytes(b"")
    if "rc" in parts:
        (root / "bin" / version / "x64").mkdir(parents=True)
        (root / "bin" / version / "x64" / "rc.exe").write_bytes(b"")


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell")
class FindsInstalledSdks(unittest.TestCase):
    def run_helper(self, command: str, roots: list[Path]) -> subprocess.CompletedProcess:
        ps = (
            "$ErrorActionPreference = 'Stop'\n"
            ". $env:LEONIDA_SDK_HELPER\n"
            "$roots = @($env:LEONIDA_SDK_ROOTS -split ';' | Where-Object { $_ })\n"
            + command
        )
        env = dict(os.environ, LEONIDA_SDK_HELPER=str(HELPER), LEONIDA_SDK_ROOTS=";".join(str(r) for r in roots))
        return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                              env=env, capture_output=True, text=True, timeout=30)

    def versions(self, roots: list[Path]) -> list[str]:
        res = self.run_helper("Get-WindowsSdkVersions $roots | ForEach-Object { $_.ToString() }", roots)
        self.assertEqual(res.returncode, 0, res.stderr)
        return res.stdout.split()

    def test_only_complete_versions_count_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            kits = Path(tmp) / "Windows Kits" / "10"
            make_sdk(kits, "10.0.22621.0")
            make_sdk(kits, "10.0.26100.0")
            make_sdk(kits, "10.0.26200.0", parts=("include", "lib"))  # a partial install without rc.exe
            make_sdk(kits, "10.0.19041.0", parts=("lib", "rc"))  # leftover libraries without headers
            (kits / "Include" / "wdf").mkdir()
            self.assertEqual(self.versions([kits]), ["10.0.26100.0", "10.0.22621.0"])

    def test_no_sdk_stops_with_the_install_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            kits = Path(tmp) / "Windows Kits" / "10"
            make_sdk(kits, "10.0.26100.0", parts=("include", "lib"))
            for roots in ([kits], [Path(tmp) / "missing"]):
                with self.subTest(roots=roots):
                    self.assertEqual(self.versions(roots), [])
                    # The message through stdout: PowerShell wraps an uncaught error at the console width.
                    res = self.run_helper("try { Assert-WindowsSdk $roots } catch { $_.Exception.Message; exit 3 }",
                                          roots)
                    self.assertEqual(res.returncode, 3, res.stderr)
                    self.assertIn("No Windows 10/11 SDK was found", res.stdout)
                    self.assertIn("winget install Microsoft.WindowsSDK.10.0.26100", res.stdout)

    def test_an_sdk_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            kits = Path(tmp) / "Windows Kits" / "10"
            make_sdk(kits, "10.0.26100.0")
            res = self.run_helper("Assert-WindowsSdk $roots; Write-Output ok", [kits])
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
