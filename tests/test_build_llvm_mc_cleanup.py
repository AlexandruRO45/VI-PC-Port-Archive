"""Exercise the native LLVM builder's robust directory cleanup handling read-only attributes."""
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell")
class BuildCleanupTest(unittest.TestCase):
    def test_install_cleans_build_artifacts_without_removing_unrelated_files(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index("function Remove-DirectoryRobust")
        func_def = text[start:text.index("if (-not (Test-Path (Join-Path $src", start)]
        install = text[text.index("New-Item -ItemType Directory -Force $Dest | Out-Null"):]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            driver = root / "test-install.ps1"
            driver.write_text(
                "param([string]$Temp, [string]$Dest, [switch]$Keep)\n"
                "$ErrorActionPreference = 'Stop'\n"
                "$NoEnv = $true\n"
                "$paths = $ExecutionContext.SessionState.Path\n"
                "$src = Join-Path $Temp 'src'\n"
                "$out = Join-Path $Temp 'build'\n"
                "$bat = Join-Path $Temp '_build.cmd'\n"
                "$probeFile = Join-Path $Temp 'probe.txt'\n"
                "$exe = Join-Path $out 'bin\\llvm-mc.exe'\n"
                + func_def + install,
                encoding="utf-8", newline="\n",
            )
            for name, has_sibling, keep in (("with-sibling", True, False),
                                            ("no-sibling", False, False),
                                            ("keep", True, True)):
                with self.subTest(case=name):
                    target = (root / name).resolve()
                    self.assertEqual(target.parent, root)
                    dest = root / ("install-" + name)
                    artifacts = [target / n for n in ("src", "build", "_build.cmd", "probe.txt")]
                    llvm_mc = target / "build" / "bin" / "llvm-mc.exe"
                    llvm_mc.parent.mkdir(parents=True)
                    llvm_mc.write_bytes(b"synthetic executable")
                    artifacts[0].mkdir()
                    for artifact in artifacts[2:]:
                        artifact.write_bytes(b"synthetic build input")
                    sibling = target / "existing-notes.txt"
                    if has_sibling:
                        sibling.write_bytes(b"user data")
                    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-File", str(driver), "-Temp", str(target), "-Dest", str(dest)]
                    if keep:
                        cmd.append("-Keep")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual((dest / "llvm-mc.exe").read_bytes(), b"synthetic executable")
                    self.assertEqual([artifact.exists() for artifact in artifacts], [keep] * len(artifacts))
                    if has_sibling:
                        self.assertEqual(sibling.read_bytes(), b"user data")
                    else:
                        self.assertFalse(target.exists())

    def test_cleanup_removes_readonly_nested_files_and_directories(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index("function Remove-DirectoryRobust")
        func_def = text[start:text.index("if (-not (Test-Path (Join-Path $src", start)].strip()

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test_temp_cleanup"
            target.mkdir()
            sub = target / ".git" / "objects" / "pack"
            sub.mkdir(parents=True)
            ro_file = sub / "sample_pack.idx"
            ro_file.write_bytes(b"DATA")
            # Mark file and directory read-only on Windows
            os.chmod(ro_file, stat.S_IREAD)
            os.chmod(sub, stat.S_IREAD)

            driver = Path(directory) / "test-clean.ps1"
            driver.write_text(
                "param([string]$Path)\n"
                "$ErrorActionPreference = 'Stop'\n"
                + func_def + "\n"
                "Remove-DirectoryRobust $Path\n",
                encoding="utf-8", newline="\n",
            )
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver), "-Path", str(target)],
                capture_output=True, text=True, check=True, timeout=30,
            )
            self.assertFalse(target.exists(), f"Target directory should have been removed: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
