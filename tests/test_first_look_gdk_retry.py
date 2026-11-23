"""Verify that First Look only uses GDK calls from a successful current scan."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT_TOOL = """\
import os
from pathlib import Path
import sys

tool = Path(__file__).stem
args = sys.argv[1:]
if tool == "pe_inspect":
    report = Path(args[-1]) / (Path(args[0]).name + ".json")
    report.write_text('{"imports":[],"sections":[]}', encoding="utf-8", newline="\\n")
elif tool == "gdk_runtime_calls":
    if os.environ.get("LEONIDA_TEST_GDK_FAIL") == "1":
        sys.exit(1)
    report = Path(args[-1]) / (Path(args[0]).name + ".gdk_calls.json")
    report.write_text("{}", encoding="utf-8", newline="\\n")
elif tool == "exe_checklist":
    Path(args[1]).write_text(str("--gdk-calls" in args), encoding="utf-8", newline="\\n")
"""
RUN = r"""
Set-Alias -Name (Join-Path $PWD 'runtime\build\gta6rt.exe') -Value (Join-Path $PWD 'loader.cmd')
& .\scripts\first_look.ps1 -Exe (Join-Path $PWD 'fixture.exe')
exit $LASTEXITCODE
"""


@unittest.skipUnless(os.name == "nt", "Windows PowerShell first-look diagnostics")
class FirstLookGdkRetryTest(unittest.TestCase):
    def test_failed_gdk_retry_does_not_pass_previous_report_to_checklist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("scripts", "tools", "runtime/build"):
                (root / folder).mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/first_look.ps1", root / "scripts/first_look.ps1")
            for name in ("pe_inspect", "gdk_runtime_calls", "find_hash_lookup", "exe_checklist"):
                (root / "tools" / f"{name}.py").write_text(REPORT_TOOL, encoding="utf-8", newline="\n")
            (root / "fixture.exe").write_bytes(b"MZ")
            (root / "runtime/build/gta6rt.exe").touch()
            (root / "runtime/build/gta6rt_tls_host.dll").touch()
            (root / "loader.cmd").write_text("@exit /b 0\n", encoding="ascii", newline="\r\n")

            env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                       "-Command", RUN]
            output = root / "reports/exe/fixture"
            for fail in (False, True):
                with self.subTest(scan_fails=fail):
                    env["LEONIDA_TEST_GDK_FAIL"] = str(int(fail))
                    result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual((output / "fixture.exe.gdk_calls.json").read_bytes(), b"{}")
                    self.assertEqual((output / "checklist.md").read_text("utf-8"), str(not fail))


if __name__ == "__main__":
    unittest.main()
