"""tools/conformance/native/scripts/run.ps1 returns the renderer's exit code and cleans a bracketed output folder."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "native" / "scripts" / "run.ps1"
STUB = r"""
using System;
using System.IO;
public static class Stub {
    public static int Main(string[] args) {
        string dir = args.Length > 0 ? args[0] : "";
        int stale = Directory.Exists(dir) ? Directory.GetFiles(dir).Length : 0;
        Console.WriteLine("STALE_FILES " + stale);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "new.txt"), "new");
        string code = Environment.GetEnvironmentVariable("STUB_EXIT");
        return code == null ? 1 : int.Parse(code);
    }
}
"""


def powershell():
    return shutil.which("powershell") or shutil.which("powershell.exe")


@unittest.skipUnless(os.name == "nt" and powershell(), "PowerShell research script")
class ConformanceRunPs1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        root = Path(cls._tmp.name, "native")
        (root / "scripts").mkdir(parents=True)
        (root / "build" / "Release").mkdir(parents=True)
        cls.script = root / "scripts" / "run.ps1"
        shutil.copyfile(SCRIPT, cls.script)
        src = Path(cls._tmp.name, "stub.cs")
        src.write_text(STUB, encoding="utf-8", newline="\n")
        exe = root / "build" / "Release" / "conform_native.exe"
        build = subprocess.run(
            [powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
             "Add-Type -TypeDefinition (Get-Content -Raw -LiteralPath $env:STUB_SRC) "
             "-OutputType ConsoleApplication -OutputAssembly $env:STUB_EXE"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "STUB_SRC": str(src), "STUB_EXE": str(exe)})
        if not exe.is_file():
            raise AssertionError(build.stdout + build.stderr)

    def run_script(self, out, exit_code):
        return subprocess.run(
            [powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.script),
             "-Output", str(out)], cwd=self._tmp.name, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env={**os.environ, "STUB_EXIT": str(exit_code)})

    def test_failing_renderer_propagates_exit_code(self):
        for code in (1, 2):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as case_dir:
                result = self.run_script(Path(case_dir, "out"), code)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn(f"EXITCODE {code}", result.stdout)

    def test_successful_renderer_succeeds(self):
        with tempfile.TemporaryDirectory() as case_dir:
            result = self.run_script(Path(case_dir, "out"), 0)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("EXITCODE 0", result.stdout)

    def test_bracketed_output_folder_is_cleaned(self):
        with tempfile.TemporaryDirectory() as case_dir:
            out = Path(case_dir, "out[1]")
            out.mkdir()
            (out / "stale.txt").write_text("old", encoding="utf-8", newline="\n")
            sibling = Path(case_dir, "out1")
            sibling.mkdir()
            (sibling / "keep.txt").write_text("keep", encoding="utf-8", newline="\n")
            result = self.run_script(out, 0)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("STALE_FILES 0", result.stdout)
            self.assertFalse((out / "stale.txt").exists())
            self.assertTrue((out / "new.txt").exists())
            self.assertTrue((sibling / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
