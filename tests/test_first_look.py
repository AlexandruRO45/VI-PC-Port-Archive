"""The first-look workflow reports a failed loader dry run while retaining diagnostics."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pe_inspect import build_image

ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh")


@unittest.skipUnless(sys.platform == "win32" and PWSH, "requires PowerShell 7 on Windows")
class FirstLookTests(unittest.TestCase):
    def test_failed_loader_dry_run_reports_failure_and_retains_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("scripts", "tools", "runtime/build"):
                (root / folder).mkdir(parents=True)
            script = root / "scripts" / "first_look.ps1"
            shutil.copyfile(ROOT / "scripts" / script.name, script)
            shutil.copyfile(ROOT / "tools" / "pe_inspect.py", root / "tools" / "pe_inspect.py")
            image = root / "sample.exe"
            image.write_bytes(build_image())

            # Python is a native command that exits 2 on the unknown --dry-run option.
            # The synthetic PE is only inspected and this test requires no runtime build.
            build = root / "runtime" / "build"
            shutil.copyfile(sys.executable, build / "gta6rt.exe")
            for dll in Path(sys.base_prefix).glob("python*.dll"):
                shutil.copyfile(dll, build / dll.name)
            env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
            result = subprocess.run(
                [PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script), "-Exe", str(image)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, env=env,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Loader dry run failed (exit 2)", result.stderr)
            reports = root / "reports" / "exe" / "sample"
            self.assertTrue((reports / "sample.exe.json").is_file())
            self.assertIn("xmem.dll", (reports / "shim-todo.txt").read_text(encoding="utf-8"))
            self.assertIn("unknown option --dry-run", (reports / "dryrun.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
