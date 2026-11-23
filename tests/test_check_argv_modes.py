"""runtime/test/check_argv_modes.py with subprocess.run patched: no loader, MSVC fixtures or game files."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_argv_modes", ROOT / "runtime" / "test" / "check_argv_modes.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def hang(command, **kwargs):
    raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))


class RunTest(unittest.TestCase):
    def test_a_hung_child_is_a_recorded_failure(self):
        with mock.patch.object(check.subprocess, "run", hang):
            result = check.run("x.exe", "x.exe a", ".", {})
        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["lines"], [])
        self.assertEqual(result["error"], "child exceeded 15 seconds")

    def test_a_finished_child_has_no_error(self):
        done = subprocess.CompletedProcess("x", 0, b"STATE entry argc=3 A=x W=y\n", b"")
        with mock.patch.object(check.subprocess, "run", return_value=done):
            result = check.run("x.exe", "x.exe a", ".", {})
        self.assertNotIn("error", result)
        self.assertEqual(result["exit_code"], 0)


@unittest.skipUnless(os.name == "nt", "main() needs native Windows")
class MainTest(unittest.TestCase):
    def run_main(self, tmp, work_dir, report):
        loader = tmp / "loader.exe"
        loader.write_bytes(b"")
        fixtures = tmp / "fixtures"
        fixtures.mkdir()
        for width in "aw":
            for mode in ("default", "expand", "noargs"):
                (fixtures / f"tiny_argv_{mode}_{width}.exe").write_bytes(b"")
        argv = ["check_argv_modes.py", "--loader", str(loader), "--fixtures", str(fixtures),
                "--work-dir", str(work_dir), "--report", str(report)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(check.subprocess, "run", hang), \
                contextlib.redirect_stdout(io.StringIO()):
            return check.main()

    def test_hung_children_do_not_stop_the_run_and_the_report_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            work = tmp / "work"
            report = tmp / "report.json"
            self.assertEqual(self.run_main(tmp, work, report), 1)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["pairs"], 26)  # (3 + 6 + 4 cases) for each of the two widths
            self.assertTrue(any("child exceeded 15 seconds" in failure for failure in data["failures"]))

    def test_a_missing_work_dir_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            work = tmp / "not" / "yet" / "there"
            self.run_main(tmp, work, tmp / "report.json")
            self.assertTrue(work.is_dir())


if __name__ == "__main__":
    unittest.main()
