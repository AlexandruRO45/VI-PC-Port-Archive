"""tools/probe_gfx1036_vcmp16.py without the Windows runtime binary: one ERROR line and exit 2, no traceback.

    python -I tests/test_probe_gfx1036_vcmp16.py
"""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import probe_gfx1036_vcmp16 as probe  # noqa: E402

PROBE = os.path.join(ROOT, "tools", "probe_gfx1036_vcmp16.py")


class MissingExe(unittest.TestCase):
    def test_missing_exe_is_one_error_line_and_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "vk_compute_test.exe")
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(probe, "VK_COMPUTE_TEST", missing), \
                 redirect_stdout(out), redirect_stderr(err):
                rc = probe.main()
            self.assertEqual(rc, 2)
            self.assertEqual(out.getvalue(), "")
            lines = err.getvalue().splitlines()
            self.assertEqual(len(lines), 1, err.getvalue())
            self.assertTrue(lines[0].startswith("ERROR:"), lines[0])
            self.assertIn(missing, lines[0])
            self.assertNotIn("Traceback", err.getvalue())

    def test_nothing_runs_before_the_exe_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "vk_compute_test.exe")
            with mock.patch.object(probe, "VK_COMPUTE_TEST", missing), \
                 mock.patch.object(probe, "run_on_silicon", side_effect=AssertionError("ran a probe")), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(probe.main(), 2)

    def test_cli_without_the_exe_exits_2_without_a_traceback(self):
        if os.path.isfile(probe.VK_COMPUTE_TEST):
            self.skipTest("the runtime exe is built here; the missing-exe path cannot run")
        p = subprocess.run([sys.executable, "-I", PROBE], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2, p.stderr)
        lines = p.stderr.splitlines()
        self.assertEqual(len(lines), 1, p.stderr)
        self.assertTrue(lines[0].startswith("ERROR:"), lines[0])
        self.assertIn("vk_compute_test.exe", lines[0])
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
