"""The gfx1036 v_cmp16 probe has a command line: --help prints usage instead of starting the probe.

Run with ``python -I tests/test_probe_gfx1036_cli.py``. No GPU, game files, LLVM or SPIR-V tools are used.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import probe_gfx1036_vcmp16 as T  # noqa: E402


class ProbeGfx1036CliTest(unittest.TestCase):
    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = T.main(list(args))
        return rc, out.getvalue(), err.getvalue()

    def test_help_prints_usage_and_does_not_start_the_probe(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            T.main(["--help"])
        self.assertEqual(stop.exception.code, 0, err.getvalue())
        self.assertIn("usage:", out.getvalue())
        self.assertNotIn("PROBING GFX1036", out.getvalue())

    def test_unknown_option_is_a_usage_error(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            T.main(["--wave-size", "16"])
        self.assertEqual(stop.exception.code, 2, err.getvalue())
        self.assertNotIn("PROBING GFX1036", out.getvalue())

    def test_missing_harness_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "vk_compute_test.exe")
            rc, out, err = self.call("--harness", missing)
        self.assertEqual(rc, 2, err)
        self.assertNotIn("PROBING GFX1036", out)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn(missing, err)
        self.assertIn("vk_compute_test", err)


if __name__ == "__main__":
    unittest.main()
