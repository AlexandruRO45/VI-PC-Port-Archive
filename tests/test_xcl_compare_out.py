"""Unit tests for the tools/xcl_compare.py --out guard: a --out that names an existing folder must be refused at
main start, before either title runs, instead of an IsADirectoryError traceback from the report write at the end.

    python -I tests/test_xcl_compare_out.py

No game data and no XCL checkout: --xcl names an empty folder, so a run that got past the guard stops at its own
"no built XCL" error.
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "xcl_compare.py")


def run_xcl_compare(*args):
    return subprocess.run([sys.executable, "-I", TOOL] + list(args), capture_output=True, text=True, timeout=120)


class XclCompareOutTests(unittest.TestCase):
    def test_out_naming_a_folder_is_refused_at_start(self):
        with tempfile.TemporaryDirectory() as t:
            xcl = os.path.join(t, "xcl")
            os.makedirs(xcl)
            report_dir = os.path.join(t, "report.json")
            os.makedirs(report_dir)
            p = run_xcl_compare("--xcl", xcl, "--out", report_dir)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("--out is a folder, not a report file", p.stderr)
            self.assertIn(report_dir, p.stderr)
            self.assertEqual(os.listdir(report_dir), [])  # nothing was written into the folder

    @unittest.skipUnless(sys.platform == "win32", "xcl_compare runs Windows executables")
    def test_a_report_path_in_a_missing_folder_is_not_refused(self):
        with tempfile.TemporaryDirectory() as t:
            xcl = os.path.join(t, "xcl")
            os.makedirs(xcl)
            p = run_xcl_compare("--xcl", xcl, "--out", os.path.join(t, "new", "report.json"))
            self.assertEqual(p.returncode, 1)  # stops at the missing XCL build, as before the guard
            self.assertIn("no built XCL", p.stderr)


if __name__ == "__main__":
    unittest.main()
