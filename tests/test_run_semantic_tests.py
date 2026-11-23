"""scripts/run_semantic_tests.py on a throwaway suite: it finds what unittest discover finds, and a failing, crashing
or unimportable file always makes it exit 1 with that file named, in parallel and with -j 1.

    python -I tests/test_run_semantic_tests.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_semantic_tests as r  # noqa: E402

PASS = "import unittest\nimport helper\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(helper.X, 1)\n" \
       "    def test_b(self): pass\n"
FAIL = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(1, 2, 'boom-msg')\n"
BROKEN = "import no_such_module_anywhere\n"


def suite(tmp, files):
    for rel, text in files.items():
        path = os.path.join(tmp, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return tmp


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = r.main(list(argv))
    return code, out.getvalue()


class RunnerTest(unittest.TestCase):
    def test_finds_what_discover_finds(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_one.py": "", "helper.py": "", "pkg/__init__.py": "", "pkg/test_two.py": "",
                        "plain_dir/test_three.py": "", "testx.py": ""})
            # discover enters packages only, and its pattern is test*.py
            self.assertEqual(r.test_modules(tmp), ["test_one", "testx", "pkg.test_two"])

    def test_all_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p1.py": PASS, "test_p2.py": PASS})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("2 files, 4 tests, 0 files failed", out)

    def test_a_failure_or_import_error_is_reported_and_fails_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "test_f.py": FAIL, "test_b.py": BROKEN})
            code, out = run(tmp, "-j", "3")
            self.assertEqual(code, 1, out)
            self.assertIn("==== FAILED: test_f", out)
            self.assertIn("boom-msg", out)                         # the failing file's own output is shown
            self.assertIn("==== FAILED: test_b", out)
            self.assertIn("2 files failed", out)
            self.assertTrue(out.rstrip().endswith("test_b, test_f"), out)

    def test_one_job_runs_discover_and_keeps_its_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "test_f.py": FAIL})
            self.assertEqual(run(tmp, "-j", "1")[0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS})
            self.assertEqual(run(tmp, "-j", "1")[0], 0)

    def test_an_empty_suite_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1)
            self.assertIn("no test*.py files", out)

    def test_timeout_preserves_output_and_continues_with_other_files(self):
        slow = ("import sys, time\nprint('before-timeout', flush=True)\n"
                "print('stderr-before-timeout', file=sys.stderr, flush=True)\n"
                "time.sleep(6)\nprint('after-timeout', flush=True)\n" + "# slow file\n" * 100)
        for jobs in (1, 2):
            with self.subTest(jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                suite(tmp, {"helper.py": "X = 1\n", "test_slow.py": slow, "test_pass.py": PASS})
                result = subprocess.run(
                    [sys.executable, "-I", os.path.join(ROOT, "scripts", "run_semantic_tests.py"),
                     tmp, "-j", str(jobs), "--timeout", "2", "-v"],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("==== FAILED: test_slow", result.stdout)
                self.assertIn("TIMEOUT", result.stdout)
                self.assertIn("before-timeout", result.stdout)
                self.assertIn("stderr-before-timeout", result.stdout)
                self.assertNotIn("after-timeout", result.stdout)
                self.assertIn("==== test_pass", result.stdout)
                self.assertIn("2 files, 2 tests, 1 files failed", result.stdout)

    def test_timeout_must_be_a_positive_finite_number(self):
        for value in ("0", "-1", "nan", "inf", "-inf", "1e309", "1e100", "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                    r.main(["unused", "--timeout=" + value])
                self.assertEqual(caught.exception.code, 2)

    def test_a_silent_module_can_time_out_with_a_fractional_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_silent.py": "import time\ntime.sleep(6)\n"})
            code, count, out = r.run_module(tmp, "test_silent", False, timeout=0.2)
            self.assertEqual((code, count), (1, 0))
            self.assertIn("TIMEOUT after 0.2 seconds", out)

    def test_timeout_output_can_be_bytes_text_or_absent(self):
        for output in (b"partial output", "partial output", None):
            with self.subTest(output=output), patch.object(
                    r.subprocess, "run", side_effect=subprocess.TimeoutExpired("unittest", 1, output=output)):
                code, count, out = r.run_module("unused", "test_one", False, timeout=1)
                self.assertEqual((code, count), (1, 0))
                self.assertIn("TIMEOUT after 1 seconds", out)
                if output is not None:
                    self.assertTrue(out.startswith("partial output"), out)

    def test_timeout_keeps_normal_success_and_failure_results(self):
        for jobs in (1, 2):
            with self.subTest(jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS})
                code, out = run(tmp, "-j", str(jobs), "--timeout", "2")
                self.assertEqual(code, 0, out)
                self.assertIn("1 files, 2 tests, 0 files failed", out)
                suite(tmp, {"test_f.py": FAIL})
                code, out = run(tmp, "-j", str(jobs), "--timeout", "2")
                self.assertEqual(code, 1, out)
                self.assertIn("==== FAILED: test_f", out)
                self.assertIn("boom-msg", out)
                self.assertNotIn("TIMEOUT", out)

    def test_an_empty_suite_with_timeout_fails(self):
        for jobs in (1, 2):
            with self.subTest(jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                code, out = run(tmp, "-j", str(jobs), "--timeout", "2")
                self.assertEqual(code, 1)
                self.assertIn("no test*.py files", out)


if __name__ == "__main__":
    unittest.main()
