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
            self.assertIn("0 tests skipped", out)

    def test_parallel_summary_counts_all_skipped(self):
        skipped = "import unittest\nclass T(unittest.TestCase):\n    @unittest.skip('optional')\n    def test_skip(self): pass\n"
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_skipped.py": skipped})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("1 tests skipped", out)

    def test_skipped_setup_class_can_exceed_tests_run(self):
        skipped = "import unittest\nclass T(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls): raise unittest.SkipTest('optional')\n    def test_one(self): pass\n"
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_skipped.py": skipped})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("1 files, 0 tests, 0 files failed", out)
            self.assertIn("1 tests skipped", out)

    def test_parallel_summary_counts_skips_even_in_failed_files(self):
        skipped = "import unittest\nclass S(unittest.TestCase):\n    @unittest.skip('optional')\n    def test_skip(self): pass\n"
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS + skipped,
                        "test_f.py": FAIL + skipped})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1, out)
            self.assertIn("2 tests skipped", out)

    def test_test_output_cannot_override_unittest_summary(self):
        printed = "import unittest\nclass T(unittest.TestCase):\n    def test_print(self): print('OK')\n    @unittest.skip('optional')\n    def test_skip(self): pass\n"
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_printed.py": printed})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("1 tests skipped", out)

    def test_nested_runner_counts_outer_summary(self):
        nested = '''import sys, unittest
class T(unittest.TestCase):
    def test_nested(self):
        unittest.TextTestRunner(stream=sys.stdout).run(unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]))
    @unittest.skip("optional")
    def test_skip(self): pass
'''
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_nested.py": nested})
            for args in ((), ("-v",)):
                with self.subTest(args=args):
                    code, out = run(tmp, "-j", "2", *args)
                    self.assertEqual(code, 0, out)
                    self.assertIn("1 files, 2 tests, 0 files failed", out)
                    self.assertIn("1 tests skipped", out)

    def test_parallel_summary_reports_unknown_skip_count_on_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_crash.py": "import os\nos._exit(2)\n"})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1, out)
            self.assertIn("test and skip count unknown", out)

    def test_successful_exit_without_result_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_exit.py": "import os\nos._exit(0)\n"})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("test and skip count unknown", out)

    def test_empty_module_keeps_unittest_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_empty.py": ""})
            expected = subprocess.run([sys.executable, "-m", "unittest", "test_empty"], cwd=tmp,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode
            code, ran, skipped, _ = r.run_module(tmp, "test_empty", False)
            self.assertEqual((code, ran, skipped), (expected, 0, 0))

    def test_module_sees_unittest_arguments(self):
        module = '''import os, sys, unittest
class T(unittest.TestCase):
    def test_arguments(self):
        self.assertEqual(os.path.basename(sys.argv[0]), "__main__.py")
        self.assertEqual(sys.argv[1:], ["test_args"])
'''
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_args.py": module})
            code, _, _, out = r.run_module(tmp, "test_args", False)
            self.assertEqual(code, 0, out)

    def test_buffered_nested_runner_counts_outer_summary(self):
        nested = '''import io, unittest
class T(unittest.TestCase):
    def test_nested(self):
        output = io.StringIO()
        unittest.TextTestRunner(stream=output).run(unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]))
        print(output.getvalue())
    @unittest.skip("optional")
    def test_skip(self): pass
'''
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_nested.py": nested})
            for args in ((), ("-v",)):
                with self.subTest(args=args):
                    code, out = run(tmp, "-j", "2", *args)
                    self.assertEqual(code, 0, out)
                    self.assertIn("1 files, 2 tests, 0 files failed", out)
                    self.assertIn("1 tests skipped", out)

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


if __name__ == "__main__":
    unittest.main()
