"""scripts/run_semantic_tests.py on a throwaway suite: it finds what unittest discover finds, and a failing, crashing
or unimportable file always makes it exit 1 with that file named, in parallel and with -j 1.

    python -I tests/test_run_semantic_tests.py
"""
import contextlib
import io
import os
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
    def test_finds_top_level_test_files_and_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_one.py": "", "helper.py": "", "pkg/__init__.py": "", "pkg/test_two.py": "",
                        "plain_dir/test_three.py": "", "testx.py": ""})
            self.assertEqual(r.test_modules(tmp), ["pkg", "test_one.py", "testx.py"])

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

    def test_package_initialization_failures_fail_the_run(self):
        for package_init, diagnosis in ((FAIL, "boom-msg"), (BROKEN, "no_such_module_anywhere")):
            with self.subTest(diagnosis=diagnosis), tempfile.TemporaryDirectory() as tmp:
                suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "pkg/__init__.py": package_init})
                code, out = run(tmp, "-j", "2")
                self.assertEqual(code, 1, out)
                self.assertIn("==== FAILED: pkg", out)
                self.assertIn(diagnosis, out)

    def test_package_load_tests_hook_controls_which_tests_run(self):
        hook = ("def load_tests(loader, tests, pattern):\n"
                "    return loader.loadTestsFromName('pkg.selected')\n")
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "pkg/__init__.py": hook,
                        "pkg/selected.py": PASS, "pkg/test_excluded.py": FAIL})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("4 tests, 0 ", out)

    def test_plain_package_still_discovers_nested_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "pkg/__init__.py": "",
                        "pkg/nested/__init__.py": "", "pkg/nested/test_p.py": PASS})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("4 tests, 0 ", out)

    def test_empty_targets_do_not_fail_a_nonempty_suite(self):
        for empty in ("pkg/__init__.py", "test_empty.py"):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as tmp:
                suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, empty: ""})
                code, out = run(tmp, "-j", "2")
                self.assertEqual(code, 0, out)
                self.assertIn("2 tests, 0 ", out)

    def test_suite_with_only_empty_targets_fails_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"pkg/__init__.py": "", "test_empty.py": ""})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1, out)
            self.assertIn("no tests discovered", out)

    def test_package_with_only_explicit_skips_succeeds(self):
        module_skip = "\ndef setUpModule():\n    raise unittest.SkipTest('optional runtime unavailable')\n"
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"pkg/__init__.py": FAIL + module_skip})
            code, out = run(tmp, "-j", "2", "-v")
            self.assertEqual(code, 0, out)
            self.assertIn("OK (skipped=1)", out)

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
