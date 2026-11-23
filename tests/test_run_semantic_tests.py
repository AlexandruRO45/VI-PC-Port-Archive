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
NESTED = ("import os, sys, unittest\nclass Inner(unittest.TestCase):\n    def test_i(self): pass\n"
          "class T(unittest.TestCase):\n    def test_a(self):\n"
          "        unittest.TextTestRunner(stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(Inner))\n"
          "        sys.stdout.flush()\n        os._exit(0)\n")


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

    def test_an_early_exit_with_status_0_fails_the_run(self):
        # os._exit(0) at import, inside a test, and after a nested runner printed its own passing summary: the
        # process exits 0 but unittest never finished (TEST-1844).
        cases = {
            "import": "import os\nos._exit(0)\n",
            "in_test": "import os, unittest\nclass T(unittest.TestCase):\n    def test_a(self): os._exit(0)\n",
            "nested": NESTED,
        }
        for name, text in cases.items():
            for jobs in ("1", "2"):
                with self.subTest(name, jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                    suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "test_x.py": text})
                    code, out = run(tmp, "-j", jobs)
                    self.assertEqual(code, 1, out)
                    self.assertIn("before unittest reported completing", out)
                    if jobs == "2":
                        self.assertIn("==== FAILED: test_x", out)
                        self.assertIn("2 files, 2 tests, 1 files failed", out)

    def test_the_completion_line_is_not_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS})
            code, ran, out = r.run_module(tmp, "test_p", False)
            self.assertEqual((code, ran), (0, 2), out)
            self.assertNotIn("leonida-semantic-done-", out)
            self.assertIn("OK", out)

    def test_an_empty_suite_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1)
            self.assertIn("no test*.py files", out)


if __name__ == "__main__":
    unittest.main()
