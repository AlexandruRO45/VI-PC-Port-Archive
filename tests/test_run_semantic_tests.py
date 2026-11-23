"""scripts/run_semantic_tests.py: runs each test file in its own process and fails the run on any bad file.

    python -I tests/test_run_semantic_tests.py

Synthetic test files in a temporary folder; no translator, no game files.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_semantic_tests.py")

PASS = '''
import unittest
class T(unittest.TestCase):
    def test_a(self): pass
    def test_b(self): pass
'''
FAIL = '''
import unittest
class T(unittest.TestCase):
    def test_wrong(self): self.assertEqual(1, 2, "the failing assertion")
'''
HANG = '''
import time, unittest
class T(unittest.TestCase):
    def test_stuck(self): time.sleep(60)
'''
EMPTY = '''
import unittest
'''


def run(folder, *extra):
    return subprocess.run([sys.executable, "-I", SCRIPT, "--start", folder, "--jobs", "2", *extra],
                          capture_output=True, text=True, timeout=120)


class RunSemanticTests(unittest.TestCase):
    def folder(self, **files):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: [os.remove(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".py")])
        for name, body in files.items():
            with open(os.path.join(d, name + ".py"), "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(body))
        return d

    def test_passing_files_exit_0_and_count_the_tests(self):
        r = run(self.folder(test_one=PASS, test_two=PASS))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("2 files, 4 tests, 0 files not ok", r.stdout)

    def test_a_failing_file_exits_1_and_shows_its_output(self):
        r = run(self.folder(test_good=PASS, test_bad=FAIL))
        self.assertEqual(r.returncode, 1)
        self.assertIn("===== test_bad.py: failed", r.stdout)
        self.assertIn("the failing assertion", r.stdout)
        self.assertNotIn("===== test_good.py", r.stdout)

    def test_a_stuck_file_is_stopped_at_the_timeout_and_fails_the_run(self):
        d = self.folder(test_good=PASS, test_stuck=HANG)
        r = run(d, "--timeout", "3")
        self.assertEqual(r.returncode, 1)
        self.assertIn("TIMED OUT: test_stuck.py", r.stdout)
        self.assertIn("1 files not ok", r.stdout)

    def test_a_file_that_runs_no_test_fails(self):
        r = run(self.folder(test_empty=EMPTY))
        self.assertEqual(r.returncode, 1)
        self.assertIn("no test ran in this file", r.stdout)

    def test_a_folder_without_test_files_fails(self):
        r = run(self.folder(helper=PASS))
        self.assertEqual(r.returncode, 1)
        self.assertIn("no test*.py file", r.stderr)


if __name__ == "__main__":
    unittest.main()
