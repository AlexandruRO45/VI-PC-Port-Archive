"""scripts/run_semantic_tests.py: runs each test file in its own process and fails the run on any bad file.

    python -I tests/test_run_semantic_tests.py

Synthetic test files in a temporary folder; no translator, no game files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
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
UNICODE = '''
import unittest
class T(unittest.TestCase):
    def test_pixel(self): self.fail("unexpected pixel \\U0001F642")
'''
GRANDCHILD = '''
import subprocess, sys, time, unittest
class T(unittest.TestCase):
    def test_stuck(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])   # inherits stdout and stderr
        open({pidfile!r}, "w").write(str(p.pid))
        time.sleep(60)
'''
EMPTY = '''
import unittest
'''


def alive(pid):
    """True while the process runs (a zombie does not count)."""
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except ProcessLookupError:
        return False
    except OSError:   # no /proc: os.kill succeeded, so it is there
        return True


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
        self.assertIn("(timed out, tests not counted)", r.stdout)

    @unittest.skipIf(os.name == "nt", "checks the process by pid through /proc or os.kill")
    def test_a_timeout_stops_a_grandchild_that_holds_the_output_open(self):
        pidfile = os.path.join(tempfile.mkdtemp(), "pid")
        self.addCleanup(shutil.rmtree, os.path.dirname(pidfile), ignore_errors=True)
        d = self.folder(test_stuck=GRANDCHILD.format(pidfile=pidfile))
        t0 = time.monotonic()
        r = run(d, "--timeout", "3")
        pid = int(open(pidfile).read())
        self.addCleanup(lambda: alive(pid) and os.kill(pid, 9))
        self.assertLess(time.monotonic() - t0, 25)
        self.assertEqual(r.returncode, 1)
        self.assertIn("TIMED OUT: test_stuck.py", r.stdout)
        for _ in range(50):
            if not alive(pid):
                break
            time.sleep(0.1)
        self.assertFalse(alive(pid), "the grandchild of the timed-out file is still running")

    def test_files_in_a_sub_package_are_run_too(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        os.mkdir(os.path.join(d, "pkg"))
        for rel, body in (("test_top.py", PASS), ("pkg/__init__.py", ""), ("pkg/test_inner.py", FAIL)):
            with open(os.path.join(d, *rel.split("/")), "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(body))
        r = run(d)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("===== pkg/test_inner.py: failed", r.stdout)
        self.assertIn("2 files, 3 tests", r.stdout)

    def test_a_failure_report_survives_a_stdout_that_cannot_encode_it(self):
        # cp1251 on Windows, here ASCII: the runner must still print the assertion (escaped) and the summary
        env = dict(os.environ, PYTHONIOENCODING="ascii")
        r = subprocess.run([sys.executable, SCRIPT, "--start", self.folder(test_unicode=UNICODE), "--jobs", "1"],
                           capture_output=True, text=True, encoding="ascii", errors="replace", env=env, timeout=120)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("unexpected pixel \\U0001f642", r.stdout)
        self.assertIn("FAILED: test_unicode.py", r.stdout)
        self.assertNotIn("UnicodeEncodeError", r.stderr)

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
