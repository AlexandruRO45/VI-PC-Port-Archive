"""scripts/run_semantic_tests.py on a throwaway suite: it finds what unittest discover finds, and a failing, crashing
or unimportable file always makes it exit 1 with that file named, in parallel and with -j 1. On Linux, the default -j
follows the CPUs the runner may use: one usable CPU runs discover, two run two files at a time. Only the default
reads them, and one it cannot read falls back to the host's count; --help and an explicit -j never need it.

    python -I tests/test_run_semantic_tests.py
"""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

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


PID = "import os\nimport unittest\nclass T(unittest.TestCase):\n    def test_a(self): print('worker', os.getpid())\n"


@unittest.skipUnless(hasattr(os, "sched_setaffinity"), "needs a settable CPU affinity mask (Linux)")
class UsableCpusTest(unittest.TestCase):
    """The default -j counts the CPUs the runner may use (taskset, a container's cpuset), not every CPU of the host."""

    def run_on(self, cpus, tmp):
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "run_semantic_tests.py"), tmp, "-v"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                           preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        return p.returncode, p.stdout

    def test_one_usable_cpu_runs_discover(self):
        cpu = min(os.sched_getaffinity(0))
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_p1.py": PID, "test_p2.py": PID, "test_p3.py": PID})
            code, out = self.run_on({cpu}, tmp)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(set(re.findall(r"worker (\d+)", out))), 1, out)  # one discover process
        self.assertNotIn(" jobs", out)

    def test_two_usable_cpus_run_two_jobs(self):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 3:
            self.skipTest("needs 3 or more usable CPUs to tell 2 jobs from all of them")
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_p1.py": PID, "test_p2.py": PID, "test_p3.py": PID})
            code, out = self.run_on(set(cpus[:2]), tmp)
        self.assertEqual(code, 0, out)
        self.assertIn("3 files, 3 tests, 0 files failed", out)
        self.assertTrue(out.rstrip().endswith("with 2 jobs"), out)


def refused(*args):
    raise PermissionError(1, "Operation not permitted")  # sched_getaffinity under a seccomp or LSM policy


class CpuProbeTest(unittest.TestCase):
    """Reading the CPU affinity mask can fail or give no answer: only the default -j reads it, and then falls back."""

    def test_help_and_an_explicit_job_count_never_probe_the_cpus(self):
        probe = mock.Mock(side_effect=refused)
        with mock.patch.object(os, "process_cpu_count", probe, create=True), \
                mock.patch.object(os, "sched_getaffinity", probe, create=True):
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(io.StringIO()) as out:
                r.main(["--help"])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn("--jobs", out.getvalue())
            with tempfile.TemporaryDirectory() as tmp:
                suite(tmp, {"helper.py": "X = 1\n", "test_p1.py": PASS, "test_p2.py": PASS})
                code, out = run(tmp, "-j", "2")
                self.assertEqual(code, 0, out)
                self.assertTrue(out.rstrip().endswith("with 2 jobs"), out)
                self.assertEqual(run(tmp, "-j", "1")[0], 0)
        probe.assert_not_called()

    def test_a_refused_or_empty_probe_falls_back_to_the_host_count_then_1(self):
        def cpus(**os_api):  # usable_cpus() on a Python whose os module has only these
            with mock.patch.object(r, "os", types.SimpleNamespace(**os_api)):
                return r.usable_cpus()

        for name in ("process_cpu_count", "sched_getaffinity"):  # Python 3.13+, Python 3.12
            self.assertEqual(cpus(**{name: refused}, cpu_count=lambda: 3), 3, name)
            self.assertEqual(cpus(**{name: refused}, cpu_count=lambda: None), 1, name)
        self.assertEqual(cpus(process_cpu_count=lambda: None, cpu_count=lambda: 3), 3)
        self.assertEqual(cpus(sched_getaffinity=lambda pid: {0, 1}, cpu_count=lambda: 3), 2)
        self.assertEqual(cpus(cpu_count=lambda: None), 1)
        with tempfile.TemporaryDirectory() as tmp:  # and the default -j of a run takes that count
            suite(tmp, {"helper.py": "X = 1\n", "test_p1.py": PASS, "test_p2.py": PASS})
            with mock.patch.object(os, "process_cpu_count", refused, create=True), \
                    mock.patch.object(os, "sched_getaffinity", refused, create=True), \
                    mock.patch.object(os, "cpu_count", return_value=2):
                code, out = run(tmp)
            self.assertEqual(code, 0, out)
            self.assertTrue(out.rstrip().endswith("with 2 jobs"), out)


if __name__ == "__main__":
    unittest.main()
