"""Tests for scripts/run_stdlib_tests.py (parallel standard-library test runner).

    python -I tests/test_run_stdlib_tests.py

Exercises discovery parity, parallel execution, run-all-then-fail reporting, timeout handling,
empty-discovery refusal, and environment forwarding under ``python -I`` on synthetic temp trees.
Standard library only; no game files, no bash.
"""

import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_PATH = os.path.join(ROOT, "scripts", "run_stdlib_tests.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("run_stdlib_tests", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(root, rel, content):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


class RunStdlibTestsTest(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def test_discover_tests_matches_ci_predicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_file(tmp, "tests/test_alpha.py", 'if __name__ == "__main__":\n    pass\n')
            write_file(tmp, "tests/sub/test_nested.py", 'if __name__ == "__main__":\n    pass\n')
            write_file(tmp, "tests/test_pytest.py", 'import pytest\nif __name__ == "__main__":\n    pass\n')
            write_file(tmp, "tests/test_numpy.py", '  HAVE_NUMPY = True\nif __name__ == "__main__":\n    pass\n')
            write_file(tmp, "tests/test_no_main.py", "def test_only():\n    pass\n")
            write_file(tmp, "tests/semantic/test_sem.py", 'if __name__ == "__main__":\n    pass\n')
            write_file(tmp, "tests/helper.py", 'if __name__ == "__main__":\n    pass\n')

            found = self.runner.discover_tests(tmp)
            self.assertEqual(found, ["tests/sub/test_nested.py", "tests/test_alpha.py"])

    def test_runs_every_file_past_failures_and_reports_all_errors(self):
        failing = (
            "import sys\n"
            'if __name__ == "__main__":\n'
            '    print("boom from", __file__, file=sys.stderr)\n'
            "    sys.exit(2)\n"
        )
        passing = (
            "import sys\n"
            'if __name__ == "__main__":\n'
            '    print("ok")\n'
            "    sys.exit(0)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_file(tmp, "tests/test_a_fails.py", failing)
            write_file(tmp, "tests/test_b_passes.py", passing)
            write_file(tmp, "tests/test_c_fails.py", failing)
            write_file(tmp, "tests/test_d_pytest.py", "import pytest\n" + failing)
            write_file(tmp, "tests/semantic/test_e.py", failing)
            write_file(tmp, "tests/test_f_nomain.py", "raise SystemExit(9)\n")

            proc = subprocess.run(
                [sys.executable, "-I", RUNNER_PATH, "--root", tmp, "-j", "3"],
                capture_output=True,
                text=True,
                timeout=60,
            )

        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, out)
        ran = sorted(re.findall(r"^== (\S+)$", out, re.M))
        self.assertEqual(
            ran,
            ["tests/test_a_fails.py", "tests/test_b_passes.py", "tests/test_c_fails.py"],
            out,
        )
        errors = re.findall(r"^::error file=(\S+)::", out, re.M)
        self.assertEqual(errors, ["tests/test_a_fails.py", "tests/test_c_fails.py"], out)
        self.assertIn("boom from", out)
        self.assertIn("1 passed, 2 failed", out)

    def test_executes_tests_concurrently(self):
        # Two tests rendezvous via temp files: sequential execution would time out in one test,
        # while -j 2 lets both create their marker and see the peer's marker immediately.
        body = (
            "import os, sys, time\n"
            'if __name__ == "__main__":\n'
            "    me = os.path.join(os.getcwd(), {mine!r})\n"
            "    peer = os.path.join(os.getcwd(), {other!r})\n"
            '    with open(me, "w", encoding="utf-8", newline="\\n") as f:\n'
            '        f.write("ready")\n'
            "    deadline = time.monotonic() + 10.0\n"
            "    while not os.path.exists(peer):\n"
            "        if time.monotonic() > deadline:\n"
            '            sys.exit("peer did not run concurrently")\n'
            "        time.sleep(0.02)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_file(tmp, "tests/test_peer_1.py", body.format(mine="peer1.ready", other="peer2.ready"))
            write_file(tmp, "tests/test_peer_2.py", body.format(mine="peer2.ready", other="peer1.ready"))

            buf = io.StringIO()
            files = self.runner.discover_tests(tmp)
            rc = self.runner.run_suite(tmp, files, jobs=2, stream=buf)

        self.assertEqual(rc, 0, buf.getvalue())
        self.assertNotIn("::error", buf.getvalue())

    def test_timeout_fails_slow_test_and_still_runs_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_file(
                tmp,
                "tests/test_a_hangs.py",
                'import time\nif __name__ == "__main__":\n    time.sleep(30)\n',
            )
            write_file(
                tmp,
                "tests/test_b_fast.py",
                'if __name__ == "__main__":\n    print("fast-ok")\n',
            )

            buf = io.StringIO()
            files = self.runner.discover_tests(tmp)
            rc = self.runner.run_suite(tmp, files, jobs=2, timeout=2.0, stream=buf)

        out = buf.getvalue()
        self.assertEqual(rc, 1, out)
        self.assertIn("fast-ok", out)
        self.assertIn("::error file=tests/test_a_hangs.py::tests/test_a_hangs.py timed out", out)

    def test_zero_discovered_tests_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "tests"))
            buf = io.StringIO()
            rc = self.runner.run_suite(tmp, [], jobs=2, stream=buf)
        self.assertEqual(rc, 1)
        self.assertIn("0 standard-library test files", buf.getvalue())

    def test_env_vars_pass_through_isolated_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_file(
                tmp,
                "tests/test_env_check.py",
                "import os, sys\n"
                'if __name__ == "__main__":\n'
                '    assert os.environ.get("RPF9DUMP_DLL") == "fake/rpf9dump.dll"\n',
            )
            write_file(
                tmp,
                "tests/test_custom_marker.py",
                'CUSTOM_SKIP = 1\nif __name__ == "__main__":\n    raise SystemExit(3)\n',
            )
            env = dict(
                os.environ,
                RPF9DUMP_DLL="fake/rpf9dump.dll",
                PYTEST_FILE_MARKERS=r"^CUSTOM_SKIP =",
            )
            proc = subprocess.run(
                [sys.executable, "-I", RUNNER_PATH, "--root", tmp, "-j", "2"],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out)
        self.assertEqual(re.findall(r"^== (\S+)$", out, re.M), ["tests/test_env_check.py"], out)

    def test_captured_non_ascii_output_matches_child_encoding(self):
        encoding = self.runner.child_output_encoding()
        probe = None
        for candidate in ("caf\u00e9", "\u043f\u0440\u0438\u0432\u0435\u0442", "\u3042"):
            try:
                candidate.encode(encoding)
            except UnicodeEncodeError:
                continue
            probe = candidate
            break
        if probe is None:
            self.skipTest(f"locale encoding {encoding!r} cannot represent probe strings")

        with tempfile.TemporaryDirectory() as tmp:
            write_file(
                tmp,
                "tests/test_non_ascii.py",
                f'if __name__ == "__main__":\n    print({probe!r})\n',
            )
            res = self.runner.run_one(tmp, "tests/test_non_ascii.py")

        self.assertEqual(res.returncode, 0, res.output)
        self.assertIn(probe, res.output)
        self.assertNotIn("\ufffd", res.output)
        self.assertNotIn("\r\n", res.output)

    def test_decode_bytes_uses_explicit_child_encoding_and_normalizes_crlf(self):
        self.assertEqual(self.runner._decode_bytes(b"caf\xe9\r\n", encoding="cp1252"), "caf\u00e9\n")

    def test_timeout_kills_helper_process_tree_and_does_not_wait_for_inherited_pipe(self):
        # Covers both helper modes: inheriting stdout/stderr (which would otherwise block communicate()
        # until the helper exits) and redirecting to DEVNULL (which would otherwise leave an orphan
        # helper running after the parent is killed).
        helper_code = (
            "import os, sys, time; "
            "open(sys.argv[1], 'w', encoding='utf-8', newline='\\n').write('started'); "
            "time.sleep(3.5); "
            "open(sys.argv[2], 'w', encoding='utf-8', newline='\\n').write('late')"
        )
        for mode in ("inherited", "devnull"):
            with self.subTest(mode=mode):
                redirect = ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" if mode == "devnull" else ""
                with tempfile.TemporaryDirectory() as tmp:
                    started = os.path.join(tmp, "started.txt")
                    late = os.path.join(tmp, "late.txt")
                    write_file(
                        tmp,
                        "tests/test_spawns_helper.py",
                        "import subprocess, sys\n"
                        'if __name__ == "__main__":\n'
                        f"    subprocess.run([sys.executable, '-I', '-c', {helper_code!r}, "
                        f"{started!r}, {late!r}]{redirect})\n",
                    )
                    t0 = time.monotonic()
                    res = self.runner.run_one(tmp, "tests/test_spawns_helper.py", timeout=1.5)
                    wall = time.monotonic() - t0

                    self.assertTrue(res.timed_out, res.output)
                    self.assertEqual(res.returncode, 124, res.output)
                    self.assertTrue(os.path.exists(started), "helper process did not start before timeout")
                    self.assertLess(wall, 3.2, f"runner waited past timeout ({wall:.2f}s) for {mode} helper")
                    # Wait past the helper's 3.5s sleep to verify it was killed, not left running.
                    remaining = 4.0 - (time.monotonic() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
                    self.assertFalse(os.path.exists(late), f"{mode} helper survived timeout and wrote late marker")


if __name__ == "__main__":
    unittest.main()
