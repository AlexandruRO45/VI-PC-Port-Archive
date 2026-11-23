"""CPU-only, synthetic Windows process and trace tests for the boot runner."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_boot


GUEST = """
import ctypes, os, pathlib, subprocess, sys, time
out = pathlib.Path.cwd()
assert pathlib.Path(sys.argv[sys.argv.index('--overlay') + 1]) == out / 'overlay'
print('synthetic stdout', flush=True)
print('[d3d12-trace] #1 Fixture ok', file=sys.stderr, flush=True)
scenario = pathlib.Path(sys.argv[2]).read_text()
if scenario != 'parser':
    with pathlib.Path(sys.argv[sys.argv.index('--trace') + 1]).open('xb') as f:
        f.write(b'{"kind":"map","module":"fixture"}\\n')
if scenario in ('hang', 'orphan'):
    child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'])
    (out / 'child.pid').write_bytes(str(child.pid).encode())
    if scenario == 'hang':
        time.sleep(120)
if scenario == 'crash':
    ctypes.windll.kernel32.SetErrorMode(2)
    ctypes.windll.kernel32.RaiseFailFastException(None, None, 0)
sys.exit(17 if scenario == 'nonzero' else 0)
"""


@unittest.skipUnless(os.name == "nt", "Windows job objects")
class Boot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pkg = self.root / "package"
        self.pkg.mkdir()
        self.image = self.pkg / "synthetic"
        self.out = self.root / "private output"
        self.fixture = self.root / "guest.py"
        self.fixture.write_text(GUEST, encoding="utf-8", newline="\n")

    def invoke(self, scenario, *extra, runtime=sys.executable, fail_capture=False, mode="run"):
        self.image.write_bytes(scenario.encode())
        args = [mode, "--runtime", runtime, "--image", self.image, "--game-dir", self.pkg,
                "--out", self.out, "--deadline", "2" if scenario == "hang" else "10", *extra]
        command = run_boot.loader_command

        def synthetic(*args):
            exe, *options = command(*args)
            return [exe, "-I", str(self.fixture), *options]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run_boot, "loader_command", synthetic))
            if fail_capture:
                # Real writer and exit status, with an invalid target PID.
                capture = run_boot.diagnostic
                stack.enter_context(patch.object(run_boot, "diagnostic",
                                                lambda _, *a: capture(0, *a)))
            rc = run_boot.main(list(map(str, args)))
        return rc, json.loads((self.out / "result.json").read_bytes())

    def test_exits_and_partial_parser(self):
        for scenario, expected, parser in (("normal", 0, 0), ("nonzero", 17, 0),
                                           ("crash", 0xc0000602, 0), ("parser", 0, 2)):
            with self.subTest(scenario=scenario):
                self.out = self.root / scenario
                rc, r = self.invoke(scenario)
                self.assertEqual((r["status"], r["loader_exit_code"], r["parser_exit_code"]),
                                 ("exited", expected, parser))
                self.assertEqual(rc, int(bool(expected or parser)))
                self.assertIn(b"synthetic stdout", (self.out / "stdout.log").read_bytes())
                summary = json.loads((self.out / "summary.json").read_bytes())
                self.assertEqual(summary["d3d12"]["calls"], 1)
                self.assertEqual(summary["startup"] is None, scenario == "parser")
                self.assertEqual(r["runtime_sha256"], hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())

    def test_earlier_passes(self):
        for mode in ("dry-run", "init"):
            self.out = self.root / mode
            rc, report = self.invoke("normal", mode=mode)
            self.assertEqual(rc, 0)
            self.assertIn("--" + mode, report["command"])
            self.assertEqual("--trace-d3d12" in report["command"], mode == "init")
            summary = json.loads((self.out / "summary.json").read_bytes())
            self.assertEqual(summary["d3d12"] is None, mode == "dry-run")

    def assert_child_stopped(self):
        import _winapi
        pid = int((self.out / "child.pid").read_bytes())
        try:
            process = _winapi.OpenProcess(0x100000, False, pid)
        except OSError as exc:
            self.assertEqual(exc.winerror, 87)  # PID no longer exists
        else:
            try:
                self.assertEqual(_winapi.WaitForSingleObject(process, 2000), 0)
            finally:
                _winapi.CloseHandle(process)

    def test_timeout_and_capture(self):
        for fail in (False, True):
            self.out = self.root / str(fail)
            rc, r = self.invoke("hang", "--diagnose-after", "0.5", fail_capture=fail)
            self.assertEqual((rc, r["status"], r["loader_exit_code"]), (1, "timeout", None))
            self.assertEqual([c["status"] for c in r["captures"]], ["failed" if fail else "captured"] * 2)
            if not fail:
                self.assertEqual((self.out / "deadline.dmp").read_bytes()[:4], b"MDMP")
            self.assertEqual(r["parser_exit_code"], 0)
            self.assert_child_stopped()

    def test_stalled_dump_writer(self):
        self.out.mkdir()
        writer = self.root / "boot_windows.py"
        writer.write_text("import os, pathlib, time\n"
                          "pathlib.Path('child.pid').write_bytes(str(os.getpid()).encode())\n"
                          "time.sleep(120)\n", encoding="utf-8", newline="\n")
        with patch.object(run_boot, "HERE", self.root):
            result = run_boot.diagnostic(os.getpid(), self.out, "requested", 1)
        self.assertEqual(result["status"], "timeout")
        self.assert_child_stopped()

    def test_orphan(self):
        rc, _ = self.invoke("orphan")
        self.assertEqual(rc, 0)
        self.assert_child_stopped()

    def test_launch_failure(self):
        rc, r = self.invoke("normal", runtime=self.image)
        self.assertEqual((rc, r["status"], r["loader_exit_code"]), (1, "launch_failure", None))
        self.assertIn("process_error", r)

    def test_collisions(self):
        for self.out in (self.pkg / "nested", self.root / "existing"):
            if self.out.name == "existing":
                self.out.mkdir()
                (self.out / "startup.jsonl").write_bytes(b"keep")
            with self.assertRaises(SystemExit):
                self.invoke("normal")
        self.assertEqual((self.out / "startup.jsonl").read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
