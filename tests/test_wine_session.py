"""Stopping a Wine prefix's leftover wineserver (runtime/test/wine_session.py) and the scripts that do it.

    python -I tests/test_wine_session.py

No Wine and no X server: wine, wineserver and xvfb-run are small Python scripts in a scratch folder that write
each call to a log, so what is under test is who gets called, in which order, and that a hung wineserver is left.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "runtime" / "test"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load("wine_session", TEST_DIR / "wine_session.py")
tiny = load("check_tiny_frame", TEST_DIR / "check_tiny_frame.py")
sys.path.insert(0, str(ROOT / "scripts"))
import run_wine_tests as rwt  # noqa: E402

# Each dummy logs "<name> <arguments> | WINEPREFIX" to $CALLS_LOG. wineserver -w sleeps when $HANG_W is set;
# wine prints $WINE_OUTPUT (a file) and then runs its arguments as a Python program when they name a .py file;
# xvfb-run drops -a and runs the rest.
LOGGER = (
    "import os, sys, time\n"
    "name = os.path.basename(sys.argv[0])\n"
    "with open(os.environ['CALLS_LOG'], 'a') as log:\n"
    "    log.write(f\"{name} {' '.join(sys.argv[1:])} | {os.environ.get('WINEPREFIX', '')}\\n\")\n"
)
WINESERVER = LOGGER + "if sys.argv[1:] == ['-w'] and os.environ.get('HANG_W'):\n    time.sleep(120)\n"
WINE = LOGGER + (
    "if os.environ.get('WINE_OUTPUT'):\n"
    "    sys.stdout.write(open(os.environ['WINE_OUTPUT']).read())\n"
    "if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):\n"
    "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
)
XVFB_RUN = LOGGER + (
    "args = [a for a in sys.argv[1:] if a != '-a']\n"
    "os.execv(args[0], args)\n"
)


@unittest.skipIf(os.name == "nt", "the dummy programs are POSIX scripts")
class Dummies(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="wine-session-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self.log = self.dir / "calls.log"
        self.wine = self.script(self.bin / "wine", WINE)
        self.wineserver = self.script(self.bin / "wineserver", WINESERVER)
        self.xvfb_run = self.script(self.bin / "xvfb-run", XVFB_RUN)
        self.env = dict(os.environ, CALLS_LOG=str(self.log), WINEPREFIX=str(self.dir / "prefix"))
        patch = mock.patch.dict(os.environ, self.env)
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def script(path, body):
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def calls(self):
        """The logged calls without the prefix, e.g. ["wineserver -k", "wine x"]."""
        if not self.log.exists():
            return []
        return [line.split(" | ")[0].strip() for line in self.log.read_text(encoding="utf-8").splitlines()]


class StopWineserverTest(Dummies):
    def test_the_prefix_server_is_ended_then_waited_for(self):
        note = session.stop_wineserver([self.xvfb_run, "-a", self.wine], self.env)
        self.assertEqual(self.calls(), ["wineserver -k", "wineserver -w"])
        prefixes = {line.split(" | ")[1] for line in self.log.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(prefixes, {str(self.dir / "prefix")})
        self.assertIn("gone", note)

    def test_the_wineserver_beside_wine_is_used_not_the_one_on_path(self):
        other = self.dir / "other"
        other.mkdir()
        self.script(other / "wineserver", WINESERVER.replace("name = os.path.basename(sys.argv[0])",
                                                             "name = 'path-wineserver'"))
        with mock.patch.dict(os.environ, {"PATH": str(other)}):
            session.stop_wineserver([self.wine], self.env)
        self.assertEqual(self.calls(), ["wineserver -k", "wineserver -w"])

    def test_a_wineserver_that_never_returns_is_given_up_on(self):
        started = time.monotonic()
        note = session.stop_wineserver([self.wine], dict(self.env, HANG_W="1"), timeout=1)
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(self.calls(), ["wineserver -k", "wineserver -w"])
        self.assertIn("did not return", note)

    def test_a_command_without_wine_stops_nothing_and_warns(self):
        note = session.stop_wineserver([sys.executable], self.env)
        self.assertEqual(self.calls(), [])
        self.assertTrue(note.startswith("WARNING: no wine"), note)

    def test_a_server_still_there_after_k_and_w_is_not_called_gone(self):
        # A wineserver of another Wine build cannot end this prefix's server: -k and -w return, its socket stays.
        (self.dir / "prefix").mkdir()
        root = self.dir / "tmp"
        with mock.patch.object(session, "SERVER_ROOT", str(root)):
            socket = Path(session.server_socket(self.env))
            socket.parent.mkdir(parents=True)
            socket.touch()
            note = session.stop_wineserver([self.wine], self.env)
            self.assertTrue(note.startswith("WARNING"), note)
            self.assertNotIn("gone", note)
            socket.unlink()  # the control: once the socket is gone, so is the server
            self.assertIn("gone", session.stop_wineserver([self.wine], self.env))


class WineBinaryTest(unittest.TestCase):
    def test_wrappers_named_after_wine_count_and_wineserver_does_not(self):
        for word in ("wine", "/usr/bin/wine64", "/usr/bin/wine-stable", "wine-staging", "/opt/x/wine64-devel"):
            with self.subTest(word=word):
                self.assertEqual(session.wine_binary(["xvfb-run", "-a", word], env={}), word)
        for command in (["wineserver", "-k"], ["winecfg"], ["xvfb-run", "-a", "proton-ge"]):
            with self.subTest(command=command):
                self.assertIsNone(session.wine_binary(command, env={}))

    def test_proton_run_names_protons_own_wine(self):
        with tempfile.TemporaryDirectory() as folder:
            proton = Path(folder) / "proton"
            wine = Path(folder) / "files" / "bin" / "wine"
            wine.parent.mkdir(parents=True)
            proton.touch()
            wine.touch()
            self.assertEqual(session.wine_binary(["xvfb-run", "-a", str(proton), "run"], env={}), str(wine))

    def test_wineloader_names_the_wine_of_a_wrapper_with_another_name(self):
        self.assertEqual(session.wine_binary(["my-runner"], env={"WINELOADER": "/opt/w/bin/wine"}), "/opt/w/bin/wine")


class RunTest(Dummies):
    def test_a_process_left_holding_the_output_does_not_hold_the_run(self):
        # The program exits at once but leaves a child (as Wine leaves wineserver) that keeps its output open.
        program = self.script(self.bin / "leaves_a_child.py", (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print('done', flush=True)\n"))
        started = time.monotonic()
        code, output = session.run([program], timeout=30)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual((code, output.strip()), (0, "done"))

    def test_a_run_past_its_timeout_is_killed_with_its_children(self):
        program = self.script(self.bin / "never_ends.py", "import time\ntime.sleep(60)\n")
        started = time.monotonic()
        code, _ = session.run([program], timeout=1)
        self.assertLess(time.monotonic() - started, 20)
        self.assertIsNone(code)

    def test_ctrl_c_kills_the_run_with_its_children_and_still_interrupts(self):
        pids = self.dir / "pids.txt"
        program = self.script(self.bin / "waits.py", (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open({str(pids)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(60)\n"))
        harness = ("import importlib.util, sys\n"
                   f"spec = importlib.util.spec_from_file_location('wine_session', {str(TEST_DIR / 'wine_session.py')!r})\n"
                   "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                   f"m.run([{program!r}], timeout=60)\n")
        runner = subprocess.Popen([sys.executable, "-I", "-c", harness], stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 20
            while not (pids.exists() and pids.read_text()) and time.monotonic() < deadline:
                time.sleep(0.1)
            child = int(pids.read_text())
            runner.send_signal(signal.SIGINT)
            self.assertNotEqual(runner.wait(timeout=30), 0)
            for _ in range(100):  # SIGKILL has been sent; give the kernel a moment to end the child
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail("the program's child survived Ctrl-C")
        finally:
            runner.kill()
            runner.wait()


    def test_ctrl_c_under_xvfb_run_still_stops_the_prefix_after_the_run(self):
        # The prefix's wineserver has a session of its own, so the group kill does not reach it: the stop after the
        # run must happen on Ctrl-C too, not only on a normal exit.
        stops = []
        program = self.script(self.bin / "sleeps.py", "import time\ntime.sleep(60)\n")
        real_wait = subprocess.Popen.wait
        interrupted = []

        def wait(process, timeout=None):
            if not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt
            return real_wait(process, timeout=timeout)

        with mock.patch.object(session, "starts_x_server", return_value=True), \
                mock.patch.object(session, "stop_and_say", side_effect=lambda command, env: stops.append(command)), \
                mock.patch.object(subprocess.Popen, "wait", wait):
            with self.assertRaises(KeyboardInterrupt):
                session.run([program], timeout=60)
        self.assertEqual(len(stops), 2, "the prefix is stopped before the run and again after the interrupt")


class CheckTinyFrameTest(Dummies):
    """check_tiny_frame.py stops the prefix before a run only when its runner starts its own X server."""

    def run_check(self, *runner):
        output = self.dir / "output.txt"
        good = ["    shim          d3d12_xs.dll!D3D12XboxCreateDevice"]
        good += [f"[d3d12-trace] #{number} {call}" for number, call in enumerate(tiny.EXPECTED, 1)]
        good += ["tiny_frame: OK", "game exited with code 0", ""]
        output.write_text("\n".join(good), encoding="utf-8")
        guest = self.dir / "tiny_frame.exe"
        guest.write_text("", encoding="utf-8")
        command = [sys.executable, "-I", str(TEST_DIR / "check_tiny_frame.py"), "gta6rt.exe", str(guest)]
        command += [f"--runner={word}" for word in runner]
        return subprocess.run(command, env=dict(self.env, WINE_OUTPUT=str(output)), capture_output=True, text=True,
                              timeout=120)

    def test_an_xvfb_run_runner_stops_the_prefix_before_and_after_the_run(self):
        run = self.run_check(self.xvfb_run, "-a", self.wine)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        calls = self.calls()
        self.assertEqual(calls[:2], ["wineserver -k", "wineserver -w"])
        self.assertEqual([call.split()[0] for call in calls[2:4]], ["xvfb-run", "wine"])
        self.assertEqual(calls[4:], ["wineserver -k", "wineserver -w"])

    def test_a_plain_wine_runner_leaves_the_prefix_alone(self):
        run = self.run_check(self.wine)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual([call.split()[0] for call in self.calls()], ["wine"])


class CheckTinyFrameProbeTest(unittest.TestCase):
    def test_a_probe_that_never_exits_fails_before_the_run(self):
        argv = ["check_tiny_frame.py", "gta6rt.exe", "tiny_frame.exe", "--probe", "d3d12_com_test.exe",
                "--runner", "wine"]
        out = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), \
                mock.patch.object(tiny.wine_session, "run", return_value=(None, "")) as run:
            self.assertEqual(tiny.main(), 1)
        self.assertEqual(run.call_count, 1)  # the loader was not started
        self.assertIn("FAIL: the probe d3d12_com_test.exe did not exit", out.getvalue())


class RunWineTestsTest(Dummies):
    """run_wine_tests.py stops the prefix once before the tests (each test then runs under the same X server)."""

    def run_main(self, wine):
        artifact = self.dir / "artifact"
        artifact.mkdir()
        (artifact / "ok.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        listing = self.dir / "ctest.json"
        tests = [{"name": f"ok{n}", "command": ["D:/win/root/artifact/ok.py"]} for n in (1, 2)]
        listing.write_text(json.dumps({"tests": tests}), encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", "D:/win/root", "--root", str(self.dir),
                "--wine", wine]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return rwt.main()

    def test_the_prefix_is_stopped_once_before_the_tests(self):
        self.assertEqual(self.run_main(self.wine), 0)
        self.assertEqual([call.split()[0] + (" " + call.split()[1] if call.startswith("wineserver") else "")
                          for call in self.calls()], ["wineserver -k", "wineserver -w", "wine", "wine"])

    def test_with_an_xvfb_run_wine_each_test_runs_between_two_stops(self):
        self.assertEqual(self.run_main(f"{self.xvfb_run} -a {self.wine}"), 0)
        names = [call.split()[0] for call in self.calls()]
        stop = ["wineserver", "wineserver"]
        self.assertEqual(names, stop + (stop + ["xvfb-run", "wine"] + stop) * 2)

    def test_a_test_that_leaves_a_process_holding_its_output_passes_in_time(self):
        # As Wine leaves wineserver: the program exits at once, a child it started keeps its output open.
        artifact = self.dir / "artifact"
        artifact.mkdir()
        (artifact / "leaves.py").write_text(
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print('done', flush=True)\n", encoding="utf-8")
        listing = self.dir / "ctest.json"
        listing.write_text(json.dumps({"tests": [{"name": "leaves", "command": ["D:/win/root/artifact/leaves.py"]}]}),
                           encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", "D:/win/root", "--root", str(self.dir),
                "--wine", self.wine, "--timeout", "10"]
        out = io.StringIO()
        started = time.monotonic()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            rwt.main()
        self.assertLess(time.monotonic() - started, 9)
        self.assertIn("[PASS] leaves", out.getvalue())


if __name__ == "__main__":
    unittest.main()
