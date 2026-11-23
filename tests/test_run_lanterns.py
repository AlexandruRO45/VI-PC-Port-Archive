"""run_lanterns.py: its timeout keeps its promise against a launcher that never ends and a child that leaves its
session, each run gets its own folder, and only a complete transcript (exit 0, the stage's whole PASS line once with
numbers that fit, the shim's Vulkan device line) counts as a pass.

    python -I tests/test_run_lanterns.py
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "runtime", "test", "lanterns", "run_lanterns.py")
sys.path.insert(0, os.path.dirname(RUNNER))

import run_lanterns  # noqa: E402

FACTORY = "[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy 0x1f2e3d4c (Vulkan device 0x5a6b7c8d)"
PASS = "Lanterns stage 1: PASS, 90000 of 98304 pixels exact, 8304 edge pixels plausible, depth-off control changed 20000 pixels"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def fake_run(tmp, script, timeout="2", env=None, wine=False):
    """Run the runner with a shell script as gta6rt; returns its result and the runner's out folder."""
    fake = os.path.join(tmp, "gta6rt")
    with open(fake, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n" + script)
    os.chmod(fake, 0o755)
    shaders = os.path.join(tmp, "shaders")
    os.makedirs(os.path.join(shaders, "spirv"), exist_ok=True)
    open(os.path.join(shaders, "spirv", "index.json"), "w").close()
    out = os.path.join(tmp, "out")
    r = subprocess.run([sys.executable, "-I", RUNNER, "--gta6rt", fake, "--guest", fake, "--shaders", shaders,
                        "--out", out, "--timeout", timeout] + (["--wine"] if wine else []),
                       capture_output=True, text=True, timeout=60, env=env)
    return r, out


@unittest.skipIf(os.name == "nt", "the stalled launcher is a POSIX shell script")
class StalledLauncher(unittest.TestCase):
    def test_timeout_kills_the_launcher_and_its_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            pids = os.path.join(tmp, "pids")
            # A "gta6rt" that starts a child, and a grandchild that leaves the session and loses its parent (so killing
            # the process group misses it), then hangs, as a guest stuck on a fence or a hung GPU would.
            escape = os.path.join(tmp, "escape.py")
            with open(escape, "w", encoding="utf-8") as f:
                f.write("import os, sys, time\nos.setsid()\n"
                        "with open(sys.argv[1], 'a') as f:\n    f.write(f'{os.getpid()}\\n')\ntime.sleep(300)\n")
            script = (f"echo $$ > {pids}\nsleep 300 &\necho $! >> {pids}\n"
                      f"( '{sys.executable}' '{escape}' '{pids}' & )\nwait\n")
            started = time.monotonic()
            r, _ = fake_run(tmp, script)
            took = time.monotonic() - started
            with open(pids, encoding="utf-8") as f:
                launched = [int(line) for line in f.read().split()]
            self.addCleanup(lambda: [os.kill(p, 9) for p in launched if alive(p)])
            self.assertEqual(r.returncode, run_lanterns.TIMEOUT_EXIT, r.stderr)
            self.assertLess(took, 30)
            self.assertEqual(len(launched), 3)
            deadline = time.monotonic() + 5
            while any(alive(p) for p in launched) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse([p for p in launched if alive(p)])

    def test_wine_mode_stops_the_prefix_wineserver_first(self):
        # A wineserver and a wine that only log their calls: -k, then -w, then the launch. A wineserver that hangs
        # fails the run within its timeout, before any launch.
        for hang, want_code, want_calls in ((False, 1, ["wineserver -k", "wineserver -w", "wine"]),
                                            (True, run_lanterns.CLEANUP_EXIT, ["wineserver -k"])):
            with self.subTest(hang=hang), tempfile.TemporaryDirectory() as tmp:
                calls = os.path.join(tmp, "calls")
                bin_dir = os.path.join(tmp, "bin")
                os.makedirs(bin_dir)
                for name, body in (("wineserver", f'echo "wineserver $1" >> {calls}\n' + ("exec sleep 300\n" if hang else "")),
                                   ("wine", f"echo wine >> {calls}\n")):
                    with open(os.path.join(bin_dir, name), "w", encoding="utf-8") as f:
                        f.write("#!/bin/sh\n" + body)
                    os.chmod(os.path.join(bin_dir, name), 0o755)
                env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ.get("PATH", ""))
                started = time.monotonic()
                r = fake_run(tmp, "exit 0\n", env=env, wine=True)[0]
                self.assertEqual(r.returncode, want_code, r.stderr)
                self.assertLess(time.monotonic() - started, run_lanterns.WINESERVER_TIMEOUT + 15)
                with open(calls, encoding="utf-8") as f:
                    self.assertEqual(f.read().split("\n")[:-1], want_calls)

    def test_each_run_has_its_own_folder(self):
        # Two runs of a launcher that exits at once without a PASS line: both fail, each in a folder of its own.
        with tempfile.TemporaryDirectory() as tmp:
            first, out = fake_run(tmp, "echo started\n")
            second, _ = fake_run(tmp, "echo started\n")
            self.assertEqual((first.returncode, second.returncode), (1, 1))
            runs = sorted(os.listdir(out))
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertTrue(run.startswith("stage1-"))
                self.assertTrue(os.path.isfile(os.path.join(out, run, "gta6rt-stage1.stdout")))


class Transcript(unittest.TestCase):
    def test_only_a_complete_transcript_passes(self):
        check = run_lanterns.check_transcript
        self.assertEqual(check(1, 0, PASS + "\n", FACTORY + "\n"), [])
        self.assertEqual(check(1, 0, PASS + "\r\n", FACTORY + "\r\n"), [])
        cases = {
            "nonzero exit": (1, PASS + "\n", FACTORY + "\n"),
            "no PASS line": (0, "lanterns: stage 1 FAILED\n", FACTORY + "\n"),
            "PASS twice": (0, PASS + "\n" + PASS + "\n", FACTORY + "\n"),
            "PASS inside another line": (0, "echo " + PASS + "\n", FACTORY + "\n"),
            "no device line": (0, PASS + "\n", ""),
            "host device, not the shim's": (0, PASS + "\n", "[rt] d3d12_xs: D3D12XboxCreateDevice -> host device\n"),
            "null device pointer": (0, PASS + "\n", FACTORY.replace("0x5a6b7c8d", "0x0") + "\n"),
            "PASS and then words": (0, "Lanterns stage 1: PASS, nonsense\n", FACTORY + "\n"),
            "PASS with counts that do not add up": (0, PASS.replace("8304 edge", "8303 edge") + "\n", FACTORY + "\n"),
            "PASS with no depth-off change": (0, PASS.replace("changed 20000", "changed 0") + "\n", FACTORY + "\n"),
        }
        for name, (code, out, err) in cases.items():
            with self.subTest(name):
                self.assertTrue(check(1, code, out, err))


if __name__ == "__main__":
    unittest.main()
