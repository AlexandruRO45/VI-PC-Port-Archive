"""run_lanterns.py: its timeout keeps its promise against a launcher that never ends, and only a complete transcript
(exit 0, the PASS line once, the shim's Vulkan device line) counts as a pass.

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


@unittest.skipIf(os.name == "nt", "the stalled launcher is a POSIX shell script")
class StalledLauncher(unittest.TestCase):
    def test_timeout_kills_the_launcher_and_its_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            pids = os.path.join(tmp, "pids")
            # A "gta6rt" that starts a child and then hangs, as a guest stuck on a fence or a hung GPU would.
            fake = os.path.join(tmp, "gta6rt")
            with open(fake, "w", encoding="utf-8") as f:
                f.write(f"#!/bin/sh\necho $$ > {pids}\nsleep 300 &\necho $! >> {pids}\nwait\n")
            os.chmod(fake, 0o755)
            shaders = os.path.join(tmp, "shaders")
            os.makedirs(os.path.join(shaders, "spirv"))
            open(os.path.join(shaders, "spirv", "index.json"), "w").close()
            started = time.monotonic()
            r = subprocess.run([sys.executable, "-I", RUNNER, "--gta6rt", fake, "--guest", fake, "--shaders", shaders,
                                "--out", os.path.join(tmp, "out"), "--timeout", "2"],
                               capture_output=True, text=True, timeout=60)
            took = time.monotonic() - started
            self.assertEqual(r.returncode, run_lanterns.TIMEOUT_EXIT, r.stderr)
            self.assertLess(took, 20)
            with open(pids, encoding="utf-8") as f:
                launched = [int(line) for line in f.read().split()]
            self.assertEqual(len(launched), 2)
            deadline = time.monotonic() + 5
            while any(alive(p) for p in launched) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse([p for p in launched if alive(p)])


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
        }
        for name, (code, out, err) in cases.items():
            with self.subTest(name):
                self.assertTrue(check(1, code, out, err))


if __name__ == "__main__":
    unittest.main()
