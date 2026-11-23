"""runtime/test/shaders/probe_signed_zero.py reports a missing harness or SPIRV-Tools as one line and exit 2.

No GPU, MSVC or game files needed.

Run:
    python -I tests/test_probe_signed_zero.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "runtime", "test", "shaders", "probe_signed_zero.py")


def run_probe(harness, path, **extra_env):
    env = {k: v for k, v in os.environ.items() if k not in ("PATH", "LEONIDA_SPIRV_TOOLS")}
    env["PATH"] = path
    env.update(extra_env)
    return subprocess.run([sys.executable, "-I", PROBE, harness], env=env, capture_output=True, text=True)


class ProbeSignedZeroMissingTools(unittest.TestCase):
    def check_one_line(self, r):
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)

    def test_missing_harness(self):
        with tempfile.TemporaryDirectory() as empty:
            r = run_probe(os.path.join(empty, "no_such_harness.exe"), empty)
        self.check_one_line(r)
        self.assertIn("harness", r.stderr)

    def test_missing_spirv_tools(self):
        with tempfile.TemporaryDirectory() as empty:
            harness = os.path.join(empty, "harness.exe")
            open(harness, "wb").close()
            r = run_probe(harness, empty)
            self.check_one_line(r)
            self.assertIn("spirv-as", r.stderr)
            # A LEONIDA_SPIRV_TOOLS folder without the tools is the same single line.
            self.check_one_line(run_probe(harness, empty, LEONIDA_SPIRV_TOOLS=empty))


if __name__ == "__main__":
    unittest.main()
