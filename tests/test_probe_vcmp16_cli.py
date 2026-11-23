"""tools/probe_gfx1036_vcmp16.py answers its command line before it touches the GPU.

    python -I tests/test_probe_vcmp16_cli.py

The probe used to ignore its arguments: `--help` and an unknown flag both started the hardware run, and on
a machine without spirv-as `--help` ended in a `FileNotFoundError` traceback from
`subprocess.run(["spirv-as", ...])`. It now exits 0 with the usage, 2 with one error line, and names a
missing spirv-as before the first module is written. None of that needs a GPU, spirv-as or a built
vk_compute_test, which is why this test can run anywhere.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "probe_gfx1036_vcmp16.py")


def run(*args, env=None):
    return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, timeout=120,
                          env=env)


class ProbeCommandLineTest(unittest.TestCase):
    def test_help_prints_the_usage_and_starts_no_probe(self):
        r = run("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("usage:", r.stdout)
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("PROBING", r.stdout)

    def test_an_unknown_flag_is_refused(self):
        r = run("--no-such-flag")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("usage:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("PROBING", r.stdout)

    def test_missing_spirv_as_is_named_before_the_probe(self):
        r = run(env=dict(os.environ, PATH=""))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertEqual(r.stderr.strip(), "ERROR: spirv-as is not on PATH (SPIRV-Tools, or the Vulkan SDK's bin)")
        self.assertNotIn("PROBING", r.stdout)


if __name__ == "__main__":
    unittest.main()
