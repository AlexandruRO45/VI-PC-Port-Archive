"""`--help` of the gfx1036 v_cmp probe prints usage instead of starting the probe.

The probe shells out to `spirv-as` and `runtime/build/vk_compute_test.exe` and needs a gfx1036. Without an argument
parser it ignored `--help`, started the probe and died on the first translation with
`FileNotFoundError: 'spirv-as'` wherever the SPIR-V tools are not on PATH. The test runs it with an empty PATH, so
reaching the probe at all is the failure.

No GPU and no game files; the probe itself is untouched.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "probe_gfx1036_vcmp16.py")


class ProbeHelpTests(unittest.TestCase):
    def test_help_prints_usage_without_spirv_tools(self):
        with tempfile.TemporaryDirectory() as empty_path:
            env = dict(os.environ, PATH=empty_path)
            proc = subprocess.run([sys.executable, "-I", TOOL, "--help"], capture_output=True, text=True,
                                  cwd=ROOT, env=env, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("usage:", proc.stdout)
        self.assertIn("--help", proc.stdout)
        self.assertNotIn("PROBING GFX1036", proc.stdout)


if __name__ == "__main__":
    unittest.main()
