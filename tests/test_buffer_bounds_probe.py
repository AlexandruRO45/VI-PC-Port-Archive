"""CPU-only unit tests for tools/buffer_bounds_probe.py (SHADER-341)."""
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(REPO, "tools", "buffer_bounds_probe.py")


class BufferBoundsProbeTests(unittest.TestCase):
    def test_self_test(self):
        proc = subprocess.run([sys.executable, "-I", PROBE, "--self-test"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("self-test OK", proc.stdout)

    def test_non_gfx10_or_missing_oracle_exits_77(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        proc = subprocess.run([sys.executable, "-I", PROBE], env=env,
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 77, proc.stderr or proc.stdout)

    def test_bad_waves(self):
        # int() also takes "3_2", "+32" and non-ASCII digits ("\u0663\u0662") as 32, so those used to reach the
        # oracle as a wave32 run instead of being rejected.
        for waves in ("16", "3_2", "+32", "\u0663\u0662", "abc", ""):
            with self.subTest(waves=waves):
                proc = subprocess.run([sys.executable, "-I", PROBE, "--waves", waves],
                                      capture_output=True, text=True, check=False)
                self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)
                self.assertIn("--waves", proc.stderr)


if __name__ == "__main__":
    unittest.main()
