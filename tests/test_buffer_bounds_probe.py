"""CPU-only unit tests for tools/buffer_bounds_probe.py (SHADER-341)."""
import io
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PROBE = os.path.join(TOOLS, "buffer_bounds_probe.py")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import buffer_bounds_probe as probe_mod  # noqa: E402
from hw_oracle import runner as hw_runner  # noqa: E402


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
        proc = subprocess.run([sys.executable, "-I", PROBE, "--waves", "16"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)

    def test_json_folder_is_refused_before_the_gpu_run(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, "-I", PROBE, "--json", folder],
                                  env=self.no_oracle_env(), capture_output=True, text=True,
                                  check=False)
            self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)
            self.assertIn("ERROR: --json", proc.stderr)
            self.assertIn("is a directory", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

    def test_json_in_a_missing_folder_is_refused_before_the_gpu_run(self):
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "missing", "report.json")
            proc = subprocess.run([sys.executable, "-I", PROBE, "--json", target],
                                  env=self.no_oracle_env(), capture_output=True, text=True,
                                  check=False)
            self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)
            self.assertIn("ERROR: --json", proc.stderr)
            self.assertIn("does not exist", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(target))

    def no_oracle_env(self):
        """An env with no hardware oracle, so a missing guard shows up as the 77 skip rather than a sweep."""
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        return env

    def test_short_or_invalid_oracle_results_exit_2_and_close_oracle(self):
        n = len(probe_mod.CASES)
        valid = [[0x11111111, 0, 0, 0] for _ in range(n)]
        bad_payloads = [
            [],
            valid[:-1],
            [None] + valid[1:],
            [[]] + valid[1:],
            ["REJECTED: disallowed instruction"] + valid[1:],
        ]
        for bad in bad_payloads:
            with self.subTest(bad=bad[0] if bad else "empty"):
                fake = unittest.mock.MagicMock()
                fake.arch = "gfx1030"
                fake.device = "Fake GPU"
                fake.run.return_value = bad
                err = io.StringIO()
                with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=fake), \
                     unittest.mock.patch.object(sys, "argv", ["buffer_bounds_probe.py", "--waves", "32"]), \
                     redirect_stderr(err):
                    rc = probe_mod.main()
                self.assertEqual(rc, 2)
                self.assertIn("ERROR:", err.getvalue())
                fake.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
