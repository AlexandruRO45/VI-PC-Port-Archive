"""CPU-only unit tests for tools/buffer_bounds_probe.py (SHADER-341)."""
import io
import os
import tempfile
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
PROBE = os.path.join(REPO, "tools", "buffer_bounds_probe.py")


class BufferBoundsProbeTests(unittest.TestCase):
    def test_json_directory_is_a_write_error(self):
        import buffer_bounds_probe as probe
        from hw_oracle import runner

        oracle = mock.Mock(arch="gfx1036")
        with tempfile.TemporaryDirectory() as report, redirect_stderr(io.StringIO()) as error:
            with mock.patch.object(sys, "argv", [PROBE, "--json", report]), \
                    mock.patch.object(runner, "open_oracle", return_value=oracle), \
                    mock.patch.object(probe, "probe", return_value={"waves": {}}):
                self.assertEqual(probe.main(), 2)
            self.assertIn(f"ERROR: cannot write {report}", error.getvalue())
        oracle.close.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
