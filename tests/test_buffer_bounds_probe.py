"""CPU-only unit tests for tools/buffer_bounds_probe.py (SHADER-341)."""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PROBE = os.path.join(TOOLS, "buffer_bounds_probe.py")

sys.dont_write_bytecode = True
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import buffer_bounds_probe as bbp  # noqa: E402
from hw_oracle import asm as hw_asm, runner as hw_runner  # noqa: E402

# A minimal probe() result: one wave, one row that matches both published columns.
FAKE_REPORT = {"arch": "gfx1035", "device": "fake", "waves": {"32": [
    {"name": "mode0_stride16_off13", "got": 1, "oob": False,
     "gfx10_oob": False, "gfx11_oob": False}]}}


def fake_oracle():
    return unittest.mock.MagicMock(arch="gfx1035", device="fake")


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

    def test_directory_json_refused_before_hardware(self):
        # LEONIDA_AMD_CLANG points nowhere, so a run that reached the oracle would
        # exit 77 (unavailable) instead of 2: a 2 here proves --json is checked first.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-I", PROBE, "--json", tmp],
                                  env=env, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)
            self.assertEqual(len(proc.stderr.splitlines()), 1, proc.stderr)
            self.assertTrue(proc.stderr.startswith("ERROR: "), proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertNotIn("SKIP:", proc.stderr)

    def test_invalid_json_rejected_before_open_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_ancestor = os.path.join(tmp, "not_a_dir")
            with open(file_ancestor, "w", encoding="utf-8") as f:
                f.write("x")
            bad_paths = [
                "",
                "   ",
                "bad\x00.json",
                tmp,
                os.path.join(tmp, "sub") + os.sep,
                os.path.join(file_ancestor, "out.json"),
                os.path.join(file_ancestor, "nested", "out.json"),
            ]
            for bad in bad_paths:
                with self.subTest(json_path=bad):
                    err = io.StringIO()
                    with unittest.mock.patch.object(hw_runner, "open_oracle") as open_oracle, \
                         redirect_stderr(err):
                        rc = bbp.main(["--json", bad])
                    open_oracle.assert_not_called()
                    self.assertEqual(rc, 2)
                    self.assertEqual(len(err.getvalue().splitlines()), 1, err.getvalue())
                    self.assertTrue(err.getvalue().startswith("ERROR: "), err.getvalue())
                    self.assertNotIn("Traceback", err.getvalue())

    def test_missing_parent_created_and_json_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "missing", "parents", "report.json")
            self.assertFalse(os.path.exists(os.path.dirname(out_path)))
            out, err = io.StringIO(), io.StringIO()
            with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=fake_oracle()), \
                 unittest.mock.patch.object(hw_asm, "family", return_value="gfx10"), \
                 unittest.mock.patch.object(bbp, "probe", return_value=FAKE_REPORT), \
                 redirect_stdout(out), redirect_stderr(err):
                rc = bbp.main(["--json", out_path])
            self.assertEqual(rc, 0, err.getvalue() or out.getvalue())
            self.assertEqual(err.getvalue(), "")
            self.assertTrue(os.path.isfile(out_path))
            with open(out_path, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(json.loads(raw.decode("utf-8")), FAKE_REPORT)

    def test_write_json_report_file_parent_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_ancestor = os.path.join(tmp, "not_a_dir")
            with open(file_ancestor, "w", encoding="utf-8") as f:
                f.write("x")
            out_path = os.path.join(file_ancestor, "out.json")
            err = bbp.write_json_report(out_path, {"ok": True})
            self.assertIsNotNone(err)
            self.assertIn("cannot create directory for --json", err)
            self.assertIn(out_path, err)

    def test_write_failure_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.json")
            out, err = io.StringIO(), io.StringIO()
            with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=fake_oracle()), \
                 unittest.mock.patch.object(hw_asm, "family", return_value="gfx10"), \
                 unittest.mock.patch.object(bbp, "probe", return_value=FAKE_REPORT), \
                 unittest.mock.patch("builtins.open", side_effect=OSError(28, "disk full")), \
                 redirect_stdout(out), redirect_stderr(err):
                rc = bbp.main(["--json", out_path])
            self.assertEqual(rc, 2, out.getvalue())
            self.assertEqual(len(err.getvalue().splitlines()), 1, err.getvalue())
            self.assertTrue(err.getvalue().startswith("ERROR: "), err.getvalue())
            self.assertIn("disk full", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
