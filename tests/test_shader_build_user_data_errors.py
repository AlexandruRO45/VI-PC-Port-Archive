"""runtime/test/shaders/build.py reports a bad LEONIDA_USER_DATA_SHIFT or fssm-data word as one error line.

The shift is read before llvm-mc runs, so the subprocess cases need no LLVM. The fssm-data word is parsed by
asm_fssm_fields, which is called directly on a tiny hand-written header.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders" / "build.py"


class BuildUserDataErrorTest(unittest.TestCase):
    def run_build(self, shift):
        with tempfile.TemporaryDirectory(prefix="leonida-build-") as out:
            env = {**os.environ, "LEONIDA_USER_DATA_SHIFT": shift}
            return subprocess.run([sys.executable, "-I", str(BUILD), out, "cs_barrier_loop"],
                                  capture_output=True, text=True, env=env, timeout=120)

    def check_one_line(self, r, needle):
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)
        self.assertIn(needle, r.stderr)

    def test_malformed_shift(self):
        self.check_one_line(self.run_build("abc"), "LEONIDA_USER_DATA_SHIFT must be a whole number")

    def test_shift_out_of_range(self):
        self.check_one_line(self.run_build("28"), "outside 0..27")

    def test_malformed_fssm_data_word(self):
        spec = importlib.util.spec_from_file_location("shader_build_under_test", BUILD)
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        with tempfile.TemporaryDirectory(prefix="leonida-build-") as d:
            asm = Path(d) / "ps_x.asm"
            asm.write_text("; fssm-at: 0x10\n; fssm-data: 0x1 zzz\ns_endpgm\n", encoding="utf-8", newline="\n")
            with self.assertRaises(SystemExit) as cm:
                build.asm_fssm_fields(str(asm))
        message = str(cm.exception.code)
        self.assertIn("ps_x.asm", message)
        self.assertIn("fssm-data", message)
        self.assertEqual(len(message.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
