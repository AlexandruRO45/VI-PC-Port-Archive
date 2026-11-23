"""runtime/test/shaders/build.py: a build target with no .asm source is one error line, not a traceback.

    python -I tests/test_shader_build_missing_source.py

docs/testing.md's standalone v_mullit readback builds vs_ids and ps_mullit; ps_mullit's source is
tests/semantic/fixtures/ps_mullit.asm, which is not in the tree. build.py lists ps_mullit as a build target, so it did
not report it as unknown: it made the output folder and then died inside Path(...).read_text with a FileNotFoundError
traceback (and a raw ../../../ path). Standard library only; the check stops before llvm-mc, so no assembler,
SPIR-V tools, GPU or game data is needed.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "runtime" / "test" / "shaders" / "build.py"
MULLIT_SOURCE = "tests/semantic/fixtures/ps_mullit.asm"


def run_build(out, *names):
    return subprocess.run([sys.executable, "-I", str(BUILD), str(out), *names],
                          capture_output=True, text=True, encoding="utf-8")


class MissingSourceTest(unittest.TestCase):
    def test_the_documented_v_mullit_target_is_one_error_line(self):
        # The command docs/testing.md gives for the v_mullit readback, and the same target on its own.
        for names in (("vs_ids", "ps_mullit"), ("ps_mullit",)):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "out"
                result = run_build(out, *names)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("FileNotFoundError", result.stderr)
                lines = result.stderr.rstrip("\n").splitlines()
                self.assertEqual(len(lines), 1, result.stderr)
                self.assertTrue(lines[0].startswith("No test shader source for: "), lines[0])
                self.assertIn("ps_mullit", lines[0])
                self.assertIn(MULLIT_SOURCE, lines[0].replace(os.sep, "/"))
                self.assertFalse(out.exists(), "the output folder was made before the source was found missing")

    def test_a_target_whose_source_exists_is_still_accepted(self):
        # ms_tri is in the tree, so the source check passes; it stops at the missing DXBC container as before.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = run_build(out, "ms_tri")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("not built (need a DXBC container with signatures): ms_tri", result.stdout)
            self.assertEqual(result.stderr, "")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
