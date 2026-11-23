"""The assembly goes to spirv-as and the binary to spirv-val over pipes: no .spvasm for a shader that passes.

    python -I tests/test_assemble_stdin.py

--no-asm used to write each module's .spvasm (~35 GB for the corpus), reopen it for spirv-as and delete it. Standard
library only; no game files.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

MINIMAL_CS = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
BROKEN = "OpCapability Shader\nOpNotAnOpcode %x\n"


def run_main(src, out, text):
    argv = ["rdna2spirv.py", src, "--out", out, "--no-asm"]
    with patch.object(sys, "argv", argv), patch.object(rdna2spirv, "translate", lambda p: text), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        status = rdna2spirv.main()
    return status, err.getvalue()


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class AssemblyOverPipes(unittest.TestCase):
    def test_the_text_needs_no_file_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            sv = os.path.join(tmp, "m.spv")
            result = rdna2spirv.assemble_spirv(os.path.join(tmp, "never_written.spvasm"), sv, asm_text=MINIMAL_CS)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(os.listdir(tmp), ["m.spv"])
            with open(sv, "rb") as f:
                self.assertEqual(f.read(4), b"\x03\x02\x23\x07")

    def test_a_rejected_module_publishes_nothing_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            sv = os.path.join(tmp, "m.spv")
            result = rdna2spirv.assemble_spirv(os.path.join(tmp, "x.spvasm"), sv, asm_text=BROKEN)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("OpNotAnOpcode", result.stderr)
            self.assertEqual(os.listdir(tmp), [])

    def test_no_asm_leaves_only_the_module_for_a_shader_that_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(src)
            open(os.path.join(src, "cs_ok.s"), "w").close()
            os.makedirs(out)
            with open(os.path.join(out, "cs_ok.spvasm"), "w") as f:  # left by an earlier rejected run
                f.write(BROKEN)
            status, _ = run_main(src, out, MINIMAL_CS)
            self.assertIn(status, (0, None))
            self.assertIn("cs_ok.spv", os.listdir(out))
            self.assertNotIn("cs_ok.spvasm", os.listdir(out))

    def test_no_asm_keeps_the_text_of_a_rejected_shader_and_names_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(src)
            open(os.path.join(src, "cs_bad.s"), "w").close()
            status, err = run_main(src, out, BROKEN)
            self.assertIn(status, (0, None))
            self.assertIn("cs_bad", err)
            self.assertNotIn("cs_bad.spv", os.listdir(out))
            with open(os.path.join(out, "cs_bad.spvasm"), encoding="utf-8") as f:
                self.assertEqual(f.read(), BROKEN)


if __name__ == "__main__":
    unittest.main()
