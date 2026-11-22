"""--native-rounding: v_floor_f32 / v_ceil_f32 / v_fract_f32 as plain GLSL Floor / Ceil / Fract.

By default each one goes through Gen.round_f32_flushed, which gives RDNA2's results for denormal operands, Fract's
round-up to 1.0, NaNs and infinities. With the flag the module holds only the GLSL function: no bit test of the
operand, no 0x3f7fffff clamp, no inf/NaN select. On ordinary operands both translations give the same words, run
through the interpreter, and the command-line option reaches the generator. (The edge cases are the driver's:
GPU-checked with LEONIDA_NATIVE_ROUNDING=1, docs/testing.md.)

    python -m unittest tests/semantic/test_native_rounding.py -v
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from support import STUB_DXBC, mrt0_words, rdna2spirv, translate_lines

OPS = (("v_fract_f32", "Fract"), ("v_floor_f32", "Floor"), ("v_ceil_f32", "Ceil"))
VALUES = (1.25, -1.25, 3.75, -0.5)


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def shader(op):
    lines = []
    for k, v in enumerate(VALUES):
        lines += [f"v_mov_b32_e32 v{1 + k}, 0x{f2u(v):x}", f"{op}_e32 v{1 + k}, v{1 + k}"]
    return lines + ["exp mrt0 v1, v2, v3, v4 done vm", "s_endpgm"]


def translate(op, native):
    with patch.object(rdna2spirv, "NATIVE_ROUNDING", native):
        return translate_lines(f"ps_{op}_{'native' if native else 'exact'}", shader(op))


class NativeRounding(unittest.TestCase):
    def test_only_the_glsl_function_with_the_flag(self):
        for op, fn in OPS:
            with self.subTest(op=op):
                native, exact = translate(op, True), translate(op, False)
                self.assertEqual(native.count(f"%glsl {fn} "), len(VALUES))
                for marker in ("%c7f800000", "%c3f7fffff", "%cffc00000"):
                    self.assertNotIn(f"{marker} ", native.split("%main = OpFunction")[1])
                self.assertIn("%c7f800000", exact)          # the default keeps its NaN / infinity tests

    def test_same_words_on_ordinary_operands(self):
        for op, _ in OPS:
            with self.subTest(op=op):
                self.assertEqual(mrt0_words(translate(op, True)), mrt0_words(translate(op, False)))

    def test_the_command_line_option_reaches_the_translator(self):
        """rdna2spirv.py --native-rounding as a process (the other tests set the module global directly)."""
        out = {}
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            with open(os.path.join(tmp, "disasm", "ps_cli.s"), "w", encoding="utf-8", newline="\n") as f:
                f.write("; ps_cli  hand-written test program\n" + "".join(f"\t{ln}\n" for ln in shader("v_fract_f32")))
            with open(os.path.join(tmp, "ps_cli.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            for name, flags in (("native", ["--native-rounding"]), ("exact", [])):
                spirv = os.path.join(tmp, name)
                subprocess.run([sys.executable, "-I", rdna2spirv.__file__, os.path.join(tmp, "disasm"), "--out", spirv,
                                *flags], check=True, capture_output=True)
                with open(os.path.join(spirv, "ps_cli.spvasm"), encoding="utf-8") as f:
                    out[name] = f.read()
        self.assertNotIn("%c3f7fffff", out["native"])
        self.assertIn("%c3f7fffff", out["exact"])


if __name__ == "__main__":
    unittest.main()
