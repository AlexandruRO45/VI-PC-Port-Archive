"""v_madak_f32 and v_madmk_f32 flush f32 denormals like v_mad_f32 (SHADER-118).

They are the VOP2 form of v_mad_f32 with a 32-bit literal: LLVM only forms them by shrinking a V_MAD_F32
(SIShrinkInstructions), and forms V_MAD_F32 only with f32 denormals flushed (SITargetLowering::isFMADLegal). v_mad_f32
has no f32 denormal support, measured natively on gfx1010 (docs/design.md, "Denormal flush in v_mad_f32, ..."): a
subnormal operand, the rounded product or the sum reads as a zero of its sign. The translator applied that to
v_mad_f32/v_mac_f32 but not to madak/madmk, so `v_madak_f32 v0, v1, v2, 0` with a denormal v1 gave a non-zero result.

Hand-written assembly only (no game files), run on the CPU through tests/semantic/spirv_mini_interp.py.

    python -m unittest discover -s tests/semantic -p test_madak_madmk_denorm.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for path in (TOOLS_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

ONE, TWO_31, ZERO, NEG_ZERO = 0x3F800000, 0x4F000000, 0x00000000, 0x80000000
DENORM, NEG_DENORM = 0x00000001, 0x80000001
FLT_MIN_X1_5, NEG_FLT_MIN = 0x00C00000, 0x80800000   # 1.5 * 2^-126, -2^-126
TWO_M70 = 0x1C800000                                 # 2^-70: its square, 2^-140, is subnormal
INF, QNAN = 0x7F800000, 0x7FC00000


class MadakMadmkDenormFlush(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(cls.tmp.name, "disasm"))
        cls.src = os.path.join(cls.tmp.name, "disasm", "ps_madak_denorm.s")
        with open(os.path.join(cls.tmp.name, "ps_madak_denorm.dxbc"), "wb") as f:
            f.write(STUB_DXBC)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_op(self, line, v1, v2):
        lines = [f"v_mov_b32 v1, 0x{v1:08x}", f"v_mov_b32 v2, 0x{v2:08x}", line,
                 "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
        asm = ("; ps_madak_denorm\n" + "".join(f"\t{line}\n" for line in lines)
               + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n")
        with open(self.src, "w", encoding="utf-8", newline="\n") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(self.src)
        return si.run_fragment(spv)["%out_mrt0"][0]

    def check(self, cases):
        for name, line, v1, v2, want in cases:
            with self.subTest(name):
                got = self.run_op(line, v1, v2)
                self.assertEqual(got, want, f"{line} with v1=0x{v1:08x} v2=0x{v2:08x}: got 0x{got:08x}, "
                                            f"want 0x{want:08x}")

    def test_madak(self):  # D = S0 * S1 + K, printed `vdst, src0, vsrc1, K`
        self.check([
            ("denormal operand", "v_madak_f32 v0, v1, v2, 0x0", DENORM, TWO_31, ZERO),
            ("negative denormal operand keeps the product's sign until the add",
             "v_madak_f32 v0, v1, v2, 0x80000000", NEG_DENORM, TWO_31, NEG_ZERO),
            ("denormal product", "v_madak_f32 v0, v1, v2, 0x0", TWO_M70, TWO_M70, ZERO),
            ("denormal sum", f"v_madak_f32 v0, v1, v2, 0x{NEG_FLT_MIN:08x}", FLT_MIN_X1_5, ONE, ZERO),
            ("denormal literal", "v_madak_f32 v0, v1, v2, 0x1", ONE, ZERO, ZERO),
            ("normal values are untouched", "v_madak_f32 v0, v1, v2, 0x40a00000", 0x40000000, 0x40400000, 0x41300000),
            ("infinity passes through", "v_madak_f32 v0, v1, v2, 0x0", INF, ONE, INF),
            ("NaN passes through", "v_madak_f32 v0, v1, v2, 0x0", QNAN, ONE, QNAN),
        ])

    def test_madmk(self):  # D = S0 * K + S1, printed `vdst, src0, K, vsrc1`
        self.check([
            ("denormal operand", "v_madmk_f32 v0, v1, 0x4f000000, v2", DENORM, ZERO, ZERO),
            ("denormal literal", "v_madmk_f32 v0, v1, 0x1, v2", TWO_31, ZERO, ZERO),
            ("denormal product", "v_madmk_f32 v0, v1, 0x1c800000, v2", TWO_M70, ZERO, ZERO),
            ("denormal sum", f"v_madmk_f32 v0, v1, 0x3f800000, v2", FLT_MIN_X1_5, NEG_FLT_MIN, ZERO),
            ("denormal addend", "v_madmk_f32 v0, v1, 0x3f800000, v2", ZERO, DENORM, ZERO),
            ("normal values are untouched", "v_madmk_f32 v0, v1, 0x40a00000, v2", 0x40000000, 0x40400000, 0x41500000),
        ])


if __name__ == "__main__":
    unittest.main()
