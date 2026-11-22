"""v_mul_f64 and v_fma_f64 (runtime/test/shaders/cs_w32_f64_mul_fma.asm, docs/design.md "v_mul_f64, v_fma_f64").

One rounding of the exact x * y (+ z) to binary64, to nearest even; f64 denormals kept; VOP3 omod ignored and clamp
applied, as measured natively on RDNA2 gfx1036 with f64 denormals kept. Run through tests/spirv_interp.py, whose
binary64 multiply and GLSL Fma are exact-then-rounded. Before, both raised Unsupported.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import spirv_interp  # noqa: E402
import support  # noqa: E402
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def run(ins, *srcs, mode=None):
    """Translate `ins` with v[2:3], v[4:5], v[6:7] = srcs (64-bit patterns); return the 64-bit result in v[10:11]."""
    lines = []
    for reg, s in zip((2, 4, 6), srcs):
        lines += [f"v_mov_b32 v{reg}, {s & M32:#x}", f"v_mov_b32 v{reg + 1}, {s >> 32:#x}"]
    if mode is not None:
        lines.append(f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}")
    lines += [ins, "exp mrt0 v10, v11, v10, v11 done vm", "s_endpgm"]
    r = spirv_interp.run(support.translate_lines("ps_f64_mul_fma", lines))
    return (r.vgpr[10] & M32) | (r.vgpr[11] & M32) << 32


X = bits(1 + 2.0 ** -30)
Z = bits(-(1 + 2.0 ** -29))
MUL = "v_mul_f64 v[10:11], v[2:3], v[4:5]"
FMA = "v_fma_f64 v[10:11], v[2:3], v[4:5], v[6:7]"


class MulFmaF64(unittest.TestCase):
    # (instruction, sources, hand-derived result, gfx1036 measurement): the fixture's four words and more.
    CASES = [
        (MUL, (bits(1.5), bits(-2.0)), 0xC008000000000000),                 # -3.0
        (FMA, (X, X, Z), 0x3C30000000000000),                              # fused: 2^-60
        (MUL, (bits(1 + 2.0 ** -52), bits(1 + 2.0 ** -52)), 0x3FF0000000000002),   # 1 + 2^-51
        (MUL + " mul:2", (bits(1.5), bits(1.5)), 0x4002000000000000),      # omod ignored: 2.25
        (MUL, (X, X), 0x3FF0000000800000),                                 # 1 + 2^-29, the rounded product
        (MUL, (0x0010000000000000, bits(0.5)), 0x0008000000000000),        # 2^-1023, subnormal kept
        (FMA, (0x0010000000000000, bits(0.5), 0x0010000000000000), 0x0018000000000000),   # 1.5 * 2^-1022
        (FMA + " clamp mul:2", (bits(1.5), bits(1.5), bits(-1.0)), 0x3FF0000000000000),   # 1.25 clamped: 1.0
        (FMA + " div:2", (bits(1.5), bits(1.5), bits(-1.0)), 0x3FF4000000000000),  # omod ignored: 1.25
    ]

    def test_words(self):
        for ins, srcs, want in self.CASES:
            with self.subTest(ins=ins, srcs=[hex(s) for s in srcs]):
                self.assertEqual(f"{run(ins, *srcs):#018x}", f"{want:#018x}")

    def test_fixture_words_are_the_first_four_cases(self):
        want = []
        for _, _, w in self.CASES[:4]:
            want += [w & M32, w >> 32]
        self.assertEqual(want, [0, 0xC0080000, 0, 0x3C300000, 2, 0x3FF00000, 0, 0x40020000])
        sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
        from compute_fixture import compute_header
        header = compute_header(os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_w32_f64_mul_fma.asm"))
        self.assertIn("OpExtInst %double %glsl Fma", support.translate_fixture("cs_w32_f64_mul_fma", fssm=bytes(header)))

    def test_reference(self):
        # Plain-Python reference for the hand-derived values: exact rationals rounded once by float().
        from fractions import Fraction as F
        x = F(1) + F(1, 2 ** 30)
        self.assertEqual(float(x * x + F(-(1 + 2.0 ** -29))), 2.0 ** -60)
        self.assertEqual(float(F(1.5) * F(1.5)), 2.25)
        self.assertEqual(bits(float((F(1) + F(1, 2 ** 52)) ** 2)), 0x3FF0000000000002)

    def test_toward_zero_mode_is_refused(self):
        for ins, srcs in ((MUL, (bits(1.5), bits(1.5))), (FMA, (X, X, Z))):
            with self.subTest(ins=ins):
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "MODE.FP_ROUND"):
                    run(ins, *srcs, mode=3)


if __name__ == "__main__":
    unittest.main()
