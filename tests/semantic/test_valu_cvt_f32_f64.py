"""v_cvt_f32_f64 (RDNA2 VOP1 / VOP3): D.f32 = (float) S0.f64, rounded to nearest even.

Every case is translated by tools/rdna2spirv.py and executed with tests/spirv_interp.py (the interpreter that
evaluates binary64). The expected words are written down by hand next to each case and rechecked against a
plain-Python reference (struct packs a double to binary32 with round to nearest even); seeded random operands compare
the translated result with it. Covers ties, overflow to infinity, signed zeros, infinities and NaN, VOP3 abs/neg and
mul:2 / mul:4 / div:2 / clamp, and SGPR-pair, inline-constant and literal sources (a 32-bit literal supplies bits 63:32
of the binary64 operand).

`v_cvt_f64_f32` is not translated at all; test_valu_f64_unsupported.py pins that refusal.
Standard library only: no GPU, no LLVM, no game data.
"""
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
from f64_ref import POS_INF, NEG_INF, POS_ZERO, NEG_ZERO, QNAN, bits, run, value  # noqa: E402

F32_NAN = 0x7FC00000


def f32_bits(x):
    """binary32 word of a Python float: round to nearest even, overflow to infinity."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def ref(b):
    x = value(b)
    if x != x:
        return F32_NAN
    return f32_bits(x)


def same32(a, b):
    return a == b or ((a & 0x7F800000) == 0x7F800000 and (a & 0x7FFFFF) and (b & 0x7F800000) == 0x7F800000 and (b & 0x7FFFFF))


FLT_MAX = 3.4028234663852886e38
HALF_ULP_ABOVE_FLT_MAX = FLT_MAX + 2.0 ** 103          # 2^128 - 2^103: the tie between FLT_MAX and 2^128

# (source word, D), written by hand
CASES = [
    (bits(1.0), 0x3F800000),
    (bits(-2.5), 0xC0200000),
    (bits(0.1), 0x3DCCCCCD),                         # 0.1 in binary32
    (bits(1.0 + 2.0 ** -24), 0x3F800000),            # exactly half an f32 ulp above 1: ties to even, stays 1
    (bits(1.0 + 2.0 ** -24 + 2.0 ** -52), 0x3F800001),   # a little more: rounds up
    (bits(-(1.0 + 2.0 ** -24 + 2.0 ** -52)), 0xBF800001),
    (bits(1.0 + 3 * 2.0 ** -24), 0x3F800002),        # 1.5 ulp above 1 ties to even: 2 ulps
    (bits(FLT_MAX), 0x7F7FFFFF),
    (bits(HALF_ULP_ABOVE_FLT_MAX), 0x7F800000),      # tie between FLT_MAX (odd) and 2^128 rounds to even: infinity
    (bits(3.4028235e38), 0x7F7FFFFF),                # just below that tie: FLT_MAX
    (bits(1e300), 0x7F800000),                       # overflow
    (bits(-1e300), 0xFF800000),
    (POS_ZERO, 0x00000000),
    (NEG_ZERO, 0x80000000),
    (1, 0x00000000),                                 # the smallest binary64 subnormal underflows to +0
    (0x8000000000000001, 0x80000000),                # and to -0
    (POS_INF, 0x7F800000),
    (NEG_INF, 0xFF800000),
    (QNAN, F32_NAN),
]

# (source word, source operand text, modifier text, D), by hand; the VOP3 form is used
MODS = [
    (bits(0.75), "-v[0:1]", "", 0xBF400000),
    (bits(-0.75), "|v[0:1]|", "", 0x3F400000),
    (bits(0.75), "-|v[0:1]|", "", 0xBF400000),
    (bits(0.75), "v[0:1]", " mul:2", 0x3FC00000),             # 1.5
    (bits(0.75), "v[0:1]", " mul:4", 0x40400000),             # 3.0
    (bits(0.75), "v[0:1]", " div:2", 0x3EC00000),             # 0.375
    (bits(0.75), "v[0:1]", " clamp", 0x3F400000),             # inside [0, 1]: unchanged
    (bits(2.5), "v[0:1]", " clamp", 0x3F800000),              # -> 1
    (bits(-2.5), "v[0:1]", " clamp", 0x00000000),             # -> 0
    (bits(1.5), "v[0:1]", " div:2 clamp", 0x3F400000),        # 0.75 (omod first)
    (bits(0.75), "v[0:1]", " mul:2 clamp", 0x3F800000),       # 1.5 -> 1
]


class CvtF32F64(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for x, want in CASES:
            with self.subTest(x=hex(x)):
                self.assertTrue(same32(ref(x), want), hex(ref(x)))

    def test_translated_cases(self):
        for x, want in CASES:
            with self.subTest(x=hex(x)):
                got = run("v_cvt_f32_f64 v4, v[0:1]", x, dst="v4")
                self.assertTrue(same32(got, want), hex(got))

    def test_modifiers(self):
        for x, operand, mods, want in MODS:
            with self.subTest(x=hex(x), operand=operand, mods=mods):
                self.assertEqual(run(f"v_cvt_f32_f64_e64 v4, {operand}{mods}", x, dst="v4"), want)

    def test_sgpr_pair_inline_constant_and_literal_sources(self):
        for x in (bits(0.1), bits(-2.5), bits(1.5), bits(2.0), bits(-0.5)):
            lo, hi = x & 0xFFFFFFFF, x >> 32
            forms = [("sgpr pair", "v_cvt_f32_f64 v4, s[2:3]", [f"s_mov_b32 s2, {lo:#x}", f"s_mov_b32 s3, {hi:#x}"]),
                     ("sgpr pair, e64", "v_cvt_f32_f64_e64 v4, s[2:3]",
                      [f"s_mov_b32 s2, {lo:#x}", f"s_mov_b32 s3, {hi:#x}"])]
            if lo == 0:
                forms.append(("literal = bits 63:32", f"v_cvt_f32_f64 v4, {hi:#x}", []))
            if value(x) in (0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0):
                forms.append(("inline constant", f"v_cvt_f32_f64 v4, {value(x)!r}", []))
            for form, line, setup in forms:
                with self.subTest(x=hex(x), form=form):
                    self.assertEqual(run(line, setup=setup, dst="v4"), ref(x))

    def test_literal_is_the_high_dword(self):
        self.assertEqual(run("v_cvt_f32_f64 v4, 0x3ff80000", dst="v4"), 0x3FC00000)       # 1.5
        self.assertEqual(run("v_cvt_f32_f64 v4, 0xc0040000", dst="v4"), 0xC0200000)       # -2.5

    def test_random_operands(self):
        rng = random.Random(0xCF3264)
        for _ in range(80):
            kind = rng.random()
            if kind < 0.5:
                x = bits(rng.uniform(-1e6, 1e6))
            elif kind < 0.7:
                x = bits(rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-60, 45))
            else:
                x = rng.getrandbits(64)
            with self.subTest(x=hex(x)):
                got = run("v_cvt_f32_f64 v4, v[0:1]", x, dst="v4")
                self.assertTrue(same32(got, ref(x)), (hex(got), hex(ref(x))))


if __name__ == "__main__":
    unittest.main()
