"""v_sqrt_f64 (RDNA2 VOP1 / VOP3): D.f64 = sqrt(S0.f64), binary64.

Every case is translated by tools/rdna2spirv.py and executed with tests/spirv_interp.py (the interpreter that
evaluates binary64). The expected words are written down by hand next to each case and rechecked against the
plain-Python reference below; seeded random operands compare the translated result with the reference.
Covers the special values, VOP3 abs/neg and clamp, and SGPR-pair, inline-constant and literal sources (a 32-bit
literal supplies bits 63:32 of a binary64 operand, bits 31:0 are zero). The omod (mul/div) rows are left out: whether
it applies to the f64 ops is SHADER-721 (#4371).
Standard library only: no GPU, no LLVM, no game data.
"""
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
from f64_ref import (DBL_MAX, MIN_NORMAL, MIN_SUBNORMAL, NEG_INF, NEG_ZERO, POS_INF, POS_ZERO, QNAN, bits,  # noqa: E402
                     is_nan, run, same, ulp_distance, value)

OP = "v_sqrt_f64"


def ref(b):
    """Reference for one binary64 word."""
    x = value(b)
    if is_nan(b) or x < 0:
        return QNAN
    return bits(math.sqrt(x))              # correctly rounded; sqrt(-0) = -0


# (source, D), written by hand
CASES = [
    (bits(4.0), bits(2.0)),
    (bits(2.25), bits(1.5)),
    (bits(0.25), bits(0.5)),
    (bits(1.0), bits(1.0)),
    (bits(2.0), 0x3FF6A09E667F3BCD),               # 1.4142135623730951
    (bits(3.0), 0x3FFBB67AE8584CAA),               # 1.7320508075688772
    (bits(4096.0 * 4096.0), bits(4096.0)),
    (MIN_SUBNORMAL, 0x1E60000000000000),           # sqrt(2^-1074) = 2^-537
    (MIN_NORMAL, 0x2000000000000000),              # sqrt(2^-1022) = 2^-511
    (DBL_MAX, 0x5FEFFFFFFFFFFFFF),
    (POS_ZERO, POS_ZERO),
    (NEG_ZERO, NEG_ZERO),                          # sqrt(-0) = -0
    (POS_INF, POS_INF),
    (bits(-4.0), QNAN),                            # negative operand
    (NEG_INF, QNAN),
    (QNAN, QNAN),
]

# (source, source operand text, modifier text, D), by hand; the VOP3 form is used
MODS = [
    (bits(-4.0), "v[0:1]", "", QNAN),
    (bits(-4.0), "-v[0:1]", "", bits(2.0)),
    (bits(-4.0), "|v[0:1]|", "", bits(2.0)),
    (bits(-4.0), "-|v[0:1]|", "", QNAN),
    (bits(4.0), "-v[0:1]", "", QNAN),
    (bits(4.0), "v[0:1]", " clamp", bits(1.0)),                # 2 -> 1
    (bits(0.25), "v[0:1]", " clamp", bits(0.5)),
]


def close(got, want):
    return same(got, want) if True else (same(got, want) or ulp_distance(got, want) <= 0)


class SqrtF64(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for x, want in CASES:
            with self.subTest(x=hex(x)):
                self.assertTrue(close(ref(x), want), hex(ref(x)))

    def test_translated_cases(self):
        for x, want in CASES:
            with self.subTest(x=hex(x)):
                got = run(f"{OP} v[4:5], v[0:1]", x)
                self.assertTrue(same(got, want), hex(got))

    def test_modifiers(self):
        for x, operand, mods, want in MODS:
            with self.subTest(x=hex(x), operand=operand, mods=mods):
                got = run(f"{OP}_e64 v[4:5], {operand}{mods}", x)
                self.assertTrue(same(got, want), hex(got))

    def test_sgpr_pair_inline_constant_and_literal_sources(self):
        for x in (bits(0.25), bits(4.0), bits(1.5), bits(-2.0), bits(1234.5)):
            lo, hi = x & 0xFFFFFFFF, x >> 32
            forms = [("sgpr pair", "s[2:3]", [f"s_mov_b32 s2, {lo:#x}", f"s_mov_b32 s3, {hi:#x}"]),
                     ("sgpr pair, e64", "s[2:3]", [f"s_mov_b32 s2, {lo:#x}", f"s_mov_b32 s3, {hi:#x}"])]
            if lo == 0:
                forms.append(("literal = bits 63:32", f"{hi:#x}", []))
            if value(x) in (0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0):
                forms.append(("inline constant", repr(value(x)), []))
            for form, text, setup in forms:
                with self.subTest(x=hex(x), form=form):
                    opname = OP + ("_e64" if "e64" in form else "")
                    self.assertTrue(close(run(f"{opname} v[4:5], {text}", setup=setup), ref(x)))

    def test_literal_is_the_high_dword(self):
        # 0x40100000 is the high word of 4.0 (not the value 2^-1074 * 0x40100000, which a low-word reading gives)
        self.assertEqual(run("v_sqrt_f64 v[4:5], 0x40100000"), bits(2.0))
        self.assertEqual(run("v_sqrt_f64_e64 v[4:5], 0x40100000"), bits(2.0))
        self.assertEqual(run("v_sqrt_f64 v[4:5], 0x3ff80000"), bits(math.sqrt(1.5)))

    def test_a_whole_binary64_word_written_by_hand_is_that_value(self):
        # wider than a dword: not a high dword (a low-word reading would build +0.0)
        self.assertEqual(run("v_sqrt_f64 v[4:5], 0x4010000000000000"), bits(2.0))   # 4.0

    def test_random_operands(self):
        rng = random.Random(0x5A7F64)
        pool = [POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, MIN_SUBNORMAL, MIN_NORMAL, DBL_MAX, bits(1.0), bits(-1.0)]
        for _ in range(60):
            kind = rng.random()
            if kind < 0.15:
                x = rng.choice(pool)
            elif kind < 0.6:
                x = bits(rng.uniform(-1000.0, 1000.0))
            else:
                x = rng.getrandbits(64)
            with self.subTest(x=hex(x)):
                self.assertTrue(close(run(f"{OP} v[4:5], v[0:1]", x), ref(x)), (hex(x), hex(ref(x))))


if __name__ == "__main__":
    unittest.main()
