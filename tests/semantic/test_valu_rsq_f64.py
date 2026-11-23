"""v_rsq_f64 (RDNA2 VOP1 / VOP3): D.f64 = 1 / sqrt(S0.f64), binary64 (the hardware result is within 1 ulp; the tests allow that).

Every case is translated by tools/rdna2spirv.py and executed with tests/spirv_interp.py (the interpreter that
evaluates binary64). The expected words are written down by hand next to each case and rechecked against the
plain-Python reference below; seeded random operands compare the translated result with the reference within 1 ulp.
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

OP = "v_rsq_f64"


def ref(b):
    """Reference for one binary64 word."""
    x = value(b)
    if is_nan(b) or x < 0:
        return QNAN
    if x == 0:
        return POS_INF if b == POS_ZERO else NEG_INF
    return bits(1.0 / math.sqrt(x))        # two roundings: within an ulp of the correctly rounded value


# (source, D), written by hand
CASES = [
    (bits(4.0), bits(0.5)),
    (bits(0.25), bits(2.0)),
    (bits(1.0), bits(1.0)),
    (bits(16.0), bits(0.25)),
    (bits(2.0 ** -100), bits(2.0 ** 50)),
    (MIN_NORMAL, 0x5FE0000000000000),              # 2^511
    (POS_ZERO, POS_INF),
    (NEG_ZERO, NEG_INF),
    (POS_INF, POS_ZERO),
    (bits(-4.0), QNAN),                            # negative operand
    (NEG_INF, QNAN),
    (QNAN, QNAN),
]

# (source, source operand text, modifier text, D), by hand; the VOP3 form is used
MODS = [
    (bits(-4.0), "-v[0:1]", "", bits(0.5)),
    (bits(-4.0), "|v[0:1]|", "", bits(0.5)),
    (bits(4.0), "-v[0:1]", "", QNAN),
    (bits(-4.0), "-|v[0:1]|", "", QNAN),
    (bits(4.0), "v[0:1]", " clamp", bits(0.5)),
    (bits(0.25), "v[0:1]", " clamp", bits(1.0)),               # 2 -> 1
]


def close(got, want):
    return same(got, want) if False else (same(got, want) or ulp_distance(got, want) <= 1)


class RsqF64(unittest.TestCase):
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
        self.assertEqual(run("v_rsq_f64 v[4:5], 0x40100000"), bits(0.5))
        self.assertEqual(run("v_rsq_f64_e64 v[4:5], 0x40100000"), bits(0.5))
        self.assertEqual(run("v_rsq_f64 v[4:5], 0x40300000"), bits(0.25))          # 16.0

    def test_random_operands(self):
        rng = random.Random(0x25F64)
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
