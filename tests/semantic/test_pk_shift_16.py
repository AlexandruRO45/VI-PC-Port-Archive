"""v_pk_lshlrev_b16, v_pk_lshrrev_b16 and v_pk_ashrrev_i16 on every op_sel / op_sel_hi combination.

RDNA2 ISA (VOP3P): for each result half, the shift amount is bits [3:0] of one half of src0 and the value is one half
of src1 (the shifted operand is src1: the "rev" forms); the low result half uses op_sel[0] (src0) and op_sel[1] (src1),
the high half op_sel_hi[0] and op_sel_hi[1], a set bit meaning the high half of that source. ashr shifts the 16-bit
value arithmetically, lshr logically, lshl keeps the low 16 bits. The lowering is written without OpBitField{S,U}Extract
and with the two halves as different expressions, because the AMD Windows driver (25.10.36 and 26.10.1) miscompiled
the extract form's wave32 high half (SHADER-777, found with tools/valu_fuzz.py --family f16 --seed 1; programs
1:534:32, 1:1146:32, 1:1540:32, 1:1556:32 and 1:378:64 on that driver). The reference here is plain Python on the
ISA text; runtime/test/shaders/cs_w32_pk_shift.asm is the same instructions on a GPU, with hand-derived words.
"""
import itertools
import random
import unittest

import support

OPS = {"v_pk_ashrrev_i16": "ashr", "v_pk_lshrrev_b16": "lshr", "v_pk_lshlrev_b16": "shl"}
EXPECTED_FIXTURE = [0x00000000, 0xB2C5B2C5, 0xFD96FD96, 0xF658FF4D, 0x1658074D, 0x96283420, 0x00003C00, 0xB2C5E9A1]


def s16(x):
    return x - 0x10000 if x & 0x8000 else x


def reference(kind, src0, src1, sel, selhi):
    out = 0
    for half, (s_sel, v_sel) in enumerate((sel, selhi)):
        amount = ((src0 >> (16 * s_sel)) & 0xFFFF) & 15
        value = (src1 >> (16 * v_sel)) & 0xFFFF
        if kind == "ashr":
            r = (s16(value) >> amount) & 0xFFFF
        elif kind == "lshr":
            r = value >> amount
        else:
            r = (value << amount) & 0xFFFF
        out |= r << (16 * half)
    return out


def run(op, src0, src1, sel, selhi):
    lines = [f"v_mov_b32_e32 v1, {src0:#x}", f"v_mov_b32_e32 v2, {src1:#x}",
             f"{op} v3, v1, v2 op_sel:[{sel[0]},{sel[1]}] op_sel_hi:[{selhi[0]},{selhi[1]}]",
             "v_mov_b32_e32 v4, 0", "exp mrt0 v3, v4, v4, v4 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_pk_shift", lines))[0]


class PkShift16(unittest.TestCase):
    def test_every_combination_matches_the_reference(self):
        rng = random.Random(1556)
        values = [0, 0xFFFFFFFF, 0x80008000, 0x7FFF7FFF, 0xB2C5E9A1, 0x3C00]
        for op, kind in OPS.items():
            for bits in itertools.product((0, 1), repeat=4):
                sel, selhi = (bits[0], bits[1]), (bits[2], bits[3])
                pairs = [(rng.choice(values), rng.choice(values)) for _ in range(3)] + \
                        [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(3)]
                for src0, src1 in pairs:
                    with self.subTest(op=op, sel=sel, selhi=selhi, src0=hex(src0), src1=hex(src1)):
                        self.assertEqual(run(op, src0, src1, sel, selhi),
                                         reference(kind, src0, src1, sel, selhi))

    def test_the_cs_w32_pk_shift_fixture(self):
        # the six shift cases of cs_w32_pk_shift.asm, S and T constants, A and B as lane 0 holds them
        s, t, a, b = 0xBC005176, 0x00030005, 0x00003C00, 0xB2C5E9A1
        cases = [("ashr", s, a, (1, 1), (1, 1)), ("ashr", s, b, (1, 1), (1, 1)), ("ashr", t, b, (0, 1), (0, 1)),
                 ("ashr", t, b, (0, 0), (1, 1)), ("lshr", t, b, (0, 0), (1, 1)), ("shl", t, b, (0, 0), (1, 1))]
        self.assertEqual([reference(*c) for c in cases], EXPECTED_FIXTURE[:6])


if __name__ == "__main__":
    unittest.main()
