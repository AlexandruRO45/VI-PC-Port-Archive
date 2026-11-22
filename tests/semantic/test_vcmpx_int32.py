"""v_cmpx_{eq,ne,lt,le,gt,ge}_{i32,u32}, e32 and e64, on values where signed and unsigned disagree.

RDNA2 ISA (VOPC): _U32 compares S0.u32 with S1.u32, _I32 compares S0.i32 with S1.i32; EQ, NE, LT, LE, GT, GE are the
usual relations. A v_cmpx writes EXEC = EXEC & compare and nothing else, so VCC is kept and a lane that is off stays
off. The shader is runtime/test/shaders/ps_vcmpx_int32.asm (also run on the GPU by check.py and run_gpu_tests.py); its
header derives the four words by hand. The reference here is plain Python on the same operand pairs.
"""
import operator
import unittest

import support

EXPECTED = [0x984DCEE8, 0x963E52E8, 0xB73612E8, 0xB9458EE8]      # derivation in ps_vcmpx_int32.asm
HEAP = (1, 0, 0x80000000, 0x12345678, 3, 5, 0, 0)
# (A, B) as the shader sees them: A is src0 in e32 (a literal, an inline constant or v5), B a VGPR.
PAIRS = ((0x80000000, 1), (0xFFFFFFFF, 0), (1, 0x80000000), (0x12345678, 0x12345678), (3, 5))
OPS = (("eq", operator.eq), ("ne", operator.ne), ("lt", operator.lt), ("le", operator.le),
       ("gt", operator.gt), ("ge", operator.ge))


def signed(x):
    return x - (1 << 32) if x >> 31 else x


def reference_word(signed_compare, swapped):
    """One mrt0 word: bit 5*k + i is op k on pair i. Bit 30 is 0 (EXEC emptied before the op stays empty) and
    bit 31 is 1 (VCC survives a v_cmpx whose result is false in every lane)."""
    word = 1 << 31
    for k, (_, rel) in enumerate(OPS):
        for i, (a, b) in enumerate(PAIRS):
            s0, s1 = (b, a) if swapped else (a, b)
            if signed_compare:
                s0, s1 = signed(s0), signed(s1)
            if rel(s0, s1):
                word |= 1 << (5 * k + i)
    return word


class VcmpxInt32(unittest.TestCase):
    def test_reference_matches_the_header_words(self):
        ref = [reference_word(True, False), reference_word(False, False),
               reference_word(True, True), reference_word(False, True)]
        self.assertEqual([f"{w:#010x}" for w in ref], [f"{w:#010x}" for w in EXPECTED])

    def test_ps_vcmpx_int32(self):
        spv = support.translate_fixture("ps_vcmpx_int32")
        words = support.mrt0_words(spv, push_constants=[0x1000, 0, 32, 0] + [0] * 28,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(HEAP)})
        self.assertEqual([f"{w:#010x}" for w in words], [f"{w:#010x}" for w in EXPECTED])


if __name__ == "__main__":
    unittest.main()
