"""v_cmp_{eq,ne,lt,le,gt,ge}_{i16,u16}, e32 (VCC) and e64 (SGPR pair), on values where signed and unsigned disagree.

RDNA2 ISA (VOPC): _U16 compares S0.u16 with S1.u16, _I16 compares S0.i16 with S1.i16, on the low 16 bits of the
operands only; EQ, NE, LT, LE, GT, GE are the usual relations. The shader is runtime/test/shaders/ps_vcmp_int16.asm
(also run on the GPU by check.py and run_gpu_tests.py); its header derives the four words by hand. The reference here
is plain Python on the same operand pairs, with junk in the high halves of the register operands.
"""
import operator
import unittest

import support

EXPECTED = [0x184DCEE8, 0x163E52E8, 0x373612E8, 0x39458EE8]      # derivation in ps_vcmp_int16.asm
HEAP = (0xAAAA0001, 0x55550000, 0x12348000, 0xABCD1234, 0xFFFF0003, 5, 0, 0)
# (A, B) as 16-bit values: A is src0 in e32 (a 16-bit literal, an inline constant or the low half of v5), B the low
# half of a VGPR.
PAIRS = ((0x8000, 0x0001), (0xFFFF, 0x0000), (0x0001, 0x8000), (0x1234, 0x1234), (0x0003, 0x0005))
OPS = (("eq", operator.eq), ("ne", operator.ne), ("lt", operator.lt), ("le", operator.le),
       ("gt", operator.gt), ("ge", operator.ge))


def signed(x):
    return x - (1 << 16) if x >> 15 else x


def reference_word(signed_compare, swapped):
    """One mrt0 word: bit 5*k + i is op k on pair i; bits 30 and 31 are 0."""
    word = 0
    for k, (_, rel) in enumerate(OPS):
        for i, (a, b) in enumerate(PAIRS):
            s0, s1 = (b, a) if swapped else (a, b)
            if signed_compare:
                s0, s1 = signed(s0), signed(s1)
            if rel(s0, s1):
                word |= 1 << (5 * k + i)
    return word


class VcmpInt16(unittest.TestCase):
    def test_reference_matches_the_header_words(self):
        ref = [reference_word(True, False), reference_word(False, False),
               reference_word(True, True), reference_word(False, True)]
        self.assertEqual([f"{w:#010x}" for w in ref], [f"{w:#010x}" for w in EXPECTED])

    def test_ps_vcmp_int16(self):
        spv = support.translate_fixture("ps_vcmp_int16")
        words = support.mrt0_words(spv, push_constants=[0x1000, 0, 32, 0] + [0] * 28,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(HEAP)})
        self.assertEqual([f"{w:#010x}" for w in words], [f"{w:#010x}" for w in EXPECTED])


if __name__ == "__main__":
    unittest.main()
