"""A literal dword on an f64 source of v_rcp_f64, v_rsq_f64, v_sqrt_f64, v_frexp_mant_f64 and
v_frexp_exp_i32_f64 is bits 63:32 of the operand (docs/design.md "f64 literal sources"; SHADER-865).

    cd tests/semantic && python -m unittest test_f64_literal_ops

gfx1030 llvm-mc prints 1.5 as the literal word 0x3ff80000; these ops put it in bits 31:0 instead, so
v_frexp_exp_i32_f64 of 1.5 gave -1044 and the three %double ops built 0x000000003ff80000.
"""
import re
import unittest

import support

ONE_POINT_FIVE = 0x3FF8000000000000
EXP = "exp mrt0 v0, v1, v2, v3 done vm"


def translate(lines):
    return support.translate_lines("ps_f64_literal", list(lines) + [EXP])


def literal_doubles(spv):
    """The 64-bit values built from two OpConstant %uint words and bitcast to %double: the literal operands."""
    words = {name: int(value) for name, value in re.findall(r"(%\w+) = OpConstant %uint (\d+)", spv)}
    pairs = {name: (words[high] << 32) | words[low]
             for name, low, high in re.findall(r"(%\w+) = OpCompositeConstruct %v2uint_d (%\w+) (%\w+)", spv)
             if low in words and high in words}
    return sorted(pairs[pair] for name, pair in re.findall(r"(%\w+) = OpBitcast %double (%\w+)", spv) if pair in pairs)


class LiteralWordIsTheHighDword(unittest.TestCase):
    def test_rcp_rsq_sqrt_build_the_literal_as_bits_63_32(self):
        for insn in ("v_rcp_f64 v[0:1], 0x3ff80000", "v_rsq_f64 v[0:1], 0x3ff80000", "v_sqrt_f64 v[0:1], 0x3ff80000"):
            with self.subTest(insn=insn):
                doubles = literal_doubles(translate([insn]))   # v_rcp_f64 also builds its 1.0
                self.assertIn(ONE_POINT_FIVE, doubles)
                self.assertNotIn(0x3FF80000, doubles)          # the literal in bits 31:0: the bug

    def test_frexp_exp_of_a_literal(self):
        words = support.mrt0_words(translate(["v_frexp_exp_i32_f64 v0, 0x3ff80000"]))
        self.assertEqual(words[0], 1)                       # 1.5 = 0.75 * 2^1

    def test_frexp_mant_of_a_literal(self):
        words = support.mrt0_words(translate(["v_frexp_mant_f64 v[0:1], 0x3ff80000"]))
        self.assertEqual((words[0], words[1]), (0, 0x3FE80000))   # 0.75

    def test_an_inline_integer_stays_a_sign_extended_integer(self):
        # Inline integers are not literal words: -1 is 0xffffffffffffffff, a NaN, whose frexp exponent is 0.
        words = support.mrt0_words(translate(["v_frexp_exp_i32_f64 v0, -1"]))
        self.assertEqual(words[0], 0)

    def test_a_hand_written_64_bit_literal_keeps_its_value(self):
        words = support.mrt0_words(translate(["v_frexp_exp_i32_f64 v0, 0x4010000000000000"]))   # 4.0 = 0.5 * 2^3
        self.assertEqual(words[0], 3)


if __name__ == "__main__":
    unittest.main()
