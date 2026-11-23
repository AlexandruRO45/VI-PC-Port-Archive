import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402

LITERALS = [
    ("0x40000000", 0x3FE0000000000000, 2),
    ("0x3ff00000", 0x3FE0000000000000, 1),
    ("0x3ff80000", 0x3FE8000000000000, 1),
    ("0xc0080000", 0xBFE8000000000000, 2),
    ("0x00080000", 0x3FE0000000000000, -1022),
    ("0x7ff00000", 0x7FF0000000000000, 0),
]

INLINE = [
    ("2.0", 0x3FE0000000000000, 2),
    ("1.0", 0x3FE0000000000000, 1),
    ("-4.0", 0xBFE0000000000000, 3),
    ("2", 0x3FE0000000000000, -1072),
    ("-1", 0xFFFFFFFFFFFFFFFF, 0),
]


def frexp(source, suffix=""):
    spv = support.translate_lines("ps_frexp_f64_literal", [
        f"v_frexp_mant_f64{suffix} v[0:1], {source}",
        f"v_frexp_exp_i32_f64{suffix} v2, {source}",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ])
    out = support.run_fragment(spv)["%out_mrt0"]
    return spv, out[0] | (out[1] << 32), out[2]


class FrexpF64Literal(unittest.TestCase):
    def check(self, cases, suffix=""):
        for source, mant, exp in cases:
            with self.subTest(source=source, suffix=suffix):
                _, got_mant, got_exp = frexp(source, suffix)
                self.assertEqual(hex(got_mant), hex(mant))
                self.assertEqual(got_exp, exp & 0xFFFFFFFF)

    def test_literal_is_bits_63_32(self):
        self.check(LITERALS)

    def test_literal_is_bits_63_32_in_the_vop3_form(self):
        self.check(LITERALS, "_e64")

    def test_inline_constants_keep_their_binary64_reading(self):
        self.check(INLINE)

    def test_a_whole_binary64_word_written_by_hand_is_that_value(self):
        self.check([("0x3ff8000000000000", 0x3FE8000000000000, 1)])

    def test_literal_source_needs_no_float64(self):
        spv, _, _ = frexp("0x3ff80000")
        self.assertNotIn("OpCapability Float64", spv)


if __name__ == "__main__":
    unittest.main()
