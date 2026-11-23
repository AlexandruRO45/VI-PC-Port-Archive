"""RDNA2 text-operand spellings survive parsing and SPIR-V translation.

Only hand-written synthetic disassembly; no game data or GPU required.
"""
from pathlib import Path
import unittest

import support


class FloatOperandSpellings(unittest.TestCase):
    def test_extended_float_literals_reach_mrt0_as_f32_bits(self):
        fixture = Path(__file__).with_name("fixtures") / "float_operand_spellings.asm"
        spvasm = support.translate_lines("ps_float_operand_spellings",
                                         fixture.read_text(encoding="utf-8").splitlines())
        # Independent IEEE-754 f32 bit patterns for 1, 1/2, 1, 1/4.
        self.assertEqual(support.mrt0_words(spvasm), [
            0x3F800000, 0x3F000000, 0x3F800000, 0x3E800000,
        ])

    def test_f32_midpoints_match_hex_controls(self):
        # The decimal strings around each midpoint round to the same binary64.
        cases = [
            ("1.000000059604644775390625000001e0", 0x3F800001),
            ("1.000000059604644775390625", 0x3F800000),
            ("1.000000059604644775390624999999", 0x3F800000),
            ("1.000000178813934326171874999999", 0x3F800001),
            ("1.000000178813934326171875", 0x3F800002),
        ]
        for literal, word in cases:
            with self.subTest(literal=literal):
                negative = word | 0x80000000
                module = support.translate_lines("ps_midpoint", [
                    f"v_mov_b32 v0, {literal}", f"v_mov_b32 v1, {word:#x}",
                    f"v_mov_b32 v2, -{literal}", f"v_mov_b32 v3, {negative:#x}",
                    "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm",
                ])
                self.assertEqual(support.mrt0_words(module), [word, word, negative, negative])

    def test_f16_midpoint_matches_hex_control(self):
        # Just above the midpoint between half words 0x3c00 and 0x3c01.
        literal = "1.000488281250000000000001e0"
        module = support.translate_lines("ps_widths", [
            f"v_cvt_f32_f16 v0, {literal}", "v_cvt_f32_f16 v1, 0x3c01",
            "exp mrt0 v0, v1, v0, v1 done vm", "s_endpgm",
        ])
        self.assertEqual(support.mrt0_words(module), [0x3f802000] * 4)

    def test_double_sign_is_rejected(self):
        with self.assertRaisesRegex(support.rdna2spirv.Unsupported, "operand --1e0"):
            support.translate_lines("ps_double_sign", [
                "v_mov_b32 v0, --1e0", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm",
            ])


if __name__ == "__main__":
    unittest.main()
