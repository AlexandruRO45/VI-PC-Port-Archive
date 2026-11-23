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


if __name__ == "__main__":
    unittest.main()
