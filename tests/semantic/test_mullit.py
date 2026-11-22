"""V_MULLIT_F32 in the inferred mode (IEEE off, f32 denormals flushed)."""
import unittest
from pathlib import Path

from support import mrt0_words, translate_lines


class MullitTests(unittest.TestCase):
    def test_mullit(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "ps_mullit.asm"
        module = translate_lines(fixture.stem, fixture.read_text(encoding="ascii").splitlines())
        lowest, inf, one = 0xFF7FFFFF, 0x7F800000, 0x3F800000
        for lhs, rhs, gate, expected in (
            (0x3E800000, 0x3F000000, one, 0x3E000000),       # lit: 0.25 * 0.5
            (one, one, 0, lowest), (one, one, 0xBF800000, lowest),
            (one, one, 1, lowest), (one, one, 0x7FC00001, lowest),
            (0, lowest, one, lowest), (one, 0xFF800000, one, lowest),
            (one, 0x7FC00001, one, lowest), (0, inf, one, 0),
            (0x7FC12345, 0, one, 0), (0x7FA12345, one, one, 0x7FA12345),
            (0x80000001, one, one, 0), (0x80800000, 0x3F000000, one, 0x80000000),
            (0x3F7FFFFE, 0x00800001, one, 0x00800000),
            (0x3F7FFFFF, 0x00800000, one, 0),
            (0x7F7FFFFF, 0x40000000, inf, inf),
        ):
            with self.subTest(lhs=hex(lhs), rhs=hex(rhs), gate=hex(gate)):
                words = mrt0_words(module, push_constants=[lhs, rhs, gate] + [0] * 29)
                self.assertEqual(words[3], expected ^ 0x40000000)
                self.assertEqual(words[2], lowest)
                if lhs == 0x3E800000:
                    self.assertEqual(words[0:2], [0x3E000000, 0xBE000000])
                if rhs == 0xFF800000 and lhs == one:
                    self.assertEqual(words[1], 0xFF800000)  # -1 * abs(-inf)
