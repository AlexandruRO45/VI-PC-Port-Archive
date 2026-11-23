"""v_fract_f16 of a NaN, measured on gfx1036 in the game's MODE (IEEE 0, DX10_CLAMP 1; docs/design.md, float
min/max/med3 with NaN and v_fract): the source half comes back unchanged, payload and signalling bit included, with
neg/abs applied to the sign bit only; with clamp the result is +0."""
import unittest
from pathlib import Path

from support import mrt0_words, translate_lines

# source half -> (plain, -x, |x|, -x clamp), each the low half under the kept high half of the fixture
CASES = {
    0x7C01: (0x7C01, 0xFC01, 0x7C01, 0x0000),     # signalling NaN
    0xFE11: (0xFE11, 0x7E11, 0x7E11, 0x0000),     # negative quiet NaN with a payload
    0x7C00: (0xFE00, 0xFE00, 0xFE00, 0x0000),     # +inf (unchanged rule)
    0x3E00: (0x3800, 0x3800, 0x3800, 0x3800),     # 1.5: fract 0.5, and fract(-1.5) = 0.5
}


class FractF16NanTests(unittest.TestCase):
    def test_fract_f16_nan(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "ps_fract_f16_nan.asm"
        module = translate_lines(fixture.stem, fixture.read_text(encoding="ascii").splitlines())
        for source, (plain, neg, ab, clamped) in CASES.items():
            with self.subTest(source=hex(source)):
                words = mrt0_words(module, push_constants=[source] + [0] * 31)
                self.assertEqual([hex(w) for w in words],
                                 [hex(0x13570000 | plain), hex(0x24680000 | neg), hex(0x369A0000 | ab),
                                  hex(0x12340000 | clamped)])


if __name__ == "__main__":
    unittest.main()
