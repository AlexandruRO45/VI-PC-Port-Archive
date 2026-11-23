from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

def ref_v_add_i16(s0: int, s1: int, clamp: bool = False) -> int:
    x = (s0 & 0x7FFF) - (s0 & 0x8000)
    y = (s1 & 0x7FFF) - (s1 & 0x8000)
    res = x + y
    if clamp:
        res = max(-32768, min(32767, res))
    return res & MASK16


class TestVAddI16Reference(unittest.TestCase):
    def test_ref_add_positive(self):
        self.assertEqual(ref_v_add_i16(100, 200), 300)

    def test_ref_add_negative(self):
        self.assertEqual(ref_v_add_i16(0xFFFF, 0xFFFF), 0xFFFE)  # -1 + -1 = -2

    def test_ref_add_wrap(self):
        self.assertEqual(ref_v_add_i16(0x7FFF, 1), 0x8000)  # 32767 + 1 = -32768 (wrap)

    def test_ref_add_clamp(self):
        self.assertEqual(ref_v_add_i16(0x7FFF, 1, clamp=True), 0x7FFF)  # 32767 + 1 clamped
        self.assertEqual(ref_v_add_i16(0x8000, 0xFFFF, clamp=True), 0x8000)  # -32768 + -1 clamped


if __name__ == "__main__":
    unittest.main()
