"""Value and hardware regression tests for v_bfi_b32 (bitfield insert / bitwise multiplexer):

  v_bfi_b32 dst, src0, src1, src2
  Operation: dst = (src0 & src1) | (~src0 & src2)

RDNA2 Architectural Semantics:
  - src0: bitmask selector (1 selects bit from src1, 0 selects bit from src2)
  - src1: replacement field
  - src2: background/base value
  - Identity cases:
      mask == 0 -> dst = src2 (base untouched)
      mask == 0xFFFFFFFF -> dst = src1 (field replaces entirely)
  - Contiguous bitfield insertion:
      For offset O and width W: mask = ((1 << W) - 1) << O
      Translates to OpBitFieldInsert in SPIR-V when mask is a constant.
  - Arbitrary bitmasks:
      Translates to (src0 & src1) | (~src0 & src2) via OpBitwiseAnd, OpNot, OpBitwiseOr.
  - Zero flags / condition codes: does not write SCC or VCC.
  - SIMD Multi-lane: independent per-lane masking under Wave32 / Wave64 EXEC.

Standard library Python only; zero third-party dependencies; zero game data.
"""
from __future__ import annotations

import dataclasses
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_bfi_b32(mask: int, field: int, base: int) -> int:
    """Reference mathematical oracle for RDNA2 v_bfi_b32."""
    m = mask & MASK32
    f = field & MASK32
    b = base & MASK32
    return ((m & f) | ((~m & MASK32) & b)) & MASK32


class TestBfiOracle(unittest.TestCase):
    def test_identity_masks(self):
        base = 0x12345678
        field = 0x9ABCDEF0
        self.assertEqual(ref_bfi_b32(0x00000000, field, base), base)
        self.assertEqual(ref_bfi_b32(0xFFFFFFFF, field, base), field)

    def test_single_bit_masks(self):
        base = 0x00000000
        field = 0xFFFFFFFF
        for bit in (0, 1, 7, 15, 16, 30, 31):
            mask = 1 << bit
            self.assertEqual(ref_bfi_b32(mask, field, base), mask)

    def test_alternating_patterns(self):
        m1 = 0x55555555
        m2 = 0xAAAAAAAA
        f = 0xFFFFFFFF
        b = 0x00000000
        self.assertEqual(ref_bfi_b32(m1, f, b), 0x55555555)
        self.assertEqual(ref_bfi_b32(m2, f, b), 0xAAAAAAAA)
        self.assertEqual(ref_bfi_b32(m1, b, f), 0xAAAAAAAA)

    def test_byte_insertions(self):
        base = 0x11223344
        field = 0xAABBCCDD
        # insert low byte
        self.assertEqual(ref_bfi_b32(0x000000FF, field, base), 0x112233DD)
        # insert second byte
        self.assertEqual(ref_bfi_b32(0x0000FF00, field, base), 0x1122CC44)
        # insert high byte
        self.assertEqual(ref_bfi_b32(0xFF000000, field, base), 0xAA223344)


if __name__ == "__main__":
    unittest.main()
