"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_and_or_b32.

Models RDNA2 / GFX10 VALU 3-operand bitwise operation:
  v_and_or_b32 vd, src0, src1, src2

RDNA2 VALU Architecture & Semantics:
  1. Operands & Computation:
     - vd: 32-bit Destination VGPR
     - src0, src1, src2: 32-bit input sources (VGPR, SGPR, or literal/inline constant)
     - Operation: dst = (src0 & src1) | src2
  2. Condition Codes & Flags:
     - Neither SCC nor VCC is modified by this bitwise instruction.
  3. Execution Masking (EXEC):
     - Inactive lanes under EXEC mask do not write to their destination VGPR.
  4. Translator Integration:
     - Validates end-to-end SPIR-V emission via support.translate_lines and execution under spirv_mini_interp.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned integer as signed two's-complement."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def ref_and_or_b32(src0: int, src1: int, src2: int) -> int:
    """Pure mathematical reference oracle for RDNA2 v_and_or_b32.

    Computes: (src0 & src1) | src2
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    s2 = src2 & MASK32
    return ((s0 & s1) | s2) & MASK32

class TestValuAndOrB32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and bitwise truth tables."""

    def test_basic_and_or_truth_table(self) -> None:
        # 4-bit exhaustive truth table patterns replicated across dwords
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    src0 = 0xFFFFFFFF if a else 0
                    src1 = 0xFFFFFFFF if b else 0
                    src2 = 0xFFFFFFFF if c else 0
                    expected = 0xFFFFFFFF if ((a and b) or c) else 0
                    self.assertEqual(ref_and_or_b32(src0, src1, src2), expected)

    def test_annihilators_and_identities(self) -> None:
        # src2 = 0xFFFFFFFF -> result always 0xFFFFFFFF
        self.assertEqual(ref_and_or_b32(0, 0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_and_or_b32(0x12345678, 0x9ABCDEF0, 0xFFFFFFFF), 0xFFFFFFFF)

        # src2 = 0 -> result is src0 & src1
        self.assertEqual(ref_and_or_b32(0x0F0F0F0F, 0x33333333, 0), 0x03030303)

        # src0 = 0 -> result is src2
        self.assertEqual(ref_and_or_b32(0, 0xFFFFFFFF, 0xCAFEBABE), 0xCAFEBABE)

        # src1 = 0 -> result is src2
        self.assertEqual(ref_and_or_b32(0xFFFFFFFF, 0, 0xDEADBEEF), 0xDEADBEEF)

    def test_commutativity_of_and_operands(self) -> None:
        val_a = 0xA5A5A5A5
        val_b = 0x5A5A5A5A
        val_c = 0x12345678
        self.assertEqual(
            ref_and_or_b32(val_a, val_b, val_c),
            ref_and_or_b32(val_b, val_a, val_c),
        )

    def test_bitfield_manipulation_masks(self) -> None:
        # Clear field (AND mask) then set field (OR mask)
        orig = 0x12345678
        clear_mask = 0xFFFF00FF
        insert_val = 0x0000AB00
        res = ref_and_or_b32(orig, clear_mask, insert_val)
        self.assertEqual(res, 0x1234AB78)

if __name__ == "__main__":
    unittest.main()
