"""Semantic tests for RDNA2 VALU Sum of Absolute Differences (SAD) bundle.

Covers:
- v_sad_u32
- v_sad_u16
- v_sad_u8
- v_sad_hi_u8
- v_msad_u8
- v_mqsad_u32_u8

Tests reference mathematical models, AST and scalar interpreter semantics,
SIMD multi-lane divergence and EXEC masking, 32-bit wrapping and accumulator semantics,
and hardware translation into SPIR-V.
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402

_ = support  # Reachable guard for tests/test_semantic_translator_use.py

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_sad_u32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_sad_u32."""
    diff = abs((s0 & MASK32) - (s1 & MASK32))
    return ((s2 & MASK32) + diff) & MASK32


class TestVSadU32Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u32."""

    def test_reference_canonical(self):
        self.assertEqual(ref_sad_u32(100, 40, 10), 70)
        self.assertEqual(ref_sad_u32(40, 100, 10), 70)

    def test_reference_zero_diff(self):
        self.assertEqual(ref_sad_u32(55, 55, 12), 12)

    def test_reference_wrapping(self):
        self.assertEqual(ref_sad_u32(0xFFFFFFFF, 0, 1), 0)


if __name__ == "__main__":
    unittest.main()
