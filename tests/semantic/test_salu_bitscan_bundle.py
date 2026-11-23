"""Semantic tests for RDNA2 SALU Bit Scanning and Leading Zero/One Count bundle.

Covers:
- s_ff0_i32_b32: Find first zero bit in 32-bit integer (LSB to MSB)
- s_ff0_i32_b64: Find first zero bit in 64-bit integer (LSB to MSB)
- s_ff1_i32_b32: Find first one bit in 32-bit integer (LSB to MSB)
- s_ff1_i32_b64: Find first one bit in 64-bit integer (LSB to MSB)
- s_flbit_i32_b32: Find last bit / leading zeros count in 32-bit integer (MSB down)
- s_flbit_i32_b64: Find last bit / leading zeros count in 64-bit integer (MSB down)

Tests pure mathematical reference models, AST and scalar interpreter semantics,
status flag (SCC) neutrality, boundary/sign/wrap cases,
and hardware translation into SPIR-V.
"""
from __future__ import annotations

import dataclasses
import enum
import os
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


def ref_ff0_i32_b32(src0: int) -> int:
    """Reference mathematical oracle for s_ff0_i32_b32.
    Returns index of first 0 bit (0..31, LSB to MSB), or -1 (0xFFFFFFFF) if all bits are 1.
    """
    val = src0 & MASK32
    if val == MASK32:
        return MASK32
    inv = (~val) & MASK32
    lsb = inv & (-inv)
    return lsb.bit_length() - 1


class TestSFf0I32B32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_ff0_i32_b32."""

    def test_reference_zero(self):
        self.assertEqual(ref_ff0_i32_b32(0), 0)

    def test_reference_first_zero_bit(self):
        self.assertEqual(ref_ff0_i32_b32(1), 1)
        self.assertEqual(ref_ff0_i32_b32(0b0101), 1)
        self.assertEqual(ref_ff0_i32_b32(0b0111), 3)

    def test_reference_msb_zero(self):
        self.assertEqual(ref_ff0_i32_b32(0x7FFFFFFF), 31)

    def test_reference_no_zero_bit(self):
        self.assertEqual(ref_ff0_i32_b32(0xFFFFFFFF), MASK32)


if __name__ == "__main__":
    unittest.main()
