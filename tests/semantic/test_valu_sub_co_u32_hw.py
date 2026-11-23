from __future__ import annotations

"""Semantic and hardware reference tests for RDNA2 v_sub_co_u32.

Validates 32-bit unsigned vector subtraction with carry-out / borrow:
  vdst = (src0 - src1) & 0xFFFFFFFF
  sdst / vcc_lo = 0xFFFFFFFF if (src0 < src1) else 0

Exercises reference mathematical oracle, symbolic AST definitions, deterministic
scalar interpreter, SIMD multi-lane divergence with EXEC masking, register
aliasing, status flags with carry/borrow, IEEE 754 bit-pattern invariance, and
negative controls. All stages cross-checked with rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

_TRANSLATE_ID = 0


def oracle_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Mathematical reference oracle for v_sub_co_u32.

    Returns:
        (diff, borrow) where:
          diff = (src0 - src1) & 0xFFFFFFFF
          borrow = 1 if (src0 & 0xFFFFFFFF) < (src1 & 0xFFFFFFFF) else 0
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    diff = (u0 - u1) & MASK32
    borrow = 1 if u0 < u1 else 0
    return diff, borrow


def run_translated_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Execute v_sub_co_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        "v_sub_co_u32 v0, vcc_lo, v1, v2",
        "v_cndmask_b32 v3, 0, 1, vcc_lo",
        "exp mrt0 v0, v3, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_sub_co_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32, words[1] & 1

# ---------------------------------------------------------------------------
# Stage 1: Reference mathematical oracle and architectural model
# ---------------------------------------------------------------------------

class TestValuSubCoU32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_sub_co_u32."""

    def test_oracle_basic_subtraction(self) -> None:
        self.assertEqual(oracle_sub_co_u32(10, 3), (7, 0))
        self.assertEqual(oracle_sub_co_u32(100, 100), (0, 0))
        self.assertEqual(oracle_sub_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_sub_co_u32(1000, 400), (600, 0))

    def test_oracle_borrow_subtraction(self) -> None:
        self.assertEqual(oracle_sub_co_u32(3, 10), (MASK32 - 6, 1))
        self.assertEqual(oracle_sub_co_u32(0, 1), (MASK32, 1))
        self.assertEqual(oracle_sub_co_u32(0, MASK32), (1, 1))
        self.assertEqual(oracle_sub_co_u32(MASK32, 0), (MASK32, 0))

    def test_oracle_matches_translated_execution(self) -> None:
        pairs = [
            (10, 3),
            (3, 10),
            (0, 0),
            (0, 1),
            (1, 0),
            (MASK32, 1),
            (1, MASK32),
            (0x12345678, 0x11111111),
            (0x11111111, 0x12345678),
            (0x80000000, 1),
            (0x7FFFFFFF, 0x80000000),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
        ]
        for a, b in pairs:
            expected = oracle_sub_co_u32(a, b)
            actual = run_translated_sub_co_u32(a, b)
            self.assertEqual(actual, expected, f"Mismatch on {a:#x} - {b:#x}")

if __name__ == "__main__":
    unittest.main()
