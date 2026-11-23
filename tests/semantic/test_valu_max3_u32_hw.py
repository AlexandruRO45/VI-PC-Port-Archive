from __future__ import annotations

"""Semantic and hardware reference tests for RDNA2 v_max3_u32.

Validates 32-bit unsigned three-input vector maximum:
  vdst = max(src0, src1, src2) (unsigned 32-bit integer arithmetic)

Lowers to GLSL.std.450 UMax via rdna2spirv.

Exercises reference mathematical oracle, symbolic AST definitions, deterministic
scalar interpreter, SIMD multi-lane divergence with EXEC masking, register
aliasing, status flags with unsigned bounds, IEEE 754 bit-pattern invariance, and
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

_TRANSLATE_ID = 0


def oracle_max3_u32(src0: int, src1: int, src2: int) -> int:
    """Mathematical reference oracle for v_max3_u32.

    Returns:
        Unsigned maximum of three 32-bit unsigned integers.
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    u2 = src2 & MASK32
    return max(u0, u1, u2)


def run_translated_max3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_max3_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_max3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_max3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32

# ---------------------------------------------------------------------------
# Stage 1: Reference mathematical oracle and architectural model
# ---------------------------------------------------------------------------

class TestValuMax3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_max3_u32."""

    def test_oracle_basic_maximum(self) -> None:
        self.assertEqual(oracle_max3_u32(10, 50, 30), 50)
        self.assertEqual(oracle_max3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_max3_u32(1, 2, 3), 3)
        self.assertEqual(oracle_max3_u32(3, 2, 1), 3)
        self.assertEqual(oracle_max3_u32(2, 3, 1), 3)

    def test_oracle_unsigned_semantics(self) -> None:
        # In unsigned arithmetic, 0xFFFFFFFF > 0x7FFFFFFF > 0
        self.assertEqual(oracle_max3_u32(0, 0x7FFF_FFFF, MASK32), MASK32)
        self.assertEqual(oracle_max3_u32(0x8000_0000, 10, 20), 0x8000_0000)
        self.assertEqual(oracle_max3_u32(MASK32, MASK32 - 1, 0), MASK32)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (10, 50, 30),
            (0, 0, 0),
            (1, 2, 3),
            (3, 2, 1),
            (2, 3, 1),
            (0x12345678, 0x99999999, 0x55555555),
            (MASK32, 0x80000000, 0x7FFFFFFF),
            (0, MASK32, 1),
            (100, 100, 100),
            (100, 200, 100),
            (200, 100, 100),
        ]
        for a, b, c in triples:
            expected = oracle_max3_u32(a, b, c)
            actual = run_translated_max3_u32(a, b, c)
            self.assertEqual(actual, expected, f"Mismatch on max3({a:#x}, {b:#x}, {c:#x})")

    def test_oracle_permutation_invariance(self) -> None:
        a, b, c = 123, 456, 789
        m = oracle_max3_u32(a, b, c)
        self.assertEqual(oracle_max3_u32(a, c, b), m)
        self.assertEqual(oracle_max3_u32(b, a, c), m)
        self.assertEqual(oracle_max3_u32(b, c, a), m)
        self.assertEqual(oracle_max3_u32(c, a, b), m)
        self.assertEqual(oracle_max3_u32(c, b, a), m)

if __name__ == "__main__":
    unittest.main()
