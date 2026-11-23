from __future__ import annotations

"""Semantic regression test suite and reference verification for v_max3_i32.

RDNA2 Vector ALU instruction:
    v_max3_i32 vdst, vsrc0, vsrc1, vsrc2

Semantics:
    Signed two's complement 32-bit maximum across three operands:
        vdst = max(int32(vsrc0), int32(vsrc1), int32(vsrc2))

Reference mathematical oracle, properties, signed vs unsigned divergence,
AST definitions, deterministic scalar interpreter, SIMD multi-lane divergence,
register aliasing, condition codes invariance, two's complement and IEEE 754
bit pattern edge cases, and translator-backed SPIR-V execution.

Standard library Python only (python -I), zero game data.
"""

import itertools
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFFFFFF
SIGN_BIT32 = 0x80000000
INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF


def to_signed_i32(x: int) -> int:
    """Interpret raw 32-bit integer as two's complement signed 32-bit integer."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_BIT32) else x


def to_unsigned_u32(x: int) -> int:
    """Mask integer to 32-bit unsigned integer range [0, 2^32 - 1]."""
    return x & MASK32


def ref_v_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for v_max3_i32: signed maximum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return max(v0, v1, v2) & MASK32


def ref_pairwise_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed maximum: max(max(s0, s1), s2)."""
    step1 = max(to_signed_i32(s0), to_signed_i32(s1))
    return max(step1, to_signed_i32(s2)) & MASK32

class TestVMax3I32ReferenceOracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and core properties for v_max3_i32."""

    def test_basic_signed_max(self) -> None:
        """Verify standard signed maximum calculations with positive and negative inputs."""
        self.assertEqual(ref_v_max3_i32(10, 20, 30), 30)
        self.assertEqual(ref_v_max3_i32(-10, -20, -5), (-5) & MASK32)
        self.assertEqual(ref_v_max3_i32(-1, 0, 1), 1)
        self.assertEqual(ref_v_max3_i32(-100, 50, -200), 50)
        self.assertEqual(ref_v_max3_i32(0, 0, 0), 0)

    def test_signed_vs_unsigned_divergence(self) -> None:
        """Verify divergence between signed maximum and unsigned maximum on identical bit patterns."""
        # 0xFFFFFFFF is -1 as i32, UINT32_MAX as u32.
        # With 0 and 10: signed max is 10 (-1 < 0 < 10); unsigned max is 0xFFFFFFFF.
        self.assertEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), 10)
        self.assertNotEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), max(0, 0xFFFFFFFF, 10))

        # 0x80000000 is INT32_MIN as i32, 2^31 as u32.
        # With 0x7FFFFFFF (INT32_MAX) and 0:
        self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), 0x7FFFFFFF)
        # Unsigned max would pick 0x80000000 (2147483648 > 2147483647).
        self.assertNotEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), max(0x7FFFFFFF, 0x80000000, 0))

    def test_pairwise_reduction_equivalence(self) -> None:
        """Verify 3-way signed max identically matches chained pairwise signed max."""
        samples = [
            (0, 0, 0),
            (1, 2, 3),
            (-5, 10, -20),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x7FFFFFFE),
            (0x12345678, 0x87654321, 0x00000000),
            (0xFFFFFFFE, 0xFFFFFFFF, 0x00000001),
        ]
        for s0, s1, s2 in samples:
            with self.subTest(s0=hex(s0 & MASK32), s1=hex(s1 & MASK32), s2=hex(s2 & MASK32)):
                self.assertEqual(ref_v_max3_i32(s0, s1, s2), ref_pairwise_max3_i32(s0, s1, s2))

    def test_idempotent_and_identity_properties(self) -> None:
        """Verify idempotent max(x, x, x) == x and neutral/absorbing element laws."""
        test_values = [0, 1, -1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 42, -9999]
        for val in test_values:
            raw = val & MASK32
            # Idempotence: max(x, x, x) == x
            self.assertEqual(ref_v_max3_i32(val, val, val), raw)
            # Neutral element: INT32_MIN (0x80000000)
            self.assertEqual(ref_v_max3_i32(val, 0x80000000, 0x80000000), raw)
            # Absorbing element: INT32_MAX (0x7FFFFFFF)
            self.assertEqual(ref_v_max3_i32(val, 0x7FFFFFFF, 0), 0x7FFFFFFF)

    def test_permutation_invariance(self) -> None:
        """Verify that all 3! = 6 permutations yield identical results."""
        triples = [
            (0x80000000, 0, 0x7FFFFFFF),
            (0xFFFFFFFF, 1, 0x80000000),
            (42, 42, 42),
            (0x12345678, 0x9ABCDEF0, 0x00000000),
            (0x80000000, 0x80000001, 0x80000002),
            (0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF),
        ]
        for triple in triples:
            expected = ref_v_max3_i32(*triple)
            for perm in itertools.permutations(triple):
                self.assertEqual(ref_v_max3_i32(*perm), expected)

if __name__ == "__main__":
    unittest.main()
