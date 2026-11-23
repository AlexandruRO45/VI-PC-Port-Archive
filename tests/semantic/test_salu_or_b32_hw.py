"""Semantic reference oracle and architectural model for RDNA2 s_or_b32.

AMD RDNA2 / GFX10 Scalar ALU (SALU) Architecture:
  - 32-bit Scalar Bitwise OR:
    - RDNA2 ISA: D.u32 = S0.u32 | S1.u32
    - Destination: SGPR (s0..s105) or special registers (exec_lo, exec_hi, vcc_lo, vcc_hi, m0, null)
    - Sources: SGPRs, inline constants (0..64, -1..-16), 32-bit literals, special registers
    - Condition code SCC:
        SCC = 1 if (D != 0) else 0
  - Mathematical and algebraic properties:
    - Identity element:       A | 0 == A
    - Annihilator / Dominance: A | 0xFFFFFFFF == 0xFFFFFFFF
    - Idempotence:            A | A == A
    - Commutativity:          A | B == B | A
    - Associativity:          (A | B) | C == A | (B | C)
    - De Morgan duality:      ~(A | B) == (~A) & (~B)
"""
from __future__ import annotations

import ast
import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32: int = 0xFFFFFFFF


def ref_s_or_b32(s0: int, s1: int) -> int:
    """Pure reference oracle for RDNA2 32-bit scalar bitwise OR: (s0 | s1) & 0xFFFFFFFF."""
    return ((s0 & MASK32) | (s1 & MASK32)) & MASK32


def ref_scc(val: int) -> int:
    """RDNA2 Scalar Condition Code for bitwise SALU operations: 1 if val != 0 else 0."""
    return 1 if (val & MASK32) != 0 else 0


class SaluOrB32OracleTests(unittest.TestCase):
    """Reference mathematical oracle and architectural model tests for s_or_b32."""

    def test_truth_table_exhaustive(self):
        """Verify full 2-input 1-bit truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1."""
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1,
        }
        for (b0, b1), expected in truth_table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK32 if b0 else 0
                s1 = MASK32 if b1 else 0
                want = MASK32 if expected else 0
                got = ref_s_or_b32(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), expected)

    def test_algebraic_identities(self):
        """Test algebraic laws: identity, annihilator, idempotence, commutativity, associativity."""
        test_values = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x87654321,
            0xAAAAAAAA,
            0x55555555,
            0x80000000,
            0x00000001,
        ]
        for a in test_values:
            with self.subTest(identity=hex(a)):
                self.assertEqual(ref_s_or_b32(a, 0), a)
                self.assertEqual(ref_s_or_b32(0, a), a)
                self.assertEqual(ref_s_or_b32(a, MASK32), MASK32)
                self.assertEqual(ref_s_or_b32(MASK32, a), MASK32)
                self.assertEqual(ref_s_or_b32(a, a), a)

        for a in test_values:
            for b in test_values:
                with self.subTest(commutative=(hex(a), hex(b))):
                    self.assertEqual(ref_s_or_b32(a, b), ref_s_or_b32(b, a))

        for a in test_values[:4]:
            for b in test_values[:4]:
                for c in test_values[:4]:
                    with self.subTest(associative=(hex(a), hex(b), hex(c))):
                        left = ref_s_or_b32(ref_s_or_b32(a, b), c)
                        right = ref_s_or_b32(a, ref_s_or_b32(b, c))
                        self.assertEqual(left, right)

    def test_de_morgan_duality(self):
        """Test De Morgan duality: ~(A | B) == (~A) & (~B)."""
        rng = random.Random(0x028332)
        for _ in range(64):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            or_ab = ref_s_or_b32(a, b)
            not_or_ab = (~or_ab) & MASK32
            not_a = (~a) & MASK32
            not_b = (~b) & MASK32
            and_not_a_not_b = (not_a & not_b) & MASK32
            self.assertEqual(not_or_ab, and_not_a_not_b)

    def test_boundary_bit_patterns(self):
        """Test single-bit walking, byte boundaries, and alternating bit patterns."""
        for i in range(32):
            bit = 1 << i
            self.assertEqual(ref_s_or_b32(bit, 0), bit)
            self.assertEqual(ref_s_or_b32(bit, bit), bit)
            self.assertEqual(ref_scc(ref_s_or_b32(bit, 0)), 1)

        for i in range(31):
            b0 = 1 << i
            b1 = 1 << (i + 1)
            combined = ref_s_or_b32(b0, b1)
            self.assertEqual(combined, b0 | b1)
            self.assertEqual(ref_scc(combined), 1)

    def test_scc_oracle_mapping(self):
        """Verify SCC mapping across exact zero and non-zero boundary transitions."""
        self.assertEqual(ref_scc(0x00000000), 0)
        self.assertEqual(ref_scc(0x00000001), 1)
        self.assertEqual(ref_scc(0x80000000), 1)
        self.assertEqual(ref_scc(0x7FFFFFFF), 1)
        self.assertEqual(ref_scc(0xFFFFFFFF), 1)
        self.assertEqual(ref_scc(ref_s_or_b32(0, 0)), 0)

    def test_translator_smoke(self):
        """Smoke test verifying tools/rdna2spirv.py translates s_or_b32 correctly."""
        lines = [
            "s_mov_b32 s0, 0x12",
            "s_mov_b32 s1, 0x20",
            "s_or_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_or_smoke", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x32)
        self.assertEqual(out[1], 1)

if __name__ == "__main__":
    unittest.main()
