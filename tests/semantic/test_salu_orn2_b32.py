"""Semantic interpreter tests and reference hardware-matching oracle for RDNA2 s_orn2_b32.

Models AMD RDNA2 / GFX10 Scalar ALU (SALU) SOP2 32-bit bitwise OR-NOT2 operation:
  s_orn2_b32 sdst, ssrc0, ssrc1

RDNA2 SALU Bitwise Architecture & Semantics:
  1. Instruction Operands:
     - sdst: Destination SGPR receiving the 32-bit bitwise result.
     - ssrc0: First 32-bit scalar source operand (SGPR, literal, or inline constant).
     - ssrc1: Second 32-bit scalar source operand (SGPR, literal, or inline constant).
  2. Bitwise Logic:
     sdst = (ssrc0 | (~ssrc1)) & 0xFFFFFFFF
     Note that ssrc1 is bitwise inverted (NOT) prior to the bitwise OR with ssrc0.
  3. Scalar Condition Code (SCC):
     SCC is updated based on the 32-bit result:
     SCC = 1 if (sdst != 0) else 0.
  4. Non-commutativity & Algebraic Properties:
     - Non-commutative: (A | ~B) != (B | ~A) in general.
     - Self-identity: (A | ~A) == 0xFFFFFFFF, with SCC = 1 for all A.
     - Zero-absorption: (0 | ~B) == ~B.
     - Maximum element: (A | ~0) == 0xFFFFFFFF; (A | ~0xFFFFFFFF) == A.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import spirv_mini_interp as si
    import support
    TRANSLATOR_ERROR = None
except Exception as _err:
    rdna2spirv = None  # type: ignore
    si = None  # type: ignore
    support = None  # type: ignore
    TRANSLATOR_ERROR = str(_err)

MASK32: int = 0xFFFFFFFF

def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def ref_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Pure bitwise reference oracle for s_orn2_b32: D = S0 | ~S1, SCC = (D != 0).

    Returns tuple (sdst, scc) normalized to 32-bit unsigned integers.
    """
    s0_u = u32(s0)
    s1_u = u32(s1)
    res = u32(s0_u | (~s1_u))
    scc = 1 if res != 0 else 0
    return res, scc


class TestSaluOrn2B32Oracle(unittest.TestCase):
    """Unit tests verifying the reference mathematical oracle against RDNA2 hardware semantics."""

    def test_oracle_truth_table_basic(self) -> None:
        # 0 | ~0 = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0x00000000, 0x00000000), (0xFFFFFFFF, 1))
        # 0 | ~0xFFFFFFFF = 0x00000000, scc = 0 (only way to produce 0)
        self.assertEqual(ref_orn2_b32(0x00000000, 0xFFFFFFFF), (0x00000000, 0))
        # 0xFFFFFFFF | ~0 = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0xFFFFFFFF, 0x00000000), (0xFFFFFFFF, 1))
        # 0xFFFFFFFF | ~0xFFFFFFFF = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 1))

    def test_oracle_self_identity(self) -> None:
        # For any word X, X | ~X == 0xFFFFFFFF, SCC == 1
        test_vectors = [
            0x00000000, 0xFFFFFFFF, 0x12345678, 0xAAAAAAAA,
            0x55555555, 0x80000000, 0x7FFFFFFF, 0x00000001,
            0xCAFEBABE, 0xDEADBEEF,
        ]
        for val in test_vectors:
            res, scc = ref_orn2_b32(val, val)
            self.assertEqual(res, 0xFFFFFFFF, f"Self-identity failed for {val:#x}")
            self.assertEqual(scc, 1, f"SCC failed for {val:#x}")

    def test_oracle_non_commutativity(self) -> None:
        # A | ~B != B | ~A when A != B
        a = 0x0F0F0F0F
        b = 0x33333333
        res_ab, _ = ref_orn2_b32(a, b)  # 0x0F0F0F0F | 0xCCCCCCCC = 0xCFCFCFCF
        res_ba, _ = ref_orn2_b32(b, a)  # 0x33333333 | 0xF0F0F0F0 = 0xF3F3F3F3
        self.assertEqual(res_ab, 0xCFCFCFCF)
        self.assertEqual(res_ba, 0xF3F3F3F3)
        self.assertNotEqual(res_ab, res_ba)

    def test_oracle_bit_decomposition(self) -> None:
        # Check all 32 independent bit positions
        for bit in range(32):
            mask = 1 << bit
            # s0=0, s1=0: bit in ~s1 is 1 -> res has 1
            res, _ = ref_orn2_b32(0, 0)
            self.assertTrue(bool(res & mask))
            # s0=0, s1=mask: bit in ~s1 is 0 -> res has 0
            res, _ = ref_orn2_b32(0, mask)
            self.assertFalse(bool(res & mask))
            # s0=mask, s1=mask: bit in s0 is 1 -> res has 1
            res, _ = ref_orn2_b32(mask, mask)
            self.assertTrue(bool(res & mask))
            # s0=mask, s1=0: both are 1 -> res has 1
            res, _ = ref_orn2_b32(mask, 0)
            self.assertTrue(bool(res & mask))

    def test_oracle_exhaustive_nibbles(self) -> None:
        # Exhaustive verification across all 16x16 4-bit nibbles
        for n0 in range(16):
            for n1 in range(16):
                expected_nibble = (n0 | (~n1 & 0xF)) & 0xF
                expected_res = sum(expected_nibble << (shift * 4) for shift in range(8))
                res, _ = ref_orn2_b32(
                    sum(n0 << (shift * 4) for shift in range(8)),
                    sum(n1 << (shift * 4) for shift in range(8)),
                )
                self.assertEqual(res, expected_res)

if __name__ == "__main__":
    unittest.main()
