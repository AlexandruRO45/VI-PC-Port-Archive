from __future__ import annotations

"""Semantic reference tests and translator-backed validation for AMD RDNA2 s_add_u32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, s_add_u32 computes the 32-bit unsigned addition
  of two scalar source operands with carry-out generation:
      D.u = (S0.u + S1.u) mod 2^32
      SCC = (S0.u + S1.u >= 2^32) ? 1 : 0

  - Format: SOP2 (Scalar Operation with 2 inputs, 1 destination).
  - Operands:
      S0 and S1 can be scalar general-purpose registers (SGPR), inline constants
      (-16..64, float constants), or 32-bit literal constants.
      D is a 32-bit destination SGPR.
  - Condition code register (SCC):
      Updated with the 1-bit unsigned carry-out: 1 if unsigned sum overflows 2^32,
      0 otherwise. Prior SCC state does not affect the calculation.
  - SPIR-V translation (tools/rdna2spirv.py):
      Lowers to OpIAddCarry %umul %s0 %s1.
      Member 0: 32-bit wrapped sum stored to destination SGPR.
      Member 1: 32-bit carry-out flag stored to %scc variable.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

from dataclasses import dataclass
import enum
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


# --- Section 1: Mathematical Reference Oracle ---

def ref_s_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for RDNA2 s_add_u32.

    Returns (wrapped_sum, scc_carry).
    """
    a = s0 & MASK32
    b = s1 & MASK32
    full = a + b
    wrapped = full & MASK32
    carry = 1 if full >= (1 << 32) else 0
    return wrapped, carry


def oracle_add_u32_analysis(s0: int, s1: int) -> dict[str, int | bool]:
    """Detailed architectural breakdown of s_add_u32 execution."""
    a = s0 & MASK32
    b = s1 & MASK32
    full = a + b
    has_carry = full >= (1 << 32)
    return {
        "s0": a,
        "s1": b,
        "full_sum": full,
        "carry": has_carry,
        "scc": 1 if has_carry else 0,
        "wrapped_result": full & MASK32,
    }


class SaluAddU32ReferenceOracleTests(unittest.TestCase):
    """Commit 1: Reference oracle tests for RDNA2 s_add_u32."""

    def test_oracle_zero_identity(self) -> None:
        """Adding zero to any 32-bit value preserves the value and sets SCC=0."""
        cases = [0, 1, 42, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                res0, scc0 = ref_s_add_u32(val, 0)
                self.assertEqual(res0, val)
                self.assertEqual(scc0, 0)

                res1, scc1 = ref_s_add_u32(0, val)
                self.assertEqual(res1, val)
                self.assertEqual(scc1, 0)

    def test_oracle_basic_addition_no_carry(self) -> None:
        """Sums strictly below 2^32 must produce exact integer sum with SCC=0."""
        cases = [
            (10, 20, 30),
            (0x1000, 0x2000, 0x3000),
            (0x12345678, 0x01020304, 0x1336597C),
            (0x7FFFFFFE, 1, 0x7FFFFFFF),
            (0x40000000, 0x3FFFFFFF, 0x7FFFFFFF),
            (0x7FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFE),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
        ]
        for a, b, expected_sum in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_u32(a, b)
                self.assertEqual(res, expected_sum)
                self.assertEqual(scc, 0)

    def test_oracle_carry_generation(self) -> None:
        """Sums at or above 2^32 must set SCC=1 and wrap modulo 2^32."""
        cases = [
            (0xFFFFFFFF, 1, 0, 1),
            (0xFFFFFFFF, 2, 1, 1),
            (0xFFFFFFFE, 2, 0, 1),
            (0x80000000, 0x80000000, 0, 1),
            (0x80000000, 0x80000001, 1, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 1),
            (0xC0000000, 0x50000000, 0x10000000, 1),
        ]
        for a, b, exp_res, exp_scc in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_u32(a, b)
                self.assertEqual(res, exp_res)
                self.assertEqual(scc, exp_scc)

    def test_oracle_commutativity(self) -> None:
        """s_add_u32 must be strictly commutative across all operand ranges."""
        pairs = [
            (0, 0),
            (0, 0xFFFFFFFF),
            (1, 0xFFFFFFFF),
            (0x12345678, 0x87654321),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xCAFEBABE, 0xDEADBEEF),
            (0xAAAAAAAA, 0x55555555),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                r1, s1 = ref_s_add_u32(a, b)
                r2, s2 = ref_s_add_u32(b, a)
                self.assertEqual(r1, r2)
                self.assertEqual(s1, s2)

    def test_oracle_analysis_breakdown(self) -> None:
        """Detailed analysis oracle must compute full sums and carry flags."""
        info = oracle_add_u32_analysis(0xFFFFFFFF, 1)
        self.assertEqual(info["s0"], 0xFFFFFFFF)
        self.assertEqual(info["s1"], 1)
        self.assertEqual(info["full_sum"], 0x100000000)
        self.assertTrue(info["carry"])
        self.assertEqual(info["scc"], 1)
        self.assertEqual(info["wrapped_result"], 0)

        info2 = oracle_add_u32_analysis(0x1234, 0x5678)
        self.assertFalse(info2["carry"])
        self.assertEqual(info2["scc"], 0)
        self.assertEqual(info2["wrapped_result"], 0x68AC)


if __name__ == "__main__":
    unittest.main()
