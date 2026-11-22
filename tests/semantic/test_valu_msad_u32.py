from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_msad_u32.

AMD RDNA VALU Instruction Set Architecture:
  v_msad_u32 dst, src0, src1, src2

Operation:
  Performs 32-bit masked sum of absolute difference with 32-bit unsigned accumulation:
    diff = 0 if src1 == 0 else |src0.u32 - src1.u32|
    dst = (src2.u32 + diff) mod 2^32

Mathematical and Architectural Properties:
  - Masking semantics:
    - Operation is masked out (omitted) if and only if reference word src1 == 0.
    - If src1 == 0, diff is strictly 0 regardless of src0 (even if src0 == 0xFFFFFFFF).
    - If src1 != 0, diff is the unsigned absolute difference |src0 - src1| in [0, 2^32 - 1].
  - Asymmetry / Non-commutativity:
    - Unlike unmasked v_sad_u32, v_msad_u32 is NOT commutative with respect to src0 and src1:
      the mask is dictated solely by src1.
  - Range and bounds:
    - Unmasked difference is in [0, 2^32 - 1] (0xFFFFFFFF).
    - Minimum diff is 0 (masked or equal operands).
  - Accumulator wrapping:
    - src2 is an unsigned 32-bit integer.
    - The addition (src2 + diff) wraps modulo 2^32 with standard unsigned arithmetic.
  - Clamp modifier:
    - In AMD RDNA2 ISA / Leonida translator contract, clamp modifier is unsupported
      and rejected rather than guessed.
"""

from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MAX_DIFF_U32: int = MASK32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def ref_abs_diff_u32(s0: int, s1: int) -> int:
    """Compute strictly unsigned 32-bit absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def oracle_v_msad_u32(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u32.

    Computes masked absolute difference conditioned on src1 non-zero,
    and adds accumulator src2. Result wraps modulo 2^32.
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    acc = src2 & MASK32

    if u1 == 0:
        diff = 0
    else:
        diff = (u0 - u1) if u0 >= u1 else (u1 - u0)

    return (acc + diff) & MASK32


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMsadU32ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_msad_u32 architectural bit model and reference oracle."""

    def test_masked_when_s1_is_zero(self) -> None:
        # When src1 is 0, masked out -> diff = 0 regardless of src0
        for s0 in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x80000000]:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(oracle_v_msad_u32(s0, 0, 0), 0)
                self.assertEqual(oracle_v_msad_u32(s0, 0, 500), 500)
                self.assertEqual(oracle_v_msad_u32(s0, 0, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_unmasked_known_values(self) -> None:
        # s0 = 1000, s1 = 1500, s2 = 100
        # diff = |1000 - 1500| = 500
        # result = 100 + 500 = 600
        self.assertEqual(oracle_v_msad_u32(1000, 1500, 100), 600)
        self.assertEqual(oracle_v_msad_u32(2000, 1000, 0), 1000)

    def test_masking_asymmetry_non_commutativity(self) -> None:
        # s0 has 1000, s1 has 0 -> masked (result = 0)
        # s0 has 0, s1 has 1000 -> active (|0 - 1000| = 1000)
        res_s0_s1 = oracle_v_msad_u32(1000, 0, 0)
        res_s1_s0 = oracle_v_msad_u32(0, 1000, 0)

        self.assertEqual(res_s0_s1, 0)
        self.assertEqual(res_s1_s0, 1000)
        self.assertNotEqual(res_s0_s1, res_s1_s0)

    def test_maximum_delta_accumulation(self) -> None:
        # Max delta: |0 - 0xFFFFFFFF| = 0xFFFFFFFF with s1 = 0xFFFFFFFF!
        sad_max = oracle_v_msad_u32(0, 0xFFFFFFFF, 0)
        self.assertEqual(sad_max, MAX_DIFF_U32)
        self.assertEqual(sad_max, 0xFFFFFFFF)

    def test_accumulator_32bit_wraparound(self) -> None:
        # diff = 0xFFFFFFFF, acc = 1 -> wraps to 0
        self.assertEqual(oracle_v_msad_u32(0, 0xFFFFFFFF, 1), 0)
        # diff = 0xFFFFFFFF, acc = 2 -> wraps to 1
        self.assertEqual(oracle_v_msad_u32(0, 0xFFFFFFFF, 2), 1)

    def test_identity_when_equal_and_nonzero(self) -> None:
        # When src0 == src1 and src1 != 0, diff = 0, so result = src2
        val = 0x12345678
        self.assertEqual(oracle_v_msad_u32(val, val, 0xABCDEF01), 0xABCDEF01)
        self.assertEqual(oracle_v_msad_u32(0xFFFFFFFF, 0xFFFFFFFF, 42), 42)


if __name__ == "__main__":
    unittest.main()
