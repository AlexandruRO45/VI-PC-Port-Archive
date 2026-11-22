from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_msad_u16.

AMD RDNA VALU Instruction Set Architecture:
  v_msad_u16 dst, src0, src1, src2

Operation:
  Performs 2-channel masked sum of absolute differences across packed 16-bit
  unsigned half-word lanes within 32-bit dwords, with 32-bit unsigned accumulation:
    For each half-word lane i in {0, 1}:
      s0_i = (src0 >> (16 * i)) & 0xFFFF
      s1_i = (src1 >> (16 * i)) & 0xFFFF
      diff_i = 0 if s1_i == 0 else |s0_i - s1_i|
    sad_sum = diff_0 + diff_1
    dst = (src2 + sad_sum) mod 2^32

Mathematical and Architectural Properties:
  - Masking semantics:
    - Half-word lane i is masked out (omitted) if and only if reference half-word s1_i == 0.
    - If s1_i == 0, diff_i is strictly 0 regardless of s0_i (even if s0_i == 0xFFFF).
    - If s1_i != 0, diff_i is the unsigned absolute difference |s0_i - s1_i| in [0, 65535].
  - Asymmetry / Non-commutativity:
    - Unlike unmasked v_sad_u16, v_msad_u16 is NOT commutative with respect to src0 and src1:
      the mask is dictated solely by src1.
  - Range and bounds:
    - Each unmasked 16-bit difference is in [0, 65535] (0xFFFF).
    - Maximum sad_sum across 2 lanes is 2 * 65535 = 131070 (0x1FFFE).
    - Minimum sad_sum is 0 (all lanes masked, or all active lanes equal).
  - Accumulator wrapping:
    - src2 is an unsigned 32-bit integer.
    - The addition (src2 + sad_sum) wraps modulo 2^32 with standard unsigned arithmetic.
  - Clamp modifier:
    - In AMD RDNA2 ISA / Leonida translator contract, clamp modifier is unsupported
      and rejected rather than guessed.
"""

from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
HALF_SHIFTS: Tuple[int, int] = (0, 16)
MAX_SAD_SUM_U16: int = 2 * 65535  # 131070 (0x1FFFE)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def unpack_halves_u32(val: int) -> Tuple[int, int]:
    """Unpack a 32-bit unsigned word into 2 16-bit half-word lanes (h0, h1)."""
    val &= MASK32
    return (
        val & MASK16,
        (val >> 16) & MASK16,
    )


def pack_halves_u32(h0: int, h1: int) -> int:
    """Pack two 16-bit unsigned integers into a 32-bit unsigned word."""
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def ref_msad_half(s0_half: int, s1_half: int) -> int:
    """Compute single 16-bit half-word lane masked absolute difference.

    If s1_half is 0, the lane is masked out and contributes 0.
    Otherwise, returns the unsigned absolute difference |s0_half - s1_half|.
    """
    h0 = s0_half & MASK16
    h1 = s1_half & MASK16
    if h1 == 0:
        return 0
    return (h0 - h1) if h0 >= h1 else (h1 - h0)


def oracle_v_msad_u16(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u16.

    Unpacks 2 half-word lanes from src0 and src1, computes masked absolute differences
    conditioned on src1 half-word non-zero, sums the differences, and adds accumulator src2.
    Result wraps modulo 2^32.
    """
    s0_lanes = unpack_halves_u32(src0)
    s1_lanes = unpack_halves_u32(src1)

    diff0 = ref_msad_half(s0_lanes[0], s1_lanes[0])
    diff1 = ref_msad_half(s0_lanes[1], s1_lanes[1])

    sad_sum = diff0 + diff1
    acc = src2 & MASK32
    return (acc + sad_sum) & MASK32


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMsadU16ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_msad_u16 architectural bit model and reference oracle."""

    def test_half_unpack_pack_roundtrip(self) -> None:
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55A3C7E,
            0x00010002,
            0x80004000,
        ]
        for pat in patterns:
            with self.subTest(pattern=hex(pat)):
                h0, h1 = unpack_halves_u32(pat)
                repacked = pack_halves_u32(h0, h1)
                self.assertEqual(repacked, pat)

    def test_all_lanes_masked_when_s1_is_zero(self) -> None:
        # When src1 is 0, both 16-bit half lanes are masked out -> diff = 0 regardless of src0
        for s0 in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x80008000]:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(oracle_v_msad_u16(s0, 0x00000000, 0), 0)
                self.assertEqual(oracle_v_msad_u16(s0, 0x00000000, 500), 500)
                self.assertEqual(oracle_v_msad_u16(s0, 0x00000000, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_all_lanes_unmasked_known_values(self) -> None:
        # s0 = [1000, 2000], s1 = [1500, 1000], s2 = 100
        # diffs: |1000-1500|=500, |2000-1000|=1000 -> sum = 1500
        # result = 100 + 1500 = 1600
        s0 = pack_halves_u32(1000, 2000)
        s1 = pack_halves_u32(1500, 1000)
        self.assertEqual(oracle_v_msad_u16(s0, s1, 100), 1600)

    def test_partial_masking_selective_skips(self) -> None:
        # S0 = [0xFFFF, 0x4000]
        # S1 = [0x0000, 0x3000]  (lane 0 masked out, lane 1 active)
        # diffs: lane 0: masked (0), lane 1: |0x4000 - 0x3000| = 0x1000 (4096)
        # Total sad_sum = 4096
        s0 = pack_halves_u32(0xFFFF, 0x4000)
        s1 = pack_halves_u32(0x0000, 0x3000)
        self.assertEqual(oracle_v_msad_u16(s0, s1, 0), 4096)
        self.assertEqual(oracle_v_msad_u16(s0, s1, 10000), 14096)

    def test_masking_asymmetry_non_commutativity(self) -> None:
        # Half 0: s0 has 1000, s1 has 0 -> masked in (s0, s1)
        # Half 1: s0 has 0, s1 has 2000 -> active in (s0, s1) (|0 - 2000| = 2000)
        s0 = pack_halves_u32(1000, 0)
        s1 = pack_halves_u32(0, 2000)

        res_0_1 = oracle_v_msad_u16(s0, s1, 0)
        res_1_0 = oracle_v_msad_u16(s1, s0, 0)

        # In res_0_1: s1 has [0, 2000] -> lane 0 masked, lane 1 has 2000 -> sum = 2000
        self.assertEqual(res_0_1, 2000)
        # In res_1_0: s0 in s1 position has [1000, 0] -> lane 0 has 1000, lane 1 masked -> sum = 1000
        self.assertEqual(res_1_0, 1000)
        self.assertNotEqual(res_0_1, res_1_0)

    def test_maximum_delta_accumulation(self) -> None:
        # Max delta: both 16-bit lanes have max difference: |0x0000 - 0xFFFF| = 65535, with s1 = 0xFFFF!
        # Across 2 lanes: 2 * 65535 = 131070 (0x1FFFE).
        s0_zero = 0x00000000
        s1_max = 0xFFFFFFFF
        sad_max = oracle_v_msad_u16(s0_zero, s1_max, 0)
        self.assertEqual(sad_max, MAX_SAD_SUM_U16)
        self.assertEqual(sad_max, 131070)

    def test_accumulator_32bit_wraparound(self) -> None:
        # When acc + diff exceeds 0xFFFFFFFF, wraps modulo 2^32
        s0 = 0x00000000
        s1 = 0xFFFFFFFF  # diff = 131070
        # acc = 0xFFFFFFFF -> (0xFFFFFFFF + 131070) mod 2^32 = 131069
        self.assertEqual(oracle_v_msad_u16(s0, s1, 0xFFFFFFFF), 131069)

        # Exact wrap boundary: acc = 0xFFFE_0002 (4294836226) + 131070 (0x1FFFE) = 0x1_0000_0000 -> 0
        acc_exact = 0xFFFE_0002
        self.assertEqual(oracle_v_msad_u16(s0, s1, acc_exact), 0)

    def test_identity_when_equal_and_nonzero(self) -> None:
        # When src0 == src1 and both half-words != 0, diff = 0 for both lanes, so result = src2
        val = 0x12345678
        self.assertEqual(oracle_v_msad_u16(val, val, 0x12345678), 0x12345678)
        self.assertEqual(oracle_v_msad_u16(0xFFFFFFFF, 0xFFFFFFFF, 42), 42)


if __name__ == "__main__":
    unittest.main()
