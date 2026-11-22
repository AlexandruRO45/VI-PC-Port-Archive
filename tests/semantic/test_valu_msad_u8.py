from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_msad_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_msad_u8 dst, src0, src1, src2

Operation:
  Performs 4-channel masked sum of absolute differences across packed 8-bit
  unsigned byte lanes within 32-bit dwords, with 32-bit unsigned accumulation:
    For each byte lane i in {0, 1, 2, 3}:
      s0_i = (src0 >> (8 * i)) & 0xFF
      s1_i = (src1 >> (8 * i)) & 0xFF
      diff_i = 0 if s1_i == 0 else |s0_i - s1_i|
    sad_sum = diff_0 + diff_1 + diff_2 + diff_3
    dst = (src2 + sad_sum) mod 2^32

Mathematical and Architectural Properties:
  - Masking semantics:
    - Byte lane i is masked out (omitted) if and only if reference byte s1_i == 0.
    - If s1_i == 0, diff_i is strictly 0 regardless of s0_i (even if s0_i == 0xFF).
    - If s1_i != 0, diff_i is the unsigned absolute difference |s0_i - s1_i| in [0, 255].
  - Asymmetry / Non-commutativity:
    - Unlike unmasked v_sad_u8, v_msad_u8 is NOT commutative with respect to src0 and src1:
      the mask is dictated solely by src1.
  - Range and bounds:
    - Each unmasked byte difference is in [0, 255].
    - Maximum sad_sum across 4 lanes is 4 * 255 = 1020 (0x3FC).
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
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
BYTE_SHIFTS: Tuple[int, int, int, int] = (0, 8, 16, 24)
MAX_SAD_SUM: int = 4 * 255  # 1020 (0x3FC)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def unpack_bytes_u32(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into 4 byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit unsigned word."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_msad_byte(s0_byte: int, s1_byte: int) -> int:
    """Compute single byte-lane masked absolute difference.

    If s1_byte is 0, the lane is masked out and contributes 0.
    Otherwise, returns the unsigned absolute difference |s0_byte - s1_byte|.
    """
    b0 = s0_byte & MASK8
    b1 = s1_byte & MASK8
    if b1 == 0:
        return 0
    return (b0 - b1) if b0 >= b1 else (b1 - b0)


def oracle_v_msad_u8(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u8.

    Unpacks 4 byte lanes from src0 and src1, computes masked absolute differences
    conditioned on src1 byte non-zero, sums the differences, and adds accumulator src2.
    Result wraps modulo 2^32.
    """
    s0_lanes = unpack_bytes_u32(src0)
    s1_lanes = unpack_bytes_u32(src1)

    diff0 = ref_msad_byte(s0_lanes[0], s1_lanes[0])
    diff1 = ref_msad_byte(s0_lanes[1], s1_lanes[1])
    diff2 = ref_msad_byte(s0_lanes[2], s1_lanes[2])
    diff3 = ref_msad_byte(s0_lanes[3], s1_lanes[3])

    sad_sum = diff0 + diff1 + diff2 + diff3
    acc = src2 & MASK32
    return (acc + sad_sum) & MASK32


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMsadU8ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_msad_u8 architectural bit model and reference oracle."""

    def test_byte_unpack_pack_roundtrip(self) -> None:
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55A3C7E,
            0x01020304,
            0x80402010,
        ]
        for pat in patterns:
            with self.subTest(pattern=hex(pat)):
                b0, b1, b2, b3 = unpack_bytes_u32(pat)
                repacked = pack_bytes_u32(b0, b1, b2, b3)
                self.assertEqual(repacked, pat)

    def test_all_lanes_masked_when_s1_is_zero(self) -> None:
        # When src1 is 0, all 4 byte lanes are masked out -> diff = 0 regardless of src0
        for s0 in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x80808080]:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(oracle_v_msad_u8(s0, 0x00000000, 0), 0)
                self.assertEqual(oracle_v_msad_u8(s0, 0x00000000, 100), 100)
                self.assertEqual(oracle_v_msad_u8(s0, 0x00000000, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_all_lanes_unmasked_known_values(self) -> None:
        # s0 = [10, 20, 30, 40], s1 = [15, 20, 25, 50], s2 = 100
        # diffs: |10-15|=5, |20-20|=0, |30-25|=5, |40-50|=10 -> sum = 20
        # result = 100 + 20 = 120
        s0 = pack_bytes_u32(10, 20, 30, 40)
        s1 = pack_bytes_u32(15, 20, 25, 50)
        self.assertEqual(oracle_v_msad_u8(s0, s1, 100), 120)

    def test_partial_masking_selective_skips(self) -> None:
        # S0 = [0xFF, 0x20, 0xFF, 0x40]
        # S1 = [0x00, 0x20, 0x00, 0x30]  (lanes 0 and 2 masked out)
        # diffs: lane 0: masked (0), lane 1: |0x20 - 0x20|=0, lane 2: masked (0), lane 3: |0x40 - 0x30|=0x10 (16)
        # Total sad_sum = 16
        s0 = 0x40FF20FF
        s1 = 0x30002000
        self.assertEqual(oracle_v_msad_u8(s0, s1, 0), 16)
        self.assertEqual(oracle_v_msad_u8(s0, s1, 1000), 1016)

    def test_masking_asymmetry_non_commutativity(self) -> None:
        # When zeros appear, swapping s0 and s1 gives different results:
        # Case 1: s0 has 0, s1 has non-zero -> S1 != 0, so diff = |0 - X| = X
        # Case 2: s0 has non-zero, s1 has 0 -> S1 == 0, so diff = 0 (masked!)
        s0 = pack_bytes_u32(0, 10, 0, 20)
        s1 = pack_bytes_u32(10, 0, 20, 0)
        res_0_1 = oracle_v_msad_u8(s0, s1, 0)
        res_1_0 = oracle_v_msad_u8(s1, s0, 0)

        # In res_0_1: s1 has [10, 0, 20, 0] -> lanes 0, 2 active (|0-10|=10, |0-20|=20) -> sum = 30
        self.assertEqual(res_0_1, 30)
        # In res_1_0: s0 in s1 position has [0, 10, 0, 20] -> lanes 1, 3 active (|0-10|=10, |0-20|=20) -> sum = 30
        # If we pick asymmetric values:
        s_a = pack_bytes_u32(50, 0, 0, 0)
        s_b = pack_bytes_u32(0, 50, 0, 0)
        # v_msad(s_a, s_b, 0): s_b has lane 1 = 50 != 0 -> |0 - 50| = 50. Total = 50
        # v_msad(s_b, s_a, 0): s_a has lane 0 = 50 != 0 -> |0 - 50| = 50.
        # Now asymmetrical:
        s_x = pack_bytes_u32(100, 200, 0, 0)
        s_y = pack_bytes_u32(0, 50, 0, 0)
        # msad(x, y, 0): y has lane 0 = 0 (masked), lane 1 = 50 (|200-50|=150) -> sum = 150
        # msad(y, x, 0): x has lane 0 = 100 (|0-100|=100), lane 1 = 200 (|50-200|=150) -> sum = 250
        self.assertEqual(oracle_v_msad_u8(s_x, s_y, 0), 150)
        self.assertEqual(oracle_v_msad_u8(s_y, s_x, 0), 250)
        self.assertNotEqual(oracle_v_msad_u8(s_x, s_y, 0), oracle_v_msad_u8(s_y, s_x, 0))

    def test_maximum_delta_accumulation(self) -> None:
        # Max delta: all 4 lanes max difference: |0xFF - 0x00| is not allowed for s1 (s1=0 masks it),
        # but |0x00 - 0xFF| = 255, with s1 = 0xFF!
        # Or |0xFF - 0x01| = 254 with s1 = 0x01.
        # Max difference per lane is |0x00 - 0xFF| = 255 when s0 = 0, s1 = 0xFF.
        # Across 4 lanes: 4 * 255 = 1020 (0x3FC).
        s0_zero = 0x00000000
        s1_max = 0xFFFFFFFF
        sad_max = oracle_v_msad_u8(s0_zero, s1_max, 0)
        self.assertEqual(sad_max, MAX_SAD_SUM)
        self.assertEqual(sad_max, 1020)

    def test_accumulator_32bit_wraparound(self) -> None:
        # When acc + diff exceeds 0xFFFFFFFF, wraps modulo 2^32
        s0 = 0x00000000
        s1 = 0xFFFFFFFF  # diff = 1020
        # acc = 0xFFFFFFFF -> (0xFFFFFFFF + 1020) mod 2^32 = 1019
        self.assertEqual(oracle_v_msad_u8(s0, s1, 0xFFFFFFFF), 1019)

        # Exact wrap boundary: acc = 0xFFFF_FC04 (4294966276) + 1020 = 0x1_0000_0000 -> 0
        acc_exact = 0xFFFF_FC04
        self.assertEqual(oracle_v_msad_u8(s0, s1, acc_exact), 0)

    def test_identity_when_equal_and_nonzero(self) -> None:
        # When src0 == src1 and all bytes != 0, diff = 0 for all lanes, so result = src2
        val = 0x01020304
        self.assertEqual(oracle_v_msad_u8(val, val, 0x12345678), 0x12345678)
        self.assertEqual(oracle_v_msad_u8(0xFFFFFFFF, 0xFFFFFFFF, 42), 42)


if __name__ == "__main__":
    unittest.main()
