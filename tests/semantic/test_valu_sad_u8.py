from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_sad_u8, v_sad_hi_u8, and v_msad_u8.

AMD RDNA / GFX10 Vector ALU Instruction Set Architecture:
- v_sad_u8 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Computes the sum of absolute differences between four pairs of packed 8-bit unsigned
    integers in src0 and src1, and adds the result to the 32-bit accumulator src2:
        diff0 = |src0[7:0] - src1[7:0]|
        diff1 = |src0[15:8] - src1[15:8]|
        diff2 = |src0[23:16] - src1[23:16]|
        diff3 = |src0[31:24] - src1[31:24]|
        sad = diff0 + diff1 + diff2 + diff3
        dst = (src2 + sad) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

- v_sad_hi_u8 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Computes the 4-byte SAD, shifts the intermediate sum left by 16 bits (MSB alignment),
    and accumulates into src2:
        sad = sum_{i=0..3} |src0.byte[i] - src1.byte[i]|
        dst = (src2 + (sad << 16)) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

- v_msad_u8 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Masked sum of absolute differences across 4 byte lanes with src1 serving as template/mask:
    If src1.byte[i] == 0, that byte lane is masked out (difference is omitted / 0):
        sad = sum_{i=0..3, src1.byte[i] != 0} |src0.byte[i] - src1.byte[i]|
        dst = (src2 + sad) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

import math
import struct
import unittest
from typing import List, Sequence, Tuple, Union

# Architectural bitmasks and saturation limits
MASK8 = 0xFF
MASK16 = 0xFFFF
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

CLAMP_MAX_U32 = MASK32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def extract_byte(word32: int, byte_idx: int) -> int:
    """Extract unsigned 8-bit byte at byte_idx (0..3) from a 32-bit word."""
    return (word32 >> (8 * (byte_idx & 3))) & MASK8


def pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four unsigned 8-bit bytes into a 32-bit word [b3, b2, b1, b0]."""
    return ((b0 & MASK8) |
            ((b1 & MASK8) << 8) |
            ((b2 & MASK8) << 16) |
            ((b3 & MASK8) << 24))


def ref_byte_abs_diff(b0: int, b1: int) -> int:
    """Calculate strictly unsigned 8-bit absolute difference |b0.u8 - b1.u8|."""
    u0 = b0 & MASK8
    u1 = b1 & MASK8
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u8_sum(s0: int, s1: int) -> int:
    """Compute 4-element byte-wise sum of absolute differences between two 32-bit words.

    Result is bounded in [0, 4 * 255] = [0, 1020].
    """
    total = 0
    for i in range(4):
        total += ref_byte_abs_diff(extract_byte(s0, i), extract_byte(s1, i))
    return total


def ref_msad_u8_sum(s0: int, s1: int) -> int:
    """Compute masked 4-element byte-wise sum of absolute differences.

    Bytes where s1.byte[i] == 0 are masked out (difference is omitted).
    """
    total = 0
    for i in range(4):
        b1 = extract_byte(s1, i)
        if b1 == 0:
            continue
        total += ref_byte_abs_diff(extract_byte(s0, i), b1)
    return total


def ref_sad_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_u8.

    dst = (s2 + sum_{i=0..3} |s0.byte[i] - s1.byte[i]|) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_sad_u8_sum(s0, s1)
    acc = s2 & MASK32
    full_sum = acc + sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_sad_hi_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_hi_u8.

    dst = (s2 + (sum_{i=0..3} |s0.byte[i] - s1.byte[i]| << 16)) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_sad_u8_sum(s0, s1)
    acc = s2 & MASK32
    shifted_sad = (sad << 16)
    full_sum = acc + shifted_sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_msad_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_msad_u8.

    dst = (s2 + sum_{i=0..3, s1.byte[i] != 0} |s0.byte[i] - s1.byte[i]|) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_msad_u8_sum(s0, s1)
    acc = s2 & MASK32
    full_sum = acc + sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


# ==============================================================================
# Unit Tests: Reference Mathematical Oracles
# ==============================================================================

class TestValuSadU8Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracles of v_sad_u8, v_sad_hi_u8, and v_msad_u8."""

    def test_byte_abs_diff_bounds_and_commutativity(self) -> None:
        pairs = [
            (0, 0),
            (0, 255),
            (255, 0),
            (128, 128),
            (10, 250),
            (250, 10),
            (127, 128),
        ]
        for b0, b1 in pairs:
            diff = ref_byte_abs_diff(b0, b1)
            self.assertGreaterEqual(diff, 0)
            self.assertLessEqual(diff, 255)
            self.assertEqual(diff, ref_byte_abs_diff(b1, b0))

    def test_unsigned_vs_signed_byte_difference(self) -> None:
        # 0x00 vs 0xFF:
        # In unsigned math: |0 - 255| = 255
        # In signed 8-bit math: 0xFF is -1, |0 - (-1)| = 1
        self.assertEqual(ref_byte_abs_diff(0x00, 0xFF), 255)
        self.assertEqual(ref_byte_abs_diff(0xFF, 0x00), 255)

    def test_sad_u8_sum_all_zero_and_all_max(self) -> None:
        # Identical words -> 0
        self.assertEqual(ref_sad_u8_sum(0x1234_5678, 0x1234_5678), 0)
        self.assertEqual(ref_sad_u8_sum(0x0000_0000, 0x0000_0000), 0)
        self.assertEqual(ref_sad_u8_sum(0xFFFF_FFFF, 0xFFFF_FFFF), 0)

        # Max possible difference: each byte diff is 255 -> 4 * 255 = 1020 (0x3FC)
        self.assertEqual(ref_sad_u8_sum(0x0000_0000, 0xFFFF_FFFF), 1020)
        self.assertEqual(ref_sad_u8_sum(0xFFFF_FFFF, 0x0000_0000), 1020)

    def test_sad_u8_sum_individual_byte_positions(self) -> None:
        # Test each byte lane independently
        s0 = pack_bytes(10, 20, 30, 40)
        s1 = pack_bytes(15, 20, 30, 40)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 5)

        s1 = pack_bytes(10, 25, 30, 40)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 5)

        s1 = pack_bytes(10, 20, 10, 40)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 20)

        s1 = pack_bytes(10, 20, 30, 0)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 40)

    def test_sad_u8_accumulation_and_identity(self) -> None:
        s0 = pack_bytes(1, 2, 3, 4)
        s1 = pack_bytes(5, 6, 7, 8)
        # diffs: |1-5|=4, |2-6|=4, |3-7|=4, |4-8|=4 -> sum = 16
        self.assertEqual(ref_sad_u8_sum(s0, s1), 16)
        self.assertEqual(ref_sad_u8(s0, s1, 100), 116)
        self.assertEqual(ref_sad_u8(s0, s1, 0), 16)

    def test_sad_u8_wraparound_modulo_2_32(self) -> None:
        # s2 = 0xFFFFFFFF, sad = 1 -> wraps to 0
        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 1)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32, clamp=False), 0)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32 - 1, clamp=False), MASK32)

    def test_sad_u8_clamp_saturation(self) -> None:
        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32, clamp=True), MASK32)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32 - 1, clamp=True), MASK32)
        self.assertEqual(ref_sad_u8(s0, s1, 50, clamp=True), 51)

    def test_sad_hi_u8_upper_word_shift_and_accumulation(self) -> None:
        s0 = pack_bytes(1, 2, 3, 4)
        s1 = pack_bytes(2, 3, 4, 5)
        # diffs: 1+1+1+1 = 4. Shifted: 4 << 16 = 0x0004_0000 = 262144
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0), 0x0004_0000)
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0x1234), 0x0004_1234)

    def test_sad_hi_u8_clamping_saturation(self) -> None:
        # Max sad = 1020 (0x3FC) -> shifted = 0x03FC_0000
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000
        # s2 = 0xFC05_0000 -> 0xFC05_0000 + 0x03FC_0000 = 0x1_0001_0000 (overflows 32-bit)
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0xFC05_0000, clamp=False), 0x0001_0000)
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0xFC05_0000, clamp=True), MASK32)

    def test_msad_u8_zero_masking_skips_inactive_bytes(self) -> None:
        # s1 has zeros in bytes 0 and 2
        s0 = pack_bytes(100, 20, 200, 40)
        s1 = pack_bytes(0, 25, 0, 50)
        # byte 0: s1=0 -> skipped (diff 0)
        # byte 1: s1=25 -> |20 - 25| = 5
        # byte 2: s1=0 -> skipped (diff 0)
        # byte 3: s1=50 -> |40 - 50| = 10
        # sum = 15
        self.assertEqual(ref_msad_u8_sum(s0, s1), 15)
        self.assertEqual(ref_msad_u8(s0, s1, 100), 115)

        # Contrast with unmasked sad_u8:
        # byte 0 diff = 100, byte 2 diff = 200 -> total = 100 + 5 + 200 + 10 = 315
        self.assertEqual(ref_sad_u8_sum(s0, s1), 315)

    def test_msad_u8_all_zeros_mask_yields_zero_sad(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000  # all template bytes are 0
        self.assertEqual(ref_msad_u8_sum(s0, s1), 0)
        self.assertEqual(ref_msad_u8(s0, s1, 0x1234_5678), 0x1234_5678)

    def test_msad_u8_non_zero_reference_matching(self) -> None:
        # When no bytes in s1 are zero, msad equals sad
        s0 = pack_bytes(10, 20, 30, 40)
        s1 = pack_bytes(1, 2, 3, 4)
        self.assertEqual(ref_msad_u8_sum(s0, s1), ref_sad_u8_sum(s0, s1))
        self.assertEqual(ref_msad_u8(s0, s1, 50), ref_sad_u8(s0, s1, 50))

    def test_msad_u8_clamp_behavior(self) -> None:
        s0 = pack_bytes(255, 255, 255, 255)
        s1 = pack_bytes(1, 1, 1, 1)
        # diff = 4 * 254 = 1016
        self.assertEqual(ref_msad_u8(s0, s1, MASK32, clamp=True), MASK32)
        self.assertEqual(ref_msad_u8(s0, s1, MASK32, clamp=False), 1015)

    def test_sad_u8_and_sad_hi_u8_multi_precision_composition(self) -> None:
        # Combining sad_u8 and sad_hi_u8 enables dual-accumulator multi-precision tracking:
        s0 = pack_bytes(100, 150, 200, 250)
        s1 = pack_bytes(50, 75, 100, 125)
        sad = ref_sad_u8_sum(s0, s1)  # 50 + 75 + 100 + 125 = 350
        res_lo = ref_sad_u8(s0, s1, 0)
        res_hi = ref_sad_hi_u8(s0, s1, 0)
        self.assertEqual(res_lo, 350)
        self.assertEqual(res_hi, 350 << 16)
        self.assertEqual((res_lo << 16) & MASK32, res_hi)


if __name__ == "__main__":
    unittest.main()
