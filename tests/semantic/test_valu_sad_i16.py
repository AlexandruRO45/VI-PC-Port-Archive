from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for AMD RDNA2 v_sad_i16.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_sad_i16 (VOP3 3-operand vector ALU instruction):
    Computes the 2-channel sum of signed 16-bit absolute differences between
    packed signed halfword lanes of src0 and src1, and accumulates into the 32-bit
    unsigned accumulator src2:
        h0_0 = to_s16(src0 & 0xFFFF)
        h0_1 = to_s16((src0 >> 16) & 0xFFFF)
        h1_0 = to_s16(src1 & 0xFFFF)
        h1_1 = to_s16((src1 >> 16) & 0xFFFF)
        diff0 = abs(h0_0 - h1_0)
        diff1 = abs(h0_1 - h1_1)
        sum_diff = diff0 + diff1
        dst.u32 = (sum_diff + src2.u32) mod 2^32
        carry_out = 1 if (sum_diff + src2.u32 >= 2^32) else 0

    Outputs:
        Vector Destination (VDST): 32-bit VGPR storing dst.u32.
        Carry generation: Tracks whether (sum_diff + src2) exceeded 2^32.

    Mathematical Bounds:
        - h0_i, h1_i in [-32768, 32767] (signed two's complement int16).
        - diff_i = |h0_i - h1_i| in [0, 65535] (0x0000 to 0xFFFF).
        - Maximum difference per lane: |32767 - (-32768)| = 65535.
        - Total sum_diff = diff0 + diff1 in [0, 131070] (0x00000 to 0x1FFFE).
        - src2.u32 in [0, 2^32 - 1] (0 to 0xFFFFFFFF).
        - Maximum full sum: 131070 + 0xFFFFFFFF = 0x1_0001_FFFD.
        - Therefore, carry_out is strictly binary {0, 1}.

    Differences from v_sad_u16:
        - In v_sad_u16, 0xFFFF is unsigned 65535, giving |65535 - 0| = 65535.
        - In v_sad_i16, 0xFFFF is signed -1, giving |-1 - 0| = 1.
        - In v_sad_i16, 0x8000 (-32768) vs 0x7FFF (32767) gives max difference 65535.

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and limits
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def to_s16(val: int) -> int:
    """Convert integer to signed 16-bit integer [-32768, 32767]."""
    val &= MASK16
    return val - 65536 if (val & 0x8000) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def unpack_signed_halfwords(val: int) -> Tuple[int, int]:
    """Unpack 32-bit word into two signed 16-bit integers (h0, h1)."""
    val &= MASK32
    return to_s16(val & MASK16), to_s16((val >> 16) & MASK16)


def pack_halfwords(h0: int, h1: int) -> int:
    """Pack two 16-bit integers into an unsigned 32-bit dword."""
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def ref_abs_diff_s16(h0: int, h1: int) -> int:
    """Signed 16-bit absolute difference: |to_s16(h0) - to_s16(h1)|."""
    s0 = to_s16(h0)
    s1 = to_s16(h1)
    return abs(s0 - s1)


def ref_sad_i16_sum(s0: int, s1: int) -> int:
    """Compute the 2-channel sum of signed 16-bit absolute differences in [0, 131070]."""
    h0_0, h0_1 = unpack_signed_halfwords(s0)
    h1_0, h1_1 = unpack_signed_halfwords(s1)
    return abs(h0_0 - h1_0) + abs(h0_1 - h1_1)


def ref_sad_i16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_sad_i16:

    dst = (sum_abs_diff_s16(s0, s1) + s2) & 0xFFFFFFFF
    """
    sum_diff = ref_sad_i16_sum(s0, s1)
    acc = s2 & MASK32
    return (sum_diff + acc) & MASK32


def ref_sad_i16_full(s0: int, s1: int, s2: int) -> Tuple[int, int, int]:
    """Return (dst_u32, carry_out, full_sum64) for v_sad_i16."""
    sum_diff = ref_sad_i16_sum(s0, s1)
    acc = s2 & MASK32
    full = sum_diff + acc
    carry = (full >> 32) & 1
    dst = full & MASK32
    return dst, carry, full


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestVSadI16Oracle(unittest.TestCase):
    """Test suite verifying mathematical bounds, properties, and edge cases of the oracle."""

    def test_oracle_zero_inputs(self) -> None:
        """All-zero inputs yield zero result."""
        self.assertEqual(ref_sad_i16(0, 0, 0), 0)
        dst, carry, full = ref_sad_i16_full(0, 0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 0)
        self.assertEqual(full, 0)

    def test_oracle_identical_inputs(self) -> None:
        """When s0 == s1, absolute differences are all zero and result is s2."""
        patterns = [0x00000000, 0x12345678, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF]
        for pat in patterns:
            acc = 0xDEADBEEF
            self.assertEqual(ref_sad_i16(pat, pat, acc), acc)
            self.assertEqual(ref_sad_i16_sum(pat, pat), 0)

    def test_oracle_signed_boundary_differences(self) -> None:
        """Max difference between min negative (-32768, 0x8000) and max positive (32767, 0x7FFF)."""
        s0 = pack_halfwords(0x8000, 0x8000)  # -32768, -32768
        s1 = pack_halfwords(0x7FFF, 0x7FFF)  # 32767, 32767

        # |32767 - (-32768)| = 65535 per lane -> total sum = 131070 (0x1FFFE)
        sum_diff = ref_sad_i16_sum(s0, s1)
        self.assertEqual(sum_diff, 131070)
        self.assertEqual(sum_diff, 0x1FFFE)
        self.assertEqual(ref_sad_i16(s0, s1, 0), 0x1FFFE)

    def test_oracle_negative_one_representation(self) -> None:
        """0xFFFF represents signed -1, differing from 0 by exactly 1."""
        s0 = pack_halfwords(0xFFFF, 0xFFFF)  # -1, -1
        s1 = pack_halfwords(0x0000, 0x0000)  # 0, 0

        # In signed math, |-1 - 0| = 1 per lane -> total sum = 2
        sum_diff = ref_sad_i16_sum(s0, s1)
        self.assertEqual(sum_diff, 2)
        self.assertEqual(ref_sad_i16(s0, s1, 0), 2)

    def test_oracle_single_halfword_lanes(self) -> None:
        """Verify each 16-bit lane contributes independently."""
        # Lane 0: -32768 vs 0 -> 32768
        self.assertEqual(ref_sad_i16(pack_halfwords(0x8000, 0), 0, 0), 32768)
        # Lane 1: -32768 vs 0 -> 32768
        self.assertEqual(ref_sad_i16(pack_halfwords(0, 0x8000), 0, 0), 32768)

    def test_oracle_commutativity(self) -> None:
        """Commutativity: ref_sad_i16(a, b, c) == ref_sad_i16(b, a, c)."""
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xFFFF0000, 0x0000FFFF),
            (0x80008000, 0x7FFF7FFF),
            (0xFFFFFFFF, 0x00000000),
        ]
        acc = 0xCAFEBABE
        for s0, s1 in test_pairs:
            self.assertEqual(ref_sad_i16(s0, s1, acc), ref_sad_i16(s1, s0, acc))
            self.assertEqual(ref_sad_i16_sum(s0, s1), ref_sad_i16_sum(s1, s0))

    def test_oracle_accumulation_with_wrap(self) -> None:
        """32-bit truncation when sum_diff + s2 exceeds 0xFFFFFFFF."""
        s0 = pack_halfwords(0x8000, 0x8000)
        s1 = pack_halfwords(0x7FFF, 0x7FFF)
        # sum_diff = 131070 (0x1FFFE); acc = 0xFFFF_0001
        # 0x1FFFE + 0xFFFF0001 = 0x1_0000FFFF -> wrapped = 0x0000FFFF
        dst, carry, full = ref_sad_i16_full(s0, s1, 0xFFFF_0001)
        self.assertEqual(dst, 0x0000_FFFF)
        self.assertEqual(carry, 1)
        self.assertEqual(full, 0x1_0000_FFFF)


if __name__ == "__main__":
    unittest.main()
