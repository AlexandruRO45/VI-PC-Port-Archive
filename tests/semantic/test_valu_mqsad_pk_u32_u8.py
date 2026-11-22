from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mqsad_pk_u32_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_mqsad_pk_u32_u8 dst, src0, src1, src2 [clamp]

Format:
  VOP3 (Vector ALU 3-source instruction, 128-bit destination).

Operands:
  - dst:  128-bit VGPR quad v[d:d+3], holding 4 unsigned 32-bit dwords:
            dst[31:0]   = output dword 0
            dst[63:32]  = output dword 1
            dst[95:64]  = output dword 2
            dst[127:96] = output dword 3
  - src0: 64-bit VGPR pair v[s0:s0+1], holding 8 packed 8-bit unsigned bytes (sliding reference window):
            byte 0: src0[7:0]
            byte 1: src0[15:8]
            byte 2: src0[23:16]
            byte 3: src0[31:24]
            byte 4: src0[39:32]
            byte 5: src0[47:40]
            byte 6: src0[55:48]
            byte 7: src0[63:56]  (unused in quad-SAD sliding windows)
  - src1: 32-bit VGPR, SGPR, or literal, holding 4 packed 8-bit unsigned bytes (query template with byte-zero masking):
            byte 0: src1[7:0]
            byte 1: src1[15:8]
            byte 2: src1[23:16]
            byte 3: src1[31:24]
            Masking rule: If query byte src1.byte[j] == 0, that byte is masked out (diff = 0).
  - src2: 128-bit VGPR quad, SGPR quad, or literal (accumulator):
            accum[0]: src2[31:0]
            accum[1]: src2[63:32]
            accum[2]: src2[95:64]
            accum[3]: src2[127:96]

Operation:
  Computes 4 overlapping 4-byte Masked Sum of Absolute Differences (MSAD) across the 8-byte
  reference window in src0 against the 4-byte query template in src1, accumulating into
  the 4 32-bit dwords in src2:

    D[31:0]   = MSAD_U8(S0[31:0],  S1[31:0], S2[31:0]);
    D[63:32]  = MSAD_U8(S0[39:8],  S1[31:0], S2[63:32]);
    D[95:64]  = MSAD_U8(S0[47:16], S1[31:0], S2[95:64]);
    D[127:96] = MSAD_U8(S0[55:24], S1[31:0], S2[127:96]);

  Where MSAD_U8(ref4, query4, acc32) evaluates:
    sad = sum_{j=0..3} (|ref4_byte[j] - query4_byte[j]| if query4_byte[j] != 0 else 0)
    total = sad + acc32
    result = min(total, 0xFFFFFFFF) if CLAMP else (total & 0xFFFFFFFF)

Mathematical Properties and Bounds:
  - Each individual byte difference is in [0, 255].
  - 4-byte masked SAD sum is in [0, 4 * 255] = [0, 1020] (0x03FC).
  - Query byte 0 acts as a per-byte mask: when src1.byte[j] == 0, no difference is accumulated.
  - Accumulator is 32-bit unsigned [0, 2^32 - 1].
  - Unclamped accumulation wraps modulo 2^32 (& 0xFFFFFFFF).
  - Clamped accumulation saturates at 2^32 - 1 (0xFFFFFFFF).
  - src0 byte 7 (bits 63:56) is never referenced by the 4 sliding windows of length 4.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK128: int = (1 << 128) - 1


def u8(val: int) -> int:
    """Normalize integer to unsigned 8-bit byte."""
    return val & MASK8


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def u128(val: int) -> int:
    """Normalize integer to unsigned 128-bit quadword."""
    return val & MASK128


def extract_byte(val: int, byte_idx: int) -> int:
    """Extract byte at index byte_idx (0-based) from an integer."""
    return (val >> (8 * byte_idx)) & MASK8


def extract_u32(val: int, dword_idx: int) -> int:
    """Extract 32-bit dword at index dword_idx (0-based) from an integer."""
    return (val >> (32 * dword_idx)) & MASK32


def pack_u32x4(d0: int, d1: int, d2: int, d3: int) -> int:
    """Pack four 32-bit unsigned integers into a 128-bit integer."""
    return (
        (u32(d0))
        | (u32(d1) << 32)
        | (u32(d2) << 64)
        | (u32(d3) << 96)
    ) & MASK128


def unpack_u32x4(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 128-bit integer into four 32-bit unsigned integers (d0, d1, d2, d3)."""
    return (
        extract_u32(val, 0),
        extract_u32(val, 1),
        extract_u32(val, 2),
        extract_u32(val, 3),
    )


def pack_u8x8(b0: int, b1: int, b2: int, b3: int, b4: int, b5: int, b6: int, b7: int) -> int:
    """Pack eight unsigned bytes into a 64-bit qword."""
    return (
        (u8(b0))
        | (u8(b1) << 8)
        | (u8(b2) << 16)
        | (u8(b3) << 24)
        | (u8(b4) << 32)
        | (u8(b5) << 40)
        | (u8(b6) << 48)
        | (u8(b7) << 56)
    ) & MASK64


def ref_msad_u8_quad(slice_4bytes: int, query_4bytes: int) -> int:
    """Reference mathematical oracle for a 4-byte Masked Sum of Absolute Differences."""
    total = 0
    for j in range(4):
        ref_b = extract_byte(slice_4bytes, j)
        qry_b = extract_byte(query_4bytes, j)
        if qry_b != 0:
            total += abs(ref_b - qry_b)
    return total


def ref_mqsad_pk_u32_u8(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_mqsad_pk_u32_u8.

    Args:
        src0: 64-bit unsigned reference window (8 packed bytes).
        src1: 32-bit unsigned query template (4 packed bytes with byte-0 masking).
        src2: 128-bit unsigned accumulator (4 packed 32-bit dwords).
        clamp: If True, saturates each 32-bit sum at 0xFFFFFFFF.

    Returns:
        128-bit unsigned integer containing 4 packed 32-bit accumulated MSADs.
    """
    out_elements = []
    for k in range(4):
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_msad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u32(src2, k)
        raw_sum = sad + acc
        if clamp:
            res = min(raw_sum, MASK32)
        else:
            res = raw_sum & MASK32
        out_elements.append(res)

    return pack_u32x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])


def ref_mqsad_pk_u32_u8_details(
    src0: int, src1: int, src2: int, clamp: bool = False
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[bool, bool, bool, bool], int]:
    """Extended reference oracle returning (sads, raw_sums, carries, packed_result)."""
    sads = []
    raw_sums = []
    carries = []
    out_elements = []
    for k in range(4):
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_msad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u32(src2, k)
        raw = sad + acc
        carry = raw > MASK32
        res = min(raw, MASK32) if clamp else (raw & MASK32)
        sads.append(sad)
        raw_sums.append(raw)
        carries.append(carry)
        out_elements.append(res)

    packed = pack_u32x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])
    return (
        (sads[0], sads[1], sads[2], sads[3]),
        (raw_sums[0], raw_sums[1], raw_sums[2], raw_sums[3]),
        (carries[0], carries[1], carries[2], carries[3]),
        packed,
    )


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMqsadPkU32U8Oracle(unittest.TestCase):
    """Unit tests for RDNA2 v_mqsad_pk_u32_u8 architectural bit model and reference oracle."""

    def test_zero_inputs(self) -> None:
        self.assertEqual(ref_mqsad_pk_u32_u8(0, 0, 0), 0)
        self.assertEqual(ref_mqsad_pk_u32_u8(0, 0, 0, clamp=True), 0)

    def test_masking_byte_zero_skips_difference(self) -> None:
        # Ref: [100, 200, 150, 50, ...]
        # Query: [100, 0, 150, 0] -> bytes 1 and 3 are 0 (masked out!)
        # Window 0:
        #   byte 0: |100 - 100| = 0
        #   byte 1: query is 0 -> skipped (diff = 0 despite ref being 200!)
        #   byte 2: |150 - 150| = 0
        #   byte 3: query is 0 -> skipped (diff = 0 despite ref being 50!)
        # Total SAD = 0!
        ref = pack_u8x8(100, 200, 150, 50, 0, 0, 0, 0)
        query = pack_u8x8(100, 0, 150, 0, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(10, 20, 30, 40)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 10)  # SAD 0 + 10 = 10

    def test_sliding_window_offsets(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 0, 0, 0)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 0)
        self.assertEqual(d1, 40)
        self.assertEqual(d2, 80)
        self.assertEqual(d3, 120)

    def test_maximum_individual_sad(self) -> None:
        # All 0xFF vs 0x01 (nonzero, so not masked) -> SAD = 4 * 254 = 1016
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = pack_u8x8(0x01, 0x01, 0x01, 0x01, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 10, 100, 1000)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 1016)
        self.assertEqual(d1, 1026)
        self.assertEqual(d2, 1116)
        self.assertEqual(d3, 2016)

    def test_accumulator_addition_and_wrapping(self) -> None:
        # SAD = 1016. Accumulator = 0xFFFFFFFF (2^32 - 1).
        # Without clamp: 0xFFFFFFFF + 1016 = 0x1000003F7 -> wraps to 1015 (0x03F7)
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = pack_u8x8(0x01, 0x01, 0x01, 0x01, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0xFFFF_FFFF, 0xFFFF_FFFE, 0x0000_0001, 0x0000_0000)

        res = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=False)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, (1016 + 0xFFFF_FFFF) & MASK32)  # 1015
        self.assertEqual(d1, (1016 + 0xFFFF_FFFE) & MASK32)  # 1014
        self.assertEqual(d2, 1017)
        self.assertEqual(d3, 1016)

    def test_clamp_modifier_saturation(self) -> None:
        # With clamp: saturated at 0xFFFFFFFF
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = pack_u8x8(0x01, 0x01, 0x01, 0x01, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0xFFFF_FFFF, 0xFFFF_FFFE, 0x0000_0001, 0x0000_0000)

        res = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=True)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 0xFFFF_FFFF)
        self.assertEqual(d1, 0xFFFF_FFFF)
        self.assertEqual(d2, 1017)
        self.assertEqual(d3, 1016)

    def test_unused_byte7_invariance(self) -> None:
        ref_a = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0x00)
        ref_b = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0xFF)
        query = 0x1234_5678
        acc = pack_u32x4(100, 200, 300, 400)

        res_a = ref_mqsad_pk_u32_u8(ref_a, query, acc)
        res_b = ref_mqsad_pk_u32_u8(ref_b, query, acc)
        self.assertEqual(res_a, res_b)

    def test_ref_details_breakdown(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 0xFFFF_FFFF, 0, 0)

        sads, raw_sums, carries, packed = ref_mqsad_pk_u32_u8_details(ref, query, acc, clamp=False)
        self.assertEqual(sads, (0, 40, 80, 120))
        self.assertEqual(raw_sums, (0, 40 + 0xFFFF_FFFF, 80, 120))
        self.assertEqual(carries, (False, True, False, False))
        self.assertEqual(unpack_u32x4(packed), (0, (40 + 0xFFFF_FFFF) & MASK32, 80, 120))


if __name__ == "__main__":
    unittest.main()
