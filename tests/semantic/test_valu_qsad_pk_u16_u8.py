from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_qsad_pk_u16_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_qsad_pk_u16_u8 dst, src0, src1, src2 [clamp]

Format:
  VOP3 (Vector ALU 3-source instruction, 64-bit destination).

Operands:
  - dst:  64-bit VGPR pair v[d:d+1], holding 4 packed 16-bit unsigned integers:
            dst[15:0]  = packed halfword 0
            dst[31:16] = packed halfword 1
            dst[47:32] = packed halfword 2
            dst[63:48] = packed halfword 3
  - src0: 64-bit VGPR pair v[s0:s0+1], holding 8 packed 8-bit unsigned bytes (sliding reference window):
            byte 0: src0[7:0]
            byte 1: src0[15:8]
            byte 2: src0[23:16]
            byte 3: src0[31:24]
            byte 4: src0[39:32]
            byte 5: src0[47:40]
            byte 6: src0[55:48]
            byte 7: src0[63:56]  (unused in quad-SAD sliding windows)
  - src1: 32-bit VGPR, SGPR, or literal, holding 4 packed 8-bit unsigned bytes (query template):
            byte 0: src1[7:0]
            byte 1: src1[15:8]
            byte 2: src1[23:16]
            byte 3: src1[31:24]
  - src2: 64-bit VGPR pair, SGPR pair, or literal/constant (accumulator):
            accum[0]: src2[15:0]
            accum[1]: src2[31:16]
            accum[2]: src2[47:32]
            accum[3]: src2[63:48]

Operation:
  Computes 4 overlapping 4-byte Sum of Absolute Differences (SAD) across the 8-byte
  reference window in src0 against the 4-byte query template in src1, accumulating into
  the 4 packed 16-bit integers in src2:

    D[15:0]  = SAD_U8(S0[31:0],  S1[31:0], S2[15:0]);
    D[31:16] = SAD_U8(S0[39:8],  S1[31:0], S2[31:16]);
    D[47:32] = SAD_U8(S0[47:16], S1[31:0], S2[47:32]);
    D[63:48] = SAD_U8(S0[55:24], S1[31:0], S2[63:48]);

  Where SAD_U8(ref4, query4, acc16) evaluates:
    sad = sum_{j=0..3} |ref4_byte[j] - query4_byte[j]|
    total = sad + acc16
    result = min(total, 0xFFFF) if CLAMP else (total & 0xFFFF)

Mathematical Properties and Bounds:
  - Each individual byte difference |ref - query| is in [0, 255].
  - 4-byte SAD sum is in [0, 4 * 255] = [0, 1020] (0x03FC).
  - Accumulator is 16-bit unsigned [0, 65535].
  - Unclamped accumulation wraps modulo 65536 (& 0xFFFF).
  - Clamped accumulation saturates at 65535 (0xFFFF).
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


def extract_byte(val: int, byte_idx: int) -> int:
    """Extract byte at index byte_idx (0-based) from an integer."""
    return (val >> (8 * byte_idx)) & MASK8


def extract_u16(val: int, half_idx: int) -> int:
    """Extract 16-bit halfword at index half_idx (0-based) from an integer."""
    return (val >> (16 * half_idx)) & MASK16


def pack_u16x4(e0: int, e1: int, e2: int, e3: int) -> int:
    """Pack four 16-bit unsigned integers into a 64-bit qword."""
    return (
        (u16(e0))
        | (u16(e1) << 16)
        | (u16(e2) << 32)
        | (u16(e3) << 48)
    ) & MASK64


def unpack_u16x4(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 64-bit integer into four 16-bit unsigned integers (e0, e1, e2, e3)."""
    return (
        extract_u16(val, 0),
        extract_u16(val, 1),
        extract_u16(val, 2),
        extract_u16(val, 3),
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


def unpack_u8x8(val: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """Unpack a 64-bit integer into eight unsigned bytes (b0..b7)."""
    return tuple(extract_byte(val, i) for i in range(8))  # type: ignore[return-value]


def ref_sad_u8_quad(slice_4bytes: int, query_4bytes: int) -> int:
    """Reference mathematical oracle for a 4-byte Sum of Absolute Differences."""
    total = 0
    for j in range(4):
        ref_b = extract_byte(slice_4bytes, j)
        qry_b = extract_byte(query_4bytes, j)
        total += abs(ref_b - qry_b)
    return total


def ref_qsad_pk_u16_u8(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_qsad_pk_u16_u8.

    Args:
        src0: 64-bit unsigned reference window (8 packed bytes).
        src1: 32-bit unsigned query template (4 packed bytes).
        src2: 64-bit unsigned accumulator (4 packed 16-bit halfwords).
        clamp: If True, saturates each 16-bit sum at 0xFFFF.

    Returns:
        64-bit unsigned qword containing 4 packed 16-bit accumulated SADs.
    """
    out_elements = []
    for k in range(4):
        # 4 bytes starting at byte offset k
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_sad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u16(src2, k)
        raw_sum = sad + acc
        if clamp:
            res = min(raw_sum, MASK16)
        else:
            res = raw_sum & MASK16
        out_elements.append(res)

    return pack_u16x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])


def ref_qsad_pk_u16_u8_details(
    src0: int, src1: int, src2: int, clamp: bool = False
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[bool, bool, bool, bool], int]:
    """Extended reference oracle returning (sads, raw_sums, carries, packed_result)."""
    sads = []
    raw_sums = []
    carries = []
    out_elements = []
    for k in range(4):
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_sad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u16(src2, k)
        raw = sad + acc
        carry = raw > MASK16
        res = min(raw, MASK16) if clamp else (raw & MASK16)
        sads.append(sad)
        raw_sums.append(raw)
        carries.append(carry)
        out_elements.append(res)

    packed = pack_u16x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])
    return (
        (sads[0], sads[1], sads[2], sads[3]),
        (raw_sums[0], raw_sums[1], raw_sums[2], raw_sums[3]),
        (carries[0], carries[1], carries[2], carries[3]),
        packed,
    )


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuQsadPkU16U8Oracle(unittest.TestCase):
    """Unit tests for RDNA2 v_qsad_pk_u16_u8 architectural bit model and reference oracle."""

    def test_zero_inputs(self) -> None:
        self.assertEqual(ref_qsad_pk_u16_u8(0, 0, 0), 0)
        self.assertEqual(ref_qsad_pk_u16_u8(0, 0, 0, clamp=True), 0)

    def test_identical_reference_and_query_zero_sad(self) -> None:
        # Repeating pattern 0x04030201 in query and reference
        query = 0x0403_0201
        ref = (query << 32) | query  # 8 bytes: [1,2,3,4, 1,2,3,4]
        # At window offset 0: bytes 1,2,3,4 vs 1,2,3,4 -> SAD = 0
        # At window offset 4: bytes 1,2,3,4 vs 1,2,3,4 -> SAD = 0
        # Accumulator values
        acc = pack_u16x4(10, 20, 30, 40)
        res = ref_qsad_pk_u16_u8(ref, query, acc)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 10)  # SAD 0 + 10 = 10

    def test_sliding_window_offsets(self) -> None:
        # Construct reference: [10, 20, 30, 40, 50, 60, 70, 80]
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        # Query: [10, 20, 30, 40]
        # Window 0: [10,20,30,40] vs [10,20,30,40] -> SAD = 0
        # Window 1: [20,30,40,50] vs [10,20,30,40] -> diffs (10+10+10+10) = 40
        # Window 2: [30,40,50,60] vs [10,20,30,40] -> diffs (20+20+20+20) = 80
        # Window 3: [40,50,60,70] vs [10,20,30,40] -> diffs (30+30+30+30) = 120
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(0, 0, 0, 0)

        res = ref_qsad_pk_u16_u8(ref, query, acc)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 0)
        self.assertEqual(e1, 40)
        self.assertEqual(e2, 80)
        self.assertEqual(e3, 120)

    def test_maximum_individual_sad(self) -> None:
        # All 0xFF vs 0x00 -> SAD = 4 * 255 = 1020 (0x03FC)
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x0000_0000
        acc = pack_u16x4(0, 10, 100, 1000)

        res = ref_qsad_pk_u16_u8(ref, query, acc)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 1020)
        self.assertEqual(e1, 1030)
        self.assertEqual(e2, 1120)
        self.assertEqual(e3, 2020)

    def test_accumulator_addition_and_wrapping(self) -> None:
        # SAD = 1020. Accumulator = 0xFFFF (65535).
        # Without clamp: 65535 + 1020 = 66555 = 0x103FB -> wraps to 0x03FB (1019)
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x0000_0000
        acc = pack_u16x4(0xFFFF, 0xFFFE, 0x0001, 0x0000)

        res = ref_qsad_pk_u16_u8(ref, query, acc, clamp=False)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, (1020 + 0xFFFF) & 0xFFFF)  # 1019
        self.assertEqual(e1, (1020 + 0xFFFE) & 0xFFFF)  # 1018
        self.assertEqual(e2, 1021)
        self.assertEqual(e3, 1020)

    def test_clamp_modifier_saturation(self) -> None:
        # With clamp: saturated at 0xFFFF
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x0000_0000
        acc = pack_u16x4(0xFFFF, 0xFFFE, 0x0001, 0x0000)

        res = ref_qsad_pk_u16_u8(ref, query, acc, clamp=True)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 0xFFFF)
        self.assertEqual(e1, 0xFFFF)
        self.assertEqual(e2, 1021)
        self.assertEqual(e3, 1020)

    def test_unused_byte7_invariance(self) -> None:
        # Modifying byte 7 of src0 must have zero effect on any output element
        ref_a = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0x00)
        ref_b = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0xFF)
        query = 0x1234_5678
        acc = pack_u16x4(100, 200, 300, 400)

        res_a = ref_qsad_pk_u16_u8(ref_a, query, acc)
        res_b = ref_qsad_pk_u16_u8(ref_b, query, acc)
        self.assertEqual(res_a, res_b)

    def test_ref_details_breakdown(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(0, 0xFFFF, 0, 0)

        sads, raw_sums, carries, packed = ref_qsad_pk_u16_u8_details(ref, query, acc, clamp=False)
        self.assertEqual(sads, (0, 40, 80, 120))
        self.assertEqual(raw_sums, (0, 40 + 0xFFFF, 80, 120))
        self.assertEqual(carries, (False, True, False, False))
        self.assertEqual(unpack_u16x4(packed), (0, (40 + 0xFFFF) & 0xFFFF, 80, 120))


if __name__ == "__main__":
    unittest.main()
