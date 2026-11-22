from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mqsad_u32_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_mqsad_u32_u8 dst, src0, src1, src2
  Syntax: v_mqsad_u32_u8 v[d:d+3], src0[0:1], src1, v[s2:s2+3]

Format:
  VOP3 (3-operand vector ALU instruction with multi-register operands).

Operands:
  - dst:  128-bit vector destination spanning 4 consecutive VGPRs: v[d], v[d+1], v[d+2], v[d+3]
  - src0: 64-bit source spanning 2 registers (VGPR pair, SGPR pair, or 64-bit constant)
  - src1: 32-bit source (1 VGPR, SGPR, or 32-bit constant)
  - src2: 128-bit vector accumulator spanning 4 consecutive VGPRs: v[s2], v[s2+1], v[s2+2], v[s2+3]

Operation:
  Computes four masked byte-level sums of absolute differences (SAD) with four
  full 32-bit accumulators:
    For each dword k in 0..3:
      W_k = (src0 >> (8 * k)) & 0xFFFFFFFF  (sliding 32-bit window across 64-bit src0)
      sad_k = 0
      For each byte i in 0..3:
        b0 = (W_k >> (8 * i)) & 0xFF
        b1 = (src1 >> (8 * i)) & 0xFF
        If b1 != 0:
          sad_k += |b0 - b1|
        Else:
          (byte skipped, adds 0)
      dst[k] = (src2[k] + sad_k) & 0xFFFFFFFF

Mathematical Properties and Bounds:
  - Operands:
      src0 in [0, 2^64 - 1]
      src1 in [0, 2^32 - 1]
      src2[k] in [0, 2^32 - 1]
  - Each byte difference |b0 - b1| is in [0, 255].
  - The maximum sum of 4 byte differences is 4 * 255 = 1020 (0x3FC).
  - Accumulation sum64 = src2[k] + sad_k is in [0, 2^32 - 1 + 1020] = [0, 0x1000003FB].
  - The carry-out past 32 bits into bit 32 is strictly in {0, 1}.
  - Zero-byte mask rule: When byte i of src1 is 0, that byte position is masked out and
    contributes 0 to the sum of differences regardless of the value in W_k.
  - The 8th byte of src0 (bits 63:56) is unused by the 4 windows (window 3 uses bytes 3..6).
"""

from typing import List, Sequence, Tuple
import unittest

# Architectural bitmasks
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK128: int = (1 << 128) - 1

MAX_BYTE_SAD: int = 4 * 255  # 1020 (0x3FC)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def extract_byte(word: int, byte_idx: int) -> int:
    """Extract byte at byte_idx (0..3) from a 32-bit word."""
    return (word >> (8 * (byte_idx & 3))) & MASK8


def byte_window_u32(val64: int, byte_offset: int) -> int:
    """Extract 32-bit sliding window W_k at byte offset k from 64-bit val: bits [8*k+31 : 8*k]."""
    return (val64 >> (8 * (byte_offset & 7))) & MASK32


def ref_masked_byte_sad_u32(win32: int, mask32: int) -> int:
    """Compute masked sum of absolute differences of 4 bytes.

    A byte where mask32's byte is 0 adds nothing.
    """
    total = 0
    for i in range(4):
        b_win = extract_byte(win32, i)
        b_mask = extract_byte(mask32, i)
        if b_mask != 0:
            diff = (b_win - b_mask) if b_win >= b_mask else (b_mask - b_win)
            total += diff
    return total


def ref_mqsad_u32_u8(s0_64: int, s1_32: int, s2_quad: Sequence[int]) -> List[int]:
    """Pure-Python bit-accurate reference mathematical oracle for AMD RDNA2 v_mqsad_u32_u8.

    Args:
        s0_64: 64-bit integer source operand.
        s1_32: 32-bit integer source operand (mask and reference bytes).
        s2_quad: Sequence of 4 unsigned 32-bit accumulator dwords.

    Returns:
        List of 4 unsigned 32-bit dwords representing v[d:d+3].
    """
    if len(s2_quad) != 4:
        raise ValueError(f"src2 accumulator must have 4 elements, got {len(s2_quad)}")

    res = []
    for k in range(4):
        win = byte_window_u32(s0_64, k)
        sad = ref_masked_byte_sad_u32(win, s1_32)
        acc = s2_quad[k] & MASK32
        res.append((acc + sad) & MASK32)
    return res


def ref_mqsad_full_acc(s0_64: int, s1_32: int, s2_quad: Sequence[int]) -> List[Tuple[int, int]]:
    """Compute (lo32, carry_out) for each of the 4 accumulators."""
    if len(s2_quad) != 4:
        raise ValueError(f"src2 accumulator must have 4 elements, got {len(s2_quad)}")

    out = []
    for k in range(4):
        win = byte_window_u32(s0_64, k)
        sad = ref_masked_byte_sad_u32(win, s1_32)
        acc = s2_quad[k] & MASK32
        full_sum = acc + sad
        out.append((full_sum & MASK32, (full_sum >> 32) & 1))
    return out


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMqsadU32U8Oracle(unittest.TestCase):
    """Unit tests for AMD RDNA2 v_mqsad_u32_u8 architectural bit model and reference oracle."""

    def test_known_vectors_basic(self) -> None:
        # Zero inputs -> zero outputs
        self.assertEqual(ref_mqsad_u32_u8(0, 0, [0, 0, 0, 0]), [0, 0, 0, 0])

        # All-zero src1: all mask bytes are 0, so SAD is 0 regardless of src0
        self.assertEqual(
            ref_mqsad_u32_u8(0xDEAD_BEEF_CAFE_BABE, 0x0000_0000, [10, 20, 30, 40]),
            [10, 20, 30, 40],
        )

    def test_sliding_window_byte_shifts(self) -> None:
        # src0 has distinct bytes: byte0=0x01, byte1=0x02, byte2=0x03, byte3=0x04,
        # byte4=0x05, byte5=0x06, byte6=0x07, byte7=0x08.
        src0 = 0x0807060504030201
        self.assertEqual(byte_window_u32(src0, 0), 0x04030201)
        self.assertEqual(byte_window_u32(src0, 1), 0x05040302)
        self.assertEqual(byte_window_u32(src0, 2), 0x06050403)
        self.assertEqual(byte_window_u32(src0, 3), 0x07060504)

    def test_isa_known_vector_sads(self) -> None:
        # Standard ISA known vector:
        # src0 = 0x0807060504030201, src1 = 0x04030201, accs = [0, 0, 0, 0]
        # Window 0: 0x04030201 vs 0x04030201 -> diffs (0, 0, 0, 0) -> SAD = 0
        # Window 1: 0x05040302 vs 0x04030201 -> diffs (1, 1, 1, 1) -> SAD = 4
        # Window 2: 0x06050403 vs 0x04030201 -> diffs (2, 2, 2, 2) -> SAD = 8
        # Window 3: 0x07060504 vs 0x04030201 -> diffs (3, 3, 3, 3) -> SAD = 12
        got = ref_mqsad_u32_u8(0x0807060504030201, 0x04030201, [0, 0, 0, 0])
        self.assertEqual(got, [0, 4, 8, 12])

        # With non-zero accumulators
        got_acc = ref_mqsad_u32_u8(0x0807060504030201, 0x04030201, [100, 200, 300, 400])
        self.assertEqual(got_acc, [100, 204, 308, 412])

    def test_masking_zero_byte_skips_accumulation(self) -> None:
        # src1 has only byte 0 non-zero: 0x00000005
        # Bytes 1, 2, 3 are 0 and MUST NOT contribute to the SAD
        win = 0xFFFFFFFF
        mask = 0x00000005
        # Only byte 0: |0xFF - 0x05| = 255 - 5 = 250
        self.assertEqual(ref_masked_byte_sad_u32(win, mask), 250)

        # Byte 1 only non-zero: 0x00000500
        mask_b1 = 0x00000500
        self.assertEqual(ref_masked_byte_sad_u32(win, mask_b1), 250)

        # Byte 2 only non-zero: 0x00050000
        mask_b2 = 0x00050000
        self.assertEqual(ref_masked_byte_sad_u32(win, mask_b2), 250)

        # Byte 3 only non-zero: 0x05000000
        mask_b3 = 0x05000000
        self.assertEqual(ref_masked_byte_sad_u32(win, mask_b3), 250)

    def test_accumulator_wrapping_u32(self) -> None:
        # src2 has accumulator 0xFFFFFFFF.
        # Adding SAD of 4 should wrap to 3: (0xFFFFFFFF + 4) mod 2^32 = 3.
        src0 = 0x0807060504030201
        src1 = 0x04030201
        accs = [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF]
        got = ref_mqsad_u32_u8(src0, src1, accs)
        # SADs are [0, 4, 8, 12]
        self.assertEqual(got, [
            0xFFFF_FFFF,
            3,
            7,
            11,
        ])

        full = ref_mqsad_full_acc(src0, src1, accs)
        self.assertEqual(full, [
            (0xFFFF_FFFF, 0),
            (3, 1),
            (7, 1),
            (11, 1),
        ])

    def test_maximum_possible_byte_sad(self) -> None:
        # Max difference occurs when win has 0xFF for all 4 bytes and mask has 0x01 (or vice versa with mask non-zero)
        # diff per byte = |0xFF - 0x01| = 254.
        # If win has 0x00 and mask has 0xFF, diff per byte = |0x00 - 0xFF| = 255.
        win_0 = 0x00000000
        mask_ff = 0xFFFFFFFF
        self.assertEqual(ref_masked_byte_sad_u32(win_0, mask_ff), 4 * 255)
        self.assertEqual(ref_masked_byte_sad_u32(win_0, mask_ff), MAX_BYTE_SAD)

    def test_upper_byte_of_src0_is_unused(self) -> None:
        # Byte 7 of src0 (bits 63:56) is never in window 0, 1, 2, or 3
        src0_a = 0x0007060504030201
        src0_b = 0xFF07060504030201
        src1 = 0x04030201
        accs = [1, 2, 3, 4]
        self.assertEqual(ref_mqsad_u32_u8(src0_a, src1, accs), ref_mqsad_u32_u8(src0_b, src1, accs))

    def test_invalid_accumulator_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            ref_mqsad_u32_u8(0, 0, [0, 0, 0])
        with self.assertRaises(ValueError):
            ref_mqsad_full_acc(0, 0, [0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
