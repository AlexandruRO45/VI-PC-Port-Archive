from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_dot4_u32_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_dot4_u32_u8 dst, src0, src1, src2

Format:
  VOP3 (3-operand vector ALU instruction with 32-bit operands).

Operands:
  - dst:  32-bit vector destination register (VGPR)
  - src0: 32-bit vector or scalar register, or 32-bit inline/literal constant
  - src1: 32-bit vector or scalar register, or 32-bit inline/literal constant
  - src2: 32-bit accumulator register (VGPR, SGPR, or constant)

Operation:
  Computes the 4-element unsigned 8-bit dot product with 32-bit unsigned accumulation:
    b0 = (src0 >> 0) & 0xFF
    b1 = (src0 >> 8) & 0xFF
    b2 = (src0 >> 16) & 0xFF
    b3 = (src0 >> 24) & 0xFF

    c0 = (src1 >> 0) & 0xFF
    c1 = (src1 >> 8) & 0xFF
    c2 = (src1 >> 16) & 0xFF
    c3 = (src1 >> 24) & 0xFF

    dot = (b0 * c0) + (b1 * c1) + (b2 * c2) + (b3 * c3)
    dst = (src2 + dot) mod 2^32

Mathematical Properties and Bounds:
  - Operands:
      src0 in [0, 2^32 - 1]
      src1 in [0, 2^32 - 1]
      src2 in [0, 2^32 - 1]
  - Each byte product (b_i * c_i) is in [0, 255 * 255] = [0, 65025].
  - The sum of four products dot is in [0, 4 * 65025] = [0, 260100] (0x0003F804).
  - Accumulation sum64 = src2 + dot is in [0, 2^32 - 1 + 260100] = [0, 0x10003F803].
  - Result wraps modulo 2^32 into [0, 2^32 - 1].
  - Carry-out into bit 32 is strictly bounded in {0, 1}.

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from typing import List, Sequence, Tuple
import unittest

# Architectural bitmasks
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = (1 << 64) - 1

MAX_BYTE_PRODUCT: int = 255 * 255  # 65025
MAX_DOT4_SUM: int = 4 * MAX_BYTE_PRODUCT  # 260100 (0x3F804)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def unpack_u8(dword: int) -> Tuple[int, int, int, int]:
    """Unpack 32-bit dword into four unsigned 8-bit integers (byte0, byte1, byte2, byte3)."""
    dword &= MASK32
    return (
        dword & MASK8,
        (dword >> 8) & MASK8,
        (dword >> 16) & MASK8,
        (dword >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four unsigned 8-bit integers into a 32-bit dword."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_dot4_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Pure-Python bit-accurate reference oracle for AMD RDNA2 v_dot4_u32_u8.

    dst = (s2 + sum(s0[i] * s1[i] for i in 0..3)) mod 2^32
    """
    b = unpack_u8(s0)
    c = unpack_u8(s1)
    dot = (b[0] * c[0]) + (b[1] * c[1]) + (b[2] * c[2]) + (b[3] * c[3])
    acc = s2 & MASK32
    return (acc + dot) & MASK32


def ref_dot4_u32_u8_full(s0: int, s1: int, s2: int = 0) -> Tuple[int, int]:
    """Compute (lo32, carry_out) for (src2 + dot)."""
    b = unpack_u8(s0)
    c = unpack_u8(s1)
    dot = (b[0] * c[0]) + (b[1] * c[1]) + (b[2] * c[2]) + (b[3] * c[3])
    acc = s2 & MASK32
    full_sum = acc + dot
    return full_sum & MASK32, (full_sum >> 32) & 1


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuDot4U32U8Oracle(unittest.TestCase):
    """Unit tests for AMD RDNA2 v_dot4_u32_u8 architectural bit model and reference oracle."""

    def test_known_vectors_basic(self) -> None:
        # 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
        self.assertEqual(ref_dot4_u32_u8(0x01020304, 0x05060708, 0), 70)
        # With accumulator: 70 + 1000 = 1070
        self.assertEqual(ref_dot4_u32_u8(0x01020304, 0x05060708, 1000), 1070)

    def test_zero_identities(self) -> None:
        # Zero operand in s0 or s1 results in the accumulator value
        test_words = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x12345678]
        for w in test_words:
            for acc in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
                with self.subTest(w=hex(w), acc=hex(acc)):
                    self.assertEqual(ref_dot4_u32_u8(0, w, acc), acc & MASK32)
                    self.assertEqual(ref_dot4_u32_u8(w, 0, acc), acc & MASK32)

    def test_commutativity_of_multipliers(self) -> None:
        # s0 . s1 == s1 . s0
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFFFFFFFF, 0x12345678),
            (0x80808080, 0x7F7F7F7F),
            (0xA1B2C3D4, 0x1E2D3C4B),
            (0x00FF00FF, 0xFF00FF00),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 100, 0x12345678, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4_u32_u8(s0, s1, s2),
                        ref_dot4_u32_u8(s1, s0, s2),
                    )

    def test_identity_broadcast_sums_bytes(self) -> None:
        # Multiplying with 0x01010101 computes the sum of the four unsigned bytes
        test_words = [
            (0x01020304, 1 + 2 + 3 + 4),
            (0xFFFFFFFF, 255 * 4),
            (0x10203040, 0x10 + 0x20 + 0x30 + 0x40),
            (0x000000FF, 255),
            (0x0000FF00, 255),
            (0x00FF0000, 255),
            (0xFF000000, 255),
        ]
        for word, expected_sum in test_words:
            with self.subTest(word=hex(word)):
                self.assertEqual(ref_dot4_u32_u8(word, 0x01010101, 0), expected_sum)
                self.assertEqual(ref_dot4_u32_u8(0x01010101, word, 0), expected_sum)

    def test_byte_lane_isolation(self) -> None:
        # Each byte position operates independently without carry bleed into neighbor bytes
        for byte_idx in range(4):
            for val_a in (1, 15, 128, 255):
                for val_b in (1, 10, 100, 255):
                    s0 = val_a << (8 * byte_idx)
                    s1 = val_b << (8 * byte_idx)
                    expected = val_a * val_b
                    with self.subTest(lane=byte_idx, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), expected)

    def test_maximum_unsigned_product(self) -> None:
        # 4 * (255 * 255) = 260100 (0x3F804)
        self.assertEqual(ref_dot4_u32_u8(0xFFFFFFFF, 0xFFFFFFFF, 0), MAX_DOT4_SUM)
        self.assertEqual(MAX_DOT4_SUM, 260100)

    def test_accumulator_wrapping_modulo_32(self) -> None:
        # Wrap boundary: 260100 + (0xFFFFFFFF - 260100 + 1) = 0 mod 2^32
        acc = (MASK32 - MAX_DOT4_SUM + 1) & MASK32
        lo, carry = ref_dot4_u32_u8_full(0xFFFFFFFF, 0xFFFFFFFF, acc)
        self.assertEqual(lo, 0)
        self.assertEqual(carry, 1)

        # Non-zero remainder wrap
        lo2, carry2 = ref_dot4_u32_u8_full(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(lo2, 260099)
        self.assertEqual(carry2, 1)

    def test_unsigned_vs_signed_divergence(self) -> None:
        # When high bit is set (e.g. 0xFF), unsigned treats it as 255, whereas signed would treat it as -1
        # 0x000000FF * 0x00000001: unsigned is 255, signed would be -1 (0xFFFFFFFF)
        self.assertEqual(ref_dot4_u32_u8(0x000000FF, 0x00000001, 0), 255)
        # 0x000000FF * 0x000000FF: unsigned is 65025, signed would be (-1)*(-1) = 1
        self.assertEqual(ref_dot4_u32_u8(0x000000FF, 0x000000FF, 0), 65025)


if __name__ == "__main__":
    unittest.main()
