from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_dot2_f32_f16.

AMD RDNA VALU Instruction Set Architecture:
  v_dot2_f32_f16 dst, src0, src1, src2 [clamp]

Format:
  VOP3P / VOP3 (3-operand vector ALU instruction with single-precision output).

Operands:
  dst:  32-bit vector register (v[dst]) storing an IEEE 754 single-precision float (binary32).
  src0: 32-bit register (v[s0] or s[s0]) storing a packed pair of IEEE 754 binary16 half-precision floats:
        - src0.f16[0]: low 16 bits [15:0]
        - src0.f16[1]: high 16 bits [31:16]
  src1: 32-bit register (v[s1] or s[s1]) storing a packed pair of IEEE 754 binary16 half-precision floats:
        - src1.f16[0]: low 16 bits [15:0]
        - src1.f16[1]: high 16 bits [31:16]
  src2: 32-bit register (v[s2], s[s2], or literal) storing an IEEE 754 single-precision float accumulator (binary32).

Operation:
  Converts half-precision float pairs into the single-precision domain and performs
  a 2-element dot product accumulated onto a 32-bit float value:
    prod0   = float32(src0.f16[0]) * float32(src1.f16[0])
    prod1   = float32(src0.f16[1]) * float32(src1.f16[1])
    res.f32 = prod0 + prod1 + src2.f32

Clamp Modifier:
  When clamp is set, the final single-precision result is clamped to [0.0, 1.0]:
    if isnan(res) or res < 0.0: res = +0.0
    elif res > 1.0:             res = 1.0
"""

import math
import struct
from typing import Tuple
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF


def f16_to_f32(bits: int) -> float:
    """Convert an IEEE 754 binary16 uint16 bitfield to Python float (float32 compatible)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary16 uint16 bitfield."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f32(bits: int) -> float:
    """Convert an IEEE 754 binary32 uint32 bitfield to Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Convert a Python float to an IEEE 754 binary32 uint32 bitfield."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def pack_f16x2(lo_bits: int, hi_bits: int) -> int:
    """Pack two 16-bit binary16 integers into a 32-bit packed dword."""
    return (lo_bits & MASK16) | ((hi_bits & MASK16) << 16)


def unpack_f16x2(packed32: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit binary16 integers (lo, hi)."""
    return packed32 & MASK16, (packed32 >> 16) & MASK16


def ref_dot2_f32_f16(
    s0_packed: int,
    s1_packed: int,
    s2_f32_bits: int,
    *,
    clamp: bool = False,
) -> float:
    """Reference mathematical oracle for v_dot2_f32_f16 returning Python float."""
    s0_lo, s0_hi = unpack_f16x2(s0_packed)
    s1_lo, s1_hi = unpack_f16x2(s1_packed)

    a0 = f16_to_f32(s0_lo)
    a1 = f16_to_f32(s0_hi)
    b0 = f16_to_f32(s1_lo)
    b1 = f16_to_f32(s1_hi)
    c = bits_to_f32(s2_f32_bits)

    res = (a0 * b0) + (a1 * b1) + c

    if clamp:
        if math.isnan(res) or res < 0.0:
            return 0.0
        if res > 1.0:
            return 1.0
    return res


def ref_dot2_f32_f16_bits(
    s0_packed: int,
    s1_packed: int,
    s2_f32_bits: int,
    *,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_dot2_f32_f16 returning binary32 integer bits."""
    return f32_to_bits(ref_dot2_f32_f16(s0_packed, s1_packed, s2_f32_bits, clamp=clamp))


class TestValuDot2F32F16ReferenceOracle(unittest.TestCase):
    """Architectural reference tests validating mathematical model for v_dot2_f32_f16."""

    def test_basic_dot2_with_accumulator(self) -> None:
        """(1.0, 2.0) . (3.0, 4.0) + 5.0 = 1*3 + 2*4 + 5 = 16.0"""
        s0 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(2.0))
        s1 = pack_f16x2(f16_to_bits(3.0), f16_to_bits(4.0))
        s2 = f32_to_bits(5.0)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 16.0, places=5)
        self.assertEqual(ref_dot2_f32_f16_bits(s0, s1, s2), f32_to_bits(16.0))

    def test_zero_sources_accumulator_pass_through(self) -> None:
        """(0.0, 0.0) . (1.0, 1.0) + 42.5 = 42.5"""
        s0 = pack_f16x2(f16_to_bits(0.0), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s2 = f32_to_bits(42.5)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 42.5, places=5)

    def test_negative_elements_and_cancellation(self) -> None:
        """(-1.5, 2.0) . (2.0, -3.0) + 10.0 = -3.0 + (-6.0) + 10.0 = 1.0"""
        s0 = pack_f16x2(f16_to_bits(-1.5), f16_to_bits(2.0))
        s1 = pack_f16x2(f16_to_bits(2.0), f16_to_bits(-3.0))
        s2 = f32_to_bits(10.0)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 1.0, places=5)

    def test_orthogonal_vectors(self) -> None:
        """(1.0, 0.0) . (0.0, 1.0) + 7.0 = 0 + 7.0 = 7.0"""
        s0 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(0.0), f16_to_bits(1.0))
        s2 = f32_to_bits(7.0)
        res = ref_dot2_f32_f16(s0, s1, s2)
        self.assertAlmostEqual(res, 7.0, places=5)

    def test_clamp_upper_saturation(self) -> None:
        """Values greater than 1.0 saturate to 1.0 with clamp=True."""
        s0 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(1.0))
        s2 = f32_to_bits(0.0)
        # 1*1 + 1*1 = 2.0 -> clamped to 1.0
        res = ref_dot2_f32_f16(s0, s1, s2, clamp=True)
        self.assertEqual(res, 1.0)
        self.assertEqual(ref_dot2_f32_f16_bits(s0, s1, s2, clamp=True), f32_to_bits(1.0))

    def test_clamp_lower_saturation(self) -> None:
        """Negative values saturate to 0.0 with clamp=True."""
        s0 = pack_f16x2(f16_to_bits(-2.0), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(0.0))
        s2 = f32_to_bits(0.0)
        # -2*1 = -2.0 -> clamped to 0.0
        res = ref_dot2_f32_f16(s0, s1, s2, clamp=True)
        self.assertEqual(res, 0.0)
        self.assertEqual(ref_dot2_f32_f16_bits(s0, s1, s2, clamp=True), f32_to_bits(0.0))

    def test_clamp_in_range_preservation(self) -> None:
        """Values strictly inside [0.0, 1.0] are preserved with clamp=True."""
        s0 = pack_f16x2(f16_to_bits(0.5), f16_to_bits(0.0))
        s1 = pack_f16x2(f16_to_bits(1.0), f16_to_bits(0.0))
        s2 = f32_to_bits(0.25)
        # 0.5 * 1.0 + 0.25 = 0.75
        res = ref_dot2_f32_f16(s0, s1, s2, clamp=True)
        self.assertAlmostEqual(res, 0.75, places=5)


if __name__ == "__main__":
    unittest.main()
