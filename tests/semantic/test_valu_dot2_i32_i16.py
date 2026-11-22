from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_dot2_i32_i16.

AMD RDNA VALU Instruction Set Architecture:
  v_dot2_i32_i16 dst, src0, src1, src2 [clamp]

Format:
  VOP3P / VOP3 (3-operand vector ALU instruction with signed 32-bit output).

Operands:
  dst:  32-bit vector register (v[dst]) storing a two's-complement signed 32-bit integer.
  src0: 32-bit register (v[s0] or s[s0]) storing a packed pair of signed 16-bit integers:
        - src0.i16[0]: low 16 bits [15:0]
        - src0.i16[1]: high 16 bits [31:16]
  src1: 32-bit register (v[s1] or s[s1]) storing a packed pair of signed 16-bit integers:
        - src1.i16[0]: low 16 bits [15:0]
        - src1.i16[1]: high 16 bits [31:16]
  src2: 32-bit register (v[s2], s[s2], or literal) storing a signed 32-bit integer accumulator.

Operation:
  Sign-extends each 16-bit halfword to a 32-bit signed integer, computes two products,
  and sums them with the 32-bit signed accumulator:
    prod0   = int32(src0.i16[0]) * int32(src1.i16[0])
    prod1   = int32(src0.i16[1]) * int32(src1.i16[1])
    full    = prod0 + prod1 + int32(src2)

Clamp Modifier:
  When clamp is set, the final result saturates to signed 32-bit bounds [-2^31, 2^31 - 1]:
    if full > 0x7FFFFFFF:       res = 0x7FFFFFFF
    elif full < -0x80000000:    res = 0x80000000
    else:                       res = full & 0xFFFFFFFF
  When unclamped, the full sum wraps modulo 2^32.
"""

from typing import Tuple
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF

INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_INT_MIN: int = 0x8000_0000
UINT32_INT_MAX: int = 0x7FFF_FFFF


def s16(val: int) -> int:
    """Sign-extend a 16-bit integer to Python signed integer."""
    val &= MASK16
    return val - 0x1_0000 if (val & 0x8000) else val


def s32(val: int) -> int:
    """Sign-extend a 32-bit integer to Python signed integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & 0x8000_0000) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def pack_i16x2(lo: int, hi: int) -> int:
    """Pack two signed 16-bit integers into an unsigned 32-bit dword."""
    return (lo & MASK16) | ((hi & MASK16) << 16)


def unpack_i16x2(packed32: int) -> Tuple[int, int]:
    """Unpack an unsigned 32-bit dword into two signed 16-bit integers (lo, hi)."""
    return s16(packed32 & MASK16), s16((packed32 >> 16) & MASK16)


def ref_dot2_i32_i16(
    s0_packed: int,
    s1_packed: int,
    s2_i32: int,
    *,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_dot2_i32_i16 returning uint32 bits."""
    a0, a1 = unpack_i16x2(s0_packed)
    b0, b1 = unpack_i16x2(s1_packed)
    c = s32(s2_i32)

    full = (a0 * b0) + (a1 * b1) + c

    if clamp:
        if full > INT32_MAX:
            return UINT32_INT_MAX
        if full < INT32_MIN:
            return UINT32_INT_MIN
        return u32(full)
    return u32(full)


class TestValuDot2I32I16ReferenceOracle(unittest.TestCase):
    """Architectural reference tests validating mathematical model for v_dot2_i32_i16."""

    def test_basic_positive_dot_product(self) -> None:
        """(2, 3) . (4, 5) + 6 = 8 + 15 + 6 = 29"""
        s0 = pack_i16x2(2, 3)
        s1 = pack_i16x2(4, 5)
        s2 = 6
        res = ref_dot2_i32_i16(s0, s1, s2)
        self.assertEqual(res, 29)

    def test_signed_negative_elements(self) -> None:
        """(-2, 3) . (4, -5) + 10 = -8 + (-15) + 10 = -13"""
        s0 = pack_i16x2(-2, 3)
        s1 = pack_i16x2(4, -5)
        s2 = 10
        res = ref_dot2_i32_i16(s0, s1, s2)
        self.assertEqual(res, u32(-13))

    def test_zero_operands(self) -> None:
        """(0, 0) . (100, -200) + 12345 = 12345"""
        s0 = pack_i16x2(0, 0)
        s1 = pack_i16x2(100, -200)
        s2 = 12345
        res = ref_dot2_i32_i16(s0, s1, s2)
        self.assertEqual(res, 12345)

    def test_unclamped_modulo_wrapping_overflow(self) -> None:
        """Two maximum products (-32768 * -32768 * 2) = 2^31 = 2147483648 -> wraps to 0x80000000."""
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(-32768, -32768)
        s2 = 0
        # (-32768)^2 + (-32768)^2 = 2^30 + 2^30 = 2^31 = 0x80000000
        res_unclamped = ref_dot2_i32_i16(s0, s1, s2, clamp=False)
        self.assertEqual(res_unclamped, 0x8000_0000)

    def test_clamp_positive_saturation(self) -> None:
        """Positive overflow saturates to 0x7FFFFFFF (INT32_MAX) when clamp=True."""
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(-32768, -32768)
        s2 = 0
        res_clamped = ref_dot2_i32_i16(s0, s1, s2, clamp=True)
        self.assertEqual(res_clamped, UINT32_INT_MAX)

    def test_clamp_negative_saturation(self) -> None:
        """Negative underflow saturates to 0x80000000 (INT32_MIN) when clamp=True."""
        # (-32768 * 32767) * 2 = -2 * 1073709056 = -2147418112
        # Adding accumulator -100000 causes signed underflow < -2147483648
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(32767, 32767)
        s2 = u32(-100_000)
        res_clamped = ref_dot2_i32_i16(s0, s1, s2, clamp=True)
        self.assertEqual(res_clamped, UINT32_INT_MIN)


if __name__ == "__main__":
    unittest.main()
