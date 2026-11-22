from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for AMD RDNA2 v_mad_u32_u16.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_mad_u32_u16 (VOP3 3-operand vector ALU instruction):
    Computes an unsigned 32-bit multiply-accumulate operation with mixed 16-bit and 32-bit operands:
        s0.u16 = (src0 >> (16 if op_sel[0] else 0)) & 0xFFFF
        s1.u16 = (src1 >> (16 if op_sel[1] else 0)) & 0xFFFF
        prod.u32 = s0.u16 * s1.u16
        full_sum = prod.u32 + (src2.u32 & 0xFFFFFFFF)
        dst.u32 = min(0xFFFFFFFF, full_sum) if clamp else (full_sum mod 2^32)
        carry_out = 1 if (full_sum >= 2^32) else 0

    Outputs:
        Vector Destination (VDST): 32-bit VGPR storing dst.u32.
        Carry generation: Tracks whether full_sum exceeded 2^32 (overflowing 32-bit unsigned range).

    Mathematical Bounds:
        - s0.u16 in [0, 65535] (0x0000 to 0xFFFF)
        - s1.u16 in [0, 65535] (0x0000 to 0xFFFF)
        - src2.u32 in [0, 2^32 - 1] (0 to 0xFFFFFFFF)
        - Maximum product: 65535^2 = 4,294,836,225 = 0xFFFE0001 (strictly < 2^32)
        - Maximum full sum: 0xFFFE0001 + 0xFFFFFFFF = 0x1FFFE0000 (strictly < 2^33)
        - Therefore, carry_out is strictly binary {0, 1}.

    Modifiers:
        - op_sel[0]: Selects low (0) or high (1) 16 bits of src0 register.
        - op_sel[1]: Selects low (0) or high (1) 16 bits of src1 register.
        - clamp: When enabled, saturates the destination result to 0xFFFFFFFF on overflow.

    Signed Variant (v_mad_i32_i16):
        Interprets the selected 16-bit halfwords as signed two's complement integers,
        multiplies them into a signed 32-bit product, and accumulates with signed 32-bit src2,
        with optional signed clamping to [-2^31, 2^31 - 1].

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from typing import Sequence, Tuple
import math
import struct
import unittest

# Architectural bitmasks and constants
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

SIGN_BIT_16: int = 0x8000
SIGN_BIT_32: int = 0x8000_0000

MODULO_16: int = 1 << 16
MODULO_32: int = 1 << 32

CLAMP_MAX_U32: int = 0xFFFF_FFFF
CLAMP_MIN_I32: int = -(1 << 31)
CLAMP_MAX_I32: int = (1 << 31) - 1


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned halfword as a signed two's-complement integer."""
    val &= MASK16
    return val - MODULO_16 if (val & SIGN_BIT_16) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - MODULO_32 if (val & SIGN_BIT_32) else val


def extract_halfword(word32: int, high: bool) -> int:
    """Extract lower (high=False) or upper (high=True) 16-bit halfword from a 32-bit dword."""
    shift = 16 if high else 0
    return (word32 >> shift) & MASK16


def pack_halfwords(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords (lo, hi) into a single 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def ref_mad_u32_u16(
    src0: int,
    src1: int,
    src2: int,
    *,
    op_sel: Tuple[bool, bool] = (False, False),
    clamp: bool = False,
) -> Tuple[int, int]:
    """Bit-accurate reference mathematical oracle for v_mad_u32_u16.

    Computes:
        s0 = extract_halfword(src0, op_sel[0])
        s1 = extract_halfword(src1, op_sel[1])
        prod = s0 * s1
        full_sum = prod + (src2 & 0xFFFFFFFF)
        dst = min(0xFFFFFFFF, full_sum) if clamp else (full_sum & 0xFFFFFFFF)
        carry_out = 1 if full_sum >= 2^32 else 0

    Returns:
        (dst.u32, carry_out)
    """
    s0 = extract_halfword(src0, op_sel[0])
    s1 = extract_halfword(src1, op_sel[1])
    prod = s0 * s1
    acc = src2 & MASK32
    full_sum = prod + acc
    carry = 1 if full_sum >= MODULO_32 else 0
    if clamp:
        dst = min(CLAMP_MAX_U32, full_sum)
    else:
        dst = full_sum & MASK32
    return dst, carry


def ref_mad_i32_i16(
    src0: int,
    src1: int,
    src2: int,
    *,
    op_sel: Tuple[bool, bool] = (False, False),
    clamp: bool = False,
) -> Tuple[int, int]:
    """Bit-accurate reference mathematical oracle for signed v_mad_i32_i16.

    Returns:
        (dst_i32_as_u32, overflow)
    """
    s0 = s16(extract_halfword(src0, op_sel[0]))
    s1 = s16(extract_halfword(src1, op_sel[1]))
    prod = s0 * s1
    acc = s32(src2)
    full_sum = prod + acc
    overflow = 1 if (full_sum < CLAMP_MIN_I32 or full_sum > CLAMP_MAX_I32) else 0
    if clamp:
        clamped = max(CLAMP_MIN_I32, min(CLAMP_MAX_I32, full_sum))
        dst = clamped & MASK32
    else:
        dst = full_sum & MASK32
    return dst, overflow


class TestValuMadU32U16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_multiplication_commutativity(self) -> None:
        pairs = [
            (0, 0),
            (1, 42),
            (0x1234, 0x5678),
            (MASK16, 1),
            (MASK16, MASK16),
            (0x8000, 2),
            (0x7FFF, 0xFFFF),
        ]
        s2 = 0xCAFE_BABE
        for a, b in pairs:
            res_ab = ref_mad_u32_u16(a, b, s2)
            res_ba = ref_mad_u32_u16(b, a, s2)
            self.assertEqual(res_ab, res_ba, f"Commutativity failed for ({hex(a)}, {hex(b)})")

    def test_multiplicative_zero_identity(self) -> None:
        for s2 in [0, 1, MASK16, MASK32, 0x1234_5678]:
            self.assertEqual(ref_mad_u32_u16(0, 0x1234, s2), (s2 & MASK32, 0))
            self.assertEqual(ref_mad_u32_u16(0x5678, 0, s2), (s2 & MASK32, 0))
            self.assertEqual(ref_mad_u32_u16(0, 0, s2), (s2 & MASK32, 0))

    def test_multiplicative_one_identity(self) -> None:
        for s1 in [0, 1, 0x1234, MASK16]:
            for s2 in [0, 100, MASK32 - 10, MASK32]:
                expected_sum = (s1 & MASK16) + (s2 & MASK32)
                expected_dst = expected_sum & MASK32
                expected_carry = 1 if expected_sum >= MODULO_32 else 0
                self.assertEqual(ref_mad_u32_u16(1, s1, s2), (expected_dst, expected_carry))
                self.assertEqual(ref_mad_u32_u16(s1, 1, s2), (expected_dst, expected_carry))

    def test_additive_zero_identity(self) -> None:
        test_pairs = [
            (0, 0),
            (10, 20),
            (0x100, 0x200),
            (MASK16, 1),
            (MASK16, MASK16),
        ]
        for s0, s1 in test_pairs:
            expected_prod = (s0 & MASK16) * (s1 & MASK16)
            # Carry is strictly 0 because max 16x16 product = 0xFFFE0001 < 2^32
            self.assertEqual(ref_mad_u32_u16(s0, s1, 0), (expected_prod, 0))

    def test_max_product_bounds(self) -> None:
        # 0xFFFF * 0xFFFF = (2^16 - 1)^2 = 2^32 - 2^17 + 1 = 0xFFFE_0001
        expected_prod = 0xFFFE_0001
        self.assertEqual(ref_mad_u32_u16(MASK16, MASK16, 0), (expected_prod, 0))

    def test_op_sel_halfword_extraction(self) -> None:
        src0 = pack_halfwords(0x1111, 0x2222)  # lo=0x1111, hi=0x2222
        src1 = pack_halfwords(0x3333, 0x4444)  # lo=0x3333, hi=0x4444
        src2 = 10

        # op_sel = (False, False) -> 0x1111 * 0x3333 + 10
        res00, _ = ref_mad_u32_u16(src0, src1, src2, op_sel=(False, False))
        self.assertEqual(res00, 0x1111 * 0x3333 + 10)

        # op_sel = (True, False) -> 0x2222 * 0x3333 + 10
        res10, _ = ref_mad_u32_u16(src0, src1, src2, op_sel=(True, False))
        self.assertEqual(res10, 0x2222 * 0x3333 + 10)

        # op_sel = (False, True) -> 0x1111 * 0x4444 + 10
        res01, _ = ref_mad_u32_u16(src0, src1, src2, op_sel=(False, True))
        self.assertEqual(res01, 0x1111 * 0x4444 + 10)

        # op_sel = (True, True) -> 0x2222 * 0x4444 + 10
        res11, _ = ref_mad_u32_u16(src0, src1, src2, op_sel=(True, True))
        self.assertEqual(res11, 0x2222 * 0x4444 + 10)

    def test_carry_generation_boundary(self) -> None:
        # s0 = 0x1000, s1 = 0x1000 -> prod = 0x0100_0000
        # Case A: sum == 2^32 - 1 -> dst = 0xFFFFFFFF, carry = 0
        s0 = 0x1000
        s1 = 0x1000
        s2_a = 0xFF00_0000 - 1  # 0xFEFF_FFFF
        dst_a, carry_a = ref_mad_u32_u16(s0, s1, s2_a)
        self.assertEqual(dst_a, MASK32)
        self.assertEqual(carry_a, 0)

        # Case B: sum == 2^32 -> dst = 0, carry = 1
        s2_b = 0xFF00_0000
        dst_b, carry_b = ref_mad_u32_u16(s0, s1, s2_b)
        self.assertEqual(dst_b, 0)
        self.assertEqual(carry_b, 1)

        # Case C: sum == 2^32 + 5 -> dst = 5, carry = 1
        s2_c = 0xFF00_0005
        dst_c, carry_c = ref_mad_u32_u16(s0, s1, s2_c)
        self.assertEqual(dst_c, 5)
        self.assertEqual(carry_c, 1)

    def test_clamp_saturation_on_overflow(self) -> None:
        s0 = 0x1000
        s1 = 0x1000  # prod = 0x0100_0000
        s2 = 0xFF00_0005  # full_sum = 0x1_0000_0005

        # Without clamp: wraps around
        dst_unclamped, carry_unclamped = ref_mad_u32_u16(s0, s1, s2, clamp=False)
        self.assertEqual(dst_unclamped, 5)
        self.assertEqual(carry_unclamped, 1)

        # With clamp: saturates to 0xFFFFFFFF
        dst_clamped, carry_clamped = ref_mad_u32_u16(s0, s1, s2, clamp=True)
        self.assertEqual(dst_clamped, CLAMP_MAX_U32)
        self.assertEqual(carry_clamped, 1)

    def test_max_saturation_extremes(self) -> None:
        # s0 = 0xFFFF, s1 = 0xFFFF, s2 = 0xFFFFFFFF
        # prod = 0xFFFE_0001
        # full_sum = 0xFFFE_0001 + 0xFFFFFFFF = 0x1_FFFE_0000
        s0 = MASK16
        s1 = MASK16
        s2 = MASK32

        dst_wrap, carry_wrap = ref_mad_u32_u16(s0, s1, s2, clamp=False)
        self.assertEqual(dst_wrap, 0xFFFE_0000)
        self.assertEqual(carry_wrap, 1)

        dst_sat, carry_sat = ref_mad_u32_u16(s0, s1, s2, clamp=True)
        self.assertEqual(dst_sat, CLAMP_MAX_U32)
        self.assertEqual(carry_sat, 1)

    def test_signed_vs_unsigned_distinction(self) -> None:
        # 0xFFFF in 16-bit:
        # Unsigned: 65535
        # Signed: -1
        # s0 = 0xFFFF, s1 = 0xFFFF, s2 = 0
        # Unsigned: 65535 * 65535 = 0xFFFE_0001
        # Signed: (-1) * (-1) + 0 = 1
        dst_u, carry_u = ref_mad_u32_u16(MASK16, MASK16, 0)
        dst_s, overflow_s = ref_mad_i32_i16(MASK16, MASK16, 0)
        self.assertEqual(dst_u, 0xFFFE_0001)
        self.assertEqual(carry_u, 0)
        self.assertEqual(dst_s, 1)
        self.assertEqual(overflow_s, 0)

    def test_dsp_dot_product_fir_accumulation(self) -> None:
        # Accumulating a 4-tap FIR filter: y = sum(x[i] * h[i]) + acc
        x = [100, 200, 300, 400]
        h = [5, 15, 25, 35]
        acc = 1000

        expected = acc
        current_acc = acc
        for xi, hi in zip(x, h):
            expected += xi * hi
            current_acc, carry = ref_mad_u32_u16(xi, hi, current_acc)
            self.assertEqual(carry, 0)

        self.assertEqual(current_acc, expected)


if __name__ == "__main__":
    unittest.main()
