from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for AMD RDNA2 v_sad_u32_u16.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_sad_u32_u16 (VOP3 3-operand vector ALU instruction):
    Computes an unsigned 32-bit sum of absolute differences with mixed 16-bit and 32-bit operands:
        s0.u16 = (src0 >> (16 if op_sel[0] else 0)) & 0xFFFF
        s1.u16 = (src1 >> (16 if op_sel[1] else 0)) & 0xFFFF
        diff.u32 = abs(s0.u16 - s1.u16)
        full_sum = diff.u32 + (src2.u32 & 0xFFFFFFFF)
        dst.u32 = min(0xFFFFFFFF, full_sum) if clamp else (full_sum mod 2^32)
        carry_out = 1 if (full_sum >= 2^32) else 0

    Outputs:
        Vector Destination (VDST): 32-bit VGPR storing dst.u32.
        Carry generation: Tracks whether full_sum exceeded 2^32 (overflowing 32-bit unsigned range).

    Mathematical Bounds:
        - s0.u16 in [0, 65535] (0x0000 to 0xFFFF)
        - s1.u16 in [0, 65535] (0x0000 to 0xFFFF)
        - src2.u32 in [0, 2^32 - 1] (0 to 0xFFFFFFFF)
        - Maximum difference: |65535 - 0| = 65535 = 0xFFFF
        - Maximum full sum: 0xFFFF + 0xFFFFFFFF = 0x10000FFFE
        - Therefore, carry_out is strictly binary {0, 1}.

    Modifiers:
        - op_sel[0]: Selects low (0) or high (1) 16 bits of src0 register.
        - op_sel[1]: Selects low (0) or high (1) 16 bits of src1 register.
        - clamp: When enabled, saturates the destination result to 0xFFFFFFFF on overflow.

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


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def ref_extract_u16(val: int, high_halfword: bool = False) -> int:
    """Extract selected 16-bit unsigned halfword from a 32-bit word."""
    shift = 16 if high_halfword else 0
    return (val >> shift) & MASK16


def ref_abs_diff_u16(s0: int, s1: int) -> int:
    """Pure-Python reference oracle for 16-bit unsigned absolute difference: |s0 - s1|."""
    u0 = s0 & MASK16
    u1 = s1 & MASK16
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u32_u16(
    s0: int,
    s1: int,
    s2: int,
    op_sel0: bool = False,
    op_sel1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_sad_u32_u16:

    dst = min(0xFFFFFFFF, |s0.u16 - s1.u16| + s2.u32) if clamp else (|s0.u16 - s1.u16| + s2.u32) & 0xFFFFFFFF
    """
    u0 = ref_extract_u16(s0, op_sel0)
    u1 = ref_extract_u16(s1, op_sel1)
    diff = ref_abs_diff_u16(u0, u1)
    acc = s2 & MASK32
    full = diff + acc
    if clamp and full > MASK32:
        return MASK32
    return full & MASK32


def ref_sad_u32_u16_full(
    s0: int,
    s1: int,
    s2: int,
    op_sel0: bool = False,
    op_sel1: bool = False,
    clamp: bool = False,
) -> Tuple[int, int, int]:
    """Return (dst_u32, carry_out, full_sum64) for v_sad_u32_u16."""
    u0 = ref_extract_u16(s0, op_sel0)
    u1 = ref_extract_u16(s1, op_sel1)
    diff = ref_abs_diff_u16(u0, u1)
    acc = s2 & MASK32
    full = diff + acc
    carry = (full >> 32) & 1
    dst = MASK32 if (clamp and full > MASK32) else (full & MASK32)
    return dst, carry, full


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestVSadU32U16Oracle(unittest.TestCase):
    """Test suite verifying mathematical bounds, properties, and edge cases of the oracle."""

    def test_oracle_zero_inputs(self) -> None:
        """All-zero inputs yield zero result."""
        self.assertEqual(ref_sad_u32_u16(0, 0, 0), 0)
        dst, carry, full = ref_sad_u32_u16_full(0, 0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 0)
        self.assertEqual(full, 0)

    def test_oracle_identical_inputs(self) -> None:
        """When s0 == s1, absolute difference is zero and result is accumulator s2."""
        patterns = [0x00000000, 0x00001234, 0x0000FFFF, 0x12345678, 0xFFFFFFFF]
        for pat in patterns:
            acc = 0xCAFEBABE
            self.assertEqual(ref_sad_u32_u16(pat, pat, acc), acc)

    def test_oracle_maximum_difference(self) -> None:
        """Max difference between 16-bit operands (|0xFFFF - 0x0000| = 65535)."""
        s0 = 0x0000FFFF
        s1 = 0x00000000
        self.assertEqual(ref_sad_u32_u16(s0, s1, 0), 65535)

    def test_oracle_op_sel_halfword_selection(self) -> None:
        """Verify op_sel modifiers properly select low vs high halfwords."""
        s0 = 0x1234_5678
        s1 = 0x1234_0000

        # op_sel0=False (low: 0x5678), op_sel1=True (high: 0x1234)
        # diff = |0x5678 - 0x1234| = 0x4444 = 17476
        res = ref_sad_u32_u16(s0, s1, 0, op_sel0=False, op_sel1=True)
        self.assertEqual(res, 0x4444)

        # op_sel0=True (high: 0x1234), op_sel1=True (high: 0x1234) -> diff = 0
        res_zero = ref_sad_u32_u16(s0, s1, 0, op_sel0=True, op_sel1=True)
        self.assertEqual(res_zero, 0)

    def test_oracle_commutativity(self) -> None:
        """Commutativity: ref_sad_u32_u16(s0, s1, s2) == ref_sad_u32_u16(s1, s0, s2)."""
        test_pairs = [
            (0x1234, 0x5678),
            (0xFFFF, 0x0000),
            (0x8000, 0x0001),
            (0x7FFF, 0x8000),
        ]
        acc = 0x1000
        for s0, s1 in test_pairs:
            self.assertEqual(ref_sad_u32_u16(s0, s1, acc), ref_sad_u32_u16(s1, s0, acc))

    def test_oracle_accumulation_with_wrap(self) -> None:
        """32-bit truncation when diff + s2 exceeds 0xFFFFFFFF."""
        s0 = 0xFFFF
        s1 = 0x0000
        # diff = 65535 (0xFFFF); acc = 0xFFFFFFFF
        # 0xFFFF + 0xFFFFFFFF = 0x1_0000_FFFE -> wrapped = 0x0000FFFE
        dst, carry, full = ref_sad_u32_u16_full(s0, s1, 0xFFFF_FFFF)
        self.assertEqual(dst, 0x0000_FFFE)
        self.assertEqual(carry, 1)
        self.assertEqual(full, 0x1_0000_FFFE)

    def test_oracle_clamp_modifier(self) -> None:
        """Clamp modifier saturates to 0xFFFFFFFF on overflow instead of wrapping."""
        s0 = 0xFFFF
        s1 = 0x0000
        acc = 0xFFFF_FFFF
        # Without clamp -> 0x0000FFFE
        self.assertEqual(ref_sad_u32_u16(s0, s1, acc, clamp=False), 0x0000_FFFE)
        # With clamp -> saturated to 0xFFFFFFFF
        self.assertEqual(ref_sad_u32_u16(s0, s1, acc, clamp=True), 0xFFFF_FFFF)


if __name__ == "__main__":
    unittest.main()
