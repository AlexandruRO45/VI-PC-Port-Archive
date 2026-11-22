from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mad_i32_i16.

AMD RDNA2 VALU Instruction Set Architecture:
  v_mad_i32_i16 dst, src0, src1, src2 [clamp] [op_sel:x,y]

Format:
  VOP3 (3-operand vector ALU instruction).

Operation:
  Multiplies two signed 16-bit integers, adds a 32-bit signed integer accumulator,
  and writes the 32-bit result to a vector register:
    s0_16   = (src0 >> 16) if op_sel[0] else src0
    s1_16   = (src1 >> 16) if op_sel[1] else src1
    prod32  = int16(s0_16) * int16(s1_16)
    sum32   = prod32 + int32(src2)
    dst.i32 = clamp_i32(sum32) if clamp else (sum32 & 0xFFFFFFFF)

Mathematical Properties and Bounds:
  - src0.i16 and src1.i16 are signed 16-bit integers in [-32768, 32767].
  - Product prod32 is strictly bounded in [-1073709056, 1073741824].
    Maximum positive product: (-32768) * (-32768) = 1,073,741,824 (0x40000000).
    Minimum negative product: (-32768) * 32767 = -1,073,709,056.
  - The product always fits without overflow in signed 32-bit integer arithmetic.
  - Accumulation with 32-bit signed addend wraps modulo 2^32 unless clamp is enabled.
"""

from dataclasses import dataclass, field
import struct
import unittest

# Architectural Constants
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
SIGN16: int = 0x8000
SIGN32: int = 0x8000_0000
MODULO16: int = 0x1_0000
MODULO32: int = 0x1_0000_0000

INT16_MIN: int = -0x8000
INT16_MAX: int = 0x7FFF
UINT16_MAX: int = 0xFFFF
INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF

NUM_VGPRS: int = 256
NUM_SGPRS: int = 106


def to_i16(val: int) -> int:
    """Normalize integer to signed 16-bit integer in [-32768, 32767]."""
    val = val & MASK16
    return val - MODULO16 if (val & SIGN16) else val


def to_u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit word in [0, 65535]."""
    return val & MASK16


def to_i32(val: int) -> int:
    """Normalize integer to signed 32-bit integer in [-2^31, 2^31 - 1]."""
    val = val & MASK32
    return val - MODULO32 if (val & SIGN32) else val


def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword in [0, 2^32 - 1]."""
    return val & MASK32


def clamp_i32(val: int) -> int:
    """Clamp signed integer to 32-bit signed range [INT32_MIN, INT32_MAX]."""
    if val < INT32_MIN:
        return INT32_MIN
    if val > INT32_MAX:
        return INT32_MAX
    return val


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def ref_mul_i16(s0: int, s1: int) -> int:
    """Compute exact signed 16-bit * 16-bit product -> signed 32-bit integer."""
    i0 = to_i16(s0)
    i1 = to_i16(s1)
    return i0 * i1


def ref_mad_i32_i16(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    op_sel: tuple[int, int] = (0, 0),
) -> int:
    """Pure-Python reference oracle for v_mad_i32_i16.

    Returns unsigned 32-bit dword result.
    """
    s0_16 = (s0 >> 16) if op_sel[0] else s0
    s1_16 = (s1 >> 16) if op_sel[1] else s1

    prod = ref_mul_i16(s0_16, s1_16)
    acc = to_i32(s2)
    raw_sum = prod + acc

    if clamp:
        res_signed = clamp_i32(raw_sum)
        return to_u32(res_signed)

    return to_u32(raw_sum)


# ==============================================================================
# Architectural Hardware Model
# ==============================================================================

@dataclass
class RegisterFile:
    """Architectural model representing RDNA2 VGPRs and SGPRs."""
    vgprs: list[int] = field(default_factory=lambda: [0] * NUM_VGPRS)
    sgprs: list[int] = field(default_factory=lambda: [0] * NUM_SGPRS)

    def read_vgpr(self, idx: int) -> int:
        if not (0 <= idx < NUM_VGPRS):
            raise IndexError(f"VGPR index {idx} out of range [0, {NUM_VGPRS - 1}]")
        return self.vgprs[idx] & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_VGPRS):
            raise IndexError(f"VGPR index {idx} out of range [0, {NUM_VGPRS - 1}]")
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        return self.sgprs[idx] & MASK32

    def write_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        self.sgprs[idx] = val & MASK32


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle
# ==============================================================================

class TestReferenceMathematicalOracle(unittest.TestCase):
    """Exhaustive mathematical verification of pure-Python reference oracle."""

    def test_zero_identities(self) -> None:
        self.assertEqual(ref_mad_i32_i16(0, 0, 0), 0)
        self.assertEqual(ref_mad_i32_i16(0, 100, 0), 0)
        self.assertEqual(ref_mad_i32_i16(100, 0, 0), 0)
        self.assertEqual(ref_mad_i32_i16(0, 0, 0x1234_5678), 0x1234_5678)

    def test_one_identity(self) -> None:
        self.assertEqual(ref_mad_i32_i16(1, 42, 0), 42)
        self.assertEqual(ref_mad_i32_i16(42, 1, 100), 142)

    def test_signed_multiplication_signs(self) -> None:
        # positive * positive
        self.assertEqual(ref_mad_i32_i16(50, 40, 0), 2000)
        # positive * negative
        self.assertEqual(ref_mad_i32_i16(50, -40, 0), to_u32(-2000))
        # negative * positive
        self.assertEqual(ref_mad_i32_i16(-50, 40, 0), to_u32(-2000))
        # negative * negative
        self.assertEqual(ref_mad_i32_i16(-50, -40, 0), 2000)

    def test_signed_boundary_products(self) -> None:
        # INT16_MIN * INT16_MIN = (-32768) * (-32768) = 1073741824 = 0x4000_0000
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, INT16_MIN, 0), 0x4000_0000)

        # INT16_MIN * 1 = -32768
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, 1, 0), to_u32(INT16_MIN))

        # INT16_MIN * -1 = 32768 = 0x8000
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, -1, 0), 0x8000)

        # INT16_MAX * INT16_MAX = 32767 * 32767 = 1073676289
        self.assertEqual(ref_mad_i32_i16(INT16_MAX, INT16_MAX, 0), 1073676289)

        # INT16_MIN * INT16_MAX = (-32768) * 32767 = -1073709056
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, INT16_MAX, 0), to_u32(-1073709056))

    def test_accumulation_modulo32_wrapping(self) -> None:
        # (-1) * 1 + 1 = 0
        self.assertEqual(ref_mad_i32_i16(-1, 1, 1), 0)

        # INT32_MAX + 1 wraps to INT32_MIN (0x80000000)
        self.assertEqual(ref_mad_i32_i16(1, 1, INT32_MAX), 0x8000_0000)

        # INT32_MIN - 1 wraps to INT32_MAX (0x7FFFFFFF)
        self.assertEqual(ref_mad_i32_i16(-1, 1, INT32_MIN), 0x7FFF_FFFF)

    def test_clamp_modifier(self) -> None:
        # Overflow without clamp wraps; with clamp clamps to INT32_MAX
        val_unclamped = ref_mad_i32_i16(1, 10, INT32_MAX, clamp=False)
        val_clamped = ref_mad_i32_i16(1, 10, INT32_MAX, clamp=True)
        self.assertEqual(val_clamped, INT32_MAX)
        self.assertNotEqual(val_unclamped, val_clamped)

        # Underflow with clamp clamps to INT32_MIN
        val_unclamped_under = ref_mad_i32_i16(-1, 10, INT32_MIN, clamp=False)
        val_clamped_under = ref_mad_i32_i16(-1, 10, INT32_MIN, clamp=True)
        self.assertEqual(val_clamped_under, to_u32(INT32_MIN))
        self.assertNotEqual(val_unclamped_under, val_clamped_under)

    def test_op_sel_halfword_selection(self) -> None:
        # Packed dword: high 16 = 20, low 16 = 10 -> 0x0014_000A
        packed_s0 = (20 << 16) | 10
        # Packed dword: high 16 = -3, low 16 = 5 -> 0xFFFD_0005
        packed_s1 = (to_u16(-3) << 16) | 5

        # op_sel = (0, 0): uses low words (10 * 5 = 50)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(0, 0)), 50)

        # op_sel = (1, 0): uses high word of s0, low of s1 (20 * 5 = 100)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(1, 0)), 100)

        # op_sel = (0, 1): uses low of s0, high of s1 (10 * -3 = -30)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(0, 1)), to_u32(-30))

        # op_sel = (1, 1): uses high words (20 * -3 = -60)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(1, 1)), to_u32(-60))

    def test_architectural_register_file(self) -> None:
        rf = RegisterFile()
        rf.write_vgpr(0, 0x1234_5678)
        self.assertEqual(rf.read_vgpr(0), 0x1234_5678)

        rf.write_sgpr(10, 0x9ABC_DEF0)
        self.assertEqual(rf.read_sgpr(10), 0x9ABC_DEF0)

        with self.assertRaises(IndexError):
            rf.read_vgpr(256)
        with self.assertRaises(IndexError):
            rf.read_sgpr(106)


if __name__ == "__main__":
    unittest.main()
