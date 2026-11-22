from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mad_i64_i32.

AMD RDNA2 VALU Instruction Set Architecture:
  v_mad_i64_i32 dst, sdst, src0, src1, src2

Format:
  VOP3B / VOP3 (3-operand vector ALU instruction with scalar destination).

Operation:
  Multiplies two signed 32-bit integers, accumulates a 64-bit integer, and writes
  the 64-bit result to a vector register pair while generating a carry-out bitmask
  in a scalar register:
    prod64  = int64(src0.i32) * int64(src1.i32)
    sum64   = uint64(prod64) + uint64(src2.i64)
    dst.i64 = sum64 & 0xFFFFFFFFFFFFFFFF
    sdst    = 1 if (uint64(prod64) + uint64(src2.i64) >= 2^64) else 0

Operands:
  - dst: 64-bit vector register pair v[d:d+1] (d in [0, 254]).
  - sdst: Scalar condition / carry destination register (s[s:s+1] or VCC).
  - src0: 32-bit signed integer (VGPR, SGPR, literal, or inline constant).
  - src1: 32-bit signed integer (VGPR, SGPR, literal, or inline constant).
  - src2: 64-bit signed/unsigned integer (VGPR pair, SGPR pair, or literal).
"""

from dataclasses import dataclass, field
import struct
import unittest

# Architectural Constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN32: int = 0x8000_0000
SIGN64: int = 0x8000_0000_0000_0000
MODULO32: int = 0x1_0000_0000
MODULO64: int = 0x1_0000_0000_0000_0000

INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
INT64_MIN: int = -0x8000_0000_0000_0000
INT64_MAX: int = 0x7FFF_FFFF_FFFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF
UINT64_MAX: int = 0xFFFF_FFFF_FFFF_FFFF

NUM_VGPRS: int = 256
NUM_SGPRS: int = 106


def to_i32(val: int) -> int:
    """Normalize integer to signed 32-bit integer in [-2^31, 2^31 - 1]."""
    val = val & MASK32
    return val - MODULO32 if (val & SIGN32) else val


def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword in [0, 2^32 - 1]."""
    return val & MASK32


def to_i64(val: int) -> int:
    """Normalize integer to signed 64-bit integer in [-2^63, 2^63 - 1]."""
    val = val & MASK64
    return val - MODULO64 if (val & SIGN64) else val


def to_u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword in [0, 2^64 - 1]."""
    return val & MASK64


def split_u64(val: int) -> tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32) dwords."""
    u = val & MASK64
    return u & MASK32, (u >> 32) & MASK32


def combine_u32(lo: int, hi: int) -> int:
    """Combine two 32-bit dwords into unsigned 64-bit qword."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


# ==============================================================================
# Reference Mathematical Oracles
# ==============================================================================

def ref_mul_i32(s0: int, s1: int) -> int:
    """Compute exact signed 32-bit * 32-bit multiplication -> signed 64-bit product."""
    i0 = to_i32(s0)
    i1 = to_i32(s1)
    return i0 * i1


def ref_mad_i64_i32(s0: int, s1: int, s2: int) -> tuple[int, int]:
    """Pure-Python reference oracle for v_mad_i64_i32: (dst_u64, carry_out).

    Computes:
      prod64 = int64(s0.i32) * int64(s1.i32)
      raw_sum = (prod64 & MASK64) + (s2 & MASK64)
      dst = raw_sum & MASK64
      carry = 1 if raw_sum >= 2^64 else 0
    """
    prod = ref_mul_i32(s0, s1)
    prod_u64 = prod & MASK64
    s2_u64 = s2 & MASK64
    raw_sum = prod_u64 + s2_u64
    dst = raw_sum & MASK64
    carry = 1 if raw_sum >= MODULO64 else 0
    return dst, carry


def ref_mad_i64_i32_split(s0: int, s1: int, s2_lo: int, s2_hi: int) -> tuple[int, int, int]:
    """Reference mathematical oracle with split 32-bit dwords for s2.

    Returns (dst_lo, dst_hi, carry_out).
    """
    s2 = combine_u32(s2_lo, s2_hi)
    dst_u64, carry = ref_mad_i64_i32(s0, s1, s2)
    lo, hi = split_u64(dst_u64)
    return lo, hi, carry


# ==============================================================================
# Architectural Hardware Model
# ==============================================================================

@dataclass
class RegisterFile:
    """Architectural model representing RDNA2 VGPRs and SGPRs for scalar simulation."""
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

    def read_vgpr_64(self, idx: int) -> int:
        if not (0 <= idx < NUM_VGPRS - 1):
            raise IndexError(f"VGPR 64-bit pair index {idx} out of range [0, {NUM_VGPRS - 2}]")
        return combine_u32(self.vgprs[idx], self.vgprs[idx + 1])

    def write_vgpr_64(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_VGPRS - 1):
            raise IndexError(f"VGPR 64-bit pair index {idx} out of range [0, {NUM_VGPRS - 2}]")
        lo, hi = split_u64(val)
        self.vgprs[idx] = lo
        self.vgprs[idx + 1] = hi

    def read_sgpr(self, idx: int) -> int:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        return self.sgprs[idx] & MASK32

    def write_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        self.sgprs[idx] = val & MASK32

    def read_sgpr_64(self, idx: int) -> int:
        if not (0 <= idx < NUM_SGPRS - 1):
            raise IndexError(f"SGPR 64-bit pair index {idx} out of range [0, {NUM_SGPRS - 2}]")
        return combine_u32(self.sgprs[idx], self.sgprs[idx + 1])

    def write_sgpr_64(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_SGPRS - 1):
            raise IndexError(f"SGPR 64-bit pair index {idx} out of range [0, {NUM_SGPRS - 2}]")
        lo, hi = split_u64(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle & Architectural Model
# ==============================================================================

class TestReferenceMathematicalOracle(unittest.TestCase):
    """Exhaustive mathematical verification of pure-Python reference oracle."""

    def test_zero_identities(self) -> None:
        # 0 * x + 0 = 0, carry = 0
        self.assertEqual(ref_mad_i64_i32(0, 0, 0), (0, 0))
        self.assertEqual(ref_mad_i64_i32(0, 12345, 0), (0, 0))
        self.assertEqual(ref_mad_i64_i32(12345, 0, 0), (0, 0))
        self.assertEqual(ref_mad_i64_i32(0, 0, 0x1234_5678_9ABC_DEF0), (0x1234_5678_9ABC_DEF0, 0))

    def test_one_identity(self) -> None:
        # 1 * x + 0 = x, carry = 0
        self.assertEqual(ref_mad_i64_i32(1, 42, 0), (42, 0))
        self.assertEqual(ref_mad_i64_i32(42, 1, 100), (142, 0))

    def test_signed_multiplication_signs(self) -> None:
        # positive * positive
        dst, c = ref_mad_i64_i32(100, 200, 0)
        self.assertEqual(dst, 20000)
        self.assertEqual(c, 0)

        # positive * negative
        dst, c = ref_mad_i64_i32(100, -200, 0)
        expected = to_u64(-20000)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

        # negative * positive
        dst, c = ref_mad_i64_i32(-100, 200, 0)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

        # negative * negative
        dst, c = ref_mad_i64_i32(-100, -200, 0)
        self.assertEqual(dst, 20000)
        self.assertEqual(c, 0)

    def test_signed_boundary_products(self) -> None:
        # INT32_MIN * INT32_MIN = (-2^31) * (-2^31) = 2^62 = 0x4000_0000_0000_0000
        dst, c = ref_mad_i64_i32(INT32_MIN, INT32_MIN, 0)
        self.assertEqual(dst, 0x4000_0000_0000_0000)
        self.assertEqual(c, 0)

        # INT32_MIN * 1 = -2^31 = 0xFFFF_FFFF_8000_0000
        dst, c = ref_mad_i64_i32(INT32_MIN, 1, 0)
        self.assertEqual(dst, to_u64(INT32_MIN))
        self.assertEqual(c, 0)

        # INT32_MIN * -1 = 2^31 = 0x0000_0000_8000_0000
        dst, c = ref_mad_i64_i32(INT32_MIN, -1, 0)
        self.assertEqual(dst, 0x8000_0000)
        self.assertEqual(c, 0)

        # INT32_MAX * INT32_MAX = (2^31 - 1)^2 = 2^62 - 2^32 + 1
        expected = (INT32_MAX * INT32_MAX) & MASK64
        dst, c = ref_mad_i64_i32(INT32_MAX, INT32_MAX, 0)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

        # INT32_MIN * INT32_MAX = (-2^31) * (2^31 - 1) = -2^62 + 2^31
        expected = (INT32_MIN * INT32_MAX) & MASK64
        dst, c = ref_mad_i64_i32(INT32_MIN, INT32_MAX, 0)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

    def test_commutativity_of_multiplication(self) -> None:
        operands = [
            (0, 0),
            (1, -1),
            (-500, 300),
            (INT32_MIN, INT32_MAX),
            (0x1234_5678, -0x1234_5678),
        ]
        s2_values = [0, 1, -1, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000]
        for s0, s1 in operands:
            for s2 in s2_values:
                with self.subTest(s0=s0, s1=s1, s2=s2):
                    res1, c1 = ref_mad_i64_i32(s0, s1, s2)
                    res2, c2 = ref_mad_i64_i32(s1, s0, s2)
                    self.assertEqual(res1, res2)
                    self.assertEqual(c1, c2)

    def test_carry_out_generation_boundary(self) -> None:
        # (-1 * 1) = -1 (0xFFFF_FFFF_FFFF_FFFF).
        # Adding 1 gives 0x1_0000_0000_0000_0000 -> dst = 0, carry = 1
        dst, c = ref_mad_i64_i32(-1, 1, 1)
        self.assertEqual(dst, 0)
        self.assertEqual(c, 1)

        # Adding 0 gives 0xFFFF_FFFF_FFFF_FFFF -> dst = MASK64, carry = 0
        dst, c = ref_mad_i64_i32(-1, 1, 0)
        self.assertEqual(dst, MASK64)
        self.assertEqual(c, 0)

        # Max positive product with s2 = UINT64_MAX:
        # 1 * 1 + UINT64_MAX = 0 with carry = 1
        dst, c = ref_mad_i64_i32(1, 1, UINT64_MAX)
        self.assertEqual(dst, 0)
        self.assertEqual(c, 1)

    def test_split_oracle_reconstruction(self) -> None:
        s0 = -12345
        s1 = 67890
        s2_lo = 0xDEAD_BEEF
        s2_hi = 0xCAFE_BABE
        dst_lo, dst_hi, carry = ref_mad_i64_i32_split(s0, s1, s2_lo, s2_hi)
        full_dst, full_c = ref_mad_i64_i32(s0, s1, combine_u32(s2_lo, s2_hi))
        self.assertEqual(combine_u32(dst_lo, dst_hi), full_dst)
        self.assertEqual(carry, full_c)

    def test_architectural_register_file(self) -> None:
        rf = RegisterFile()
        rf.write_vgpr(0, 0x12345678)
        rf.write_vgpr(1, 0x9ABCDEF0)
        self.assertEqual(rf.read_vgpr_64(0), 0x9ABC_DEF0_1234_5678)

        rf.write_vgpr_64(10, 0xFEDC_BA98_7654_3210)
        self.assertEqual(rf.read_vgpr(10), 0x7654_3210)
        self.assertEqual(rf.read_vgpr(11), 0xFEDC_BA98)

        rf.write_sgpr_64(2, 0x0123_4567_89AB_CDEF)
        self.assertEqual(rf.read_sgpr_64(2), 0x0123_4567_89AB_CDEF)

        with self.assertRaises(IndexError):
            rf.read_vgpr(256)
        with self.assertRaises(IndexError):
            rf.read_vgpr_64(255)
        with self.assertRaises(IndexError):
            rf.read_sgpr(106)
        with self.assertRaises(IndexError):
            rf.read_sgpr_64(105)


if __name__ == "__main__":
    unittest.main()
