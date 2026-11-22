"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_fmax_x2.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory double-precision (64-bit) float atomic maximum:
  buffer_atomic_fmax_x2 v[d:d+1], vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: 64-bit VGPR pair v[d:d+1] containing the double-precision float (f64) value to compare atomically.
       If 'glc' is enabled, the pre-operation 64-bit dword pair at memory is returned to v[d:d+1].
     - vaddr: VGPR containing the byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = soffset + inst_offset + (vaddr if offen else 0)
     If effective_offset + 8 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic maximum is suppressed (memory is NOT modified).
       If 'glc' is set, 0 (64-bit 0) is returned to v[d:d+1].
  3. Atomic Operation:
     For in-bounds accesses:
       old_bits = memory_64[effective_address]
       new_bits = fmax_f64(old_bits, vdata)
       memory_64[effective_address] = new_bits
       if glc: v[d:d+1] = old_bits
  4. IEEE 754 Double-Precision Float Max Ordering Rules:
     - NaN suppression: if one operand is NaN, return the non-NaN operand.
     - If both operands are NaN, return canonical quiet NaN (0x7FF8000000000000).
     - Signed zero ordering: -0.0 < +0.0.
  5. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPR pairs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF

POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
QNAN_F64: int = 0x7FF8000000000000
SNAN_F64: int = 0x7FF0000000000001


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Splits a 64-bit integer into low and high 32-bit unsigned dwords."""
    return val64 & MASK32, (val64 >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Joins low and high 32-bit dwords into a 64-bit unsigned integer."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def bits_to_f64(bits: int) -> float:
    """Reinterpret 64-bit unsigned integer as IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Reinterpret IEEE 754 double-precision float as 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_nan_f64(bits: int) -> bool:
    """Checks if 64-bit pattern is any IEEE 754 NaN."""
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_qnan_f64(bits: int) -> bool:
    """Checks if 64-bit pattern is a quiet NaN."""
    return is_nan_f64(bits) and bool(bits & 0x0008000000000000)


def is_snan_f64(bits: int) -> bool:
    """Checks if 64-bit pattern is a signaling NaN."""
    return is_nan_f64(bits) and not bool(bits & 0x0008000000000000)


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Tuple key that orders -0.0 strictly less than +0.0."""
    return (v, math.copysign(1.0, v))


def ref_fmax_f64(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 64-bit float max.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FF8000000000000).
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK64
    s1 &= MASK64
    nan0 = is_nan_f64(s0)
    nan1 = is_nan_f64(s1)
    if nan0 and nan1:
        return QNAN_F64
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f64(s0)
    f1 = bits_to_f64(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


@dataclasses.dataclass(frozen=True)
class BufferResourceDescriptor:
    """AMD GFX10 128-bit Buffer Resource Descriptor (V#)."""
    base_address: int
    stride: int
    num_records: int
    format_flags: int = 0

    @classmethod
    def from_dwords(cls, d0: int, d1: int, d2: int, d3: int) -> BufferResourceDescriptor:
        base_lo = d0 & MASK32
        base_hi = d1 & 0xFFFF
        base = (base_hi << 32) | base_lo
        stride = (d1 >> 16) & 0xFFFF
        num_records = d2 & MASK32
        return cls(base_address=base, stride=stride, num_records=num_records, format_flags=d3)

    def to_dwords(self) -> tuple[int, int, int, int]:
        d0 = self.base_address & MASK32
        d1 = ((self.base_address >> 32) & 0xFFFF) | ((self.stride & 0xFFFF) << 16)
        d2 = self.num_records & MASK32
        d3 = self.format_flags & MASK32
        return d0, d1, d2, d3


class BufferMemory:
    """Byte-addressable memory buffer simulating GPU global / buffer device memory."""

    def __init__(self, size: int = 65536) -> None:
        self.data: bytearray = bytearray(size)

    def read_u32(self, addr: int) -> int:
        if addr < 0 or addr + 4 > len(self.data):
            return 0
        return struct.unpack_from("<I", self.data, addr)[0]

    def write_u32(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 4 > len(self.data):
            return
        struct.pack_into("<I", self.data, addr, val & MASK32)

    def read_u64(self, addr: int) -> int:
        if addr < 0 or addr + 8 > len(self.data):
            return 0
        return struct.unpack_from("<Q", self.data, addr)[0]

    def write_u64(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 8 > len(self.data):
            return
        struct.pack_into("<Q", self.data, addr, val & MASK64)


def ref_buffer_atomic_fmax_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    comp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_fmax_x2.

    Returns:
      (return_val_u64, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS (64-bit access requires 8 bytes)
    if effective_offset + 8 > desc.num_records:
        return (0 if glc else (comp_val & MASK64)), False

    phys_addr = (desc.base_address + effective_offset) & 0xFFFFFFFFFFFFFFFF
    old_bits = mem.read_u64(phys_addr)
    # Double-precision float maximum comparison
    new_bits = ref_fmax_f64(old_bits, comp_val & MASK64)
    mem.write_u64(phys_addr, new_bits)

    ret_val = old_bits if glc else (comp_val & MASK64)
    return ret_val, True


class TestBufferAtomicFmaxX2Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_fmax_x2_in_bounds(self) -> None:
        self.mem.write_u64(100, f64_to_bits(1.5))
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=100, comp_val=f64_to_bits(3.5), glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f64(self.mem.read_u64(100)), 3.5)

    def test_atomic_fmax_x2_smaller_value_no_change(self) -> None:
        self.mem.write_u64(100, f64_to_bits(10.0))
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=100, comp_val=f64_to_bits(5.0), glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f64(self.mem.read_u64(100)), 10.0)

    def test_atomic_fmax_x2_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u64(200, f64_to_bits(2.0))
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=200, comp_val=f64_to_bits(8.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f64(ret), 2.0)
        self.assertEqual(bits_to_f64(self.mem.read_u64(200)), 8.0)

    def test_atomic_fmax_x2_signed_zeros(self) -> None:
        # -0.0 < +0.0
        self.mem.write_u64(0, NEG_ZERO_F64)
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=0, comp_val=POS_ZERO_F64, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, NEG_ZERO_F64)
        self.assertEqual(self.mem.read_u64(0), POS_ZERO_F64)

        # +0.0 vs -0.0 -> stays +0.0
        ret2, in_bounds2 = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=0, comp_val=NEG_ZERO_F64, glc=True
        )
        self.assertTrue(in_bounds2)
        self.assertEqual(ret2, POS_ZERO_F64)
        self.assertEqual(self.mem.read_u64(0), POS_ZERO_F64)

    def test_atomic_fmax_x2_infinities(self) -> None:
        # -inf vs 100.0 -> 100.0 wins
        self.mem.write_u64(8, NEG_INF_F64)
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=8, comp_val=f64_to_bits(100.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, NEG_INF_F64)
        self.assertEqual(bits_to_f64(self.mem.read_u64(8)), 100.0)

        # +inf vs 1e100 -> +inf wins
        self.mem.write_u64(16, f64_to_bits(1e100))
        ret2, in_bounds2 = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=16, comp_val=POS_INF_F64, glc=True
        )
        self.assertTrue(in_bounds2)
        self.assertEqual(self.mem.read_u64(16), POS_INF_F64)

    def test_atomic_fmax_x2_nan_suppression(self) -> None:
        # NaN in memory, valid float comp_val -> valid float wins
        self.mem.write_u64(24, QNAN_F64)
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=24, comp_val=f64_to_bits(42.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, QNAN_F64)
        self.assertEqual(bits_to_f64(self.mem.read_u64(24)), 42.0)

        # Valid float in memory, NaN comp_val -> memory remains valid float
        ret2, in_bounds2 = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=24, comp_val=SNAN_F64, glc=True
        )
        self.assertTrue(in_bounds2)
        self.assertEqual(bits_to_f64(self.mem.read_u64(24)), 42.0)

        # Dual NaN -> produces canonical quiet NaN
        self.mem.write_u64(32, SNAN_F64)
        ret3, in_bounds3 = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=32, comp_val=QNAN_F64, glc=True
        )
        self.assertTrue(in_bounds3)
        self.assertEqual(self.mem.read_u64(32), QNAN_F64)

    def test_out_of_bounds_rejection(self) -> None:
        self.mem.write_u64(1024, f64_to_bits(50.0))
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=0, vaddr=1024, comp_val=f64_to_bits(100.0), glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(bits_to_f64(self.mem.read_u64(1024)), 50.0)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(24) = 104
        self.mem.write_u64(104, f64_to_bits(10.0))
        ret, in_bounds = ref_buffer_atomic_fmax_x2(
            self.mem, self.desc, soffset=64, vaddr=24, comp_val=f64_to_bits(50.0), inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f64(ret), 10.0)
        self.assertEqual(bits_to_f64(self.mem.read_u64(104)), 50.0)

    def test_split_join_dwords(self) -> None:
        val = 0x123456789ABCDEF0
        lo, hi = split_dwords(val)
        self.assertEqual(lo, 0x9ABCDEF0)
        self.assertEqual(hi, 0x12345678)
        self.assertEqual(join_dwords(lo, hi), val)


if __name__ == "__main__":
    unittest.main()
