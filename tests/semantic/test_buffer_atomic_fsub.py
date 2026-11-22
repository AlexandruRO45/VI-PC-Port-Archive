"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_fsub.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory 32-bit floating point atomic subtraction:
  buffer_atomic_fsub vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the IEEE 754 32-bit float value. If 'glc' is enabled,
       the pre-operation 32-bit float value at memory is returned to vdata.
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
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic float sub is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation (IEEE 754-2008 single-precision subtraction):
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = ieee754_sub_f32(old_val, vdata)
       memory[effective_address] = new_val
       if glc: vdata = old_val
  4. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPRs.

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
EXP_MASK: int = 0x7F800000
MANT_MASK: int = 0x007FFFFF
SIGN_MASK: int = 0x80000000

POS_ZERO: int = 0x00000000
NEG_ZERO: int = 0x80000000
POS_INF: int = 0x7F800000
NEG_INF: int = 0xFF800000
QNAN_DEFAULT: int = 0x7FC00000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_MASK) else val


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def u32_to_f32(val: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", u32(val)))[0]


def f32_to_u32(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def ieee754_sub_f32(a_u32: int, b_u32: int) -> int:
    """Bit-accurate IEEE 754 single-precision subtraction matching RDNA2 hardware semantics."""
    a_u32 = u32(a_u32)
    b_u32 = u32(b_u32)

    a_exp = (a_u32 >> 23) & 0xFF
    a_mant = a_u32 & MANT_MASK
    b_exp = (b_u32 >> 23) & 0xFF
    b_mant = b_u32 & MANT_MASK

    a_is_nan = (a_exp == 0xFF and a_mant != 0)
    b_is_nan = (b_exp == 0xFF and b_mant != 0)

    # Any NaN operand produces a canonical quiet NaN
    if a_is_nan or b_is_nan:
        return QNAN_DEFAULT

    a_is_inf = (a_exp == 0xFF and a_mant == 0)
    b_is_inf = (b_exp == 0xFF and b_mant == 0)
    a_sign = (a_u32 >> 31) & 1
    b_sign = (b_u32 >> 31) & 1

    # (+Inf) - (+Inf) or (-Inf) - (-Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign == b_sign):
        return QNAN_DEFAULT

    # (+Inf) - finite or (+Inf) - (-Inf) -> +Inf
    # (-Inf) - finite or (-Inf) - (+Inf) -> -Inf
    if a_is_inf:
        return a_u32
    if b_is_inf:
        return b_u32 ^ SIGN_MASK

    # Handle IEEE -0.0 - +0.0 = -0.0
    if a_u32 == NEG_ZERO and b_u32 == POS_ZERO:
        return NEG_ZERO

    a_f = u32_to_f32(a_u32)
    b_f = u32_to_f32(b_u32)

    try:
        res_f = a_f - b_f
    except OverflowError:
        return POS_INF if (a_f > b_f) else NEG_INF

    res_u32 = f32_to_u32(res_f)
    # When difference is -0.0 but operands were not (-0.0, +0.0), IEEE 754 rounds to +0.0 in nearest-even
    if res_u32 == NEG_ZERO and not (a_u32 == NEG_ZERO and b_u32 == POS_ZERO):
        res_u32 = POS_ZERO

    return res_u32


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


def ref_buffer_atomic_fsub(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    sub_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_fsub.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (sub_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = ieee754_sub_f32(old_val, sub_val & MASK32)
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (sub_val & MASK32)
    return ret_val, True

class TestBufferAtomicFsubOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_float_sub_in_bounds(self) -> None:
        c_10 = f32_to_u32(10.0)
        c_4_5 = f32_to_u32(4.5)
        c_5_5 = f32_to_u32(5.5)

        self.mem.write_u32(100, c_10)
        ret, in_bounds = ref_buffer_atomic_fsub(
            self.mem, self.desc, soffset=0, vaddr=100, sub_val=c_4_5, glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(100), c_5_5)

    def test_atomic_fsub_glc_returns_pre_op_value(self) -> None:
        c_25 = f32_to_u32(25.0)
        c_5 = f32_to_u32(5.0)
        c_20 = f32_to_u32(20.0)

        self.mem.write_u32(200, c_25)
        ret, in_bounds = ref_buffer_atomic_fsub(
            self.mem, self.desc, soffset=0, vaddr=200, sub_val=c_5, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, c_25)
        self.assertEqual(self.mem.read_u32(200), c_20)

    def test_out_of_bounds_rejection(self) -> None:
        c_val = f32_to_u32(123.0)
        self.mem.write_u32(1024, c_val)
        ret, in_bounds = ref_buffer_atomic_fsub(
            self.mem, self.desc, soffset=0, vaddr=1024, sub_val=f32_to_u32(1.0), glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), c_val)

    def test_signed_zero_subtraction(self) -> None:
        # +0.0 - +0.0 = +0.0
        self.mem.write_u32(0, POS_ZERO)
        ret, in_bounds = ref_buffer_atomic_fsub(self.mem, self.desc, soffset=0, vaddr=0, sub_val=POS_ZERO, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(0), POS_ZERO)

        # -0.0 - +0.0 = -0.0
        self.mem.write_u32(4, NEG_ZERO)
        ret, in_bounds = ref_buffer_atomic_fsub(self.mem, self.desc, soffset=0, vaddr=4, sub_val=POS_ZERO, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(4), NEG_ZERO)

        # -0.0 - -0.0 = +0.0
        self.mem.write_u32(8, NEG_ZERO)
        ret, in_bounds = ref_buffer_atomic_fsub(self.mem, self.desc, soffset=0, vaddr=8, sub_val=NEG_ZERO, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(8), POS_ZERO)

    def test_infinity_and_nan_arithmetic(self) -> None:
        # (+Inf) - (+Inf) = NaN
        self.mem.write_u32(0, POS_INF)
        ret, in_bounds = ref_buffer_atomic_fsub(self.mem, self.desc, soffset=0, vaddr=0, sub_val=POS_INF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(0), QNAN_DEFAULT)

        # (+Inf) - (-Inf) = +Inf
        self.mem.write_u32(4, POS_INF)
        ret, in_bounds = ref_buffer_atomic_fsub(self.mem, self.desc, soffset=0, vaddr=4, sub_val=NEG_INF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(4), POS_INF)

        # 10.0 - NaN = NaN
        self.mem.write_u32(8, f32_to_u32(10.0))
        ret, in_bounds = ref_buffer_atomic_fsub(self.mem, self.desc, soffset=0, vaddr=8, sub_val=QNAN_DEFAULT, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(8), QNAN_DEFAULT)

    def test_inst_offset_and_soffset_combination(self) -> None:
        c_1 = f32_to_u32(200.0)
        c_2 = f32_to_u32(50.0)
        self.mem.write_u32(100, c_1)
        ret, in_bounds = ref_buffer_atomic_fsub(
            self.mem, self.desc, soffset=64, vaddr=20, sub_val=c_2, inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, c_1)
        self.assertEqual(self.mem.read_u32(100), f32_to_u32(150.0))

if __name__ == "__main__":
    unittest.main()
