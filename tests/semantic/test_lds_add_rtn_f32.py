from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_add_rtn_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Add with Return:
Performs an atomic single-precision IEEE 754 floating-point addition on 32-bit LDS memory
and returns the original (pre-operation) value.

Syntax:
    ds_add_rtn_f32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val_u32 = LDS[dword_index]
    LDS[dword_index] = f32_to_u32(u32_to_f32(old_val_u32) + u32_to_f32(vdata0))
    vdst = old_val_u32
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Constants
QNAN_DEFAULT: int = 0x7FC0_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def u32_to_f32(val: int) -> float:
    """Unpack 32-bit unsigned dword to Python float."""
    return struct.unpack("<f", struct.pack("<I", u32(val)))[0]


def f32_to_u32(val: float) -> int:
    """Pack Python float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def ieee754_add_f32(a_u32: int, b_u32: int) -> int:
    """Bit-accurate IEEE 754 single-precision addition matching RDNA2 hardware semantics."""
    a_u32 = u32(a_u32)
    b_u32 = u32(b_u32)

    a_exp = (a_u32 >> 23) & 0xFF
    a_mant = a_u32 & 0x7F_FFFF
    b_exp = (b_u32 >> 23) & 0xFF
    b_mant = b_u32 & 0x7F_FFFF

    a_is_nan = (a_exp == 0xFF and a_mant != 0)
    b_is_nan = (b_exp == 0xFF and b_mant != 0)

    # Any NaN operand produces a canonical quiet NaN
    if a_is_nan or b_is_nan:
        return QNAN_DEFAULT

    a_is_inf = (a_exp == 0xFF and a_mant == 0)
    b_is_inf = (b_exp == 0xFF and b_mant == 0)
    a_sign = (a_u32 >> 31) & 1
    b_sign = (b_u32 >> 31) & 1

    # (+Inf) + (-Inf) or (-Inf) + (+Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign != b_sign):
        return QNAN_DEFAULT

    # If either is Inf, result is that Inf
    if a_is_inf:
        return a_u32
    if b_is_inf:
        return b_u32

    # Float addition
    a_f = u32_to_f32(a_u32)
    b_f = u32_to_f32(b_u32)

    # Handle IEEE -0.0 + -0.0 = -0.0
    if a_u32 == 0x8000_0000 and b_u32 == 0x8000_0000:
        return 0x8000_0000

    try:
        res_f = a_f + b_f
    except OverflowError:
        return 0x7F80_0000 if (a_f > 0 or b_f > 0) else 0xFF80_0000

    return f32_to_u32(res_f)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_add_rtn_f32(current_mem: int, data: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_add_rtn_f32.

    Returns:
        (old_val, new_val)
    """
    old_val = u32(current_mem)
    data_masked = u32(data)
    new_val = ieee754_add_f32(old_val, data_masked)
    return old_val, new_val


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = u32(val)

    def atomic_add_f32(self, vaddr: int, data: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic float addition with return on LDS dword.

        Returns (old_val, new_val).
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_val = oracle_ds_add_rtn_f32(curr, data)
        self.memory[idx] = new_val
        return old_val, new_val


class TestLdsAddRtnF32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_float_add_identity_zero(self) -> None:
        """Adding +0.0 to a float preserves its exact bit representation."""
        addr = 0x40
        val = f32_to_u32(42.5)
        self.lds.write_dword(addr, val)
        old_val, new_val = self.lds.atomic_add_f32(addr, f32_to_u32(0.0))
        self.assertEqual(old_val, val)
        self.assertEqual(new_val, val)
        self.assertEqual(self.lds.read_dword(addr), val)

    def test_float_add_basic_positive(self) -> None:
        """1.5 + 2.5 == 4.0."""
        addr = 0x80
        val1 = f32_to_u32(1.5)
        val2 = f32_to_u32(2.5)
        self.lds.write_dword(addr, val1)
        old_val, new_val = self.lds.atomic_add_f32(addr, val2)
        self.assertEqual(old_val, val1)
        self.assertEqual(u32_to_f32(new_val), 4.0)
        self.assertEqual(self.lds.read_dword(addr), f32_to_u32(4.0))

    def test_float_add_positive_and_negative(self) -> None:
        """10.0 + (-3.5) == 6.5."""
        addr = 0x90
        val1 = f32_to_u32(10.0)
        val2 = f32_to_u32(-3.5)
        self.lds.write_dword(addr, val1)
        old_val, new_val = self.lds.atomic_add_f32(addr, val2)
        self.assertEqual(old_val, val1)
        self.assertEqual(u32_to_f32(new_val), 6.5)

    def test_float_add_signed_zeros(self) -> None:
        """+0.0 + -0.0 == +0.0, and -0.0 + -0.0 == -0.0."""
        addr = 0xA0
        # +0.0 + -0.0 = +0.0
        self.lds.write_dword(addr, 0x0000_0000)
        old1, new1 = self.lds.atomic_add_f32(addr, 0x8000_0000)
        self.assertEqual(old1, 0x0000_0000)
        self.assertEqual(new1, 0x0000_0000)

        # -0.0 + -0.0 = -0.0
        self.lds.write_dword(addr, 0x8000_0000)
        old2, new2 = self.lds.atomic_add_f32(addr, 0x8000_0000)
        self.assertEqual(old2, 0x8000_0000)
        self.assertEqual(new2, 0x8000_0000)

    def test_offset_calculation_and_wrap(self) -> None:
        """Offset is 16-bit unsigned and added to vaddr modulo 2^32."""
        base = 0x1000
        offset = 0x24
        eff = calc_lds_effective_address(base, offset)
        self.assertEqual(eff, 0x1024)

        # 16-bit offset masking
        eff_masked = calc_lds_effective_address(base, 0x1_0004)
        self.assertEqual(eff_masked, 0x1004)

        # 32-bit address wraparound
        eff_wrap = calc_lds_effective_address(0xFFFF_FFF0, 0x0020)
        self.assertEqual(eff_wrap, 0x0000_0010)

    def test_dword_alignment_truncation(self) -> None:
        """Low 2 bits of effective address are ignored for dword indexing."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx2, 0x100)
        self.assertEqual(idx3, 0x100)

    def test_bank_distribution(self) -> None:
        """32 consecutive dwords map to 32 distinct LDS banks [0..31]."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))
        self.assertEqual(calc_lds_bank(32 * 4), 0)

    def test_lds_capacity_wrap(self) -> None:
        """LDS access wraps modulo LDS capacity (default 64 KiB = 16384 dwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + LDS_SIZE_BYTES
        idx_base = calc_lds_dword_index(addr_base)
        idx_wrapped = calc_lds_dword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)


if __name__ == "__main__":
    unittest.main()
