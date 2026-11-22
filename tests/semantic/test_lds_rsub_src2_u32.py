"""Semantic reference oracle and architectural model for ds_rsub_src2_u32.

RDNA2 / GFX10 Local Data Share (LDS) Dual-Source Atomic Reverse Subtraction:
Performs a 32-bit atomic reverse subtraction of secondary operand LDS[B] minus primary operand LDS[A],
storing the result back into LDS[A]:
  tmp_a = LDS[A]
  tmp_b = LDS[B]
  LDS[A] = (tmp_b - tmp_a) mod 2^32

Syntax:
    ds_rsub_src2_u32 vaddr [offset0:offset8] [offset1:offset8]

Semantics:
    Address A = (vaddr + offset0 * 4) & 0xFFFFFFFF
    Address B = (vaddr + offset1 * 4) & 0xFFFFFFFF
    idx_a = (Address A >> 2) & (lds_dwords - 1)
    idx_b = (Address B >> 2) & (lds_dwords - 1)
    val_a = LDS[idx_a]
    val_b = LDS[idx_b]
    LDS[idx_a] = (val_b - val_a) & 0xFFFFFFFF

Non-returning:
    This instruction does not write to vector general-purpose registers (VGPRs).

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import struct
import sys
from typing import Any, Tuple
import unittest

# Architectural Constants for RDNA2 Local Data Share (LDS)
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK32: int = 0xFFFFFFFF
OFFSET8_MASK: int = 0xFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    capacity_dwords: int = LDS_SIZE_DWORDS,
) -> tuple[int, int, int, int]:
    """Calculate effective byte addresses and dword indices for LDS Dual-Source operands A and B."""
    byte_a = (vaddr + ((offset0 & OFFSET8_MASK) * 4)) & MASK32
    idx_a = (byte_a >> 2) & (capacity_dwords - 1)

    byte_b = (vaddr + ((offset1 & OFFSET8_MASK) * 4)) & MASK32
    idx_b = (byte_b >> 2) & (capacity_dwords - 1)

    return byte_a, idx_a, byte_b, idx_b


def oracle_ds_rsub_src2_u32(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_rsub_src2_u32.
    
    Reverse subtraction: res = (val_b - val_a) mod 2^32.
    """
    a = val_a & MASK32
    b = val_b & MASK32
    return (b - a) & MASK32


class LDSModel:
    """Architectural memory model representing the RDNA2 Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords: int = capacity_dwords
        self.memory: list[int] = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        self.memory[idx] = val & MASK32

    def atomic_rsub_src2_u32(self, vaddr: int, offset0: int = 0, offset1: int = 0) -> int:
        """Execute atomic dual-source reverse subtraction on LDS."""
        byte_a, idx_a, byte_b, idx_b = calc_lds_src2_addresses(
            vaddr, offset0, offset1, self.capacity_dwords
        )
        val_a = self.memory[idx_a]
        val_b = self.memory[idx_b]
        res = oracle_ds_rsub_src2_u32(val_a, val_b)
        self.memory[idx_a] = res
        return res


class TestLdsRsubSrc2ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model of ds_rsub_src2_u32."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_basic_reverse_subtraction(self) -> None:
        # LDS[A] = 30, LDS[B] = 100 -> new LDS[A] = 100 - 30 = 70
        res = oracle_ds_rsub_src2_u32(val_a=30, val_b=100)
        self.assertEqual(res, 70)

    def test_reverse_subtraction_underflow_wrapping(self) -> None:
        # LDS[A] = 100, LDS[B] = 30 -> new LDS[A] = (30 - 100) mod 2^32 = -70 mod 2^32 = 0xFFFFFFBA
        res = oracle_ds_rsub_src2_u32(val_a=100, val_b=30)
        self.assertEqual(res, (30 - 100) & MASK32)
        self.assertEqual(res, 0xFFFFFFBA)

    def test_self_subtraction_zeroes(self) -> None:
        # When A == B (same index), val_a == val_b -> (val_b - val_a) = 0
        for val in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            self.assertEqual(oracle_ds_rsub_src2_u32(val, val), 0)

    def test_lds_model_dual_source_execution(self) -> None:
        # Seed slot A (byte 0x10) with 25 and slot B (byte 0x20) with 100
        self.lds.write_dword(0x10, 25)
        self.lds.write_dword(0x20, 100)

        # vaddr = 0x00, offset0 = 4 (byte 16 = 0x10), offset1 = 8 (byte 32 = 0x20)
        res = self.lds.atomic_rsub_src2_u32(vaddr=0x00, offset0=4, offset1=8)
        self.assertEqual(res, 75)
        # LDS[A] updated to 75
        self.assertEqual(self.lds.read_dword(0x10), 75)
        # LDS[B] untouched (remains 100)
        self.assertEqual(self.lds.read_dword(0x20), 100)

    def test_self_addressing_in_lds_model(self) -> None:
        # Both offset0 and offset1 point to byte 0x40
        self.lds.write_dword(0x40, 0xABCDEF01)
        res = self.lds.atomic_rsub_src2_u32(vaddr=0x40, offset0=0, offset1=0)
        self.assertEqual(res, 0)
        self.assertEqual(self.lds.read_dword(0x40), 0)

    def test_dword_alignment_drops_low_bits(self) -> None:
        # vaddr = 0x13 -> drops low 2 bits to 0x10
        self.lds.write_dword(0x10, 40)
        self.lds.write_dword(0x18, 90)
        # offset0 = 0 -> address 0x13 -> dword 4 (byte 0x10)
        # offset1 = 2 -> address 0x1B -> dword 6 (byte 0x18)
        res = self.lds.atomic_rsub_src2_u32(vaddr=0x13, offset0=0, offset1=2)
        self.assertEqual(res, 50)
        self.assertEqual(self.lds.read_dword(0x10), 50)
        self.assertEqual(self.lds.read_dword(0x18), 90)

    def test_address_wrapping_modulo_lds_size(self) -> None:
        mem = LDSModel(capacity_dwords=256)  # 256 dwords (1024 bytes)
        mem.write_dword(0x00, 15)
        mem.write_dword(0x20, 50)
        # Byte 1024 wraps to 0x00 (dword 0)
        res = mem.atomic_rsub_src2_u32(vaddr=1024, offset0=0, offset1=8)  # offset1=8 -> byte 32 (0x20 = dword 8)
        self.assertEqual(res, 35)
        self.assertEqual(mem.read_dword(0x00), 35)


if __name__ == "__main__":
    unittest.main()
