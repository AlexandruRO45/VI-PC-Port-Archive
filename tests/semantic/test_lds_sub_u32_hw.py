"""Reference mathematical oracle and hardware translator tests for ds_sub_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Subtraction:
Performs an atomic 32-bit subtraction of operand from LDS memory location:
    LDS[address + offset] = (LDS[address + offset] - data) mod 2^32

Syntax:
    ds_sub_u32 vaddr, vdata [offset:N]

Semantics:
    Effective Address = (vaddr + offset) & 0xFFFFFFFF
    idx = (Effective Address >> 2) & (lds_dwords - 1)
    old_val = LDS[idx]
    LDS[idx] = (old_val - data) & 0xFFFFFFFF

Non-returning:
    Does not write back to vector general-purpose registers (VGPRs).

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
from typing import Any
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

# Architectural Constants for RDNA2 Local Data Share (LDS)
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK32: int = 0xFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
SIGN_BIT32: int = 0x80000000

# Compute Shader RSRC2 metadata: 4 user SGPRs, workgroup id in s4, LDS_SIZE = 4 KiB
RSRC2_4K: int = 0x40088
VSHARP: list[int] = [0, 0, 0x1000, 0x31016FAC]


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def calc_lds_address(
    vaddr: int,
    offset: int = 0,
    capacity_dwords: int = LDS_SIZE_DWORDS,
) -> tuple[int, int]:
    """Calculate effective byte address and dword index for LDS single-address operand."""
    eff_byte = (vaddr + (offset & OFFSET16_MASK)) & MASK32
    idx = (eff_byte >> 2) & (capacity_dwords - 1)
    return eff_byte, idx


def oracle_ds_sub_u32(orig_val: int, sub_val: int) -> int:
    """Pure mathematical reference oracle for ds_sub_u32.
    
    Subtraction: res = (orig_val - sub_val) mod 2^32.
    """
    orig = orig_val & MASK32
    sub = sub_val & MASK32
    return (orig - sub) & MASK32


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

    def atomic_sub_u32(self, vaddr: int, sub_val: int, offset: int = 0) -> int:
        """Execute atomic subtraction on LDS: LDS[addr] = (LDS[addr] - sub_val) mod 2^32."""
        eff_byte, idx = calc_lds_address(vaddr, offset, self.capacity_dwords)
        old_val = self.memory[idx]
        new_val = oracle_ds_sub_u32(old_val, sub_val)
        self.memory[idx] = new_val
        return new_val


def make_cs_fssm(rsrc2: int = RSRC2_4K, size: tuple[int, int, int] = (32, 1, 1)) -> bytes:
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, rsrc2, *size)
    return bytes(blob)


def translate_compute(lines: list[str], name: str = "cs_ds_sub_u32") -> str:
    body = list(lines) + ["s_endpgm"]
    return support.translate_lines(name, body, fssm=make_cs_fssm())


def run_compute_kernel(
    lines: list[str],
    lanes: int = 32,
    inputs_for: Any | None = None,
    memory: dict[int, int] | None = None,
) -> dict[int, int]:
    spv = translate_compute(lines)
    if memory is None:
        memory = {}
    if inputs_for is None:
        inputs_for = lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i}
    return si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=inputs_for,
        push_constants=VSHARP + [0] * 28,
        memory=memory,
    )


class TestLdsSubU32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and hardware translator execution."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_oracle_basic_subtraction(self) -> None:
        # 100 - 30 = 70
        res = oracle_ds_sub_u32(orig_val=100, sub_val=30)
        self.assertEqual(res, 70)

    def test_oracle_underflow_wrapping(self) -> None:
        # 30 - 100 = -70 mod 2^32 = 0xFFFFFFBA
        res = oracle_ds_sub_u32(orig_val=30, sub_val=100)
        self.assertEqual(res, (30 - 100) & MASK32)
        self.assertEqual(res, 0xFFFFFFBA)

    def test_oracle_zero_identity(self) -> None:
        for val in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            self.assertEqual(oracle_ds_sub_u32(val, 0), val)
            self.assertEqual(oracle_ds_sub_u32(val, val), 0)

    def test_lds_model_execution_with_offset(self) -> None:
        # Seed address 0x10 with 50
        self.lds.write_dword(0x10, 50)
        # vaddr = 0x00, offset = 0x10 (byte 16 = dword 4), sub_val = 15
        res = self.lds.atomic_sub_u32(vaddr=0x00, sub_val=15, offset=0x10)
        self.assertEqual(res, 35)
        self.assertEqual(self.lds.read_dword(0x10), 35)

    def test_translator_hardware_execution(self) -> None:
        # Translates a compute kernel with ds_sub_u32 and executes on spirv_mini_interp
        lines = [
            "v_mov_b32_e32 v1, 0x10",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 15",
            "ds_sub_u32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpAtomicISub", spv)
        mem = run_compute_kernel(lines, lanes=1)
        # Verify result in output buffer is 50 - 15 = 35
        self.assertEqual(mem.get(0), 35)


if __name__ == "__main__":
    unittest.main()
