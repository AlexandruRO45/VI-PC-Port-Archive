"""Reference mathematical oracle and hardware translator tests for ds_min_i32.

RDNA2 / GFX10 Local Data Share (LDS) Signed 32-Bit Atomic Minimum:
Performs an atomic signed 32-bit integer minimum of operand and LDS memory location:
    LDS[address + offset] = min_i32(LDS[address + offset], data)

Syntax:
    ds_min_i32 vaddr, vdata [offset:N]
    ds_min_rtn_i32 vdst, vaddr, vdata [offset:N]

Semantics:
    Effective Address = (vaddr + offset) & 0xFFFFFFFF
    idx = (Effective Address >> 2) & (lds_dwords - 1)
    old_val = LDS[idx]
    LDS[idx] = min_i32(old_val, data)
    if rtn: vdst = old_val

Two's-Complement Signed 32-bit Ordering:
    Values are interpreted as signed integers in [-2^31, 2^31 - 1].
    Bit 31 is the sign bit: negative values compare strictly less than positive values.
    For example:
        -1 (0xFFFFFFFF) < 0 (0x00000000)
        INT32_MIN (0x80000000) < INT32_MAX (0x7FFFFFFF)
    In contrast to ds_min_u32 where 0x00000000 < 0xFFFFFFFF and 1 < 0x80000000.

Translator Mapping:
    Mapped to SPIR-V OpAtomicSMin in the Workgroup storage class.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import random
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
INT32_MIN: int = 0x80000000
INT32_MAX: int = 0x7FFFFFFF

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


def oracle_ds_min_i32(m: int, d: int) -> int:
    """Pure mathematical reference oracle for ds_min_i32.
    
    Signed 32-bit integer minimum: returns the operand with smaller signed value.
    """
    m32 = u32(m)
    d32 = u32(d)
    return m32 if s32(m32) <= s32(d32) else d32


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

    def atomic_min_i32(self, vaddr: int, data: int, offset: int = 0) -> tuple[int, int]:
        """Execute atomic signed minimum on LDS: returns (old_val, new_val)."""
        eff_byte, idx = calc_lds_address(vaddr, offset, self.capacity_dwords)
        old_val = self.memory[idx]
        new_val = oracle_ds_min_i32(old_val, data)
        self.memory[idx] = new_val
        return old_val, new_val


def make_cs_fssm(rsrc2: int = RSRC2_4K, size: tuple[int, int, int] = (32, 1, 1)) -> bytes:
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, rsrc2, *size)
    return bytes(blob)


def translate_compute(lines: list[str], name: str = "cs_ds_min_i32") -> str:
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


class TestLdsMinI32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and hardware translator execution."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_oracle_basic_min(self) -> None:
        self.assertEqual(oracle_ds_min_i32(100, 30), 30)
        self.assertEqual(oracle_ds_min_i32(30, 100), 30)
        self.assertEqual(oracle_ds_min_i32(-5, -3), u32(-5))
        self.assertEqual(oracle_ds_min_i32(-3, -5), u32(-5))
        self.assertEqual(oracle_ds_min_i32(0, 0), 0)

    def test_oracle_signed_boundary_extremes(self) -> None:
        self.assertEqual(oracle_ds_min_i32(INT32_MIN, INT32_MAX), INT32_MIN)
        self.assertEqual(oracle_ds_min_i32(INT32_MAX, INT32_MIN), INT32_MIN)
        self.assertEqual(oracle_ds_min_i32(0xFFFFFFFF, 0), 0xFFFFFFFF)
        self.assertEqual(oracle_ds_min_i32(0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(oracle_ds_min_i32(INT32_MIN, 0xFFFFFFFF), INT32_MIN)

    def test_signed_vs_unsigned_divergence(self) -> None:
        self.assertEqual(oracle_ds_min_i32(1, 0x80000000), 0x80000000)
        self.assertEqual(oracle_ds_min_i32(0x80000000, 1), 0x80000000)

    def test_lds_model_execution_with_offset(self) -> None:
        self.lds.write_dword(0x10, 50)
        old_val, new_val = self.lds.atomic_min_i32(vaddr=0x00, data=15, offset=0x10)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 15)
        self.assertEqual(self.lds.read_dword(0x10), 15)

    def test_translator_hardware_execution_non_returning(self) -> None:
        lines = [
            "v_mov_b32_e32 v1, 0x10",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 15",
            "ds_min_i32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpAtomicSMin", spv)
        mem = run_compute_kernel(lines, lanes=1)
        self.assertEqual(mem.get(0), 15)

    def test_translator_hardware_execution_returning(self) -> None:
        lines = [
            "v_mov_b32_e32 v1, 0x20",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, -20",
            "ds_min_rtn_i32 v5, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpAtomicSMin", spv)
        mem = run_compute_kernel(lines, lanes=1)
        self.assertEqual(mem.get(0), 100)
        self.assertEqual(mem.get(4), u32(-20))
