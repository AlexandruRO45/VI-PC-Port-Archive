"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_dec_u32 and ds_dec_rtn_u32.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - Single 32-bit Atomic Unsigned Decrement with Wrap / Clamp Threshold:
    - RDNA2 ISA: M = (M == 0 || M > DATA) ? DATA : M - 1 (unsigned comparison)
    - ds_dec_u32 vaddr, vdata [offset:N]:
      Non-returning form:
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[dword_idx]
        LDS[dword_idx] = vdata if (old_val == 0 or old_val > vdata) else (old_val - 1)
      The prior value at LDS[dword_idx] is discarded.
    - ds_dec_rtn_u32 vdst, vaddr, vdata [offset:N]:
      Returning form:
        Atomically updates memory according to the same formula and stores pre-atomic
        value in vdst.
  - Active lane masking:
    - Governed by EXEC mask (Wave32 / Wave64). Inactive lanes do not mutate LDS storage
      and leave destination VGPRs unmodified.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


def ds_dec_u32_oracle(old_val: int, data: int) -> int:
    """Pure reference oracle for RDNA2 ds_dec_u32: (M == 0 || M > DATA) ? DATA : M - 1."""
    m = old_val & MASK32
    d = data & MASK32
    if m == 0 or m > d:
        return d
    return (m - 1) & MASK32


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.dwords: List[int] = [0] * num_dwords

    def read_dword(self, byte_addr: int) -> int:
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        return self.dwords[idx]

    def write_dword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        self.dwords[idx] = val & MASK32

    def atomic_dec(self, byte_addr: int, data: int) -> int:
        """Atomic decrement with threshold wrap. Returns pre-atomic dword value."""
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        old = self.dwords[idx]
        self.dwords[idx] = ds_dec_u32_oracle(old, data)
        return old


def ds_dec_u32_execute_model(
    lds: LDSMemory, vaddr: int, vdata: int, offset: int = 0
) -> Tuple[Optional[int], int]:
    """Executes non-returning ds_dec_u32. Returns (None, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_dec(eff_byte, vdata)
    new_val = lds.read_dword(eff_byte)
    return None, new_val


def ds_dec_rtn_u32_execute_model(
    lds: LDSMemory, vaddr: int, vdata: int, offset: int = 0
) -> Tuple[int, int]:
    """Executes returning ds_dec_rtn_u32. Returns (old_val, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_dec(eff_byte, vdata)
    new_val = lds.read_dword(eff_byte)
    return old_val, new_val


class TestDSDecU32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_decrement_sequence(self) -> None:
        # DATA = 5: sequence 5 -> 4 -> 3 -> 2 -> 1 -> 0 -> 5
        cur = 5
        data = 5
        expected_seq = [4, 3, 2, 1, 0, 5]
        for want in expected_seq:
            cur = ds_dec_u32_oracle(cur, data)
            self.assertEqual(cur, want)

    def test_oracle_clamp_when_greater_than_data(self) -> None:
        # If M > DATA, resets/clamps to DATA
        data = 10
        for val in (11, 20, 100, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(ds_dec_u32_oracle(val, data), data)

    def test_oracle_zero_wraps_to_data(self) -> None:
        for data in (1, 5, 42, 0x12345678, 0xFFFFFFFF):
            with self.subTest(data=hex(data)):
                self.assertEqual(ds_dec_u32_oracle(0, data), data)

    def test_lds_memory_address_wrapping_and_alignment(self) -> None:
        lds = LDSMemory(num_dwords=1024)  # 4 KiB
        lds.write_dword(0x40, 5)
        for off in (0, 1, 2, 3):
            self.assertEqual(lds.read_dword(0x40 + off), 5)
        self.assertEqual(lds.read_dword(0x40 + 4096), 5)

    def test_model_non_returning_and_returning(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x100, 5)

        old, new = ds_dec_rtn_u32_execute_model(lds, 0x100, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new, 4)

        rtn, new2 = ds_dec_u32_execute_model(lds, 0x100, 5)
        self.assertIsNone(rtn)
        self.assertEqual(new2, 3)


class DSOpcode(str, enum.Enum):
    DS_DEC_U32 = "ds_dec_u32"
    DS_DEC_RTN_U32 = "ds_dec_rtn_u32"


@dataclasses.dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 0xFFFF):
            raise ValueError(f"OffsetImmediate out of range [0, 0xFFFF]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSDecInstruction:
    opcode: DSOpcode
    vaddr: VGPR
    vdata: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = dataclasses.field(default_factory=lambda: OffsetImmediate(0))

    def __post_init__(self) -> None:
        if self.opcode == DSOpcode.DS_DEC_U32 and self.vdst is not None:
            raise ValueError(f"{self.opcode} must not have a destination register")
        if self.opcode == DSOpcode.DS_DEC_RTN_U32 and self.vdst is None:
            raise ValueError(f"{self.opcode} requires a destination register")

    def to_asm(self) -> str:
        off_str = f" {self.offset}" if self.offset.value != 0 else ""
        if self.opcode == DSOpcode.DS_DEC_RTN_U32:
            return f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata}{off_str}"
        return f"{self.opcode.value} {self.vaddr}, {self.vdata}{off_str}"


class TestDSDecU32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_non_returning(self) -> None:
        inst = DSDecInstruction(
            opcode=DSOpcode.DS_DEC_U32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(16),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_DEC_U32)
        self.assertEqual(inst.vaddr.index, 1)
        self.assertEqual(inst.vdata.index, 2)
        self.assertIsNone(inst.vdst)
        self.assertEqual(inst.offset.value, 16)
        self.assertEqual(inst.to_asm(), "ds_dec_u32 v1, v2 offset:16")

    def test_ast_construction_returning(self) -> None:
        inst = DSDecInstruction(
            opcode=DSOpcode.DS_DEC_RTN_U32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(0),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_DEC_RTN_U32)
        self.assertEqual(inst.vdst.index, 5)
        self.assertEqual(inst.to_asm(), "ds_dec_rtn_u32 v5, v1, v2")


if __name__ == "__main__":
    unittest.main()
