from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_rsub_rtn_u32.

Models unsigned 32-bit integer reverse-subtraction and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_rsub_rtn_u32: Atomic reverse subtraction of unsigned 32-bit integer in LDS memory,
    returning the previous memory content into destination VGPR.

Syntax:
    ds_rsub_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    new_mem = (vdata0 - old_val) & 0xFFFFFFFF
    LDS[dword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. Reverse Subtraction (RSUB):
     Computes (vdata0 - old_val) rather than (old_val - vdata0).
     The incoming operand vdata0 serves as the minuend, and the current memory content
     serves as the subtrahend.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-arithmetic value of LDS[dword_index] and returns it
     into the destination VGPR (vdst).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but dword-aligned for 32-bit operations.
     The bottom 2 bits of the effective address are dropped (eff_addr >> 2).
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity and Banking:
     Standard allocation is 64 KiB (16,384 dwords) organized into 32 memory banks.
     Bank index = dword_index % 32.

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
from dataclasses import dataclass
import enum
from enum import Enum, auto
import re
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import unittest

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536      # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384      # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32                  # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256             # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106             # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the lowest 2 bits are dropped by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


def ref_ds_rsub_rtn_u32(current_mem: int, vdata0: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_rsub_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = (vdata0 - old_val) & 0xFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    new_mem = u32(u32(vdata0) - old_val)
    return old_val, new_mem


class LDSMemory:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.dwords = [0] * self.num_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        return self.dwords[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        self.dwords[idx] = u32(val)

    def atomic_rsub_rtn_u32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit integer reverse subtraction with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        curr = self.dwords[idx]
        old_val, new_mem = ref_ds_rsub_rtn_u32(curr, vdata0)
        self.dwords[idx] = new_mem
        return old_val, new_mem


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2, 4):
            raise ValueError(f"Unsupported register count: {self.count}")
        if self.index + self.count > 256:
            raise ValueError(f"Register range exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count > 1:
            return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class DSOffset:
    """Immediate byte offset modifier for DS instructions."""
    offset16: int = 0

    def __post_init__(self) -> None:
        if self.offset16 < 0 or self.offset16 > 65535:
            raise ValueError(f"offset16 must be in unsigned 16-bit range [0, 65535]: {self.offset16}")


class Opcode(enum.Enum):
    """LDS instruction opcodes."""
    DS_RSUB_RTN_U32 = "ds_rsub_rtn_u32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_rsub_rtn_u32 instruction."""
    opcode: Opcode
    vdst: Register
    vaddr: Register
    vdata0: Register
    offset: DSOffset = dataclasses.field(default_factory=DSOffset)

    def __post_init__(self) -> None:
        if not self.vdst.is_vgpr or self.vdst.count != 1:
            raise ValueError(f"vdst must be a single 32-bit VGPR: {self.vdst}")
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR: {self.vaddr}")
        if not self.vdata0.is_vgpr or self.vdata0.count != 1:
            raise ValueError(f"vdata0 must be a single 32-bit VGPR: {self.vdata0}")

    def to_assembly(self) -> str:
        vd_str = self.vdst.format_asm()
        va_str = self.vaddr.format_asm()
        v0_str = self.vdata0.format_asm()
        off_str = f" offset:{self.offset.offset16}" if self.offset.offset16 != 0 else ""
        return f"{self.opcode.value} {vd_str}, {va_str}, {v0_str}{off_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of LDS instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarLDSInterpreter:
    """Deterministic scalar interpreter for ds_rsub_rtn_u32."""

    def __init__(self, lds_size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(lds_size_bytes)
        self.vcc: int = 0
        self.last_condition_codes: Any | None = None
        self.last_status_flags: Any | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= len(self.vgprs)) or (kind == RegKind.SGPR and idx >= len(self.sgprs)):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= len(self.vgprs)) or (kind == RegKind.SGPR and idx >= len(self.sgprs)):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def step(self, inst: Instruction) -> tuple[int, int]:
        """Executes a single ds_rsub_rtn_u32 instruction."""
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        vdata0 = self.read_reg_u32(RegKind.VGPR, inst.vdata0.index)

        old_val, new_mem = self.lds.atomic_rsub_rtn_u32(
            vaddr=vaddr,
            vdata0=vdata0,
            offset=inst.offset.offset16,
        )

        self.write_reg_u32(RegKind.VGPR, inst.vdst.index, old_val)
        self.trace.append(f"{inst.to_assembly()} => old={old_val:#010x}, new={new_mem:#010x}")
        return old_val, new_mem

    def run_program(self, prog: Program) -> list[tuple[int, int]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestLdsRsubRtnScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(lds_size_bytes=4096)

    def test_scalar_execution_basic(self) -> None:
        # Memory at 0x10 holds 20. vaddr=0x10, vdata0=100.
        self.interp.lds.write_dword(0x10, 20)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)  # vaddr
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)   # vdata0

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 20)
        self.assertEqual(new_mem, 80)
        # vdst receives old_val
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 20)
        # LDS memory updated to 80
        self.assertEqual(self.interp.lds.read_dword(0x10), 80)

    def test_scalar_execution_with_offset(self) -> None:
        # Memory at 0x24 holds 50. vaddr=0x20, offset=4, vdata0=200.
        self.interp.lds.write_dword(0x24, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 200)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset16=4),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_mem, 150)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 50)
        self.assertEqual(self.interp.lds.read_dword(0x24), 150)

    def test_scalar_chained_execution(self) -> None:
        self.interp.lds.write_dword(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 200)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 4),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 3),
        ))

        results = self.interp.run_program(prog)
        # Step 1: 100 - 10 = 90 (old=10)
        self.assertEqual(results[0], (10, 90))
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 10)
        # Step 2: 200 - 90 = 110 (old=90)
        self.assertEqual(results[1], (90, 110))
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 4), 90)
        self.assertEqual(self.interp.lds.read_dword(0), 110)

    def test_scalar_underflow_wrap(self) -> None:
        self.interp.lds.write_dword(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 20)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, u32(20 - 100))
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 100)
        self.assertEqual(self.interp.lds.read_dword(0), u32(20 - 100))


class TestLdsRsubRtnAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v1 = Register(RegKind.VGPR, 1)
        v2 = Register(RegKind.VGPR, 2)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")
        self.assertEqual(v2.format_asm(), "v2")

    def test_instruction_formatting(self) -> None:
        inst_no_off = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst_no_off.to_assembly(), "ds_rsub_rtn_u32 v0, v1, v2")

        inst_with_off = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 6),
            vdata0=Register(RegKind.VGPR, 7),
            offset=DSOffset(offset16=128),
        )
        self.assertEqual(inst_with_off.to_assembly(), "ds_rsub_rtn_u32 v5, v6, v7 offset:128")

    def test_instruction_validation(self) -> None:
        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_RTN_U32,
                vdst=Register(RegKind.SGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                vdata0=Register(RegKind.VGPR, 2),
            )

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_RTN_U32,
                vdst=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.SGPR, 1),
                vdata0=Register(RegKind.VGPR, 2),
            )

        # vdata0 cannot be a multi-register pair
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_RTN_U32,
                vdst=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                vdata0=Register(RegKind.VGPR, 2, count=2),
            )

    def test_offset_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSOffset(offset16=-1)
        with self.assertRaises(ValueError):
            DSOffset(offset16=65536)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset16=4),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 3),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 4),
        ))
        expected = "ds_rsub_rtn_u32 v0, v1, v2 offset:4\nds_rsub_rtn_u32 v3, v1, v4"
        self.assertEqual(prog.to_assembly(), expected)


class TestLDSMemoryAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS memory model."""

    def test_oracle_basic_rsub(self) -> None:
        # 50 - 10 = 40
        old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=10, vdata0=50)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_mem, 40)

        # Reverse order: 10 - 50 = -40 = 0xFFFFFFD8
        old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=50, vdata0=10)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_mem, u32(-40))
        self.assertEqual(new_mem, 0xFFFFFFD8)

    def test_oracle_identity(self) -> None:
        # X - X = 0
        for x in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=x, vdata0=x)
            self.assertEqual(old_val, x & MASK32)
            self.assertEqual(new_mem, 0)

    def test_oracle_zero_cases(self) -> None:
        # 0 - 0 = 0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0), (0, 0))
        # vdata0 - 0 = vdata0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 500), (0, 500))
        # 0 - current_mem = -current_mem
        self.assertEqual(ref_ds_rsub_rtn_u32(500, 0), (500, u32(-500)))

    def test_oracle_wraparound(self) -> None:
        # 0 - 0xFFFFFFFF = 1
        self.assertEqual(ref_ds_rsub_rtn_u32(0xFFFFFFFF, 0), (0xFFFFFFFF, 1))
        # 0xFFFFFFFF - 0 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0xFFFFFFFF), (0, 0xFFFFFFFF))
        # 1 - 2 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(2, 1), (2, 0xFFFFFFFF))

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_dwords, 16384)
        self.assertEqual(len(mem.dwords), 16384)

        # Invalid non-power-of-two or non-positive sizes
        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory(invalid_size)

    def test_lds_memory_read_write(self) -> None:
        mem = LDSMemory(1024)  # 256 dwords
        mem.write_dword(0x00, 0x12345678)
        self.assertEqual(mem.read_dword(0x00), 0x12345678)

        mem.write_dword(0x40, 0xABCDEF01)
        self.assertEqual(mem.read_dword(0x40), 0xABCDEF01)

    def test_lds_memory_unaligned_byte_address(self) -> None:
        mem = LDSMemory(1024)
        # Low 2 bits dropped: 0x100, 0x101, 0x102, 0x103 all map to dword index 0x100 >> 2
        mem.write_dword(0x100, 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x101), 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x102), 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x103), 0xDEADBEEF)

    def test_lds_memory_address_wrapping(self) -> None:
        mem = LDSMemory(1024)  # 256 dwords (1024 bytes)
        # Byte address 1024 wraps to 0
        mem.write_dword(0, 0xCAFE)
        self.assertEqual(mem.read_dword(1024), 0xCAFE)
        self.assertEqual(mem.read_dword(2048), 0xCAFE)

    def test_lds_memory_bank_calculation(self) -> None:
        # Bank is (eff_addr >> 2) % 32
        self.assertEqual(calc_lds_bank(0), 0)
        self.assertEqual(calc_lds_bank(4), 1)
        self.assertEqual(calc_lds_bank(124), 31)
        self.assertEqual(calc_lds_bank(128), 0)  # Next bank cycle

    def test_lds_atomic_rsub_rtn_u32(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x20, 100)

        # Execute rsub with vdata0=250: old=100, new = 250 - 100 = 150
        old_val, new_mem = mem.atomic_rsub_rtn_u32(vaddr=0x20, vdata0=250)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 150)
        self.assertEqual(mem.read_dword(0x20), 150)

        # Subsequent rsub with vdata0=50: old=150, new = 50 - 150 = -100 = 0xFFFFFF9C
        old_val2, new_mem2 = mem.atomic_rsub_rtn_u32(vaddr=0x20, vdata0=50)
        self.assertEqual(old_val2, 150)
        self.assertEqual(new_mem2, u32(-100))
        self.assertEqual(mem.read_dword(0x20), u32(-100))

    def test_lds_atomic_rsub_rtn_with_offset(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x50, 40)

        # vaddr = 0x40, offset = 0x10 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_rsub_rtn_u32(vaddr=0x40, vdata0=100, offset=0x10)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 60)
        self.assertEqual(mem.read_dword(0x50), 60)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_RSUB_RTN_U32 = "ds_rsub_rtn_u32"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < limit):
            raise ValueError(f"Register index out of range: {self.reg_type.value}{self.index}")

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class Immediate:
    """32-bit unsigned immediate value."""
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u32(self.value))

    @property
    def name(self) -> str:
        return hex(self.value)


@dataclass(frozen=True)
class OffsetImmediate:
    """16-bit immediate byte offset descriptor."""
    offset: int

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"Offset out of 16-bit range: {self.offset}")


@dataclass(frozen=True)
class DSRSubRtnInstruction:
    """Symbolic AST node representing a ds_rsub_rtn_u32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Union[RegisterOperand, Immediate]
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_RSUB_RTN_U32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be VGPR, got {self.vdst.reg_type.value}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be VGPR, got {self.vaddr.reg_type.value}")
        if isinstance(self.vdata0, RegisterOperand) and self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 register must be VGPR, got {self.vdata0.reg_type.value}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of 16-bit range: {self.offset}")

    def disasm(self) -> str:
        vdata_str = self.vdata0.name
        parts = [f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {vdata_str}"]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)

    @classmethod
    def parse(cls, line: str) -> DSRSubRtnInstruction:
        raw = line.split("//")[0].split(";")[0].strip()
        if not raw:
            raise ValueError("Empty instruction line")
        tokens = raw.replace(",", " ").split()
        if not tokens:
            raise ValueError("No tokens found")
        mnemonic = tokens[0].lower()
        if mnemonic != DSOpcode.DS_RSUB_RTN_U32.value:
            raise ValueError(f"Unknown opcode: {mnemonic}")

        if len(tokens) < 4:
            raise ValueError(f"Insufficient operands for {mnemonic}")

        def parse_reg(tok: str) -> RegisterOperand:
            m = re.match(r"^([vs])\[?(\d+)\]?$", tok.lower())
            if not m:
                raise ValueError(f"Invalid register operand: {tok}")
            rtype = RegType.VGPR if m.group(1) == "v" else RegType.SGPR
            idx = int(m.group(2))
            return RegisterOperand(idx, rtype)

        vdst = parse_reg(tokens[1])
        vaddr = parse_reg(tokens[2])
        vdata_tok = tokens[3]
        if vdata_tok.lower().startswith("v") or vdata_tok.lower().startswith("s"):
            vdata0: Union[RegisterOperand, Immediate] = parse_reg(vdata_tok)
        else:
            vdata0 = Immediate(int(vdata_tok, 0))

        offset = 0
        gds = False
        for opt in tokens[4:]:
            opt_l = opt.lower()
            if opt_l == "gds":
                gds = True
            elif opt_l.startswith("offset:"):
                offset = int(opt_l.split(":", 1)[1], 0)
            else:
                raise ValueError(f"Unexpected token in instruction: {opt}")

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset, gds=gds)


class TestDSRSubRtnAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_and_disasm(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v1 = RegisterOperand(1, RegType.VGPR)
        v2 = RegisterOperand(2, RegType.VGPR)

        inst_simple = DSRSubRtnInstruction(vdst=v0, vaddr=v1, vdata0=v2)
        self.assertEqual(inst_simple.disasm(), "ds_rsub_rtn_u32 v0, v1, v2")
        self.assertEqual(inst_simple.offset, 0)
        self.assertFalse(inst_simple.gds)

        inst_opts = DSRSubRtnInstruction(vdst=v0, vaddr=v1, vdata0=v2, offset=32, gds=True)
        self.assertEqual(inst_opts.disasm(), "ds_rsub_rtn_u32 v0, v1, v2 offset:32 gds")

        inst_imm = DSRSubRtnInstruction(vdst=v0, vaddr=v1, vdata0=Immediate(0x100), offset=16)
        self.assertEqual(inst_imm.disasm(), "ds_rsub_rtn_u32 v0, v1, 0x100 offset:16")

    def test_ast_parse_canonical(self) -> None:
        inst1 = DSRSubRtnInstruction.parse("ds_rsub_rtn_u32 v0, v1, v2")
        self.assertEqual(inst1.vdst, RegisterOperand(0, RegType.VGPR))
        self.assertEqual(inst1.vaddr, RegisterOperand(1, RegType.VGPR))
        self.assertEqual(inst1.vdata0, RegisterOperand(2, RegType.VGPR))
        self.assertEqual(inst1.offset, 0)
        self.assertFalse(inst1.gds)

        inst2 = DSRSubRtnInstruction.parse("ds_rsub_rtn_u32 v[5], v[6], v[7] offset:64 gds")
        self.assertEqual(inst2.vdst, RegisterOperand(5, RegType.VGPR))
        self.assertEqual(inst2.vaddr, RegisterOperand(6, RegType.VGPR))
        self.assertEqual(inst2.vdata0, RegisterOperand(7, RegType.VGPR))
        self.assertEqual(inst2.offset, 64)
        self.assertTrue(inst2.gds)

        inst3 = DSRSubRtnInstruction.parse("ds_rsub_rtn_u32 v10, v11, 0x200 offset:0x10")
        self.assertEqual(inst3.vdata0, Immediate(0x200))
        self.assertEqual(inst3.offset, 16)

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=s0, vaddr=v0, vdata0=v0)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=s0, vdata0=v0)

        # vdata0 cannot be SGPR
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=v0, vdata0=s0)

        # Negative offset
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=-1)

        # Offset exceeding 16 bits
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=65536)

        # Register index out of bounds
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)


if __name__ == "__main__":
    unittest.main()
