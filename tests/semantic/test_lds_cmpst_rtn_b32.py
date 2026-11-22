from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Compare-and-Swap with Return:
Performs a 32-bit atomic compare-and-swap on LDS memory and returns the original value.

Syntax:
    ds_cmpst_rtn_b32 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    if old_val == vdata0:
        LDS[dword_index] = vdata1
    vdst = old_val
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
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


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_cmpst_rtn_b32(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_rtn_b32.

    Compares current_mem with cmp_val; if equal, replaces memory with new_val.
    Always returns the pre-operation original memory value.

    Returns:
        Tuple[int, int, bool]: (returned_old_val, new_mem_val, swapped)
    """
    old_val = u32(current_mem)
    cmp_masked = u32(cmp_val)
    new_masked = u32(new_val)

    if old_val == cmp_masked:
        return old_val, new_masked, True
    return old_val, old_val, False


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def atomic_cmpst_rtn_b32(
        self,
        vaddr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic 32-bit compare-and-swap on LDS.

        Returns:
            Tuple[int, int, bool]: (old_val, new_mem, swapped)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        curr = self.memory[idx]
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(curr, cmp_val, new_val)
        self.memory[idx] = new_mem
        return old_val, new_mem, swapped


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) compare-and-swap atomic opcodes."""
    DS_CMPST_RTN_B32 = "ds_cmpst_rtn_b32"


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
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}[{self.index}] "
                f"(expected 0 <= index < {max_limit})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Immediate:
    """32-bit immediate constant operand."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u32(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand = Union[RegisterOperand, Immediate]


@dataclass(frozen=True)
class DSCmpstRtnInstruction:
    """Symbolic AST node for ds_cmpst_rtn_b32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Operand  # Compare value
    vdata1: Operand  # New value
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_CMPST_RTN_B32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR, got {self.vdst.reg_type}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR, got {self.vaddr.reg_type}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(
                f"offset out of bounds: {self.offset} (expected 0 <= offset <= {OFFSET16_MASK})"
            )

    def disassemble(self) -> str:
        """Emit canonical AMD assembly string."""
        parts = [self.opcode.value, str(self.vdst), str(self.vaddr), str(self.vdata0), str(self.vdata1)]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"



class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_match_and_swap(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(0x12345678, 0x12345678, 0x9ABCDEF0)
        self.assertTrue(swapped)
        self.assertEqual(old_val, 0x12345678)
        self.assertEqual(new_mem, 0x9ABCDEF0)

    def test_oracle_mismatch_no_swap(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(0x11112222, 0x33334444, 0x99990000)
        self.assertFalse(swapped)
        self.assertEqual(old_val, 0x11112222)
        self.assertEqual(new_mem, 0x11112222)

    def test_oracle_zero_values(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(0, 0, 42)
        self.assertTrue(swapped)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_mem, 42)

    def test_oracle_32bit_masking(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(0x1_00000000, 0, 0x2_00000005)
        self.assertTrue(swapped)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_mem, 5)

    def test_effective_address_calculation(self) -> None:
        eff = calc_lds_effective_address(0x2000, 0x40)
        self.assertEqual(eff, 0x2040)

        eff_wrap = calc_lds_effective_address(0xFFFFFFFF, 1)
        self.assertEqual(eff_wrap, 0)

    def test_dword_index_and_alignment(self) -> None:
        self.assertEqual(calc_lds_dword_index(0x2000), 0x800)
        # Low 2 bits ignored
        self.assertEqual(calc_lds_dword_index(0x2001), 0x800)
        self.assertEqual(calc_lds_dword_index(0x2002), 0x800)
        self.assertEqual(calc_lds_dword_index(0x2003), 0x800)
        self.assertEqual(calc_lds_dword_index(0x2004), 0x801)

    def test_bank_calculation(self) -> None:
        self.assertEqual(calc_lds_bank(0x00), 0)
        self.assertEqual(calc_lds_bank(0x04), 1)
        self.assertEqual(calc_lds_bank(0x7C), 31)
        self.assertEqual(calc_lds_bank(0x80), 0)

    def test_architectural_memory_atomic_cmpst(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        vaddr = 0x100
        lds.write_dword(vaddr, 0xAAAA)

        # Successful compare-and-swap
        old, new_val, swapped = lds.atomic_cmpst_rtn_b32(vaddr=vaddr, cmp_val=0xAAAA, new_val=0xBBBB, offset=0)
        self.assertTrue(swapped)
        self.assertEqual(old, 0xAAAA)
        self.assertEqual(new_val, 0xBBBB)
        self.assertEqual(lds.read_dword(vaddr), 0xBBBB)

        # Failed compare-and-swap (cmp mismatch)
        old2, new_val2, swapped2 = lds.atomic_cmpst_rtn_b32(vaddr=vaddr, cmp_val=0xCCCC, new_val=0xDDDD, offset=0)
        self.assertFalse(swapped2)
        self.assertEqual(old2, 0xBBBB)
        self.assertEqual(new_val2, 0xBBBB)
        self.assertEqual(lds.read_dword(vaddr), 0xBBBB)


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)
        vdata1 = RegisterOperand(3, RegType.VGPR)
        instr = DSCmpstRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_CMPST_RTN_B32)
        self.assertEqual(instr.vdst.index, 0)
        self.assertEqual(instr.vaddr.index, 1)
        self.assertEqual(instr.offset, 16)
        self.assertFalse(instr.gds)

    def test_ast_with_immediate_operands(self) -> None:
        vdst = RegisterOperand(5, RegType.VGPR)
        vaddr = RegisterOperand(6, RegType.VGPR)
        cmp_imm = Immediate(0x100)
        new_imm = Immediate(0x200)
        instr = DSCmpstRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=cmp_imm, vdata1=new_imm, offset=0)

        self.assertEqual(instr.vdata0, Immediate(0x100))
        self.assertEqual(instr.vdata1, Immediate(0x200))

    def test_disassembly_formatting(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)
        vdata1 = RegisterOperand(3, RegType.VGPR)

        instr_no_offset = DSCmpstRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=0)
        self.assertEqual(
            instr_no_offset.disassemble(),
            "ds_cmpst_rtn_b32 v[0], v[1], v[2], v[3]",
        )

        instr_with_offset = DSCmpstRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=40)
        self.assertEqual(
            instr_with_offset.disassemble(),
            "ds_cmpst_rtn_b32 v[0], v[1], v[2], v[3], offset:40",
        )

        instr_gds = DSCmpstRtnInstruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=8, gds=True)
        self.assertEqual(
            instr_gds.disassemble(),
            "ds_cmpst_rtn_b32 v[0], v[1], v[2], v[3], offset:8, gds",
        )

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            DSCmpstRtnInstruction(vdst=s0, vaddr=v0, vdata0=v0, vdata1=v0)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSCmpstRtnInstruction(vdst=v0, vaddr=s0, vdata0=v0, vdata1=v0)

        # Negative offset
        with self.assertRaises(ValueError):
            DSCmpstRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, vdata1=v0, offset=-1)

        # Offset exceeding 16 bits
        with self.assertRaises(ValueError):
            DSCmpstRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, vdata1=v0, offset=65536)


if __name__ == "__main__":
    unittest.main()

