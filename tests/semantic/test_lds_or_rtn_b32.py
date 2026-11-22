"""Semantic reference oracle and architectural model for ds_or_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Bitwise Atomic OR with Return:
Performs a 32-bit atomic bitwise OR operation on LDS memory and returns
the previous memory value to the destination VGPR.

Syntax:
    ds_or_rtn_b32 vdst, vaddr, vdata [offset:offset16]
    ds_or_b32 vaddr, vdata [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_capacity_dwords - 1)
    old_val = LDS[dword_index]
    new_val = old_val | vdata
    LDS[dword_index] = new_val
    vdst = old_val
"""
from __future__ import annotations

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
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


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
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_or_b32(current_mem: int, or_val: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_or_rtn_b32.
    
    Returns:
        (returned_old_val, new_mem_val)
    """
    old_val = current_mem & MASK_32
    vdata = or_val & MASK_32
    new_mem = (old_val | vdata) & MASK_32
    return old_val, new_mem


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
        self.memory[idx] = val & MASK_32

    def atomic_or_b32(
        self,
        vaddr: int,
        or_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic bitwise OR on LDS dword, returning (old_val, new_val)."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_or_b32(curr, or_val)
        self.memory[idx] = new_mem
        return old_val, new_mem


class TestLdsOrRtnB32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_identity_zero(self) -> None:
        """OR with zero leaves memory unchanged and returns original value."""
        old, mem = oracle_ds_or_b32(0x12345678, 0x00000000)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, 0x12345678)

    def test_oracle_annihilator_all_ones(self) -> None:
        """OR with 0xFFFFFFFF sets all 32 bits."""
        old, mem = oracle_ds_or_b32(0x00000000, 0xFFFFFFFF)
        self.assertEqual(old, 0x00000000)
        self.assertEqual(mem, 0xFFFFFFFF)

        old2, mem2 = oracle_ds_or_b32(0x12345678, 0xFFFFFFFF)
        self.assertEqual(old2, 0x12345678)
        self.assertEqual(mem2, 0xFFFFFFFF)

    def test_oracle_idempotence(self) -> None:
        """OR with self produces identical value."""
        val = 0xA5A55A5A
        old, mem = oracle_ds_or_b32(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, val)

    def test_oracle_disjoint_bitmasks(self) -> None:
        """OR of disjoint bitmasks sets the union of bits without loss."""
        mask_a = 0xAAAA0000
        mask_b = 0x00005555
        old, mem = oracle_ds_or_b32(mask_a, mask_b)
        self.assertEqual(old, mask_a)
        self.assertEqual(mem, 0xAAAA5555)

    def test_oracle_overlapping_bitmasks(self) -> None:
        """OR of overlapping bitmasks correctly unifies set bits."""
        val_a = 0b1100_1010
        val_b = 0b0110_0111
        old, mem = oracle_ds_or_b32(val_a, val_b)
        self.assertEqual(old, val_a)
        self.assertEqual(mem, 0b1110_1111)

    def test_oracle_incremental_bit_accumulation(self) -> None:
        """Incrementally setting bits one by one constructs all 32 bits."""
        accum = 0
        for bit in range(32):
            bitmask = 1 << bit
            old, accum = oracle_ds_or_b32(accum, bitmask)
            self.assertEqual(old, (1 << bit) - 1)
            self.assertEqual(accum, (1 << (bit + 1)) - 1)
        self.assertEqual(accum, MASK_32)

    def test_effective_address_calculation_with_offset(self) -> None:
        """Effective address correctly adds base and 16-bit offset."""
        eff = calc_lds_effective_address(0x1000, 0x0024)
        self.assertEqual(eff, 0x1024)
        self.assertEqual(calc_lds_dword_index(eff), 0x1024 >> 2)

    def test_effective_address_wrapping_modulo_2_32(self) -> None:
        """Effective address wraps cleanly at 2^32 boundary."""
        eff = calc_lds_effective_address(0xFFFF_FFFE, 0x0004)
        self.assertEqual(eff, 0x0000_0002)

    def test_dword_alignment_drops_low_bits(self) -> None:
        """Hardware drops low 2 bits; byte offsets 0, 1, 2, 3 map to same dword."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, idx1)
        self.assertEqual(idx1, idx2)
        self.assertEqual(idx2, idx3)
        self.assertEqual(idx0, 0x100)

    def test_bank_distribution_across_32_banks(self) -> None:
        """Sequential dwords map cleanly across all 32 LDS banks."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))

    def test_lds_capacity_wrapping(self) -> None:
        """Addresses past capacity wrap modulo capacity_dwords."""
        cap_bytes = 65536
        idx_low = calc_lds_dword_index(0x100)
        idx_wrapped = calc_lds_dword_index(0x100 + cap_bytes)
        self.assertEqual(idx_low, idx_wrapped)

    def test_model_atomic_or_returns_pre_op_value(self) -> None:
        """LDSModel atomic_or_b32 updates memory and returns the pre-operation value."""
        lds = LDSModel()
        addr = 0x200
        lds.write_dword(addr, 0x00FF0000)

        old_val, new_val = lds.atomic_or_b32(addr, 0x0000FF00)
        self.assertEqual(old_val, 0x00FF0000)
        self.assertEqual(new_val, 0x00FFFF00)
        self.assertEqual(lds.read_dword(addr), 0x00FFFF00)

    def test_model_repeated_atomic_or_accumulation(self) -> None:
        """Repeated atomic OR on the same address accumulates bits."""
        lds = LDSModel()
        addr = 0x300
        lds.write_dword(addr, 0x01)

        old1, new1 = lds.atomic_or_b32(addr, 0x02)
        self.assertEqual(old1, 0x01)
        self.assertEqual(new1, 0x03)

        old2, new2 = lds.atomic_or_b32(addr, 0x04)
        self.assertEqual(old2, 0x03)
        self.assertEqual(new2, 0x07)

        old3, new3 = lds.atomic_or_b32(addr, 0x08)
        self.assertEqual(old3, 0x07)
        self.assertEqual(new3, 0x0F)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class VGPR:
    """Vector General Purpose Register operand (e.g. v[0] .. v[255])."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of architectural range [0, 255]")

    @property
    def name(self) -> str:
        return f"v[{self.index}]"


@dataclass(frozen=True)
class SGPR:
    """Scalar General Purpose Register operand (e.g. s[0] .. s[105])."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of architectural range [0, 105]")

    @property
    def name(self) -> str:
        return f"s[{self.index}]"


@dataclass(frozen=True)
class Immediate:
    """32-bit literal or inline integer constant."""
    value: int

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK_32:X}"


Operand = Union[VGPR, SGPR, Immediate]


class DSOpcode(Enum):
    """LDS Bitwise OR opcodes (returning and non-returning variants)."""
    DS_OR_RTN_B32 = auto()
    DS_OR_B32 = auto()

    @property
    def is_return(self) -> bool:
        return self == DSOpcode.DS_OR_RTN_B32

    @property
    def op_name(self) -> str:
        return "ds_or_rtn_b32" if self.is_return else "ds_or_b32"


@dataclass(frozen=True)
class DSOrInstruction:
    """Symbolic AST node for ds_or_rtn_b32 / ds_or_b32 instruction."""
    opcode: DSOpcode
    dst: Optional[VGPR]
    addr: VGPR
    data: Operand
    offset: int = 0

    def __post_init__(self) -> None:
        if self.opcode.is_return:
            if not isinstance(self.dst, VGPR):
                raise TypeError(f"Destination must be a VGPR for returning opcode, got {type(self.dst).__name__}")
        else:
            if self.dst is not None:
                raise ValueError("Destination must be None for non-returning ds_or_b32 opcode")
        if not isinstance(self.addr, VGPR):
            raise TypeError(f"Address operand must be a VGPR, got {type(self.addr).__name__}")
        if not isinstance(self.data, (VGPR, SGPR, Immediate)):
            raise TypeError(f"Data operand must be VGPR, SGPR, or Immediate, got {type(self.data).__name__}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset {self.offset} out of 16-bit range [0, 65535]")

    def disassemble(self) -> str:
        offset_part = f" offset:{self.offset}" if self.offset != 0 else ""
        if self.opcode.is_return:
            assert self.dst is not None
            return f"ds_or_rtn_b32 {self.dst.name}, {self.addr.name}, {self.data.name}{offset_part}"
        return f"ds_or_b32 {self.addr.name}, {self.data.name}{offset_part}"


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_ast_construction_returning(self) -> None:
        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
            offset=16,
        )
        self.assertEqual(instr.opcode, DSOpcode.DS_OR_RTN_B32)
        self.assertTrue(instr.opcode.is_return)
        self.assertIsNotNone(instr.dst)
        assert instr.dst is not None
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.addr.index, 1)
        self.assertEqual(instr.offset, 16)

    def test_instruction_ast_construction_non_returning(self) -> None:
        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_B32,
            dst=None,
            addr=VGPR(10),
            data=Immediate(0x1234),
        )
        self.assertEqual(instr.opcode, DSOpcode.DS_OR_B32)
        self.assertFalse(instr.opcode.is_return)
        self.assertIsNone(instr.dst)
        self.assertEqual(instr.addr.index, 10)
        self.assertEqual(instr.offset, 0)

    def test_disassembly_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(6),
            data=Immediate(0x1234),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_or_rtn_b32 v[5], v[6], 0x1234",
        )

        instr_off = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(12),
            addr=VGPR(13),
            data=SGPR(10),
            offset=64,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_or_rtn_b32 v[12], v[13], s[10] offset:64",
        )

    def test_disassembly_non_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSOrInstruction(
            opcode=DSOpcode.DS_OR_B32,
            dst=None,
            addr=VGPR(2),
            data=VGPR(3),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_or_b32 v[2], v[3]",
        )

        instr_off = DSOrInstruction(
            opcode=DSOpcode.DS_OR_B32,
            dst=None,
            addr=VGPR(2),
            data=VGPR(3),
            offset=128,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_or_b32 v[2], v[3] offset:128",
        )

    def test_dst_type_enforcement_rejects_sgpr_and_none(self) -> None:
        with self.assertRaises(TypeError):
            DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=SGPR(0),  # type: ignore[arg-type]
                addr=VGPR(1),
                data=Immediate(0),
            )
        with self.assertRaises(TypeError):
            DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=None,
                addr=VGPR(1),
                data=Immediate(0),
            )

    def test_addr_type_enforcement_rejects_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(0),
                addr=SGPR(1),  # type: ignore[arg-type]
                data=Immediate(0),
            )

    def test_non_returning_rejects_non_none_dst(self) -> None:
        with self.assertRaises(ValueError):
            DSOrInstruction(
                opcode=DSOpcode.DS_OR_B32,
                dst=VGPR(0),
                addr=VGPR(1),
                data=Immediate(0),
            )

    def test_register_index_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            SGPR(106)
        with self.assertRaises(ValueError):
            SGPR(-1)

    def test_offset_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(0),
                addr=VGPR(1),
                data=Immediate(0),
                offset=65536,
            )
        with self.assertRaises(ValueError):
            DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(0),
                addr=VGPR(1),
                data=Immediate(0),
                offset=-1,
            )

    def test_operand_imm_and_reg_equality(self) -> None:
        v0_a = VGPR(0)
        v0_b = VGPR(0)
        v1 = VGPR(1)
        self.assertEqual(v0_a, v0_b)
        self.assertNotEqual(v0_a, v1)
        imm1 = Immediate(42)
        imm2 = Immediate(42)
        self.assertEqual(imm1, imm2)


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register and scratchpad state for scalar execution."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    lds: LDSModel = field(default_factory=LDSModel)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK_32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK_32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK_32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK_32


class ScalarInterpreter:
    """Deterministic scalar interpreter for ds_or_rtn_b32 / ds_or_b32."""

    def resolve_operand(self, op: Operand, state: ScalarMachineState) -> int:
        if isinstance(op, VGPR):
            return state.get_vgpr(op.index)
        if isinstance(op, SGPR):
            return state.get_sgpr(op.index)
        if isinstance(op, Immediate):
            return op.value & MASK_32
        raise TypeError(f"Unknown operand type: {type(op).__name__}")

    def step(
        self,
        instr: DSOrInstruction,
        state: ScalarMachineState,
    ) -> Tuple[Optional[int], int]:
        """Execute a single ds_or_rtn_b32 / ds_or_b32 instruction deterministically.
        
        All operands are latched before write-back to ensure operand preservation.
        Returns:
            (returned_old_val if returning else None, updated_mem_val)
        """
        # 1. Latch inputs
        vaddr = state.get_vgpr(instr.addr.index)
        data_val = self.resolve_operand(instr.data, state)

        # 2. Compute effective byte address and fetch current LDS value
        eff_addr = calc_lds_effective_address(vaddr, instr.offset)
        curr_mem = state.lds.read_dword(eff_addr)

        # 3. Compute bitwise OR
        old_val, next_mem = oracle_ds_or_b32(curr_mem, data_val)

        # 4. Atomic memory writeback
        state.lds.write_dword(eff_addr, next_mem)

        # 5. Write return value to destination VGPR if returning variant
        if instr.opcode.is_return:
            assert instr.dst is not None
            state.set_vgpr(instr.dst.index, old_val)
            return old_val, next_mem

        return None, next_mem


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_step_returning_vgpr_operand(self) -> None:
        addr = 0x200
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0x00FF0000)
        self.state.lds.write_dword(addr, 0x0000FF00)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x0000FF00)
        self.assertEqual(mem, 0x00FFFF00)
        self.assertEqual(self.state.get_vgpr(0), 0x0000FF00)
        self.assertEqual(self.state.lds.read_dword(addr), 0x00FFFF00)

    def test_scalar_step_returning_sgpr_operand(self) -> None:
        addr = 0x300
        self.state.set_vgpr(1, addr)
        self.state.set_sgpr(10, 0x12340000)
        self.state.lds.write_dword(addr, 0x00005678)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(3),
            addr=VGPR(1),
            data=SGPR(10),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x00005678)
        self.assertEqual(mem, 0x12345678)
        self.assertEqual(self.state.get_vgpr(3), 0x00005678)
        self.assertEqual(self.state.lds.read_dword(addr), 0x12345678)

    def test_scalar_step_returning_immediate_operand(self) -> None:
        addr = 0x400
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(addr, 0xAA00AA00)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(4),
            addr=VGPR(1),
            data=Immediate(0x00550055),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xAA00AA00)
        self.assertEqual(mem, 0xAA55AA55)
        self.assertEqual(self.state.get_vgpr(4), 0xAA00AA00)
        self.assertEqual(self.state.lds.read_dword(addr), 0xAA55AA55)

    def test_scalar_step_non_returning_opcode(self) -> None:
        addr = 0x500
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0x0000000F)
        self.state.lds.write_dword(addr, 0x000000F0)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_B32,
            dst=None,
            addr=VGPR(1),
            data=VGPR(2),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertIsNone(old)
        self.assertEqual(mem, 0x000000FF)
        self.assertEqual(self.state.lds.read_dword(addr), 0x000000FF)

    def test_scalar_step_with_immediate_offset(self) -> None:
        base_addr = 0x600
        offset = 0x08
        eff_addr = base_addr + offset
        self.state.set_vgpr(2, base_addr)
        self.state.lds.write_dword(eff_addr, 0x10101010)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(2),
            data=Immediate(0x01010101),
            offset=offset,
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x10101010)
        self.assertEqual(mem, 0x11111111)
        self.assertEqual(self.state.get_vgpr(5), 0x10101010)
        self.assertEqual(self.state.lds.read_dword(eff_addr), 0x11111111)
        # Base dword remains untouched
        self.assertEqual(self.state.lds.read_dword(base_addr), 0)

    def test_scalar_step_idempotent_or_retains_value(self) -> None:
        addr = 0x700
        val = 0xCAFE_BABE
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(addr, val)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(6),
            addr=VGPR(1),
            data=Immediate(0),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, val)
        self.assertEqual(mem, val)
        self.assertEqual(self.state.lds.read_dword(addr), val)

    def test_scalar_step_bit_accumulation_sequence(self) -> None:
        addr = 0x800
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(addr, 0)

        for bit in range(8):
            instr = DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(7),
                addr=VGPR(1),
                data=Immediate(1 << bit),
            )
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, (1 << bit) - 1)
            self.assertEqual(mem, (1 << (bit + 1)) - 1)
            self.assertEqual(self.state.get_vgpr(7), (1 << bit) - 1)

    def test_scalar_step_out_of_bounds_address_wrapping(self) -> None:
        cap_bytes = 65536
        addr = cap_bytes + 0x100
        eff_wrapped = 0x100
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(eff_wrapped, 0x11110000)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(8),
            addr=VGPR(1),
            data=Immediate(0x00002222),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x11110000)
        self.assertEqual(mem, 0x11112222)
        self.assertEqual(self.state.lds.read_dword(eff_wrapped), 0x11112222)


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state supporting Wave32 and Wave64 execution."""
    num_lanes: int = 32
    exec_mask: int = 0xFFFF_FFFF
    vgprs: List[List[int]] = field(default_factory=lambda: [])
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    lds: LDSModel = field(default_factory=LDSModel)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        elif len(self.vgprs) != self.num_lanes:
            raise ValueError(f"Expected {self.num_lanes} lane VGPR sets, got {len(self.vgprs)}")

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes - 1}]")
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK_32

    def set_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK_32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK_32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK_32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and atomic serialization."""

    def resolve_lane_operand(self, op: Operand, lane: int, state: SIMDState) -> int:
        if isinstance(op, VGPR):
            return state.get_vgpr(lane, op.index)
        if isinstance(op, SGPR):
            return state.get_sgpr(op.index)
        if isinstance(op, Immediate):
            return op.value & MASK_32
        raise TypeError(f"Unknown operand type: {type(op).__name__}")

    def step(
        self,
        instr: DSOrInstruction,
        state: SIMDState,
    ) -> List[Tuple[int, Optional[int], int]]:
        """Execute instruction across active lanes in ascending order.
        
        Hardware atomic operations on LDS serialize collisions deterministically
        in lane index order (0 to num_lanes - 1).
        Returns:
            List of (lane, returned_old_val, updated_mem_val) for all active lanes.
        """
        results: List[Tuple[int, Optional[int], int]] = []

        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue

            # 1. Latch lane inputs
            vaddr = state.get_vgpr(lane, instr.addr.index)
            data_val = self.resolve_lane_operand(instr.data, lane, state)

            # 2. Compute effective byte address
            eff_addr = calc_lds_effective_address(vaddr, instr.offset)

            # 3. Read current value and perform atomic bitwise OR
            curr_mem = state.lds.read_dword(eff_addr)
            old_val, next_mem = oracle_ds_or_b32(curr_mem, data_val)
            state.lds.write_dword(eff_addr, next_mem)

            # 4. If returning variant, write to lane's destination VGPR
            if instr.opcode.is_return:
                assert instr.dst is not None
                state.set_vgpr(lane, instr.dst.index, old_val)
                results.append((lane, old_val, next_mem))
            else:
                results.append((lane, None, next_mem))

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_independent_addresses(self) -> None:
        """All 32 lanes active with independent addresses."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(32):
            addr = 0x200 + lane * 4
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 1 << lane)
            state.lds.write_dword(addr, 0x1000 + lane)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)

        for lane in range(32):
            addr = 0x200 + lane * 4
            expected_old = 0x1000 + lane
            expected_mem = (0x1000 + lane) | (1 << lane)
            self.assertEqual(state.get_vgpr(lane, 0), expected_old)
            self.assertEqual(state.lds.read_dword(addr), expected_mem)

    def test_wave64_uniform_full_lds_spread(self) -> None:
        """64 lanes active in Wave64 mode across LDS banks."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x100 + lane * 4
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xFF)
            state.lds.write_dword(addr, lane)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)

        for lane in range(64):
            addr = 0x100 + lane * 4
            self.assertEqual(state.get_vgpr(lane, 5), lane)
            self.assertEqual(state.lds.read_dword(addr), lane | 0xFF)

    def test_wave32_exec_mask_alternating_lanes(self) -> None:
        """Even lanes active (0x55555555), odd lanes inactive."""
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        interp = SIMDInterpreter()

        sentinel = 0xDEAD_BEEF
        for lane in range(32):
            addr = 0x400 + lane * 4
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xF0)
            state.lds.write_dword(addr, 0x0F)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 16)

        for lane in range(32):
            addr = 0x400 + lane * 4
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr(lane, 0), 0x0F)
                self.assertEqual(state.lds.read_dword(addr), 0xFF)
            else:
                # Inactive lane must remain untouched
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(addr), 0x0F)

    def test_wave32_exec_mask_upper_half_disabled(self) -> None:
        """Lower 16 lanes active, upper 16 inactive."""
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        interp = SIMDInterpreter()
        sentinel = 0xCAFE_0000

        for lane in range(32):
            addr = 0x500 + lane * 4
            state.set_vgpr(lane, 3, sentinel)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xAA)
            state.lds.write_dword(addr, 0x55)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(3),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 16)

        for lane in range(16):
            self.assertEqual(state.get_vgpr(lane, 3), 0x55)
            self.assertEqual(state.lds.read_dword(0x500 + lane * 4), 0xFF)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr(lane, 3), sentinel)
            self.assertEqual(state.lds.read_dword(0x500 + lane * 4), 0x55)

    def test_wave32_exec_mask_single_lane(self) -> None:
        """Only lane 17 active."""
        target_lane = 17
        state = SIMDState(num_lanes=32, exec_mask=(1 << target_lane))
        interp = SIMDInterpreter()

        addr = 0x600
        state.set_vgpr(target_lane, 1, addr)
        state.set_vgpr(target_lane, 2, 0x01)
        state.lds.write_dword(addr, 0x10)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], target_lane)
        self.assertEqual(state.get_vgpr(target_lane, 0), 0x10)
        self.assertEqual(state.lds.read_dword(addr), 0x11)

    def test_wave32_exec_mask_all_zero(self) -> None:
        """All lanes disabled (exec_mask = 0). No modification to memory or registers."""
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        interp = SIMDInterpreter()

        addr = 0x700
        state.lds.write_dword(addr, 0x42)
        for lane in range(32):
            state.set_vgpr(lane, 0, 0x9999)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xFFFF)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 0)
        self.assertEqual(state.lds.read_dword(addr), 0x42)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x9999)

    def test_wave32_same_address_collision_atomic_serialization(self) -> None:
        """Multiple active lanes target the exact same LDS dword.
        
        Hardware serializes atomics in ascending order of lane index:
        Lane 0 updates LDS and gets old value, then Lane 1, etc.
        """
        state = SIMDState(num_lanes=32, exec_mask=0x0000_000F)  # Lanes 0..3 active
        interp = SIMDInterpreter()
        common_addr = 0x800
        state.lds.write_dword(common_addr, 0x0000)

        # Lane 0 sets bit 0, Lane 1 sets bit 1, Lane 2 sets bit 2, Lane 3 sets bit 3
        for lane in range(4):
            state.set_vgpr(lane, 1, common_addr)
            state.set_vgpr(lane, 2, 1 << lane)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 4)

        # Lane 0 executed first: saw 0x00, wrote 0x01
        self.assertEqual(state.get_vgpr(0, 0), 0x00)
        # Lane 1 executed second: saw 0x01, wrote 0x03
        self.assertEqual(state.get_vgpr(1, 0), 0x01)
        # Lane 2 executed third: saw 0x03, wrote 0x07
        self.assertEqual(state.get_vgpr(2, 0), 0x03)
        # Lane 3 executed fourth: saw 0x07, wrote 0x0F
        self.assertEqual(state.get_vgpr(3, 0), 0x07)

        # Final memory holds accumulated OR of all 4 lanes
        self.assertEqual(state.lds.read_dword(common_addr), 0x0F)

    def test_wave64_exec_mask_64bit_full_coverage(self) -> None:
        """64-bit mask with odd lanes active (0xAAAAAAAAAAAAAAAA)."""
        mask64 = 0xAAAA_AAAA_AAAA_AAAA
        state = SIMDState(num_lanes=64, exec_mask=mask64)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x900 + lane * 4
            state.set_vgpr(lane, 4, 0)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0x55)
            state.lds.write_dword(addr, 0xAA)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(4),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)

        for lane in range(64):
            addr = 0x900 + lane * 4
            if (mask64 >> lane) & 1:
                self.assertEqual(state.get_vgpr(lane, 4), 0xAA)
                self.assertEqual(state.lds.read_dword(addr), 0xFF)
            else:
                self.assertEqual(state.get_vgpr(lane, 4), 0)
                self.assertEqual(state.lds.read_dword(addr), 0xAA)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_alias_dst_equals_addr(self) -> None:
        """Destination VGPR aliases Address VGPR: v[0] = ds_or_rtn_b32 v[0], v[1].
        
        Address must be latched in the operand stage before v[0] is overwritten with old_val.
        """
        state = ScalarMachineState()
        addr = 0x500
        state.set_vgpr(0, addr)        # v[0] is address AND destination
        state.set_vgpr(1, 0x00FF00FF)  # v[1] is data operand
        state.lds.write_dword(addr, 0xFF00FF00)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(0),
            data=VGPR(1),
        )
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0xFF00FF00)
        self.assertEqual(mem, 0xFFFFFFFF)
        # v[0] now contains the returned old_val
        self.assertEqual(state.get_vgpr(0), 0xFF00FF00)
        # Memory at original addr was properly updated
        self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)

    def test_alias_dst_equals_data(self) -> None:
        """Destination VGPR aliases Data VGPR: v[1] = ds_or_rtn_b32 v[0], v[1].
        
        Data must be latched before v[1] is overwritten with old_val.
        """
        state = ScalarMachineState()
        addr = 0x600
        state.set_vgpr(0, addr)
        state.set_vgpr(1, 0x0F0F0F0F)  # v[1] is data AND destination
        state.lds.write_dword(addr, 0xF0F0F0F0)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(1),
            addr=VGPR(0),
            data=VGPR(1),
        )
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0xF0F0F0F0)
        self.assertEqual(mem, 0xFFFFFFFF)
        self.assertEqual(state.get_vgpr(1), 0xF0F0F0F0)
        self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)

    def test_alias_addr_equals_data(self) -> None:
        """Address VGPR aliases Data VGPR: v[2] = ds_or_rtn_b32 v[0], v[0]."""
        state = ScalarMachineState()
        addr_and_data = 0x200
        state.set_vgpr(0, addr_and_data)
        state.lds.write_dword(addr_and_data, 0x01)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(2),
            addr=VGPR(0),
            data=VGPR(0),
        )
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0x01)
        self.assertEqual(mem, 0x01 | 0x200)
        self.assertEqual(state.get_vgpr(2), 0x01)
        self.assertEqual(state.lds.read_dword(addr_and_data), 0x01 | 0x200)
        self.assertEqual(state.get_vgpr(0), addr_and_data)

    def test_full_operand_aliasing_three_way(self) -> None:
        """Destination, Address, and Data all share the same register: v[5] = ds_or_rtn_b32 v[5], v[5]."""
        state = ScalarMachineState()
        val = 0x400
        state.set_vgpr(5, val)
        state.lds.write_dword(val, 0x1000)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(5),
            data=VGPR(5),
        )
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0x1000)
        self.assertEqual(mem, 0x1000 | 0x400)
        # v[5] receives old_val
        self.assertEqual(state.get_vgpr(5), 0x1000)
        # Memory at address 0x400 was updated
        self.assertEqual(state.lds.read_dword(val), 0x1000 | 0x400)

    def test_unrelated_vgprs_unmodified(self) -> None:
        """Canary test: ensure all 256 VGPRs outside destination remain untouched."""
        state = ScalarMachineState()
        canary_base = 0xDEAD0000
        for r in range(256):
            state.set_vgpr(r, canary_base | r)

        addr = 0x100
        state.set_vgpr(10, addr)
        state.set_vgpr(11, 0x55)
        state.lds.write_dword(addr, 0xAA)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(9),
            addr=VGPR(10),
            data=VGPR(11),
        )
        self.scalar_interp.step(instr, state)

        # Destination v[9] was updated
        self.assertEqual(state.get_vgpr(9), 0xAA)
        # Operands unchanged
        self.assertEqual(state.get_vgpr(10), addr)
        self.assertEqual(state.get_vgpr(11), 0x55)

        # Check all other 253 registers
        for r in range(256):
            if r not in (9, 10, 11):
                self.assertEqual(state.get_vgpr(r), canary_base | r)

    def test_simd_multi_lane_aliasing(self) -> None:
        """All 32 lanes alias destination with address: v[0] = ds_or_rtn_b32 v[0], v[1]."""
        state = SIMDState(num_lanes=32)
        for lane in range(32):
            addr = 0x300 + lane * 4
            state.set_vgpr(lane, 0, addr)        # v[0] = addr AND dst
            state.set_vgpr(lane, 1, 1 << lane)   # v[1] = data
            state.lds.write_dword(addr, 0x8000_0000)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(0),
            data=VGPR(1),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            addr = 0x300 + lane * 4
            self.assertEqual(state.get_vgpr(lane, 0), 0x8000_0000)
            self.assertEqual(state.lds.read_dword(addr), 0x8000_0000 | (1 << lane))


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AtomicBitwiseFlags:
    """Architectural condition codes, status flags, and arithmetic analysis for ds_or_rtn_b32."""
    zero: bool
    negative: bool
    old_zero: bool
    old_negative: bool
    parity_even: bool
    popcount: int
    changed: bool
    bits_added: int
    unaligned_addr: bool
    bank: int
    carry_or_sum: bool
    borrow: bool


def evaluate_ds_or_flags(
    old_val: int,
    or_val: int,
    new_mem: int,
    eff_addr: int,
) -> AtomicBitwiseFlags:
    """Evaluate condition codes, bitwise status flags, and arithmetic carry/borrow metrics."""
    u_old = old_val & MASK_32
    u_or = or_val & MASK_32
    u_new = new_mem & MASK_32

    zero = (u_new == 0)
    negative = bool(u_new & 0x8000_0000)
    old_zero = (u_old == 0)
    old_negative = bool(u_old & 0x8000_0000)

    popcount = bin(u_new).count("1")
    parity_even = (popcount % 2 == 0)

    bits_added = (u_or & (~u_old)) & MASK_32
    changed = (bits_added != 0)

    unaligned = bool(eff_addr & 0x3)
    bank = calc_lds_bank(eff_addr)

    carry_or_sum = ((u_old + u_or) > MASK_32)
    borrow = (u_old < u_or)

    return AtomicBitwiseFlags(
        zero=zero,
        negative=negative,
        old_zero=old_zero,
        old_negative=old_negative,
        parity_even=parity_even,
        popcount=popcount,
        changed=changed,
        bits_added=bits_added,
        unaligned_addr=unaligned,
        bank=bank,
        carry_or_sum=carry_or_sum,
        borrow=borrow,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_flags_zero_result(self) -> None:
        flags = evaluate_ds_or_flags(
            old_val=0,
            or_val=0,
            new_mem=0,
            eff_addr=0x100,
        )
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.old_zero)
        self.assertFalse(flags.old_negative)
        self.assertTrue(flags.parity_even)
        self.assertEqual(flags.popcount, 0)
        self.assertFalse(flags.changed)
        self.assertEqual(flags.bits_added, 0)
        self.assertFalse(flags.unaligned_addr)
        self.assertEqual(flags.bank, (0x100 >> 2) & 31)

    def test_flags_negative_msb_set(self) -> None:
        flags = evaluate_ds_or_flags(
            old_val=0x0000_0001,
            or_val=0x8000_0000,
            new_mem=0x8000_0001,
            eff_addr=0x200,
        )
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertFalse(flags.old_zero)
        self.assertFalse(flags.old_negative)
        self.assertEqual(flags.popcount, 2)
        self.assertTrue(flags.parity_even)
        self.assertTrue(flags.changed)
        self.assertEqual(flags.bits_added, 0x8000_0000)

    def test_flags_old_zero_and_negative(self) -> None:
        flags = evaluate_ds_or_flags(
            old_val=0x8000_0000,
            or_val=0x0000_0002,
            new_mem=0x8000_0002,
            eff_addr=0x300,
        )
        self.assertFalse(flags.old_zero)
        self.assertTrue(flags.old_negative)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.changed)
        self.assertEqual(flags.bits_added, 0x0000_0002)

    def test_flags_parity_and_popcount(self) -> None:
        # Odd popcount: 0b111 = 3 bits set -> parity_even is False
        flags_odd = evaluate_ds_or_flags(0, 0b111, 0b111, eff_addr=0x400)
        self.assertEqual(flags_odd.popcount, 3)
        self.assertFalse(flags_odd.parity_even)

        # Even popcount: 0b1111 = 4 bits set -> parity_even is True
        flags_even = evaluate_ds_or_flags(0, 0b1111, 0b1111, eff_addr=0x400)
        self.assertEqual(flags_even.popcount, 4)
        self.assertTrue(flags_even.parity_even)

    def test_flags_changed_and_bits_added(self) -> None:
        # Subsumed mask: ORing 0x01 into memory that already has 0x03
        flags_unchanged = evaluate_ds_or_flags(
            old_val=0x03,
            or_val=0x01,
            new_mem=0x03,
            eff_addr=0x500,
        )
        self.assertFalse(flags_unchanged.changed)
        self.assertEqual(flags_unchanged.bits_added, 0)

        # New bits added: ORing 0x04 into memory that has 0x03
        flags_changed = evaluate_ds_or_flags(
            old_val=0x03,
            or_val=0x04,
            new_mem=0x07,
            eff_addr=0x500,
        )
        self.assertTrue(flags_changed.changed)
        self.assertEqual(flags_changed.bits_added, 0x04)

    def test_flags_unaligned_address_detection(self) -> None:
        flags_aligned = evaluate_ds_or_flags(0, 0, 0, eff_addr=0x1000)
        self.assertFalse(flags_aligned.unaligned_addr)

        for byte_off in (1, 2, 3):
            flags_unaligned = evaluate_ds_or_flags(0, 0, 0, eff_addr=0x1000 + byte_off)
            self.assertTrue(flags_unaligned.unaligned_addr)

    def test_flags_bank_selection(self) -> None:
        for bank_idx in range(32):
            addr = bank_idx * 4
            flags = evaluate_ds_or_flags(0, 0, 0, eff_addr=addr)
            self.assertEqual(flags.bank, bank_idx)

    def test_flags_carry_and_borrow_metrics(self) -> None:
        # Carry if addition would overflow: 0xFFFFFFFF + 1 > 0xFFFFFFFF
        flags_carry = evaluate_ds_or_flags(
            old_val=0xFFFF_FFFF,
            or_val=0x0000_0001,
            new_mem=0xFFFF_FFFF,
            eff_addr=0x600,
        )
        self.assertTrue(flags_carry.carry_or_sum)

        # No carry if sum fits: 0x1000 + 0x2000
        flags_no_carry = evaluate_ds_or_flags(
            old_val=0x1000,
            or_val=0x2000,
            new_mem=0x3000,
            eff_addr=0x600,
        )
        self.assertFalse(flags_no_carry.carry_or_sum)

        # Borrow when old_val < or_val
        flags_borrow = evaluate_ds_or_flags(
            old_val=10,
            or_val=20,
            new_mem=30,
            eff_addr=0x600,
        )
        self.assertTrue(flags_borrow.borrow)

        # No borrow when old_val >= or_val
        flags_no_borrow = evaluate_ds_or_flags(
            old_val=20,
            or_val=10,
            new_mem=30,
            eff_addr=0x600,
        )
        self.assertFalse(flags_no_borrow.borrow)


# ==============================================================================
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

# IEEE 754 Single-Precision (binary32) Architectural Constants
F32_POS_ZERO: int = 0x0000_0000
F32_NEG_ZERO: int = 0x8000_0000
F32_POS_INF: int = 0x7F80_0000
F32_NEG_INF: int = 0xFF80_0000
F32_CANONICAL_QNAN: int = 0x7FC0_0000
F32_SIGNALING_NAN: int = 0x7F80_0001
F32_MIN_SUBNORMAL: int = 0x0000_0001
F32_MAX_SUBNORMAL: int = 0x007F_FFFF
F32_MIN_NORMAL: int = 0x0080_0000
F32_MAX_NORMAL: int = 0x7F7F_FFFF


def f32_to_bits(val: float) -> int:
    """Convert float to IEEE 754 32-bit unsigned integer representation."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Convert IEEE 754 32-bit integer representation to Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK_32))[0]


def is_f32_nan(bits: int) -> bool:
    """Check if bits represent an IEEE 754 NaN."""
    return ((bits & 0x7F80_0000) == 0x7F80_0000) and ((bits & 0x007F_FFFF) != 0)


def is_f32_subnormal(bits: int) -> bool:
    """Check if bits represent an IEEE 754 subnormal number."""
    return ((bits & 0x7F80_0000) == 0) and ((bits & 0x007F_FFFF) != 0)


class TestIEEE754BoundaryExtremesAndBitwiseLimits(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.lds = LDSModel()
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState(lds=self.lds)

    def test_signed_zeros_bitwise_or(self) -> None:
        """ORing sign bit 0x80000000 into +0.0 results in -0.0."""
        addr = 0x100
        self.lds.write_dword(addr, F32_POS_ZERO)
        self.state.set_vgpr(0, addr)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(1),
            addr=VGPR(0),
            data=Immediate(F32_NEG_ZERO),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, F32_POS_ZERO)
        self.assertEqual(mem, F32_NEG_ZERO)
        self.assertEqual(self.state.get_vgpr(1), F32_POS_ZERO)
        self.assertEqual(self.lds.read_dword(addr), F32_NEG_ZERO)

    def test_subnormals_preservation_no_ftz(self) -> None:
        """LDS bitwise operations preserve subnormals without flush-to-zero (FTZ)."""
        self.assertTrue(is_f32_subnormal(F32_MIN_SUBNORMAL))
        self.assertTrue(is_f32_subnormal(F32_MAX_SUBNORMAL))

        addr = 0x200
        self.lds.write_dword(addr, F32_MIN_SUBNORMAL)
        self.state.set_vgpr(0, addr)

        instr = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(2),
            addr=VGPR(0),
            data=Immediate(0x0000_0002),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, F32_MIN_SUBNORMAL)
        self.assertEqual(mem, 0x0000_0003)
        self.assertTrue(is_f32_subnormal(mem))

    def test_signaling_nan_preserved_unless_quiet_bit_explicitly_set(self) -> None:
        """Signaling NaNs are not quietened unless quiet bit (bit 22) is set in mask."""
        addr = 0x300
        self.lds.write_dword(addr, F32_SIGNALING_NAN)
        self.state.set_vgpr(0, addr)

        # OR with payload bits that keep bit 22 clear
        instr_keep_snan = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(3),
            addr=VGPR(0),
            data=Immediate(0x0000_0010),
        )
        old, mem = self.interp.step(instr_keep_snan, self.state)
        self.assertEqual(old, F32_SIGNALING_NAN)
        self.assertEqual(mem, 0x7F80_0011)
        self.assertTrue(is_f32_nan(mem))
        # Bit 22 is quiet bit: must still be 0 for signaling NaN
        self.assertEqual(mem & 0x0040_0000, 0)

        # Now explicitly set quiet bit
        instr_quiet = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(3),
            addr=VGPR(0),
            data=Immediate(0x0040_0000),
        )
        old2, mem2 = self.interp.step(instr_quiet, self.state)
        self.assertEqual(old2, 0x7F80_0011)
        self.assertEqual(mem2, 0x7FC0_0011)
        # Bit 22 is now set (Quiet NaN)
        self.assertNotEqual(mem2 & 0x0040_0000, 0)

    def test_infinities_sign_bit_and_payload_injection(self) -> None:
        """ORing sign bit transforms +inf into -inf; ORing payload turns inf into NaN."""
        addr = 0x400
        self.lds.write_dword(addr, F32_POS_INF)
        self.state.set_vgpr(0, addr)

        # OR with sign bit -> negative infinity
        instr_neg_inf = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(4),
            addr=VGPR(0),
            data=Immediate(F32_NEG_ZERO),
        )
        old, mem = self.interp.step(instr_neg_inf, self.state)
        self.assertEqual(old, F32_POS_INF)
        self.assertEqual(mem, F32_NEG_INF)

        # OR with fraction payload -> NaN
        instr_nan = DSOrInstruction(
            opcode=DSOpcode.DS_OR_RTN_B32,
            dst=VGPR(4),
            addr=VGPR(0),
            data=Immediate(0x0000_0001),
        )
        old2, mem2 = self.interp.step(instr_nan, self.state)
        self.assertEqual(old2, F32_NEG_INF)
        self.assertEqual(mem2, 0xFF80_0001)
        self.assertTrue(is_f32_nan(mem2))

    def test_walking_ones_boundary_coverage(self) -> None:
        """Test bitwise OR across all 32 single-bit patterns."""
        addr = 0x500
        for bit in range(32):
            self.lds.write_dword(addr, 0)
            self.state.set_vgpr(0, addr)
            mask = 1 << bit

            instr = DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(5),
                addr=VGPR(0),
                data=Immediate(mask),
            )
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, 0)
            self.assertEqual(mem, mask)
            self.assertEqual(self.state.get_vgpr(5), 0)
            self.assertEqual(self.lds.read_dword(addr), mask)

    def test_walking_zeros_boundary_coverage(self) -> None:
        """Test bitwise OR when starting from inverted single-bit masks."""
        addr = 0x600
        for bit in range(32):
            inv_mask = (~(1 << bit)) & MASK_32
            self.lds.write_dword(addr, inv_mask)
            self.state.set_vgpr(0, addr)
            fill_bit = 1 << bit

            instr = DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(6),
                addr=VGPR(0),
                data=Immediate(fill_bit),
            )
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, inv_mask)
            self.assertEqual(mem, MASK_32)
            self.assertEqual(self.state.get_vgpr(6), inv_mask)
            self.assertEqual(self.lds.read_dword(addr), MASK_32)

    def test_integer_boundary_extremes(self) -> None:
        """Sequential OR across critical 32-bit boundary integers."""
        extremes = [0x0000_0000, 0x0000_0001, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF]
        addr = 0x700
        self.state.set_vgpr(0, addr)
        accum = 0
        self.lds.write_dword(addr, accum)

        for val in extremes:
            instr = DSOrInstruction(
                opcode=DSOpcode.DS_OR_RTN_B32,
                dst=VGPR(7),
                addr=VGPR(0),
                data=Immediate(val),
            )
            expected_old = accum
            accum = accum | val
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, expected_old)
            self.assertEqual(mem, accum)


if __name__ == "__main__":
    unittest.main()
