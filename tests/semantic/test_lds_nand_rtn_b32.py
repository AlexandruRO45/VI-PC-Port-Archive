"""Semantic reference oracle and architectural model for ds_nand_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Bitwise Atomic NAND with Return:
Performs a 32-bit atomic bitwise NAND operation on LDS memory and returns
the previous memory value to the destination VGPR.

Syntax:
    ds_nand_rtn_b32 vdst, vaddr, vdata [offset:offset16]
    ds_nand_b32 vaddr, vdata [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_capacity_dwords - 1)
    old_val = LDS[dword_index]
    new_val = ~(old_val & vdata) & 0xFFFFFFFF
    LDS[dword_index] = new_val
    vdst = old_val
"""
from __future__ import annotations

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


def oracle_ds_nand_b32(current_mem: int, nand_val: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nand_rtn_b32.
    
    Returns:
        (returned_old_val, new_mem_val)
    """
    old_val = current_mem & MASK_32
    vdata = nand_val & MASK_32
    new_mem = (~(old_val & vdata)) & MASK_32
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

    def atomic_nand_b32(
        self,
        vaddr: int,
        nand_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic bitwise NAND on LDS dword, returning (old_val, new_val)."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_nand_b32(curr, nand_val)
        self.memory[idx] = new_mem
        return old_val, new_mem


class TestLdsNandRtnB32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_nand_with_zero(self) -> None:
        """NAND with 0 always sets all 32 bits (~0 == 0xFFFFFFFF)."""
        old, mem = oracle_ds_nand_b32(0x12345678, 0x00000000)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, 0xFFFFFFFF)

        old2, mem2 = oracle_ds_nand_b32(0x00000000, 0x00000000)
        self.assertEqual(old2, 0x00000000)
        self.assertEqual(mem2, 0xFFFFFFFF)

    def test_oracle_nand_with_all_ones_inversion(self) -> None:
        """NAND with 0xFFFFFFFF computes bitwise NOT (~x)."""
        val = 0xAA55AA55
        old, mem = oracle_ds_nand_b32(val, 0xFFFFFFFF)
        self.assertEqual(old, val)
        self.assertEqual(mem, 0x55AA55AA)

    def test_oracle_self_nand_inversion(self) -> None:
        """x NAND x computes ~x."""
        val = 0x12345678
        old, mem = oracle_ds_nand_b32(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, (~val) & MASK_32)

    def test_oracle_disjoint_bitmasks(self) -> None:
        """NAND of disjoint bitmasks produces 0xFFFFFFFF because AND is 0."""
        mask_a = 0xAAAA0000
        mask_b = 0x00005555
        old, mem = oracle_ds_nand_b32(mask_a, mask_b)
        self.assertEqual(old, mask_a)
        self.assertEqual(mem, 0xFFFFFFFF)

    def test_oracle_overlapping_bitmasks(self) -> None:
        """NAND of overlapping bitmasks correctly inverts common set bits."""
        val_a = 0b1100_1010
        val_b = 0b0110_0111
        # val_a & val_b = 0b0100_0010
        # ~(val_a & val_b) has 0 at bits 1 and 6, 1 elsewhere
        old, mem = oracle_ds_nand_b32(val_a, val_b)
        self.assertEqual(old, val_a)
        expected = (~0b0100_0010) & MASK_32
        self.assertEqual(mem, expected)

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

    def test_model_atomic_nand_returns_pre_op_value(self) -> None:
        """LDSModel atomic_nand_b32 updates memory and returns the pre-operation value."""
        lds = LDSModel()
        addr = 0x200
        lds.write_dword(addr, 0x00FF00FF)

        old_val, new_val = lds.atomic_nand_b32(addr, 0x0000FFFF)
        # old & data = 0x000000FF
        # new = ~0x000000FF = 0xFFFFFF00
        self.assertEqual(old_val, 0x00FF00FF)
        self.assertEqual(new_val, 0xFFFFFF00)
        self.assertEqual(lds.read_dword(addr), 0xFFFFFF00)


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
    """LDS Bitwise NAND opcodes (returning and non-returning variants)."""
    DS_NAND_RTN_B32 = auto()
    DS_NAND_B32 = auto()

    @property
    def is_return(self) -> bool:
        return self == DSOpcode.DS_NAND_RTN_B32

    @property
    def op_name(self) -> str:
        return "ds_nand_rtn_b32" if self.is_return else "ds_nand_b32"


@dataclass(frozen=True)
class DSNandInstruction:
    """Symbolic AST node for ds_nand_rtn_b32 / ds_nand_b32 instruction."""
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
                raise ValueError("Destination must be None for non-returning ds_nand_b32 opcode")
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
            return f"ds_nand_rtn_b32 {self.dst.name}, {self.addr.name}, {self.data.name}{offset_part}"
        return f"ds_nand_b32 {self.addr.name}, {self.data.name}{offset_part}"


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_ast_construction_returning(self) -> None:
        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
            offset=16,
        )
        self.assertEqual(instr.opcode, DSOpcode.DS_NAND_RTN_B32)
        self.assertTrue(instr.opcode.is_return)
        self.assertIsNotNone(instr.dst)
        assert instr.dst is not None
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.addr.index, 1)
        self.assertEqual(instr.offset, 16)

    def test_instruction_ast_construction_non_returning(self) -> None:
        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_B32,
            dst=None,
            addr=VGPR(10),
            data=Immediate(0x1234),
        )
        self.assertEqual(instr.opcode, DSOpcode.DS_NAND_B32)
        self.assertFalse(instr.opcode.is_return)
        self.assertIsNone(instr.dst)
        self.assertEqual(instr.addr.index, 10)
        self.assertEqual(instr.offset, 0)

    def test_disassembly_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(6),
            data=Immediate(0x1234),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_nand_rtn_b32 v[5], v[6], 0x1234",
        )

        instr_off = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(12),
            addr=VGPR(13),
            data=SGPR(10),
            offset=64,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_nand_rtn_b32 v[12], v[13], s[10] offset:64",
        )

    def test_disassembly_non_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_B32,
            dst=None,
            addr=VGPR(2),
            data=VGPR(3),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_nand_b32 v[2], v[3]",
        )

        instr_off = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_B32,
            dst=None,
            addr=VGPR(2),
            data=VGPR(3),
            offset=128,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_nand_b32 v[2], v[3] offset:128",
        )

    def test_dst_type_enforcement_rejects_sgpr_and_none(self) -> None:
        with self.assertRaises(TypeError):
            DSNandInstruction(
                opcode=DSOpcode.DS_NAND_RTN_B32,
                dst=SGPR(0),  # type: ignore[arg-type]
                addr=VGPR(1),
                data=Immediate(0),
            )
        with self.assertRaises(TypeError):
            DSNandInstruction(
                opcode=DSOpcode.DS_NAND_RTN_B32,
                dst=None,
                addr=VGPR(1),
                data=Immediate(0),
            )

    def test_addr_type_enforcement_rejects_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            DSNandInstruction(
                opcode=DSOpcode.DS_NAND_RTN_B32,
                dst=VGPR(0),
                addr=SGPR(1),  # type: ignore[arg-type]
                data=Immediate(0),
            )

    def test_non_returning_rejects_non_none_dst(self) -> None:
        with self.assertRaises(ValueError):
            DSNandInstruction(
                opcode=DSOpcode.DS_NAND_B32,
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
            DSNandInstruction(
                opcode=DSOpcode.DS_NAND_RTN_B32,
                dst=VGPR(0),
                addr=VGPR(1),
                data=Immediate(0),
                offset=65536,
            )
        with self.assertRaises(ValueError):
            DSNandInstruction(
                opcode=DSOpcode.DS_NAND_RTN_B32,
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
    """Deterministic scalar interpreter for ds_nand_rtn_b32 / ds_nand_b32."""

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
        instr: DSNandInstruction,
        state: ScalarMachineState,
    ) -> Tuple[Optional[int], int]:
        """Execute a single ds_nand_rtn_b32 / ds_nand_b32 instruction deterministically.
        
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

        # 3. Compute bitwise NAND
        old_val, next_mem = oracle_ds_nand_b32(curr_mem, data_val)

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
        self.state.set_vgpr(2, 0x00FF00FF)
        self.state.lds.write_dword(addr, 0x0F0F0F0F)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        # old & data = 0x000F000F
        # ~0x000F000F = 0xFFF0FFF0
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x0F0F0F0F)
        self.assertEqual(mem, 0xFFF0FFF0)
        self.assertEqual(self.state.get_vgpr(0), 0x0F0F0F0F)
        self.assertEqual(self.state.lds.read_dword(addr), 0xFFF0FFF0)

    def test_scalar_step_returning_sgpr_operand(self) -> None:
        addr = 0x300
        self.state.set_vgpr(1, addr)
        self.state.set_sgpr(10, 0xFFFFFFFF)
        self.state.lds.write_dword(addr, 0x12345678)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(3),
            addr=VGPR(1),
            data=SGPR(10),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, (~0x12345678) & MASK_32)
        self.assertEqual(self.state.get_vgpr(3), 0x12345678)

    def test_scalar_step_returning_immediate_operand(self) -> None:
        addr = 0x400
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(addr, 0xAAAA5555)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(4),
            addr=VGPR(1),
            data=Immediate(0x00000000),
        )
        # Any value NAND 0 is 0xFFFFFFFF
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xAAAA5555)
        self.assertEqual(mem, 0xFFFFFFFF)
        self.assertEqual(self.state.get_vgpr(4), 0xAAAA5555)
        self.assertEqual(self.state.lds.read_dword(addr), 0xFFFFFFFF)

    def test_scalar_step_non_returning_opcode(self) -> None:
        addr = 0x500
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0xFF00FF00)
        self.state.lds.write_dword(addr, 0xFF00FF00)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_B32,
            dst=None,
            addr=VGPR(1),
            data=VGPR(2),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertIsNone(old)
        self.assertEqual(mem, 0x00FF00FF)
        self.assertEqual(self.state.lds.read_dword(addr), 0x00FF00FF)

    def test_scalar_step_with_immediate_offset(self) -> None:
        base_addr = 0x600
        offset = 0x08
        eff_addr = base_addr + offset
        self.state.set_vgpr(2, base_addr)
        self.state.lds.write_dword(eff_addr, 0xF0F0F0F0)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(2),
            data=Immediate(0xFFFFFFFF),
            offset=offset,
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xF0F0F0F0)
        self.assertEqual(mem, 0x0F0F0F0F)
        self.assertEqual(self.state.get_vgpr(5), 0xF0F0F0F0)
        self.assertEqual(self.state.lds.read_dword(eff_addr), 0x0F0F0F0F)
        self.assertEqual(self.state.lds.read_dword(base_addr), 0)

    def test_scalar_step_nand_double_inversion_cycle(self) -> None:
        """Two sequential NAND with 0xFFFFFFFF inverts and then restores original bits."""
        addr = 0x700
        initial_val = 0x1234ABCD
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(addr, initial_val)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(6),
            addr=VGPR(1),
            data=Immediate(0xFFFFFFFF),
        )
        # Step 1: invert
        old1, mem1 = self.interp.step(instr, self.state)
        self.assertEqual(old1, initial_val)
        self.assertEqual(mem1, (~initial_val) & MASK_32)

        # Step 2: invert again
        old2, mem2 = self.interp.step(instr, self.state)
        self.assertEqual(old2, (~initial_val) & MASK_32)
        self.assertEqual(mem2, initial_val)

    def test_scalar_step_out_of_bounds_address_wrapping(self) -> None:
        cap_bytes = 65536
        addr = cap_bytes + 0x100
        eff_wrapped = 0x100
        self.state.set_vgpr(1, addr)
        self.state.lds.write_dword(eff_wrapped, 0x11110000)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(8),
            addr=VGPR(1),
            data=Immediate(0x00000000),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x11110000)
        self.assertEqual(mem, 0xFFFFFFFF)
        self.assertEqual(self.state.lds.read_dword(eff_wrapped), 0xFFFFFFFF)


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
        instr: DSNandInstruction,
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

            # 3. Read current value and perform atomic bitwise NAND
            curr_mem = state.lds.read_dword(eff_addr)
            old_val, next_mem = oracle_ds_nand_b32(curr_mem, data_val)
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
            state.set_vgpr(lane, 2, 0xFFFFFFFF)
            state.lds.write_dword(addr, 1 << lane)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)

        for lane in range(32):
            addr = 0x200 + lane * 4
            expected_old = 1 << lane
            expected_mem = (~(1 << lane)) & MASK_32
            self.assertEqual(state.get_vgpr(lane, 0), expected_old)
            self.assertEqual(state.lds.read_dword(addr), expected_mem)

    def test_wave64_uniform_full_lds_spread(self) -> None:
        """64 lanes active in Wave64 mode across LDS banks."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x100 + lane * 4
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0)
            state.lds.write_dword(addr, lane)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)

        for lane in range(64):
            addr = 0x100 + lane * 4
            self.assertEqual(state.get_vgpr(lane, 5), lane)
            self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)

    def test_wave32_exec_mask_alternating_lanes(self) -> None:
        """Even lanes active (0x55555555), odd lanes inactive."""
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        interp = SIMDInterpreter()

        sentinel = 0xDEAD_BEEF
        for lane in range(32):
            addr = 0x400 + lane * 4
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xFF)
            state.lds.write_dword(addr, 0xFF)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 16)

        for lane in range(32):
            addr = 0x400 + lane * 4
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr(lane, 0), 0xFF)
                self.assertEqual(state.lds.read_dword(addr), 0xFFFFFF00)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(addr), 0xFF)

    def test_wave32_exec_mask_upper_half_disabled(self) -> None:
        """Lower 16 lanes active, upper 16 inactive."""
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        interp = SIMDInterpreter()
        sentinel = 0xCAFE_0000

        for lane in range(32):
            addr = 0x500 + lane * 4
            state.set_vgpr(lane, 3, sentinel)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xAAAAAAAA)
            state.lds.write_dword(addr, 0xAAAAAAAA)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(3),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 16)

        for lane in range(16):
            self.assertEqual(state.get_vgpr(lane, 3), 0xAAAAAAAA)
            self.assertEqual(state.lds.read_dword(0x500 + lane * 4), 0x55555555)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr(lane, 3), sentinel)
            self.assertEqual(state.lds.read_dword(0x500 + lane * 4), 0xAAAAAAAA)

    def test_wave32_exec_mask_single_lane(self) -> None:
        """Only lane 17 active."""
        target_lane = 17
        state = SIMDState(num_lanes=32, exec_mask=(1 << target_lane))
        interp = SIMDInterpreter()

        addr = 0x600
        state.set_vgpr(target_lane, 1, addr)
        state.set_vgpr(target_lane, 2, 0x01)
        state.lds.write_dword(addr, 0x01)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], target_lane)
        self.assertEqual(state.get_vgpr(target_lane, 0), 0x01)
        self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFE)

    def test_wave32_exec_mask_all_zero(self) -> None:
        """All lanes disabled (exec_mask = 0)."""
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        interp = SIMDInterpreter()

        addr = 0x700
        state.lds.write_dword(addr, 0x42)
        for lane in range(32):
            state.set_vgpr(lane, 0, 0x9999)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0xFFFF)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
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
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0003)  # Lanes 0 and 1 active
        interp = SIMDInterpreter()
        common_addr = 0x800
        state.lds.write_dword(common_addr, 0xFFFF_FFFF)

        # Lane 0: NAND with 0x000000FF -> old = 0xFFFFFFFF, mem = 0xFFFFFF00
        # Lane 1: NAND with 0x0000FF00 -> old = 0xFFFFFF00, mem = ~(0xFFFFFF00 & 0x0000FF00) = ~0x0000FF00 = 0xFFFF00FF
        state.set_vgpr(0, 1, common_addr)
        state.set_vgpr(0, 2, 0x0000_00FF)
        state.set_vgpr(1, 1, common_addr)
        state.set_vgpr(1, 2, 0x0000_FF00)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 2)

        self.assertEqual(state.get_vgpr(0, 0), 0xFFFF_FFFF)
        self.assertEqual(state.get_vgpr(1, 0), 0xFFFF_FF00)
        self.assertEqual(state.lds.read_dword(common_addr), 0xFFFF_00FF)

    def test_wave64_exec_mask_64bit_full_coverage(self) -> None:
        """64-bit mask with odd lanes active (0xAAAAAAAAAAAAAAAA)."""
        mask64 = 0xAAAA_AAAA_AAAA_AAAA
        state = SIMDState(num_lanes=64, exec_mask=mask64)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x900 + lane * 4
            state.set_vgpr(lane, 4, 0)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, 0)
            state.lds.write_dword(addr, 0x1234)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(4),
            addr=VGPR(1),
            data=VGPR(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)

        for lane in range(64):
            addr = 0x900 + lane * 4
            if (mask64 >> lane) & 1:
                self.assertEqual(state.get_vgpr(lane, 4), 0x1234)
                self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)
            else:
                self.assertEqual(state.get_vgpr(lane, 4), 0)
                self.assertEqual(state.lds.read_dword(addr), 0x1234)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_alias_dst_equals_addr(self) -> None:
        """Destination VGPR aliases Address VGPR: v[0] = ds_nand_rtn_b32 v[0], v[1].
        
        Address must be latched in the operand stage before v[0] is overwritten with old_val.
        """
        state = ScalarMachineState()
        addr = 0x500
        state.set_vgpr(0, addr)        # v[0] is address AND destination
        state.set_vgpr(1, 0xFFFFFFFF)  # v[1] is data operand
        state.lds.write_dword(addr, 0x12345678)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(0),
            data=VGPR(1),
        )
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, (~0x12345678) & MASK_32)
        # v[0] now contains the returned old_val
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        # Memory at original addr was properly updated
        self.assertEqual(state.lds.read_dword(addr), (~0x12345678) & MASK_32)

    def test_alias_dst_equals_data(self) -> None:
        """Destination VGPR aliases Data VGPR: v[1] = ds_nand_rtn_b32 v[0], v[1].
        
        Data must be latched before v[1] is overwritten with old_val.
        """
        state = ScalarMachineState()
        addr = 0x600
        state.set_vgpr(0, addr)
        state.set_vgpr(1, 0x0F0F0F0F)  # v[1] is data AND destination
        state.lds.write_dword(addr, 0xF0F0F0F0)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(1),
            addr=VGPR(0),
            data=VGPR(1),
        )
        # 0xF0F0F0F0 & 0x0F0F0F0F = 0 -> ~0 = 0xFFFFFFFF
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0xF0F0F0F0)
        self.assertEqual(mem, 0xFFFFFFFF)
        self.assertEqual(state.get_vgpr(1), 0xF0F0F0F0)
        self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)

    def test_alias_addr_equals_data(self) -> None:
        """Address VGPR aliases Data VGPR: v[2] = ds_nand_rtn_b32 v[0], v[0]."""
        state = ScalarMachineState()
        addr_and_data = 0x200
        state.set_vgpr(0, addr_and_data)
        state.lds.write_dword(addr_and_data, 0x200)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(2),
            addr=VGPR(0),
            data=VGPR(0),
        )
        # 0x200 NAND 0x200 = ~0x200
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0x200)
        self.assertEqual(mem, (~0x200) & MASK_32)
        self.assertEqual(state.get_vgpr(2), 0x200)
        self.assertEqual(state.lds.read_dword(addr_and_data), (~0x200) & MASK_32)
        self.assertEqual(state.get_vgpr(0), addr_and_data)

    def test_full_operand_aliasing_three_way(self) -> None:
        """Destination, Address, and Data all share the same register: v[5] = ds_nand_rtn_b32 v[5], v[5]."""
        state = ScalarMachineState()
        val = 0x400
        state.set_vgpr(5, val)
        state.lds.write_dword(val, 0)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(5),
            addr=VGPR(5),
            data=VGPR(5),
        )
        # 0 & 0x400 = 0 -> ~0 = 0xFFFFFFFF
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0)
        self.assertEqual(mem, 0xFFFFFFFF)
        self.assertEqual(state.get_vgpr(5), 0)
        self.assertEqual(state.lds.read_dword(val), 0xFFFFFFFF)

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

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(9),
            addr=VGPR(10),
            data=VGPR(11),
        )
        self.scalar_interp.step(instr, state)

        self.assertEqual(state.get_vgpr(9), 0xAA)
        self.assertEqual(state.get_vgpr(10), addr)
        self.assertEqual(state.get_vgpr(11), 0x55)

        for r in range(256):
            if r not in (9, 10, 11):
                self.assertEqual(state.get_vgpr(r), canary_base | r)

    def test_simd_multi_lane_aliasing(self) -> None:
        """All 32 lanes alias destination with address: v[0] = ds_nand_rtn_b32 v[0], v[1]."""
        state = SIMDState(num_lanes=32)
        for lane in range(32):
            addr = 0x300 + lane * 4
            state.set_vgpr(lane, 0, addr)
            state.set_vgpr(lane, 1, 0)
            state.lds.write_dword(addr, 0x1000 + lane)

        instr = DSNandInstruction(
            opcode=DSOpcode.DS_NAND_RTN_B32,
            dst=VGPR(0),
            addr=VGPR(0),
            data=VGPR(1),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            addr = 0x300 + lane * 4
            self.assertEqual(state.get_vgpr(lane, 0), 0x1000 + lane)
            self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
