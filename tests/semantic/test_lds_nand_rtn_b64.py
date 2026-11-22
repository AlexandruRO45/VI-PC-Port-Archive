"""Semantic reference oracle and architectural model for ds_nand_rtn_b64.

RDNA2 / GFX10 Local Data Share (LDS) 64-Bit Bitwise Atomic NAND with Return:
Performs a 64-bit atomic bitwise NAND operation on LDS memory and returns
the previous memory value to the destination VGPR pair v[dst:dst+1].

Syntax:
    ds_nand_rtn_b64 v[dst:dst+1], vaddr, v[data:data+1] [offset:offset16]
    ds_nand_b64 vaddr, v[data:data+1] [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_capacity_qwords - 1)
    old_val = LDS[qword_index]
    new_val = ~(old_val & vdata) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_val
    v[dst:dst+1] = old_val
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
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.
    
    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.
    
    64-bit LDS accesses are 8-byte aligned; the low 3 bits are ignored by hardware.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_nand_b64(current_mem: int, nand_val: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nand_rtn_b64.
    
    Returns:
        (returned_old_val, new_mem_val)
    """
    old_val = current_mem & MASK_64
    vdata = nand_val & MASK_64
    new_mem = (~(old_val & vdata)) & MASK_64
    return old_val, new_mem


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.capacity_dwords = capacity_qwords * 2
        self.memory = [0] * self.capacity_dwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        low = self.memory[dword_idx]
        high = self.memory[dword_idx + 1]
        return ((high << 32) | low) & MASK_64

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        self.memory[dword_idx] = val & MASK_32
        self.memory[dword_idx + 1] = (val >> 32) & MASK_32

    def atomic_nand_b64(
        self,
        vaddr: int,
        nand_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic bitwise NAND on LDS 64-bit qword, returning (old_val, new_val)."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem = oracle_ds_nand_b64(curr, nand_val)
        self.write_qword(eff_addr, new_mem)
        return old_val, new_mem


class TestLdsNandRtnB64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_nand_with_zero(self) -> None:
        """NAND with 0 always sets all 64 bits (~0 == 0xFFFFFFFFFFFFFFFF)."""
        old, mem = oracle_ds_nand_b64(0x12345678_9ABCDEF0, 0x00000000_00000000)
        self.assertEqual(old, 0x12345678_9ABCDEF0)
        self.assertEqual(mem, MASK_64)

    def test_oracle_nand_with_all_ones_inversion(self) -> None:
        """NAND with MASK_64 computes 64-bit bitwise NOT (~x)."""
        val = 0xAA55AA55_55AAAA55
        old, mem = oracle_ds_nand_b64(val, MASK_64)
        self.assertEqual(old, val)
        self.assertEqual(mem, 0x55AA55AA_AA5555AA)

    def test_oracle_self_nand_inversion(self) -> None:
        """x NAND x computes ~x across 64 bits."""
        val = 0x12345678_9ABCDEF0
        old, mem = oracle_ds_nand_b64(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, (~val) & MASK_64)

    def test_oracle_disjoint_bitmasks(self) -> None:
        """NAND of disjoint 64-bit masks produces MASK_64."""
        mask_a = 0xFFFF_0000_0000_0000
        mask_b = 0x0000_0000_FFFF_0000
        old, mem = oracle_ds_nand_b64(mask_a, mask_b)
        self.assertEqual(old, mask_a)
        self.assertEqual(mem, MASK_64)

    def test_oracle_overlapping_bitmasks(self) -> None:
        """NAND of overlapping 64-bit masks inverts only common bits."""
        val_a = 0x0000_0000_FFFF_FFFF
        val_b = 0x0000_FFFF_FFFF_0000
        # val_a & val_b = 0x0000_0000_FFFF_0000
        old, mem = oracle_ds_nand_b64(val_a, val_b)
        self.assertEqual(old, val_a)
        expected = (~0x0000_0000_FFFF_0000) & MASK_64
        self.assertEqual(mem, expected)

    def test_effective_address_calculation_with_offset(self) -> None:
        """Effective address correctly adds base and 16-bit offset."""
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)
        self.assertEqual(calc_lds_qword_index(eff), 0x1040 >> 3)

    def test_effective_address_wrapping_modulo_2_32(self) -> None:
        """Effective address wraps cleanly at 2^32 boundary."""
        eff = calc_lds_effective_address(0xFFFF_FFF8, 0x0010)
        self.assertEqual(eff, 0x0000_0008)

    def test_qword_alignment_drops_low_3_bits(self) -> None:
        """Hardware drops low 3 bits; byte offsets 0..7 all map to same qword."""
        idx_base = calc_lds_qword_index(0x400)
        for byte_off in range(8):
            self.assertEqual(calc_lds_qword_index(0x400 + byte_off), idx_base)
        self.assertEqual(idx_base, 0x400 >> 3)

    def test_lds_capacity_wrapping(self) -> None:
        """Addresses past capacity wrap modulo capacity_qwords."""
        cap_bytes = 65536
        idx_low = calc_lds_qword_index(0x100)
        idx_wrapped = calc_lds_qword_index(0x100 + cap_bytes)
        self.assertEqual(idx_low, idx_wrapped)

    def test_model_atomic_nand_returns_pre_op_value(self) -> None:
        """LDSModel atomic_nand_b64 updates memory and returns the pre-operation value."""
        lds = LDSModel()
        addr = 0x200
        lds.write_qword(addr, 0x00FF00FF_00FF00FF)

        old_val, new_val = lds.atomic_nand_b64(addr, 0x0000FFFF_0000FFFF)
        # old & data = 0x000000FF_000000FF
        # new = ~0x000000FF_000000FF = 0xFFFFFF00_FFFFFF00
        self.assertEqual(old_val, 0x00FF00FF_00FF00FF)
        self.assertEqual(new_val, 0xFFFFFF00_FFFFFF00)
        self.assertEqual(lds.read_qword(addr), 0xFFFFFF00_FFFFFF00)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

@dataclass(frozen=True)
class VGPR:
    """32-bit Vector General Purpose Register (e.g. v[0] .. v[255])."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of architectural range [0, 255]")

    @property
    def name(self) -> str:
        return f"v[{self.index}]"


@dataclass(frozen=True)
class VGPRPair:
    """64-bit aligned VGPR pair operand (e.g. v[0:1] .. v[254:255])."""
    base_index: int

    def __post_init__(self) -> None:
        if not (0 <= self.base_index <= 254):
            raise ValueError(f"VGPRPair base_index {self.base_index} out of range [0, 254]")

    @property
    def name(self) -> str:
        return f"v[{self.base_index}:{self.base_index + 1}]"


@dataclass(frozen=True)
class SGPRPair:
    """64-bit aligned SGPR pair operand (e.g. s[0:1] .. s[104:105])."""
    base_index: int

    def __post_init__(self) -> None:
        if not (0 <= self.base_index <= 104):
            raise ValueError(f"SGPRPair base_index {self.base_index} out of range [0, 104]")

    @property
    def name(self) -> str:
        return f"s[{self.base_index}:{self.base_index + 1}]"


@dataclass(frozen=True)
class Immediate64:
    """64-bit integer immediate constant."""
    value: int

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK_64:X}"


Operand64 = Union[VGPRPair, SGPRPair, Immediate64]


class DSOpcode64(Enum):
    """LDS 64-bit Bitwise NAND opcodes (returning and non-returning variants)."""
    DS_NAND_RTN_B64 = auto()
    DS_NAND_B64 = auto()

    @property
    def is_return(self) -> bool:
        return self == DSOpcode64.DS_NAND_RTN_B64

    @property
    def op_name(self) -> str:
        return "ds_nand_rtn_b64" if self.is_return else "ds_nand_b64"


@dataclass(frozen=True)
class DSNand64Instruction:
    """Symbolic AST node for ds_nand_rtn_b64 / ds_nand_b64 instruction."""
    opcode: DSOpcode64
    dst: Optional[VGPRPair]
    addr: VGPR
    data: Operand64
    offset: int = 0

    def __post_init__(self) -> None:
        if self.opcode.is_return:
            if not isinstance(self.dst, VGPRPair):
                raise TypeError(f"Destination must be a VGPRPair for returning opcode, got {type(self.dst).__name__}")
        else:
            if self.dst is not None:
                raise ValueError("Destination must be None for non-returning ds_nand_b64 opcode")
        if not isinstance(self.addr, VGPR):
            raise TypeError(f"Address operand must be a 32-bit VGPR, got {type(self.addr).__name__}")
        if not isinstance(self.data, (VGPRPair, SGPRPair, Immediate64)):
            raise TypeError(f"Data operand must be VGPRPair, SGPRPair, or Immediate64, got {type(self.data).__name__}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset {self.offset} out of 16-bit range [0, 65535]")

    def disassemble(self) -> str:
        offset_part = f" offset:{self.offset}" if self.offset != 0 else ""
        if self.opcode.is_return:
            assert self.dst is not None
            return f"ds_nand_rtn_b64 {self.dst.name}, {self.addr.name}, {self.data.name}{offset_part}"
        return f"ds_nand_b64 {self.addr.name}, {self.data.name}{offset_part}"


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_ast_construction_returning(self) -> None:
        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(2),
            data=VGPRPair(4),
            offset=16,
        )
        self.assertEqual(instr.opcode, DSOpcode64.DS_NAND_RTN_B64)
        self.assertTrue(instr.opcode.is_return)
        self.assertIsNotNone(instr.dst)
        assert instr.dst is not None
        self.assertEqual(instr.dst.base_index, 0)
        self.assertEqual(instr.addr.index, 2)
        self.assertEqual(instr.offset, 16)

    def test_instruction_ast_construction_non_returning(self) -> None:
        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_B64,
            dst=None,
            addr=VGPR(10),
            data=Immediate64(0x1234_5678_9ABC_DEF0),
        )
        self.assertEqual(instr.opcode, DSOpcode64.DS_NAND_B64)
        self.assertFalse(instr.opcode.is_return)
        self.assertIsNone(instr.dst)
        self.assertEqual(instr.addr.index, 10)
        self.assertEqual(instr.offset, 0)

    def test_disassembly_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(6),
            addr=VGPR(5),
            data=Immediate64(0x1234),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_nand_rtn_b64 v[6:7], v[5], 0x1234",
        )

        instr_off = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(12),
            addr=VGPR(14),
            data=SGPRPair(10),
            offset=64,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_nand_rtn_b64 v[12:13], v[14], s[10:11] offset:64",
        )

    def test_disassembly_non_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_B64,
            dst=None,
            addr=VGPR(2),
            data=VGPRPair(4),
        )
        self.assertEqual(
            instr_no_off.disassemble(),
            "ds_nand_b64 v[2], v[4:5]",
        )

        instr_off = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_B64,
            dst=None,
            addr=VGPR(2),
            data=VGPRPair(4),
            offset=128,
        )
        self.assertEqual(
            instr_off.disassemble(),
            "ds_nand_b64 v[2], v[4:5] offset:128",
        )

    def test_dst_type_enforcement_rejects_single_vgpr_and_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=SGPRPair(0),  # type: ignore[arg-type]
                addr=VGPR(1),
                data=Immediate64(0),
            )
        with self.assertRaises(TypeError):
            DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=None,
                addr=VGPR(1),
                data=Immediate64(0),
            )

    def test_addr_type_enforcement_rejects_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=VGPRPair(0),
                addr=SGPRPair(0),  # type: ignore[arg-type]
                data=Immediate64(0),
            )

    def test_non_returning_rejects_non_none_dst(self) -> None:
        with self.assertRaises(ValueError):
            DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_B64,
                dst=VGPRPair(0),
                addr=VGPR(1),
                data=Immediate64(0),
            )

    def test_register_index_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPRPair(255)  # 255:256 exceeds 255
        with self.assertRaises(ValueError):
            VGPRPair(-1)
        with self.assertRaises(ValueError):
            SGPRPair(105)  # 105:106 exceeds 105
        with self.assertRaises(ValueError):
            SGPRPair(-1)

    def test_offset_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=VGPRPair(0),
                addr=VGPR(1),
                data=Immediate64(0),
                offset=65536,
            )
        with self.assertRaises(ValueError):
            DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=VGPRPair(0),
                addr=VGPR(1),
                data=Immediate64(0),
                offset=-1,
            )

    def test_operand_equality_and_hashing(self) -> None:
        v0_a = VGPRPair(0)
        v0_b = VGPRPair(0)
        v2 = VGPRPair(2)
        self.assertEqual(v0_a, v0_b)
        self.assertNotEqual(v0_a, v2)
        imm1 = Immediate64(0x1234_5678)
        imm2 = Immediate64(0x1234_5678)
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

    def get_vgpr_pair(self, base_index: int) -> int:
        low = self.vgprs[base_index] & MASK_32
        high = self.vgprs[base_index + 1] & MASK_32
        return ((high << 32) | low) & MASK_64

    def set_vgpr_pair(self, base_index: int, val: int) -> None:
        self.vgprs[base_index] = val & MASK_32
        self.vgprs[base_index + 1] = (val >> 32) & MASK_32

    def get_sgpr_pair(self, base_index: int) -> int:
        low = self.sgprs[base_index] & MASK_32
        high = self.sgprs[base_index + 1] & MASK_32
        return ((high << 32) | low) & MASK_64

    def set_sgpr_pair(self, base_index: int, val: int) -> None:
        self.sgprs[base_index] = val & MASK_32
        self.sgprs[base_index + 1] = (val >> 32) & MASK_32


class ScalarInterpreter:
    """Deterministic scalar interpreter for ds_nand_rtn_b64 / ds_nand_b64."""

    def resolve_operand(self, op: Operand64, state: ScalarMachineState) -> int:
        if isinstance(op, VGPRPair):
            return state.get_vgpr_pair(op.base_index)
        if isinstance(op, SGPRPair):
            return state.get_sgpr_pair(op.base_index)
        if isinstance(op, Immediate64):
            return op.value & MASK_64
        raise TypeError(f"Unknown operand type: {type(op).__name__}")

    def step(
        self,
        instr: DSNand64Instruction,
        state: ScalarMachineState,
    ) -> Tuple[Optional[int], int]:
        """Execute a single ds_nand_rtn_b64 / ds_nand_b64 instruction deterministically.
        
        All operands are latched before write-back to ensure operand preservation.
        Returns:
            (returned_old_val if returning else None, updated_mem_val)
        """
        # 1. Latch inputs
        vaddr = state.get_vgpr(instr.addr.index)
        data_val = self.resolve_operand(instr.data, state)

        # 2. Compute effective byte address and fetch current LDS 64-bit value
        eff_addr = calc_lds_effective_address(vaddr, instr.offset)
        curr_mem = state.lds.read_qword(eff_addr)

        # 3. Compute 64-bit bitwise NAND
        old_val, next_mem = oracle_ds_nand_b64(curr_mem, data_val)

        # 4. Atomic memory writeback
        state.lds.write_qword(eff_addr, next_mem)

        # 5. Write return value to destination VGPR pair if returning variant
        if instr.opcode.is_return:
            assert instr.dst is not None
            state.set_vgpr_pair(instr.dst.base_index, old_val)
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
        self.state.set_vgpr_pair(4, 0x00FF00FF_00FF00FF)
        self.state.lds.write_qword(addr, 0x0F0F0F0F_0F0F0F0F)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        # old & data = 0x000F000F_000F000F
        # ~0x000F000F_000F000F = 0xFFF0FFF0_FFF0FFF0
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(mem, 0xFFF0FFF0_FFF0FFF0)
        self.assertEqual(self.state.get_vgpr_pair(2), 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(self.state.lds.read_qword(addr), 0xFFF0FFF0_FFF0FFF0)

    def test_scalar_step_returning_sgpr_operand(self) -> None:
        addr = 0x300
        self.state.set_vgpr(1, addr)
        self.state.set_sgpr_pair(10, MASK_64)
        self.state.lds.write_qword(addr, 0x12345678_9ABCDEF0)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=SGPRPair(10),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x12345678_9ABCDEF0)
        self.assertEqual(mem, (~0x12345678_9ABCDEF0) & MASK_64)
        self.assertEqual(self.state.get_vgpr_pair(2), 0x12345678_9ABCDEF0)

    def test_scalar_step_returning_immediate_operand(self) -> None:
        addr = 0x400
        self.state.set_vgpr(1, addr)
        self.state.lds.write_qword(addr, 0xAAAA5555_AAAA5555)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(4),
            addr=VGPR(1),
            data=Immediate64(0),
        )
        # Any value NAND 0 is MASK_64
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xAAAA5555_AAAA5555)
        self.assertEqual(mem, MASK_64)
        self.assertEqual(self.state.get_vgpr_pair(4), 0xAAAA5555_AAAA5555)
        self.assertEqual(self.state.lds.read_qword(addr), MASK_64)

    def test_scalar_step_non_returning_opcode(self) -> None:
        addr = 0x500
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr_pair(2, 0xFF00FF00_FF00FF00)
        self.state.lds.write_qword(addr, 0xFF00FF00_FF00FF00)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_B64,
            dst=None,
            addr=VGPR(1),
            data=VGPRPair(2),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertIsNone(old)
        self.assertEqual(mem, 0x00FF00FF_00FF00FF)
        self.assertEqual(self.state.lds.read_qword(addr), 0x00FF00FF_00FF00FF)

    def test_scalar_step_with_immediate_offset(self) -> None:
        base_addr = 0x600
        offset = 0x10
        eff_addr = base_addr + offset
        self.state.set_vgpr(2, base_addr)
        self.state.lds.write_qword(eff_addr, 0xF0F0F0F0_F0F0F0F0)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(6),
            addr=VGPR(2),
            data=Immediate64(MASK_64),
            offset=offset,
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xF0F0F0F0_F0F0F0F0)
        self.assertEqual(mem, 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(self.state.get_vgpr_pair(6), 0xF0F0F0F0_F0F0F0F0)
        self.assertEqual(self.state.lds.read_qword(eff_addr), 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(self.state.lds.read_qword(base_addr), 0)

    def test_scalar_step_nand_double_inversion_cycle(self) -> None:
        addr = 0x700
        initial_val = 0x1234ABCD_5678EF01
        self.state.set_vgpr(1, addr)
        self.state.lds.write_qword(addr, initial_val)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(8),
            addr=VGPR(1),
            data=Immediate64(MASK_64),
        )
        old1, mem1 = self.interp.step(instr, self.state)
        self.assertEqual(old1, initial_val)
        self.assertEqual(mem1, (~initial_val) & MASK_64)

        old2, mem2 = self.interp.step(instr, self.state)
        self.assertEqual(old2, (~initial_val) & MASK_64)
        self.assertEqual(mem2, initial_val)

    def test_scalar_step_out_of_bounds_address_wrapping(self) -> None:
        cap_bytes = 65536
        addr = cap_bytes + 0x100
        eff_wrapped = 0x100
        self.state.set_vgpr(1, addr)
        self.state.lds.write_qword(eff_wrapped, 0x11110000_22220000)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(10),
            addr=VGPR(1),
            data=Immediate64(0),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x11110000_22220000)
        self.assertEqual(mem, MASK_64)
        self.assertEqual(self.state.lds.read_qword(eff_wrapped), MASK_64)


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

    def get_vgpr_pair(self, lane: int, base_index: int) -> int:
        low = self.vgprs[lane][base_index] & MASK_32
        high = self.vgprs[lane][base_index + 1] & MASK_32
        return ((high << 32) | low) & MASK_64

    def set_vgpr_pair(self, lane: int, base_index: int, val: int) -> None:
        self.vgprs[lane][base_index] = val & MASK_32
        self.vgprs[lane][base_index + 1] = (val >> 32) & MASK_32

    def get_sgpr_pair(self, base_index: int) -> int:
        low = self.sgprs[base_index] & MASK_32
        high = self.sgprs[base_index + 1] & MASK_32
        return ((high << 32) | low) & MASK_64

    def set_sgpr_pair(self, base_index: int, val: int) -> None:
        self.sgprs[base_index] = val & MASK_32
        self.sgprs[base_index + 1] = (val >> 32) & MASK_32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and atomic serialization."""

    def resolve_lane_operand(self, op: Operand64, lane: int, state: SIMDState) -> int:
        if isinstance(op, VGPRPair):
            return state.get_vgpr_pair(lane, op.base_index)
        if isinstance(op, SGPRPair):
            return state.get_sgpr_pair(op.base_index)
        if isinstance(op, Immediate64):
            return op.value & MASK_64
        raise TypeError(f"Unknown operand type: {type(op).__name__}")

    def step(
        self,
        instr: DSNand64Instruction,
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

            # 1. Latch lane inputs (vaddr is 32-bit VGPR, data is 64-bit)
            vaddr = state.get_vgpr(lane, instr.addr.index)
            data_val = self.resolve_lane_operand(instr.data, lane, state)

            # 2. Compute effective byte address
            eff_addr = calc_lds_effective_address(vaddr, instr.offset)

            # 3. Read current value and perform atomic bitwise NAND
            curr_mem = state.lds.read_qword(eff_addr)
            old_val, next_mem = oracle_ds_nand_b64(curr_mem, data_val)
            state.lds.write_qword(eff_addr, next_mem)

            # 4. If returning variant, write to lane's destination VGPR pair
            if instr.opcode.is_return:
                assert instr.dst is not None
                state.set_vgpr_pair(lane, instr.dst.base_index, old_val)
                results.append((lane, old_val, next_mem))
            else:
                results.append((lane, None, next_mem))

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_independent_addresses(self) -> None:
        """All 32 lanes active with independent 64-bit addresses."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(32):
            addr = 0x200 + lane * 8  # 8 bytes per qword
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr_pair(lane, 4, MASK_64)
            state.lds.write_qword(addr, 1 << lane)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)

        for lane in range(32):
            addr = 0x200 + lane * 8
            expected_old = 1 << lane
            expected_mem = (~(1 << lane)) & MASK_64
            self.assertEqual(state.get_vgpr_pair(lane, 2), expected_old)
            self.assertEqual(state.lds.read_qword(addr), expected_mem)

    def test_wave64_uniform_full_lds_spread(self) -> None:
        """64 lanes active in Wave64 mode across LDS banks."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x100 + lane * 8
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr_pair(lane, 4, 0)
            state.lds.write_qword(addr, lane)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(6),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)

        for lane in range(64):
            addr = 0x100 + lane * 8
            self.assertEqual(state.get_vgpr_pair(lane, 6), lane)
            self.assertEqual(state.lds.read_qword(addr), MASK_64)

    def test_wave32_exec_mask_alternating_lanes(self) -> None:
        """Even lanes active (0x55555555), odd lanes inactive."""
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        interp = SIMDInterpreter()

        sentinel = 0xDEAD_BEEF_CAFE_0000
        for lane in range(32):
            addr = 0x400 + lane * 8
            state.set_vgpr_pair(lane, 0, sentinel)
            state.set_vgpr(lane, 2, addr)
            state.set_vgpr_pair(lane, 4, 0xFF)
            state.lds.write_qword(addr, 0xFF)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(2),
            data=VGPRPair(4),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 16)

        for lane in range(32):
            addr = 0x400 + lane * 8
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr_pair(lane, 0), 0xFF)
                self.assertEqual(state.lds.read_qword(addr), (~0xFF) & MASK_64)
            else:
                self.assertEqual(state.get_vgpr_pair(lane, 0), sentinel)
                self.assertEqual(state.lds.read_qword(addr), 0xFF)

    def test_wave32_exec_mask_upper_half_disabled(self) -> None:
        """Lower 16 lanes active, upper 16 inactive."""
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        interp = SIMDInterpreter()
        sentinel = 0xCAFE_0000_1234_5678

        for lane in range(32):
            addr = 0x500 + lane * 8
            state.set_vgpr_pair(lane, 2, sentinel)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr_pair(lane, 4, 0xAAAAAAAAAAAAAAAA)
            state.lds.write_qword(addr, 0xAAAAAAAAAAAAAAAA)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 16)

        for lane in range(16):
            self.assertEqual(state.get_vgpr_pair(lane, 2), 0xAAAAAAAAAAAAAAAA)
            self.assertEqual(state.lds.read_qword(0x500 + lane * 8), 0x5555555555555555)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr_pair(lane, 2), sentinel)
            self.assertEqual(state.lds.read_qword(0x500 + lane * 8), 0xAAAAAAAAAAAAAAAA)

    def test_wave32_exec_mask_single_lane(self) -> None:
        """Only lane 17 active."""
        target_lane = 17
        state = SIMDState(num_lanes=32, exec_mask=(1 << target_lane))
        interp = SIMDInterpreter()

        addr = 0x600
        state.set_vgpr(target_lane, 1, addr)
        state.set_vgpr_pair(target_lane, 2, 0x01)
        state.lds.write_qword(addr, 0x01)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(1),
            data=VGPRPair(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], target_lane)
        self.assertEqual(state.get_vgpr_pair(target_lane, 0), 0x01)
        self.assertEqual(state.lds.read_qword(addr), MASK_64 ^ 0x01)

    def test_wave32_exec_mask_all_zero(self) -> None:
        """All lanes disabled (exec_mask = 0)."""
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        interp = SIMDInterpreter()

        addr = 0x700
        state.lds.write_qword(addr, 0x42)
        for lane in range(32):
            state.set_vgpr_pair(lane, 0, 0x9999)
            state.set_vgpr(lane, 2, addr)
            state.set_vgpr_pair(lane, 4, 0xFFFF)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(2),
            data=VGPRPair(4),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 0)
        self.assertEqual(state.lds.read_qword(addr), 0x42)
        for lane in range(32):
            self.assertEqual(state.get_vgpr_pair(lane, 0), 0x9999)

    def test_wave32_same_address_collision_atomic_serialization(self) -> None:
        """Multiple active lanes target the exact same LDS 64-bit qword.
        
        Hardware serializes atomics in ascending order of lane index:
        Lane 0 updates LDS and gets old value, then Lane 1, etc.
        """
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0003)  # Lanes 0 and 1 active
        interp = SIMDInterpreter()
        common_addr = 0x800
        state.lds.write_qword(common_addr, MASK_64)

        # Lane 0: NAND with 0x000000FF_00000000 -> old = MASK_64, mem = ~0x000000FF_00000000
        # Lane 1: NAND with 0x0000FF00_00000000
        state.set_vgpr(0, 1, common_addr)
        state.set_vgpr_pair(0, 2, 0x0000_00FF_0000_0000)
        state.set_vgpr(1, 1, common_addr)
        state.set_vgpr_pair(1, 2, 0x0000_FF00_0000_0000)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(1),
            data=VGPRPair(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 2)

        self.assertEqual(state.get_vgpr_pair(0, 0), MASK_64)
        self.assertEqual(state.get_vgpr_pair(1, 0), (~0x0000_00FF_0000_0000) & MASK_64)
        expected_final = (~0x0000_FF00_0000_0000) & MASK_64
        self.assertEqual(state.lds.read_qword(common_addr), expected_final)

    def test_wave64_exec_mask_64bit_full_coverage(self) -> None:
        """64-bit mask with odd lanes active (0xAAAAAAAAAAAAAAAA)."""
        mask64 = 0xAAAA_AAAA_AAAA_AAAA
        state = SIMDState(num_lanes=64, exec_mask=mask64)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x900 + lane * 8
            state.set_vgpr_pair(lane, 4, 0)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr_pair(lane, 2, 0)
            state.lds.write_qword(addr, 0x1234_5678)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(4),
            addr=VGPR(1),
            data=VGPRPair(2),
        )
        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)

        for lane in range(64):
            addr = 0x900 + lane * 8
            if (mask64 >> lane) & 1:
                self.assertEqual(state.get_vgpr_pair(lane, 4), 0x1234_5678)
                self.assertEqual(state.lds.read_qword(addr), MASK_64)
            else:
                self.assertEqual(state.get_vgpr_pair(lane, 4), 0)
                self.assertEqual(state.lds.read_qword(addr), 0x1234_5678)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_alias_dst_overlaps_addr(self) -> None:
        """Destination VGPR pair v[0:1] overlaps Address VGPR v[0].
        
        Address must be latched in the operand stage before v[0:1] is overwritten with old_val.
        """
        state = ScalarMachineState()
        addr = 0x500
        state.set_vgpr(0, addr)        # v[0] holds address
        state.set_vgpr(1, 0x1234)      # v[1] holds dummy high word
        state.set_vgpr_pair(2, MASK_64) # v[2:3] holds data
        state.lds.write_qword(addr, 0xAAAA_AAAA_5555_5555)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(0),
            data=VGPRPair(2),
        )
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0xAAAA_AAAA_5555_5555)
        self.assertEqual(mem, 0x5555_5555_AAAA_AAAA)
        # v[0:1] now contains the returned old_val
        self.assertEqual(state.get_vgpr_pair(0), 0xAAAA_AAAA_5555_5555)
        # Memory at original addr was properly updated
        self.assertEqual(state.lds.read_qword(addr), 0x5555_5555_AAAA_AAAA)

    def test_alias_dst_equals_data(self) -> None:
        """Destination VGPR pair aliases Data VGPR pair: v[2:3] = ds_nand_rtn_b64 v[2:3], v[0], v[2:3].
        
        Data must be latched before v[2:3] is overwritten with old_val.
        """
        state = ScalarMachineState()
        addr = 0x600
        state.set_vgpr(0, addr)
        state.set_vgpr_pair(2, 0x0F0F0F0F_0F0F0F0F)  # v[2:3] is data AND destination
        state.lds.write_qword(addr, 0xF0F0F0F0_F0F0F0F0)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(0),
            data=VGPRPair(2),
        )
        # 0xF0F0F0F0_F0F0F0F0 & 0x0F0F0F0F_0F0F0F0F = 0 -> ~0 = MASK_64
        old, mem = self.scalar_interp.step(instr, state)
        self.assertEqual(old, 0xF0F0F0F0_F0F0F0F0)
        self.assertEqual(mem, MASK_64)
        self.assertEqual(state.get_vgpr_pair(2), 0xF0F0F0F0_F0F0F0F0)
        self.assertEqual(state.lds.read_qword(addr), MASK_64)

    def test_unrelated_vgprs_unmodified(self) -> None:
        """Canary test: ensure all 256 VGPRs outside destination pair remain untouched."""
        state = ScalarMachineState()
        canary_base = 0xDEAD0000
        for r in range(256):
            state.set_vgpr(r, canary_base | r)

        addr = 0x100
        state.set_vgpr(10, addr)
        state.set_vgpr_pair(12, 0x55)
        state.lds.write_qword(addr, 0xAA)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(8),
            addr=VGPR(10),
            data=VGPRPair(12),
        )
        self.scalar_interp.step(instr, state)

        # Destination v[8] and v[9] were updated
        self.assertEqual(state.get_vgpr(8), 0xAA)
        self.assertEqual(state.get_vgpr(9), 0)

        # Operands unchanged
        self.assertEqual(state.get_vgpr(10), addr)
        self.assertEqual(state.get_vgpr(12), 0x55)
        self.assertEqual(state.get_vgpr(13), 0)

        # Check all other 252 registers
        for r in range(256):
            if r not in (8, 9, 10, 12, 13):
                self.assertEqual(state.get_vgpr(r), canary_base | r)

    def test_simd_multi_lane_aliasing(self) -> None:
        """All 32 lanes alias destination base with address: v[0:1] = ds_nand_rtn_b64 v[0:1], v[0], v[2:3]."""
        state = SIMDState(num_lanes=32)
        for lane in range(32):
            addr = 0x300 + lane * 8
            state.set_vgpr(lane, 0, addr)
            state.set_vgpr(lane, 1, 0)
            state.set_vgpr_pair(lane, 2, 0)
            state.lds.write_qword(addr, 0x1000 + lane)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(0),
            data=VGPRPair(2),
        )
        self.simd_interp.step(instr, state)

        for lane in range(32):
            addr = 0x300 + lane * 8
            self.assertEqual(state.get_vgpr_pair(lane, 0), 0x1000 + lane)
            self.assertEqual(state.lds.read_qword(addr), MASK_64)


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AtomicBitwiseFlags64:
    """Architectural condition codes, status flags, and arithmetic analysis for ds_nand_rtn_b64."""
    zero: bool
    negative: bool
    old_zero: bool
    old_negative: bool
    parity_even: bool
    popcount: int
    changed: bool
    unaligned_addr: bool
    bank: int
    carry_or_sum: bool
    borrow: bool


def evaluate_ds_nand_b64_flags(
    old_val: int,
    nand_val: int,
    new_mem: int,
    eff_addr: int,
) -> AtomicBitwiseFlags64:
    """Evaluate condition codes, bitwise status flags, and arithmetic carry/borrow metrics."""
    u_old = old_val & MASK_64
    u_nand = nand_val & MASK_64
    u_new = new_mem & MASK_64

    zero = (u_new == 0)
    negative = bool(u_new & 0x8000_0000_0000_0000)
    old_zero = (u_old == 0)
    old_negative = bool(u_old & 0x8000_0000_0000_0000)

    popcount = bin(u_new).count("1")
    parity_even = (popcount % 2 == 0)
    changed = (u_new != u_old)

    unaligned = bool(eff_addr & 0x7)  # 64-bit qword accesses require 8-byte alignment
    bank = calc_lds_bank(eff_addr)

    carry_or_sum = ((u_old + u_nand) > MASK_64)
    borrow = (u_old < u_nand)

    return AtomicBitwiseFlags64(
        zero=zero,
        negative=negative,
        old_zero=old_zero,
        old_negative=old_negative,
        parity_even=parity_even,
        popcount=popcount,
        changed=changed,
        unaligned_addr=unaligned,
        bank=bank,
        carry_or_sum=carry_or_sum,
        borrow=borrow,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_flags_zero_result(self) -> None:
        # Both MASK_64 -> ~(MASK_64 & MASK_64) = 0!
        flags = evaluate_ds_nand_b64_flags(
            old_val=MASK_64,
            nand_val=MASK_64,
            new_mem=0,
            eff_addr=0x100,
        )
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.old_zero)
        self.assertTrue(flags.old_negative)
        self.assertTrue(flags.parity_even)
        self.assertEqual(flags.popcount, 0)
        self.assertTrue(flags.changed)
        self.assertFalse(flags.unaligned_addr)
        self.assertEqual(flags.bank, (0x100 >> 2) & 31)

    def test_flags_negative_msb_set(self) -> None:
        flags = evaluate_ds_nand_b64_flags(
            old_val=0,
            nand_val=0,
            new_mem=MASK_64,
            eff_addr=0x200,
        )
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.old_zero)
        self.assertFalse(flags.old_negative)
        self.assertEqual(flags.popcount, 64)
        self.assertTrue(flags.parity_even)
        self.assertTrue(flags.changed)

    def test_flags_old_zero_and_negative(self) -> None:
        flags = evaluate_ds_nand_b64_flags(
            old_val=0x8000_0000_0000_0000,
            nand_val=0x8000_0000_0000_0000,
            new_mem=0x7FFF_FFFF_FFFF_FFFF,
            eff_addr=0x300,
        )
        self.assertFalse(flags.old_zero)
        self.assertTrue(flags.old_negative)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.changed)

    def test_flags_parity_and_popcount(self) -> None:
        flags_odd = evaluate_ds_nand_b64_flags(0, 0, 0b111, eff_addr=0x400)
        self.assertEqual(flags_odd.popcount, 3)
        self.assertFalse(flags_odd.parity_even)

        flags_even = evaluate_ds_nand_b64_flags(0, 0, 0b1111, eff_addr=0x400)
        self.assertEqual(flags_even.popcount, 4)
        self.assertTrue(flags_even.parity_even)

    def test_flags_unaligned_address_detection(self) -> None:
        flags_aligned = evaluate_ds_nand_b64_flags(0, 0, 0, eff_addr=0x1000)
        self.assertFalse(flags_aligned.unaligned_addr)

        for byte_off in range(1, 8):
            flags_unaligned = evaluate_ds_nand_b64_flags(0, 0, 0, eff_addr=0x1000 + byte_off)
            self.assertTrue(flags_unaligned.unaligned_addr)

    def test_flags_bank_selection(self) -> None:
        for bank_idx in range(32):
            addr = bank_idx * 4
            flags = evaluate_ds_nand_b64_flags(0, 0, 0, eff_addr=addr)
            self.assertEqual(flags.bank, bank_idx)

    def test_flags_carry_and_borrow_metrics(self) -> None:
        flags_carry = evaluate_ds_nand_b64_flags(
            old_val=MASK_64,
            nand_val=0x0000_0000_0000_0001,
            new_mem=0,
            eff_addr=0x600,
        )
        self.assertTrue(flags_carry.carry_or_sum)

        flags_borrow = evaluate_ds_nand_b64_flags(
            old_val=10,
            nand_val=20,
            new_mem=0,
            eff_addr=0x600,
        )
        self.assertTrue(flags_borrow.borrow)


# ==============================================================================
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

# IEEE 754 Double-Precision (binary64) Architectural Constants
F64_POS_ZERO: int = 0x0000_0000_0000_0000
F64_NEG_ZERO: int = 0x8000_0000_0000_0000
F64_POS_INF: int = 0x7FF0_0000_0000_0000
F64_NEG_INF: int = 0xFFF0_0000_0000_0000
F64_CANONICAL_QNAN: int = 0x7FF8_0000_0000_0000
F64_SIGNALING_NAN: int = 0x7FF0_0000_0000_0001
F64_MIN_SUBNORMAL: int = 0x0000_0000_0000_0001
F64_MAX_SUBNORMAL: int = 0x000F_FFFF_FFFF_FFFF
F64_MIN_NORMAL: int = 0x0010_0000_0000_0000
F64_MAX_NORMAL: int = 0x7FEF_FFFF_FFFF_FFFF


def f64_to_bits(val: float) -> int:
    """Convert float to IEEE 754 64-bit unsigned integer representation."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert IEEE 754 64-bit integer representation to Python float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK_64))[0]


def is_f64_nan(bits: int) -> bool:
    """Check if bits represent an IEEE 754 double-precision NaN."""
    return ((bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000) and ((bits & 0x000F_FFFF_FFFF_FFFF) != 0)


def is_f64_subnormal(bits: int) -> bool:
    """Check if bits represent an IEEE 754 double-precision subnormal number."""
    return ((bits & 0x7FF0_0000_0000_0000) == 0) and ((bits & 0x000F_FFFF_FFFF_FFFF) != 0)


class TestIEEE754BoundaryExtremesAndBitwiseLimits(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.lds = LDSModel()
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState(lds=self.lds)

    def test_signed_zeros_bitwise_nand(self) -> None:
        """NAND with +0.0 results in MASK_64; -0.0 NAND -0.0 clears sign bit."""
        addr = 0x100
        self.lds.write_qword(addr, F64_POS_ZERO)
        self.state.set_vgpr(0, addr)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(0),
            data=Immediate64(F64_POS_ZERO),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, F64_POS_ZERO)
        self.assertEqual(mem, MASK_64)

        # -0.0 NAND -0.0
        self.lds.write_qword(addr, F64_NEG_ZERO)
        instr_neg = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(0),
            data=Immediate64(F64_NEG_ZERO),
        )
        old2, mem2 = self.interp.step(instr_neg, self.state)
        self.assertEqual(old2, F64_NEG_ZERO)
        self.assertEqual(mem2, 0x7FFF_FFFF_FFFF_FFFF)  # Sign bit 63 cleared!

    def test_subnormals_preservation_no_ftz(self) -> None:
        """Double-precision subnormal patterns inverted without flush-to-zero."""
        self.assertTrue(is_f64_subnormal(F64_MIN_SUBNORMAL))
        self.assertTrue(is_f64_subnormal(F64_MAX_SUBNORMAL))

        addr = 0x200
        self.lds.write_qword(addr, F64_MIN_SUBNORMAL)
        self.state.set_vgpr(0, addr)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(4),
            addr=VGPR(0),
            data=Immediate64(MASK_64),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, F64_MIN_SUBNORMAL)
        self.assertEqual(mem, (~F64_MIN_SUBNORMAL) & MASK_64)

    def test_signaling_nan_inversion(self) -> None:
        """Double-precision signaling NaNs inverted cleanly."""
        addr = 0x300
        self.lds.write_qword(addr, F64_SIGNALING_NAN)
        self.state.set_vgpr(0, addr)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(4),
            addr=VGPR(0),
            data=Immediate64(MASK_64),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, F64_SIGNALING_NAN)
        self.assertEqual(mem, (~F64_SIGNALING_NAN) & MASK_64)

    def test_infinities_nand_behavior(self) -> None:
        addr = 0x400
        self.lds.write_qword(addr, F64_POS_INF)
        self.state.set_vgpr(0, addr)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(6),
            addr=VGPR(0),
            data=Immediate64(MASK_64),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, F64_POS_INF)
        self.assertEqual(mem, (~F64_POS_INF) & MASK_64)

    def test_walking_ones_boundary_coverage(self) -> None:
        """Test NAND of each of 64 single bits with MASK_64."""
        addr = 0x500
        for bit in range(64):
            mask = 1 << bit
            self.lds.write_qword(addr, mask)
            self.state.set_vgpr(0, addr)

            instr = DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=VGPRPair(8),
                addr=VGPR(0),
                data=Immediate64(MASK_64),
            )
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, mask)
            self.assertEqual(mem, (~mask) & MASK_64)

    def test_walking_zeros_boundary_coverage(self) -> None:
        """Test NAND of each inverted single-bit mask with MASK_64."""
        addr = 0x600
        for bit in range(64):
            inv_mask = (~(1 << bit)) & MASK_64
            self.lds.write_qword(addr, inv_mask)
            self.state.set_vgpr(0, addr)

            instr = DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=VGPRPair(10),
                addr=VGPR(0),
                data=Immediate64(MASK_64),
            )
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, inv_mask)
            self.assertEqual(mem, 1 << bit)

    def test_integer_boundary_extremes(self) -> None:
        extremes = [0, 1, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFE, MASK_64]
        addr = 0x700
        self.state.set_vgpr(0, addr)

        for val in extremes:
            self.lds.write_qword(addr, val)
            instr = DSNand64Instruction(
                opcode=DSOpcode64.DS_NAND_RTN_B64,
                dst=VGPRPair(12),
                addr=VGPR(0),
                data=Immediate64(MASK_64),
            )
            old, mem = self.interp.step(instr, self.state)
            self.assertEqual(old, val)
            self.assertEqual(mem, (~val) & MASK_64)


# ==============================================================================
# Phase 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

def parse_ds_nand_b64_assembly(asm_str: str) -> DSNand64Instruction:
    """Parse AMD assembly string into a DSNand64Instruction AST node.
    
    Accepts formats:
        ds_nand_rtn_b64 v[dst:dst+1], vaddr, v[data:data+1] [offset:N]
        ds_nand_b64 vaddr, v[data:data+1] [offset:N]
    """
    text = asm_str.strip()
    if not text:
        raise ValueError("Empty assembly string")

    parts = text.replace(",", " ").split()
    if not parts:
        raise ValueError("Empty assembly string")

    op_str = parts[0].lower()
    if op_str == "ds_nand_rtn_b64":
        opcode = DSOpcode64.DS_NAND_RTN_B64
    elif op_str == "ds_nand_b64":
        opcode = DSOpcode64.DS_NAND_B64
    else:
        raise ValueError(f"Unknown opcode '{op_str}', expected ds_nand_rtn_b64 or ds_nand_b64")

    offset = 0
    clean_parts: List[str] = [parts[0]]
    for token in parts[1:]:
        if token.lower().startswith("offset:"):
            try:
                offset_val = int(token.split(":", 1)[1], 0)
            except ValueError as exc:
                raise ValueError(f"Invalid offset format: {token}") from exc
            if not (0 <= offset_val <= 65535):
                raise ValueError(f"Offset {offset_val} out of 16-bit range [0, 65535]")
            offset = offset_val
        else:
            clean_parts.append(token)

    def parse_reg_pair(token: str, expected_type: str) -> Union[VGPRPair, SGPRPair]:
        t = token.lower()
        if t.startswith("v[") and t.endswith("]") and ":" in t:
            inner = t[2:-1]
            base_str, hi_str = inner.split(":", 1)
            prefix = "v"
        elif t.startswith("s[") and t.endswith("]") and ":" in t:
            inner = t[2:-1]
            base_str, hi_str = inner.split(":", 1)
            prefix = "s"
        else:
            raise ValueError(f"Invalid 64-bit register pair syntax: '{token}', expected v[N:N+1] or s[N:N+1]")

        if prefix != expected_type:
            raise TypeError(f"Expected register type '{expected_type}', got '{prefix}' in '{token}'")
        base_idx = int(base_str)
        hi_idx = int(hi_str)
        if hi_idx != base_idx + 1:
            raise ValueError(f"Register pair must be consecutive, got {base_idx}:{hi_idx}")
        return VGPRPair(base_idx) if prefix == "v" else SGPRPair(base_idx)

    def parse_vgpr(token: str) -> VGPR:
        t = token.lower()
        if t.startswith("v[") and t.endswith("]"):
            idx = int(t[2:-1])
        elif t.startswith("v") and t[1:].isdigit():
            idx = int(t[1:])
        else:
            raise ValueError(f"Invalid VGPR syntax: '{token}'")
        return VGPR(idx)

    def parse_operand64(token: str) -> Operand64:
        t = token.lower()
        if t.startswith("v"):
            return parse_reg_pair(token, "v")
        if t.startswith("s"):
            return parse_reg_pair(token, "s")
        val = int(token, 0)
        return Immediate64(val)

    if opcode.is_return:
        if len(clean_parts) != 4:
            raise ValueError(f"ds_nand_rtn_b64 expects 3 operands (vdst, vaddr, vdata), got {len(clean_parts) - 1}")
        dst_pair = parse_reg_pair(clean_parts[1], "v")
        assert isinstance(dst_pair, VGPRPair)
        addr_reg = parse_vgpr(clean_parts[2])
        data_op = parse_operand64(clean_parts[3])
        return DSNand64Instruction(opcode=opcode, dst=dst_pair, addr=addr_reg, data=data_op, offset=offset)
    else:
        if len(clean_parts) != 3:
            raise ValueError(f"ds_nand_b64 expects 2 operands (vaddr, vdata), got {len(clean_parts) - 1}")
        addr_reg = parse_vgpr(clean_parts[1])
        data_op = parse_operand64(clean_parts[2])
        return DSNand64Instruction(opcode=opcode, dst=None, addr=addr_reg, data=data_op, offset=offset)


class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Phase 8: Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_assembly_parser_valid_syntaxes(self) -> None:
        inst1 = parse_ds_nand_b64_assembly("ds_nand_rtn_b64 v[0:1], v2, v[4:5]")
        self.assertEqual(inst1.opcode, DSOpcode64.DS_NAND_RTN_B64)
        self.assertEqual(inst1.dst, VGPRPair(0))
        self.assertEqual(inst1.addr, VGPR(2))
        self.assertEqual(inst1.data, VGPRPair(4))
        self.assertEqual(inst1.offset, 0)

        inst2 = parse_ds_nand_b64_assembly("ds_nand_rtn_b64 v[6:7], v5, s[10:11] offset:32")
        self.assertEqual(inst2.dst, VGPRPair(6))
        self.assertEqual(inst2.addr, VGPR(5))
        self.assertEqual(inst2.data, SGPRPair(10))
        self.assertEqual(inst2.offset, 32)

        inst3 = parse_ds_nand_b64_assembly("ds_nand_b64 v3, 0x1234 offset:0x10")
        self.assertEqual(inst3.opcode, DSOpcode64.DS_NAND_B64)
        self.assertIsNone(inst3.dst)
        self.assertEqual(inst3.addr, VGPR(3))
        self.assertEqual(inst3.data, Immediate64(0x1234))
        self.assertEqual(inst3.offset, 16)

    def test_syntax_rejection_invalid_instructions(self) -> None:
        invalid_cases = [
            "",
            "   ",
            "v_add_f32 v0, v1, v2",
            "ds_nand_rtn_b64",
            "ds_nand_rtn_b64 v[0:1], v2",
            "ds_nand_b64 v2, v[4:5], v[6:7]",
            "ds_nand_rtn_b64 v0, v2, v[4:5]",            # Single VGPR instead of pair for dst
            "ds_nand_rtn_b64 s[0:1], v2, v[4:5]",        # SGPRPair as destination
            "ds_nand_rtn_b64 v[0:1], s2, v[4:5]",        # SGPR as address
            "ds_nand_rtn_b64 v[0:2], v2, v[4:5]",        # Non-consecutive pair
            "ds_nand_rtn_b64 v[255:256], v2, v[4:5]",    # Out of bounds register pair
            "ds_nand_rtn_b64 v[0:1], v2, v[4:5] offset:65536",
            "ds_nand_rtn_b64 v[0:1], v2, v[4:5] offset:-1",
            "ds_nand_rtn_b64 v[0:1], v2, v[4:5] offset:bad",
        ]
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises((ValueError, TypeError)):
                    parse_ds_nand_b64_assembly(case)

    def test_architectural_model_negative_controls(self) -> None:
        with self.assertRaises(ValueError):
            LDSModel(capacity_qwords=0)
        with self.assertRaises(ValueError):
            LDSModel(capacity_qwords=1000)
        with self.assertRaises(ValueError):
            LDSModel(capacity_qwords=-16)

        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)
        state32 = SIMDState(num_lanes=32)
        with self.assertRaises(IndexError):
            state32.is_lane_active(32)
        with self.assertRaises(IndexError):
            state32.is_lane_active(-1)

    def test_mutation_detection_and_instead_of_nand(self) -> None:
        """Mutant: bitwise AND instead of bitwise NAND."""
        def mutant_and(a: int, b: int) -> int:
            return a & b

        val_a = 0x0000_0000_FFFF_0000
        val_b = 0x0000_0000_00FF_FF00
        ref_old, ref_new = oracle_ds_nand_b64(val_a, val_b)
        mutant_res = mutant_and(val_a, val_b)
        self.assertEqual(ref_new, (~0x0000_0000_00FF_0000) & MASK_64)
        self.assertEqual(mutant_res, 0x0000_0000_00FF_0000)
        self.assertNotEqual(ref_new, mutant_res)

    def test_mutation_detection_or_instead_of_nand(self) -> None:
        """Mutant: bitwise OR instead of bitwise NAND."""
        def mutant_or(a: int, b: int) -> int:
            return a | b

        val_a = MASK_64
        val_b = MASK_64
        ref_old, ref_new = oracle_ds_nand_b64(val_a, val_b)
        mutant_res = mutant_or(val_a, val_b)
        self.assertEqual(ref_new, 0)
        self.assertEqual(mutant_res, MASK_64)
        self.assertNotEqual(ref_new, mutant_res)

    def test_mutation_detection_non_returning_destination(self) -> None:
        """Mutant: failing to return pre-operation value to destination VGPR pair."""
        class NonReturningMutant(ScalarInterpreter):
            def step(self, instr: DSNand64Instruction, state: ScalarMachineState) -> Tuple[Optional[int], int]:
                old_val, next_mem = super().step(instr, state)
                if instr.opcode.is_return:
                    assert instr.dst is not None
                    state.set_vgpr_pair(instr.dst.base_index, next_mem)
                return old_val, next_mem

        state = ScalarMachineState()
        addr = 0x100
        state.set_vgpr(1, addr)
        state.set_vgpr_pair(4, 0x02)
        state.lds.write_qword(addr, 0x01)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        mutant = NonReturningMutant()
        mutant.step(instr, state)
        self.assertNotEqual(state.get_vgpr_pair(2), 0x01)
        self.assertEqual(state.get_vgpr_pair(2), MASK_64)

    def test_mutation_detection_premature_writeback_aliasing(self) -> None:
        """Mutant: Destination register written before address is latched."""
        class PrematureWritebackInterpreter(ScalarInterpreter):
            def step(self, instr: DSNand64Instruction, state: ScalarMachineState) -> Tuple[Optional[int], int]:
                if instr.opcode.is_return:
                    assert instr.dst is not None
                    state.set_vgpr_pair(instr.dst.base_index, 0xBAD0)
                vaddr = state.get_vgpr(instr.addr.index)
                eff_addr = calc_lds_effective_address(vaddr, instr.offset)
                curr_mem = state.lds.read_qword(eff_addr)
                data_val = self.resolve_operand(instr.data, state)
                old_val, next_mem = oracle_ds_nand_b64(curr_mem, data_val)
                state.lds.write_qword(eff_addr, next_mem)
                return old_val, next_mem

        state = ScalarMachineState()
        addr = 0x200
        state.set_vgpr(0, addr)
        state.set_vgpr(1, 0)
        state.set_vgpr_pair(2, 0)
        state.lds.write_qword(addr, 0xAA)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(0),
            data=VGPRPair(2),
        )
        mutant = PrematureWritebackInterpreter()
        old, mem = mutant.step(instr, state)
        self.assertNotEqual(state.lds.read_qword(addr), MASK_64)
        self.assertEqual(state.lds.read_qword(0xBAD0), MASK_64)

    def test_mutation_detection_inverted_lane_collision_order(self) -> None:
        """Mutant: SIMD collision serialized in reverse order (lane N to lane 0)."""
        class ReverseSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: DSNand64Instruction, state: SIMDState) -> List[Tuple[int, Optional[int], int]]:
                results = []
                for lane in reversed(range(state.num_lanes)):
                    if not state.is_lane_active(lane):
                        continue
                    vaddr = state.get_vgpr(lane, instr.addr.index)
                    data_val = self.resolve_lane_operand(instr.data, lane, state)
                    eff_addr = calc_lds_effective_address(vaddr, instr.offset)
                    curr_mem = state.lds.read_qword(eff_addr)
                    old_val, next_mem = oracle_ds_nand_b64(curr_mem, data_val)
                    state.lds.write_qword(eff_addr, next_mem)
                    if instr.opcode.is_return:
                        assert instr.dst is not None
                        state.set_vgpr_pair(lane, instr.dst.base_index, old_val)
                    results.append((lane, old_val, next_mem))
                return results

        state = SIMDState(num_lanes=32, exec_mask=0x3)  # Lanes 0 and 1 active
        common_addr = 0x300
        state.lds.write_qword(common_addr, MASK_64)

        state.set_vgpr(0, 1, common_addr)
        state.set_vgpr_pair(0, 4, 0x0000_00FF_0000_0000)
        state.set_vgpr(1, 1, common_addr)
        state.set_vgpr_pair(1, 4, 0x0000_FF00_0000_0000)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        mutant = ReverseSIMDInterpreter()
        mutant.step(instr, state)

        # In reverse order, Lane 0 executed second and saw Lane 1's intermediate memory!
        self.assertNotEqual(state.get_vgpr_pair(0, 2), MASK_64)
        self.assertEqual(state.get_vgpr_pair(0, 2), (~0x0000_FF00_0000_0000) & MASK_64)

    def test_mutation_detection_ignore_exec_mask(self) -> None:
        """Mutant: Inactive lanes execute and corrupt registers/memory."""
        class InactiveExecMutant(SIMDInterpreter):
            def step(self, instr: DSNand64Instruction, state: SIMDState) -> List[Tuple[int, Optional[int], int]]:
                orig_mask = state.exec_mask
                state.exec_mask = 0xFFFF_FFFF
                try:
                    return super().step(instr, state)
                finally:
                    state.exec_mask = orig_mask

        state = SIMDState(num_lanes=32, exec_mask=0x1)  # Only lane 0 active
        sentinel = 0xC001_CAFE_0000_0000
        for lane in range(32):
            state.set_vgpr_pair(lane, 2, sentinel)
            state.set_vgpr(lane, 1, 0x400 + lane * 8)
            state.set_vgpr_pair(lane, 4, 0xFF)

        instr = DSNand64Instruction(
            opcode=DSOpcode64.DS_NAND_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        mutant = InactiveExecMutant()
        mutant.step(instr, state)

        self.assertNotEqual(state.get_vgpr_pair(5, 2), sentinel)

    def test_mutation_detection_dropped_immediate_offset(self) -> None:
        def mutant_calc_addr(vaddr: int, offset: int = 0) -> int:
            return vaddr & MASK_32

        base = 0x200
        off = 0x10
        correct = calc_lds_effective_address(base, off)
        buggy = mutant_calc_addr(base, off)
        self.assertEqual(correct, 0x210)
        self.assertEqual(buggy, 0x200)
        self.assertNotEqual(correct, buggy)

    def test_mutation_detection_wrong_alignment_shift(self) -> None:
        # For 64-bit qword, shift must be >> 3 (8 bytes), not >> 2 (4 bytes dword)
        def mutant_qword_idx(eff_addr: int, capacity: int = 8192) -> int:
            return (eff_addr >> 2) & (capacity - 1)

        addr = 0x100
        correct = calc_lds_qword_index(addr)
        buggy = mutant_qword_idx(addr)
        self.assertEqual(correct, 0x20)
        self.assertEqual(buggy, 0x40)
        self.assertNotEqual(correct, buggy)


if __name__ == "__main__":
    unittest.main()
