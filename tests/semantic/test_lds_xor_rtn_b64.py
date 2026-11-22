"""Semantic reference oracle and architectural model for ds_xor_rtn_b64.

RDNA2 / GFX10 Local Data Share (LDS) 64-Bit Bitwise Atomic XOR with Return:
Performs a 64-bit atomic bitwise XOR operation on LDS memory and returns
the previous memory value to the destination VGPR pair v[dst:dst+1].

Syntax:
    ds_xor_rtn_b64 v[dst:dst+1], vaddr, v[data:data+1] [offset:offset16]
    ds_xor_b64 vaddr, v[data:data+1] [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_capacity_qwords - 1)
    old_val = LDS[qword_index]
    new_mem = (old_val ^ vdata) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_mem
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
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_xor_b64(current_mem: int, xor_val: int) -> Tuple[int, int]:
    old_val = current_mem & MASK_64
    vdata = xor_val & MASK_64
    new_mem = (old_val ^ vdata) & MASK_64
    return old_val, new_mem


class LDSModel:
    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.capacity_dwords = capacity_qwords * 2
        self.memory = [0] * self.capacity_dwords

    def read_qword(self, eff_addr: int) -> int:
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        low = self.memory[dword_idx]
        high = self.memory[dword_idx + 1]
        return ((high << 32) | low) & MASK_64

    def write_qword(self, eff_addr: int, val: int) -> None:
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        self.memory[dword_idx] = val & MASK_32
        self.memory[dword_idx + 1] = (val >> 32) & MASK_32

    def atomic_xor_b64(
        self,
        vaddr: int,
        xor_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem = oracle_ds_xor_b64(curr, xor_val)
        self.write_qword(eff_addr, new_mem)
        return old_val, new_mem


class TestLdsXorRtnB64ReferenceOracle(unittest.TestCase):
    def test_oracle_xor_with_zero(self) -> None:
        old, mem = oracle_ds_xor_b64(0x12345678_9ABCDEF0, 0)
        self.assertEqual(old, 0x12345678_9ABCDEF0)
        self.assertEqual(mem, 0x12345678_9ABCDEF0)

    def test_oracle_xor_self_cancels_to_zero(self) -> None:
        val = 0xAA55AA55_55AAAA55
        old, mem = oracle_ds_xor_b64(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, 0)

    def test_oracle_xor_with_all_ones_inversion(self) -> None:
        val = 0x12345678_9ABCDEF0
        old, mem = oracle_ds_xor_b64(val, MASK_64)
        self.assertEqual(old, val)
        self.assertEqual(mem, (~val) & MASK_64)

    def test_oracle_double_xor_restores_original(self) -> None:
        val = 0xCAFEBABE_DEADBEEF
        key = 0x11223344_55667788
        old1, enc = oracle_ds_xor_b64(val, key)
        old2, dec = oracle_ds_xor_b64(enc, key)
        self.assertEqual(dec, val)

    def test_effective_address_calculation_with_offset(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)
        self.assertEqual(calc_lds_qword_index(eff), 0x1040 >> 3)

    def test_effective_address_wrapping_modulo_2_32(self) -> None:
        eff = calc_lds_effective_address(0xFFFF_FFF8, 0x0010)
        self.assertEqual(eff, 0x0000_0008)

    def test_qword_alignment_drops_low_3_bits(self) -> None:
        idx_base = calc_lds_qword_index(0x400)
        for byte_off in range(8):
            self.assertEqual(calc_lds_qword_index(0x400 + byte_off), idx_base)
        self.assertEqual(idx_base, 0x400 >> 3)

    def test_lds_capacity_wrapping(self) -> None:
        cap_bytes = 65536
        idx_low = calc_lds_qword_index(0x100)
        idx_wrapped = calc_lds_qword_index(0x100 + cap_bytes)
        self.assertEqual(idx_low, idx_wrapped)

    def test_model_atomic_xor_returns_pre_op_value(self) -> None:
        lds = LDSModel()
        addr = 0x200
        lds.write_qword(addr, 0x00FF00FF_00FF00FF)

        old_val, new_val = lds.atomic_xor_b64(addr, 0x0000FFFF_0000FFFF)
        self.assertEqual(old_val, 0x00FF00FF_00FF00FF)
        self.assertEqual(new_val, 0x00FFFF00_00FFFF00)
        self.assertEqual(lds.read_qword(addr), 0x00FFFF00_00FFFF00)

# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

@dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of architectural range [0, 255]")

    @property
    def name(self) -> str:
        return f"v[{self.index}]"


@dataclass(frozen=True)
class VGPRPair:
    base_index: int

    def __post_init__(self) -> None:
        if not (0 <= self.base_index <= 254):
            raise ValueError(f"VGPRPair base_index {self.base_index} out of range [0, 254]")

    @property
    def name(self) -> str:
        return f"v[{self.base_index}:{self.base_index + 1}]"


@dataclass(frozen=True)
class SGPRPair:
    base_index: int

    def __post_init__(self) -> None:
        if not (0 <= self.base_index <= 104):
            raise ValueError(f"SGPRPair base_index {self.base_index} out of range [0, 104]")

    @property
    def name(self) -> str:
        return f"s[{self.base_index}:{self.base_index + 1}]"


@dataclass(frozen=True)
class Immediate64:
    value: int

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK_64:X}"


Operand64 = Union[VGPRPair, SGPRPair, Immediate64]


class DSOpcode64(Enum):
    DS_XOR_RTN_B64 = auto()
    DS_XOR_B64 = auto()

    @property
    def is_return(self) -> bool:
        return self == DSOpcode64.DS_XOR_RTN_B64

    @property
    def op_name(self) -> str:
        return "ds_xor_rtn_b64" if self.is_return else "ds_xor_b64"


@dataclass(frozen=True)
class DSXor64Instruction:
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
                raise ValueError("Destination must be None for non-returning ds_xor_b64 opcode")
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
            return f"ds_xor_rtn_b64 {self.dst.name}, {self.addr.name}, {self.data.name}{offset_part}"
        return f"ds_xor_b64 {self.addr.name}, {self.data.name}{offset_part}"


class TestSymbolicAST(unittest.TestCase):
    def test_instruction_ast_construction_returning(self) -> None:
        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(0),
            addr=VGPR(2),
            data=VGPRPair(4),
            offset=16,
        )
        self.assertEqual(instr.opcode, DSOpcode64.DS_XOR_RTN_B64)
        self.assertTrue(instr.opcode.is_return)
        self.assertEqual(instr.dst.base_index, 0)
        self.assertEqual(instr.addr.index, 2)
        self.assertEqual(instr.offset, 16)

    def test_instruction_ast_construction_non_returning(self) -> None:
        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_B64,
            dst=None,
            addr=VGPR(10),
            data=Immediate64(0x1234_5678_9ABC_DEF0),
        )
        self.assertEqual(instr.opcode, DSOpcode64.DS_XOR_B64)
        self.assertFalse(instr.opcode.is_return)
        self.assertIsNone(instr.dst)

    def test_disassembly_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(6),
            addr=VGPR(5),
            data=Immediate64(0x1234),
        )
        self.assertEqual(instr_no_off.disassemble(), "ds_xor_rtn_b64 v[6:7], v[5], 0x1234")

        instr_off = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(12),
            addr=VGPR(14),
            data=SGPRPair(10),
            offset=64,
        )
        self.assertEqual(instr_off.disassemble(), "ds_xor_rtn_b64 v[12:13], v[14], s[10:11] offset:64")

    def test_disassembly_non_returning_with_and_without_offset(self) -> None:
        instr_no_off = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_B64,
            dst=None,
            addr=VGPR(2),
            data=VGPRPair(4),
        )
        self.assertEqual(instr_no_off.disassemble(), "ds_xor_b64 v[2], v[4:5]")

    def test_dst_type_enforcement_rejects_sgpr(self) -> None:
        with self.assertRaises(TypeError):
            DSXor64Instruction(
                opcode=DSOpcode64.DS_XOR_RTN_B64,
                dst=SGPRPair(0),
                addr=VGPR(1),
                data=Immediate64(0),
            )

# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
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
        instr: DSXor64Instruction,
        state: ScalarMachineState,
    ) -> Tuple[Optional[int], int]:
        vaddr = state.get_vgpr(instr.addr.index)
        data_val = self.resolve_operand(instr.data, state)

        eff_addr = calc_lds_effective_address(vaddr, instr.offset)
        curr_mem = state.lds.read_qword(eff_addr)

        old_val, next_mem = oracle_ds_xor_b64(curr_mem, data_val)
        state.lds.write_qword(eff_addr, next_mem)

        if instr.opcode.is_return:
            assert instr.dst is not None
            state.set_vgpr_pair(instr.dst.base_index, old_val)
            return old_val, next_mem

        return None, next_mem


class TestScalarInterpreter(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_step_returning_vgpr_operand(self) -> None:
        addr = 0x200
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr_pair(4, 0x00FF00FF_00FF00FF)
        self.state.lds.write_qword(addr, 0x0F0F0F0F_0F0F0F0F)

        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=VGPRPair(4),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(mem, 0x0FF00FF0_0FF00FF0)
        self.assertEqual(self.state.get_vgpr_pair(2), 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(self.state.lds.read_qword(addr), 0x0FF00FF0_0FF00FF0)

    def test_scalar_step_returning_sgpr_operand(self) -> None:
        addr = 0x300
        self.state.set_vgpr(1, addr)
        self.state.set_sgpr_pair(10, MASK_64)
        self.state.lds.write_qword(addr, 0x12345678_9ABCDEF0)

        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(2),
            addr=VGPR(1),
            data=SGPRPair(10),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0x12345678_9ABCDEF0)
        self.assertEqual(mem, (~0x12345678_9ABCDEF0) & MASK_64)

    def test_scalar_step_returning_immediate_operand(self) -> None:
        addr = 0x400
        self.state.set_vgpr(1, addr)
        self.state.lds.write_qword(addr, 0xAAAA5555_AAAA5555)

        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(4),
            addr=VGPR(1),
            data=Immediate64(0),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xAAAA5555_AAAA5555)
        self.assertEqual(mem, 0xAAAA5555_AAAA5555)

    def test_scalar_step_non_returning_opcode(self) -> None:
        addr = 0x500
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr_pair(2, 0xFF00FF00_FF00FF00)
        self.state.lds.write_qword(addr, 0xFF00FF00_FF00FF00)

        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_B64,
            dst=None,
            addr=VGPR(1),
            data=VGPRPair(2),
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertIsNone(old)
        self.assertEqual(mem, 0)
        self.assertEqual(self.state.lds.read_qword(addr), 0)

    def test_scalar_step_with_immediate_offset(self) -> None:
        base_addr = 0x600
        offset = 0x10
        eff_addr = base_addr + offset
        self.state.set_vgpr(2, base_addr)
        self.state.lds.write_qword(eff_addr, 0xF0F0F0F0_F0F0F0F0)

        instr = DSXor64Instruction(
            opcode=DSOpcode64.DS_XOR_RTN_B64,
            dst=VGPRPair(6),
            addr=VGPR(2),
            data=Immediate64(MASK_64),
            offset=offset,
        )
        old, mem = self.interp.step(instr, self.state)
        self.assertEqual(old, 0xF0F0F0F0_F0F0F0F0)
        self.assertEqual(mem, 0x0F0F0F0F_0F0F0F0F)

if __name__ == "__main__":
    unittest.main()
