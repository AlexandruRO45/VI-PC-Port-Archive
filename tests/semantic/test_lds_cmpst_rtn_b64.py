from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_rtn_b64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Compare-and-Swap with Return:
Performs a 64-bit atomic compare-and-swap on LDS memory and returns the original 64-bit value.

Syntax:
    ds_cmpst_rtn_b64 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    addr_lo = effective_byte_address & ~3
    old_lo = LDS[addr_lo]
    old_hi = LDS[addr_lo + 4]
    old_val64 = (old_hi << 32) | old_lo
    if old_val64 == vdata0:
        LDS[addr_lo] = vdata1 & 0xFFFFFFFF
        LDS[addr_lo + 4] = (vdata1 >> 32) & 0xFFFFFFFF
    vdst = old_val64
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
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned integer as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit unsigned integer into (lo, hi) 32-bit words."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


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


def ref_ds_cmpst_rtn_b64(
    current_mem64: int,
    cmp_val64: int,
    new_val64: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_rtn_b64.

    Compares current_mem64 with cmp_val64; if equal, replaces memory with new_val64.
    Always returns the pre-operation original memory value.

    Returns:
        Tuple[int, int, bool]: (returned_old_val, new_mem_val, swapped)
    """
    old_val = u64(current_mem64)
    cmp_masked = u64(cmp_val64)
    new_masked = u64(new_val64)

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

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword (two consecutive dwords) from LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        return combine_u32s_to_u64(self.memory[idx_lo], self.memory[idx_hi])

    def write_qword(self, eff_addr: int, val64: int) -> None:
        """Write a 64-bit qword (two consecutive dwords) to LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        lo, hi = split_u64_to_u32s(val64)
        self.memory[idx_lo] = lo
        self.memory[idx_hi] = hi

    def atomic_cmpst_rtn_b64(
        self,
        vaddr: int,
        cmp_val64: int,
        new_val64: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic 64-bit compare-and-swap on LDS.

        Returns:
            Tuple[int, int, bool]: (old_val, new_mem, swapped)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(curr, cmp_val64, new_val64)
        if swapped:
            self.write_qword(eff_addr, new_mem)
        return old_val, new_mem, swapped


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) 64-bit compare-and-swap atomic opcodes."""
    DS_CMPST_RTN_B64 = "ds_cmpst_rtn_b64"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegSingle:
    """32-bit single register representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(f"Register index out of bounds: {self.reg_type.value}[{self.index}]")

    def __str__(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"


@dataclass(frozen=True)
class RegPair:
    """64-bit consecutive register pair representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = (DEFAULT_NUM_VGPRS - 1) if self.reg_type == RegType.VGPR else (DEFAULT_NUM_SGPRS - 1)
        if not (0 <= self.index < max_limit):
            raise ValueError(f"Register pair base index out of bounds: {self.reg_type.value}[{self.index}:{self.index + 1}]")

    def __str__(self) -> str:
        return f"{self.reg_type.value}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Immediate64:
    """64-bit immediate constant operand."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u64(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand64 = Union[RegPair, Immediate64]


@dataclass(frozen=True)
class DSCmpstRtnB64Instruction:
    """Symbolic AST node for ds_cmpst_rtn_b64 instruction."""
    vdst: RegPair
    vaddr: RegSingle
    vdata0: Operand64  # 64-bit compare value
    vdata1: Operand64  # 64-bit new value
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_CMPST_RTN_B64

    def __post_init__(self) -> None:
        if not isinstance(self.vdst, RegPair) or self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR RegPair, got {self.vdst}")
        if not isinstance(self.vaddr, RegSingle) or self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR RegSingle, got {self.vaddr}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of bounds: {self.offset}")

    def disassemble(self) -> str:
        """Emit canonical AMD assembly string."""
        parts = [self.opcode.value, str(self.vdst), str(self.vaddr), str(self.vdata0), str(self.vdata1)]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising 32-bit/64-bit register files and LDS memory."""

    def __init__(self, lds_size_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def get_u64_vgpr(self, base_index: int) -> int:
        return combine_u32s_to_u64(self.get_vgpr(base_index), self.get_vgpr(base_index + 1))

    def set_u64_vgpr(self, base_index: int, val64: int) -> None:
        lo, hi = split_u64_to_u32s(val64)
        self.set_vgpr(base_index, lo)
        self.set_vgpr(base_index + 1, hi)

    def get_reg_single(self, reg: RegSingle) -> int:
        if reg.reg_type == RegType.VGPR:
            return self.get_vgpr(reg.index)
        return self.get_sgpr(reg.index)

    def set_reg_single(self, reg: RegSingle, val: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_vgpr(reg.index, val)
        else:
            self.set_sgpr(reg.index, val)

    def get_reg_pair(self, reg: RegPair) -> int:
        if reg.reg_type == RegType.VGPR:
            return self.get_u64_vgpr(reg.index)
        return combine_u32s_to_u64(self.get_sgpr(reg.index), self.get_sgpr(reg.index + 1))

    def set_reg_pair(self, reg: RegPair, val64: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_u64_vgpr(reg.index, val64)
        else:
            lo, hi = split_u64_to_u32s(val64)
            self.set_sgpr(reg.index, lo)
            self.set_sgpr(reg.index + 1, hi)

    def resolve_operand64(self, op: Operand64) -> int:
        if isinstance(op, Immediate64):
            return op.val
        return self.get_reg_pair(op)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_cmpst_rtn_b64."""

    def execute(self, instr: DSCmpstRtnB64Instruction, state: ScalarState) -> Tuple[int, int, bool]:
        """Execute a single ds_cmpst_rtn_b64 instruction on scalar state.

        Returns:
            Tuple[int, int, bool]: (old_val64, new_mem64, swapped)
        """
        # 1. Latch inputs prior to any register writeback
        vaddr = state.get_reg_single(instr.vaddr)
        cmp_val = state.resolve_operand64(instr.vdata0)
        new_val = state.resolve_operand64(instr.vdata1)

        # 2. Perform atomic LDS 64-bit compare-and-swap
        old_val, new_mem, swapped = state.lds.atomic_cmpst_rtn_b64(
            vaddr=vaddr,
            cmp_val64=cmp_val,
            new_val64=new_val,
            offset=instr.offset,
        )

        # 3. Write pre-operation original 64-bit value back to destination register pair
        state.set_reg_pair(instr.vdst, old_val)

        return old_val, new_mem, swapped


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState:
    """SIMD wavefront execution state supporting Wave32 and Wave64 execution modes."""

    def __init__(
        self,
        num_lanes: int = 64,
        exec_mask: Optional[int] = None,
        lds_size_dwords: int = LDS_SIZE_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = ((1 << num_lanes) - 1) if exec_mask is None else (exec_mask & ((1 << num_lanes) - 1))
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))

    def get_lane_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index]

    def set_lane_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = u32(val)

    def get_lane_u64_vgpr(self, lane: int, base_index: int) -> int:
        return combine_u32s_to_u64(self.get_lane_vgpr(lane, base_index), self.get_lane_vgpr(lane, base_index + 1))

    def set_lane_u64_vgpr(self, lane: int, base_index: int, val64: int) -> None:
        lo, hi = split_u64_to_u32s(val64)
        self.set_lane_vgpr(lane, base_index, lo)
        self.set_lane_vgpr(lane, base_index + 1, hi)

    def resolve_lane_operand64(self, op: Operand64, lane: int) -> int:
        if isinstance(op, Immediate64):
            return op.val
        if op.reg_type == RegType.SGPR:
            return combine_u32s_to_u64(self.sgprs[op.index], self.sgprs[op.index + 1])
        return self.get_lane_u64_vgpr(lane, op.index)


class SIMDInterpreter:
    """SIMD multi-lane instruction interpreter for ds_cmpst_rtn_b64."""

    def execute(
        self,
        instr: DSCmpstRtnB64Instruction,
        state: SIMDState,
    ) -> List[Optional[int]]:
        """Execute ds_cmpst_rtn_b64 across SIMD lanes with EXEC masking and collision serialization.

        Returns:
            List[Optional[int]]: Per-lane returned old_val64 for active lanes, or None for inactive lanes.
        """
        results: List[Optional[int]] = [None] * state.num_lanes

        # 1. Latch per-lane inputs prior to memory updates
        lane_inputs: list[tuple[int, int, int, int] | None] = [None] * state.num_lanes
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue
            vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
            cmp_val = state.resolve_lane_operand64(instr.vdata0, lane)
            new_val = state.resolve_lane_operand64(instr.vdata1, lane)
            eff_addr = calc_lds_effective_address(vaddr, instr.offset)
            lane_inputs[lane] = (lane, eff_addr, cmp_val, new_val)

        # 2. Execute 64-bit atomic operations in ascending lane order (hardware serialization order)
        for lane in range(state.num_lanes):
            entry = lane_inputs[lane]
            if entry is None:
                continue
            _, eff_addr, cmp_val, new_val = entry
            curr_val = state.lds.read_qword(eff_addr)
            old_val, updated_mem, swapped = ref_ds_cmpst_rtn_b64(curr_val, cmp_val, new_val)
            if swapped:
                state.lds.write_qword(eff_addr, updated_mem)
            results[lane] = old_val

        # 3. Writeback to destination VGPR pairs for active lanes
        for lane in range(state.num_lanes):
            if results[lane] is not None:
                state.set_lane_u64_vgpr(lane, instr.vdst.index, results[lane])  # type: ignore

        return results





class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_64bit_match_and_swap(self) -> None:
        val = 0x1122_3344_5566_7788
        new_val = 0x99AA_BBCC_DDEE_FF00
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(val, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, new_val)

    def test_oracle_64bit_low_dword_mismatch(self) -> None:
        val = 0x1122_3344_5566_7788
        cmp_val = 0x1122_3344_5566_0000 # Low dword differs
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(val, cmp_val, 0x9999)
        self.assertFalse(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, val)

    def test_oracle_64bit_high_dword_mismatch(self) -> None:
        val = 0x1122_3344_5566_7788
        cmp_val = 0x0000_0000_5566_7788 # High dword differs
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(val, cmp_val, 0x9999)
        self.assertFalse(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, val)

    def test_dword_splitting_and_combining(self) -> None:
        val = 0xDEAD_BEEF_CAFE_BABE
        lo, hi = split_u64_to_u32s(val)
        self.assertEqual(lo, 0xCAFE_BABE)
        self.assertEqual(hi, 0xDEAD_BEEF)
        self.assertEqual(combine_u32s_to_u64(lo, hi), val)

    def test_effective_address_calculation(self) -> None:
        eff = calc_lds_effective_address(0x4000, 0x20)
        self.assertEqual(eff, 0x4020)

        eff_wrap = calc_lds_effective_address(0xFFFFFFFF, 1)
        self.assertEqual(eff_wrap, 0)

    def test_architectural_memory_qword_read_write(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x80
        val = 0x0123_4567_89AB_CDEF
        lds.write_qword(addr, val)

        self.assertEqual(lds.read_qword(addr), val)
        # Check individual dwords: lo at idx, hi at idx+1
        idx = calc_lds_dword_index(addr, 1024)
        self.assertEqual(lds.memory[idx], 0x89AB_CDEF)
        self.assertEqual(lds.memory[idx + 1], 0x0123_4567)

    def test_architectural_memory_64bit_boundary_wrap(self) -> None:
        # Wrap across LDS capacity: last dword at size - 1, second dword wraps to 0
        lds = LDSMemory(size_dwords=1024)
        last_byte_addr = (1024 - 1) * 4 # 0xFFC
        val = 0xAAAA_BBBB_CCCC_DDDD
        lds.write_qword(last_byte_addr, val)

        self.assertEqual(lds.read_qword(last_byte_addr), val)
        self.assertEqual(lds.memory[1023], 0xCCCC_DDDD)
        self.assertEqual(lds.memory[0], 0xAAAA_BBBB)

    def test_architectural_atomic_cmpst_b64(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x100
        initial_val = 0x5555_6666_7777_8888
        new_val = 0x1111_2222_3333_4444

        lds.write_qword(addr, initial_val)

        # Successful 64-bit CAS
        old, mem, swapped = lds.atomic_cmpst_rtn_b64(vaddr=addr, cmp_val64=initial_val, new_val64=new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, initial_val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_qword(addr), new_val)

        # Failed 64-bit CAS
        old2, mem2, swapped2 = lds.atomic_cmpst_rtn_b64(vaddr=addr, cmp_val64=0x9999, new_val64=0xAAAA)
        self.assertFalse(swapped2)
        self.assertEqual(old2, new_val)
        self.assertEqual(mem2, new_val)
        self.assertEqual(lds.read_qword(addr), new_val)


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        vdst = RegPair(0, RegType.VGPR)
        vaddr = RegSingle(2, RegType.VGPR)
        vdata0 = RegPair(4, RegType.VGPR)
        vdata1 = RegPair(6, RegType.VGPR)

        instr = DSCmpstRtnB64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_CMPST_RTN_B64)
        self.assertEqual(instr.vdst.index, 0)
        self.assertEqual(instr.vaddr.index, 2)
        self.assertEqual(instr.offset, 16)
        self.assertFalse(instr.gds)

    def test_ast_with_immediate_operands(self) -> None:
        vdst = RegPair(10, RegType.VGPR)
        vaddr = RegSingle(12, RegType.VGPR)
        cmp_imm = Immediate64(0x1122_3344_5566_7788)
        new_imm = Immediate64(0x99AA_BBCC_DDEE_FF00)

        instr = DSCmpstRtnB64Instruction(vdst=vdst, vaddr=vaddr, vdata0=cmp_imm, vdata1=new_imm, offset=0)
        self.assertEqual(instr.vdata0, cmp_imm)
        self.assertEqual(instr.vdata1, new_imm)

    def test_disassembly_formatting(self) -> None:
        vdst = RegPair(0, RegType.VGPR)
        vaddr = RegSingle(2, RegType.VGPR)
        vdata0 = RegPair(4, RegType.VGPR)
        vdata1 = RegPair(6, RegType.VGPR)

        instr_no_off = DSCmpstRtnB64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1)
        self.assertEqual(instr_no_off.disassemble(), "ds_cmpst_rtn_b64 v[0:1], v[2], v[4:5], v[6:7]")

        instr_off = DSCmpstRtnB64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=32)
        self.assertEqual(instr_off.disassemble(), "ds_cmpst_rtn_b64 v[0:1], v[2], v[4:5], v[6:7], offset:32")

        instr_gds = DSCmpstRtnB64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=8, gds=True)
        self.assertEqual(instr_gds.disassemble(), "ds_cmpst_rtn_b64 v[0:1], v[2], v[4:5], v[6:7], offset:8, gds")

    def test_ast_validation_errors(self) -> None:
        v0 = RegPair(0, RegType.VGPR)
        v1 = RegSingle(1, RegType.VGPR)
        s0_pair = RegPair(0, RegType.SGPR)
        s0_single = RegSingle(0, RegType.SGPR)

        # vdst cannot be SGPR or RegSingle
        with self.assertRaises(ValueError):
            DSCmpstRtnB64Instruction(vdst=s0_pair, vaddr=v1, vdata0=v0, vdata1=v0)
        with self.assertRaises(ValueError):
            DSCmpstRtnB64Instruction(vdst=v1, vaddr=v1, vdata0=v0, vdata1=v0)  # type: ignore

        # vaddr cannot be SGPR or RegPair
        with self.assertRaises(ValueError):
            DSCmpstRtnB64Instruction(vdst=v0, vaddr=s0_single, vdata0=v0, vdata1=v0)
        with self.assertRaises(ValueError):
            DSCmpstRtnB64Instruction(vdst=v0, vaddr=v0, vdata0=v0, vdata1=v0)  # type: ignore

        # Offset validation
        with self.assertRaises(ValueError):
            DSCmpstRtnB64Instruction(vdst=v0, vaddr=v1, vdata0=v0, vdata1=v0, offset=-1)
        with self.assertRaises(ValueError):
            DSCmpstRtnB64Instruction(vdst=v0, vaddr=v1, vdata0=v0, vdata1=v0, offset=65536)


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState(lds_size_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_64bit_cmpst_match_execution(self) -> None:
        addr = 0x40
        initial_val = 0x1122_3344_5566_7788
        new_val = 0x99AA_BBCC_DDEE_FF00

        self.state.lds.write_qword(addr, initial_val)
        self.state.set_reg_single(RegSingle(1), addr)
        self.state.set_reg_pair(RegPair(2), initial_val)  # v[2:3] = cmp_val
        self.state.set_reg_pair(RegPair(4), new_val)      # v[4:5] = new_val
        self.state.set_reg_pair(RegPair(6), 0xDEAD_BEEF)  # v[6:7] = sentinel

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            vdata1=RegPair(4),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, initial_val)
        self.assertEqual(new_mem, new_val)
        self.assertEqual(self.state.get_reg_pair(RegPair(6)), initial_val)
        self.assertEqual(self.state.lds.read_qword(addr), new_val)

    def test_scalar_64bit_cmpst_mismatch_execution(self) -> None:
        addr = 0x40
        initial_val = 0x1122_3344_5566_7788
        cmp_val = 0x1122_3344_0000_0000

        self.state.lds.write_qword(addr, initial_val)
        self.state.set_reg_single(RegSingle(1), addr)
        self.state.set_reg_pair(RegPair(2), cmp_val)
        self.state.set_reg_pair(RegPair(4), 0x9999_9999_9999_9999)

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            vdata1=RegPair(4),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertFalse(swapped)
        self.assertEqual(old_val, initial_val)
        self.assertEqual(new_mem, initial_val)
        self.assertEqual(self.state.get_reg_pair(RegPair(6)), initial_val)
        self.assertEqual(self.state.lds.read_qword(addr), initial_val)

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x100
        offset = 0x18
        eff_addr = base_addr + offset

        initial_val = 0xCAFE_BABE_0123_4567
        new_val = 0x5555_AAAA_6666_BBBB

        self.state.lds.write_qword(eff_addr, initial_val)
        self.state.set_reg_single(RegSingle(1), base_addr)

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(0),
            vaddr=RegSingle(1),
            vdata0=Immediate64(initial_val),
            vdata1=Immediate64(new_val),
            offset=offset,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, initial_val)
        self.assertEqual(new_mem, new_val)
        self.assertEqual(self.state.get_reg_pair(RegPair(0)), initial_val)
        self.assertEqual(self.state.lds.read_qword(eff_addr), new_val)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        # Each lane has its own 8-byte aligned address
        for lane in range(32):
            addr = lane * 8
            val = 0x1000_0000_0000_0000 + lane
            state.lds.write_qword(addr, val)
            state.set_lane_vgpr(lane, 1, addr) # v1: vaddr

            # Even lanes match, odd lanes mismatch
            cmp_val = val if (lane % 2 == 0) else 0x9999
            state.set_lane_u64_vgpr(lane, 2, cmp_val) # v[2:3]
            state.set_lane_u64_vgpr(lane, 4, 0x5000_0000_0000_0000 + lane) # v[4:5]
            state.set_lane_u64_vgpr(lane, 6, 0xDEAD_BEEF_DEAD_BEEF) # v[6:7]

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            vdata1=RegPair(4),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            expected_old = 0x1000_0000_0000_0000 + lane
            self.assertEqual(results[lane], expected_old)
            self.assertEqual(state.get_lane_u64_vgpr(lane, 6), expected_old)
            if lane % 2 == 0:
                self.assertEqual(state.lds.read_qword(lane * 8), 0x5000_0000_0000_0000 + lane)
            else:
                self.assertEqual(state.lds.read_qword(lane * 8), expected_old)

    def test_wave64_divergent_exec_mask(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        sentinel = 0xCAFE_BABE_CAFE_BABE
        for lane in range(64):
            addr = lane * 8
            val = 0x2000_0000_0000_0000 + lane
            state.lds.write_qword(addr, val)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_u64_vgpr(lane, 2, val) # Match
            state.set_lane_u64_vgpr(lane, 4, 0x3000_0000_0000_0000 + lane)
            state.set_lane_u64_vgpr(lane, 6, sentinel)

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            vdata1=RegPair(4),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            val = 0x2000_0000_0000_0000 + lane
            if lane % 2 == 0:
                self.assertEqual(results[lane], val)
                self.assertEqual(state.get_lane_u64_vgpr(lane, 6), val)
                self.assertEqual(state.lds.read_qword(lane * 8), 0x3000_0000_0000_0000 + lane)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_u64_vgpr(lane, 6), sentinel)
                self.assertEqual(state.lds.read_qword(lane * 8), val)

    def test_multi_lane_collision_chained_64bit_cas(self) -> None:
        common_addr = 0x80
        state = SIMDState(num_lanes=32, exec_mask=0x7, lds_size_dwords=1024) # Lanes 0..2 active
        interp = SIMDInterpreter()

        state.lds.write_qword(common_addr, 0x10)

        chain = [(0x10, 0x20), (0x20, 0x30), (0x30, 0x40)]
        for lane, (cmp_val, new_val) in enumerate(chain):
            state.set_lane_vgpr(lane, 1, common_addr)
            state.set_lane_u64_vgpr(lane, 2, cmp_val)
            state.set_lane_u64_vgpr(lane, 4, new_val)
            state.set_lane_u64_vgpr(lane, 6, 0)

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            vdata1=RegPair(4),
        )

        results = interp.execute(instr, state)

        self.assertEqual(results[0], 0x10)
        self.assertEqual(state.get_lane_u64_vgpr(0, 6), 0x10)

        self.assertEqual(results[1], 0x20)
        self.assertEqual(state.get_lane_u64_vgpr(1, 6), 0x20)

        self.assertEqual(results[2], 0x30)
        self.assertEqual(state.get_lane_u64_vgpr(2, 6), 0x30)

        self.assertEqual(state.lds.read_qword(common_addr), 0x40)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        addr = 0x100
        val = 0x7777_8888_9999_AAAA
        state.lds.write_qword(addr, val)

        for lane in range(32):
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_u64_vgpr(lane, 2, val)
            state.set_lane_u64_vgpr(lane, 4, 0x1111)
            state.set_lane_u64_vgpr(lane, 6, 0x5555)

        instr = DSCmpstRtnB64Instruction(
            vdst=RegPair(6),
            vaddr=RegSingle(1),
            vdata0=RegPair(2),
            vdata1=RegPair(4),
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_u64_vgpr(lane, 6), 0x5555)
        self.assertEqual(state.lds.read_qword(addr), val)


if __name__ == "__main__":
    unittest.main()



