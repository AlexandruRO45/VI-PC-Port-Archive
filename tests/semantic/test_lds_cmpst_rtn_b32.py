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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising register files and LDS memory."""

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

    def get_reg(self, reg: RegisterOperand) -> int:
        if reg.reg_type == RegType.VGPR:
            return self.get_vgpr(reg.index)
        return self.get_sgpr(reg.index)

    def set_reg(self, reg: RegisterOperand, val: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_vgpr(reg.index, val)
        else:
            self.set_sgpr(reg.index, val)

    def resolve_operand(self, op: Operand) -> int:
        if isinstance(op, Immediate):
            return op.val
        return self.get_reg(op)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_cmpst_rtn_b32."""

    def execute(self, instr: DSCmpstRtnInstruction, state: ScalarState) -> Tuple[int, int, bool]:
        """Execute a single ds_cmpst_rtn_b32 instruction on scalar state.

        Returns:
            Tuple[int, int, bool]: (old_val, new_mem, swapped)
        """
        # 1. Latch inputs prior to any register writeback
        vaddr = state.get_reg(instr.vaddr)
        cmp_val = state.resolve_operand(instr.vdata0)
        new_val = state.resolve_operand(instr.vdata1)

        # 2. Perform atomic LDS compare-and-swap
        old_val, new_mem, swapped = state.lds.atomic_cmpst_rtn_b32(
            vaddr=vaddr,
            cmp_val=cmp_val,
            new_val=new_val,
            offset=instr.offset,
        )

        # 3. Write pre-operation original value back to destination register
        state.set_reg(instr.vdst, old_val)

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

    def resolve_lane_operand(self, op: Operand, lane: int) -> int:
        if isinstance(op, Immediate):
            return op.val
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        return self.get_lane_vgpr(lane, op.index)


class SIMDInterpreter:
    """SIMD multi-lane instruction interpreter for ds_cmpst_rtn_b32."""

    def execute(
        self,
        instr: DSCmpstRtnInstruction,
        state: SIMDState,
    ) -> List[Optional[int]]:
        """Execute ds_cmpst_rtn_b32 across SIMD lanes with EXEC masking and collision serialization.

        Returns:
            List[Optional[int]]: Per-lane returned old_val for active lanes, or None for inactive lanes.
        """
        results: List[Optional[int]] = [None] * state.num_lanes

        # 1. Latch per-lane inputs prior to any memory updates
        lane_inputs: list[tuple[int, int, int, int, int] | None] = [None] * state.num_lanes
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue
            vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
            cmp_val = state.resolve_lane_operand(instr.vdata0, lane)
            new_val = state.resolve_lane_operand(instr.vdata1, lane)
            eff_addr = calc_lds_effective_address(vaddr, instr.offset)
            dword_idx = calc_lds_dword_index(eff_addr, state.lds.size_dwords)
            lane_inputs[lane] = (lane, dword_idx, eff_addr, cmp_val, new_val)

        # 2. Execute atomic operations in ascending lane order (hardware serialization order)
        for lane in range(state.num_lanes):
            entry = lane_inputs[lane]
            if entry is None:
                continue
            _, dword_idx, eff_addr, cmp_val, new_val = entry
            curr_val = state.lds.memory[dword_idx]
            old_val, updated_mem, swapped = ref_ds_cmpst_rtn_b32(curr_val, cmp_val, new_val)
            state.lds.memory[dword_idx] = updated_mem
            results[lane] = old_val

        # 3. Writeback to destination VGPRs for active lanes
        for lane in range(state.num_lanes):
            if results[lane] is not None:
                state.set_lane_vgpr(lane, instr.vdst.index, results[lane])  # type: ignore

        return results





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


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState(lds_size_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_cmpst_match_execution(self) -> None:
        addr = 0x40
        self.state.lds.write_dword(addr, 0x1234)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0x1234)      # cmp_val matches
        self.state.set_vgpr(3, 0x9999)      # new_val
        self.state.set_vgpr(0, 0xDEADBEEF)  # sentinel

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, 0x1234)
        self.assertEqual(new_mem, 0x9999)
        self.assertEqual(self.state.get_vgpr(0), 0x1234)  # vdst received old_val
        self.assertEqual(self.state.lds.read_dword(addr), 0x9999)

    def test_scalar_cmpst_mismatch_execution(self) -> None:
        addr = 0x40
        self.state.lds.write_dword(addr, 0x1234)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0x5555)      # cmp_val does not match
        self.state.set_vgpr(3, 0x9999)      # new_val
        self.state.set_vgpr(0, 0xDEADBEEF)  # sentinel

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertFalse(swapped)
        self.assertEqual(old_val, 0x1234)
        self.assertEqual(new_mem, 0x1234)
        self.assertEqual(self.state.get_vgpr(0), 0x1234)  # vdst still gets old_val
        self.assertEqual(self.state.lds.read_dword(addr), 0x1234)

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x100
        offset = 0x10
        eff_addr = base_addr + offset

        self.state.lds.write_dword(eff_addr, 0x7777)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, 0x7777)
        self.state.set_vgpr(3, 0x8888)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
            offset=offset,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, 0x7777)
        self.assertEqual(new_mem, 0x8888)
        self.assertEqual(self.state.get_vgpr(0), 0x7777)
        self.assertEqual(self.state.lds.read_dword(eff_addr), 0x8888)

    def test_scalar_execution_with_immediate_operands(self) -> None:
        addr = 0x200
        self.state.lds.write_dword(addr, 50)
        self.state.set_vgpr(1, addr)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=Immediate(50),
            vdata1=Immediate(100),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_mem, 100)
        self.assertEqual(self.state.get_vgpr(0), 50)
        self.assertEqual(self.state.lds.read_dword(addr), 100)

    def test_scalar_unaligned_byte_address(self) -> None:
        self.state.lds.write_dword(0x300, 42)
        self.state.set_vgpr(1, 0x302)  # Unaligned address: low 2 bits dropped -> 0x300

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=Immediate(42),
            vdata1=Immediate(99),
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)
        self.assertTrue(swapped)
        self.assertEqual(old_val, 42)
        self.assertEqual(new_mem, 99)
        self.assertEqual(self.state.lds.read_dword(0x300), 99)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        # Initialize: even lanes match and swap, odd lanes mismatch
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(addr, 100 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            # Even lanes cmp matches, odd lanes cmp does not match
            cmp_val = (100 + lane) if (lane % 2 == 0) else 9999
            state.set_lane_vgpr(lane, 2, cmp_val)
            state.set_lane_vgpr(lane, 3, 500 + lane)
            state.set_lane_vgpr(lane, 0, 0xDEADBEEF)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertEqual(results[lane], 100 + lane)
            self.assertEqual(state.get_lane_vgpr(lane, 0), 100 + lane)
            if lane % 2 == 0:
                self.assertEqual(state.lds.read_dword(lane * 4), 500 + lane)
            else:
                self.assertEqual(state.lds.read_dword(lane * 4), 100 + lane)

    def test_wave64_divergent_exec_mask(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        sentinel = 0xCAFEBABE
        for lane in range(64):
            addr = lane * 4
            state.lds.write_dword(addr, 200)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 200) # Match
            state.set_lane_vgpr(lane, 3, 300) # New val
            state.set_lane_vgpr(lane, 0, sentinel)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
            offset=0,
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            if lane % 2 == 0:
                # Active
                self.assertEqual(results[lane], 200)
                self.assertEqual(state.get_lane_vgpr(lane, 0), 200)
                self.assertEqual(state.lds.read_dword(lane * 4), 300)
            else:
                # Inactive
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(lane * 4), 200)

    def test_multi_lane_collision_chained_cas(self) -> None:
        # 4 lanes all target address 0x80 in sequential chained CAS order
        common_addr = 0x80
        state = SIMDState(num_lanes=32, exec_mask=0xF, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        state.lds.write_dword(common_addr, 10)

        # Lane 0: expects 10, writes 20
        # Lane 1: expects 20, writes 30
        # Lane 2: expects 30, writes 40
        # Lane 3: expects 40, writes 50
        chain = [(10, 20), (20, 30), (30, 40), (40, 50)]
        for lane, (cmp_val, new_val) in enumerate(chain):
            state.set_lane_vgpr(lane, 1, common_addr)
            state.set_lane_vgpr(lane, 2, cmp_val)
            state.set_lane_vgpr(lane, 3, new_val)
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        self.assertEqual(results[0], 10)
        self.assertEqual(state.get_lane_vgpr(0, 0), 10)

        self.assertEqual(results[1], 20)
        self.assertEqual(state.get_lane_vgpr(1, 0), 20)

        self.assertEqual(results[2], 30)
        self.assertEqual(state.get_lane_vgpr(2, 0), 30)

        self.assertEqual(results[3], 40)
        self.assertEqual(state.get_lane_vgpr(3, 0), 40)

        self.assertEqual(state.lds.read_dword(common_addr), 50)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0, lds_size_dwords=1024)
        interp = SIMDInterpreter()

        addr = 0x100
        state.lds.write_dword(addr, 0x1111)
        for lane in range(32):
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 0x1111)
            state.set_lane_vgpr(lane, 3, 0x2222)
            state.set_lane_vgpr(lane, 0, 0x7777)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_vgpr(lane, 0), 0x7777)
        self.assertEqual(state.lds.read_dword(addr), 0x1111)


if __name__ == "__main__":
    unittest.main()



