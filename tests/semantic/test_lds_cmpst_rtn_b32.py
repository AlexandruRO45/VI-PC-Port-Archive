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

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
MIN_NORMAL_F32: int = 0x00800000
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF
ONE_F32: int = 0x3F800000


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0



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


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Evaluation
# ==============================================================================

@dataclass(frozen=True)
class ComparisonStatusFlags:
    """Condition codes and status flags for 32-bit compare-and-swap operations."""
    equal: bool
    less_unsigned: bool
    less_signed: bool
    zero_result: bool
    sign_result: bool
    borrow: bool


def evaluate_cmpst_status_flags(
    old_val: int,
    cmp_val: int,
    new_mem: int,
) -> ComparisonStatusFlags:
    """Evaluate condition codes, comparison flags, and borrow generation.

    Args:
        old_val: Memory value retrieved from LDS.
        cmp_val: Expected value compared against memory.
        new_mem: Memory value after compare-and-swap.

    Returns:
        ComparisonStatusFlags: Evaluated relational and arithmetic status flags.
    """
    u_old = u32(old_val)
    u_cmp = u32(cmp_val)
    u_res = u32(new_mem)

    equal = (u_old == u_cmp)
    less_unsigned = (u_old < u_cmp)
    less_signed = (s32(u_old) < s32(u_cmp))
    zero_result = (u_res == 0)
    sign_result = bool(u_res & 0x80000000)
    borrow = less_unsigned

    return ComparisonStatusFlags(
        equal=equal,
        less_unsigned=less_unsigned,
        less_signed=less_signed,
        zero_result=zero_result,
        sign_result=sign_result,
        borrow=borrow,
    )






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


class TestRegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        # ds_cmpst_rtn_b32 v0, v0, v1, v2
        state = ScalarState(lds_size_dwords=1024)
        addr = 0x60
        initial_mem = 0x1111
        cmp_val = 0x1111
        new_val = 0x2222

        state.lds.write_dword(addr, initial_mem)
        state.set_vgpr(0, addr)    # v0 = addr (also vdst)
        state.set_vgpr(1, cmp_val) # v1 = cmp_val
        state.set_vgpr(2, new_val) # v2 = new_val

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            vdata1=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, new_val)
        self.assertEqual(state.get_vgpr(0), initial_mem) # v0 received old_val
        self.assertEqual(state.get_vgpr(1), cmp_val)     # v1 preserved
        self.assertEqual(state.get_vgpr(2), new_val)     # v2 preserved
        self.assertEqual(state.lds.read_dword(addr), new_val)

    def test_aliasing_vdst_equals_vdata0(self) -> None:
        # ds_cmpst_rtn_b32 v1, v0, v1, v2
        state = ScalarState(lds_size_dwords=1024)
        addr = 0x80
        initial_mem = 0x3333
        cmp_val = 0x3333
        new_val = 0x4444

        state.lds.write_dword(addr, initial_mem)
        state.set_vgpr(0, addr)    # v0 = addr
        state.set_vgpr(1, cmp_val) # v1 = cmp_val (also vdst)
        state.set_vgpr(2, new_val) # v2 = new_val

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(1, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            vdata1=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, new_val)
        self.assertEqual(state.get_vgpr(0), addr)        # v0 preserved
        self.assertEqual(state.get_vgpr(1), initial_mem) # v1 received old_val
        self.assertEqual(state.get_vgpr(2), new_val)     # v2 preserved
        self.assertEqual(state.lds.read_dword(addr), new_val)

    def test_aliasing_vdst_equals_vdata1(self) -> None:
        # ds_cmpst_rtn_b32 v2, v0, v1, v2
        state = ScalarState(lds_size_dwords=1024)
        addr = 0x90
        initial_mem = 0x5555
        cmp_val = 0x5555
        new_val = 0x6666

        state.lds.write_dword(addr, initial_mem)
        state.set_vgpr(0, addr)    # v0 = addr
        state.set_vgpr(1, cmp_val) # v1 = cmp_val
        state.set_vgpr(2, new_val) # v2 = new_val (also vdst)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(2, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            vdata1=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem, swapped = self.interp.execute(instr, state)

        self.assertTrue(swapped)
        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, new_val)
        self.assertEqual(state.get_vgpr(0), addr)        # v0 preserved
        self.assertEqual(state.get_vgpr(1), cmp_val)     # v1 preserved
        self.assertEqual(state.get_vgpr(2), initial_mem) # v2 received old_val
        self.assertEqual(state.lds.read_dword(addr), new_val)

    def test_aliasing_vdata0_equals_vdata1(self) -> None:
        # ds_cmpst_rtn_b32 v3, v0, v1, v1 (cmp_val == new_val)
        state = ScalarState(lds_size_dwords=1024)
        addr = 0xA0
        initial_mem = 0x7777

        state.lds.write_dword(addr, initial_mem)
        state.set_vgpr(0, addr)
        state.set_vgpr(1, initial_mem) # v1 is both cmp_val and new_val

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(3, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            vdata1=RegisterOperand(1, RegType.VGPR),
        )

        old_val, new_mem, swapped = self.interp.execute(instr, state)
        self.assertTrue(swapped)
        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_mem, initial_mem)
        self.assertEqual(state.get_vgpr(3), initial_mem)
        self.assertEqual(state.get_vgpr(1), initial_mem) # v1 preserved

    def test_quadruple_aliasing_all_v0(self) -> None:
        # ds_cmpst_rtn_b32 v0, v0, v0, v0
        state = ScalarState(lds_size_dwords=1024)
        val = 0x40  # Address, cmp_val, and new_val are all 0x40

        state.lds.write_dword(val, val) # Matches val
        state.set_vgpr(0, val)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(0, RegType.VGPR),
            vdata1=RegisterOperand(0, RegType.VGPR),
        )

        old_val, new_mem, swapped = self.interp.execute(instr, state)
        self.assertTrue(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, val)
        self.assertEqual(state.get_vgpr(0), val)
        self.assertEqual(state.lds.read_dword(val), val)

    def test_unreferenced_registers_preservation(self) -> None:
        state = ScalarState(lds_size_dwords=1024)
        state.set_vgpr(1, 0x100) # vaddr
        state.set_vgpr(2, 0x10)  # vdata0
        state.set_vgpr(3, 0x20)  # vdata1

        for r in range(4, 25):
            state.set_vgpr(r, 0x5000 + r)
        for s in range(12):
            state.set_sgpr(s, 0x6000 + s)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )
        self.interp.execute(instr, state)

        for r in range(4, 25):
            self.assertEqual(state.get_vgpr(r), 0x5000 + r)
        for s in range(12):
            self.assertEqual(state.get_sgpr(s), 0x6000 + s)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds_size_dwords=1024)
        simd_interp = SIMDInterpreter()

        # Each lane has vdst == vaddr: v0 is addr and dst
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(addr, 0x100 + lane)
            state.set_lane_vgpr(lane, 0, addr)       # v0 = addr and dst
            state.set_lane_vgpr(lane, 1, 0x100 + lane) # v1 = cmp_val (match)
            state.set_lane_vgpr(lane, 2, 0x700 + lane) # v2 = new_val

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
            vdata1=RegisterOperand(2, RegType.VGPR),
        )

        results = simd_interp.execute(instr, state)

        for lane in range(32):
            self.assertEqual(results[lane], 0x100 + lane)
            self.assertEqual(state.get_lane_vgpr(lane, 0), 0x100 + lane) # Received old_val
            self.assertEqual(state.get_lane_vgpr(lane, 1), 0x100 + lane) # Preserved
            self.assertEqual(state.get_lane_vgpr(lane, 2), 0x700 + lane) # Preserved
            self.assertEqual(state.lds.read_dword(lane * 4), 0x700 + lane)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_equality_flags(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(0x1000, 0x1000, 0x2000)
        self.assertTrue(swapped)
        flags = evaluate_cmpst_status_flags(old_val, 0x1000, new_mem)

        self.assertTrue(flags.equal)
        self.assertFalse(flags.less_unsigned)
        self.assertFalse(flags.less_signed)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.zero_result)
        self.assertFalse(flags.sign_result)

    def test_less_unsigned_and_borrow(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(10, 20, 99)
        self.assertFalse(swapped)
        flags = evaluate_cmpst_status_flags(old_val, 20, new_mem)

        self.assertFalse(flags.equal)
        self.assertTrue(flags.less_unsigned)
        self.assertTrue(flags.less_signed)
        self.assertTrue(flags.borrow)

    def test_greater_unsigned_no_borrow(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(50, 20, 99)
        self.assertFalse(swapped)
        flags = evaluate_cmpst_status_flags(old_val, 20, new_mem)

        self.assertFalse(flags.equal)
        self.assertFalse(flags.less_unsigned)
        self.assertFalse(flags.less_signed)
        self.assertFalse(flags.borrow)

    def test_signed_vs_unsigned_divergence(self) -> None:
        # 0xFFFFFFFF (-1 signed, 4294967295 unsigned) vs 0x00000001 (+1)
        old_val = 0xFFFFFFFF
        cmp_val = 0x00000001
        _, new_mem, _ = ref_ds_cmpst_rtn_b32(old_val, cmp_val, 0)
        flags = evaluate_cmpst_status_flags(old_val, cmp_val, new_mem)

        self.assertFalse(flags.equal)
        self.assertFalse(flags.less_unsigned) # Unsigned: 0xFFFFFFFF > 1
        self.assertFalse(flags.borrow)
        self.assertTrue(flags.less_signed)    # Signed: -1 < +1

    def test_zero_result_flag(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(42, 42, 0)
        self.assertTrue(swapped)
        self.assertEqual(new_mem, 0)
        flags = evaluate_cmpst_status_flags(old_val, 42, new_mem)
        self.assertTrue(flags.zero_result)
        self.assertFalse(flags.sign_result)

    def test_negative_sign_result_flag(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(0, 0, 0x80000000)
        self.assertTrue(swapped)
        self.assertEqual(new_mem, 0x80000000)
        flags = evaluate_cmpst_status_flags(old_val, 0, new_mem)
        self.assertTrue(flags.sign_result)
        self.assertFalse(flags.zero_result)


class TestIEEE754Boundaries(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros_bitwise_inequality(self) -> None:
        # In IEEE 754, +0.0 == -0.0, but in bitwise CAS: 0x00000000 != 0x80000000
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(POS_ZERO_F32, NEG_ZERO_F32, 0x1234)
        self.assertFalse(swapped)
        self.assertEqual(old_val, POS_ZERO_F32)
        self.assertEqual(new_mem, POS_ZERO_F32)

    def test_matching_signed_zero(self) -> None:
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(NEG_ZERO_F32, NEG_ZERO_F32, POS_ZERO_F32)
        self.assertTrue(swapped)
        self.assertEqual(old_val, NEG_ZERO_F32)
        self.assertEqual(new_mem, POS_ZERO_F32)

    def test_subnormal_no_ftz_swap(self) -> None:
        self.assertTrue(is_f32_subnormal(MIN_SUBNORMAL_F32))
        self.assertTrue(is_f32_subnormal(MAX_SUBNORMAL_F32))

        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(MIN_SUBNORMAL_F32, MIN_SUBNORMAL_F32, MAX_SUBNORMAL_F32)
        self.assertTrue(swapped)
        self.assertEqual(old_val, MIN_SUBNORMAL_F32)
        self.assertEqual(new_mem, MAX_SUBNORMAL_F32)
        self.assertTrue(is_f32_subnormal(new_mem))

    def test_signaling_nan_preserved_no_quietening(self) -> None:
        self.assertTrue(is_f32_nan(SNAN_F32))
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(SNAN_F32, SNAN_F32, 0x5555)
        self.assertTrue(swapped)
        self.assertEqual(old_val, SNAN_F32)
        self.assertEqual(new_mem, 0x5555)
        # Ensure SNAN returned to vdst preserves signaling bit (does not quieten to QNAN)
        self.assertNotEqual(old_val, QNAN_F32)

    def test_nan_payload_mismatch(self) -> None:
        # Different NaN bit patterns must not match
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(QNAN_F32, SNAN_F32, 0x9999)
        self.assertFalse(swapped)
        self.assertEqual(old_val, QNAN_F32)
        self.assertEqual(new_mem, QNAN_F32)

    def test_infinities_distinct(self) -> None:
        # Positive and negative infinity are distinct bit patterns
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b32(POS_INF_F32, NEG_INF_F32, 0)
        self.assertFalse(swapped)
        self.assertEqual(old_val, POS_INF_F32)
        self.assertEqual(new_mem, POS_INF_F32)

    def test_subnormal_in_scalar_interpreter(self) -> None:
        state = ScalarState(lds_size_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x20
        state.lds.write_dword(addr, MIN_SUBNORMAL_F32)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, MIN_SUBNORMAL_F32)
        state.set_vgpr(3, MAX_SUBNORMAL_F32)

        instr = DSCmpstRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )

        old_val, new_mem, swapped = interp.execute(instr, state)
        self.assertTrue(swapped)
        self.assertEqual(old_val, MIN_SUBNORMAL_F32)
        self.assertEqual(new_mem, MAX_SUBNORMAL_F32)
        self.assertEqual(state.get_vgpr(0), MIN_SUBNORMAL_F32)
        self.assertEqual(state.lds.read_dword(addr), MAX_SUBNORMAL_F32)


if __name__ == "__main__":
    unittest.main()






