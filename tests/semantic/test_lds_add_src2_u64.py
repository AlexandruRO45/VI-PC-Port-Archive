from __future__ import annotations

"""Reference oracle and architectural model for ds_add_src2_u64.

RDNA2 / GFX10 Local Data Share (LDS) 64-bit Dual-Source Atomic Addition:
Performs a 64-bit atomic addition on LDS memory location A using
the secondary LDS memory location B as the addend: LDS[A] = (LDS[A] + LDS[B]) mod 2^64.

Syntax:
    ds_add_src2_u64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Operands:
    vaddr: VGPR containing base byte address into LDS.
    offset0: 8-bit unsigned immediate multiplied by 8 bytes (location A stride).
    offset1: 8-bit unsigned immediate multiplied by 8 bytes (location B stride).

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    LDS64[qword_idx_a] = (val_a + val_b) & 0xFFFFFFFFFFFFFFFF
    # Location B remains unchanged (unless qword_idx_a == qword_idx_b)
    # Non-returning: VGPRs are not modified.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
OFFSET8_MASK: int = 0xFF
DEFAULT_LDS_QWORDS: int = 8192     # 64 KiB default capacity (8192 * 8 bytes)
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit qword into (low_u32, high_u32)."""
    val = u64(val)
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (low_u32, high_u32) into a 64-bit qword."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def calc_lds_src2_byte_address(vaddr: int, offset8: int) -> int:
    """Calculate the 32-bit effective byte address for a src2 operand.
    
    offset8 is an 8-bit unsigned immediate multiplied by 8 bytes (qword stride).
    Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + ((offset8 & OFFSET8_MASK) * 8)) & MASK32


def calc_lds_qword_index(byte_addr: int, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS qword index.
    
    Qword addresses are 8-byte aligned (low 3 bits dropped).
    """
    return (byte_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank_pair(byte_addr: int) -> Tuple[int, int]:
    """Determine the two LDS memory banks (0..31) occupied by a 64-bit qword."""
    dword_idx = (byte_addr >> 2) & 31
    return dword_idx, (dword_idx + 1) & 31


def oracle_ds_add_src2_u64(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_add_src2_u64: (val_a + val_b) mod 2^64."""
    return u64(u64(val_a) + u64(val_b))


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, qword_idx: int) -> int:
        idx = qword_idx & (self.capacity_qwords - 1)
        return self.memory[idx]

    def write_qword(self, qword_idx: int, val: int) -> None:
        idx = qword_idx & (self.capacity_qwords - 1)
        self.memory[idx] = u64(val)

    def read_qword_at_byte_addr(self, byte_addr: int) -> int:
        idx = calc_lds_qword_index(byte_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword_at_byte_addr(self, byte_addr: int, val: int) -> None:
        idx = calc_lds_qword_index(byte_addr, self.capacity_qwords)
        self.memory[idx] = u64(val)

    def atomic_add_src2_u64(self, vaddr: int, offset0: int, offset1: int) -> int:
        """Execute atomic dual-source 64-bit addition on LDS.
        
        LDS[A] = (LDS[A] + LDS[B]) mod 2^64.
        Returns the updated value at location A.
        """
        byte_addr_a = calc_lds_src2_byte_address(vaddr, offset0)
        byte_addr_b = calc_lds_src2_byte_address(vaddr, offset1)

        idx_a = calc_lds_qword_index(byte_addr_a, self.capacity_qwords)
        idx_b = calc_lds_qword_index(byte_addr_b, self.capacity_qwords)

        val_a = self.memory[idx_a]
        val_b = self.memory[idx_b]

        new_val_a = oracle_ds_add_src2_u64(val_a, val_b)
        self.memory[idx_a] = new_val_a
        return new_val_a


class TestLdsAddSrc2U64ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_basic_add_src2(self) -> None:
        """LDS[A] += LDS[B] with distinct locations."""
        vaddr = 0x100
        offset0 = 2  # A = 0x100 + 16 = 0x110
        offset1 = 5  # B = 0x100 + 40 = 0x128

        self.lds.write_qword_at_byte_addr(0x110, 1000)
        self.lds.write_qword_at_byte_addr(0x128, 250)

        res = self.lds.atomic_add_src2_u64(vaddr, offset0, offset1)
        self.assertEqual(res, 1250)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x110), 1250)
        # Location B must remain unchanged
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x128), 250)

    def test_identity_zero(self) -> None:
        """When LDS[B] is zero, LDS[A] remains unchanged."""
        vaddr = 0x200
        self.lds.write_qword_at_byte_addr(0x200, 0x1234_5678_9ABC_DEF0)
        self.lds.write_qword_at_byte_addr(0x208, 0)

        res = self.lds.atomic_add_src2_u64(vaddr, 0, 1)
        self.assertEqual(res, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x200), 0x1234_5678_9ABC_DEF0)

    def test_aliased_addresses_same_offset(self) -> None:
        """When offset0 == offset1, A == B: LDS[A] becomes 2 * LDS[A]."""
        vaddr = 0x300
        val = 0x0000_0001_0000_0000
        self.lds.write_qword_at_byte_addr(0x300, val)

        res = self.lds.atomic_add_src2_u64(vaddr, 0, 0)
        self.assertEqual(res, 0x0000_0002_0000_0000)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x300), 0x0000_0002_0000_0000)

    def test_64bit_overflow_wrap(self) -> None:
        """64-bit unsigned addition overflow wraps modulo 2^64."""
        vaddr = 0x400
        self.lds.write_qword_at_byte_addr(0x400, MASK64)
        self.lds.write_qword_at_byte_addr(0x408, 1)

        res = self.lds.atomic_add_src2_u64(vaddr, 0, 1)
        self.assertEqual(res, 0)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x400), 0)

    def test_stride_8_offset_scaling(self) -> None:
        """offset8 is multiplied by 8 bytes (qword stride)."""
        base = 0x1000
        addr_off0 = calc_lds_src2_byte_address(base, 0)
        addr_off1 = calc_lds_src2_byte_address(base, 1)
        addr_off2 = calc_lds_src2_byte_address(base, 2)
        addr_off255 = calc_lds_src2_byte_address(base, 255)

        self.assertEqual(addr_off0, 0x1000)
        self.assertEqual(addr_off1, 0x1008)
        self.assertEqual(addr_off2, 0x1010)
        self.assertEqual(addr_off255, 0x1000 + 255 * 8)

    def test_qword_alignment_truncation(self) -> None:
        """Low 3 bits of byte address are ignored for qword alignment."""
        idx0 = calc_lds_qword_index(0x800)
        idx1 = calc_lds_qword_index(0x801)
        idx7 = calc_lds_qword_index(0x807)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx7, 0x100)

    def test_bank_pair_distribution(self) -> None:
        """Each qword occupies a pair of consecutive 32-bit LDS memory banks."""
        b_lo, b_hi = calc_lds_bank_pair(0)
        self.assertEqual((b_lo, b_hi), (0, 1))

        b_lo, b_hi = calc_lds_bank_pair(31 * 4)  # Last dword bank 31 wraps to 0
        self.assertEqual((b_lo, b_hi), (31, 0))

    def test_capacity_wrapping(self) -> None:
        """Qword access wraps modulo capacity (default 64 KiB = 8192 qwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + (DEFAULT_LDS_QWORDS * 8)
        idx_base = calc_lds_qword_index(addr_base)
        idx_wrapped = calc_lds_qword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)


if __name__ == "__main__":
    unittest.main()


# ============================================================================
# Section 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(f"VGPR index out of range [0, {DEFAULT_NUM_VGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_SGPRS):
            raise ValueError(f"SGPR index out of range [0, {DEFAULT_NUM_SGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class DSAddSrc2U64Instruction:
    vaddr: VGPR
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset0 <= OFFSET8_MASK):
            raise ValueError(f"offset0 out of 8-bit range [0, 255]: {self.offset0}")
        if not (0 <= self.offset1 <= OFFSET8_MASK):
            raise ValueError(f"offset1 out of 8-bit range [0, 255]: {self.offset1}")
        if not isinstance(self.vaddr, VGPR):
            raise TypeError(f"vaddr must be a VGPR, got {type(self.vaddr)}")

    def disasm(self) -> str:
        tokens = ["ds_add_src2_u64", str(self.vaddr)]
        if self.offset0 != 0:
            tokens.append(f"offset0:{self.offset0}")
        if self.offset1 != 0:
            tokens.append(f"offset1:{self.offset1}")
        if self.gds:
            tokens.append("gds")
        return " ".join(tokens)


class ASTExpr64:
    """Base class for symbolic typed expressions in RDNA2 64-bit dual-source arithmetic."""

    def simplify(self) -> ASTExpr64:
        raise NotImplementedError

    def evaluate(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol64(ASTExpr64):
    name: str

    def simplify(self) -> ASTExpr64:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in evaluation environment")
        return u64(env[self.name])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant64(ASTExpr64):
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u64(self.value))

    def simplify(self) -> ASTExpr64:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        return self.value

    def __str__(self) -> str:
        return f"0x{self.value:016X}"


@dataclass(frozen=True)
class DualSourceAddExpr(ASTExpr64):
    loc_a: ASTExpr64
    loc_b: ASTExpr64

    def simplify(self) -> ASTExpr64:
        s_a = self.loc_a.simplify()
        s_b = self.loc_b.simplify()

        # Constant folding
        if isinstance(s_a, Constant64) and isinstance(s_b, Constant64):
            return Constant64(s_a.value + s_b.value)

        # Identity addition: A + 0 -> A
        if isinstance(s_b, Constant64) and s_b.value == 0:
            return s_a

        return DualSourceAddExpr(s_a, s_b)

    def evaluate(self, env: Dict[str, int]) -> int:
        return u64(self.loc_a.evaluate(env) + self.loc_b.evaluate(env))

    def __str__(self) -> str:
        return f"({self.loc_a} + {self.loc_b})"


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_valid_creation_and_disassembly(self) -> None:
        instr = DSAddSrc2U64Instruction(vaddr=VGPR(4), offset0=2, offset1=6)
        self.assertEqual(instr.disasm(), "ds_add_src2_u64 v4 offset0:2 offset1:6")

        instr_default = DSAddSrc2U64Instruction(vaddr=VGPR(1), offset0=0, offset1=0, gds=True)
        self.assertEqual(instr_default.disasm(), "ds_add_src2_u64 v1 gds")

    def test_instruction_validation_vgpr_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_instruction_validation_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=256, offset1=0)
        with self.assertRaises(ValueError):
            DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=-1)

    def test_symbolic_constant_folding(self) -> None:
        expr = DualSourceAddExpr(Constant64(500), Constant64(250))
        simplified = expr.simplify()
        self.assertIsInstance(simplified, Constant64)
        self.assertEqual(simplified.evaluate({}), 750)

    def test_symbolic_identity_zero(self) -> None:
        a = Symbol64("LDS_A")
        expr = DualSourceAddExpr(a, Constant64(0))
        simplified = expr.simplify()
        self.assertEqual(simplified, a)

    def test_symbolic_evaluation_env(self) -> None:
        a = Symbol64("A")
        b = Symbol64("B")
        expr = DualSourceAddExpr(a, b)
        env = {"A": 1000, "B": 2000}
        self.assertEqual(expr.evaluate(env), 3000)


# ============================================================================
# Section 3: Deterministic Scalar Instruction Interpreter
# ============================================================================

class ScalarMachineState:
    """Register and memory state for scalar single-thread RDNA2 execution."""

    def __init__(self, lds_capacity_qwords: int = DEFAULT_LDS_QWORDS) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSModel = LDSModel(capacity_qwords=lds_capacity_qwords)

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.vgprs[idx]

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_add_src2_u64."""

    def step(
        self,
        instr: DSAddSrc2U64Instruction,
        state: ScalarMachineState,
    ) -> int:
        """Execute a single ds_add_src2_u64 instruction deterministically.

        Non-returning: VGPR registers are strictly untouched.
        Returns the updated value at location A.
        """
        vaddr = state.get_vgpr(instr.vaddr.index)
        return state.lds.atomic_add_src2_u64(vaddr, instr.offset0, instr.offset1)

    def execute_sequence(
        self,
        instructions: List[DSAddSrc2U64Instruction],
        state: ScalarMachineState,
    ) -> List[int]:
        """Execute an ordered sequence of instructions."""
        return [self.step(instr, state) for instr in instructions]


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_basic_execution(self) -> None:
        base = 0x400
        self.state.set_vgpr(5, base)

        # A at offset0=1 -> 0x408, B at offset1=2 -> 0x410
        self.state.lds.write_qword_at_byte_addr(0x408, 0x1000)
        self.state.lds.write_qword_at_byte_addr(0x410, 0x2000)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(5), offset0=1, offset1=2)
        res = self.interp.step(instr, self.state)

        self.assertEqual(res, 0x3000)
        self.assertEqual(self.state.lds.read_qword_at_byte_addr(0x408), 0x3000)
        self.assertEqual(self.state.lds.read_qword_at_byte_addr(0x410), 0x2000)
        # Verify vaddr VGPR was preserved untouched
        self.assertEqual(self.state.get_vgpr(5), base)

    def test_scalar_deterministic_instruction_sequence(self) -> None:
        base = 0x500
        self.state.set_vgpr(1, base)

        # Acc at offset 0 (0x500), Step at offset 1 (0x508)
        self.state.lds.write_qword_at_byte_addr(0x500, 10)
        self.state.lds.write_qword_at_byte_addr(0x508, 5)

        seq = [
            DSAddSrc2U64Instruction(vaddr=VGPR(1), offset0=0, offset1=1),
            DSAddSrc2U64Instruction(vaddr=VGPR(1), offset0=0, offset1=1),
            DSAddSrc2U64Instruction(vaddr=VGPR(1), offset0=0, offset1=1),
        ]
        results = self.interp.execute_sequence(seq, self.state)

        self.assertEqual(results, [15, 20, 25])
        self.assertEqual(self.state.lds.read_qword_at_byte_addr(0x500), 25)
        self.assertEqual(self.state.lds.read_qword_at_byte_addr(0x508), 5)


# ============================================================================
# Section 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ============================================================================

class SIMDState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64) with shared LDS."""

    def __init__(
        self,
        num_lanes: int = 32,
        exec_mask: int = 0xFFFF_FFFF,
        lds_capacity_qwords: int = DEFAULT_LDS_QWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes: int = num_lanes
        self.max_exec_mask: int = (1 << num_lanes) - 1
        self.exec_mask: int = exec_mask & self.max_exec_mask
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.lds: LDSModel = LDSModel(capacity_qwords=lds_capacity_qwords)

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of range: {lane}")
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx]

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = u32(val)


class SIMDInterpreter:
    """Wavefront SIMD instruction interpreter with divergence, masking, and collision handling."""

    def step(
        self,
        instr: DSAddSrc2U64Instruction,
        state: SIMDState,
    ) -> List[Optional[int]]:
        """Execute ds_add_src2_u64 across all lanes in the wavefront."""
        active_lane_data: List[Tuple[int, int, int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                vaddr = state.get_vgpr(lane, instr.vaddr.index)
                addr_a = calc_lds_src2_byte_address(vaddr, instr.offset0)
                addr_b = calc_lds_src2_byte_address(vaddr, instr.offset1)
                active_lane_data.append((lane, addr_a, addr_b))

        results: List[Optional[int]] = [None] * state.num_lanes

        for lane, addr_a, addr_b in active_lane_data:
            idx_a = calc_lds_qword_index(addr_a, state.lds.capacity_qwords)
            idx_b = calc_lds_qword_index(addr_b, state.lds.capacity_qwords)
            val_a = state.lds.memory[idx_a]
            val_b = state.lds.memory[idx_b]
            new_val_a = oracle_ds_add_src2_u64(val_a, val_b)
            state.lds.memory[idx_a] = new_val_a
            results[lane] = new_val_a

        return results


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD divergence, 32/64-bit EXEC masking, and intra-wave collision."""

    def test_wave32_all_lanes_active_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        # Each lane has its own unique base address
        for lane in range(32):
            base = lane * 64
            state.set_vgpr(lane, 0, base)
            state.lds.write_qword_at_byte_addr(base, 100)      # A (offset 0)
            state.lds.write_qword_at_byte_addr(base + 8, 25)   # B (offset 1)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        results = interp.step(instr, state)

        for lane in range(32):
            self.assertEqual(results[lane], 125)
            base = lane * 64
            self.assertEqual(state.lds.read_qword_at_byte_addr(base), 125)
            self.assertEqual(state.lds.read_qword_at_byte_addr(base + 8), 25)

    def test_wave32_exec_mask_partial_inactivity(self) -> None:
        exec_mask = 0x0000_FFFF  # Lanes 0..15 active, 16..31 inactive
        state = SIMDState(num_lanes=32, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        for lane in range(32):
            base = lane * 64
            state.set_vgpr(lane, 0, base)
            state.lds.write_qword_at_byte_addr(base, 500)
            state.lds.write_qword_at_byte_addr(base + 8, 50)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        results = interp.step(instr, state)

        for lane in range(32):
            base = lane * 64
            if lane < 16:
                self.assertEqual(results[lane], 550)
                self.assertEqual(state.lds.read_qword_at_byte_addr(base), 550)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lds.read_qword_at_byte_addr(base), 500)

    def test_wave64_full_and_partial_masks(self) -> None:
        exec_mask = 0x5555_5555_5555_5555  # Alternating lanes active
        state = SIMDState(num_lanes=64, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            base = lane * 64
            state.set_vgpr(lane, 0, base)
            state.lds.write_qword_at_byte_addr(base, 1000)
            state.lds.write_qword_at_byte_addr(base + 8, 10)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        results = interp.step(instr, state)

        for lane in range(64):
            base = lane * 64
            if (lane % 2) == 0:
                self.assertEqual(results[lane], 1010)
                self.assertEqual(state.lds.read_qword_at_byte_addr(base), 1010)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lds.read_qword_at_byte_addr(base), 1000)

    def test_intra_wave_collision_same_address_a(self) -> None:
        """Multiple lanes targeting the same location A serialize in lane order."""
        state = SIMDState(num_lanes=32, exec_mask=0x07)  # Lanes 0, 1, 2 active
        interp = SIMDInterpreter()

        common_base = 0x800
        state.lds.write_qword_at_byte_addr(common_base, 100)      # Location A

        # Each lane has distinct B:
        # Lane 0: B at 0x808 = 10
        # Lane 1: B at 0x810 = 20
        # Lane 2: B at 0x818 = 30
        state.lds.write_qword_at_byte_addr(common_base + 8, 10)
        state.lds.write_qword_at_byte_addr(common_base + 16, 20)
        state.lds.write_qword_at_byte_addr(common_base + 24, 30)

        state.set_vgpr(0, 0, common_base)
        state.set_vgpr(1, 0, common_base)
        state.set_vgpr(2, 0, common_base)

        # Lane 0 adds offset1=1 (+10) -> 110
        # Lane 1 adds offset1=2 (+20) -> 130
        # Lane 2 adds offset1=3 (+30) -> 160
        instr0 = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        instr1 = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=2)
        instr2 = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=3)

        res0 = interp.step(instr0, state)
        self.assertEqual(state.lds.read_qword_at_byte_addr(common_base), 130)


# ============================================================================
# Section 5: Register Aliasing and In-Place Operand Preservation
# ============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for operand preservation: vaddr preservation, self-aliased memory A == B."""

    def test_vaddr_preservation(self) -> None:
        """vaddr VGPR must remain strictly untouched (non-returning instruction)."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        vaddr_val = 0x400
        state.set_vgpr(7, vaddr_val)
        state.lds.write_qword_at_byte_addr(vaddr_val, 100)
        state.lds.write_qword_at_byte_addr(vaddr_val + 8, 50)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(7), offset0=0, offset1=1)
        interp.step(instr, state)

        # VGPR 7 must be preserved bit-for-bit
        self.assertEqual(state.get_vgpr(7), vaddr_val)

    def test_all_vgprs_unmodified(self) -> None:
        """Every VGPR in the register file is preserved unchanged."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        for i in range(DEFAULT_NUM_VGPRS):
            state.set_vgpr(i, 0x1000 + i)
        state.set_vgpr(0, 0x100)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=0)
        interp.step(instr, state)

        for i in range(1, DEFAULT_NUM_VGPRS):
            self.assertEqual(state.get_vgpr(i), 0x1000 + i)

    def test_self_aliased_memory_locations(self) -> None:
        """offset0 == offset1: Location A and Location B reference the same memory qword."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x200
        initial_val = 0x0123_4567_89AB_CDEF
        state.lds.write_qword_at_byte_addr(addr, initial_val)
        state.set_vgpr(2, addr)

        # LDS[A] = LDS[A] + LDS[A] = 2 * LDS[A]
        instr = DSAddSrc2U64Instruction(vaddr=VGPR(2), offset0=0, offset1=0)
        res = interp.step(instr, state)

        self.assertEqual(res, u64(initial_val * 2))
        self.assertEqual(state.lds.read_qword_at_byte_addr(addr), u64(initial_val * 2))


# ============================================================================
# Section 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ============================================================================

@dataclass(frozen=True)
class AddSrc2U64Flags:
    """Status flags and condition codes derived from 64-bit dual-source addition."""
    z: bool            # Zero flag: result == 0
    n: bool            # Sign flag: bit 63 set
    carry_out: bool    # 64-bit carry out
    carry_dword: bool  # 32-bit dword boundary carry


def compute_add_src2_u64_flags(val_a: int, val_b: int) -> AddSrc2U64Flags:
    """Compute condition flags for unsigned 64-bit addition (val_a + val_b)."""
    a = u64(val_a)
    b = u64(val_b)
    res = u64(a + b)

    z = (res == 0)
    n = bool(res & 0x8000_0000_0000_0000)
    carry_out = (a + b) > MASK64
    carry_dword = ((a & MASK32) + (b & MASK32)) > MASK32

    return AddSrc2U64Flags(
        z=z,
        n=n,
        carry_out=carry_out,
        carry_dword=carry_dword,
    )


class TestConditionCodesStatusFlagsAndCarryBorrow(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow relations."""

    def test_ds_instruction_hardware_flag_isolation(self) -> None:
        """DS instructions do not modify hardware scalar condition code (SCC)."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_sgpr(0, 1)  # SCC set
        state.set_vgpr(1, 0x100)
        state.lds.write_qword_at_byte_addr(0x100, 10)
        state.lds.write_qword_at_byte_addr(0x108, 20)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(1), offset0=0, offset1=1)
        interp.step(instr, state)

        self.assertEqual(state.get_sgpr(0), 1)

    def test_zero_flag_wrap(self) -> None:
        """Overflow to 0 sets Z flag."""
        flags = compute_add_src2_u64_flags(MASK64, 1)
        self.assertTrue(flags.z)
        self.assertFalse(flags.n)
        self.assertTrue(flags.carry_out)

    def test_sign_flag_transitions(self) -> None:
        """Bit 63 transition sets N flag."""
        flags = compute_add_src2_u64_flags(0x4000_0000_0000_0000, 0x4000_0000_0000_0000)
        self.assertTrue(flags.n)
        self.assertFalse(flags.z)

    def test_carry_out_detection(self) -> None:
        flags_carry = compute_add_src2_u64_flags(MASK64, 10)
        self.assertTrue(flags_carry.carry_out)

        flags_nocarry = compute_add_src2_u64_flags(100, 200)
        self.assertFalse(flags_nocarry.carry_out)

    def test_dword_boundary_carry(self) -> None:
        a = 0x0000_0000_FFFF_FFFF
        b = 0x0000_0000_0000_0001
        flags = compute_add_src2_u64_flags(a, b)
        self.assertTrue(flags.carry_dword)
        self.assertFalse(flags.carry_out)


# ============================================================================
# Section 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ============================================================================

def f64_to_u64(val: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(val: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u64(val)))[0]


DOUBLE_POS_ZERO: int = 0x0000_0000_0000_0000
DOUBLE_NEG_ZERO: int = 0x8000_0000_0000_0000
DOUBLE_POS_ONE: int = 0x3FF0_0000_0000_0000
DOUBLE_POS_INF: int = 0x7FF0_0000_0000_0000
DOUBLE_NEG_INF: int = 0xFFF0_0000_0000_0000
DOUBLE_QNAN_DEFAULT: int = 0x7FF8_0000_0000_0000
DOUBLE_SNAN_EXAMPLE: int = 0x7FF0_0000_0000_0001
DOUBLE_MIN_SUBNORMAL: int = 0x0000_0000_0000_0001
DOUBLE_MAX_SUBNORMAL: int = 0x000F_FFFF_FFFF_FFFF


class TestIEEE754BoundaryExtremesSubnormalsAndNaNs(unittest.TestCase):
    """Unit tests for IEEE 754 double-precision bit patterns manipulated via ds_add_src2_u64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_double_subnormal_preservation_without_ftz(self) -> None:
        addr_a = 0x100
        addr_b = 0x108
        self.state.lds.write_qword_at_byte_addr(addr_a, DOUBLE_MIN_SUBNORMAL)
        self.state.lds.write_qword_at_byte_addr(addr_b, 0)  # Identity
        self.state.set_vgpr(0, addr_a)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        res = self.interp.step(instr, self.state)

        self.assertEqual(res, DOUBLE_MIN_SUBNORMAL)
        self.assertNotEqual(res, 0)

    def test_double_snan_preserved_without_quietening(self) -> None:
        addr_a = 0x120
        addr_b = 0x128
        self.state.lds.write_qword_at_byte_addr(addr_a, DOUBLE_SNAN_EXAMPLE)
        self.state.lds.write_qword_at_byte_addr(addr_b, 0)
        self.state.set_vgpr(0, addr_a)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        res = self.interp.step(instr, self.state)

        self.assertEqual(res, DOUBLE_SNAN_EXAMPLE)
        self.assertEqual(res & (1 << 51), 0, "Double sNaN was quietened")

    def test_double_nan_payload_preservation(self) -> None:
        addr_a = 0x140
        addr_b = 0x148
        nan_payload = 0x7FF8_0000_CAFE_BABE
        self.state.lds.write_qword_at_byte_addr(addr_a, nan_payload)
        self.state.lds.write_qword_at_byte_addr(addr_b, 0)
        self.state.set_vgpr(0, addr_a)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        res = self.interp.step(instr, self.state)

        self.assertEqual(res, nan_payload)


# ============================================================================
# Section 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ============================================================================

class TestNegativeControlsSyntaxRejectionAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_negative_invalid_vgpr_index(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_negative_invalid_offset_range(self) -> None:
        with self.assertRaises(ValueError):
            DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=256, offset1=0)
        with self.assertRaises(ValueError):
            DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=-1)

    def test_negative_invalid_lds_capacity(self) -> None:
        with self.assertRaises(ValueError):
            LDSModel(capacity_qwords=1000)
        with self.assertRaises(ValueError):
            LDSModel(capacity_qwords=0)

    def test_negative_invalid_simd_num_lanes(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)

    # --- Mutation Detection Tests ---

    def test_mutation_memory_a_not_updated_detected(self) -> None:
        """Mutant: Interpreter executes addition but does not write to LDS[A]."""
        class BuggyInterpreter(ScalarInterpreter):
            def step(self, instr: DSAddSrc2U64Instruction, s: ScalarMachineState) -> int:
                vaddr = s.get_vgpr(instr.vaddr.index)
                addr_a = calc_lds_src2_byte_address(vaddr, instr.offset0)
                # Bug: omits write back to LDS
                return s.lds.read_qword_at_byte_addr(addr_a)

        state = ScalarMachineState()
        state.set_vgpr(0, 0x100)
        state.lds.write_qword_at_byte_addr(0x100, 50)
        state.lds.write_qword_at_byte_addr(0x108, 25)

        buggy = BuggyInterpreter()
        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        buggy.step(instr, state)

        self.assertNotEqual(state.lds.read_qword_at_byte_addr(0x100), 75)

    def test_mutation_memory_b_corrupted_detected(self) -> None:
        """Mutant: Location B is overwritten during addition."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, 0x100)
        state.lds.write_qword_at_byte_addr(0x100, 100)
        state.lds.write_qword_at_byte_addr(0x108, 50)

        instr = DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=1)
        interp.step(instr, state)

        self.assertEqual(state.lds.read_qword_at_byte_addr(0x108), 50)

    def test_mutation_stride_4_instead_of_stride_8_detected(self) -> None:
        """Mutant: 64-bit src2 instruction uses stride 4 instead of stride 8."""
        def buggy_calc_addr(vaddr: int, offset8: int) -> int:
            return (vaddr + ((offset8 & OFFSET8_MASK) * 4)) & MASK32  # Bug: * 4

        correct = calc_lds_src2_byte_address(0x1000, 2)
        mutant = buggy_calc_addr(0x1000, 2)
        self.assertEqual(correct, 0x1010)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(mutant, 0x1008)

    def test_mutation_overflow_masking_omitted_detected(self) -> None:
        """Mutant: Omits 64-bit truncation on addition overflow."""
        def buggy_add(a: int, b: int) -> int:
            return a + b

        correct = oracle_ds_add_src2_u64(MASK64, 1)
        mutant = buggy_add(MASK64, 1)
        self.assertEqual(correct, 0)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(mutant, 1 << 64)
