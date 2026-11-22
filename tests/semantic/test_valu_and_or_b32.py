from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_and_or_b32.

Tests the vector fused bitwise AND and OR instruction:
    v_and_or_b32 D, S0, S1, S2
Lowered semantics:
    dst = ((S0 & S1) | S2) & 0xFFFFFFFF
Mathematical and architectural properties:
    - Boolean logic: Performs bitwise AND between S0 and S1, followed by bitwise OR with S2.
    - Bit-level independence: Every bit position k in [0..31] computes independently:
        dst[k] = (S0[k] & S1[k]) | S2[k]
    - Commutativity: S0 and S1 are interchangeable: v_and_or_b32(a, b, c) == v_and_or_b32(b, a, c).
    - Degenerate forms:
        - When S2 == 0: computes pure bitwise AND: (S0 & S1).
        - When S1 == 0xFFFFFFFF: computes pure bitwise OR: (S0 | S2).
        - When S2 == 0xFFFFFFFF: constant 0xFFFFFFFF.
Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs). Overwrites all 32 bits of destination VGPR.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Sequence, Tuple, Union

# Architectural bitwise masks and constants
MASK32: int = 0xFFFF_FFFF
SIGN32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret unsigned 32-bit dword as signed two's complement integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & SIGN32) else val


def ref_and_or_b32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical computation for v_and_or_b32: (s0 & s1) | s2."""
    return (((s0 & MASK32) & (s1 & MASK32)) | (s2 & MASK32)) & MASK32


def oracle_v_and_or_b32(src0: int, src1: int, src2: int) -> int:
    """Reference architectural oracle for RDNA2 v_and_or_b32 instruction."""
    return ref_and_or_b32(src0, src1, src2)


class TestAndOrB32ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_and_or_b32."""

    def test_basic_and_or(self) -> None:
        # (0xFF & 0x0F) | 0xA0 = 0x0F | 0xA0 = 0xAF
        res = oracle_v_and_or_b32(0xFF, 0x0F, 0xA0)
        self.assertEqual(res, 0xAF)

    def test_zero_src2_identity_and(self) -> None:
        # S2 == 0 -> plain AND between S0 and S1
        self.assertEqual(oracle_v_and_or_b32(0x1234_5678, 0x00FF_00FF, 0), 0x0034_0078)
        self.assertEqual(oracle_v_and_or_b32(0xAAAA_AAAA, 0x5555_5555, 0), 0x0000_0000)

    def test_ones_src1_identity_or(self) -> None:
        # S1 == 0xFFFFFFFF -> (S0 & 0xFFFFFFFF) | S2 == S0 | S2
        s0 = 0x1234_0000
        s2 = 0x0000_5678
        self.assertEqual(oracle_v_and_or_b32(s0, 0xFFFF_FFFF, s2), 0x1234_5678)

    def test_all_ones_src2_saturation(self) -> None:
        # S2 == 0xFFFFFFFF -> result always 0xFFFFFFFF regardless of S0, S1
        self.assertEqual(oracle_v_and_or_b32(0, 0, 0xFFFF_FFFF), 0xFFFF_FFFF)
        self.assertEqual(oracle_v_and_or_b32(0x1234_5678, 0x9ABC_DEF0, 0xFFFF_FFFF), 0xFFFF_FFFF)

    def test_all_zeros_operands(self) -> None:
        self.assertEqual(oracle_v_and_or_b32(0, 0, 0), 0)

    def test_bit_level_truth_table_exhaustive(self) -> None:
        # Test all 8 combinations of (s0, s1, s2) bits:
        # s0 s1 s2 -> (s0 & s1) | s2
        #  0  0  0 -> 0
        #  0  0  1 -> 1
        #  0  1  0 -> 0
        #  0  1  1 -> 1
        #  1  0  0 -> 0
        #  1  0  1 -> 1
        #  1  1  0 -> 1
        #  1  1  1 -> 1
        for s0_bit in (0, 1):
            for s1_bit in (0, 1):
                for s2_bit in (0, 1):
                    expected = (s0_bit & s1_bit) | s2_bit
                    res = oracle_v_and_or_b32(s0_bit, s1_bit, s2_bit)
                    self.assertEqual(res, expected)

    def test_commutativity_src0_src1(self) -> None:
        triplets = [
            (0xF0F0_F0F0, 0x0F0F_0F0F, 0x1234_5678),
            (0xDEAD_BEEF, 0xCAFE_BABE, 0x0123_4567),
            (0xFFFF_0000, 0x0000_FFFF, 0xAAAA_5555),
        ]
        for a, b, c in triplets:
            self.assertEqual(oracle_v_and_or_b32(a, b, c), oracle_v_and_or_b32(b, a, c))



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Represents an architectural 32-bit register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit integer literal or inline constant."""
    value: int

    def __post_init__(self) -> None:
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
            raise ValueError(f"Literal32 value {self.value} out of 32-bit range")

    @property
    def name(self) -> str:
        return f"0x{self.value & MASK32:08X}" if self.value > 9 or self.value < 0 else str(self.value)


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand representing a register or literal source."""
    source: Union[Reg, Literal32]

    def disassemble(self) -> str:
        return self.source.name


class Opcode(Enum):
    V_AND_OR_B32 = "v_and_or_b32"


@dataclass(frozen=True)
class VAndOrB32Instruction:
    """Symbolic AST node representing an RDNA2 v_and_or_b32 VALU instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_AND_OR_B32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_disassembly(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        s5 = Reg(5, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(s5.name, "s5")

    def test_register_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)

    def test_literal_operand(self) -> None:
        lit = Literal32(0x1234)
        self.assertEqual(lit.name, "0x00001234")
        lit_small = Literal32(7)
        self.assertEqual(lit_small.name, "7")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Reg(2, RegType.VGPR))
        src2 = VOP3Operand(Literal32(0xFF))

        inst = VAndOrB32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_and_or_b32 v0, v1, v2, 0x000000FF")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VAndOrB32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)



# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_and_or_b32."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VAndOrB32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)
        s2_raw = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_and_or_b32(s0_raw, s1_raw, s2_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 0xFF)
        self.state.set_vgpr(2, 0x0F)
        self.state.set_vgpr(3, 0xA0)
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xAF)
        self.assertEqual(self.state.get_vgpr(0), 0xAF)

    def test_step_with_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 0xF0F0_0000)
        self.state.set_vgpr(5, 0x0F0F_0000)
        instr = VAndOrB32Instruction(
            dst=Reg(6, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.SGPR)),
            src1=VOP3Operand(Reg(5, RegType.VGPR)),
            src2=VOP3Operand(Literal32(0x0000_FFFF)),
        )
        res = self.interp.step(instr, self.state)
        # (0xF0F0_0000 & 0x0F0F_0000) | 0x0000_FFFF = 0x0000_0000 | 0x0000_FFFF = 0x0000_FFFF
        self.assertEqual(res, 0x0000_FFFF)
        self.assertEqual(self.state.get_vgpr(6), 0x0000_FFFF)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0x1234_5678)
        self.state.set_vgpr(11, 0x9ABC_DEF0)
        self.state.set_vgpr(12, 0x0F0F_0F0F)
        instr = VAndOrB32Instruction(
            dst=Reg(13, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(11, RegType.VGPR)),
            src2=VOP3Operand(Reg(12, RegType.VGPR)),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)



# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VAndOrB32Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]
        s2_vals = [
            self.resolve_lane_operand(instr.src2, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_and_or_b32(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xDEADBEEF

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 1, 0xFF)
            state.set_vgpr(lane, 2, lane)
            state.set_vgpr(lane, 3, 0x100)

        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((0xFF & lane) | 0x100) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, lane * 3)
            state.set_vgpr(lane, 2, 0x0F)
            state.set_vgpr(lane, 3, 0xA0)

        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = (((lane * 3) & 0x0F) | 0xA0) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0xFF)
            state.set_vgpr(lane, 2, 0x0F)
            state.set_vgpr(lane, 3, 0x50)

        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = 0x5F
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_split_halves_masking(self) -> None:
        mask = 0x0000_0000_FFFF_FFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x12)
            state.set_vgpr(lane, 2, 0x34)

        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Literal32(0x56)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), (0x12 & 0x34) | 0x56)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self) -> None:
        mask = 1 << 42
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0xAA)
            state.set_vgpr(lane, 2, 0x55)

        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Literal32(0x77)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == 42:
                self.assertEqual(state.get_vgpr(lane, 0), (0xAA & 0x55) | 0x77)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_zero_exec_mask_preserves_all(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)

        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Literal32(0xFF)),
            src1=VOP3Operand(Literal32(0xFF)),
            src2=VOP3Operand(Literal32(0xFF)),
        )
        results = self.simd_interp.step(instr, state)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_uniform_scalar_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr(7, 0x0F0F_0F0F)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane)

        # src0 is SGPR 7 (uniform broadcast), src1 is per-lane VGPR 1, src2 is literal 0x0000_F000
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(7, RegType.SGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Literal32(0x0000_F000)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((0x0F0F_0F0F & lane) | 0x0000_F000) & MASK32
            self.assertEqual(state.get_vgpr(lane, 0), expected)



# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 0xFF)
        state.set_vgpr(1, 0x0F)
        state.set_vgpr(2, 0xA0)
        # v0 = (v0 & v1) | v2 = (0xFF & 0x0F) | 0xA0 = 0xAF
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0xAF)
        self.assertEqual(state.get_vgpr(0), 0xAF)
        # v1 and v2 must be preserved
        self.assertEqual(state.get_vgpr(1), 0x0F)
        self.assertEqual(state.get_vgpr(2), 0xA0)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 0x0F)
        state.set_vgpr(1, 0xFF)
        state.set_vgpr(2, 0xA0)
        # v0 = (v1 & v0) | v2 = (0xFF & 0x0F) | 0xA0 = 0xAF
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0xAF)
        self.assertEqual(state.get_vgpr(0), 0xAF)
        self.assertEqual(state.get_vgpr(1), 0xFF)
        self.assertEqual(state.get_vgpr(2), 0xA0)

    def test_dst_alias_src2(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 0xA0)  # accumulator in v0
        state.set_vgpr(1, 0xFF)
        state.set_vgpr(2, 0x0F)
        # v0 = (v1 & v2) | v0 = (0xFF & 0x0F) | 0xA0 = 0xAF
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0xAF)
        self.assertEqual(state.get_vgpr(0), 0xAF)
        self.assertEqual(state.get_vgpr(1), 0xFF)
        self.assertEqual(state.get_vgpr(2), 0x0F)

    def test_dst_alias_all_three_sources(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 0x1234_5678)
        # v0 = (v0 & v0) | v0 = v0 (idempotent)
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x1234_5678)
        self.assertEqual(state.get_vgpr(0), 0x1234_5678)

    def test_simd_multi_lane_aliased_accumulation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, 0xF0)
            state.set_vgpr(lane, 1, 0x0F)

        # In-place lane accumulation: v0 = (v0 & v1) | lane = (0xF0 & 0x0F) | lane = 0 | lane = lane
        instr = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),  # wait, test with literal or lane
        )
        # Let's test with v1 = lane and v2 literal
        for lane in range(32):
            state.set_vgpr(lane, 0, 0xFF)
            state.set_vgpr(lane, 1, lane)

        instr_lane = VAndOrB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Literal32(0x100)),
        )
        self.simd_interp.step(instr_lane, state)
        for lane in range(32):
            expected = (lane | 0x100) & MASK32
            self.assertEqual(state.get_vgpr(lane, 0), expected)



# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AndOrStatusFlags:
    """Architectural condition codes and status flags for v_and_or_b32."""
    result_zero: bool
    result_negative: bool
    and_term_zero: bool
    or_term_zero: bool
    all_bits_set: bool
    parity_odd: bool


def evaluate_and_or_flags(s0: int, s1: int, s2: int, res32: int) -> AndOrStatusFlags:
    """Evaluate condition codes, term status, and bitwise characteristics."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    u2 = s2 & MASK32
    and_term = u0 & u1

    return AndOrStatusFlags(
        result_zero=(res32 == 0),
        result_negative=bool(res32 & SIGN32),
        and_term_zero=(and_term == 0),
        or_term_zero=(u2 == 0),
        all_bits_set=(res32 == MASK32),
        parity_odd=bool(res32 & 1),
    )


def compute_bitwise_carry_synthesis(a: int, b: int, c_in: int) -> int:
    """Synthesize full-adder carry-out using bitwise logic and v_and_or_b32 logic.

    Carry-out: Cout = (A & B) | (Cin & (A ^ B))
    Can be computed as: (A & B) | (Cin & (A | B))
    """
    g = a & b           # generate
    p = a | b           # propagate
    carry_term = c_in & p
    return oracle_v_and_or_b32(a, b, carry_term)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation synthesis."""

    def test_zero_flags(self) -> None:
        # All zero
        flags_all_zero = evaluate_and_or_flags(0, 0, 0, oracle_v_and_or_b32(0, 0, 0))
        self.assertTrue(flags_all_zero.result_zero)
        self.assertTrue(flags_all_zero.and_term_zero)
        self.assertTrue(flags_all_zero.or_term_zero)
        self.assertFalse(flags_all_zero.all_bits_set)

        # AND term non-zero, but canceled out? No, OR can only add bits.
        # So result_zero implies both AND term is zero and S2 is zero.
        flags_disjoint = evaluate_and_or_flags(0xAAAA, 0x5555, 0, oracle_v_and_or_b32(0xAAAA, 0x5555, 0))
        self.assertTrue(flags_disjoint.result_zero)
        self.assertTrue(flags_disjoint.and_term_zero)
        self.assertTrue(flags_disjoint.or_term_zero)

    def test_negative_sign_flag(self) -> None:
        # Sign bit 31 set via S2
        res_s2 = oracle_v_and_or_b32(0, 0, 0x8000_0000)
        flags_s2 = evaluate_and_or_flags(0, 0, 0x8000_0000, res_s2)
        self.assertTrue(flags_s2.result_negative)
        self.assertFalse(flags_s2.result_zero)

        # Sign bit 31 set via S0 & S1
        res_and = oracle_v_and_or_b32(0x8000_0000, 0x8000_0000, 0)
        flags_and = evaluate_and_or_flags(0x8000_0000, 0x8000_0000, 0, res_and)
        self.assertTrue(flags_and.result_negative)
        self.assertFalse(flags_and.and_term_zero)
        self.assertTrue(flags_and.or_term_zero)

    def test_all_bits_set_flag(self) -> None:
        # All 32 bits set via S2
        res_ones = oracle_v_and_or_b32(0x1234, 0x5678, 0xFFFF_FFFF)
        flags_ones = evaluate_and_or_flags(0x1234, 0x5678, 0xFFFF_FFFF, res_ones)
        self.assertTrue(flags_ones.all_bits_set)
        self.assertTrue(flags_ones.result_negative)
        self.assertTrue(flags_ones.parity_odd)

    def test_parity_flag(self) -> None:
        # Bit 0 set (odd)
        res_odd = oracle_v_and_or_b32(1, 1, 0)
        self.assertTrue(evaluate_and_or_flags(1, 1, 0, res_odd).parity_odd)

        # Bit 0 clear (even)
        res_even = oracle_v_and_or_b32(2, 2, 0)
        self.assertFalse(evaluate_and_or_flags(2, 2, 0, res_even).parity_odd)

    def test_carry_generator_synthesis(self) -> None:
        # Test full adder carry truth table across bits
        # A B Cin -> Cout
        # 0 0  0  -> 0
        # 0 0  1  -> 0
        # 0 1  0  -> 0
        # 0 1  1  -> 1
        # 1 0  0  -> 0
        # 1 0  1  -> 1
        # 1 1  0  -> 1
        # 1 1  1  -> 1
        for a in (0, 1):
            for b in (0, 1):
                for cin in (0, 1):
                    expected_cout = 1 if (a + b + cin) >= 2 else 0
                    cout = compute_bitwise_carry_synthesis(a, b, cin)
                    self.assertEqual(cout, expected_cout)


if __name__ == "__main__":
    unittest.main()





