from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_xnor_b32.

Tests the vector bitwise XNOR (equivalence) instruction:
    v_xnor_b32 D, S0, S1
Lowered semantics:
    dst = ~(S0 ^ S1) & 0xFFFFFFFF
Mathematical and architectural properties:
    - Bitwise equivalence: Evaluates bitwise equality between S0 and S1.
      Bit k of destination is 1 if S0[k] == S1[k], and 0 if S0[k] != S1[k].
    - Commutativity: S0 and S1 are interchangeable: v_xnor_b32(a, b) == v_xnor_b32(b, a).
    - Algebraic identities:
        - Self-equivalence: v_xnor_b32(a, a) == 0xFFFFFFFF (all bits match).
        - Complement-equivalence: v_xnor_b32(a, ~a) == 0x00000000 (no bits match).
        - Identity with 0xFFFFFFFF: v_xnor_b32(a, 0xFFFFFFFF) == a (pass-through).
        - Inversion with 0x00000000: v_xnor_b32(a, 0) == (~a) & 0xFFFFFFFF (bitwise NOT).
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


def ref_xnor_b32(s0: int, s1: int) -> int:
    """Reference mathematical computation for v_xnor_b32: ~(s0 ^ s1) & 0xFFFFFFFF."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (~(u0 ^ u1)) & MASK32


def oracle_v_xnor_b32(src0: int, src1: int) -> int:
    """Reference architectural oracle for RDNA2 v_xnor_b32 instruction."""
    return ref_xnor_b32(src0, src1)


class TestXnorB32ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_xnor_b32."""

    def test_basic_xnor(self) -> None:
        # Alternating bits
        self.assertEqual(oracle_v_xnor_b32(0xAAAA_AAAA, 0xAAAA_AAAA), 0xFFFF_FFFF)
        self.assertEqual(oracle_v_xnor_b32(0xAAAA_AAAA, 0x5555_5555), 0x0000_0000)

    def test_identity_with_all_ones(self) -> None:
        # ~(a ^ 0xFFFFFFFF) == ~(~a) == a
        values = [0x0000_0000, 0x1234_5678, 0x8000_0000, 0xFFFF_FFFF]
        for v in values:
            self.assertEqual(oracle_v_xnor_b32(v, 0xFFFF_FFFF), v)
            self.assertEqual(oracle_v_xnor_b32(0xFFFF_FFFF, v), v)

    def test_inversion_with_zero(self) -> None:
        # ~(a ^ 0) == ~a
        self.assertEqual(oracle_v_xnor_b32(0x0000_0000, 0), 0xFFFF_FFFF)
        self.assertEqual(oracle_v_xnor_b32(0xFFFF_FFFF, 0), 0x0000_0000)
        self.assertEqual(oracle_v_xnor_b32(0x1234_5678, 0), (~0x1234_5678) & MASK32)

    def test_self_equivalence(self) -> None:
        patterns = [0x0000_0000, 0xCAFE_BABE, 0xDEAD_BEEF, 0xFFFF_FFFF]
        for p in patterns:
            self.assertEqual(oracle_v_xnor_b32(p, p), 0xFFFF_FFFF)

    def test_complement_equivalence(self) -> None:
        patterns = [0x0000_0000, 0x1234_5678, 0xAAAA_5555, 0xFFFF_FFFF]
        for p in patterns:
            comp = (~p) & MASK32
            self.assertEqual(oracle_v_xnor_b32(p, comp), 0x0000_0000)

    def test_bit_level_truth_table(self) -> None:
        # 0 XNOR 0 = 1
        # 0 XNOR 1 = 0
        # 1 XNOR 0 = 0
        # 1 XNOR 1 = 1
        self.assertEqual(oracle_v_xnor_b32(0, 0) & 1, 1)
        self.assertEqual(oracle_v_xnor_b32(0, 1) & 1, 0)
        self.assertEqual(oracle_v_xnor_b32(1, 0) & 1, 0)
        self.assertEqual(oracle_v_xnor_b32(1, 1) & 1, 1)

    def test_commutativity(self) -> None:
        pairs = [
            (0x1122_3344, 0x5566_7788),
            (0xDEAD_BEEF, 0x0123_4567),
            (0x8000_0000, 0x7FFF_FFFF),
        ]
        for a, b in pairs:
            self.assertEqual(oracle_v_xnor_b32(a, b), oracle_v_xnor_b32(b, a))



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
class VOP2Operand:
    """Typed expression operand representing a register or literal source."""
    source: Union[Reg, Literal32]

    def disassemble(self) -> str:
        return self.source.name


class Opcode(Enum):
    V_XNOR_B32 = "v_xnor_b32"


@dataclass(frozen=True)
class VXnorB32Instruction:
    """Symbolic AST node representing an RDNA2 v_xnor_b32 VALU instruction."""
    dst: Reg
    src0: VOP2Operand
    src1: VOP2Operand
    opcode: Opcode = Opcode.V_XNOR_B32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"


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
        lit_small = Literal32(5)
        self.assertEqual(lit_small.name, "5")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP2Operand(Reg(1, RegType.VGPR))
        src1 = VOP2Operand(Literal32(0xFF))

        inst = VXnorB32Instruction(dst=dst, src0=src0, src1=src1)
        self.assertEqual(inst.disassemble(), "v_xnor_b32 v0, v1, 0x000000FF")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP2Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VXnorB32Instruction(dst=dst_sgpr, src0=src, src1=src)



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
    """Deterministic scalar instruction interpreter for v_xnor_b32."""

    def resolve_operand(self, op: VOP2Operand, state: ScalarMachineState) -> int:
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

    def step(self, instr: VXnorB32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase
        result = oracle_v_xnor_b32(s0_raw, s1_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 0xAAAA_AAAA)
        self.state.set_vgpr(2, 0x5555_5555)
        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x0000_0000)
        self.assertEqual(self.state.get_vgpr(0), 0x0000_0000)

    def test_step_with_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 0x1234_5678)
        instr = VXnorB32Instruction(
            dst=Reg(5, RegType.VGPR),
            src0=VOP2Operand(Reg(4, RegType.SGPR)),
            src1=VOP2Operand(Literal32(0xFFFF_FFFF)),
        )
        res = self.interp.step(instr, self.state)
        # xnor with 0xFFFFFFFF is identity
        self.assertEqual(res, 0x1234_5678)
        self.assertEqual(self.state.get_vgpr(5), 0x1234_5678)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0xDEAD_BEEF)
        self.state.set_vgpr(11, 0xCAFE_BABE)
        instr = VXnorB32Instruction(
            dst=Reg(12, RegType.VGPR),
            src0=VOP2Operand(Reg(10, RegType.VGPR)),
            src1=VOP2Operand(Reg(11, RegType.VGPR)),
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

    def resolve_lane_operand(self, op: VOP2Operand, state: SIMDState, lane: int) -> int:
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

    def step(self, instr: VXnorB32Instruction, state: SIMDState) -> List[Optional[int]]:
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

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_xnor_b32(s0_vals[lane], s1_vals[lane])
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
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, lane)

        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            # Same operand XNOR-ed with itself produces 0xFFFFFFFF
            self.assertEqual(results[lane], 0xFFFF_FFFF)
            self.assertEqual(state.get_vgpr(lane, 0), 0xFFFF_FFFF)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, 0xFFFF_FFFF)

        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            # XNOR with 0xFFFFFFFF preserves lane value exactly
            self.assertEqual(results[lane], lane)
            self.assertEqual(state.get_vgpr(lane, 0), lane)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x1234_5678)
            state.set_vgpr(lane, 2, 0x0000_0000)

        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(2, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        expected = (~0x1234_5678) & MASK32
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
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
            state.set_vgpr(lane, 1, 0xAAAA_AAAA)
            state.set_vgpr(lane, 2, 0x5555_5555)

        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x0000_0000)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self) -> None:
        mask = 1 << 42
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x0F0F_0F0F)
            state.set_vgpr(lane, 2, 0x0F0F_0F0F)

        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(2, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == 42:
                self.assertEqual(state.get_vgpr(lane, 0), 0xFFFF_FFFF)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_zero_exec_mask_preserves_all(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)

        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Literal32(0)),
            src1=VOP2Operand(Literal32(0)),
        )
        results = self.simd_interp.step(instr, state)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_uniform_scalar_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr(3, 0xFFFF_0000)
        for lane in range(32):
            state.set_vgpr(lane, 1, 0x0000_FFFF)

        # src0 is SGPR 3 (uniform broadcast), src1 is per-lane VGPR 1
        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(3, RegType.SGPR)),
            src1=VOP2Operand(Reg(1, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x0000_0000)



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
        state.set_vgpr(0, 0xAAAA_AAAA)
        state.set_vgpr(1, 0x5555_5555)
        # v0 = xnor(v0, v1) = ~(0xAAAA_AAAA ^ 0x5555_5555) = ~0xFFFF_FFFF = 0
        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(0, RegType.VGPR)),
            src1=VOP2Operand(Reg(1, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x0000_0000)
        self.assertEqual(state.get_vgpr(0), 0x0000_0000)
        # v1 must be preserved
        self.assertEqual(state.get_vgpr(1), 0x5555_5555)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 0x1234_5678)
        state.set_vgpr(1, 0xFFFF_FFFF)
        # v0 = xnor(v1, v0) = xnor(0xFFFFFFFF, v0) = v0
        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(1, RegType.VGPR)),
            src1=VOP2Operand(Reg(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x1234_5678)
        self.assertEqual(state.get_vgpr(0), 0x1234_5678)
        self.assertEqual(state.get_vgpr(1), 0xFFFF_FFFF)

    def test_dst_alias_both_srcs(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 0xDEAD_BEEF)
        # v0 = xnor(v0, v0) = 0xFFFFFFFF
        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(0, RegType.VGPR)),
            src1=VOP2Operand(Reg(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0xFFFF_FFFF)
        self.assertEqual(state.get_vgpr(0), 0xFFFF_FFFF)

    def test_simd_multi_lane_aliased_inversion(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, lane)

        # In-place lane bitwise inversion: v0 = xnor(v0, 0) == ~v0
        instr = VXnorB32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP2Operand(Reg(0, RegType.VGPR)),
            src1=VOP2Operand(Literal32(0)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = (~lane) & MASK32
            self.assertEqual(state.get_vgpr(lane, 0), expected)



# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class XnorStatusFlags:
    """Architectural condition codes and status flags for v_xnor_b32."""
    result_zero: bool
    result_negative: bool
    all_bits_set: bool
    operands_equal: bool
    operands_complementary: bool
    matching_bits_count: int
    parity_odd: bool


def evaluate_xnor_flags(s0: int, s1: int, res32: int) -> XnorStatusFlags:
    """Evaluate condition codes, bit equality metrics, and parity flags."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    popcount = bin(res32).count("1")

    return XnorStatusFlags(
        result_zero=(res32 == 0),
        result_negative=bool(res32 & SIGN32),
        all_bits_set=(res32 == MASK32),
        operands_equal=(u0 == u1),
        operands_complementary=(u0 == ((~u1) & MASK32)),
        matching_bits_count=popcount,
        parity_odd=bool(res32 & 1),
    )


def compute_matching_bit_metrics(s0: int, s1: int) -> Tuple[int, int]:
    """Compute number of matching bits and differing bits between two 32-bit words."""
    res = oracle_v_xnor_b32(s0, s1)
    matching = bin(res).count("1")
    differing = 32 - matching
    return matching, differing


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and bit equality metrics."""

    def test_all_matching_flags(self) -> None:
        val = 0x1234_5678
        res = oracle_v_xnor_b32(val, val)
        flags = evaluate_xnor_flags(val, val, res)
        self.assertTrue(flags.all_bits_set)
        self.assertTrue(flags.operands_equal)
        self.assertFalse(flags.operands_complementary)
        self.assertFalse(flags.result_zero)
        self.assertEqual(flags.matching_bits_count, 32)

    def test_no_matching_flags(self) -> None:
        val = 0xAAAA_5555
        comp = (~val) & MASK32
        res = oracle_v_xnor_b32(val, comp)
        flags = evaluate_xnor_flags(val, comp, res)
        self.assertTrue(flags.result_zero)
        self.assertTrue(flags.operands_complementary)
        self.assertFalse(flags.operands_equal)
        self.assertFalse(flags.all_bits_set)
        self.assertEqual(flags.matching_bits_count, 0)

    def test_negative_sign_flag(self) -> None:
        # High bits match (both have bit 31 = 1) -> result has bit 31 = 1
        s0 = 0x8000_0000
        s1 = 0x8000_0000
        res = oracle_v_xnor_b32(s0, s1)
        flags = evaluate_xnor_flags(s0, s1, res)
        self.assertTrue(flags.result_negative)

        # High bits differ -> result has bit 31 = 0
        s0_pos = 0x0000_0000
        res_diff = oracle_v_xnor_b32(s0_pos, s1)
        flags_diff = evaluate_xnor_flags(s0_pos, s1, res_diff)
        self.assertFalse(flags_diff.result_negative)

    def test_parity_flag(self) -> None:
        # Low bits match -> bit 0 of result is 1 (odd)
        res_odd = oracle_v_xnor_b32(1, 1)
        self.assertTrue(evaluate_xnor_flags(1, 1, res_odd).parity_odd)

        # Low bits differ -> bit 0 of result is 0 (even)
        res_even = oracle_v_xnor_b32(1, 0)
        self.assertFalse(evaluate_xnor_flags(1, 0, res_even).parity_odd)

    def test_matching_bit_metrics_sum_to_32(self) -> None:
        pairs = [
            (0x0000_0000, 0xFFFF_FFFF),
            (0x1234_5678, 0x8765_4321),
            (0x0F0F_0F0F, 0x00FF_00FF),
        ]
        for a, b in pairs:
            matching, differing = compute_matching_bit_metrics(a, b)
            self.assertEqual(matching + differing, 32)



# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 float bit pattern handling, boundary extremes, and raw integer casting."""

    F32_POS_ZERO: int = 0x0000_0000
    F32_NEG_ZERO: int = 0x8000_0000
    F32_POS_INF: int = 0x7F80_0000
    F32_NEG_INF: int = 0xFF80_0000
    F32_QNAN: int = 0x7FC0_0000
    F32_SNAN: int = 0x7F80_0001
    F32_MIN_SUBNORMAL: int = 0x0000_0001

    def test_ieee754_special_float_bitcasts_treated_as_raw_bits(self) -> None:
        # v_xnor_b32 is a bitwise instruction. Shaders bit-casting
        # floats (+0, -0, +/-inf, NaNs) operate on raw 32-bit words without FPU exceptions.
        specials = [
            self.F32_POS_ZERO,
            self.F32_NEG_ZERO,
            self.F32_POS_INF,
            self.F32_NEG_INF,
            self.F32_QNAN,
            self.F32_SNAN,
        ]
        for f in specials:
            # Self-equivalence: f XNOR f == 0xFFFFFFFF
            self.assertEqual(oracle_v_xnor_b32(f, f), 0xFFFF_FFFF)
            # Identity with all-ones: f XNOR 0xFFFFFFFF == f
            self.assertEqual(oracle_v_xnor_b32(f, MASK32), f)

    def test_float_sign_equality_detection(self) -> None:
        # +0.0 (0x00000000) vs -0.0 (0x80000000):
        # All bits match except bit 31 -> result is 0x7FFFFFFF
        res = oracle_v_xnor_b32(self.F32_POS_ZERO, self.F32_NEG_ZERO)
        self.assertEqual(res, 0x7FFF_FFFF)

        # +inf (0x7F800000) vs -inf (0xFF800000):
        # All bits match except bit 31 -> result is 0x7FFFFFFF
        res_inf = oracle_v_xnor_b32(self.F32_POS_INF, self.F32_NEG_INF)
        self.assertEqual(res_inf, 0x7FFF_FFFF)

    def test_subnormal_float_bits_no_flush_to_zero(self) -> None:
        # Smallest positive subnormal float (bit 0 = 1): must not be flushed to 0
        sub = self.F32_MIN_SUBNORMAL
        res = oracle_v_xnor_b32(sub, 0)
        expected = (~sub) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(res & 1, 0)  # bit 0 inverted to 0

    def test_nan_payload_exact_bit_matching(self) -> None:
        nan1 = 0x7FA5_5A3C
        nan2 = 0x7FA5_5A3C
        self.assertEqual(oracle_v_xnor_b32(nan1, nan2), 0xFFFF_FFFF)

        # Single payload bit differs:
        nan3 = nan1 ^ (1 << 4)
        res_diff = oracle_v_xnor_b32(nan1, nan3)
        self.assertEqual(res_diff, (~(1 << 4)) & MASK32)

    def test_extreme_u32_boundaries(self) -> None:
        boundaries = [0x0000_0000, 0x0000_0001, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF]
        for b in boundaries:
            self.assertEqual(oracle_v_xnor_b32(b, b), MASK32)
            self.assertEqual(oracle_v_xnor_b32(b, MASK32), b)


if __name__ == "__main__":
    unittest.main()






