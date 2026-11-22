from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_add_lshl_u32.

Tests the vector fused add and logical shift left instruction:
    v_add_lshl_u32 D, S0, S1, S2
Lowered semantics:
    sum32 = (S0 + S1) & 0xFFFFFFFF
    shift = S2 & 0x1F
    dst   = (sum32 << shift) & 0xFFFFFFFF
Mathematical and architectural properties:
    - Addition: 32-bit modular addition with wraparound mod 2^32.
    - Shift: Logical shift left by (S2 & 31); high bits shifted past bit 31 are discarded,
      and vacant low bits are zero-filled.
    - Commutativity: S0 and S1 are interchangeable: v_add_lshl_u32(a, b, c) == v_add_lshl_u32(b, a, c).
    - Shift operand masking: Bits [31:5] of S2 are ignored by hardware and translator.
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
SHIFT_MASK: int = 0x1F
SIGN32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret unsigned 32-bit dword as signed two's complement integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & SIGN32) else val


def ref_add_lshl_u32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical computation for v_add_lshl_u32: ((s0 + s1) mod 2^32) << (s2 & 31)."""
    sum32 = (s0 + s1) & MASK32
    shift = s2 & SHIFT_MASK
    return (sum32 << shift) & MASK32


def oracle_v_add_lshl_u32(src0: int, src1: int, src2: int) -> int:
    """Reference architectural oracle for RDNA2 v_add_lshl_u32 instruction."""
    return ref_add_lshl_u32(src0, src1, src2)


class TestAddLshlU32ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_add_lshl_u32."""

    def test_basic_add_lshl(self) -> None:
        # (10 + 20) << 2 = 30 << 2 = 120
        res = oracle_v_add_lshl_u32(10, 20, 2)
        self.assertEqual(res, 120)

    def test_shift_zero_identity(self) -> None:
        # Shift count 0 produces plain modular addition
        self.assertEqual(oracle_v_add_lshl_u32(100, 250, 0), 350)
        self.assertEqual(oracle_v_add_lshl_u32(0xFFFFFFFF, 1, 0), 0)

    def test_shift_masking_upper_bits(self) -> None:
        # Shift amount is strictly masked by 0x1F (only lower 5 bits count)
        base = oracle_v_add_lshl_u32(5, 5, 3)
        self.assertEqual(base, 10 << 3)
        # S2 with upper bits set: 32 + 3 = 35 -> shift 3
        self.assertEqual(oracle_v_add_lshl_u32(5, 5, 35), base)
        # S2 = 0x100 -> 256 & 31 = 0 -> shift 0
        self.assertEqual(oracle_v_add_lshl_u32(5, 5, 256), 10)
        # S2 = 0xFFFFFFFF -> 0xFFFFFFFF & 31 = 31 -> shift 31
        self.assertEqual(oracle_v_add_lshl_u32(1, 0, 0xFFFFFFFF), (1 << 31) & MASK32)

    def test_addition_modulo_overflow(self) -> None:
        # Addition wraps mod 2^32 before the shift
        s0 = 0x8000_0000
        s1 = 0x8000_0000
        # (0x80000000 + 0x80000000) mod 2^32 = 0
        self.assertEqual(oracle_v_add_lshl_u32(s0, s1, 5), 0)

        # 0xFFFFFFFF + 2 = 1 (mod 2^32) -> 1 << 4 = 16
        self.assertEqual(oracle_v_add_lshl_u32(0xFFFF_FFFF, 2, 4), 16)

    def test_shift_truncation_overflow(self) -> None:
        # Bits shifted past 31 are discarded
        # 0x00010000 << 16 = 0 (1 << 32 is truncated)
        self.assertEqual(oracle_v_add_lshl_u32(0x0001_0000, 0, 16), 0)
        # 0x40000000 << 1 = 0x80000000, << 2 = 0
        self.assertEqual(oracle_v_add_lshl_u32(0x4000_0000, 0, 1), 0x8000_0000)
        self.assertEqual(oracle_v_add_lshl_u32(0x4000_0000, 0, 2), 0)

    def test_commutativity_src0_src1(self) -> None:
        # S0 and S1 addition is strictly commutative
        pairs = [
            (1234, 5678, 4),
            (0xFFFF_0000, 0x0000_FFFF, 7),
            (0xDEAD_BEEF, 0xCAFE_BABE, 12),
        ]
        for a, b, c in pairs:
            self.assertEqual(oracle_v_add_lshl_u32(a, b, c), oracle_v_add_lshl_u32(b, a, c))

    def test_identity_zero_operands(self) -> None:
        # s0 = 0, s1 = 0 -> 0
        self.assertEqual(oracle_v_add_lshl_u32(0, 0, 15), 0)
        # s1 = 0, s2 = 0 -> identity pass-through of s0
        val = 0x1234_5678
        self.assertEqual(oracle_v_add_lshl_u32(val, 0, 0), val)
        self.assertEqual(oracle_v_add_lshl_u32(0, val, 0), val)



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
    V_ADD_LSHL_U32 = "v_add_lshl_u32"


@dataclass(frozen=True)
class VAddLshlU32Instruction:
    """Symbolic AST node representing an RDNA2 v_add_lshl_u32 VALU instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_ADD_LSHL_U32

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
        lit_small = Literal32(4)
        self.assertEqual(lit_small.name, "4")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Reg(2, RegType.VGPR))
        src2 = VOP3Operand(Literal32(3))

        inst = VAddLshlU32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_add_lshl_u32 v0, v1, v2, 3")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VAddLshlU32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)



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
    """Deterministic scalar instruction interpreter for v_add_lshl_u32."""

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

    def step(self, instr: VAddLshlU32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)
        s2_raw = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_add_lshl_u32(s0_raw, s1_raw, s2_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 15)
        self.state.set_vgpr(2, 25)
        self.state.set_vgpr(3, 3)
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # (15 + 25) << 3 = 40 << 3 = 320
        self.assertEqual(res, 320)
        self.assertEqual(self.state.get_vgpr(0), 320)

    def test_step_with_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 50)
        self.state.set_vgpr(5, 50)
        instr = VAddLshlU32Instruction(
            dst=Reg(6, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.SGPR)),
            src1=VOP3Operand(Reg(5, RegType.VGPR)),
            src2=VOP3Operand(Literal32(2)),
        )
        res = self.interp.step(instr, self.state)
        # (50 + 50) << 2 = 100 << 2 = 400
        self.assertEqual(res, 400)
        self.assertEqual(self.state.get_vgpr(6), 400)

    def test_step_with_overflow(self) -> None:
        self.state.set_vgpr(1, 0xFFFF_FFFF)
        self.state.set_vgpr(2, 1)
        self.state.set_vgpr(3, 5)
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # (0xFFFFFFFF + 1) = 0 -> 0 << 5 = 0
        self.assertEqual(res, 0)
        self.assertEqual(self.state.get_vgpr(0), 0)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0x1234_5678)
        self.state.set_vgpr(11, 0x0000_0008)
        self.state.set_vgpr(12, 4)
        instr = VAddLshlU32Instruction(
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

    def step(self, instr: VAddLshlU32Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_add_lshl_u32(s0_vals[lane], s1_vals[lane], s2_vals[lane])
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
            state.set_vgpr(lane, 3, 1)

        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((lane + lane) << 1) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, lane * 2)
            state.set_vgpr(lane, 2, 5)
            state.set_vgpr(lane, 3, 2)

        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = ((lane * 2 + 5) << 2) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 10)
            state.set_vgpr(lane, 2, 20)
            state.set_vgpr(lane, 3, 3)

        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = (30 << 3) & MASK32
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
            state.set_vgpr(lane, 1, 100)
            state.set_vgpr(lane, 2, 200)

        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Literal32(1)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 300 << 1)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self) -> None:
        mask = 1 << 42
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 7)
            state.set_vgpr(lane, 2, 3)

        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Literal32(4)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == 42:
                self.assertEqual(state.get_vgpr(lane, 0), 10 << 4)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_zero_exec_mask_preserves_all(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)

        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Literal32(10)),
            src1=VOP3Operand(Literal32(20)),
            src2=VOP3Operand(Literal32(2)),
        )
        results = self.simd_interp.step(instr, state)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_uniform_scalar_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr(5, 100)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane)

        # src0 is SGPR 5 (uniform broadcast 100), src1 is VGPR 1 (lane), src2 is literal 2
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(5, RegType.SGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Literal32(2)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((100 + lane) << 2) & MASK32
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
        state.set_vgpr(0, 25)
        state.set_vgpr(1, 15)
        # v0 = (v0 + v1) << 2 = (25 + 15) << 2 = 160
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Literal32(2)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 160)
        self.assertEqual(state.get_vgpr(0), 160)
        # v1 must be preserved intact
        self.assertEqual(state.get_vgpr(1), 15)

    def test_dst_alias_src1(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 30)
        state.set_vgpr(1, 70)
        # v0 = (v1 + v0) << 1 = (70 + 30) << 1 = 200
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Literal32(1)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 200)
        self.assertEqual(state.get_vgpr(0), 200)
        self.assertEqual(state.get_vgpr(1), 70)

    def test_dst_alias_src2(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 3)   # shift count in v0
        state.set_vgpr(1, 10)
        state.set_vgpr(2, 20)
        # v0 = (v1 + v2) << v0 = (10 + 20) << 3 = 240
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 240)
        self.assertEqual(state.get_vgpr(0), 240)
        self.assertEqual(state.get_vgpr(1), 10)
        self.assertEqual(state.get_vgpr(2), 20)

    def test_dst_alias_all_three_sources(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr(0, 2)
        # v0 = (v0 + v0) << v0 = (2 + 2) << 2 = 16
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 16)
        self.assertEqual(state.get_vgpr(0), 16)

    def test_simd_multi_lane_aliased_accumulation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 0, lane + 1)
            state.set_vgpr(lane, 1, lane + 1)

        # In-place lane accumulation: v0 = (v0 + v1) << 1
        instr = VAddLshlU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Literal32(1)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = (((lane + 1) * 2) << 1) & MASK32
            self.assertEqual(state.get_vgpr(lane, 0), expected)



# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AddLshlStatusFlags:
    """Architectural condition codes and status flags for v_add_lshl_u32."""
    result_zero: bool
    result_negative: bool
    add_carry_out: bool
    add_zero: bool
    shift_truncated_bits: bool
    odd_parity: bool


def evaluate_add_lshl_flags(s0: int, s1: int, s2: int, res32: int) -> AddLshlStatusFlags:
    """Evaluate condition codes, addition carry-out, and shift truncation flags."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    sum_full = u0 + u1
    add_carry = (sum_full > MASK32)
    sum32 = sum_full & MASK32
    shift = s2 & SHIFT_MASK
    shifted_full = sum32 << shift
    shift_truncated = ((shifted_full >> 32) != 0)

    return AddLshlStatusFlags(
        result_zero=(res32 == 0),
        result_negative=bool(res32 & SIGN32),
        add_carry_out=add_carry,
        add_zero=(sum32 == 0),
        shift_truncated_bits=shift_truncated,
        odd_parity=bool(res32 & 1),
    )


def compute_add_carry_generation(s0: int, s1: int) -> Tuple[int, int]:
    """Compute 32-bit addition sum and unsigned carry-out bit."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    total = u0 + u1
    return total & MASK32, 1 if total > MASK32 else 0


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_zero_flags(self) -> None:
        # Addition itself is zero
        s0, s1, s2 = 0x8000_0000, 0x8000_0000, 4
        res = oracle_v_add_lshl_u32(s0, s1, s2)
        flags = evaluate_add_lshl_flags(s0, s1, s2, res)
        self.assertTrue(flags.result_zero)
        self.assertTrue(flags.add_zero)
        self.assertTrue(flags.add_carry_out)

        # Addition non-zero but shifted to zero
        s0, s1, s2 = 0x0001_0000, 0, 16
        res2 = oracle_v_add_lshl_u32(s0, s1, s2)
        flags2 = evaluate_add_lshl_flags(s0, s1, s2, res2)
        self.assertTrue(flags2.result_zero)
        self.assertFalse(flags2.add_zero)
        self.assertTrue(flags2.shift_truncated_bits)

    def test_negative_sign_flag(self) -> None:
        # Result has bit 31 set: (1 + 0) << 31 = 0x80000000
        res = oracle_v_add_lshl_u32(1, 0, 31)
        flags = evaluate_add_lshl_flags(1, 0, 31, res)
        self.assertTrue(flags.result_negative)
        self.assertFalse(flags.result_zero)

        # Positive result: (1 + 1) << 1 = 4
        res_pos = oracle_v_add_lshl_u32(1, 1, 1)
        flags_pos = evaluate_add_lshl_flags(1, 1, 1, res_pos)
        self.assertFalse(flags_pos.result_negative)

    def test_add_carry_out_generation(self) -> None:
        sum32, carry = compute_add_carry_generation(0xFFFF_FFFF, 1)
        self.assertEqual(sum32, 0)
        self.assertEqual(carry, 1)

        sum32_nc, carry_nc = compute_add_carry_generation(0x1000, 0x2000)
        self.assertEqual(sum32_nc, 0x3000)
        self.assertEqual(carry_nc, 0)

    def test_odd_parity_flag(self) -> None:
        # Shift == 0 preserves sum parity
        res_odd = oracle_v_add_lshl_u32(2, 3, 0)  # 5 (odd)
        flags_odd = evaluate_add_lshl_flags(2, 3, 0, res_odd)
        self.assertTrue(flags_odd.odd_parity)

        # Shift > 0 always forces bit 0 to 0 (even)
        res_even = oracle_v_add_lshl_u32(2, 3, 1)  # 10 (even)
        flags_even = evaluate_add_lshl_flags(2, 3, 1, res_even)
        self.assertFalse(flags_even.odd_parity)



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

    def test_ieee754_special_float_bitcasts_treated_as_raw_integers(self) -> None:
        # v_add_lshl_u32 is an integer arithmetic instruction. Shaders bit-casting
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
            # Identity pass-through: (f + 0) << 0 == f
            self.assertEqual(oracle_v_add_lshl_u32(f, 0, 0), f)
            # Self-double and shift: (f + f) << 0 == (f << 1) & 0xFFFFFFFF
            self.assertEqual(oracle_v_add_lshl_u32(f, f, 0), (f << 1) & MASK32)

    def test_subnormal_float_bits_no_flush_to_zero(self) -> None:
        # Smallest positive subnormal float (bit 0 = 1): must not be flushed to 0
        sub = self.F32_MIN_SUBNORMAL
        res = oracle_v_add_lshl_u32(sub, 0, 1)
        self.assertEqual(res, 0x0000_0002)

        # Adding two subnormals without flush
        res2 = oracle_v_add_lshl_u32(sub, sub, 0)
        self.assertEqual(res2, 0x0000_0002)

    def test_nan_payload_preservation_across_shifts(self) -> None:
        custom_nan = 0x7FA5_5A3C
        # (custom_nan + 0) << 0 preserves payload bits exactly
        self.assertEqual(oracle_v_add_lshl_u32(custom_nan, 0, 0), custom_nan)

    def test_extreme_u32_boundaries(self) -> None:
        boundaries = [0x0000_0000, 0x0000_0001, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF]
        for b in boundaries:
            # Shift 0
            self.assertEqual(oracle_v_add_lshl_u32(b, 0, 0), b)
            # Shift 31
            expected_shift31 = (b << 31) & MASK32
            self.assertEqual(oracle_v_add_lshl_u32(b, 0, 31), expected_shift31)

    def test_shift_by_31_boundary(self) -> None:
        # Shifting 1 by 31 produces 0x80000000 (matches IEEE 754 -0.0f bit pattern)
        res = oracle_v_add_lshl_u32(1, 0, 31)
        self.assertEqual(res, 0x8000_0000)
        self.assertEqual(res, self.F32_NEG_ZERO)


if __name__ == "__main__":
    unittest.main()






