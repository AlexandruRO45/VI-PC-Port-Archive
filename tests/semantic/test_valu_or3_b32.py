from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_or3_b32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_or3_b32 D, S0, S1, S2
Operation:
    Performs a 3-way bitwise OR across three 32-bit operands:
        dst = (S0 | S1 | S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0, S1, S2: 32-bit unsigned integer values.
    - Full bitwise OR associativity and commutativity.
    - Zero identity: S0 | 0 | 0 == S0.
    - All-ones absorbing element: S0 | 0xFFFFFFFF | S2 == 0xFFFFFFFF.
    - Idempotency: S0 | S0 | S0 == S0.

Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    The destination is always a 32-bit vector register (VGPR).
    VALU instructions do not modify VCC unless explicitly defined as carry/comparison ops.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF


def oracle_v_or3_b32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_or3_b32."""
    s0 = src0_bits & MASK32
    s1 = src1_bits & MASK32
    s2 = src2_bits & MASK32
    return (s0 | s1 | s2) & MASK32


class TestValuOr3B32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_or3_b32."""

    def test_zero_identity(self) -> None:
        # S0 | 0 | 0 == S0
        res = oracle_v_or3_b32(0x12345678, 0, 0)
        self.assertEqual(res, 0x12345678)

    def test_all_ones_absorbing(self) -> None:
        # Any operand 0xFFFFFFFF results in 0xFFFFFFFF
        res = oracle_v_or3_b32(0x1234, 0xFFFFFFFF, 0x5678)
        self.assertEqual(res, 0xFFFFFFFF)

    def test_idempotency(self) -> None:
        # S | S | S == S
        val = 0xA5A5A5A5
        self.assertEqual(oracle_v_or3_b32(val, val, val), val)

    def test_disjoint_bitfields_combination(self) -> None:
        # Combine three disjoint bitfields into single dword
        s0 = 0xFF000000
        s1 = 0x00FF0000
        s2 = 0x0000FFFF
        res = oracle_v_or3_b32(s0, s1, s2)
        self.assertEqual(res, 0xFFFFFFFF)

    def test_commutativity_and_associativity(self) -> None:
        a, b, c = 0x12345678, 0x9ABCDEF0, 0x5555AAAA
        res1 = oracle_v_or3_b32(a, b, c)
        res2 = oracle_v_or3_b32(c, a, b)
        res3 = oracle_v_or3_b32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)


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
    """Represents a 32-bit literal integer constant."""
    bits: int

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK32:08X}"


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand (Register or Literal32)."""
    source: Union[Reg, Literal32]

    def disassemble(self) -> str:
        return self.source.name


class Opcode(Enum):
    V_OR3_B32 = "v_or3_b32"


@dataclass(frozen=True)
class VOr3B32Instruction:
    """Symbolic AST node representing a RDNA2 v_or3_b32 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_OR3_B32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"{self.opcode.value} {self.dst.name}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


class TestValuOr3B32SymbolicAST(unittest.TestCase):
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
        lit = Literal32(0x12345678)
        self.assertEqual(lit.name, "0x12345678")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Literal32(0xFF))
        src2 = VOP3Operand(Reg(2, RegType.SGPR))

        inst = VOr3B32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_or3_b32 v0, v1, 0x000000FF, s2")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VOr3B32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)
    vcc: int = 0

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for v_or3_b32."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VOr3B32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)
        s2_raw = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_or3_b32(s0_raw, s1_raw, s2_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuOr3B32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 0x1000)
        self.state.set_vgpr(2, 0x0200)
        self.state.set_vgpr(3, 0x0034)
        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x1234)
        self.assertEqual(self.state.get_vgpr(0), 0x1234)

    def test_step_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 0xAA000000)
        self.state.set_vgpr(1, 0x00BB0000)
        instr = VOr3B32Instruction(
            dst=Reg(5, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.SGPR)),
            src1=VOP3Operand(Literal32(0x0000CC00)),
            src2=VOP3Operand(Reg(1, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xAABBCC00)
        self.assertEqual(self.state.get_vgpr(5), 0xAABBCC00)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0xDEAD0000)
        self.state.set_vgpr(11, 0x0000BEEF)
        self.state.set_vgpr(12, 0x00AA0000)
        instr = VOr3B32Instruction(
            dst=Reg(13, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(11, RegType.VGPR)),
            src2=VOP3Operand(Reg(12, RegType.VGPR)),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)

    def test_unaffected_registers_preserved(self) -> None:
        self.state.set_vgpr(1, 0x10)
        self.state.set_vgpr(2, 0x20)
        self.state.set_vgpr(3, 0x01)
        self.state.set_vgpr(7, 0xCAFEFACE)
        self.state.set_sgpr(9, 0xBEEFBABE)
        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(7), 0xCAFEFACE)
        self.assertEqual(self.state.get_sgpr(9), 0xBEEFBABE)


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
    vcc: int = 0

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
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VOr3B32Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_or3_b32(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuOr3B32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xDEADBEEF

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, 0x100)
            state.set_vgpr(lane, 3, 0x1000)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = lane | 0x100 | 0x1000
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, 1 << (lane % 32))
            state.set_vgpr(lane, 2, 0x40000000)
            state.set_vgpr(lane, 3, 0x80000000)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = (1 << (lane % 32)) | 0xC0000000
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x01)
            state.set_vgpr(lane, 2, 0x02)
            state.set_vgpr(lane, 3, 0x04)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = 0x07
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
            state.set_vgpr(lane, 1, 0xAA)
            state.set_vgpr(lane, 2, 0x5500)
            state.set_vgpr(lane, 3, 0x00)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x55AA)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_lane_active(self) -> None:
        target_lane = 23
        mask = 1 << target_lane
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x10)
            state.set_vgpr(lane, 2, 0x20)
            state.set_vgpr(lane, 3, 0x40)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(results[lane], 0x70)
                self.assertEqual(state.get_vgpr(lane, 0), 0x70)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_all_lanes_disabled(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x12)
            state.set_vgpr(lane, 2, 0x34)
            state.set_vgpr(lane, 3, 0x56)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_shared_sgpr_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        state.set_sgpr(5, 0xF0000000)
        for lane in range(32):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, 0x00FF0000)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(5, RegType.SGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(16):
            expected = 0xF0FF0000 | lane
            self.assertEqual(state.get_vgpr(lane, 0), expected)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuOr3B32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_dst_aliases_src0(self) -> None:
        # v0 = v0 | v1 | v2
        self.state.set_vgpr(0, 0x10)
        self.state.set_vgpr(1, 0x20)
        self.state.set_vgpr(2, 0x05)
        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x35)
        self.assertEqual(self.state.get_vgpr(0), 0x35)
        self.assertEqual(self.state.get_vgpr(1), 0x20)
        self.assertEqual(self.state.get_vgpr(2), 0x05)

    def test_dst_aliases_src1(self) -> None:
        # v1 = v0 | v1 | v2
        self.state.set_vgpr(0, 0x01)
        self.state.set_vgpr(1, 0x02)
        self.state.set_vgpr(2, 0x04)
        instr = VOr3B32Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x07)
        self.assertEqual(self.state.get_vgpr(1), 0x07)
        self.assertEqual(self.state.get_vgpr(0), 0x01)
        self.assertEqual(self.state.get_vgpr(2), 0x04)

    def test_dst_aliases_src2(self) -> None:
        # v2 = v0 | v1 | v2
        self.state.set_vgpr(0, 0x100)
        self.state.set_vgpr(1, 0x020)
        self.state.set_vgpr(2, 0x003)
        instr = VOr3B32Instruction(
            dst=Reg(2, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x123)
        self.assertEqual(self.state.get_vgpr(2), 0x123)
        self.assertEqual(self.state.get_vgpr(0), 0x100)
        self.assertEqual(self.state.get_vgpr(1), 0x020)

    def test_dst_aliases_src0_and_src1(self) -> None:
        # v0 = v0 | v0 | v2
        self.state.set_vgpr(0, 0xAA00)
        self.state.set_vgpr(2, 0x0055)
        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xAA55)
        self.assertEqual(self.state.get_vgpr(0), 0xAA55)

    def test_dst_aliases_all_operands_idempotent(self) -> None:
        # v0 = v0 | v0 | v0
        self.state.set_vgpr(0, 0xDEADBEEF)
        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xDEADBEEF)
        self.assertEqual(self.state.get_vgpr(0), 0xDEADBEEF)

    def test_simd_multi_lane_aliasing(self) -> None:
        simd = SIMDInterpreter()
        state = SIMDState(num_lanes=64)
        for lane in range(64):
            state.set_vgpr(lane, 0, lane)
            state.set_vgpr(lane, 1, 0x100)

        # v0 = v0 | 0x200 | v1
        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Literal32(0x200)),
            src2=VOP3Operand(Reg(1, RegType.VGPR)),
        )
        simd.step(instr, state)
        for lane in range(64):
            expected = lane | 0x100 | 0x200
            self.assertEqual(state.get_vgpr(lane, 0), expected)
            self.assertEqual(state.get_vgpr(lane, 1), 0x100)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class Or3StatusFlags:
    """Architectural condition codes and status flags for v_or3_b32."""
    zero: bool
    sign: bool
    all_ones: bool
    parity: bool
    pairwise_overlap: bool
    vcc_modified: bool = False


def evaluate_or3_flags(
    res32: int,
    s0: int,
    s1: int,
    s2: int,
) -> Or3StatusFlags:
    """Evaluate condition codes, parity, and pairwise bit collision."""
    res = res32 & MASK32
    s0_u = s0 & MASK32
    s1_u = s1 & MASK32
    s2_u = s2 & MASK32

    bits_set = bin(res).count("1")
    overlap = ((s0_u & s1_u) | (s0_u & s2_u) | (s1_u & s2_u)) != 0

    return Or3StatusFlags(
        zero=(res == 0),
        sign=bool(res & 0x8000_0000),
        all_ones=(res == MASK32),
        parity=(bits_set % 2) == 0,
        pairwise_overlap=overlap,
        vcc_modified=False,
    )


def compute_u32_or3_carry(s0: int, s1: int, s2: int) -> int:
    """Compute majority function carry across 3 operands: (s0&s1)|(s0&s2)|(s1&s2)."""
    a = s0 & MASK32
    b = s1 & MASK32
    c = s2 & MASK32
    return (a & b) | (a & c) | (b & c)


class TestValuOr3B32ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def test_condition_codes_zero_flag(self) -> None:
        res = oracle_v_or3_b32(0, 0, 0)
        flags = evaluate_or3_flags(res, 0, 0, 0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.all_ones)
        self.assertTrue(flags.parity)
        self.assertFalse(flags.pairwise_overlap)

    def test_condition_codes_sign_flag(self) -> None:
        res = oracle_v_or3_b32(0x80000000, 0x10, 0x20)
        flags = evaluate_or3_flags(res, 0x80000000, 0x10, 0x20)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.all_ones)

    def test_condition_codes_all_ones_flag(self) -> None:
        res = oracle_v_or3_b32(0xFFFF0000, 0x0000FF00, 0x000000FF)
        flags = evaluate_or3_flags(res, 0xFFFF0000, 0x0000FF00, 0x000000FF)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.all_ones)
        self.assertTrue(flags.sign)
        self.assertTrue(flags.parity)  # 32 bits set -> even parity

    def test_condition_codes_parity(self) -> None:
        # 0 bits -> even (True)
        self.assertTrue(evaluate_or3_flags(0, 0, 0, 0).parity)
        # 1 bit -> odd (False)
        self.assertFalse(evaluate_or3_flags(1, 1, 0, 0).parity)
        # 2 bits -> even (True)
        self.assertTrue(evaluate_or3_flags(3, 1, 2, 0).parity)

    def test_pairwise_overlap_and_majority_carry(self) -> None:
        # No overlap across operands: disjoint bits
        res = oracle_v_or3_b32(0x01, 0x02, 0x04)
        flags = evaluate_or3_flags(res, 0x01, 0x02, 0x04)
        self.assertFalse(flags.pairwise_overlap)
        self.assertEqual(compute_u32_or3_carry(0x01, 0x02, 0x04), 0)

        # Overlap between S0 and S2
        res2 = oracle_v_or3_b32(0x05, 0x02, 0x04)
        flags2 = evaluate_or3_flags(res2, 0x05, 0x02, 0x04)
        self.assertTrue(flags2.pairwise_overlap)
        self.assertEqual(compute_u32_or3_carry(0x05, 0x02, 0x04), 0x04)

    def test_vcc_isolation(self) -> None:
        interp = ScalarVALUInterpreter()
        canary_vcc = 0xFEEDFACECAFEBEEF
        state = ScalarMachineState(vcc=canary_vcc)

        instr = VOr3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Literal32(0xFFFFFFFF)),
            src1=VOP3Operand(Literal32(0xFFFFFFFF)),
            src2=VOP3Operand(Literal32(0xFFFFFFFF)),
        )
        interp.step(instr, state)
        self.assertEqual(state.vcc, canary_vcc)

        simd = SIMDInterpreter()
        simd_state = SIMDState(num_lanes=64, vcc=canary_vcc)
        simd.step(instr, simd_state)
        self.assertEqual(simd_state.vcc, canary_vcc)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
QNAN_PAYLOAD_F32: int = 0x7FC01234
SNAN_PAYLOAD_F32: int = 0x7F804321
SUBNORM_MIN_F32: int = 0x00000001
SUBNORM_MAX_F32: int = 0x007FFFFF


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_subnormal_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision subnormal."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


class TestValuOr3B32IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_float32_component_assembly(self) -> None:
        # Reconstruct -3.0 from sign bit (bit 31), exponent 128 (2.0), mantissa 0x00400000 (0.5)
        sign = 0x80000000
        exponent = 128 << 23  # 0x40000000
        mantissa = 0x00400000
        res = oracle_v_or3_b32(sign, exponent, mantissa)
        self.assertEqual(res, 0xC0400000)
        self.assertEqual(bits_to_f32(res), -3.0)

    def test_subnormal_assembly_no_ftz(self) -> None:
        # Assemble subnormal float bits across three separate bit slices
        slice0 = 0x00000001  # bit 0
        slice1 = 0x00000020  # bit 5
        slice2 = 0x00010000  # bit 16
        res = oracle_v_or3_b32(slice0, slice1, slice2)
        expected = 0x00010021
        self.assertEqual(res, expected)
        self.assertTrue(is_subnormal_f32(res))

    def test_nan_payload_composition(self) -> None:
        # Assemble Signaling NaN: exponent 0xFF, quiet bit 0, payload bits
        exp = 0x7F800000
        payload_lo = 0x00000012
        payload_hi = 0x00003400
        res = oracle_v_or3_b32(exp, payload_lo, payload_hi)
        self.assertEqual(res, 0x7F803412)
        self.assertTrue(is_nan_f32(res))
        self.assertEqual(res & 0x00400000, 0)  # quiet bit is 0 (sNaN)

        # Assemble Quiet NaN: set quiet bit 22
        qbit = 0x00400000
        res_q = oracle_v_or3_b32(exp, qbit, payload_lo)
        self.assertEqual(res_q, 0x7FC00012)
        self.assertTrue(is_nan_f32(res_q))
        self.assertNotEqual(res_q & 0x00400000, 0)  # quiet bit is 1 (qNaN)

    def test_signed_zero_and_infinities(self) -> None:
        # +0.0 | 0 | 0 == +0.0
        self.assertEqual(oracle_v_or3_b32(POS_ZERO_F32, 0, 0), POS_ZERO_F32)

        # Negative zero: sign bit | 0 | 0
        self.assertEqual(oracle_v_or3_b32(NEG_ZERO_F32, 0, 0), NEG_ZERO_F32)

        # Positive infinity: 0x7F800000 | 0 | 0
        self.assertEqual(oracle_v_or3_b32(POS_INF_F32, 0, 0), POS_INF_F32)

        # Negative infinity: 0x80000000 | 0x7F800000 | 0 == 0xFF800000
        self.assertEqual(oracle_v_or3_b32(NEG_ZERO_F32, POS_INF_F32, 0), NEG_INF_F32)








if __name__ == "__main__":
    unittest.main()
