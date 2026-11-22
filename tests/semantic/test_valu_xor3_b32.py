from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_xor3_b32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_xor3_b32 D, S0, S1, S2
Operation:
    Performs a 3-way bitwise exclusive OR across three 32-bit operands:
        dst = (S0 ^ S1 ^ S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0, S1, S2: 32-bit unsigned integer values.
    - Full bitwise XOR associativity and commutativity.
    - Zero identity: S0 ^ 0 ^ 0 == S0.
    - Self-inversion / cancellation: S0 ^ S0 ^ S2 == S2.
    - Inversion via all-ones: S0 ^ 0xFFFFFFFF ^ 0 == ~S0 & 0xFFFFFFFF.
    - Bit parity calculation: computes odd bit parity across three inputs per bit position.

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


def oracle_v_xor3_b32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_xor3_b32."""
    s0 = src0_bits & MASK32
    s1 = src1_bits & MASK32
    s2 = src2_bits & MASK32
    return (s0 ^ s1 ^ s2) & MASK32


class TestValuXor3B32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_xor3_b32."""

    def test_zero_identity(self) -> None:
        # S0 ^ 0 ^ 0 == S0
        res = oracle_v_xor3_b32(0x12345678, 0, 0)
        self.assertEqual(res, 0x12345678)

    def test_self_cancellation(self) -> None:
        # S ^ S ^ Other == Other
        val = 0xA5A5A5A5
        other = 0x5A5A5A5A
        self.assertEqual(oracle_v_xor3_b32(val, val, other), other)

    def test_triple_identity(self) -> None:
        # S ^ S ^ S == S
        val = 0x12345678
        self.assertEqual(oracle_v_xor3_b32(val, val, val), val)

    def test_all_ones_inversion(self) -> None:
        # S ^ 0xFFFFFFFF ^ 0 == ~S
        val = 0x00FF00FF
        self.assertEqual(oracle_v_xor3_b32(val, 0xFFFFFFFF, 0), 0xFF00FF00)

    def test_commutativity_and_associativity(self) -> None:
        a, b, c = 0x12345678, 0x9ABCDEF0, 0x5555AAAA
        res1 = oracle_v_xor3_b32(a, b, c)
        res2 = oracle_v_xor3_b32(c, a, b)
        res3 = oracle_v_xor3_b32(b, c, a)
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
    V_XOR3_B32 = "v_xor3_b32"


@dataclass(frozen=True)
class VXor3B32Instruction:
    """Symbolic AST node representing a RDNA2 v_xor3_b32 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_XOR3_B32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"{self.opcode.value} {self.dst.name}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


class TestValuXor3B32SymbolicAST(unittest.TestCase):
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

        inst = VXor3B32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_xor3_b32 v0, v1, 0x000000FF, s2")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VXor3B32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)


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
    """Deterministic scalar instruction interpreter for v_xor3_b32."""

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

    def step(self, instr: VXor3B32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)
        s2_raw = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_xor3_b32(s0_raw, s1_raw, s2_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuXor3B32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 0xFF00)
        self.state.set_vgpr(2, 0x0FF0)
        self.state.set_vgpr(3, 0x00FF)
        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xF00F)
        self.assertEqual(self.state.get_vgpr(0), 0xF00F)

    def test_step_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 0xAA000000)
        self.state.set_vgpr(1, 0x00BB0000)
        instr = VXor3B32Instruction(
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
        instr = VXor3B32Instruction(
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
        instr = VXor3B32Instruction(
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

    def step(self, instr: VXor3B32Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_xor3_b32(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuXor3B32SIMDDivergence(unittest.TestCase):
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

        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = lane ^ 0x100 ^ 0x1000
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, 1 << (lane % 32))
            state.set_vgpr(lane, 2, 0x40000000)
            state.set_vgpr(lane, 3, 0x80000000)

        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = (1 << (lane % 32)) ^ 0xC0000000
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x01)
            state.set_vgpr(lane, 2, 0x03)
            state.set_vgpr(lane, 3, 0x07)

        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = 0x01 ^ 0x03 ^ 0x07  # 0x05
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
            state.set_vgpr(lane, 2, 0x55)
            state.set_vgpr(lane, 3, 0xFF)

        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_lane_active(self) -> None:
        target_lane = 25
        mask = 1 << target_lane
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x11)
            state.set_vgpr(lane, 2, 0x22)
            state.set_vgpr(lane, 3, 0x44)

        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(results[lane], 0x77)
                self.assertEqual(state.get_vgpr(lane, 0), 0x77)
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

        instr = VXor3B32Instruction(
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

        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(5, RegType.SGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(16):
            expected = 0xF0FF0000 ^ lane
            self.assertEqual(state.get_vgpr(lane, 0), expected)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuXor3B32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_dst_aliases_src0(self) -> None:
        # v0 = v0 ^ v1 ^ v2
        self.state.set_vgpr(0, 0x11)
        self.state.set_vgpr(1, 0x22)
        self.state.set_vgpr(2, 0x44)
        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x77)
        self.assertEqual(self.state.get_vgpr(0), 0x77)
        self.assertEqual(self.state.get_vgpr(1), 0x22)
        self.assertEqual(self.state.get_vgpr(2), 0x44)

    def test_dst_aliases_src1(self) -> None:
        # v1 = v0 ^ v1 ^ v2
        self.state.set_vgpr(0, 0x01)
        self.state.set_vgpr(1, 0x02)
        self.state.set_vgpr(2, 0x04)
        instr = VXor3B32Instruction(
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
        # v2 = v0 ^ v1 ^ v2
        self.state.set_vgpr(0, 0x100)
        self.state.set_vgpr(1, 0x020)
        self.state.set_vgpr(2, 0x003)
        instr = VXor3B32Instruction(
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

    def test_dst_aliases_src0_and_src1_self_cancellation(self) -> None:
        # v0 = v0 ^ v0 ^ v2 == v2
        self.state.set_vgpr(0, 0xDEADBEEF)
        self.state.set_vgpr(2, 0xCAFEBABE)
        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xCAFEBABE)
        self.assertEqual(self.state.get_vgpr(0), 0xCAFEBABE)

    def test_dst_aliases_all_operands_triple_identity(self) -> None:
        # v0 = v0 ^ v0 ^ v0 == v0
        self.state.set_vgpr(0, 0x13579BDF)
        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
            src2=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x13579BDF)
        self.assertEqual(self.state.get_vgpr(0), 0x13579BDF)

    def test_simd_multi_lane_aliasing(self) -> None:
        simd = SIMDInterpreter()
        state = SIMDState(num_lanes=64)
        for lane in range(64):
            state.set_vgpr(lane, 0, lane)
            state.set_vgpr(lane, 1, 0x100)

        # v0 = v0 ^ 0x200 ^ v1
        instr = VXor3B32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Literal32(0x200)),
            src2=VOP3Operand(Reg(1, RegType.VGPR)),
        )
        simd.step(instr, state)
        for lane in range(64):
            expected = lane ^ 0x100 ^ 0x200
            self.assertEqual(state.get_vgpr(lane, 0), expected)
            self.assertEqual(state.get_vgpr(lane, 1), 0x100)


if __name__ == "__main__":
    unittest.main()
