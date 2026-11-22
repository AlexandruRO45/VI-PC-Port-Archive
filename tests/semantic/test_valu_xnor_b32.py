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


if __name__ == "__main__":
    unittest.main()


