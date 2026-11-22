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




if __name__ == "__main__":
    unittest.main()
