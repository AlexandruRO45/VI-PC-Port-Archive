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


if __name__ == "__main__":
    unittest.main()
