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


if __name__ == "__main__":
    unittest.main()

