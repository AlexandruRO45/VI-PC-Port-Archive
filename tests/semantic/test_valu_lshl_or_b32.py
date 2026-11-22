from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_lshl_or_b32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_lshl_or_b32 D, S0, S1, S2
Operation:
    Performs a 32-bit logical shift left of S0 by the shift count specified in S1,
    then bitwise ORs S2 with the shifted result:
        dst = ((S0 << (S1 & 0x1F)) | S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0: 32-bit unsigned integer base value to shift.
    - S1: 32-bit unsigned integer shift count; only low 5 bits [4:0] are used (modulo 32).
    - S2: 32-bit unsigned integer bitmask to OR with the shifted base.
    - Zero-fills vacated bits on the right.
    - Bits shifted left past bit 31 are discarded.

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
SHIFT_MASK: int = 0x1F


def oracle_v_lshl_or_b32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_lshl_or_b32."""
    s0 = src0_bits & MASK32
    shift = src1_bits & SHIFT_MASK
    s2 = src2_bits & MASK32

    shifted = (s0 << shift) & MASK32
    return shifted | s2


class TestValuLshlOrB32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_lshl_or_b32."""

    def test_zero_shift_identity(self) -> None:
        # (s0 << 0) | s2 == s0 | s2
        res = oracle_v_lshl_or_b32(0x12340000, 0, 0x00005678)
        self.assertEqual(res, 0x12345678)

    def test_zero_or_identity(self) -> None:
        # (s0 << s1) | 0 == s0 << s1
        res = oracle_v_lshl_or_b32(0x00000001, 16, 0)
        self.assertEqual(res, 0x00010000)

    def test_shift_count_modulo_32(self) -> None:
        # S1 values 32, 64, 96 mask to shift count 0
        self.assertEqual(oracle_v_lshl_or_b32(0x1234, 32, 0), 0x1234)
        self.assertEqual(oracle_v_lshl_or_b32(0x1234, 64, 0), 0x1234)
        # S1 value 33 masks to shift count 1
        self.assertEqual(oracle_v_lshl_or_b32(0x0001, 33, 0), 0x0002)
        # S1 value 0xFFFFFFFF masks to 31
        self.assertEqual(oracle_v_lshl_or_b32(0x0001, 0xFFFFFFFF, 0), 0x80000000)

    def test_high_bits_discarded_on_shift(self) -> None:
        # (0x80000000 << 1) | 0x00000005 => 0x00000005
        self.assertEqual(oracle_v_lshl_or_b32(0x80000000, 1, 0x00000005), 0x00000005)

    def test_overlapping_bits_bitwise_or(self) -> None:
        # (0x0F << 4) | 0xFF = 0x0F0 | 0x0FF = 0x0FF
        res = oracle_v_lshl_or_b32(0x0F, 4, 0xFF)
        self.assertEqual(res, 0xFF)

    def test_pack_halves_via_shift_and_or(self) -> None:
        # (0x1234 << 16) | 0x5678 = 0x12345678
        res = oracle_v_lshl_or_b32(0x1234, 16, 0x5678)
        self.assertEqual(res, 0x12345678)


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
    V_LSHL_OR_B32 = "v_lshl_or_b32"


@dataclass(frozen=True)
class VLshlOrB32Instruction:
    """Symbolic AST node representing a RDNA2 v_lshl_or_b32 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_LSHL_OR_B32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"{self.opcode.value} {self.dst.name}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


class TestValuLshlOrB32SymbolicAST(unittest.TestCase):
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
        src1 = VOP3Operand(Literal32(8))
        src2 = VOP3Operand(Reg(2, RegType.SGPR))

        inst = VLshlOrB32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_lshl_or_b32 v0, v1, 0x00000008, s2")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VLshlOrB32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)



if __name__ == "__main__":
    unittest.main()
