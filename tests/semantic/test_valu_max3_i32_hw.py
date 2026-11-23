from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_max3_i32.

Instruction definition:
    v_max3_i32 vdst, vsrc0, vsrc1, vsrc2 [clamp]

Semantics:
    3-operand signed 32-bit integer maximum (RDNA2 VOP3).
    D.i32 = max(S0.i32, S1.i32, S2.i32)
    Operands are interpreted as two's complement signed 32-bit integers in [-2^31, 2^31 - 1].
    The largest algebraic value is written to the 32-bit destination VGPR.

Standard library only: struct, math, unittest. Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
INT32_MIN: int = -2147483648     # -2^31 (0x80000000)
INT32_MAX: int = 2147483647      # 2^31 - 1 (0x7FFFFFFF)


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def to_i32(val: int) -> int:
    """Interpret 32-bit integer as two's complement signed integer."""
    val32 = val & MASK32
    if val32 & 0x8000_0000:
        return val32 - 0x1_0000_0000
    return val32


def ref_max3_i32(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Architectural reference oracle for v_max3_i32."""
    i0 = to_i32(src0)
    i1 = to_i32(src1)
    i2 = to_i32(src2)

    res = max(i0, i1, i2)
    if clamp:
        if res < 0:
            res = 0

    return res & MASK32


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def i32(self) -> int:
        return to_i32(self.raw)


class TestMax3I32ReferenceOracle(unittest.TestCase):
    def test_all_positive_values(self):
        self.assertEqual(ref_max3_i32(10, 20, 30), 30)
        self.assertEqual(ref_max3_i32(100, 50, 75), 100)

    def test_mixed_signs(self):
        # max(-5, 10, -2) = 10
        self.assertEqual(ref_max3_i32(-5, 10, -2), 10)
        # max(-100, -200, 0) = 0
        self.assertEqual(ref_max3_i32(-100, -200, 0), 0)

    def test_all_negative_values(self):
        # max(-10, -5, -20) = -5 (0xFFFFFFFB)
        self.assertEqual(ref_max3_i32(-10, -5, -20), 0xFFFF_FFFB)
        self.assertEqual(ref_max3_i32(-1, -2, -3), 0xFFFF_FFFF)

    def test_boundary_extremes(self):
        # max(INT32_MIN, 0, INT32_MAX) = INT32_MAX
        self.assertEqual(ref_max3_i32(INT32_MIN, 0, INT32_MAX), INT32_MAX)
        # max(INT32_MIN, INT32_MIN + 1, -1) = -1
        self.assertEqual(ref_max3_i32(INT32_MIN, INT32_MIN + 1, -1), 0xFFFF_FFFF)

    def test_identical_operands(self):
        self.assertEqual(ref_max3_i32(42, 42, 42), 42)
        self.assertEqual(ref_max3_i32(-7, -7, -7), 0xFFFF_FFF9)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    @property
    def name(self) -> str:
        prefix = "v" if self.op_type == OperandType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= MASK32):
            raise ValueError(f"Literal 32-bit value out of range: {self.value}")


@dataclass
class VMax3I32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]
    src2: Union[RegisterOperand, LiteralOperand]
    clamp: bool = False

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_max3_i32 must be a VGPR")


class TestMax3I32AST(unittest.TestCase):
    def test_valid_ast_construction(self):
        dst = RegisterOperand(OperandType.VGPR, 0)
        s0 = RegisterOperand(OperandType.VGPR, 1)
        s1 = RegisterOperand(OperandType.VGPR, 2)
        s2 = RegisterOperand(OperandType.VGPR, 3)
        insn = VMax3I32Instruction(dst=dst, src0=s0, src1=s1, src2=s2, clamp=True)
        self.assertEqual(insn.dst.name, "v0")
        self.assertEqual(insn.src0.name, "v1")
        self.assertEqual(insn.src1.name, "v2")
        self.assertEqual(insn.src2.name, "v3")
        self.assertTrue(insn.clamp)

    def test_invalid_register_indices(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)

    def test_invalid_dst_type(self):
        with self.assertRaises(TypeError):
            VMax3I32Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 1),
                src1=RegisterOperand(OperandType.VGPR, 2),
                src2=RegisterOperand(OperandType.VGPR, 3),
            )

    def test_literal_operand_bounds(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)
        with self.assertRaises(ValueError):
            LiteralOperand(-0x8000_0001)


if __name__ == '__main__':
    unittest.main()
