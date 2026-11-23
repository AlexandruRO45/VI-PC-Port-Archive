from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_mad_i32_i24.

Instruction definition:
    v_mad_i32_i24 vdst, vsrc0, vsrc1, vsrc2 [clamp]

Semantics:
    3-operand signed 24-bit multiply-accumulate to 32-bit destination (RDNA2 VOP3).
    D.i32 = S0.i24 * S1.i24 + S2.i32
    - S0 and S1: bits [23:0] are extracted and sign-extended to 32/64-bit signed integers;
      bits [31:24] are ignored.
    - S2: treated as a full 32-bit signed integer.
    - Product of S0.i24 * S1.i24 is computed exactly (fits in 48 bits, range [-2^46, 2^46]).
    - Added with S2.i32 and truncated to a 32-bit signed/unsigned integer bit pattern.

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
MASK24: int = 0x00FF_FFFF
SIGN24: int = 0x0080_0000

INT24_MIN: int = -8388608        # -2^23 (0x800000)
INT24_MAX: int = 8388607         # 2^23 - 1 (0x7FFFFF)
INT32_MIN: int = -2147483648     # -2^31 (0x80000000)
INT32_MAX: int = 2147483647      # 2^31 - 1 (0x7FFFFFFF)


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def sign_extend_i24(val: int) -> int:
    """Extract lower 24 bits and sign-extend to integer."""
    val24 = val & MASK24
    if val24 & SIGN24:
        return val24 - 0x1000000
    return val24


def sign_extend_i32(val: int) -> int:
    """Treat 32-bit value as signed integer."""
    val32 = val & MASK32
    if val32 & 0x8000_0000:
        return val32 - 0x1_0000_0000
    return val32


def ref_mad_i32_i24(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Architectural reference oracle for v_mad_i32_i24."""
    s0 = sign_extend_i24(src0)
    s1 = sign_extend_i24(src1)
    s2 = sign_extend_i32(src2)

    product = s0 * s1
    res = product + s2

    if clamp:
        if res < INT32_MIN:
            res = INT32_MIN
        elif res > INT32_MAX:
            res = INT32_MAX

    return res & MASK32


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def i32(self) -> int:
        return sign_extend_i32(self.raw)

    @property
    def i24(self) -> int:
        return sign_extend_i24(self.raw)


class TestMadI32I24ReferenceOracle(unittest.TestCase):
    def test_basic_arithmetic(self):
        # 2 * 3 + 4 = 10
        self.assertEqual(ref_mad_i32_i24(2, 3, 4), 10)
        # 0 * 100 + 42 = 42
        self.assertEqual(ref_mad_i32_i24(0, 100, 42), 42)

    def test_signed_negative_sources(self):
        # (-2) * 3 + 4 = -2 (0xFFFFFFFE)
        self.assertEqual(ref_mad_i32_i24(-2, 3, 4), 0xFFFF_FFFE)
        # 2 * (-3) + 4 = -2 (0xFFFFFFFE)
        self.assertEqual(ref_mad_i32_i24(2, -3, 4), 0xFFFF_FFFE)
        # (-2) * (-3) + 4 = 10
        self.assertEqual(ref_mad_i32_i24(-2, -3, 4), 10)
        # 2 * 3 + (-10) = -4 (0xFFFFFFFC)
        self.assertEqual(ref_mad_i32_i24(2, 3, -10), 0xFFFF_FFFC)

    def test_upper_8_bits_ignored(self):
        # 0x55000002 has 2 in lower 24 bits
        self.assertEqual(ref_mad_i32_i24(0x55000002, 3, 4), 10)
        # 0xAAFFFFFE has -2 in lower 24 bits
        self.assertEqual(ref_mad_i32_i24(0xAAFFFFFE, 3, 4), 0xFFFF_FFFE)

    def test_24bit_extremes(self):
        # Max positive 24-bit: 0x7FFFFF (8388607) * 2 + 0 = 16777214 (0x00FFFFFE)
        self.assertEqual(ref_mad_i32_i24(0x7FFFFF, 2, 0), 16777214)
        # Min negative 24-bit: 0x800000 (-8388608) * 2 + 0 = -16777216 (0xFF000000)
        self.assertEqual(ref_mad_i32_i24(0x800000, 2, 0), 0xFF00_0000)

    def test_overflow_and_wrap(self):
        # 0x7FFFFF * 0x7FFFFF + 0 = 2^46 - 2^24 + 1 -> lower 32 bits: 0xFF000001
        self.assertEqual(ref_mad_i32_i24(0x7FFFFF, 0x7FFFFF, 0), 0xFF00_0001)


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
class VMadI32I24Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]
    src2: Union[RegisterOperand, LiteralOperand]
    clamp: bool = False

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_mad_i32_i24 must be a VGPR")


class TestMadI32I24AST(unittest.TestCase):
    def test_valid_ast_construction(self):
        dst = RegisterOperand(OperandType.VGPR, 0)
        s0 = RegisterOperand(OperandType.VGPR, 1)
        s1 = RegisterOperand(OperandType.VGPR, 2)
        s2 = RegisterOperand(OperandType.VGPR, 3)
        insn = VMadI32I24Instruction(dst=dst, src0=s0, src1=s1, src2=s2, clamp=True)
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
            VMadI32I24Instruction(
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
