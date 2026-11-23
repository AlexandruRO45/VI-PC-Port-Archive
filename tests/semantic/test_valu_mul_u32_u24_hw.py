from __future__ import annotations

"""Semantic reference oracle and architectural model for v_mul_u32_u24.

RDNA2 / GFX10 VOP2 24-bit Unsigned Integer Multiply:
Multiplies the low 24 bits of two unsigned integer source operands (bits [31:24] ignored)
and returns the lower 32 bits of the 48-bit product into a 32-bit VGPR.

Syntax:
    v_mul_u32_u24 vdst, vsrc0, vsrc1

Semantics:
    s0_u24 = src0 & 0x00FF_FFFF
    s1_u24 = src1 & 0x00FF_FFFF
    product_48 = s0_u24 * s1_u24
    vdst = product_48 & 0xFFFF_FFFF
"""

import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si
import support

# 24-bit and 32-bit Constants
MASK_24: int = 0x00FF_FFFF
MASK_32: int = 0xFFFF_FFFF
MAX_U24: int = 0x00FF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def u24(val: int) -> int:
    """Normalize integer to unsigned 24-bit integer."""
    return val & MASK_24


def oracle_mul_u32_u24(src0: int, src1: int) -> int:
    """Reference mathematical oracle for v_mul_u32_u24."""
    s0 = u24(src0)
    s1 = u24(src1)
    return u32(s0 * s1)


def run_translator_mul_u24(
    src0: int,
    src1: int,
    name: str = "ps_mul_u24",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    lines = [
        f"v_mov_b32 v1, {u32(src0):#x}",
        f"v_mov_b32 v2, {u32(src1):#x}",
        "v_mul_u32_u24 v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_multiplication(self) -> None:
        """Verify 100 * 250 = 25000."""
        got = oracle_mul_u32_u24(100, 250)
        self.assertEqual(got, 25000)

    def test_high_byte_masking(self) -> None:
        """Verify bits [31:24] are strictly ignored on both operands."""
        # 0xFF000002 low 24 bits = 2, 0xAA000003 low 24 bits = 3 -> 6
        got = oracle_mul_u32_u24(0xFF00_0002, 0xAA00_0003)
        self.assertEqual(got, 6)

    def test_max_u24_product(self) -> None:
        """Verify (2^24 - 1) * (2^24 - 1) low 32 bits equals 0xFE000001."""
        # (0xFFFFFF * 0xFFFFFF) = 0xFFFFFE000001; low 32 bits = 0xFE000001
        got = oracle_mul_u32_u24(MAX_U24, MAX_U24)
        self.assertEqual(got, 0xFE00_0001)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly against translator output."""
        src0 = 0x1234_5678
        src1 = 0xABCD_EF02
        expected = oracle_mul_u32_u24(src0, src1)
        translated = run_translator_mul_u24(src0, src1, name="ps_mul_oracle")
        self.assertEqual(translated, expected)


# ==============================================================================
# SECTION 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMMEDIATE = auto()


@dataclass(frozen=True)
class Operand:
    kind: OperandKind
    value: int

    def __post_init__(self) -> None:
        if self.kind == OperandKind.VGPR:
            if not (0 <= self.value <= 255):
                raise ValueError(f"VGPR index out of range: {self.value}")
        elif self.kind == OperandKind.SGPR:
            if not (0 <= self.value <= 105):
                raise ValueError(f"SGPR index out of range: {self.value}")
        elif self.kind == OperandKind.IMMEDIATE:
            if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
                raise ValueError(f"Immediate value out of 32-bit range: {self.value}")


def VGPR(idx: int) -> Operand:
    return Operand(OperandKind.VGPR, idx)


def SGPR(idx: int) -> Operand:
    return Operand(OperandKind.SGPR, idx)


def Immediate(val: int) -> Operand:
    return Operand(OperandKind.IMMEDIATE, u32(val))


@dataclass(frozen=True)
class VMulU24Instruction:
    dst: Operand
    src0: Operand
    src1: Operand

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")

    def format_operand(self, op: Operand) -> str:
        if op.kind == OperandKind.VGPR:
            return f"v{op.value}"
        elif op.kind == OperandKind.SGPR:
            return f"s{op.value}"
        else:
            return f"{op.value:#x}"

    def to_assembly(self) -> str:
        d = self.format_operand(self.dst)
        s0 = self.format_operand(self.src0)
        s1 = self.format_operand(self.src1)
        return f"v_mul_u32_u24 {d}, {s0}, {s1}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly serializes to valid RDNA2 assembly text."""
        instr = VMulU24Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        asm = instr.to_assembly()
        self.assertEqual(asm, "v_mul_u32_u24 v0, v1, v2")

    def test_ast_translation_validity(self) -> None:
        """Verify that assembly generated by AST translates cleanly."""
        instr = VMulU24Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 250",
            instr.to_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_mul_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 25000)
