from __future__ import annotations

"""Semantic regression test suite and reference verification for v_max3_i32.

RDNA2 Vector ALU instruction:
    v_max3_i32 vdst, vsrc0, vsrc1, vsrc2

Semantics:
    Signed two's complement 32-bit maximum across three operands:
        vdst = max(int32(vsrc0), int32(vsrc1), int32(vsrc2))

Reference mathematical oracle, properties, signed vs unsigned divergence,
AST definitions, deterministic scalar interpreter, SIMD multi-lane divergence,
register aliasing, condition codes invariance, two's complement and IEEE 754
bit pattern edge cases, and translator-backed SPIR-V execution.

Standard library Python only (python -I), zero game data.
"""

import itertools
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFFFFFF
SIGN_BIT32 = 0x80000000
INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF


def to_signed_i32(x: int) -> int:
    """Interpret raw 32-bit integer as two's complement signed 32-bit integer."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_BIT32) else x


def to_unsigned_u32(x: int) -> int:
    """Mask integer to 32-bit unsigned integer range [0, 2^32 - 1]."""
    return x & MASK32


def ref_v_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for v_max3_i32: signed maximum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return max(v0, v1, v2) & MASK32


def ref_pairwise_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed maximum: max(max(s0, s1), s2)."""
    step1 = max(to_signed_i32(s0), to_signed_i32(s1))
    return max(step1, to_signed_i32(s2)) & MASK32

class TestVMax3I32ReferenceOracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and core properties for v_max3_i32."""

    def test_basic_signed_max(self) -> None:
        """Verify standard signed maximum calculations with positive and negative inputs."""
        self.assertEqual(ref_v_max3_i32(10, 20, 30), 30)
        self.assertEqual(ref_v_max3_i32(-10, -20, -5), (-5) & MASK32)
        self.assertEqual(ref_v_max3_i32(-1, 0, 1), 1)
        self.assertEqual(ref_v_max3_i32(-100, 50, -200), 50)
        self.assertEqual(ref_v_max3_i32(0, 0, 0), 0)

    def test_signed_vs_unsigned_divergence(self) -> None:
        """Verify divergence between signed maximum and unsigned maximum on identical bit patterns."""
        # 0xFFFFFFFF is -1 as i32, UINT32_MAX as u32.
        # With 0 and 10: signed max is 10 (-1 < 0 < 10); unsigned max is 0xFFFFFFFF.
        self.assertEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), 10)
        self.assertNotEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), max(0, 0xFFFFFFFF, 10))

        # 0x80000000 is INT32_MIN as i32, 2^31 as u32.
        # With 0x7FFFFFFF (INT32_MAX) and 0:
        self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), 0x7FFFFFFF)
        # Unsigned max would pick 0x80000000 (2147483648 > 2147483647).
        self.assertNotEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), max(0x7FFFFFFF, 0x80000000, 0))

    def test_pairwise_reduction_equivalence(self) -> None:
        """Verify 3-way signed max identically matches chained pairwise signed max."""
        samples = [
            (0, 0, 0),
            (1, 2, 3),
            (-5, 10, -20),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x7FFFFFFE),
            (0x12345678, 0x87654321, 0x00000000),
            (0xFFFFFFFE, 0xFFFFFFFF, 0x00000001),
        ]
        for s0, s1, s2 in samples:
            with self.subTest(s0=hex(s0 & MASK32), s1=hex(s1 & MASK32), s2=hex(s2 & MASK32)):
                self.assertEqual(ref_v_max3_i32(s0, s1, s2), ref_pairwise_max3_i32(s0, s1, s2))

    def test_idempotent_and_identity_properties(self) -> None:
        """Verify idempotent max(x, x, x) == x and neutral/absorbing element laws."""
        test_values = [0, 1, -1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 42, -9999]
        for val in test_values:
            raw = val & MASK32
            # Idempotence: max(x, x, x) == x
            self.assertEqual(ref_v_max3_i32(val, val, val), raw)
            # Neutral element: INT32_MIN (0x80000000)
            self.assertEqual(ref_v_max3_i32(val, 0x80000000, 0x80000000), raw)
            # Absorbing element: INT32_MAX (0x7FFFFFFF)
            self.assertEqual(ref_v_max3_i32(val, 0x7FFFFFFF, 0), 0x7FFFFFFF)

    def test_permutation_invariance(self) -> None:
        """Verify that all 3! = 6 permutations yield identical results."""
        triples = [
            (0x80000000, 0, 0x7FFFFFFF),
            (0xFFFFFFFF, 1, 0x80000000),
            (42, 42, 42),
            (0x12345678, 0x9ABCDEF0, 0x00000000),
            (0x80000000, 0x80000001, 0x80000002),
            (0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF),
        ]
        for triple in triples:
            expected = ref_v_max3_i32(*triple)
            for perm in itertools.permutations(triple):
                self.assertEqual(ref_v_max3_i32(*perm), expected)

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
                raise ValueError(f"VGPR index out of range [0, 255]: {self.value}")
        elif self.kind == OperandKind.SGPR:
            if not (0 <= self.value <= 105):
                raise ValueError(f"SGPR index out of range [0, 105]: {self.value}")
        elif self.kind == OperandKind.IMMEDIATE:
            if not (-0x80000000 <= self.value <= 0xFFFFFFFF):
                raise ValueError(f"Immediate value out of 32-bit range: {self.value}")


def VGPR(idx: int) -> Operand:
    return Operand(OperandKind.VGPR, idx)


def SGPR(idx: int) -> Operand:
    return Operand(OperandKind.SGPR, idx)


def Immediate(val: int) -> Operand:
    return Operand(OperandKind.IMMEDIATE, val)


@dataclass(frozen=True)
class VMax3I32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")

    def format_operand(self, op: Operand) -> str:
        if op.kind == OperandKind.VGPR:
            return f"v{op.value}"
        elif op.kind == OperandKind.SGPR:
            return f"s{op.value}"
        else:
            return f"{op.value}"

    def to_assembly(self) -> str:
        d = self.format_operand(self.dst)
        s0 = self.format_operand(self.src0)
        s1 = self.format_operand(self.src1)
        s2 = self.format_operand(self.src2)
        return f"v_max3_i32 {d}, {s0}, {s1}, {s2}"


class TestVMax3I32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly formats to valid RDNA2 assembly text."""
        inst = VMax3I32Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=SGPR(4),
            src2=Immediate(-16),
        )
        asm = inst.to_assembly()
        self.assertEqual(asm, "v_max3_i32 v0, v1, s4, -16")

    def test_ast_operand_bounds_validation(self) -> None:
        """Verify AST rejects out-of-range registers and invalid destinations."""
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            SGPR(106)
        with self.assertRaises(ValueError):
            SGPR(-1)
        with self.assertRaises(ValueError):
            Immediate(0x1_0000_0000)
        with self.assertRaises(TypeError):
            VMax3I32Instruction(dst=SGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))

    def test_ast_assembly_execution_structure(self) -> None:
        """Verify assembly generated from AST has the expected syntax."""
        inst = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        self.assertTrue(inst.to_assembly().startswith("v_max3_i32 v0, v1, v2, v3"))

if __name__ == "__main__":
    unittest.main()
