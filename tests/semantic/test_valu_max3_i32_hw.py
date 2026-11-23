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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachine:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.scc: int = 0

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand type: {type(op)}")

    def write_vgpr(self, index: int, val32: int):
        self.vgprs[index] = val32 & MASK32

    def step(self, insn: VMax3I32Instruction):
        val_s0 = self.read_operand(insn.src0)
        val_s1 = self.read_operand(insn.src1)
        val_s2 = self.read_operand(insn.src2)
        result = ref_max3_i32(val_s0, val_s1, val_s2, clamp=insn.clamp)
        self.write_vgpr(insn.dst.index, result)


class TestMax3I32ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.machine = ScalarMachine()

    def test_scalar_execution_basic(self):
        self.machine.vgprs[1] = 0xFFFF_FFF6  # -10
        self.machine.vgprs[2] = 0xFFFF_FFFB  # -5
        self.machine.vgprs[3] = 0xFFFF_FFEC  # -20
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 0xFFFF_FFFB)

    def test_scalar_with_sgpr_and_literal(self):
        self.machine.sgprs[4] = 10
        self.machine.vgprs[2] = 20
        lit = LiteralOperand(-30)
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 5),
            src0=RegisterOperand(OperandType.SGPR, 4),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=lit,
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[5], 20)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDWave:
    def __init__(self, wave_size: int = 64):
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.sgprs: List[int] = [0] * 106
        self.exec_mask: int = (1 << wave_size) - 1

    def is_lane_active(self, lane_id: int) -> bool:
        return bool(self.exec_mask & (1 << lane_id))

    def step(self, insn: VMax3I32Instruction):
        for lane in range(self.wave_size):
            if not self.is_lane_active(lane):
                continue
            # Read src0
            if isinstance(insn.src0, LiteralOperand):
                val_s0 = insn.src0.value & MASK32
            elif insn.src0.op_type == OperandType.VGPR:
                val_s0 = self.vgprs[insn.src0.index][lane] & MASK32
            else:
                val_s0 = self.sgprs[insn.src0.index] & MASK32

            # Read src1
            if isinstance(insn.src1, LiteralOperand):
                val_s1 = insn.src1.value & MASK32
            elif insn.src1.op_type == OperandType.VGPR:
                val_s1 = self.vgprs[insn.src1.index][lane] & MASK32
            else:
                val_s1 = self.sgprs[insn.src1.index] & MASK32

            # Read src2
            if isinstance(insn.src2, LiteralOperand):
                val_s2 = insn.src2.value & MASK32
            elif insn.src2.op_type == OperandType.VGPR:
                val_s2 = self.vgprs[insn.src2.index][lane] & MASK32
            else:
                val_s2 = self.sgprs[insn.src2.index] & MASK32

            res = ref_max3_i32(val_s0, val_s1, val_s2, clamp=insn.clamp)
            self.vgprs[insn.dst.index][lane] = res & MASK32


class TestMax3I32SIMDDivergence(unittest.TestCase):
    def test_wave32_alternating_exec(self):
        wave = SIMDWave(wave_size=32)
        wave.exec_mask = 0x5555_5555  # Even lanes active
        for lane in range(32):
            wave.vgprs[0][lane] = 0xDEAD_BEEF
            wave.vgprs[1][lane] = to_i32(-lane)
            wave.vgprs[2][lane] = 0
            wave.vgprs[3][lane] = to_i32(lane)

        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[0][lane], lane)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xDEAD_BEEF)

    def test_wave64_divergence(self):
        wave = SIMDWave(wave_size=64)
        wave.exec_mask = 0xFFFF_FFFF_0000_0000  # Top 32 lanes active
        for lane in range(64):
            wave.vgprs[10][lane] = 0xCAFE_0000
            wave.vgprs[1][lane] = -10
            wave.vgprs[2][lane] = lane - 40
            wave.vgprs[3][lane] = -5

        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 10),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)

        for lane in range(32):
            self.assertEqual(wave.vgprs[10][lane], 0xCAFE_0000)
        for lane in range(32, 64):
            expected = ref_max3_i32(-10, lane - 40, -5)
            self.assertEqual(wave.vgprs[10][lane], expected)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestMax3I32RegisterAliasing(unittest.TestCase):
    def test_dst_aliases_src0(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 10
        machine.vgprs[1] = 50
        machine.vgprs[2] = 20
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
            src1=RegisterOperand(OperandType.VGPR, 1),
            src2=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 50)

    def test_dst_aliases_src2(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 50
        machine.vgprs[1] = 10
        machine.vgprs[2] = 20
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 0),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 50)

    def test_dst_aliases_all_sources(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 0xFFFF_FFFE  # -2
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
            src1=RegisterOperand(OperandType.VGPR, 0),
            src2=RegisterOperand(OperandType.VGPR, 0),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 0xFFFF_FFFE)


if __name__ == '__main__':
    unittest.main()
