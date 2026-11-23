from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_exp_i32_f32.

Instruction definition:
    v_frexp_exp_i32_f32 vdst, vsrc0

Semantics:
    Extracts the binary exponent of a single-precision floating-point value S0,
    such that S0 = mantissa * 2^exponent with mantissa in [0.5, 1.0).
    The exponent is returned as a 32-bit signed two's complement integer in destination VGPR.
    For normal binary32 numbers (biased exponent e in 1..254):
        exp = e - 126
    For zero (0.0, -0.0), denormals, infinities (+inf, -inf), and NaNs:
        exp = 0

Zero game data, standard library Python only (python -I, struct, math, unittest).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
EXP_MASK32: int = 0x00FF
EXP_SHIFT32: int = 23
MANT_MASK32: int = 0x007F_FFFF
EXP_BIAS_FREXP: int = 126

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000
QNAN: int = 0x7FC0_0000
SNAN: int = 0x7F80_0001
MIN_NORMAL: int = 0x0080_0000       # 2^-126
MAX_NORMAL: int = 0x7F7F_FFFF       # approx 3.4028235e+38


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def to_i32(val: int) -> int:
    """Interpret 32-bit integer as two's complement signed integer."""
    val32 = val & MASK32
    if val32 & SIGN_BIT32:
        return val32 - 0x1_0000_0000
    return val32


def f32_to_bits(f: float) -> int:
    """Pack Python float to 32-bit IEEE 754 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    """Unpack 32-bit integer bit pattern to Python float."""
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def is_normal_f32(u: int) -> bool:
    """Check if bit pattern represents a normal IEEE 754 single-precision float."""
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    return 1 <= e <= 254


def ref_frexp_exp_i32_f32(src: int) -> int:
    """Architectural reference oracle for v_frexp_exp_i32_f32.
    Computes 32-bit signed exponent in [0.5, 1.0) representation."""
    u = src & MASK32
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    if 1 <= e <= 254:
        exp = e - EXP_BIAS_FREXP
        return exp & MASK32
    return 0


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def i32(self) -> int:
        return to_i32(self.raw)

    @property
    def f32(self) -> float:
        return bits_to_f32(self.raw)


class TestFrexpExpI32F32Oracle(unittest.TestCase):
    def test_powers_of_two(self):
        # 1.0f = 0.5 * 2^1 -> exp = 1
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(1.0)), 1)
        # 0.5f = 0.5 * 2^0 -> exp = 0
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.5)), 0)
        # 2.0f = 0.5 * 2^2 -> exp = 2
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(2.0)), 2)
        # 4.0f = 0.5 * 2^3 -> exp = 3
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(4.0)), 3)
        # 0.25f = 0.5 * 2^-1 -> exp = -1 (0xFFFFFFFF)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.25)), 0xFFFF_FFFF)
        # 0.125f = 0.5 * 2^-2 -> exp = -2 (0xFFFFFFFE)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.125)), 0xFFFF_FFFE)

    def test_sign_invariance(self):
        # Negative floats have identical exponent to positive counterparts
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-1.0)), 1)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-0.5)), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-2.0)), 2)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-0.25)), 0xFFFF_FFFF)

    def test_arbitrary_normals(self):
        # 1.5 = 0.75 * 2^1 -> exp = 1
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(1.5)), 1)
        # 3.0 = 0.75 * 2^2 -> exp = 2
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(3.0)), 2)
        # 100.0 = 0.78125 * 2^7 -> exp = 7
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(100.0)), 7)

    def test_boundary_normals(self):
        # Minimum positive normal: e = 1 -> exp = 1 - 126 = -125 (0xFFFFFF83)
        self.assertEqual(ref_frexp_exp_i32_f32(MIN_NORMAL), 0xFFFF_FF83)
        self.assertEqual(to_i32(ref_frexp_exp_i32_f32(MIN_NORMAL)), -125)
        # Maximum positive normal: e = 254 -> exp = 254 - 126 = 128 (0x00000080)
        self.assertEqual(ref_frexp_exp_i32_f32(MAX_NORMAL), 128)
        # Negative boundary normals
        self.assertEqual(ref_frexp_exp_i32_f32(MIN_NORMAL | SIGN_BIT32), 0xFFFF_FF83)
        self.assertEqual(ref_frexp_exp_i32_f32(MAX_NORMAL | SIGN_BIT32), 128)

    def test_special_values_return_zero(self):
        # Zeros
        self.assertEqual(ref_frexp_exp_i32_f32(POS_ZERO), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(NEG_ZERO), 0)
        # Infinities
        self.assertEqual(ref_frexp_exp_i32_f32(POS_INF), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(NEG_INF), 0)
        # NaNs
        self.assertEqual(ref_frexp_exp_i32_f32(QNAN), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(SNAN), 0)
        # Subnormals (denormals)
        self.assertEqual(ref_frexp_exp_i32_f32(0x0000_0001), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x007F_FFFF), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x8000_0001), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x807F_FFFF), 0)


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
class VFrexpExpI32F32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_frexp_exp_i32_f32 must be a VGPR")


class TestFrexpExpI32F32AST(unittest.TestCase):
    def test_valid_ast_construction(self):
        dst = RegisterOperand(OperandType.VGPR, 0)
        src0 = RegisterOperand(OperandType.VGPR, 1)
        insn = VFrexpExpI32F32Instruction(dst=dst, src0=src0)
        self.assertEqual(insn.dst.name, "v0")
        self.assertEqual(insn.src0.name, "v1")

    def test_sgpr_and_literal_sources(self):
        dst = RegisterOperand(OperandType.VGPR, 5)
        s_src = RegisterOperand(OperandType.SGPR, 10)
        lit_src = LiteralOperand(0x3F80_0000)
        insn_s = VFrexpExpI32F32Instruction(dst=dst, src0=s_src)
        insn_l = VFrexpExpI32F32Instruction(dst=dst, src0=lit_src)
        self.assertEqual(insn_s.src0.name, "s10")
        self.assertEqual(insn_l.src0.value, 0x3F80_0000)

    def test_invalid_register_indices(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)

    def test_invalid_dst_type(self):
        with self.assertRaises(TypeError):
            VFrexpExpI32F32Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 1),
            )

    def test_literal_bounds(self):
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

    def step(self, insn: VFrexpExpI32F32Instruction):
        val_s0 = self.read_operand(insn.src0)
        result = ref_frexp_exp_i32_f32(val_s0)
        self.write_vgpr(insn.dst.index, result)


class TestFrexpExpI32F32ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.machine = ScalarMachine()

    def test_scalar_execution_vgpr(self):
        # 2.0f in v1 -> exp 2 in v0
        self.machine.vgprs[1] = f32_to_bits(2.0)
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 2)

    def test_scalar_execution_sgpr(self):
        # 0.25f in s4 -> exp -1 (0xFFFFFFFF) in v5
        self.machine.sgprs[4] = f32_to_bits(0.25)
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 5),
            src0=RegisterOperand(OperandType.SGPR, 4),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[5], 0xFFFF_FFFF)

    def test_scalar_execution_literal(self):
        # Literal 1.0f -> exp 1 in v2
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=LiteralOperand(f32_to_bits(1.0)),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[2], 1)

    def test_scalar_zero_input(self):
        self.machine.vgprs[1] = NEG_ZERO
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 0)


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

    def read_operand_lane(self, op: Union[RegisterOperand, LiteralOperand], lane: int) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index][lane] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand type: {type(op)}")

    def step(self, insn: VFrexpExpI32F32Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                val_s0 = self.read_operand_lane(insn.src0, lane)
                res = ref_frexp_exp_i32_f32(val_s0)
                self.vgprs[insn.dst.index][lane] = res


class TestFrexpExpI32F32SIMDDivergence(unittest.TestCase):
    def test_simd64_full_divergence(self):
        wave = SIMDWave(wave_size=64)
        dst = RegisterOperand(OperandType.VGPR, 0)
        src0 = RegisterOperand(OperandType.VGPR, 1)
        insn = VFrexpExpI32F32Instruction(dst=dst, src0=src0)

        # Populate lanes with powers of two: 2^(lane - 30)
        for lane in range(64):
            # Exponent shift from 1.0f (which is exp 1)
            fval = math.ldexp(1.0, lane - 30)
            wave.vgprs[1][lane] = f32_to_bits(fval)

        wave.step(insn)

        for lane in range(64):
            expected_exp = (lane - 30 + 1) & MASK32
            self.assertEqual(
                wave.vgprs[0][lane],
                expected_exp,
                f"Lane {lane} failed: got {wave.vgprs[0][lane]}, expected {expected_exp}",
            )

    def test_simd32_exec_masking(self):
        wave = SIMDWave(wave_size=32)
        dst = RegisterOperand(OperandType.VGPR, 0)
        src0 = RegisterOperand(OperandType.VGPR, 1)
        insn = VFrexpExpI32F32Instruction(dst=dst, src0=src0)

        for lane in range(32):
            wave.vgprs[0][lane] = 0xDEAD_BEEF
            wave.vgprs[1][lane] = f32_to_bits(2.0)  # exp 2

        # Active on even lanes only
        wave.exec_mask = 0x5555_5555
        wave.step(insn)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[0][lane], 2)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xDEAD_BEEF)

    def test_zero_exec_mask_no_mutation(self):
        wave = SIMDWave(wave_size=64)
        dst = RegisterOperand(OperandType.VGPR, 0)
        src0 = RegisterOperand(OperandType.VGPR, 1)
        insn = VFrexpExpI32F32Instruction(dst=dst, src0=src0)

        for lane in range(64):
            wave.vgprs[0][lane] = 0xCAFE_BABE
            wave.vgprs[1][lane] = f32_to_bits(1.0)

        wave.exec_mask = 0
        wave.step(insn)

        for lane in range(64):
            self.assertEqual(wave.vgprs[0][lane], 0xCAFE_BABE)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestFrexpExpI32F32RegisterAliasing(unittest.TestCase):
    def setUp(self):
        self.machine = ScalarMachine()

    def test_in_place_alias_vgpr(self):
        # v0 = v_frexp_exp_i32_f32(v0)
        self.machine.vgprs[0] = f32_to_bits(1.0)  # 0x3F800000
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 1)

    def test_in_place_alias_negative_exponent(self):
        # v3 = v_frexp_exp_i32_f32(v3) with 0.125f (exp -2)
        self.machine.vgprs[3] = f32_to_bits(0.125)
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 3),
            src0=RegisterOperand(OperandType.VGPR, 3),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[3], 0xFFFF_FFFE)

    def test_unrelated_registers_preserved(self):
        self.machine.vgprs[0] = 0x1111_1111
        self.machine.vgprs[1] = f32_to_bits(8.0)  # exp 4
        self.machine.vgprs[2] = 0x2222_2222
        insn = VFrexpExpI32F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 4)
        self.assertEqual(self.machine.vgprs[1], f32_to_bits(8.0))
        self.assertEqual(self.machine.vgprs[2], 0x2222_2222)


if __name__ == '__main__':
    unittest.main()
