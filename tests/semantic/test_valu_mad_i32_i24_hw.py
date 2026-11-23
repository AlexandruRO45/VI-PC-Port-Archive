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

    def step(self, insn: VMadI32I24Instruction):
        val_s0 = self.read_operand(insn.src0)
        val_s1 = self.read_operand(insn.src1)
        val_s2 = self.read_operand(insn.src2)
        result = ref_mad_i32_i24(val_s0, val_s1, val_s2, clamp=insn.clamp)
        self.write_vgpr(insn.dst.index, result)


class TestMadI32I24ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.machine = ScalarMachine()

    def test_scalar_execution_basic(self):
        self.machine.vgprs[1] = 2
        self.machine.vgprs[2] = 3
        self.machine.vgprs[3] = 4
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 10)

    def test_scalar_with_sgpr_and_literal(self):
        self.machine.sgprs[4] = 0x00FFFFFE  # -2 in 24 bits
        self.machine.vgprs[2] = 3
        lit = LiteralOperand(4)
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 5),
            src0=RegisterOperand(OperandType.SGPR, 4),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=lit,
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[5], 0xFFFF_FFFE)


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

    def step(self, insn: VMadI32I24Instruction):
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

            res = ref_mad_i32_i24(val_s0, val_s1, val_s2, clamp=insn.clamp)
            self.vgprs[insn.dst.index][lane] = res & MASK32


class TestMadI32I24SIMDDivergence(unittest.TestCase):
    def test_wave32_alternating_exec(self):
        wave = SIMDWave(wave_size=32)
        wave.exec_mask = 0x5555_5555  # Even lanes active
        for lane in range(32):
            wave.vgprs[0][lane] = 0xDEAD_BEEF
            wave.vgprs[1][lane] = 2
            wave.vgprs[2][lane] = 3
            wave.vgprs[3][lane] = 4

        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[0][lane], 10)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xDEAD_BEEF)

    def test_wave64_divergence(self):
        wave = SIMDWave(wave_size=64)
        wave.exec_mask = 0xFFFF_FFFF_0000_0000  # Top 32 lanes active
        for lane in range(64):
            wave.vgprs[10][lane] = 0xCAFE_0000
            wave.vgprs[1][lane] = lane
            wave.vgprs[2][lane] = 2
            wave.vgprs[3][lane] = 1

        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 10),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)

        for lane in range(32):
            self.assertEqual(wave.vgprs[10][lane], 0xCAFE_0000)
        for lane in range(32, 64):
            expected = ref_mad_i32_i24(lane, 2, 1)
            self.assertEqual(wave.vgprs[10][lane], expected)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestMadI32I24RegisterAliasing(unittest.TestCase):
    def test_dst_aliases_src0(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 2
        machine.vgprs[1] = 3
        machine.vgprs[2] = 4
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
            src1=RegisterOperand(OperandType.VGPR, 1),
            src2=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 10)

    def test_dst_aliases_src2(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 4
        machine.vgprs[1] = 2
        machine.vgprs[2] = 3
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 0),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 10)

    def test_dst_aliases_all_sources(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 3
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
            src1=RegisterOperand(OperandType.VGPR, 0),
            src2=RegisterOperand(OperandType.VGPR, 0),
        )
        # 3 * 3 + 3 = 12
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 12)


# ==============================================================================
# Condition Codes, Status Flags, and Invariance
# ==============================================================================

class TestMadI32I24StatusFlags(unittest.TestCase):
    def test_scc_invariance(self):
        machine = ScalarMachine()
        machine.scc = 1
        machine.vgprs[1] = 2
        machine.vgprs[2] = 3
        machine.vgprs[3] = 4
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        machine.step(insn)
        # VOP3 does not alter SCC
        self.assertEqual(machine.scc, 1)

    def test_exec_mask_invariance(self):
        wave = SIMDWave(wave_size=64)
        orig_mask = 0xAAAA_5555_AAAA_5555
        wave.exec_mask = orig_mask
        insn = VMadI32I24Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)
        self.assertEqual(wave.exec_mask, orig_mask)


# ==============================================================================
# IEEE 754 Bit-Pattern Boundary Extremes as Integer Values
# ==============================================================================

class TestMadI32I24BitPatternBoundaries(unittest.TestCase):
    def test_float_patterns_as_integers(self):
        # v_mad_i32_i24 operates on raw bits without IEEE floating point interpretation
        # Float +0.0: 0x00000000
        # Float -0.0: 0x80000000 -> lower 24 bits: 0x000000
        self.assertEqual(ref_mad_i32_i24(0x8000_0000, 10, 5), 5)
        # Float 1.0f: 0x3F800000 -> bit 23 is 1 -> sign extended is negative (-0x800000)
        s0_val = sign_extend_i24(0x3F80_0000)
        self.assertEqual(s0_val, -8388608)
        self.assertEqual(ref_mad_i32_i24(0x3F80_0000, 2, 0), 0xFF00_0000)


# ==============================================================================
# Negative Controls, Rejection, and Translator Hardware Execution
# ==============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv


class TestMadI32I24RejectionAndHardwareTranslator(unittest.TestCase):
    def test_syntax_rejection_negative_controls(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 300)
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)

    def test_hardware_translator_execution(self):
        """Validates that tools/rdna2spirv.py translates v_mad_i32_i24 and matches expected results."""
        cases = [
            (2, 3, 4, "v_mad_i32_i24 v0, v1, v2, v3", 10),
            (0x00FFFFFE, 3, 4, "v_mad_i32_i24 v0, v1, v2, v3", 0xFFFF_FFFE),
            (0x55FFFFFE, 3, 4, "v_mad_i32_i24 v0, v1, v2, v3", 0xFFFF_FFFE),
            (0x007FFFFF, 2, 0, "v_mad_i32_i24 v0, v1, v2, v3", 16777214),
            (0x00800000, 2, 0, "v_mad_i32_i24 v0, v1, v2, v3", 0xFF00_0000),
            (0x007FFFFF, 0x007FFFFF, 0, "v_mad_i32_i24 v0, v1, v2, v3", 0xFF00_0001),
            (2, 3, 4, "v_mad_i32_i24 v0, v1, v2, v3 clamp", 10),
        ]
        for s0, s1, s2, asm, expected in cases:
            prog = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {s2:#x}",
                asm,
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_mad_i32_i24_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected, f"Failed on {asm}: got {words[0]:#x}, expected {expected:#x}")


if __name__ == '__main__':
    unittest.main()
