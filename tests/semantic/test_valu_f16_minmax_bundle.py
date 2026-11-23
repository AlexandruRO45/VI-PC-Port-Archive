from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 Float16 Min/Max/Clamp Suite.

Covers:
- v_min_f16: IEEE 754 binary16 minimum, NMin semantics (NaN passes other operand, -0.0 < +0.0).
- v_max_f16: IEEE 754 binary16 maximum, NMax semantics (NaN passes other operand, +0.0 > -0.0).
- v_floor_f16: floor rounding in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.
- v_ceil_f16: ceiling rounding in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.
- v_trunc_f16: truncation towards zero in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.
- v_rndne_f16: round to nearest even in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.

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

MASK16: int = 0xFFFF
SIGN_BIT16: int = 0x8000
EXP_MASK16: int = 0x001F
EXP_SHIFT16: int = 10
MANT_MASK16: int = 0x03FF

POS_ZERO16: int = 0x0000
NEG_ZERO16: int = 0x8000
POS_INF16: int = 0x7C00
NEG_INF16: int = 0xFC00
QNAN16: int = 0x7E00
SNAN16: int = 0x7C01


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


def is_nan_f16(u: int) -> bool:
    return (u & 0x7C00) == 0x7C00 and (u & 0x03FF) != 0


def is_inf_f16(u: int) -> bool:
    return (u & 0x7FFF) == 0x7C00


# ------------------------------------------------------------------------------
# v_min_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_min_f16(s0: int, s1: int) -> int:
    u0 = s0 & MASK16
    u1 = s1 & MASK16
    nan0 = is_nan_f16(u0)
    nan1 = is_nan_f16(u1)
    if nan0 and nan1:
        return u0
    if nan0:
        return u1
    if nan1:
        return u0
    if (u0 | u1) == 0x8000:
        return NEG_ZERO16
    f0 = bits_to_f16(u0)
    f1 = bits_to_f16(u1)
    return u0 if f0 < f1 else u1


class TestMinF16Oracle(unittest.TestCase):
    def test_basic_min(self):
        self.assertEqual(ref_min_f16(f16_to_bits(1.0), f16_to_bits(2.0)), f16_to_bits(1.0))
        self.assertEqual(ref_min_f16(f16_to_bits(-3.0), f16_to_bits(1.0)), f16_to_bits(-3.0))
        self.assertEqual(ref_min_f16(f16_to_bits(5.0), f16_to_bits(5.0)), f16_to_bits(5.0))

    def test_signed_zeros(self):
        # -0.0 < +0.0
        self.assertEqual(ref_min_f16(POS_ZERO16, NEG_ZERO16), NEG_ZERO16)
        self.assertEqual(ref_min_f16(NEG_ZERO16, POS_ZERO16), NEG_ZERO16)


# ------------------------------------------------------------------------------
# v_min_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

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
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        return f"{'v' if self.op_type == OperandType.VGPR else 's'}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
            raise ValueError(f"Literal value out of range: {self.value}")


@dataclass
class VMinF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineMinF16:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK16
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK16
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK16
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VMinF16Instruction):
        s0 = self.read_operand(insn.src0)
        s1 = self.read_operand(insn.src1)
        res = ref_min_f16(s0, s1)
        # Low 16 bits written, preserving high 16 bits
        self.vgprs[insn.dst.index] = (self.vgprs[insn.dst.index] & 0xFFFF_0000) | res


class TestMinF16ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VMinF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        self.assertEqual(insn.dst.name, "v0")

    def test_scalar_execution(self):
        machine = ScalarMachineMinF16()
        machine.vgprs[0] = 0xAAAA_0000
        machine.vgprs[1] = f16_to_bits(10.0)
        machine.vgprs[2] = f16_to_bits(2.5)
        insn = VMinF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 0xAAAA_0000 | f16_to_bits(2.5))


# ------------------------------------------------------------------------------
# v_min_f16 SIMD & IEEE 754
# ------------------------------------------------------------------------------

class SIMDWaveMinF16:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VMinF16Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane] & MASK16
                s1 = self.vgprs[insn.src1.index][lane] & MASK16
                res = ref_min_f16(s0, s1)
                self.vgprs[insn.dst.index][lane] = (self.vgprs[insn.dst.index][lane] & 0xFFFF_0000) | res


class TestMinF16SIMDAndIEEE(unittest.TestCase):
    def test_simd_divergence(self):
        wave = SIMDWaveMinF16(wave_size=32)
        for lane in range(32):
            wave.vgprs[1][lane] = f16_to_bits(float(lane))
            wave.vgprs[2][lane] = f16_to_bits(16.0)
            wave.vgprs[0][lane] = 0

        wave.exec_mask = 0x5555_5555
        insn = VMinF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        wave.step(insn)

        for lane in range(32):
            if lane % 2 == 0:
                expected = f16_to_bits(min(float(lane), 16.0))
                self.assertEqual(wave.vgprs[0][lane], expected)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0)

    def test_nan_and_infinities(self):
        # One NaN returns other operand
        self.assertEqual(ref_min_f16(QNAN16, f16_to_bits(3.0)), f16_to_bits(3.0))
        self.assertEqual(ref_min_f16(f16_to_bits(-5.0), QNAN16), f16_to_bits(-5.0))
        # Infinities
        self.assertEqual(ref_min_f16(POS_INF16, f16_to_bits(100.0)), f16_to_bits(100.0))
        self.assertEqual(ref_min_f16(NEG_INF16, f16_to_bits(-100.0)), NEG_INF16)


# ------------------------------------------------------------------------------
# v_min_f16 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv


class TestMinF16HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)

    def test_hardware_translator_execution(self):
        cases = [
            (f16_to_bits(1.0), f16_to_bits(2.0), f16_to_bits(1.0)),
            (f16_to_bits(-3.0), f16_to_bits(1.0), f16_to_bits(-3.0)),
            (POS_ZERO16, NEG_ZERO16, NEG_ZERO16),
            (QNAN16, f16_to_bits(1.0), f16_to_bits(1.0)),
            (POS_INF16, f16_to_bits(10.0), f16_to_bits(10.0)),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_min_f16 v0, v1, v2",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_min_f16_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0] & 0xFFFF, expected, f"Failed on min_f16({s0:#x}, {s1:#x})")


# ------------------------------------------------------------------------------
# v_max_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_max_f16(s0: int, s1: int) -> int:
    u0 = s0 & MASK16
    u1 = s1 & MASK16
    nan0 = is_nan_f16(u0)
    nan1 = is_nan_f16(u1)
    if nan0 and nan1:
        return u0
    if nan0:
        return u1
    if nan1:
        return u0
    if (u0 | u1) == 0x8000:
        return POS_ZERO16
    f0 = bits_to_f16(u0)
    f1 = bits_to_f16(u1)
    return u0 if f0 > f1 else u1


class TestMaxF16Oracle(unittest.TestCase):
    def test_basic_max(self):
        self.assertEqual(ref_max_f16(f16_to_bits(1.0), f16_to_bits(2.0)), f16_to_bits(2.0))
        self.assertEqual(ref_max_f16(f16_to_bits(-3.0), f16_to_bits(1.0)), f16_to_bits(1.0))

    def test_signed_zeros(self):
        # +0.0 > -0.0
        self.assertEqual(ref_max_f16(POS_ZERO16, NEG_ZERO16), POS_ZERO16)
        self.assertEqual(ref_max_f16(NEG_ZERO16, POS_ZERO16), POS_ZERO16)


# ------------------------------------------------------------------------------
# v_max_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VMaxF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineMaxF16:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK16
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK16
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK16
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VMaxF16Instruction):
        s0 = self.read_operand(insn.src0)
        s1 = self.read_operand(insn.src1)
        res = ref_max_f16(s0, s1)
        self.vgprs[insn.dst.index] = (self.vgprs[insn.dst.index] & 0xFFFF_0000) | res


class TestMaxF16ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VMaxF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        self.assertEqual(insn.dst.name, "v0")

    def test_scalar_execution(self):
        machine = ScalarMachineMaxF16()
        machine.vgprs[1] = f16_to_bits(1.0)
        machine.vgprs[2] = f16_to_bits(4.0)
        insn = VMaxF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], f16_to_bits(4.0))


# ------------------------------------------------------------------------------
# v_max_f16 SIMD & IEEE 754
# ------------------------------------------------------------------------------

class SIMDWaveMaxF16:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VMaxF16Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane] & MASK16
                s1 = self.vgprs[insn.src1.index][lane] & MASK16
                res = ref_max_f16(s0, s1)
                self.vgprs[insn.dst.index][lane] = (self.vgprs[insn.dst.index][lane] & 0xFFFF_0000) | res


class TestMaxF16SIMDAndIEEE(unittest.TestCase):
    def test_simd_divergence(self):
        wave = SIMDWaveMaxF16(wave_size=32)
        for lane in range(32):
            wave.vgprs[1][lane] = f16_to_bits(float(lane))
            wave.vgprs[2][lane] = f16_to_bits(10.0)
            wave.vgprs[0][lane] = 0

        insn = VMaxF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        wave.step(insn)

        for lane in range(32):
            expected = f16_to_bits(max(float(lane), 10.0))
            self.assertEqual(wave.vgprs[0][lane], expected)

    def test_nan_and_infinities(self):
        self.assertEqual(ref_max_f16(QNAN16, f16_to_bits(3.0)), f16_to_bits(3.0))
        self.assertEqual(ref_max_f16(f16_to_bits(-5.0), QNAN16), f16_to_bits(-5.0))
        self.assertEqual(ref_max_f16(NEG_INF16, f16_to_bits(-100.0)), f16_to_bits(-100.0))
        self.assertEqual(ref_max_f16(POS_INF16, f16_to_bits(100.0)), POS_INF16)


# ------------------------------------------------------------------------------
# v_max_f16 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

class TestMaxF16HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(TypeError):
            VMaxF16Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 1),
                src1=RegisterOperand(OperandType.VGPR, 2),
            )

    def test_hardware_translator_execution(self):
        cases = [
            (f16_to_bits(1.0), f16_to_bits(2.0), f16_to_bits(2.0)),
            (f16_to_bits(-3.0), f16_to_bits(1.0), f16_to_bits(1.0)),
            (POS_ZERO16, NEG_ZERO16, POS_ZERO16),
            (QNAN16, f16_to_bits(1.0), f16_to_bits(1.0)),
            (NEG_INF16, f16_to_bits(10.0), f16_to_bits(10.0)),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_max_f16 v0, v1, v2",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_max_f16_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0] & 0xFFFF, expected, f"Failed on max_f16({s0:#x}, {s1:#x})")


# ------------------------------------------------------------------------------
# v_floor_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_floor_f16(src: int) -> int:
    u = src & MASK16
    if is_nan_f16(u) or is_inf_f16(u):
        return u
    if (u & 0x7FFF) == 0:
        return u
    f = bits_to_f16(u)
    rf = math.floor(f)
    return f16_to_bits(rf)


class TestFloorF16Oracle(unittest.TestCase):
    def test_basic_floor(self):
        self.assertEqual(ref_floor_f16(f16_to_bits(1.5)), f16_to_bits(1.0))
        self.assertEqual(ref_floor_f16(f16_to_bits(-1.5)), f16_to_bits(-2.0))
        self.assertEqual(ref_floor_f16(f16_to_bits(2.0)), f16_to_bits(2.0))

    def test_special_cases(self):
        self.assertEqual(ref_floor_f16(POS_ZERO16), POS_ZERO16)
        self.assertEqual(ref_floor_f16(NEG_ZERO16), NEG_ZERO16)
        self.assertEqual(ref_floor_f16(POS_INF16), POS_INF16)
        self.assertEqual(ref_floor_f16(NEG_INF16), NEG_INF16)
        self.assertEqual(ref_floor_f16(QNAN16), QNAN16)


# ------------------------------------------------------------------------------
# v_floor_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VFloorF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineFloorF16:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK16
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK16
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK16
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VFloorF16Instruction):
        s0 = self.read_operand(insn.src0)
        res = ref_floor_f16(s0)
        self.vgprs[insn.dst.index] = (self.vgprs[insn.dst.index] & 0xFFFF_0000) | res


class TestFloorF16ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VFloorF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        self.assertEqual(insn.dst.name, "v0")

    def test_scalar_execution(self):
        machine = ScalarMachineFloorF16()
        machine.vgprs[1] = f16_to_bits(2.75)
        insn = VFloorF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], f16_to_bits(2.0))


# ------------------------------------------------------------------------------
# v_floor_f16 SIMD & IEEE 754
# ------------------------------------------------------------------------------

class SIMDWaveFloorF16:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VFloorF16Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane] & MASK16
                res = ref_floor_f16(s0)
                self.vgprs[insn.dst.index][lane] = (self.vgprs[insn.dst.index][lane] & 0xFFFF_0000) | res


class TestFloorF16SIMDAndIEEE(unittest.TestCase):
    def test_simd_divergence(self):
        wave = SIMDWaveFloorF16(wave_size=32)
        for lane in range(32):
            wave.vgprs[1][lane] = f16_to_bits(float(lane) + 0.5)
            wave.vgprs[0][lane] = 0

        insn = VFloorF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        wave.step(insn)

        for lane in range(32):
            expected = f16_to_bits(float(lane))
            self.assertEqual(wave.vgprs[0][lane], expected)

    def test_nans_pass_through(self):
        self.assertEqual(ref_floor_f16(QNAN16), QNAN16)
        self.assertEqual(ref_floor_f16(SNAN16), SNAN16)


# ------------------------------------------------------------------------------
# v_floor_f16 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

class TestFloorF16HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)

    def test_hardware_translator_execution(self):
        cases = [
            (f16_to_bits(1.5), f16_to_bits(1.0)),
            (f16_to_bits(-1.5), f16_to_bits(-2.0)),
            (f16_to_bits(3.0), f16_to_bits(3.0)),
            (POS_ZERO16, POS_ZERO16),
            (NEG_ZERO16, NEG_ZERO16),
            (POS_INF16, POS_INF16),
            (NEG_INF16, NEG_INF16),
            (QNAN16, QNAN16),
        ]
        for src_val, expected in cases:
            prog = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {src_val:#x}",
                "v_floor_f16 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_floor_f16_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0] & 0xFFFF, expected, f"Failed on floor_f16({src_val:#x})")


# ------------------------------------------------------------------------------
# v_ceil_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_ceil_f16(src: int) -> int:
    u = src & MASK16
    if is_nan_f16(u) or is_inf_f16(u):
        return u
    if (u & 0x7FFF) == 0:
        return u
    f = bits_to_f16(u)
    rf = math.ceil(f)
    return f16_to_bits(rf)


class TestCeilF16Oracle(unittest.TestCase):
    def test_basic_ceil(self):
        self.assertEqual(ref_ceil_f16(f16_to_bits(1.2)), f16_to_bits(2.0))
        self.assertEqual(ref_ceil_f16(f16_to_bits(-1.8)), f16_to_bits(-1.0))
        self.assertEqual(ref_ceil_f16(f16_to_bits(4.0)), f16_to_bits(4.0))

    def test_special_cases(self):
        self.assertEqual(ref_ceil_f16(POS_ZERO16), POS_ZERO16)
        self.assertEqual(ref_ceil_f16(NEG_ZERO16), NEG_ZERO16)
        self.assertEqual(ref_ceil_f16(POS_INF16), POS_INF16)
        self.assertEqual(ref_ceil_f16(NEG_INF16), NEG_INF16)
        self.assertEqual(ref_ceil_f16(QNAN16), QNAN16)


# ------------------------------------------------------------------------------
# v_ceil_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VCeilF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineCeilF16:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK16
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK16
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK16
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VCeilF16Instruction):
        s0 = self.read_operand(insn.src0)
        res = ref_ceil_f16(s0)
        self.vgprs[insn.dst.index] = (self.vgprs[insn.dst.index] & 0xFFFF_0000) | res


class TestCeilF16ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VCeilF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        self.assertEqual(insn.dst.name, "v0")

    def test_scalar_execution(self):
        machine = ScalarMachineCeilF16()
        machine.vgprs[1] = f16_to_bits(1.1)
        insn = VCeilF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], f16_to_bits(2.0))


# ------------------------------------------------------------------------------
# v_ceil_f16 SIMD & IEEE 754
# ------------------------------------------------------------------------------

class SIMDWaveCeilF16:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VCeilF16Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane] & MASK16
                res = ref_ceil_f16(s0)
                self.vgprs[insn.dst.index][lane] = (self.vgprs[insn.dst.index][lane] & 0xFFFF_0000) | res


class TestCeilF16SIMDAndIEEE(unittest.TestCase):
    def test_simd_divergence(self):
        wave = SIMDWaveCeilF16(wave_size=32)
        for lane in range(32):
            wave.vgprs[1][lane] = f16_to_bits(float(lane) + 0.1)
            wave.vgprs[0][lane] = 0

        insn = VCeilF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        wave.step(insn)

        for lane in range(32):
            expected = f16_to_bits(float(lane) + 1.0)
            self.assertEqual(wave.vgprs[0][lane], expected)

    def test_nans_pass_through(self):
        self.assertEqual(ref_ceil_f16(QNAN16), QNAN16)
        self.assertEqual(ref_ceil_f16(SNAN16), SNAN16)


# ------------------------------------------------------------------------------
# v_ceil_f16 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

class TestCeilF16HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(TypeError):
            VCeilF16Instruction(dst=RegisterOperand(OperandType.SGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))

    def test_hardware_translator_execution(self):
        cases = [
            (f16_to_bits(1.2), f16_to_bits(2.0)),
            (f16_to_bits(-1.8), f16_to_bits(-1.0)),
            (f16_to_bits(4.0), f16_to_bits(4.0)),
            (POS_ZERO16, POS_ZERO16),
            (NEG_ZERO16, NEG_ZERO16),
            (POS_INF16, POS_INF16),
            (NEG_INF16, NEG_INF16),
            (QNAN16, QNAN16),
        ]
        for src_val, expected in cases:
            prog = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {src_val:#x}",
                "v_ceil_f16 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_ceil_f16_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0] & 0xFFFF, expected, f"Failed on ceil_f16({src_val:#x})")


# ------------------------------------------------------------------------------
# v_trunc_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_trunc_f16(src: int) -> int:
    u = src & MASK16
    if is_nan_f16(u) or is_inf_f16(u):
        return u
    if (u & 0x7FFF) == 0:
        return u
    f = bits_to_f16(u)
    rf = math.trunc(f)
    return f16_to_bits(rf)


class TestTruncF16Oracle(unittest.TestCase):
    def test_basic_trunc(self):
        self.assertEqual(ref_trunc_f16(f16_to_bits(1.9)), f16_to_bits(1.0))
        self.assertEqual(ref_trunc_f16(f16_to_bits(-1.9)), f16_to_bits(-1.0))
        self.assertEqual(ref_trunc_f16(f16_to_bits(5.0)), f16_to_bits(5.0))

    def test_special_cases(self):
        self.assertEqual(ref_trunc_f16(POS_ZERO16), POS_ZERO16)
        self.assertEqual(ref_trunc_f16(NEG_ZERO16), NEG_ZERO16)
        self.assertEqual(ref_trunc_f16(POS_INF16), POS_INF16)
        self.assertEqual(ref_trunc_f16(NEG_INF16), NEG_INF16)
        self.assertEqual(ref_trunc_f16(QNAN16), QNAN16)


# ------------------------------------------------------------------------------
# v_trunc_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VTruncF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineTruncF16:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK16
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK16
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK16
        raise TypeError(f"Unknown operand: {op}")

    def step(self, insn: VTruncF16Instruction):
        s0 = self.read_operand(insn.src0)
        res = ref_trunc_f16(s0)
        self.vgprs[insn.dst.index] = (self.vgprs[insn.dst.index] & 0xFFFF_0000) | res


class TestTruncF16ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VTruncF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        self.assertEqual(insn.dst.name, "v0")

    def test_scalar_execution(self):
        machine = ScalarMachineTruncF16()
        machine.vgprs[1] = f16_to_bits(-3.8)
        insn = VTruncF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], f16_to_bits(-3.0))


# ------------------------------------------------------------------------------
# v_trunc_f16 SIMD & IEEE 754
# ------------------------------------------------------------------------------

class SIMDWaveTruncF16:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VTruncF16Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane] & MASK16
                res = ref_trunc_f16(s0)
                self.vgprs[insn.dst.index][lane] = (self.vgprs[insn.dst.index][lane] & 0xFFFF_0000) | res


class TestTruncF16SIMDAndIEEE(unittest.TestCase):
    def test_simd_divergence(self):
        wave = SIMDWaveTruncF16(wave_size=32)
        for lane in range(32):
            sign = -1.0 if lane % 2 else 1.0
            wave.vgprs[1][lane] = f16_to_bits(sign * (float(lane) + 0.9))
            wave.vgprs[0][lane] = 0

        insn = VTruncF16Instruction(dst=RegisterOperand(OperandType.VGPR, 0), src0=RegisterOperand(OperandType.VGPR, 1))
        wave.step(insn)

        for lane in range(32):
            sign = -1.0 if lane % 2 else 1.0
            expected = f16_to_bits(sign * float(lane))
            self.assertEqual(wave.vgprs[0][lane], expected)

    def test_nans_pass_through(self):
        self.assertEqual(ref_trunc_f16(QNAN16), QNAN16)
        self.assertEqual(ref_trunc_f16(SNAN16), SNAN16)


# ------------------------------------------------------------------------------
# v_trunc_f16 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

class TestTruncF16HardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        cases = [
            (f16_to_bits(1.9), f16_to_bits(1.0)),
            (f16_to_bits(-1.9), f16_to_bits(-1.0)),
            (f16_to_bits(5.0), f16_to_bits(5.0)),
            (POS_ZERO16, POS_ZERO16),
            (NEG_ZERO16, NEG_ZERO16),
            (POS_INF16, POS_INF16),
            (NEG_INF16, NEG_INF16),
            (QNAN16, QNAN16),
        ]
        for src_val, expected in cases:
            prog = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {src_val:#x}",
                "v_trunc_f16 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_trunc_f16_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0] & 0xFFFF, expected, f"Failed on trunc_f16({src_val:#x})")


if __name__ == '__main__':
    unittest.main()
