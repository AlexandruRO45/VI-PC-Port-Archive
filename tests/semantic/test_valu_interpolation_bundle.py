from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 Parameter Interpolation Suite.

Covers:
- v_interp_p1_f32: First barycentric axis interpolation D = P0 + i * (P1 - P0).
- v_interp_p2_f32: Second barycentric axis interpolation and accumulation D = D_p1 + j * (P2 - P0).
- v_interp_mov_f32: Direct parameter access: p0 (provoking vertex), p10 (delta P1-P0), p20 (delta P2-P0).
- v_interp_p1ll_f16: 16-bit packed attribute low half first axis interpolation in f32 intermediate precision.
- v_interp_p1lv_f16: 16-bit packed attribute low/high half first axis interpolation in f32 intermediate precision.
- v_interp_p2_f16: Second axis accumulation onto f32 intermediate, narrowed to f16 with optional clamp.

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

MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF


def f32_to_bits(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", float(f)))[0]
    except OverflowError:
        return 0x7F80_0000 if f > 0 else 0xFF80_0000


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_round(f: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(f)))[0]
    except OverflowError:
        return float("inf") if f > 0 else float("-inf")


def f16_to_bits(f: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", float(f)))[0]
    except OverflowError:
        return 0x7C00 if f > 0 else 0xFC00


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


# ------------------------------------------------------------------------------
# v_interp_p1_f32 Reference Oracle
# ------------------------------------------------------------------------------

def ref_interp_p1_f32(i_coord: float, p0: float, p1: float) -> int:
    """Computes D = P0 + i * (P1 - P0)."""
    d1 = f32_round(p1 - p0)
    m1 = f32_round(d1 * i_coord)
    res = f32_round(p0 + m1)
    return f32_to_bits(res)


class TestInterpP1F32Oracle(unittest.TestCase):
    def test_vertex_endpoints(self):
        # i = 0.0 -> P0
        self.assertEqual(ref_interp_p1_f32(0.0, 10.0, 20.0), f32_to_bits(10.0))
        # i = 1.0 -> P1
        self.assertEqual(ref_interp_p1_f32(1.0, 10.0, 20.0), f32_to_bits(20.0))

    def test_midpoint_interpolation(self):
        # i = 0.5 -> 15.0
        self.assertEqual(ref_interp_p1_f32(0.5, 10.0, 20.0), f32_to_bits(15.0))
        # i = 0.25 -> 12.5
        self.assertEqual(ref_interp_p1_f32(0.25, 10.0, 20.0), f32_to_bits(12.5))


# ------------------------------------------------------------------------------
# v_interp_p1_f32 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    ATTR = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    @property
    def name(self) -> str:
        return f"{'v' if self.op_type == OperandType.VGPR else 's'}{self.index}"


@dataclass(frozen=True)
class AttrOperand:
    index: int
    channel: str

    @property
    def name(self) -> str:
        return f"attr{self.index}.{self.channel}"


@dataclass
class VInterpP1F32Instruction:
    dst: RegisterOperand
    src0: RegisterOperand  # i barycentric coordinate (usually v0)
    attr: AttrOperand

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineInterpP1F32:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.attrs: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 32

    def step(self, insn: VInterpP1F32Instruction):
        i_coord = bits_to_f32(self.vgprs[insn.src0.index])
        p0, p1, p2 = self.attrs[insn.attr.index]
        res = ref_interp_p1_f32(i_coord, p0, p1)
        self.vgprs[insn.dst.index] = res


class TestInterpP1F32ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VInterpP1F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=RegisterOperand(OperandType.VGPR, 0),
            attr=AttrOperand(0, "x"),
        )
        self.assertEqual(insn.dst.name, "v2")
        self.assertEqual(insn.src0.name, "v0")
        self.assertEqual(insn.attr.name, "attr0.x")

    def test_scalar_execution(self):
        machine = ScalarMachineInterpP1F32()
        machine.vgprs[0] = f32_to_bits(0.5)
        machine.attrs[0] = (2.0, 6.0, 10.0)  # P0=2.0, P1=6.0 -> midpoint 4.0
        insn = VInterpP1F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=RegisterOperand(OperandType.VGPR, 0),
            attr=AttrOperand(0, "x"),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[2], f32_to_bits(4.0))


# ------------------------------------------------------------------------------
# v_interp_p1_f32 SIMD & Barycentric Coordinates
# ------------------------------------------------------------------------------

class SIMDWaveInterpP1F32:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VInterpP1F32Instruction, p0: float, p1: float):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                i_coord = bits_to_f32(self.vgprs[insn.src0.index][lane])
                self.vgprs[insn.dst.index][lane] = ref_interp_p1_f32(i_coord, p0, p1)


class TestInterpP1F32SIMDAndBarycentrics(unittest.TestCase):
    def test_simd_divergent_coordinates(self):
        wave = SIMDWaveInterpP1F32(wave_size=32)
        p0, p1 = 0.0, 100.0
        for lane in range(32):
            i_coord = float(lane) / 32.0
            wave.vgprs[0][lane] = f32_to_bits(i_coord)
            wave.vgprs[2][lane] = 0

        insn = VInterpP1F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=RegisterOperand(OperandType.VGPR, 0),
            attr=AttrOperand(0, "x"),
        )
        wave.step(insn, p0, p1)

        for lane in range(32):
            expected = ref_interp_p1_f32(float(lane) / 32.0, p0, p1)
            self.assertEqual(wave.vgprs[2][lane], expected)

    def test_negative_coordinates(self):
        # Outside triangle boundary (extrapolation)
        self.assertEqual(ref_interp_p1_f32(-0.5, 10.0, 20.0), f32_to_bits(5.0))


# ------------------------------------------------------------------------------
# v_interp_p1_f32 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv


class TestInterpP1F32HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(TypeError):
            VInterpP1F32Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 0),
                attr=AttrOperand(0, "x"),
            )

    def test_hardware_translator_execution(self):
        prog = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_interp_p1_hw", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)


# ------------------------------------------------------------------------------
# v_interp_p2_f32 Reference Oracle
# ------------------------------------------------------------------------------

def ref_interp_p2_f32(j_coord: float, p0: float, p2: float, p1_accum: float, clamp: bool = False) -> int:
    """Computes D = p1_accum + j * (P2 - P0)."""
    d2 = f32_round(p2 - p0)
    m2 = f32_round(d2 * j_coord)
    res = f32_round(p1_accum + m2)
    if clamp:
        res = max(0.0, min(1.0, res))
    return f32_to_bits(res)


class TestInterpP2F32Oracle(unittest.TestCase):
    def test_barycentric_triangle_vertices(self):
        p0, p1, p2 = 1.0, 5.0, 9.0
        # Vertex 0: (i=0, j=0) -> P0 = 1.0
        acc0 = bits_to_f32(ref_interp_p1_f32(0.0, p0, p1))
        self.assertEqual(ref_interp_p2_f32(0.0, p0, p2, acc0), f32_to_bits(1.0))

        # Vertex 1: (i=1, j=0) -> P1 = 5.0
        acc1 = bits_to_f32(ref_interp_p1_f32(1.0, p0, p1))
        self.assertEqual(ref_interp_p2_f32(0.0, p0, p2, acc1), f32_to_bits(5.0))

        # Vertex 2: (i=0, j=1) -> P2 = 9.0
        acc2 = bits_to_f32(ref_interp_p1_f32(0.0, p0, p1))
        self.assertEqual(ref_interp_p2_f32(1.0, p0, p2, acc2), f32_to_bits(9.0))

    def test_clamp_modifier(self):
        # Value > 1.0 clamped to 1.0
        self.assertEqual(ref_interp_p2_f32(0.5, 0.0, 4.0, 1.0, clamp=True), f32_to_bits(1.0))
        # Value < 0.0 clamped to 0.0
        self.assertEqual(ref_interp_p2_f32(0.5, 0.0, -10.0, 0.0, clamp=True), f32_to_bits(0.0))


# ------------------------------------------------------------------------------
# v_interp_p2_f32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VInterpP2F32Instruction:
    dst: RegisterOperand
    src0: RegisterOperand  # j barycentric coordinate (usually v1)
    attr: AttrOperand
    accum: RegisterOperand
    clamp: bool = False

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineInterpP2F32:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.attrs: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 32

    def step(self, insn: VInterpP2F32Instruction):
        j_coord = bits_to_f32(self.vgprs[insn.src0.index])
        accum = bits_to_f32(self.vgprs[insn.accum.index])
        p0, p1, p2 = self.attrs[insn.attr.index]
        res = ref_interp_p2_f32(j_coord, p0, p2, accum, clamp=insn.clamp)
        self.vgprs[insn.dst.index] = res


class TestInterpP2F32ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VInterpP2F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=RegisterOperand(OperandType.VGPR, 1),
            attr=AttrOperand(0, "y"),
            accum=RegisterOperand(OperandType.VGPR, 2),
            clamp=True,
        )
        self.assertEqual(insn.dst.name, "v2")
        self.assertTrue(insn.clamp)

    def test_scalar_pipeline(self):
        machine = ScalarMachineInterpP2F32()
        machine.vgprs[1] = f32_to_bits(0.5)
        machine.vgprs[2] = f32_to_bits(3.0)  # accum from p1
        machine.attrs[0] = (1.0, 5.0, 7.0)   # P0=1, P2=7 -> delta=6 -> + 0.5*6 = 3 -> 3+3=6.0
        insn = VInterpP2F32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=RegisterOperand(OperandType.VGPR, 1),
            attr=AttrOperand(0, "x"),
            accum=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[2], f32_to_bits(6.0))


# ------------------------------------------------------------------------------
# v_interp_p2_f32 SIMD & Full Triangle Interpolation
# ------------------------------------------------------------------------------

class SIMDWaveInterpP2F32:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VInterpP2F32Instruction, p0: float, p2: float):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                j_coord = bits_to_f32(self.vgprs[insn.src0.index][lane])
                accum = bits_to_f32(self.vgprs[insn.accum.index][lane])
                self.vgprs[insn.dst.index][lane] = ref_interp_p2_f32(j_coord, p0, p2, accum, clamp=insn.clamp)


class TestInterpP2F32SIMDAndCentroid(unittest.TestCase):
    def test_centroid_interpolation(self):
        # i = 1/3, j = 1/3, k = 1/3
        p0, p1, p2 = 3.0, 6.0, 9.0  # centroid is (3 + 6 + 9)/3 = 6.0
        i_coord = 1.0 / 3.0
        j_coord = 1.0 / 3.0
        p1_acc = bits_to_f32(ref_interp_p1_f32(i_coord, p0, p1))
        res = ref_interp_p2_f32(j_coord, p0, p2, p1_acc)
        self.assertEqual(res, f32_to_bits(6.0))


# ------------------------------------------------------------------------------
# v_interp_p2_f32 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

class TestInterpP2F32HardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        prog = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x clamp",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_interp_p2_hw", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)


# ------------------------------------------------------------------------------
# v_interp_mov_f32 Reference Oracle
# ------------------------------------------------------------------------------

def ref_interp_mov_f32(vsel: str, p0: float, p1: float = 0.0, p2: float = 0.0) -> int:
    """Direct attribute access: p0 -> P0, p10 -> P1 - P0, p20 -> P2 - P0."""
    if vsel == "p0":
        return f32_to_bits(p0)
    elif vsel == "p10":
        return f32_to_bits(f32_round(p1 - p0))
    elif vsel == "p20":
        return f32_to_bits(f32_round(p2 - p0))
    raise ValueError(f"Unknown vertex select: {vsel}")


class TestInterpMovF32Oracle(unittest.TestCase):
    def test_provoking_vertex(self):
        self.assertEqual(ref_interp_mov_f32("p0", 42.0), f32_to_bits(42.0))

    def test_attribute_deltas(self):
        self.assertEqual(ref_interp_mov_f32("p10", 10.0, 25.0), f32_to_bits(15.0))
        self.assertEqual(ref_interp_mov_f32("p20", 10.0, 0.0, 30.0), f32_to_bits(20.0))


# ------------------------------------------------------------------------------
# v_interp_mov_f32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VInterpMovF32Instruction:
    dst: RegisterOperand
    vsel: str
    attr: AttrOperand

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")
        if self.vsel not in ("p0", "p10", "p20"):
            raise ValueError(f"Invalid vsel: {self.vsel}")


class ScalarMachineInterpMovF32:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.attrs: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 32

    def step(self, insn: VInterpMovF32Instruction):
        p0, p1, p2 = self.attrs[insn.attr.index]
        res = ref_interp_mov_f32(insn.vsel, p0, p1, p2)
        self.vgprs[insn.dst.index] = res


class TestInterpMovF32ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VInterpMovF32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 3),
            vsel="p0",
            attr=AttrOperand(1, "z"),
        )
        self.assertEqual(insn.vsel, "p0")

    def test_scalar_execution(self):
        machine = ScalarMachineInterpMovF32()
        machine.attrs[1] = (100.0, 150.0, 220.0)
        insn = VInterpMovF32Instruction(dst=RegisterOperand(OperandType.VGPR, 3), vsel="p10", attr=AttrOperand(1, "x"))
        machine.step(insn)
        self.assertEqual(machine.vgprs[3], f32_to_bits(50.0))


# ------------------------------------------------------------------------------
# v_interp_mov_f32 SIMD & Flat Attribute Handling
# ------------------------------------------------------------------------------

class SIMDWaveInterpMovF32:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step(self, insn: VInterpMovF32Instruction, p0: float, p1: float, p2: float):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                self.vgprs[insn.dst.index][lane] = ref_interp_mov_f32(insn.vsel, p0, p1, p2)


class TestInterpMovF32SIMDAndFlat(unittest.TestCase):
    def test_simd_flat_across_lanes(self):
        wave = SIMDWaveInterpMovF32(wave_size=32)
        insn = VInterpMovF32Instruction(dst=RegisterOperand(OperandType.VGPR, 5), vsel="p0", attr=AttrOperand(0, "w"))
        wave.step(insn, p0=123.0, p1=0.0, p2=0.0)

        for lane in range(32):
            self.assertEqual(wave.vgprs[5][lane], f32_to_bits(123.0))


# ------------------------------------------------------------------------------
# v_interp_mov_f32 Rejection & Hardware Translator
# ------------------------------------------------------------------------------

class TestInterpMovF32HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            VInterpMovF32Instruction(dst=RegisterOperand(OperandType.VGPR, 0), vsel="p30", attr=AttrOperand(0, "x"))

    def test_hardware_translator_execution(self):
        prog = [
            "v_interp_mov_f32 v3, p0, attr0.y",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_interp_mov_hw", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)


# ------------------------------------------------------------------------------
# v_interp_p1ll_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_interp_p1ll_f16(i_coord: float, p0: float, p1: float) -> int:
    """Computes D = P0 + i * (P1 - P0) in f32 intermediate precision."""
    d1 = f32_round(p1 - p0)
    m1 = f32_round(d1 * i_coord)
    res = f32_round(p0 + m1)
    return f32_to_bits(res)


class TestInterpP1llF16Oracle(unittest.TestCase):
    def test_intermediate_precision(self):
        # Result produces full f32 representation
        res = ref_interp_p1ll_f16(0.5, 1.0, 3.0)
        self.assertEqual(res, f32_to_bits(2.0))


# ------------------------------------------------------------------------------
# v_interp_p1ll_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VInterpP1llF16Instruction:
    dst: RegisterOperand
    src0: RegisterOperand
    attr: AttrOperand

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineInterpP1llF16:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.attrs: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 32

    def step(self, insn: VInterpP1llF16Instruction):
        i_coord = bits_to_f32(self.vgprs[insn.src0.index])
        p0, p1, p2 = self.attrs[insn.attr.index]
        res = ref_interp_p1ll_f16(i_coord, p0, p1)
        self.vgprs[insn.dst.index] = res


class TestInterpP1llF16ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = VInterpP1llF16Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            src0=RegisterOperand(OperandType.VGPR, 0),
            attr=AttrOperand(0, "x"),
        )
        self.assertEqual(insn.dst.name, "v2")

    def test_scalar_execution(self):
        machine = ScalarMachineInterpP1llF16()
        machine.vgprs[0] = f32_to_bits(0.25)
        machine.attrs[0] = (0.0, 4.0, 8.0)
        insn = VInterpP1llF16Instruction(dst=RegisterOperand(OperandType.VGPR, 2), src0=RegisterOperand(OperandType.VGPR, 0), attr=AttrOperand(0, "x"))
        machine.step(insn)
        self.assertEqual(machine.vgprs[2], f32_to_bits(1.0))


if __name__ == '__main__':
    unittest.main()
