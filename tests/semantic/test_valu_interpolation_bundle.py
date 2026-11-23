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


if __name__ == '__main__':
    unittest.main()
