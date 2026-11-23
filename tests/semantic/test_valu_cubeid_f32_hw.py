"""Semantic tests for RDNA2 v_cubeid_f32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_cubeid_f32 dst, src0, src1, src2
Computes the cubemap face index (0.0 .. 5.0) from a 3D direction vector (x, y, z):
    src0 = x, src1 = y, src2 = z (IEEE 754 binary32).
Major axis selection (Z beats Y and X; Y beats X):
    if |z| >= |x| and |z| >= |y|: major = Z (z < 0 -> face 5.0 [-Z]; else face 4.0 [+Z])
    elif |y| >= |x|:              major = Y (y < 0 -> face 3.0 [-Y]; else face 2.0 [+Y])
    else:                         major = X (x < 0 -> face 1.0 [-X]; else face 0.0 [+X])
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_cubeid(x: float, y: float, z: float) -> float:
    """Bit-accurate reference oracle for RDNA2 v_cubeid_f32."""
    x, y, z = f32(x), f32(y), f32(z)
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        return 5.0 if z < 0.0 else 4.0
    elif ay >= ax:
        return 3.0 if y < 0.0 else 2.0
    else:
        return 1.0 if x < 0.0 else 0.0


def ref_cubeid_bits(x_bits: int, y_bits: int, z_bits: int) -> int:
    x = bits_to_f32(x_bits)
    y = bits_to_f32(y_bits)
    z = bits_to_f32(z_bits)
    return f32_to_bits(ref_cubeid(x, y, z))


def run_hw_cubeid(x_bits: int, y_bits: int, z_bits: int) -> int:
    """Execute v_cubeid_f32 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {x_bits:#x}",
        f"v_mov_b32 v2, {y_bits:#x}",
        f"v_mov_b32 v3, {z_bits:#x}",
        "v_cubeid_f32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cubeid_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVCubeidF32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_all_six_faces(self):
        cases = [
            # (x, y, z, expected_face)
            (4.0, -1.5, 0.75, 0.0),   # +X (0)
            (-4.0, 1.5, -0.75, 1.0),  # -X (1)
            (1.25, 3.0, -0.5, 2.0),   # +Y (2)
            (-1.25, -3.0, 0.5, 3.0),  # -Y (3)
            (-0.5, 1.25, 2.5, 4.0),   # +Z (4)
            (0.5, -1.25, -2.5, 5.0),  # -Z (5)
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(ref_cubeid(x, y, z), expected)

    def test_reference_tie_breaker_hierarchy(self):
        # |x| == |y| (> |z|): Y wins over X
        self.assertEqual(ref_cubeid(2.0, 2.0, 0.5), 2.0)    # +Y
        self.assertEqual(ref_cubeid(2.0, -2.0, 0.5), 3.0)   # -Y
        self.assertEqual(ref_cubeid(-2.0, 2.0, -0.5), 2.0)  # +Y
        self.assertEqual(ref_cubeid(-2.0, -2.0, -0.5), 3.0) # -Y

        # |y| == |z| (> |x|): Z wins over Y
        self.assertEqual(ref_cubeid(0.5, 2.0, 2.0), 4.0)    # +Z
        self.assertEqual(ref_cubeid(0.5, -2.0, -2.0), 5.0)  # -Z

        # 3-way tie: |x| == |y| == |z|: Z wins
        self.assertEqual(ref_cubeid(1.5, -1.5, 1.5), 4.0)   # +Z
        self.assertEqual(ref_cubeid(-1.5, 1.5, -1.5), 5.0)  # -Z

    def test_translator_hw_cubeid(self):
        vectors = [
            (4.0, -1.5, 0.75),
            (-4.0, 1.5, -0.75),
            (1.25, 3.0, -0.5),
            (-1.25, -3.0, 0.5),
            (-0.5, 1.25, 2.5),
            (0.5, -1.25, -2.5),
            (2.0, 2.0, 0.5),
            (0.5, 2.0, 2.0),
        ]
        for x, y, z in vectors:
            with self.subTest(x=x, y=y, z=z):
                xb = f32_to_bits(x)
                yb = f32_to_bits(y)
                zb = f32_to_bits(z)
                hw_res = run_hw_cubeid(xb, yb, zb)
                ref_res = ref_cubeid_bits(xb, yb, zb)
                self.assertEqual(hw_res, ref_res)


class RegType(enum.Enum):
    VGPR = 1
    SGPR = 2
    IMM = 3


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    value: int = 0

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")


@dataclasses.dataclass(frozen=True)
class VCubeidF32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")


class TestVCubeidF32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 2),
            src2=Operand(RegType.IMM, 0, value=f32_to_bits(1.0)),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)
        self.assertEqual(inst.src2.value, f32_to_bits(1.0))

    def test_ast_validation_rejects_invalid_dst(self):
        with self.assertRaises(TypeError):
            VCubeidF32Instruction(
                dst=Operand(RegType.SGPR, 0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )
        with self.assertRaises(TypeError):
            VCubeidF32Instruction(
                dst=Operand(RegType.IMM, 0, value=0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )

    def test_ast_validation_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, 256)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 106)
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, -1)


if __name__ == "__main__":
    unittest.main()
