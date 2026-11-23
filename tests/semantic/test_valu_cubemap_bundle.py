"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Cubemap Suite.

Subsystem Feature Bundle 16 covers the 4 RDNA2 cubemap coordinate projection instructions:
  1. v_cubeid_f32  - Cubemap face index identification (0.0 to 5.0)
  2. v_cubesc_f32  - Cubemap S coordinate projection (unnormalized S coordinate)
  3. v_cubetc_f32  - Cubemap T coordinate projection (unnormalized T coordinate)
  4. v_cubema_f32  - Cubemap Major Axis magnitude calculation (2 * major_axis, +0.0 for ±0)

RDNA2 Cubemap Projection Architecture:
  Given 3D vector coordinates (x, y, z):
  1. Major axis selection follows priority:
       |z| >= |x| and |z| >= |y|  => Z major
       else |y| >= |x|            => Y major
       else                       => X major
  2. Face mapping:
       +X (face 0): sc = -z, tc = -y, ma = 2*x
       -X (face 1): sc = +z, tc = -y, ma = 2*x
       +Y (face 2): sc = +x, tc = +z, ma = 2*y
       -Y (face 3): sc = +x, tc = -z, ma = 2*y
       +Z (face 4): sc = +x, tc = -y, ma = 2*z
       -Z (face 5): sc = -x, tc = -y, ma = 2*z
  3. v_cubema_f32 returns +0.0 (never -0.0) whenever the selected major axis is ±0.0.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
from typing import Optional
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack(">f", struct.pack(">I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack(">I", struct.pack(">f", f))[0]


def select_cube_face(x: float, y: float, z: float) -> tuple[int, float, float, float]:
    """RDNA2 hardware cubemap projection model returning (face_id, sc, tc, ma)."""
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        # Z major
        if z < 0.0:
            face = 5
            sc = -x
            tc = -y
            ma = 2.0 * z
        else:
            face = 4
            sc = x
            tc = -y
            ma = 2.0 * z
    elif ay >= ax:
        # Y major
        if y < 0.0:
            face = 3
            sc = x
            tc = -z
            ma = 2.0 * y
        else:
            face = 2
            sc = x
            tc = z
            ma = 2.0 * y
    else:
        # X major
        if x < 0.0:
            face = 1
            sc = z
            tc = -y
            ma = 2.0 * x
        else:
            face = 0
            sc = -z
            tc = -y
            ma = 2.0 * x
    # Normalize ±0.0 to +0.0 for cubema
    if ma == 0.0:
        ma = 0.0
    return face, sc, tc, ma


_TRANS_ID = 0


def run_cube_op(opcode: str, x: float, y: float, z: float, name: Optional[str] = None) -> float:
    """Translate and run a pixel shader executing single cubemap opcode with float arguments."""
    global _TRANS_ID
    _TRANS_ID += 1
    prog = name or f"ps_{opcode}_{_TRANS_ID}"
    lines = [
        f"v_mov_b32 v0, {f32_to_u32(x):#x}",
        f"v_mov_b32 v1, {f32_to_u32(y):#x}",
        f"v_mov_b32 v2, {f32_to_u32(z):#x}",
        f"{opcode} v3, v0, v1, v2",
        "exp mrt0 v3, v3, v3, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(prog, lines)
    words = mrt0_words(spv)
    return u32_to_f32(words[0])


# -----------------------------------------------------------------------------
# Opcode 1: v_cubeid_f32
# -----------------------------------------------------------------------------
def oracle_cubeid_f32(x: float, y: float, z: float) -> float:
    return float(select_cube_face(x, y, z)[0])


class VCubeIdOracleTests(unittest.TestCase):
    def test_cubeid_cardinal_faces(self):
        cases = [
            (1.0, 0.0, 0.0, 0.0),   # +X -> face 0.0
            (-1.0, 0.0, 0.0, 1.0),  # -X -> face 1.0
            (0.0, 1.0, 0.0, 2.0),   # +Y -> face 2.0
            (0.0, -1.0, 0.0, 3.0),  # -Y -> face 3.0
            (0.0, 0.0, 1.0, 4.0),   # +Z -> face 4.0
            (0.0, 0.0, -1.0, 5.0),  # -Z -> face 5.0
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(oracle_cubeid_f32(x, y, z), expected)
                hw = run_cube_op("v_cubeid_f32", x, y, z)
                self.assertEqual(hw, expected)


class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: float = 0.0


@dataclass(frozen=True)
class CubeInstructionAST:
    mnemonic: str
    dst: Operand
    src_x: Operand
    src_y: Operand
    src_z: Operand


class WavefrontSimdState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        self.vgprs = [[0.0] * 64 for _ in range(lane_count)]

    def set_vgpr(self, lane: int, reg_idx: int, val: float):
        self.vgprs[lane][reg_idx] = val

    def get_vgpr(self, lane: int, reg_idx: int) -> float:
        return self.vgprs[lane][reg_idx]


class VCubeIdAstTests(unittest.TestCase):
    def test_cubeid_ast_construction(self):
        ast = CubeInstructionAST(
            "v_cubeid_f32",
            Operand(RegKind.VGPR, index=3),
            Operand(RegKind.VGPR, index=0),
            Operand(RegKind.VGPR, index=1),
            Operand(RegKind.VGPR, index=2),
        )
        self.assertEqual(ast.mnemonic, "v_cubeid_f32")
        self.assertEqual(ast.dst.index, 3)
        self.assertEqual(ast.src_x.index, 0)
        self.assertEqual(ast.src_y.index, 1)
        self.assertEqual(ast.src_z.index, 2)

        # Wire AST emission through rdna2spirv translator
        lines = [
            "v_mov_b32 v0, 0.0",
            "v_mov_b32 v1, 5.0",
            "v_mov_b32 v2, 0.0",
            f"{ast.mnemonic} v{ast.dst.index}, v{ast.src_x.index}, v{ast.src_y.index}, v{ast.src_z.index}",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_cubeid", lines)
        self.assertEqual(u32_to_f32(mrt0_words(spv)[0]), 2.0)  # +Y face


class VCubeIdScalarInterpreterTests(unittest.TestCase):
    def test_cubeid_scalar_execution(self):
        x, y, z = 0.2, 0.5, 0.9  # Z major
        res = oracle_cubeid_f32(x, y, z)
        self.assertEqual(res, 4.0)
        hw = run_cube_op("v_cubeid_f32", x, y, z)
        self.assertEqual(hw, 4.0)


class VCubeIdSimdDivergenceTests(unittest.TestCase):
    def test_cubeid_exec_mask_divergence(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x55555555  # Even lanes active
        vectors = [
            (1.0, 0.0, 0.0),   # face 0
            (-1.0, 0.0, 0.0),  # face 1
            (0.0, 1.0, 0.0),   # face 2
            (0.0, -1.0, 0.0),  # face 3
            (0.0, 0.0, 1.0),   # face 4
            (0.0, 0.0, -1.0),  # face 5
        ]
        for i in range(32):
            wave.set_vgpr(i, 3, -999.0)
            if (wave.exec_mask >> i) & 1:
                vec = vectors[i % len(vectors)]
                wave.set_vgpr(i, 3, oracle_cubeid_f32(vec[0], vec[1], vec[2]))

        for i in range(32):
            if (0x55555555 >> i) & 1:
                expected_face = float(i % len(vectors))
                self.assertEqual(wave.get_vgpr(i, 3), expected_face)
            else:
                self.assertEqual(wave.get_vgpr(i, 3), -999.0)

        # Cross-validate through translator execution
        for i, vec in enumerate(vectors):
            hw = run_cube_op("v_cubeid_f32", vec[0], vec[1], vec[2])
            self.assertEqual(hw, float(i))


class VCubeIdEdgeCasesTests(unittest.TestCase):
    def test_cubeid_tie_breaking(self):
        # Tie between X, Y, Z: Z takes priority -> face 4.0 (+Z)
        self.assertEqual(oracle_cubeid_f32(1.0, 1.0, 1.0), 4.0)
        self.assertEqual(run_cube_op("v_cubeid_f32", 1.0, 1.0, 1.0), 4.0)

        # Tie between X and Y: Y takes priority -> face 2.0 (+Y)
        self.assertEqual(oracle_cubeid_f32(1.0, 1.0, 0.5), 2.0)
        self.assertEqual(run_cube_op("v_cubeid_f32", 1.0, 1.0, 0.5), 2.0)

        # Negative tie (-1, -1, -1): -Z takes priority -> face 5.0 (-Z)
        self.assertEqual(oracle_cubeid_f32(-1.0, -1.0, -1.0), 5.0)
        self.assertEqual(run_cube_op("v_cubeid_f32", -1.0, -1.0, -1.0), 5.0)


class VCubeIdTranslatorTests(unittest.TestCase):
    def test_cubeid_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 1.0",
            "v_mov_b32 v1, 2.0",
            "v_mov_b32 v2, 3.0",
            "v_cubeid_f32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cubeid_exec", lines)
        val = u32_to_f32(mrt0_words(spv)[0])
        self.assertEqual(val, 4.0)  # +Z face

    def test_cubeid_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cubeid", ["v_cubeid_f32"])


# -----------------------------------------------------------------------------
# Opcode 2: v_cubesc_f32
# -----------------------------------------------------------------------------
def oracle_cubesc_f32(x: float, y: float, z: float) -> float:
    return select_cube_face(x, y, z)[1]


class VCubeScOracleTests(unittest.TestCase):
    def test_cubesc_cardinal_coordinates(self):
        cases = [
            (2.0, 0.0, 0.5, -0.5),   # +X (face 0): sc = -z
            (-2.0, 0.0, 0.5, 0.5),   # -X (face 1): sc = +z
            (0.3, 2.0, 0.0, 0.3),    # +Y (face 2): sc = +x
            (0.4, -2.0, 0.0, 0.4),   # -Y (face 3): sc = +x
            (0.5, 0.0, 2.0, 0.5),    # +Z (face 4): sc = +x
            (0.5, 0.0, -2.0, -0.5),  # -Z (face 5): sc = -x
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(oracle_cubesc_f32(x, y, z), expected)
                hw = run_cube_op("v_cubesc_f32", x, y, z)
                self.assertAlmostEqual(hw, expected, places=5)


class VCubeScAstTests(unittest.TestCase):
    def test_cubesc_ast_construction(self):
        ast = CubeInstructionAST(
            "v_cubesc_f32",
            Operand(RegKind.VGPR, index=4),
            Operand(RegKind.VGPR, index=0),
            Operand(RegKind.VGPR, index=1),
            Operand(RegKind.VGPR, index=2),
        )
        self.assertEqual(ast.mnemonic, "v_cubesc_f32")
        self.assertEqual(ast.dst.index, 4)

        lines = [
            "v_mov_b32 v0, 0.3",
            "v_mov_b32 v1, 2.0",
            "v_mov_b32 v2, 0.0",
            f"{ast.mnemonic} v{ast.dst.index}, v{ast.src_x.index}, v{ast.src_y.index}, v{ast.src_z.index}",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_cubesc", lines)
        self.assertAlmostEqual(u32_to_f32(mrt0_words(spv)[0]), 0.3, places=5)


class VCubeScScalarInterpreterTests(unittest.TestCase):
    def test_cubesc_scalar_execution(self):
        # Z major ray: (1.5, 2.0, 4.0) -> sc = x = 1.5
        res = oracle_cubesc_f32(1.5, 2.0, 4.0)
        self.assertEqual(res, 1.5)
        hw = run_cube_op("v_cubesc_f32", 1.5, 2.0, 4.0)
        self.assertEqual(hw, 1.5)


class VCubeScSimdDivergenceTests(unittest.TestCase):
    def test_cubesc_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x0000FFFF  # Lower 16 active
        for i in range(32):
            wave.set_vgpr(i, 4, 111.0)
            if (wave.exec_mask >> i) & 1:
                # +Z major with varying X
                wave.set_vgpr(i, 4, oracle_cubesc_f32(float(i), 0.0, 100.0))
        for i in range(16):
            self.assertEqual(wave.get_vgpr(i, 4), float(i))
        for i in range(16, 32):
            self.assertEqual(wave.get_vgpr(i, 4), 111.0)

        # Cross-validate through translator execution
        hw = run_cube_op("v_cubesc_f32", 7.0, 0.0, 100.0)
        self.assertEqual(hw, 7.0)


class VCubeScEdgeCasesTests(unittest.TestCase):
    def test_cubesc_boundary_negation(self):
        # -Z major ray: (2.5, 1.0, -5.0) -> sc = -x = -2.5
        self.assertEqual(oracle_cubesc_f32(2.5, 1.0, -5.0), -2.5)
        self.assertEqual(run_cube_op("v_cubesc_f32", 2.5, 1.0, -5.0), -2.5)

        # +X major ray: (5.0, 1.0, 2.5) -> sc = -z = -2.5
        self.assertEqual(oracle_cubesc_f32(5.0, 1.0, 2.5), -2.5)
        self.assertEqual(run_cube_op("v_cubesc_f32", 5.0, 1.0, 2.5), -2.5)


class VCubeScTranslatorTests(unittest.TestCase):
    def test_cubesc_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 1.5",
            "v_mov_b32 v1, 2.0",
            "v_mov_b32 v2, 4.0",
            "v_cubesc_f32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cubesc_exec", lines)
        val = u32_to_f32(mrt0_words(spv)[0])
        self.assertEqual(val, 1.5)


# -----------------------------------------------------------------------------
# Opcode 3: v_cubetc_f32
# -----------------------------------------------------------------------------
def oracle_cubetc_f32(x: float, y: float, z: float) -> float:
    return select_cube_face(x, y, z)[2]


class VCubeTcOracleTests(unittest.TestCase):
    def test_cubetc_cardinal_coordinates(self):
        cases = [
            (2.0, 0.7, 0.0, -0.7),   # +X (face 0): tc = -y
            (-2.0, 0.8, 0.0, -0.8),  # -X (face 1): tc = -y
            (0.0, 2.0, 0.9, 0.9),    # +Y (face 2): tc = +z
            (0.0, -2.0, 0.9, -0.9),  # -Y (face 3): tc = -z
            (0.0, 0.5, 2.0, -0.5),   # +Z (face 4): tc = -y
            (0.0, 0.5, -2.0, -0.5),  # -Z (face 5): tc = -y
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(oracle_cubetc_f32(x, y, z), expected)
                hw = run_cube_op("v_cubetc_f32", x, y, z)
                self.assertAlmostEqual(hw, expected, places=5)


class VCubeTcAstTests(unittest.TestCase):
    def test_cubetc_ast_construction(self):
        ast = CubeInstructionAST(
            "v_cubetc_f32",
            Operand(RegKind.VGPR, index=5),
            Operand(RegKind.VGPR, index=0),
            Operand(RegKind.VGPR, index=1),
            Operand(RegKind.VGPR, index=2),
        )
        self.assertEqual(ast.mnemonic, "v_cubetc_f32")
        self.assertEqual(ast.dst.index, 5)

        lines = [
            "v_mov_b32 v0, 0.0",
            "v_mov_b32 v1, 2.0",
            "v_mov_b32 v2, 0.9",
            f"{ast.mnemonic} v{ast.dst.index}, v{ast.src_x.index}, v{ast.src_y.index}, v{ast.src_z.index}",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_cubetc", lines)
        self.assertAlmostEqual(u32_to_f32(mrt0_words(spv)[0]), 0.9, places=5)


class VCubeTcScalarInterpreterTests(unittest.TestCase):
    def test_cubetc_scalar_execution(self):
        # +Y major: (0.5, 3.0, 1.2) -> tc = +z = 1.2
        self.assertEqual(oracle_cubetc_f32(0.5, 3.0, 1.2), 1.2)
        self.assertAlmostEqual(run_cube_op("v_cubetc_f32", 0.5, 3.0, 1.2), 1.2, places=5)

        # -Y major: (0.5, -3.0, 1.2) -> tc = -z = -1.2
        self.assertEqual(oracle_cubetc_f32(0.5, -3.0, 1.2), -1.2)
        self.assertAlmostEqual(run_cube_op("v_cubetc_f32", 0.5, -3.0, 1.2), -1.2, places=5)


class VCubeTcSimdDivergenceTests(unittest.TestCase):
    def test_cubetc_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0xF0F0F0F0
        for i in range(32):
            wave.set_vgpr(i, 5, -555.0)
            if (wave.exec_mask >> i) & 1:
                # +Z major with varying Y
                wave.set_vgpr(i, 5, oracle_cubetc_f32(0.0, float(i), 100.0))
        for i in range(32):
            if (0xF0F0F0F0 >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 5), -float(i))
            else:
                self.assertEqual(wave.get_vgpr(i, 5), -555.0)

        hw = run_cube_op("v_cubetc_f32", 0.0, 4.0, 100.0)
        self.assertEqual(hw, -4.0)


class VCubeTcEdgeCasesTests(unittest.TestCase):
    def test_cubetc_zero_y_axis(self):
        # +Z major with y = 0.0 -> tc = -0.0 -> 0.0
        tc = oracle_cubetc_f32(0.0, 0.0, 1.0)
        self.assertEqual(tc, 0.0)
        hw = run_cube_op("v_cubetc_f32", 0.0, 0.0, 1.0)
        self.assertEqual(hw, 0.0)


class VCubeTcTranslatorTests(unittest.TestCase):
    def test_cubetc_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 1.0",
            "v_mov_b32 v1, 2.5",
            "v_mov_b32 v2, 5.0",
            "v_cubetc_f32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cubetc_exec", lines)
        val = u32_to_f32(mrt0_words(spv)[0])
        self.assertEqual(val, -2.5)  # +Z major: tc = -y = -2.5


# -----------------------------------------------------------------------------
# Opcode 4: v_cubema_f32
# -----------------------------------------------------------------------------
def oracle_cubema_f32(x: float, y: float, z: float) -> float:
    return select_cube_face(x, y, z)[3]


class VCubeMaOracleTests(unittest.TestCase):
    def test_cubema_magnitudes(self):
        cases = [
            (3.0, 1.0, 1.0, 6.0),    # +X major: 2 * x = 6.0
            (-3.0, 1.0, 1.0, -6.0),  # -X major: 2 * x = -6.0
            (1.0, 4.0, 1.0, 8.0),    # +Y major: 2 * y = 8.0
            (1.0, 1.0, 5.0, 10.0),   # +Z major: 2 * z = 10.0
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(oracle_cubema_f32(x, y, z), expected)
                hw = run_cube_op("v_cubema_f32", x, y, z)
                self.assertEqual(hw, expected)


class VCubeMaAstTests(unittest.TestCase):
    def test_cubema_ast_construction(self):
        ast = CubeInstructionAST(
            "v_cubema_f32",
            Operand(RegKind.VGPR, index=6),
            Operand(RegKind.VGPR, index=0),
            Operand(RegKind.VGPR, index=1),
            Operand(RegKind.VGPR, index=2),
        )
        self.assertEqual(ast.mnemonic, "v_cubema_f32")
        self.assertEqual(ast.dst.index, 6)

        lines = [
            "v_mov_b32 v0, 1.0",
            "v_mov_b32 v1, 4.0",
            "v_mov_b32 v2, 1.0",
            f"{ast.mnemonic} v{ast.dst.index}, v{ast.src_x.index}, v{ast.src_y.index}, v{ast.src_z.index}",
            "exp mrt0 v6, v6, v6, v6 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_cubema", lines)
        self.assertEqual(u32_to_f32(mrt0_words(spv)[0]), 8.0)


class VCubeMaScalarInterpreterTests(unittest.TestCase):
    def test_cubema_scalar_execution(self):
        res = oracle_cubema_f32(0.5, 1.0, 2.5)  # Z major -> 2 * 2.5 = 5.0
        self.assertEqual(res, 5.0)
        hw = run_cube_op("v_cubema_f32", 0.5, 1.0, 2.5)
        self.assertEqual(hw, 5.0)


class VCubeMaSimdDivergenceTests(unittest.TestCase):
    def test_cubema_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0xAAAAAAAA
        for i in range(32):
            wave.set_vgpr(i, 6, -1.0)
            if (wave.exec_mask >> i) & 1:
                wave.set_vgpr(i, 6, oracle_cubema_f32(0.0, 0.0, float(i)))
        for i in range(32):
            if (0xAAAAAAAA >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 6), 2.0 * float(i))
            else:
                self.assertEqual(wave.get_vgpr(i, 6), -1.0)

        hw = run_cube_op("v_cubema_f32", 0.0, 0.0, 8.0)
        self.assertEqual(hw, 16.0)


class VCubeMaEdgeCasesTests(unittest.TestCase):
    def test_cubema_signed_zero_normalization(self):
        # Hardware rule: v_cubema_f32 returns +0.0 (never -0.0) when major axis is ±0.0
        pos_zero_ma = oracle_cubema_f32(0.0, 0.0, 0.0)
        neg_zero_ma = oracle_cubema_f32(-0.0, -0.0, -0.0)
        self.assertEqual(pos_zero_ma, 0.0)
        self.assertEqual(neg_zero_ma, 0.0)
        # Sign bit check: must be positive (+0.0)
        self.assertEqual(f32_to_u32(pos_zero_ma), 0x00000000)
        self.assertEqual(f32_to_u32(neg_zero_ma), 0x00000000)

        # Cross-check through hardware translator execution
        hw_pos = run_cube_op("v_cubema_f32", 0.0, 0.0, 0.0)
        hw_neg = run_cube_op("v_cubema_f32", -0.0, -0.0, -0.0)
        self.assertEqual(f32_to_u32(hw_pos), 0x00000000)
        self.assertEqual(f32_to_u32(hw_neg), 0x00000000)


class VCubeMaTranslatorTests(unittest.TestCase):
    def test_cubema_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 1.0",
            "v_mov_b32 v1, 2.0",
            "v_mov_b32 v2, 3.0",
            "v_cubema_f32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cubema_exec", lines)
        val = u32_to_f32(mrt0_words(spv)[0])
        self.assertEqual(val, 6.0)  # 2 * 3.0

    def test_cubema_zero_translation(self):
        lines = [
            "v_mov_b32 v0, 0.0",
            "v_mov_b32 v1, 0.0",
            "v_mov_b32 v2, 0.0",
            "v_cubema_f32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cubema_zero_exec", lines)
        word = mrt0_words(spv)[0]
        self.assertEqual(word, 0x00000000)  # Exactly +0.0


class VCubemapPipelineTests(unittest.TestCase):
    """End-to-end full pipeline execution tests covering all four cubemap instructions together."""

    def test_full_cubemap_projection_pipeline(self):
        """Translate shader executing all 4 cubemap ops and export mrt0 (face, sc, tc, ma)."""
        test_vectors = [
            (4.0, -1.5, 0.75),      # +X
            (-4.0, 1.5, -0.75),     # -X
            (1.25, 3.0, -0.5),      # +Y
            (-1.25, -3.0, 0.5),     # -Y
            (-0.5, 1.25, 2.5),      # +Z
            (0.5, -1.25, -2.5),     # -Z
            (2.0, 2.0, 0.5),        # +Y beats +X tie
            (0.5, 2.0, 2.0),        # +Z beats +Y tie
        ]
        for x, y, z in test_vectors:
            with self.subTest(x=x, y=y, z=z):
                lines = [
                    f"v_mov_b32 v0, {f32_to_u32(x):#x}",
                    f"v_mov_b32 v1, {f32_to_u32(y):#x}",
                    f"v_mov_b32 v2, {f32_to_u32(z):#x}",
                    "v_cubeid_f32 v3, v0, v1, v2",
                    "v_cubesc_f32 v4, v0, v1, v2",
                    "v_cubetc_f32 v5, v0, v1, v2",
                    "v_cubema_f32 v6, v0, v1, v2",
                    "exp mrt0 v3, v4, v5, v6 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_cube_pipe_{f32_to_u32(x):x}", lines)
                words = mrt0_words(spv)
                got_face = u32_to_f32(words[0])
                got_sc = u32_to_f32(words[1])
                got_tc = u32_to_f32(words[2])
                got_ma = u32_to_f32(words[3])
                exp_face, exp_sc, exp_tc, exp_ma = select_cube_face(x, y, z)
                self.assertEqual(got_face, float(exp_face))
                self.assertAlmostEqual(got_sc, exp_sc, places=5)
                self.assertAlmostEqual(got_tc, exp_tc, places=5)
                self.assertAlmostEqual(got_ma, exp_ma, places=5)


if __name__ == "__main__":
    unittest.main()
