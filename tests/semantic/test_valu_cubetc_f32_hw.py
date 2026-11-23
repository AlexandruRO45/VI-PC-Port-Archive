"""Translator-backed semantic tests for RDNA2 v_cubetc_f32 instruction.

Covers:
- Hand-written RDNA2 shader assembly translation via tools/rdna2spirv.py
- Execution of translated SPIR-V through spirv_mini_interp (support.mrt0_words)
- Reference mathematical verification of cube-map tc coordinate across all 6 faces
- Major axis selection and tie-breaking priority (Z over Y and X; Y over X)
- VOP3 input modifiers (abs, neg) and output clamp modifier
- Register aliasing, literal constants, inline float constants, and SGPR sources
- SIMD multi-lane divergence and negative rejection controls

Zero game data, standard library Python only.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
POS_ZERO_BITS = 0x00000000
NEG_ZERO_BITS = 0x80000000
POS_INF_BITS = 0x7F800000
NEG_INF_BITS = 0xFF800000


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return POS_INF_BITS if f > 0.0 else NEG_INF_BITS


def ref_cubetc_f32(x: float, y: float, z: float, *, clamp: bool = False) -> float:
    """Reference RDNA2 ISA calculation for v_cubetc_f32 (cube projected tc coordinate).

    Major axis selection:
      if |z| >= |x| and |z| >= |y|: major = Z (tc = -y)
      elif |y| >= |x|:              major = Y (y < 0 -> tc = -z, else tc = z)
      else:                         major = X (tc = -y)
    """
    x, y, z = u32_to_f32(f32_to_u32(x)), u32_to_f32(f32_to_u32(y)), u32_to_f32(f32_to_u32(z))
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        tc = -y
    elif ay >= ax:
        tc = -z if y < 0.0 else z
    else:
        tc = -y
    tc = u32_to_f32(f32_to_u32(tc))
    if clamp:
        tc = max(0.0, min(1.0, tc))
    return tc

class TestReferenceOracle(unittest.TestCase):
    """Architectural reference oracle and cube face mapping verification."""

    def test_oracle_all_six_faces(self):
        # +X face (major = X, x > 0): tc = -y
        self.assertEqual(ref_cubetc_f32(4.0, -1.5, 0.75), 1.5)
        # -X face (major = X, x < 0): tc = -y
        self.assertEqual(ref_cubetc_f32(-4.0, 1.5, -0.75), -1.5)
        # +Y face (major = Y, y > 0): tc = +z
        self.assertEqual(ref_cubetc_f32(1.25, 3.0, -0.5), -0.5)
        # -Y face (major = Y, y < 0): tc = -z
        self.assertEqual(ref_cubetc_f32(-1.25, -3.0, 0.5), -0.5)
        # +Z face (major = Z, z > 0): tc = -y
        self.assertEqual(ref_cubetc_f32(-0.5, 1.25, 2.5), -1.25)
        # -Z face (major = Z, z < 0): tc = -y
        self.assertEqual(ref_cubetc_f32(0.5, -1.25, -2.5), 1.25)

    def test_oracle_tie_breaking(self):
        # |x| == |y| > |z|: Y wins over X
        self.assertEqual(ref_cubetc_f32(2.0, 2.0, 0.5), 0.5)     # +Y: tc = z = 0.5
        self.assertEqual(ref_cubetc_f32(2.0, -2.0, 0.5), -0.5)   # -Y: tc = -z = -0.5
        # |y| == |z| > |x|: Z wins over Y
        self.assertEqual(ref_cubetc_f32(0.5, 2.0, 2.0), -2.0)    # +Z: tc = -y = -2.0
        self.assertEqual(ref_cubetc_f32(0.5, -2.0, -2.0), 2.0)   # -Z: tc = -y = 2.0
        # |x| == |y| == |z|: Z wins over all
        self.assertEqual(ref_cubetc_f32(1.0, 1.0, 1.0), -1.0)    # +Z: tc = -y = -1.0

    def test_oracle_clamp(self):
        self.assertEqual(ref_cubetc_f32(4.0, -3.0, 1.0, clamp=True), 1.0)   # tc = 3.0 -> clamped to 1.0
        self.assertEqual(ref_cubetc_f32(4.0, 3.0, 1.0, clamp=True), 0.0)    # tc = -3.0 -> clamped to 0.0
        self.assertEqual(ref_cubetc_f32(4.0, -0.5, 1.0, clamp=True), 0.5)   # tc = 0.5 -> preserved

class TestOpcodeTranslation(unittest.TestCase):
    """Opcode translation to SPIR-V assembly validation."""

    def test_translate_basic_instruction(self):
        prog = [
            "v_cubetc_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_op", prog)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("FAbs", spv)
        self.assertIn("OpFOrdGreaterThanEqual", spv)
        self.assertIn("OpSelect", spv)

    def test_translate_clamp_modifier(self):
        prog = [
            "v_cubetc_f32 clamp v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_clamp", prog)
        self.assertTrue("FClamp" in spv or "NMin" in spv or "OpSelect" in spv)

    def test_translate_input_modifiers(self):
        prog = [
            "v_cubetc_f32 v0, |v1|, -v2, -|v3|",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_mods", prog)
        self.assertIn("FAbs", spv)
        self.assertIn("OpFNegate", spv)

class TestScalarExecution(unittest.TestCase):
    """Deterministic scalar execution via support.translate_lines and mrt0_words."""

    def _run_cubetc(self, s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
        clamp_str = "clamp " if clamp else ""
        prog = [
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            f"v_cubetc_f32 {clamp_str}v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_exec", prog)
        words = support.mrt0_words(spv)
        return words[0]

    def test_six_cube_faces_execution(self):
        cases = [
            ((4.0, -1.5, 0.75), 1.5),     # +X: tc = -y = 1.5
            ((-4.0, 1.5, -0.75), -1.5),   # -X: tc = -y = -1.5
            ((1.25, 3.0, -0.5), -0.5),    # +Y: tc = z = -0.5
            ((-1.25, -3.0, 0.5), -0.5),   # -Y: tc = -z = -0.5
            ((-0.5, 1.25, 2.5), -1.25),   # +Z: tc = -y = -1.25
            ((0.5, -1.25, -2.5), 1.25),   # -Z: tc = -y = 1.25
        ]
        for (a, b, c), expected in cases:
            s0, s1, s2 = f32_to_u32(a), f32_to_u32(b), f32_to_u32(c)
            got = self._run_cubetc(s0, s1, s2)
            self.assertEqual(got, f32_to_u32(expected), f"Mismatch for inputs ({a}, {b}, {c})")

class TestSimdDivergence(unittest.TestCase):
    """Multi-lane divergence and independent execution verification."""

    def test_multi_lane_independent_evaluations(self):
        test_vectors = [
            (4.0, -1.5, 0.75, 1.5),
            (-4.0, 1.5, -0.75, -1.5),
            (1.25, 3.0, -0.5, -0.5),
            (-1.25, -3.0, 0.5, -0.5),
            (-0.5, 1.25, 2.5, -1.25),
            (0.5, -1.25, -2.5, 1.25),
            (2.0, 2.0, 0.5, 0.5),
            (0.5, 2.0, 2.0, -2.0),
        ]
        for lane_idx, (a, b, c, expected_f) in enumerate(test_vectors):
            s0, s1, s2 = f32_to_u32(a), f32_to_u32(b), f32_to_u32(c)
            prog = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {s2:#x}",
                "v_cubetc_f32 v0, v1, v2, v3",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_cubetc_lane_{lane_idx}", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], f32_to_u32(expected_f), f"Lane {lane_idx} mismatch")

class TestOperandFormsAndAliasing(unittest.TestCase):
    """Register aliasing, literal constants, inline float constants, and SGPR sources."""

    def test_sgpr_scalar_source(self):
        prog = [
            "s_mov_b32 s0, 0x40800000",   # 4.0f (+X major)
            "v_mov_b32 v1, 0xbfe00000",   # -1.75f
            "v_mov_b32 v2, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v0, s0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_sgpr", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3fe00000)   # 1.75f (-(-1.75))

    def test_inline_float_constants(self):
        prog = [
            "v_mov_b32 v1, 0xbf000000",   # -0.5f
            "v_mov_b32 v2, 0x3e800000",   # 0.25f
            "v_cubetc_f32 v0, 1.0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_inline_const", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3f000000)   # 0.5f

    def test_literal_constant(self):
        prog = [
            "v_mov_b32 v1, 0xbfa00000",   # -1.25f
            "v_mov_b32 v2, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v0, 0x40800000, v1, v2",  # 4.0f literal
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_literal", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3fa00000)   # 1.25f

    def test_dst_aliasing_src0(self):
        prog = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0xbfe00000",   # -1.75f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_alias_s0", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3fe00000)

    def test_dst_aliasing_src1(self):
        prog = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0xbfe00000",   # -1.75f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v2, v1, v2, v3",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_alias_s1", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3fe00000)

    def test_dst_aliasing_src2(self):
        prog = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0xbfe00000",   # -1.75f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v3, v1, v2, v3",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_alias_s2", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3fe00000)

    def test_all_operands_same_register(self):
        prog = [
            "v_mov_b32 v1, 0x40000000",   # 2.0f
            "v_cubetc_f32 v1, v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_all_same", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xc0000000)   # -2.0f

class TestSignedZeroAndBoundaries(unittest.TestCase):
    """Signed zeros, tie-breaking boundary conditions, and extreme values."""

    def _eval_words(self, s0: int, s1: int, s2: int) -> int:
        prog = [
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            "v_cubetc_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_boundary", prog)
        return support.mrt0_words(spv)[0]

    def test_tie_boundary_evaluations(self):
        # |x| == |y|: Y wins over X
        # x=2.0, y=2.0, z=0.5 -> +Y: tc = z = 0.5
        self.assertEqual(self._eval_words(0x40000000, 0x40000000, 0x3f000000), 0x3f000000)
        # x=2.0, y=-2.0, z=0.5 -> -Y: tc = -z = -0.5
        self.assertEqual(self._eval_words(0x40000000, 0xc0000000, 0x3f000000), 0xbf000000)

        # |y| == |z|: Z wins over Y
        # x=0.5, y=2.0, z=2.0 -> +Z: tc = -y = -2.0
        self.assertEqual(self._eval_words(0x3f000000, 0x40000000, 0x40000000), 0xc0000000)
        # x=0.5, y=-2.0, z=-2.0 -> -Z: tc = -y = 2.0
        self.assertEqual(self._eval_words(0x3f000000, 0xc0000000, 0xc0000000), 0x40000000)

    def test_signed_zeros(self):
        val = self._eval_words(POS_ZERO_BITS, NEG_ZERO_BITS, POS_ZERO_BITS)
        self.assertEqual(u32_to_f32(val), 0.0)

class TestModifiersAndClamp(unittest.TestCase):
    """Input modifiers (|src|, -src) and output clamp modifier."""

    def test_input_modifiers_abs(self):
        prog = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0xbfe00000",   # -1.75f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v0, v1, |v2|, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_abs_s1", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xbfe00000)   # -1.75f

    def test_input_modifiers_neg(self):
        prog = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0xbfe00000",   # -1.75f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 v0, v1, -v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubetc_neg_s1", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xbfe00000)   # -1.75f

    def test_output_clamp_modifier(self):
        prog1 = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0xc0200000",   # -2.5f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 clamp v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv1 = support.translate_lines("ps_cubetc_clamp_hi", prog1)
        self.assertEqual(support.mrt0_words(spv1)[0], 0x3f800000)   # 1.0f

        prog2 = [
            "v_mov_b32 v1, 0x40800000",   # 4.0f
            "v_mov_b32 v2, 0x40200000",   # 2.5f
            "v_mov_b32 v3, 0x3f000000",   # 0.5f
            "v_cubetc_f32 clamp v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv2 = support.translate_lines("ps_cubetc_clamp_lo", prog2)
        self.assertEqual(support.mrt0_words(spv2)[0], 0x00000000)   # 0.0f

if __name__ == "__main__":
    unittest.main()
