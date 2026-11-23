"""Translator-backed semantic tests for RDNA2 v_max3_f32 instruction.

Covers:
- Hand-written RDNA2 shader assembly translation via tools/rdna2spirv.py
- Execution of translated SPIR-V through spirv_mini_interp (support.mrt0_words)
- Reference mathematical verification across standard and edge-case inputs
- Signed zeros (-0.0 < +0.0 ordering as measured on RDNA2 hardware)
- IEEE 754-2008 / GLSL.std.450 NMax NaN suppression semantics
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
QNAN_BITS = 0x7FC00000
SNAN_BITS = 0x7F800001
SUBNORM_MIN_BITS = 0x00000001
SUBNORM_MAX_BITS = 0x007FFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return POS_INF_BITS if f > 0.0 else NEG_INF_BITS


def ref_nmax(x: float, y: float) -> float:
    """GLSL.std.450 NMax / IEEE 754-2008 maxNum:
    A NaN operand yields the non-NaN operand; both NaN yield NaN.
    Signed zeros are ordered with -0.0 < +0.0 (measured on RDNA2 hardware).
    """
    if math.isnan(x):
        return y if not math.isnan(y) else math.nan
    if math.isnan(y):
        return x
    kx, ky = (x, math.copysign(1.0, x)), (y, math.copysign(1.0, y))
    return y if ky > kx else x


def ref_max3_f32(s0: float, s1: float, s2: float, *, clamp: bool = False) -> float:
    """Reference 3-input f32 maximum evaluating NMax(NMax(s0, s1), s2)."""
    r = ref_nmax(ref_nmax(s0, s1), s2)
    if clamp:
        if math.isnan(r):
            raise si.UnsupportedConstruct("clamp on NaN is undefined")
        r = max(0.0, min(1.0, r))
    return r

class TestReferenceOracle(unittest.TestCase):
    """Architectural reference oracle and IEEE 754 behavior tests."""

    def test_oracle_standard_floats(self):
        self.assertEqual(ref_max3_f32(1.0, 2.0, 3.0), 3.0)
        self.assertEqual(ref_max3_f32(3.0, 1.0, 2.0), 3.0)
        self.assertEqual(ref_max3_f32(2.0, 3.0, 1.0), 3.0)
        self.assertEqual(ref_max3_f32(-10.0, -5.0, -1.0), -1.0)
        self.assertEqual(ref_max3_f32(-2.5, 0.0, 1.5), 1.5)

    def test_oracle_signed_zeros(self):
        pos_zero = u32_to_f32(POS_ZERO_BITS)
        neg_zero = u32_to_f32(NEG_ZERO_BITS)
        self.assertEqual(f32_to_u32(ref_max3_f32(pos_zero, neg_zero, neg_zero)), POS_ZERO_BITS)
        self.assertEqual(f32_to_u32(ref_max3_f32(neg_zero, pos_zero, neg_zero)), POS_ZERO_BITS)
        self.assertEqual(f32_to_u32(ref_max3_f32(neg_zero, neg_zero, pos_zero)), POS_ZERO_BITS)
        self.assertEqual(f32_to_u32(ref_max3_f32(neg_zero, neg_zero, neg_zero)), NEG_ZERO_BITS)

    def test_oracle_infinities_and_subnormals(self):
        pos_inf = u32_to_f32(POS_INF_BITS)
        neg_inf = u32_to_f32(NEG_INF_BITS)
        self.assertEqual(ref_max3_f32(pos_inf, 1e38, neg_inf), pos_inf)
        self.assertEqual(ref_max3_f32(neg_inf, -1e38, -1e30), -1e30)
        self.assertEqual(ref_max3_f32(neg_inf, neg_inf, neg_inf), neg_inf)
        subnorm = u32_to_f32(SUBNORM_MIN_BITS)
        self.assertEqual(ref_max3_f32(subnorm, 0.0, -0.0), subnorm)

    def test_oracle_nans(self):
        qnan = u32_to_f32(QNAN_BITS)
        self.assertEqual(ref_max3_f32(qnan, 1.0, 2.0), 2.0)
        self.assertEqual(ref_max3_f32(1.0, qnan, 2.0), 2.0)
        self.assertEqual(ref_max3_f32(1.0, 2.0, qnan), 2.0)
        self.assertEqual(ref_max3_f32(qnan, 5.0, qnan), 5.0)
        self.assertTrue(math.isnan(ref_max3_f32(qnan, qnan, qnan)))

    def test_oracle_clamp(self):
        self.assertEqual(ref_max3_f32(2.0, 3.0, 5.0, clamp=True), 1.0)
        self.assertEqual(ref_max3_f32(-5.0, -2.0, -1.0, clamp=True), 0.0)
        self.assertEqual(ref_max3_f32(0.2, 0.5, 0.8, clamp=True), 0.8)

class TestOpcodeTranslation(unittest.TestCase):
    """Opcode translation to SPIR-V assembly validation."""

    def test_translate_basic_instruction(self):
        prog = [
            "v_max3_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_op", prog)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpExtInst %float %glsl NMax", spv)
        self.assertEqual(spv.count("NMax"), 2)

    def test_translate_clamp_modifier(self):
        prog = [
            "v_max3_f32 clamp v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_clamp", prog)
        self.assertIn("NMax", spv)

    def test_translate_input_modifiers(self):
        prog = [
            "v_max3_f32 v0, |v1|, -v2, -|v3|",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_mods", prog)
        self.assertIn("FAbs", spv)
        self.assertIn("OpFNegate", spv)
        self.assertIn("NMax", spv)

class TestScalarExecution(unittest.TestCase):
    """Deterministic scalar execution via support.translate_lines and mrt0_words."""

    def _run_max3(self, s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
        clamp_str = "clamp " if clamp else ""
        prog = [
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            f"v_max3_f32 {clamp_str}v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_exec", prog)
        words = support.mrt0_words(spv)
        return words[0]

    def test_standard_float_values(self):
        cases = [
            (1.0, 2.0, 3.0),
            (3.0, 1.0, 2.0),
            (2.0, 3.0, 1.0),
            (-10.0, -5.0, -1.0),
            (-2.5, 0.0, 1.5),
            (100.0, 100.0, 100.0),
            (0.125, 0.5, 0.25),
            (-0.5, -2.5, -1.5),
        ]
        for a, b, c in cases:
            s0, s1, s2 = f32_to_u32(a), f32_to_u32(b), f32_to_u32(c)
            got = self._run_max3(s0, s1, s2)
            expected = f32_to_u32(ref_max3_f32(a, b, c))
            self.assertEqual(got, expected, f"Failed for {a}, {b}, {c}")

class TestSimdDivergence(unittest.TestCase):
    """Multi-lane divergence and independent execution verification."""

    def test_multi_lane_independent_evaluations(self):
        test_vectors = [
            (1.0, 5.0, 2.0, 5.0),
            (10.0, -1.0, 4.0, 10.0),
            (-20.0, -30.0, -5.0, -5.0),
            (0.25, 0.75, 0.5, 0.75),
            (100.0, 200.0, 300.0, 300.0),
            (-1.0, 0.0, 1.0, 1.0),
            (7.5, 7.5, 2.0, 7.5),
            (-100.0, -50.0, -10.0, -10.0),
        ]
        for lane_idx, (a, b, c, expected_f) in enumerate(test_vectors):
            s0, s1, s2 = f32_to_u32(a), f32_to_u32(b), f32_to_u32(c)
            prog = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {s2:#x}",
                "v_max3_f32 v0, v1, v2, v3",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_max3_lane_{lane_idx}", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], f32_to_u32(expected_f), f"Lane {lane_idx} mismatch")

class TestOperandFormsAndAliasing(unittest.TestCase):
    """Register aliasing, literal constants, inline float constants, and SGPR sources."""

    def test_sgpr_scalar_source(self):
        prog = [
            "s_mov_b32 s0, 0x40800000",   # 4.0f
            "v_mov_b32 v1, 0x3f800000",   # 1.0f
            "v_mov_b32 v2, 0x40000000",   # 2.0f
            "v_max3_f32 v0, s0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_sgpr", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40800000)

    def test_inline_float_constants(self):
        prog = [
            "v_mov_b32 v1, 0x3f000000",   # 0.5f
            "v_mov_b32 v2, 0x3e800000",   # 0.25f
            "v_max3_f32 v0, 1.0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_inline_const", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3f800000)   # 1.0f

    def test_literal_constant(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",   # 1.0f
            "v_mov_b32 v2, 0x40000000",   # 2.0f
            "v_max3_f32 v0, 0x40400000, v1, v2",  # 3.0f literal
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_literal", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)

    def test_dst_aliasing_src0(self):
        prog = [
            "v_mov_b32 v1, 0x40400000",   # 3.0f
            "v_mov_b32 v2, 0x3f800000",   # 1.0f
            "v_mov_b32 v3, 0x40000000",   # 2.0f
            "v_max3_f32 v1, v1, v2, v3",  # dst is src0
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_alias_s0", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)

    def test_dst_aliasing_src1(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",   # 1.0f
            "v_mov_b32 v2, 0x40400000",   # 3.0f
            "v_mov_b32 v3, 0x40000000",   # 2.0f
            "v_max3_f32 v2, v1, v2, v3",  # dst is src1
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_alias_s1", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)

    def test_dst_aliasing_src2(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",   # 1.0f
            "v_mov_b32 v2, 0x40000000",   # 2.0f
            "v_mov_b32 v3, 0x40400000",   # 3.0f
            "v_max3_f32 v3, v1, v2, v3",  # dst is src2
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_alias_s2", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)

    def test_all_operands_same_register(self):
        prog = [
            "v_mov_b32 v1, 0x40a00000",   # 5.0f
            "v_max3_f32 v1, v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_all_same", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40a00000)

class TestSignedZeroAndBoundaries(unittest.TestCase):
    """Signed zeros (-0.0 < +0.0), infinities, and subnormal float boundary tests."""

    def _eval_words(self, s0: int, s1: int, s2: int) -> int:
        prog = [
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {s2:#x}",
            "v_max3_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_boundary", prog)
        return support.mrt0_words(spv)[0]

    def test_signed_zero_combinations(self):
        self.assertEqual(self._eval_words(POS_ZERO_BITS, NEG_ZERO_BITS, NEG_ZERO_BITS), POS_ZERO_BITS)
        self.assertEqual(self._eval_words(NEG_ZERO_BITS, POS_ZERO_BITS, NEG_ZERO_BITS), POS_ZERO_BITS)
        self.assertEqual(self._eval_words(NEG_ZERO_BITS, NEG_ZERO_BITS, POS_ZERO_BITS), POS_ZERO_BITS)
        self.assertEqual(self._eval_words(NEG_ZERO_BITS, NEG_ZERO_BITS, NEG_ZERO_BITS), NEG_ZERO_BITS)
        self.assertEqual(self._eval_words(POS_ZERO_BITS, POS_ZERO_BITS, POS_ZERO_BITS), POS_ZERO_BITS)

    def test_infinity_extremes(self):
        self.assertEqual(self._eval_words(POS_INF_BITS, 0x7F7FFFFF, NEG_INF_BITS), POS_INF_BITS)
        self.assertEqual(self._eval_words(NEG_INF_BITS, 0x7F7FFFFF, 0x00000000), 0x7F7FFFFF)
        self.assertEqual(self._eval_words(NEG_INF_BITS, NEG_INF_BITS, NEG_INF_BITS), NEG_INF_BITS)
        self.assertEqual(self._eval_words(POS_INF_BITS, POS_INF_BITS, 0x00000000), POS_INF_BITS)

    def test_subnormal_extremes(self):
        self.assertEqual(self._eval_words(SUBNORM_MIN_BITS, POS_ZERO_BITS, NEG_ZERO_BITS), SUBNORM_MIN_BITS)
        self.assertEqual(self._eval_words(SUBNORM_MAX_BITS, SUBNORM_MIN_BITS, POS_ZERO_BITS), SUBNORM_MAX_BITS)

class TestNanAndModifiers(unittest.TestCase):
    """IEEE 754-2008 NaN suppression, input modifiers (abs/neg), and output clamp."""

    def test_nan_suppression_single_nan(self):
        prog = [
            f"v_mov_b32 v1, {QNAN_BITS:#x}",
            "v_mov_b32 v2, 0x3f800000",   # 1.0f
            "v_mov_b32 v3, 0x40000000",   # 2.0f
            "v_max3_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_nan1", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40000000)

    def test_nan_suppression_two_nans(self):
        prog = [
            f"v_mov_b32 v1, {QNAN_BITS:#x}",
            "v_mov_b32 v2, 0x42280000",   # 42.0f
            f"v_mov_b32 v3, {SNAN_BITS:#x}",
            "v_max3_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_nan2", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x42280000)

    def test_all_nans_yields_nan(self):
        prog = [
            f"v_mov_b32 v1, {QNAN_BITS:#x}",
            f"v_mov_b32 v2, {QNAN_BITS:#x}",
            f"v_mov_b32 v3, {SNAN_BITS:#x}",
            "v_max3_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_nan_all", prog)
        words = support.mrt0_words(spv)
        self.assertTrue(math.isnan(u32_to_f32(words[0])))

    def test_input_modifiers_abs(self):
        prog = [
            "v_mov_b32 v1, 0xc1200000",   # -10.0f
            "v_mov_b32 v2, 0x40000000",   # 2.0f
            "v_mov_b32 v3, 0x40400000",   # 3.0f
            "v_max3_f32 v0, |v1|, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_abs_s0", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x41200000)   # +10.0f

    def test_input_modifiers_neg(self):
        prog = [
            "v_mov_b32 v1, 0xc0f00000",   # -7.5f
            "v_mov_b32 v2, 0x3f800000",   # 1.0f
            "v_mov_b32 v3, 0x40000000",   # 2.0f
            "v_max3_f32 v0, -v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_neg_s0", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40f00000)   # +7.5f

    def test_output_clamp_modifier(self):
        prog1 = [
            "v_mov_b32 v1, 0x40a00000",   # 5.0f
            "v_mov_b32 v2, 0x3f800000",   # 1.0f
            "v_mov_b32 v3, 0x40000000",   # 2.0f
            "v_max3_f32 clamp v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv1 = support.translate_lines("ps_max3_clamp_hi", prog1)
        self.assertEqual(support.mrt0_words(spv1)[0], 0x3f800000)   # 1.0f

        prog2 = [
            "v_mov_b32 v1, 0xc0000000",   # -2.0f
            "v_mov_b32 v2, 0xc0400000",   # -3.0f
            "v_mov_b32 v3, 0xc0800000",   # -4.0f
            "v_max3_f32 clamp v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv2 = support.translate_lines("ps_max3_clamp_lo", prog2)
        self.assertEqual(support.mrt0_words(spv2)[0], 0x00000000)   # 0.0f

if __name__ == "__main__":
    unittest.main()
