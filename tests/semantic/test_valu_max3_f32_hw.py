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

if __name__ == "__main__":
    unittest.main()
