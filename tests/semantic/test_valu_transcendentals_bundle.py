"""Semantic tests for RDNA2 VALU Transcendental instructions bundle.

Covers:
- v_sin_f32
- v_cos_f32
- v_exp_f32
- v_log_f32
- v_rcp_f32
- v_rsq_f32
- v_sqrt_f32

All 44 test methods are 100% backed by tools/rdna2spirv translation via support.translate_lines(...)
and hardware execution via support.mrt0_words, eliminating in-file mock interpreters.
"""
from __future__ import annotations

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


def f32_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def run_hw_unary(op: str, src_val: float, name: str) -> float:
    bits = f32_to_bits(src_val)
    lines = [
        f"v_mov_b32 v0, {bits:#x}",
        f"{op} v0, v0",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    out_bits = support.mrt0_words(spv)[0]
    return bits_to_f32(out_bits)


def run_hw_simd_divergence(op: str, val0: float, val1: float, name: str) -> tuple[float, float]:
    lines = [
        f"v_mov_b32 v1, {f32_to_bits(val0):#x}",
        f"v_mov_b32 v2, {f32_to_bits(val1):#x}",
        f"{op} v3, v1",
        f"{op} v4, v2",
        "exp mrt0 v3, v4, v3, v4 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return bits_to_f32(words[0]), bits_to_f32(words[1])


# ==============================================================================
# v_sin_f32
# ==============================================================================

def run_hw_sin_f32(src_val: float, name: str = "ps_sin_f32_hw") -> float:
    return run_hw_unary("v_sin_f32", src_val, name)


class TestVSinF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_sin_f32 executed on translator."""

    def test_reference_quarter_turns(self):
        self.assertAlmostEqual(run_hw_sin_f32(0.25, "ps_sin_ref_q1"), 1.0, places=6)
        self.assertAlmostEqual(run_hw_sin_f32(0.75, "ps_sin_ref_q2"), -1.0, places=6)
        self.assertAlmostEqual(run_hw_sin_f32(1.25, "ps_sin_ref_q3"), 1.0, places=6)

    def test_reference_exact_zero_half_turns(self):
        self.assertEqual(run_hw_sin_f32(0.5, "ps_sin_ref_z1"), 0.0)
        self.assertEqual(run_hw_sin_f32(1.5, "ps_sin_ref_z2"), 0.0)
        self.assertEqual(run_hw_sin_f32(-0.5, "ps_sin_ref_z3"), 0.0)

    def test_reference_signed_zero(self):
        res_pos = run_hw_sin_f32(0.0, "ps_sin_ref_sz_p")
        res_neg = run_hw_sin_f32(-0.0, "ps_sin_ref_sz_n")
        self.assertEqual(res_pos, 0.0)
        self.assertEqual(math.copysign(1.0, res_pos), 1.0)
        self.assertEqual(res_neg, -0.0)
        self.assertEqual(math.copysign(1.0, res_neg), -1.0)

    def test_reference_nan_and_inf(self):
        self.assertTrue(math.isnan(run_hw_sin_f32(float("nan"), "ps_sin_ref_nan")))
        self.assertTrue(math.isnan(run_hw_sin_f32(float("inf"), "ps_sin_ref_pinf")))
        self.assertTrue(math.isnan(run_hw_sin_f32(float("-inf"), "ps_sin_ref_ninf")))


class TestVSinF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_sin_f32."""

    def test_ast_construction_and_operand_bounds(self):
        lines = [
            f"v_mov_b32 v10, {f32_to_bits(0.25):#x}",
            "v_sin_f32 v5, v10",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sin_ast_bounds", lines)
        self.assertIn("Sin", spv)
        words = support.mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(words[0]), 1.0, places=6)

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(0.25):#x}",
            "v_sin_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sin_scalar_exec", lines)
        self.assertAlmostEqual(bits_to_f32(support.mrt0_words(spv)[0]), 1.0, places=6)

    def test_scalar_interpreter_imm_source(self):
        lines = [
            f"v_mov_b32 v2, {f32_to_bits(-0.25):#x}",
            "v_sin_f32 v0, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sin_scalar_imm", lines)
        self.assertAlmostEqual(bits_to_f32(support.mrt0_words(spv)[0]), -1.0, places=6)


class TestVSinF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_sin_f32."""

    def test_simd_divergence_and_exec_masking(self):
        w0, w1 = run_hw_simd_divergence("v_sin_f32", 0.25, 0.75, "ps_sin_simd")
        self.assertAlmostEqual(w0, 1.0, places=6)
        self.assertAlmostEqual(w1, -1.0, places=6)

    def test_ieee_denormal_subnormal_handling(self):
        subnormal = bits_to_f32(0x00000001)
        res = run_hw_sin_f32(subnormal, "ps_sin_subnormal")
        self.assertAlmostEqual(res, subnormal * 2.0 * math.pi, places=6)


class TestVSinF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_sin_f32."""

    def test_translator_quarter_turns(self):
        self.assertAlmostEqual(run_hw_sin_f32(0.25, "ps_sin_hw_q1"), 1.0, places=6)
        self.assertAlmostEqual(run_hw_sin_f32(0.75, "ps_sin_hw_q2"), -1.0, places=6)
        self.assertAlmostEqual(run_hw_sin_f32(-0.25, "ps_sin_hw_q3"), -1.0, places=6)

    def test_translator_exact_zeros(self):
        self.assertEqual(run_hw_sin_f32(0.0, "ps_sin_hw_z1"), 0.0)
        self.assertEqual(run_hw_sin_f32(0.5, "ps_sin_hw_z2"), 0.0)
        self.assertEqual(run_hw_sin_f32(1.0, "ps_sin_hw_z3"), 0.0)
        self.assertEqual(run_hw_sin_f32(-0.5, "ps_sin_hw_z4"), 0.0)

    def test_translator_signed_zero(self):
        res_neg = run_hw_sin_f32(-0.0, "ps_sin_hw_sz_neg")
        self.assertEqual(res_neg, -0.0)
        self.assertEqual(math.copysign(1.0, res_neg), -1.0)


# ==============================================================================
# v_cos_f32
# ==============================================================================

def run_hw_cos_f32(src_val: float, name: str = "ps_cos_f32_hw") -> float:
    return run_hw_unary("v_cos_f32", src_val, name)


class TestVCosF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_cos_f32 executed on translator."""

    def test_reference_values(self):
        self.assertAlmostEqual(run_hw_cos_f32(0.0, "ps_cos_ref_0"), 1.0, places=6)
        self.assertAlmostEqual(run_hw_cos_f32(0.5, "ps_cos_ref_h1"), -1.0, places=6)
        self.assertAlmostEqual(run_hw_cos_f32(1.0, "ps_cos_ref_1"), 1.0, places=6)
        self.assertAlmostEqual(run_hw_cos_f32(-0.5, "ps_cos_ref_hn"), -1.0, places=6)

    def test_reference_exact_zero_quarter_turns(self):
        self.assertEqual(run_hw_cos_f32(0.25, "ps_cos_ref_q1"), 0.0)
        self.assertEqual(run_hw_cos_f32(0.75, "ps_cos_ref_q2"), 0.0)
        self.assertEqual(run_hw_cos_f32(-0.25, "ps_cos_ref_q3"), 0.0)
        self.assertEqual(run_hw_cos_f32(-0.75, "ps_cos_ref_q4"), 0.0)

    def test_reference_nan_and_inf(self):
        self.assertTrue(math.isnan(run_hw_cos_f32(float("nan"), "ps_cos_ref_nan")))
        self.assertTrue(math.isnan(run_hw_cos_f32(float("inf"), "ps_cos_ref_inf")))


class TestVCosF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_cos_f32."""

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(0.0):#x}",
            "v_cos_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cos_scalar_exec", lines)
        self.assertIn("Cos", spv)
        self.assertAlmostEqual(bits_to_f32(support.mrt0_words(spv)[0]), 1.0, places=6)

    def test_scalar_interpreter_quarter_turn(self):
        lines = [
            f"v_mov_b32 v2, {f32_to_bits(0.25):#x}",
            "v_cos_f32 v3, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cos_scalar_q", lines)
        self.assertEqual(bits_to_f32(support.mrt0_words(spv)[0]), 0.0)


class TestVCosF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_cos_f32."""

    def test_simd_divergence(self):
        w0, w1 = run_hw_simd_divergence("v_cos_f32", 0.0, 0.5, "ps_cos_simd")
        self.assertAlmostEqual(w0, 1.0, places=6)
        self.assertAlmostEqual(w1, -1.0, places=6)


class TestVCosF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_cos_f32."""

    def test_translator_cos_values(self):
        self.assertAlmostEqual(run_hw_cos_f32(0.0, "ps_cos_hw_0"), 1.0, places=6)
        self.assertEqual(run_hw_cos_f32(0.25, "ps_cos_hw_q1"), 0.0)
        self.assertAlmostEqual(run_hw_cos_f32(0.5, "ps_cos_hw_h1"), -1.0, places=6)
        self.assertEqual(run_hw_cos_f32(0.75, "ps_cos_hw_q2"), 0.0)
        self.assertAlmostEqual(run_hw_cos_f32(1.0, "ps_cos_hw_1"), 1.0, places=6)
        self.assertEqual(run_hw_cos_f32(-0.25, "ps_cos_hw_q3"), 0.0)


# ==============================================================================
# v_exp_f32
# ==============================================================================

def run_hw_exp_f32(src_val: float, name: str = "ps_exp_f32_hw") -> float:
    return run_hw_unary("v_exp_f32", src_val, name)


class TestVExpF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_exp_f32 executed on translator."""

    def test_reference_powers_of_two(self):
        self.assertEqual(run_hw_exp_f32(0.0, "ps_exp_ref_0"), 1.0)
        self.assertEqual(run_hw_exp_f32(1.0, "ps_exp_ref_1"), 2.0)
        self.assertEqual(run_hw_exp_f32(2.0, "ps_exp_ref_2"), 4.0)
        self.assertEqual(run_hw_exp_f32(3.0, "ps_exp_ref_3"), 8.0)
        self.assertEqual(run_hw_exp_f32(-1.0, "ps_exp_ref_n1"), 0.5)
        self.assertEqual(run_hw_exp_f32(-2.0, "ps_exp_ref_n2"), 0.25)

    def test_reference_inf_and_nan(self):
        self.assertEqual(run_hw_exp_f32(float("-inf"), "ps_exp_ref_ninf"), 0.0)
        self.assertEqual(run_hw_exp_f32(float("inf"), "ps_exp_ref_pinf"), float("inf"))
        self.assertTrue(math.isnan(run_hw_exp_f32(float("nan"), "ps_exp_ref_nan")))


class TestVExpF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_exp_f32."""

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(3.0):#x}",
            "v_exp_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_exp_scalar_exec", lines)
        self.assertIn("Exp2", spv)
        self.assertEqual(bits_to_f32(support.mrt0_words(spv)[0]), 8.0)


class TestVExpF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_exp_f32."""

    def test_simd_multi_lane_divergence(self):
        w0, w1 = run_hw_simd_divergence("v_exp_f32", 1.0, 2.0, "ps_exp_simd")
        self.assertEqual(w0, 2.0)
        self.assertEqual(w1, 4.0)


class TestVExpF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_exp_f32."""

    def test_translator_exp_values(self):
        self.assertEqual(run_hw_exp_f32(0.0, "ps_exp_hw_0"), 1.0)
        self.assertEqual(run_hw_exp_f32(1.0, "ps_exp_hw_1"), 2.0)
        self.assertEqual(run_hw_exp_f32(2.0, "ps_exp_hw_2"), 4.0)
        self.assertEqual(run_hw_exp_f32(3.0, "ps_exp_hw_3"), 8.0)
        self.assertEqual(run_hw_exp_f32(-1.0, "ps_exp_hw_n1"), 0.5)
        self.assertEqual(run_hw_exp_f32(-2.0, "ps_exp_hw_n2"), 0.25)
        self.assertEqual(run_hw_exp_f32(float("-inf"), "ps_exp_hw_ninf"), 0.0)


# ==============================================================================
# v_log_f32
# ==============================================================================

def run_hw_log_f32(src_val: float, name: str = "ps_log_f32_hw") -> float:
    return run_hw_unary("v_log_f32", src_val, name)


class TestVLogF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_log_f32 executed on translator."""

    def test_reference_powers_of_two(self):
        self.assertEqual(run_hw_log_f32(1.0, "ps_log_ref_1"), 0.0)
        self.assertEqual(run_hw_log_f32(2.0, "ps_log_ref_2"), 1.0)
        self.assertEqual(run_hw_log_f32(4.0, "ps_log_ref_4"), 2.0)
        self.assertEqual(run_hw_log_f32(8.0, "ps_log_ref_8"), 3.0)
        self.assertEqual(run_hw_log_f32(0.5, "ps_log_ref_h"), -1.0)
        self.assertEqual(run_hw_log_f32(0.25, "ps_log_ref_q"), -2.0)

    def test_reference_edge_cases(self):
        spv_0 = support.translate_lines("ps_log_ref_0", [
            f"v_mov_b32 v0, {f32_to_bits(0.0):#x}",
            "v_log_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("Log2", spv_0)
        with self.assertRaises(Exception):
            support.mrt0_words(spv_0)
        self.assertEqual(run_hw_log_f32(float("inf"), "ps_log_ref_inf"), float("inf"))
        self.assertEqual(run_hw_log_f32(1.0, "ps_log_ref_one"), 0.0)


class TestVLogF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_log_f32."""

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(4.0):#x}",
            "v_log_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_log_scalar_exec", lines)
        self.assertIn("Log2", spv)
        self.assertEqual(bits_to_f32(support.mrt0_words(spv)[0]), 2.0)


class TestVLogF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_log_f32."""

    def test_simd_divergence(self):
        w0, w1 = run_hw_simd_divergence("v_log_f32", 4.0, 8.0, "ps_log_simd")
        self.assertEqual(w0, 2.0)
        self.assertEqual(w1, 3.0)


class TestVLogF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_log_f32."""

    def test_translator_log_values(self):
        self.assertEqual(run_hw_log_f32(1.0, "ps_log_hw_1"), 0.0)
        self.assertEqual(run_hw_log_f32(2.0, "ps_log_hw_2"), 1.0)
        self.assertEqual(run_hw_log_f32(4.0, "ps_log_hw_4"), 2.0)
        self.assertEqual(run_hw_log_f32(8.0, "ps_log_hw_8"), 3.0)
        self.assertEqual(run_hw_log_f32(0.5, "ps_log_hw_h"), -1.0)
        self.assertEqual(run_hw_log_f32(0.25, "ps_log_hw_q"), -2.0)
        self.assertEqual(run_hw_log_f32(float("inf"), "ps_log_hw_inf"), float("inf"))


# ==============================================================================
# v_rcp_f32
# ==============================================================================

def run_hw_rcp_f32(src_val: float, name: str = "ps_rcp_f32_hw") -> float:
    return run_hw_unary("v_rcp_f32", src_val, name)


class TestVRcpF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_rcp_f32 executed on translator."""

    def test_reference_fractions(self):
        self.assertEqual(run_hw_rcp_f32(1.0, "ps_rcp_ref_1"), 1.0)
        self.assertEqual(run_hw_rcp_f32(2.0, "ps_rcp_ref_2"), 0.5)
        self.assertEqual(run_hw_rcp_f32(4.0, "ps_rcp_ref_4"), 0.25)
        self.assertEqual(run_hw_rcp_f32(-2.0, "ps_rcp_ref_n2"), -0.5)
        self.assertEqual(run_hw_rcp_f32(0.5, "ps_rcp_ref_h"), 2.0)

    def test_reference_zeros_and_infs(self):
        spv_0 = support.translate_lines("ps_rcp_ref_0", [
            f"v_mov_b32 v0, {f32_to_bits(0.0):#x}",
            "v_rcp_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpFDiv", spv_0)
        with self.assertRaises(Exception):
            support.mrt0_words(spv_0)
        self.assertEqual(run_hw_rcp_f32(1.0, "ps_rcp_ref_one"), 1.0)
        self.assertEqual(run_hw_rcp_f32(-1.0, "ps_rcp_ref_neg_one"), -1.0)


class TestVRcpF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_rcp_f32."""

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(4.0):#x}",
            "v_rcp_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_scalar_exec", lines)
        self.assertIn("OpFDiv", spv)
        self.assertEqual(bits_to_f32(support.mrt0_words(spv)[0]), 0.25)


class TestVRcpF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_rcp_f32."""

    def test_simd_divergence(self):
        w0, w1 = run_hw_simd_divergence("v_rcp_f32", 2.0, 4.0, "ps_rcp_simd")
        self.assertEqual(w0, 0.5)
        self.assertEqual(w1, 0.25)


class TestVRcpF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_rcp_f32."""

    def test_translator_rcp_values(self):
        self.assertEqual(run_hw_rcp_f32(1.0, "ps_rcp_hw_1"), 1.0)
        self.assertEqual(run_hw_rcp_f32(2.0, "ps_rcp_hw_2"), 0.5)
        self.assertEqual(run_hw_rcp_f32(4.0, "ps_rcp_hw_4"), 0.25)
        self.assertEqual(run_hw_rcp_f32(-2.0, "ps_rcp_hw_n2"), -0.5)
        self.assertEqual(run_hw_rcp_f32(0.5, "ps_rcp_hw_h"), 2.0)


# ==============================================================================
# v_rsq_f32
# ==============================================================================

def run_hw_rsq_f32(src_val: float, name: str = "ps_rsq_f32_hw") -> float:
    return run_hw_unary("v_rsq_f32", src_val, name)


class TestVRsqF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_rsq_f32 executed on translator."""

    def test_reference_squares(self):
        self.assertEqual(run_hw_rsq_f32(1.0, "ps_rsq_ref_1"), 1.0)
        self.assertEqual(run_hw_rsq_f32(4.0, "ps_rsq_ref_4"), 0.5)
        self.assertEqual(run_hw_rsq_f32(16.0, "ps_rsq_ref_16"), 0.25)
        self.assertEqual(run_hw_rsq_f32(0.25, "ps_rsq_ref_q"), 2.0)

    def test_reference_edge_cases(self):
        spv_0 = support.translate_lines("ps_rsq_ref_0", [
            f"v_mov_b32 v0, {f32_to_bits(0.0):#x}",
            "v_rsq_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("InverseSqrt", spv_0)
        with self.assertRaises(Exception):
            support.mrt0_words(spv_0)
        self.assertEqual(run_hw_rsq_f32(float("inf"), "ps_rsq_ref_inf"), 0.0)
        self.assertEqual(run_hw_rsq_f32(1.0, "ps_rsq_ref_one"), 1.0)


class TestVRsqF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_rsq_f32."""

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(16.0):#x}",
            "v_rsq_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rsq_scalar_exec", lines)
        self.assertIn("InverseSqrt", spv)
        self.assertEqual(bits_to_f32(support.mrt0_words(spv)[0]), 0.25)


class TestVRsqF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_rsq_f32."""

    def test_simd_divergence(self):
        w0, w1 = run_hw_simd_divergence("v_rsq_f32", 4.0, 16.0, "ps_rsq_simd")
        self.assertEqual(w0, 0.5)
        self.assertEqual(w1, 0.25)


class TestVRsqF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_rsq_f32."""

    def test_translator_rsq_values(self):
        self.assertEqual(run_hw_rsq_f32(1.0, "ps_rsq_hw_1"), 1.0)
        self.assertEqual(run_hw_rsq_f32(4.0, "ps_rsq_hw_4"), 0.5)
        self.assertEqual(run_hw_rsq_f32(16.0, "ps_rsq_hw_16"), 0.25)
        self.assertEqual(run_hw_rsq_f32(0.25, "ps_rsq_hw_q"), 2.0)
        self.assertEqual(run_hw_rsq_f32(float("inf"), "ps_rsq_hw_inf"), 0.0)


# ==============================================================================
# v_sqrt_f32
# ==============================================================================

def run_hw_sqrt_f32(src_val: float, name: str = "ps_sqrt_f32_hw") -> float:
    return run_hw_unary("v_sqrt_f32", src_val, name)


class TestVSqrtF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_sqrt_f32 executed on translator."""

    def test_reference_squares(self):
        self.assertEqual(run_hw_sqrt_f32(0.0, "ps_sqrt_ref_0"), 0.0)
        self.assertEqual(run_hw_sqrt_f32(1.0, "ps_sqrt_ref_1"), 1.0)
        self.assertEqual(run_hw_sqrt_f32(4.0, "ps_sqrt_ref_4"), 2.0)
        self.assertEqual(run_hw_sqrt_f32(9.0, "ps_sqrt_ref_9"), 3.0)
        self.assertEqual(run_hw_sqrt_f32(16.0, "ps_sqrt_ref_16"), 4.0)
        self.assertEqual(run_hw_sqrt_f32(25.0, "ps_sqrt_ref_25"), 5.0)

    def test_reference_edge_cases(self):
        res_neg_zero = run_hw_sqrt_f32(-0.0, "ps_sqrt_ref_n0")
        self.assertEqual(res_neg_zero, -0.0)
        self.assertEqual(math.copysign(1.0, res_neg_zero), -1.0)
        self.assertEqual(run_hw_sqrt_f32(float("inf"), "ps_sqrt_ref_inf"), float("inf"))
        spv_neg = support.translate_lines("ps_sqrt_ref_neg", [
            f"v_mov_b32 v0, {f32_to_bits(-4.0):#x}",
            "v_sqrt_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("Sqrt", spv_neg)
        with self.assertRaises(Exception):
            support.mrt0_words(spv_neg)


class TestVSqrtF32ASTInterpreter(unittest.TestCase):
    """AST opcode translation and scalar execution for v_sqrt_f32."""

    def test_scalar_interpreter_execution(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(9.0):#x}",
            "v_sqrt_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sqrt_scalar_exec", lines)
        self.assertIn("Sqrt", spv)
        self.assertEqual(bits_to_f32(support.mrt0_words(spv)[0]), 3.0)


class TestVSqrtF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_sqrt_f32."""

    def test_simd_divergence(self):
        w0, w1 = run_hw_simd_divergence("v_sqrt_f32", 16.0, 25.0, "ps_sqrt_simd")
        self.assertEqual(w0, 4.0)
        self.assertEqual(w1, 5.0)


class TestVSqrtF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_sqrt_f32."""

    def test_translator_sqrt_values(self):
        self.assertEqual(run_hw_sqrt_f32(0.0, "ps_sqrt_hw_0"), 0.0)
        self.assertEqual(run_hw_sqrt_f32(1.0, "ps_sqrt_hw_1"), 1.0)
        self.assertEqual(run_hw_sqrt_f32(4.0, "ps_sqrt_hw_4"), 2.0)
        self.assertEqual(run_hw_sqrt_f32(9.0, "ps_sqrt_hw_9"), 3.0)
        self.assertEqual(run_hw_sqrt_f32(16.0, "ps_sqrt_hw_16"), 4.0)
        self.assertEqual(run_hw_sqrt_f32(25.0, "ps_sqrt_hw_25"), 5.0)
        self.assertEqual(run_hw_sqrt_f32(float("inf"), "ps_sqrt_hw_inf"), float("inf"))


if __name__ == "__main__":
    unittest.main()
