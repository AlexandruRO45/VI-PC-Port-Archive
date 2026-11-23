"""Bundle 18: VALU Float16 Conversions & Normalization Suite.

Tests for 6 VALU Float16 conversion instructions:
- v_cvt_f16_f32: 32-bit float to 16-bit half float conversion with rounding
- v_cvt_f32_f16: 16-bit half float to 32-bit float exact widening conversion
- v_cvt_f16_u16: 16-bit unsigned integer to 16-bit half float conversion
- v_cvt_f16_i16: 16-bit signed integer to 16-bit half float conversion
- v_cvt_u16_f16: 16-bit half float to 16-bit unsigned integer with truncation/saturation
- v_cvt_i16_f16: 16-bit half float to 16-bit signed integer with truncation/saturation

All test methods compile assembly listings directly through tools/rdna2spirv.py
via support.translate_lines and execute in spirv_mini_interp via support.mrt0_words.
Standard library Python only; zero third-party dependencies; zero game data.
"""
from __future__ import annotations

import dataclasses
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple

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

MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF


def float_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f16_bits_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(val: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return 0xFC00 if val < 0 else 0x7C00


def execute_cvt_op(op: str, input_val: int, name: str = "ps_cvt") -> int:
    """Compile and execute a synthetic pixel shader for conversion op, returning mrt0.x."""
    lines = [
        f"v_mov_b32_e32 v1, {input_val & MASK32:#x}",
        f"{op}_e32 v0, v1",
        "exp mrt0, v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


@dataclasses.dataclass(frozen=True)
class CvtAstNode:
    opcode: str
    dst: str
    src: str

    def emit_asm(self) -> str:
        return f"{self.opcode}_e32 {self.dst}, {self.src}"


# Mathematical reference oracles
def ref_v_cvt_f16_f32(f32_val: float) -> int:
    return float_to_f16_bits(f32_val)


def ref_v_cvt_f32_f16(f16_bits: int) -> float:
    return f16_bits_to_float(f16_bits & MASK16)


def ref_v_cvt_f16_u16(u16_val: int) -> int:
    return float_to_f16_bits(float(u16_val & MASK16))


def ref_v_cvt_f16_i16(i16_val: int) -> int:
    i16_val &= MASK16
    signed_val = i16_val - (1 << 16) if (i16_val & 0x8000) else i16_val
    return float_to_f16_bits(float(signed_val))


def ref_v_cvt_u16_f16(f16_bits: int) -> int:
    val = f16_bits_to_float(f16_bits & MASK16)
    if math.isnan(val) or val <= 0.0:
        return 0
    if math.isinf(val) or val >= 65535.0:
        return 65535
    return math.trunc(val)


def ref_v_cvt_i16_f16(f16_bits: int) -> int:
    val = f16_bits_to_float(f16_bits & MASK16)
    if math.isnan(val):
        return 0
    if math.isinf(val):
        return 0x7FFF if val > 0 else 0x8000
    tr = math.trunc(val)
    if tr < -32768:
        return 0x8000
    if tr > 32767:
        return 0x7FFF
    return tr & MASK16


# =============================================================================
# 1. v_cvt_f16_f32
# =============================================================================

class TestVCvtF16F32Oracle(unittest.TestCase):
    """Reference oracle tests for v_cvt_f16_f32 backed by translator execution."""

    def test_basic_values(self) -> None:
        cases = [
            (0.0, 0x0000),
            (1.0, 0x3C00),
            (-1.0, 0xBC00),
            (2.0, 0x4000),
            (0.5, 0x3800),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                res = execute_cvt_op("v_cvt_f16_f32", float_to_bits(val), "ps_cvt_f16_f32_basic")
                self.assertEqual(res & MASK16, expected)
                self.assertEqual(res & MASK16, ref_v_cvt_f16_f32(val))

    def test_infinities_and_zeros(self) -> None:
        cases = [
            (float("inf"), 0x7C00),
            (float("-inf"), 0xFC00),
            (-0.0, 0x8000),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                res = execute_cvt_op("v_cvt_f16_f32", float_to_bits(val), "ps_cvt_f16_f32_inf")
                self.assertEqual(res & MASK16, expected)


class TestVCvtF16F32AST(unittest.TestCase):
    """AST modeling and translation verification for v_cvt_f16_f32."""

    def test_ast_evaluation(self) -> None:
        node = CvtAstNode("v_cvt_f16_f32", "v0", "v1")
        lines = [
            "v_mov_b32_e32 v1, 0x3fc00000",  # 1.5f
            node.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_f16_f32_ast", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0] & MASK16, 0x3E00)


class TestVCvtF16F32SIMD(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_cvt_f16_f32."""

    def test_simd_lane_divergence(self) -> None:
        test_inputs = [0.25, 0.75, 1.25, 3.5, 10.0, 64.0, 128.0, 256.0]
        for idx, val in enumerate(test_inputs):
            with self.subTest(val=val):
                res = execute_cvt_op("v_cvt_f16_f32", float_to_bits(val), f"ps_cvt_f16_f32_div_{idx}")
                expected = ref_v_cvt_f16_f32(val)
                self.assertEqual(res & MASK16, expected)

    def test_ieee754_boundaries(self) -> None:
        cases = [
            (65504.0, 0x7BFF),  # Max normal f16
            (2.0 ** -24, 0x0001),  # Min subnormal f16
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                res = execute_cvt_op("v_cvt_f16_f32", float_to_bits(val), "ps_cvt_f16_f32_bnd")
                self.assertEqual(res & MASK16, expected)


class TestVCvtF16F32Translator(unittest.TestCase):
    """Translator execution verification for v_cvt_f16_f32."""

    def test_translator_cvt_f16_f32_execution(self) -> None:
        res = execute_cvt_op("v_cvt_f16_f32", 0x3F800000, "ps_cvt_f16_f32_exec")
        self.assertEqual(res & MASK16, 0x3C00)


# =============================================================================
# 2. v_cvt_f32_f16
# =============================================================================

class TestVCvtF32F16Oracle(unittest.TestCase):
    """Reference oracle tests for v_cvt_f32_f16 backed by translator execution."""

    def test_basic_values(self) -> None:
        cases = [
            (0x0000, 0.0),
            (0x3C00, 1.0),
            (0xBC00, -1.0),
            (0x4000, 2.0),
            (0x3800, 0.5),
        ]
        for f16, expected in cases:
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_f32_f16", f16, "ps_cvt_f32_f16_basic")
                self.assertEqual(bits_to_float(res), expected)

    def test_infinities_and_zeros(self) -> None:
        cases = [
            (0x7C00, 0x7F800000),  # +inf
            (0xFC00, 0xFF800000),  # -inf
            (0x8000, 0x80000000),  # -0.0
        ]
        for f16, expected_bits in cases:
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_f32_f16", f16, "ps_cvt_f32_f16_inf")
                self.assertEqual(res, expected_bits)


class TestVCvtF32F16AST(unittest.TestCase):
    """AST modeling and translation verification for v_cvt_f32_f16."""

    def test_ast_evaluation(self) -> None:
        node = CvtAstNode("v_cvt_f32_f16", "v0", "v1")
        lines = [
            "v_mov_b32_e32 v1, 0x4200",  # 3.0h
            node.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_f32_f16_ast", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(bits_to_float(out[0]), 3.0)


class TestVCvtF32F16SIMD(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_cvt_f32_f16."""

    def test_simd_lane_divergence(self) -> None:
        f16_vals = [0x3C00, 0x4000, 0x4200, 0x4400, 0x4600, 0x4800]
        for idx, f16 in enumerate(f16_vals):
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_f32_f16", f16, f"ps_cvt_f32_f16_div_{idx}")
                self.assertEqual(bits_to_float(res), ref_v_cvt_f32_f16(f16))

    def test_ieee754_subnormals_widening(self) -> None:
        # Half subnormal 0x0001 (2^-24) widens into normal f32 (0x33800000)
        res = execute_cvt_op("v_cvt_f32_f16", 0x0001, "ps_cvt_f32_f16_subnorm")
        self.assertEqual(res, 0x33800000)


class TestVCvtF32F16Translator(unittest.TestCase):
    """Translator execution verification for v_cvt_f32_f16."""

    def test_translator_cvt_f32_f16_execution(self) -> None:
        res = execute_cvt_op("v_cvt_f32_f16", 0x3C00, "ps_cvt_f32_f16_exec")
        self.assertEqual(res, 0x3F800000)


# =============================================================================
# 3. v_cvt_f16_u16
# =============================================================================

class TestVCvtF16U16Oracle(unittest.TestCase):
    """Reference oracle tests for v_cvt_f16_u16 backed by translator execution."""

    def test_basic_values(self) -> None:
        cases = [
            (0, 0x0000),
            (1, 0x3C00),
            (42, 0x5140),
            (100, 0x5640),
        ]
        for u16, expected in cases:
            with self.subTest(u16=u16):
                res = execute_cvt_op("v_cvt_f16_u16", u16, "ps_cvt_f16_u16_basic")
                self.assertEqual(res & MASK16, expected)
                self.assertEqual(res & MASK16, ref_v_cvt_f16_u16(u16))

    def test_boundary_values(self) -> None:
        # 65504 is the maximum finite half float (0x7BFF)
        res_65504 = execute_cvt_op("v_cvt_f16_u16", 65504, "ps_cvt_f16_u16_max_finite")
        self.assertEqual(res_65504 & MASK16, 0x7BFF)

        # 65535 overflows and rounds to +inf (0x7C00) in nearest-even rounding
        res_65535 = execute_cvt_op("v_cvt_f16_u16", 65535, "ps_cvt_f16_u16_overflow")
        self.assertEqual(res_65535 & MASK16, 0x7C00)


class TestVCvtF16U16AST(unittest.TestCase):
    """AST modeling and translation verification for v_cvt_f16_u16."""

    def test_ast_evaluation(self) -> None:
        node = CvtAstNode("v_cvt_f16_u16", "v0", "v1")
        lines = [
            "v_mov_b32_e32 v1, 2",
            node.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_f16_u16_ast", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0] & MASK16, 0x4000)  # 2.0h


class TestVCvtF16U16SIMD(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_cvt_f16_u16."""

    def test_simd_lane_divergence(self) -> None:
        u16_vals = [5, 10, 25, 50, 250, 1000]
        for idx, u16 in enumerate(u16_vals):
            with self.subTest(u16=u16):
                res = execute_cvt_op("v_cvt_f16_u16", u16, f"ps_cvt_f16_u16_div_{idx}")
                self.assertEqual(res & MASK16, ref_v_cvt_f16_u16(u16))


class TestVCvtF16U16Translator(unittest.TestCase):
    """Translator execution verification for v_cvt_f16_u16."""

    def test_translator_cvt_f16_u16_execution(self) -> None:
        res = execute_cvt_op("v_cvt_f16_u16", 42, "ps_cvt_f16_u16_exec")
        self.assertEqual(res & MASK16, 0x5140)


# =============================================================================
# 4. v_cvt_f16_i16
# =============================================================================

class TestVCvtF16I16Oracle(unittest.TestCase):
    """Reference oracle tests for v_cvt_f16_i16 backed by translator execution."""

    def test_basic_values(self) -> None:
        cases = [
            (0, 0x0000),
            (1, 0x3C00),
            (0xFFFF, 0xBC00),  # -1 -> -1.0h
            (0xFFD6, 0xD140),  # -42 -> -42.0h
        ]
        for i16, expected in cases:
            with self.subTest(i16=hex(i16)):
                res = execute_cvt_op("v_cvt_f16_i16", i16, "ps_cvt_f16_i16_basic")
                self.assertEqual(res & MASK16, expected)
                self.assertEqual(res & MASK16, ref_v_cvt_f16_i16(i16))

    def test_boundary_extremes(self) -> None:
        cases = [
            (0x8000, 0xF800),  # -32768 -> -32768.0h (0xF800)
            (0x7FFF, 0x7800),  # 32767 -> 32768.0h (0x7800)
        ]
        for i16, expected in cases:
            with self.subTest(i16=hex(i16)):
                res = execute_cvt_op("v_cvt_f16_i16", i16, "ps_cvt_f16_i16_bnd")
                self.assertEqual(res & MASK16, expected)


class TestVCvtF16I16AST(unittest.TestCase):
    """AST modeling and translation verification for v_cvt_f16_i16."""

    def test_ast_evaluation(self) -> None:
        node = CvtAstNode("v_cvt_f16_i16", "v0", "v1")
        lines = [
            "v_mov_b32_e32 v1, 0xfffe",  # -2
            node.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_f16_i16_ast", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0] & MASK16, 0xC000)  # -2.0h


class TestVCvtF16I16SIMD(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_cvt_f16_i16."""

    def test_simd_lane_divergence(self) -> None:
        i16_vals = [0xFFF6, 0xFFEC, 0x000A, 0x0014]  # -10, -20, 10, 20
        for idx, i16 in enumerate(i16_vals):
            with self.subTest(i16=hex(i16)):
                res = execute_cvt_op("v_cvt_f16_i16", i16, f"ps_cvt_f16_i16_div_{idx}")
                self.assertEqual(res & MASK16, ref_v_cvt_f16_i16(i16))


class TestVCvtF16I16Translator(unittest.TestCase):
    """Translator execution verification for v_cvt_f16_i16."""

    def test_translator_cvt_f16_i16_execution(self) -> None:
        res = execute_cvt_op("v_cvt_f16_i16", 0xFFD6, "ps_cvt_f16_i16_exec")
        self.assertEqual(res & MASK16, 0xD140)


# =============================================================================
# 5. v_cvt_u16_f16
# =============================================================================

class TestVCvtU16F16Oracle(unittest.TestCase):
    """Reference oracle tests for v_cvt_u16_f16 backed by translator execution."""

    def test_basic_values(self) -> None:
        cases = [
            (0x5140, 42),  # 42.0h -> 42
            (0x3C00, 1),   # 1.0h -> 1
            (0x3E00, 1),   # 1.5h -> 1 (truncated)
            (0x3800, 0),   # 0.5h -> 0
        ]
        for f16, expected in cases:
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_u16_f16", f16, "ps_cvt_u16_f16_basic")
                self.assertEqual(res & MASK16, expected)
                self.assertEqual(res & MASK16, ref_v_cvt_u16_f16(f16))

    def test_saturation_and_nan(self) -> None:
        cases = [
            (0xBC00, 0),      # -1.0h -> 0 (clamped)
            (0x7C00, 65535),  # +inf -> 65535
            (0xFC00, 0),      # -inf -> 0
            (0x7E00, 0),      # NaN -> 0
        ]
        for f16, expected in cases:
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_u16_f16", f16, "ps_cvt_u16_f16_sat")
                self.assertEqual(res & MASK16, expected)


class TestVCvtU16F16AST(unittest.TestCase):
    """AST modeling and translation verification for v_cvt_u16_f16."""

    def test_ast_evaluation(self) -> None:
        node = CvtAstNode("v_cvt_u16_f16", "v0", "v1")
        lines = [
            "v_mov_b32_e32 v1, 0x4880",  # 9.0h
            node.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_u16_f16_ast", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0] & MASK16, 9)


class TestVCvtU16F16SIMD(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_cvt_u16_f16."""

    def test_simd_lane_divergence(self) -> None:
        f16_vals = [0x4000, 0x4200, 0x4500, 0x4700]  # 2.0, 3.0, 5.0, 7.0
        for idx, f16 in enumerate(f16_vals):
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_u16_f16", f16, f"ps_cvt_u16_f16_div_{idx}")
                self.assertEqual(res & MASK16, ref_v_cvt_u16_f16(f16))


class TestVCvtU16F16Translator(unittest.TestCase):
    """Translator execution verification for v_cvt_u16_f16."""

    def test_translator_cvt_u16_f16_execution(self) -> None:
        res = execute_cvt_op("v_cvt_u16_f16", 0x5140, "ps_cvt_u16_f16_exec")
        self.assertEqual(res & MASK16, 42)


# =============================================================================
# 6. v_cvt_i16_f16
# =============================================================================

class TestVCvtI16F16Oracle(unittest.TestCase):
    """Reference oracle tests for v_cvt_i16_f16 backed by translator execution."""

    def test_basic_values(self) -> None:
        cases = [
            (0xD140, 0xFFD6),  # -42.0h -> -42
            (0x5140, 42),      # 42.0h -> 42
            (0xBE00, 0xFFFF),  # -1.5h -> -1
        ]
        for f16, expected in cases:
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_i16_f16", f16, "ps_cvt_i16_f16_basic")
                self.assertEqual(res & MASK16, expected)
                self.assertEqual(res & MASK16, ref_v_cvt_i16_f16(f16))

    def test_saturation_and_nan(self) -> None:
        cases = [
            (0x7C00, 0x7FFF),  # +inf -> 32767
            (0xFC00, 0x8000),  # -inf -> -32768
            (0x7E00, 0x0000),  # NaN -> 0
        ]
        for f16, expected in cases:
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_i16_f16", f16, "ps_cvt_i16_f16_sat")
                self.assertEqual(res & MASK16, expected)


class TestVCvtI16F16AST(unittest.TestCase):
    """AST modeling and translation verification for v_cvt_i16_f16."""

    def test_ast_evaluation(self) -> None:
        node = CvtAstNode("v_cvt_i16_f16", "v0", "v1")
        lines = [
            "v_mov_b32_e32 v1, 0xc000",  # -2.0h
            node.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_i16_f16_ast", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0] & MASK16, 0xFFFE)


class TestVCvtI16F16SIMD(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_cvt_i16_f16."""

    def test_simd_lane_divergence(self) -> None:
        f16_vals = [0xC200, 0xC500, 0x4200, 0x4500]  # -3.0, -5.0, 3.0, 5.0
        for idx, f16 in enumerate(f16_vals):
            with self.subTest(f16=hex(f16)):
                res = execute_cvt_op("v_cvt_i16_f16", f16, f"ps_cvt_i16_f16_div_{idx}")
                self.assertEqual(res & MASK16, ref_v_cvt_i16_f16(f16))


class TestVCvtI16F16Translator(unittest.TestCase):
    """Translator execution verification for v_cvt_i16_f16."""

    def test_translator_cvt_i16_f16_execution(self) -> None:
        res = execute_cvt_op("v_cvt_i16_f16", 0xD140, "ps_cvt_i16_f16_exec")
        self.assertEqual(res & MASK16, 0xFFD6)


# =============================================================================
# 7. Negative Controls and Mutation Detection
# =============================================================================

class TestVCvtF16NegativeControls(unittest.TestCase):
    """Negative controls and mutation detection for f16 conversions."""

    def test_syntax_rejection_insufficient_operands(self) -> None:
        lines = [
            "v_cvt_f16_f32_e32 v0",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_cvt_bad_ops", lines)

    def test_mutation_detection_signed_vs_unsigned_f16_conversion(self) -> None:
        # Negative half float -42.0h (0xD140):
        # v_cvt_i16_f16 gives -42 (0xFFD6)
        # v_cvt_u16_f16 gives 0 (saturated clamp to 0)
        res_i16 = execute_cvt_op("v_cvt_i16_f16", 0xD140, "ps_mut_i16")
        res_u16 = execute_cvt_op("v_cvt_u16_f16", 0xD140, "ps_mut_u16")
        self.assertEqual(res_i16 & MASK16, 0xFFD6)
        self.assertEqual(res_u16 & MASK16, 0)
        self.assertNotEqual(res_i16 & MASK16, res_u16 & MASK16)


if __name__ == "__main__":
    unittest.main()
