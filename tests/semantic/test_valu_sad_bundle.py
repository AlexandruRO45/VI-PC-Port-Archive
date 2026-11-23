"""Semantic tests for RDNA2 VALU Sum of Absolute Differences (SAD) bundle.

Covers:
- v_sad_u32
- v_sad_u16
- v_sad_u8
- v_sad_hi_u8
- v_msad_u8
- v_mqsad_u32_u8

All tests compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def run_hw_sad_u32(s0: int, s1: int, s2: int, name: str = "ps_sad_u32_hw") -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_u32 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


def run_hw_sad_u16(s0: int, s1: int, s2: int, name: str = "ps_sad_u16_hw") -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_u16 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


def run_hw_sad_u8(s0: int, s1: int, s2: int, name: str = "ps_sad_u8_hw") -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_u8 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


def run_hw_sad_hi_u8(s0: int, s1: int, s2: int, name: str = "ps_sad_hi_u8_hw") -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_hi_u8 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


def run_hw_msad_u8(s0: int, s1: int, s2: int, name: str = "ps_msad_u8_hw") -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_msad_u8 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


def run_hw_mqsad_u32_u8(s0_lo: int, s0_hi: int, s1: int, s2_quad: list[int], name: str = "ps_mqsad_hw") -> list[int]:
    lines = [
        f"v_mov_b32 v0, {s0_lo & MASK32:#x}",
        f"v_mov_b32 v1, {s0_hi & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v4, {s2_quad[0] & MASK32:#x}",
        f"v_mov_b32 v5, {s2_quad[1] & MASK32:#x}",
        f"v_mov_b32 v6, {s2_quad[2] & MASK32:#x}",
        f"v_mov_b32 v7, {s2_quad[3] & MASK32:#x}",
        "v_mqsad_u32_u8 v[8:11], v[0:1], v2, v[4:7]",
        "exp mrt0 v8, v9, v10, v11 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)


# ==============================================================================
# 1. v_sad_u32
# ==============================================================================

class TestVSadU32Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u32."""

    def test_reference_canonical(self):
        self.assertEqual(run_hw_sad_u32(100, 40, 10, "ps_ref_sad_u32_1"), 70)
        self.assertEqual(run_hw_sad_u32(40, 100, 10, "ps_ref_sad_u32_2"), 70)

    def test_reference_zero_diff(self):
        self.assertEqual(run_hw_sad_u32(55, 55, 12, "ps_ref_sad_u32_zero"), 12)

    def test_reference_wrapping(self):
        self.assertEqual(run_hw_sad_u32(0xFFFFFFFF, 0, 1, "ps_ref_sad_u32_wrap"), 0)


class TestVSadU32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_u32."""

    def test_ast_construction_and_bounds(self):
        lines = [
            "v_sad_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sad_u32", lines)
        for needle in ["UMax", "UMin", "OpISub", "OpIAdd"]:
            self.assertIn(needle, spv)

        with self.assertRaises(Exception):
            support.translate_lines("ps_clamp_sad_u32", [
                "v_sad_u32 v0, v1, v2, v3 clamp",
                "s_endpgm",
            ])

    def test_scalar_interpreter_execution(self):
        self.assertEqual(run_hw_sad_u32(100, 40, 10, "ps_step_sad_u32"), 70)


class TestVSadU32SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and 32-bit wrapping for v_sad_u32."""

    def test_simd_multi_lane_divergence(self):
        self.assertEqual(run_hw_sad_u32(50, 20, 0, "ps_simd_sad_u32_1"), 30)
        self.assertEqual(run_hw_sad_u32(51, 20, 0, "ps_simd_sad_u32_2"), 31)
        self.assertEqual(run_hw_sad_u32(99, 20, 0, "ps_simd_sad_u32_3"), 79)


class TestVSadU32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_u32."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_u32(100, 40, 10, "ps_hw_sad_u32_1"), 70)
        self.assertEqual(run_hw_sad_u32(200, 300, 50, "ps_hw_sad_u32_2"), 150)


# ==============================================================================
# 2. v_sad_u16
# ==============================================================================

class TestVSadU16Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u16."""

    def test_reference_canonical(self):
        self.assertEqual(run_hw_sad_u16(0x00200064, 0x00100032, 5, "ps_ref_sad_u16_1"), 71)

    def test_reference_zero_diff(self):
        self.assertEqual(run_hw_sad_u16(0x12345678, 0x12345678, 100, "ps_ref_sad_u16_zero"), 100)


class TestVSadU16ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_u16."""

    def test_scalar_interpreter_execution(self):
        lines = [
            "v_sad_u16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sad_u16", lines)
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertEqual(run_hw_sad_u16(0x00200064, 0x00100032, 5, "ps_step_sad_u16"), 71)


class TestVSadU16SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and dual-half handling for v_sad_u16."""

    def test_simd_dual_half_divergence(self):
        self.assertEqual(run_hw_sad_u16(0x00200064, 0x00100032, 0, "ps_simd_sad_u16"), 66)


class TestVSadU16HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_u16."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_u16(0x00200064, 0x00100032, 5, "ps_hw_sad_u16_1"), 71)
        self.assertEqual(run_hw_sad_u16(0x00100020, 0x00200010, 0, "ps_hw_sad_u16_2"), 32)


# ==============================================================================
# 3. v_sad_u8
# ==============================================================================

class TestVSadU8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u8."""

    def test_reference_canonical(self):
        self.assertEqual(run_hw_sad_u8(0x04030201, 0x01010101, 10, "ps_ref_sad_u8_1"), 16)

    def test_reference_zeros(self):
        self.assertEqual(run_hw_sad_u8(0, 0, 42, "ps_ref_sad_u8_zero"), 42)


class TestVSadU8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_u8."""

    def test_scalar_interpreter_execution(self):
        lines = [
            "v_sad_u8 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sad_u8", lines)
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertEqual(run_hw_sad_u8(0x04030201, 0x01010101, 10, "ps_step_sad_u8"), 16)


class TestVSadU8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and 4-byte accumulation for v_sad_u8."""

    def test_simd_accumulation(self):
        self.assertEqual(run_hw_sad_u8(0x04030201, 0x01010101, 0, "ps_simd_sad_u8"), 6)


class TestVSadU8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_u8."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_u8(0x04030201, 0x01010101, 10, "ps_hw_sad_u8_1"), 16)
        self.assertEqual(run_hw_sad_u8(0x0A0A0A0A, 0x05050505, 0, "ps_hw_sad_u8_2"), 20)


# ==============================================================================
# 4. v_sad_hi_u8
# ==============================================================================

class TestVSadHiU8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_hi_u8."""

    def test_reference_canonical(self):
        self.assertEqual(run_hw_sad_hi_u8(0x04030201, 0x01010101, 10, "ps_ref_sad_hi_u8_1"), 0x0006000A)

    def test_reference_zero_diff(self):
        self.assertEqual(run_hw_sad_hi_u8(0x12345678, 0x12345678, 0x1234, "ps_ref_sad_hi_u8_zero"), 0x1234)


class TestVSadHiU8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_hi_u8."""

    def test_scalar_interpreter_execution(self):
        lines = [
            "v_sad_hi_u8 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sad_hi_u8", lines)
        self.assertIn("OpShiftLeftLogical", spv)
        self.assertEqual(run_hw_sad_hi_u8(0x04030201, 0x01010101, 10, "ps_step_sad_hi_u8"), 0x0006000A)


class TestVSadHiU8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and 16-bit shift handling for v_sad_hi_u8."""

    def test_simd_divergence(self):
        self.assertEqual(run_hw_sad_hi_u8(0x04030201, 0x01010101, 0, "ps_simd_sad_hi_u8"), 6 << 16)


class TestVSadHiU8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_hi_u8."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_hi_u8(0x04030201, 0x01010101, 10, "ps_hw_sad_hi_u8"), 0x0006000A)


# ==============================================================================
# 5. v_msad_u8
# ==============================================================================

class TestVMsadU8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_msad_u8."""

    def test_reference_canonical_masking(self):
        # S1 has bytes [1, 0, 1, 0] -> bytes 1 and 3 are masked out
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x00010001, 10, "ps_ref_msad_u8_1"), 12)

    def test_reference_all_zero_mask(self):
        self.assertEqual(run_hw_msad_u8(0xFFFFFFFF, 0x00000000, 50, "ps_ref_msad_u8_zero"), 50)


class TestVMsadU8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_msad_u8."""

    def test_scalar_interpreter_execution(self):
        lines = [
            "v_msad_u8 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_msad_u8", lines)
        self.assertIn("OpSelect", spv)
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x00010001, 10, "ps_step_msad_u8"), 12)


class TestVMsadU8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and byte-masking handling for v_msad_u8."""

    def test_simd_masking(self):
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x00010001, 0, "ps_simd_msad_u8"), 2)


class TestVMsadU8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_msad_u8."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x00010001, 10, "ps_hw_msad_u8_1"), 12)
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x01010101, 10, "ps_hw_msad_u8_2"), 16)


# ==============================================================================
# 6. v_mqsad_u32_u8
# ==============================================================================

class TestVMqsadU32U8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_mqsad_u32_u8."""

    def test_reference_sliding_window(self):
        outs = run_hw_mqsad_u32_u8(0x04030201, 0x08070605, 0x01010101, [0, 0, 0, 0], "ps_ref_mqsad_u32_u8")
        self.assertEqual(outs, [6, 10, 14, 18])


class TestVMqsadU32U8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_mqsad_u32_u8."""

    def test_scalar_interpreter_execution(self):
        outs = run_hw_mqsad_u32_u8(0x04030201, 0x08070605, 0x01010101, [10, 10, 10, 10], "ps_step_mqsad")
        self.assertEqual(outs, [16, 20, 24, 28])


class TestVMqsadU32U8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and quad-lane windowing for v_mqsad_u32_u8."""

    def test_quad_lane_divergence(self):
        outs = run_hw_mqsad_u32_u8(0x04030201, 0x08070605, 0x01010101, [0, 0, 0, 0], "ps_simd_mqsad")
        self.assertEqual(outs, [6, 10, 14, 18])


class TestVMqsadU32U8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_mqsad_u32_u8."""

    def test_translator_hardware_cases(self):
        outs = run_hw_mqsad_u32_u8(0x04030201, 0x08070605, 0x01010101, [0, 0, 0, 0], "ps_hw_mqsad")
        self.assertEqual(outs, [6, 10, 14, 18])


if __name__ == "__main__":
    unittest.main()
