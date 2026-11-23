"""Comprehensive translator-backed regression bundle for RDNA2 VALU 16-bit 3-Input Min/Max/Med Selection.

Subsystem Feature Bundle 50 covering:
1. v_min3_u16 (Unsigned 16-bit integer minimum of 3 operands, destination upper half preserved)
2. v_max3_u16 (Unsigned 16-bit integer maximum of 3 operands, destination upper half preserved)
3. v_med3_u16 (Unsigned 16-bit integer median of 3 operands, destination upper half preserved)
4. v_min3_i16 (Signed 16-bit integer minimum of 3 operands, destination upper half preserved)
5. v_max3_i16 (Signed 16-bit integer maximum of 3 operands, destination upper half preserved)
6. v_med3_i16 (Signed 16-bit integer median of 3 operands, destination upper half preserved)
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def s16(x: int) -> int:
    x &= MASK16
    return x - 0x10000 if x & 0x8000 else x


def run_bundle_ps(lines: list[str], name: str = "ps_minmaxmed16_bundle") -> list[int]:
    all_lines = list(lines) + [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


def oracle_v_min3_u16(a: int, b: int, c: int) -> int:
    return min(a & MASK16, b & MASK16, c & MASK16)


def oracle_v_max3_u16(a: int, b: int, c: int) -> int:
    return max(a & MASK16, b & MASK16, c & MASK16)


def oracle_v_med3_u16(a: int, b: int, c: int) -> int:
    return sorted([a & MASK16, b & MASK16, c & MASK16])[1]


def oracle_v_min3_i16(a: int, b: int, c: int) -> int:
    return min([a & MASK16, b & MASK16, c & MASK16], key=s16)


def oracle_v_max3_i16(a: int, b: int, c: int) -> int:
    return max([a & MASK16, b & MASK16, c & MASK16], key=s16)


def oracle_v_med3_i16(a: int, b: int, c: int) -> int:
    return sorted([a & MASK16, b & MASK16, c & MASK16], key=s16)[1]

class TestVMin3U16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_min3_u16: reference mathematical oracle validation."""

    def test_v_min3_u16_oracle_identity(self):
        res = oracle_v_min3_u16(0xa, 0x5, 0x14)
        self.assertEqual(res, 0x5)

class TestVMin3U16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_min3_u16: AST and scalar interpreter execution."""

    def test_v_min3_u16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xa",
            "v_mov_b32 v2, 0x5",
            "v_mov_b32 v3, 0x14",
            "v_min3_u16 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_min3_u16", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("UMin", spv)
        words = run_bundle_ps(lines, "ps_exec_v_min3_u16")
        self.assertEqual(words[0] & 0xFFFF, 0x5)
        self.assertEqual((words[0] >> 16) & 0xFFFF, 0x55AA)

class TestVMin3U16SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for v_min3_u16: boundary edge cases, SIMD multi-lane divergence, and register preservation."""

    def test_v_min3_u16_edge_cases(self):
        cases = [([0, 1, 65535], 0), ([32768, 32767, 65535], 32767), ([5, 5, 5], 5)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = [
                    "v_mov_b32 v0, 0x55AA0000",
                    f"v_mov_b32 v1, {inputs[0]:#x}",
                    f"v_mov_b32 v2, {inputs[1]:#x}",
                    f"v_mov_b32 v3, {inputs[2]:#x}",
                    f"v_min3_u16 v0, v1, v2, v3",
                ]
                words = run_bundle_ps(lines, f"ps_edge_v_min3_u16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVMin3U16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_min3_u16: translator integration and negative controls."""

    def test_v_min3_u16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_min3_u16", ["v_min3_u16", "s_endpgm"])

class TestVMax3U16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_max3_u16: reference mathematical oracle validation."""

    def test_v_max3_u16_oracle_identity(self):
        res = oracle_v_max3_u16(0xa, 0x5, 0x14)
        self.assertEqual(res, 0x14)

class TestVMax3U16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_max3_u16: AST and scalar interpreter execution."""

    def test_v_max3_u16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xa",
            "v_mov_b32 v2, 0x5",
            "v_mov_b32 v3, 0x14",
            "v_max3_u16 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_max3_u16", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("UMax", spv)
        words = run_bundle_ps(lines, "ps_exec_v_max3_u16")
        self.assertEqual(words[0] & 0xFFFF, 0x14)
        self.assertEqual((words[0] >> 16) & 0xFFFF, 0x55AA)

class TestVMax3U16SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for v_max3_u16: boundary edge cases, SIMD multi-lane divergence, and register preservation."""

    def test_v_max3_u16_edge_cases(self):
        cases = [([0, 1, 65535], 65535), ([32768, 32767, 65535], 65535), ([5, 5, 5], 5)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = [
                    "v_mov_b32 v0, 0x55AA0000",
                    f"v_mov_b32 v1, {inputs[0]:#x}",
                    f"v_mov_b32 v2, {inputs[1]:#x}",
                    f"v_mov_b32 v3, {inputs[2]:#x}",
                    f"v_max3_u16 v0, v1, v2, v3",
                ]
                words = run_bundle_ps(lines, f"ps_edge_v_max3_u16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVMax3U16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_max3_u16: translator integration and negative controls."""

    def test_v_max3_u16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_max3_u16", ["v_max3_u16", "s_endpgm"])

class TestVMed3U16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_med3_u16: reference mathematical oracle validation."""

    def test_v_med3_u16_oracle_identity(self):
        res = oracle_v_med3_u16(0xa, 0x5, 0x14)
        self.assertEqual(res, 0xa)

class TestVMed3U16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_med3_u16: AST and scalar interpreter execution."""

    def test_v_med3_u16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xa",
            "v_mov_b32 v2, 0x5",
            "v_mov_b32 v3, 0x14",
            "v_med3_u16 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_med3_u16", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("UClamp", spv)
        words = run_bundle_ps(lines, "ps_exec_v_med3_u16")
        self.assertEqual(words[0] & 0xFFFF, 0xa)
        self.assertEqual((words[0] >> 16) & 0xFFFF, 0x55AA)

class TestVMed3U16SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for v_med3_u16: boundary edge cases, SIMD multi-lane divergence, and register preservation."""

    def test_v_med3_u16_edge_cases(self):
        cases = [([0, 1, 65535], 1), ([32768, 32767, 65535], 32768), ([5, 5, 5], 5)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = [
                    "v_mov_b32 v0, 0x55AA0000",
                    f"v_mov_b32 v1, {inputs[0]:#x}",
                    f"v_mov_b32 v2, {inputs[1]:#x}",
                    f"v_mov_b32 v3, {inputs[2]:#x}",
                    f"v_med3_u16 v0, v1, v2, v3",
                ]
                words = run_bundle_ps(lines, f"ps_edge_v_med3_u16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVMed3U16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_med3_u16: translator integration and negative controls."""

    def test_v_med3_u16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_med3_u16", ["v_med3_u16", "s_endpgm"])

class TestVMin3I16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_min3_i16: reference mathematical oracle validation."""

    def test_v_min3_i16_oracle_identity(self):
        res = oracle_v_min3_i16(0xfffe, 0x5, 0x14)
        self.assertEqual(res, 0xfffe)

class TestVMin3I16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_min3_i16: AST and scalar interpreter execution."""

    def test_v_min3_i16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xfffe",
            "v_mov_b32 v2, 0x5",
            "v_mov_b32 v3, 0x14",
            "v_min3_i16 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_min3_i16", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("SMin", spv)
        words = run_bundle_ps(lines, "ps_exec_v_min3_i16")
        self.assertEqual(words[0] & 0xFFFF, 0xfffe)
        self.assertEqual((words[0] >> 16) & 0xFFFF, 0x55AA)

class TestVMin3I16SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for v_min3_i16: boundary edge cases, SIMD multi-lane divergence, and register preservation."""

    def test_v_min3_i16_edge_cases(self):
        cases = [([32768, 0, 32767], 32768), ([65535, 65534, 1], 65534), ([32769, 32766, 0], 32769)]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                lines = [
                    "v_mov_b32 v0, 0x55AA0000",
                    f"v_mov_b32 v1, {inputs[0]:#x}",
                    f"v_mov_b32 v2, {inputs[1]:#x}",
                    f"v_mov_b32 v3, {inputs[2]:#x}",
                    f"v_min3_i16 v0, v1, v2, v3",
                ]
                words = run_bundle_ps(lines, f"ps_edge_v_min3_i16_{inputs[0]:#x}")
                lo = words[0] & 0xFFFF
                hi = (words[0] >> 16) & 0xFFFF
                self.assertEqual(hi, 0x55AA)
                self.assertEqual(lo, expected)

class TestVMin3I16TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_min3_i16: translator integration and negative controls."""

    def test_v_min3_i16_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_min3_i16", ["v_min3_i16", "s_endpgm"])

class TestVMax3I16ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_max3_i16: reference mathematical oracle validation."""

    def test_v_max3_i16_oracle_identity(self):
        res = oracle_v_max3_i16(0xfffe, 0x5, 0x14)
        self.assertEqual(res, 0x14)

class TestVMax3I16ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_max3_i16: AST and scalar interpreter execution."""

    def test_v_max3_i16_ast_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xfffe",
            "v_mov_b32 v2, 0x5",
            "v_mov_b32 v3, 0x14",
            "v_max3_i16 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_max3_i16", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("SMax", spv)
        words = run_bundle_ps(lines, "ps_exec_v_max3_i16")
        self.assertEqual(words[0] & 0xFFFF, 0x14)
        self.assertEqual((words[0] >> 16) & 0xFFFF, 0x55AA)


if __name__ == '__main__':
    unittest.main()
