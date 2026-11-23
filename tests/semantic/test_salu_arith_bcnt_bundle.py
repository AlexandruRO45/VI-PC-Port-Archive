"""Comprehensive translator-backed regression bundle for RDNA2 SALU Integer Arithmetic & Population Count.

Subsystem Feature Bundle 29 covering:
1. s_abs_i32       (Scalar signed 32-bit integer absolute value, SCC condition code)
2. s_absdiff_i32   (Scalar signed 32-bit integer absolute difference, SCC condition code)
3. s_addk_i32      (Scalar 32-bit addition with sign-extended 16-bit constant, carry tracking)
4. s_bcnt0_i32_b32 (Scalar 32-bit zero-bit population count)
5. s_bcnt0_i32_b64 (Scalar 64-bit zero-bit population count across SGPR pairs)
6. s_bcnt1_i32_b32 (Scalar 32-bit one-bit population count)
7. s_bcnt1_i32_b64 (Scalar 64-bit one-bit population count across SGPR pairs)

All tests in this suite compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.
"""
from __future__ import annotations

import os
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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def run_salu_bundle_ps(lines: list[str], name: str = "ps_salu_bundle") -> list[int]:
    all_lines = list(lines) + [
        "s_cselect_b32 s15, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s15",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


# ==============================================================================
# 1. s_abs_i32: Scalar Signed 32-bit Absolute Value
# ==============================================================================

class TestSAbsI32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_abs_i32: reference mathematical execution validation."""

    def test_s_abs_i32_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s1, 0xFFFFFFFD",
            "s_abs_i32 s0, s1",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_abs_i32")
        self.assertEqual(words[0], 0x3)
        self.assertEqual(words[1], 1)


class TestSAbsI32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_abs_i32: AST and deterministic scalar interpreter execution."""

    def test_s_abs_i32_ast_and_execution(self):
        ast_prog = [
            "s_abs_i32 s0, s2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_abs_i32", ast_prog)
        for needle in ["SAbs", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s1, 0xFFFFFFFD",
            "s_abs_i32 s0, s1",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_abs_i32")
        self.assertEqual(words[0], 0x3)
        self.assertEqual(words[1], 1)


class TestSAbsI32SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_abs_i32: boundary edge cases and status flag tracking."""

    def test_s_abs_i32_edge_cases(self):
        cases = [
            (["s_mov_b32 s1, 0", "s_abs_i32 s0, s1"], 0, 0),
            (["s_mov_b32 s1, 1", "s_abs_i32 s0, s1"], 1, 1),
            (["s_mov_b32 s1, 0xFFFFFFFF", "s_abs_i32 s0, s1"], 1, 1),
            (["s_mov_b32 s1, 0x7FFFFFFF", "s_abs_i32 s0, s1"], 2147483647, 1),
            (["s_mov_b32 s1, 0x80000000", "s_abs_i32 s0, s1"], 2147483648, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_abs_i32_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSAbsI32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_abs_i32: translator integration and negative controls."""

    def test_s_abs_i32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_abs_i32", ["s_abs_i32", "s_endpgm"])


# ==============================================================================
# 2. s_absdiff_i32: Scalar Signed 32-bit Absolute Difference
# ==============================================================================

class TestSAbsdiffI32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_absdiff_i32: reference mathematical execution validation."""

    def test_s_absdiff_i32_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s1, 5",
            "s_mov_b32 s2, 9",
            "s_absdiff_i32 s0, s1, s2",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_absdiff_i32")
        self.assertEqual(words[0], 0x4)
        self.assertEqual(words[1], 1)


class TestSAbsdiffI32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_absdiff_i32: AST and deterministic scalar interpreter execution."""

    def test_s_absdiff_i32_ast_and_execution(self):
        ast_prog = [
            "s_absdiff_i32 s0, s2, s3",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_absdiff_i32", ast_prog)
        for needle in ["SAbs", "OpISub", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s1, 5",
            "s_mov_b32 s2, 9",
            "s_absdiff_i32 s0, s1, s2",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_absdiff_i32")
        self.assertEqual(words[0], 0x4)
        self.assertEqual(words[1], 1)


class TestSAbsdiffI32SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_absdiff_i32: boundary edge cases and status flag tracking."""

    def test_s_absdiff_i32_edge_cases(self):
        cases = [
            (["s_mov_b32 s1, 5", "s_mov_b32 s2, 5", "s_absdiff_i32 s0, s1, s2"], 0, 0),
            (["s_mov_b32 s1, 5", "s_mov_b32 s2, 9", "s_absdiff_i32 s0, s1, s2"], 4, 1),
            (["s_mov_b32 s1, 9", "s_mov_b32 s2, 5", "s_absdiff_i32 s0, s1, s2"], 4, 1),
            (["s_mov_b32 s1, 0", "s_mov_b32 s2, 0xFFFFFFFF", "s_absdiff_i32 s0, s1, s2"], 1, 1),
            (["s_mov_b32 s1, 0x7FFFFFFF", "s_mov_b32 s2, 0", "s_absdiff_i32 s0, s1, s2"], 2147483647, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_absdiff_i32_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSAbsdiffI32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_absdiff_i32: translator integration and negative controls."""

    def test_s_absdiff_i32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_absdiff_i32", ["s_absdiff_i32", "s_endpgm"])


# ==============================================================================
# 3. s_addk_i32: Scalar 32-bit Add with 16-bit Signed Immediate
# ==============================================================================

class TestSAddkI32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_addk_i32: reference mathematical execution validation."""

    def test_s_addk_i32_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s0, 10",
            "s_addk_i32 s0, 5",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_addk_i32")
        self.assertEqual(words[0], 0xf)
        self.assertEqual(words[1], 0)


class TestSAddkI32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_addk_i32: AST and deterministic scalar interpreter execution."""

    def test_s_addk_i32_ast_and_execution(self):
        ast_prog = [
            "s_mov_b32 s0, 10",
            "s_addk_i32 s0, 5",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_addk_i32", ast_prog)
        for needle in ["OpIAdd", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s0, 10",
            "s_addk_i32 s0, 5",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_addk_i32")
        self.assertEqual(words[0], 0xf)
        self.assertEqual(words[1], 0)


class TestSAddkI32SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_addk_i32: boundary edge cases and status flag tracking."""

    def test_s_addk_i32_edge_cases(self):
        cases = [
            (["s_mov_b32 s0, 10", "s_addk_i32 s0, 5"], 15, 0),
            (["s_mov_b32 s0, 10", "s_addk_i32 s0, -3"], 7, 0),
            (["s_mov_b32 s0, 0", "s_addk_i32 s0, 0"], 0, 0),
            (["s_mov_b32 s0, 0xFFFFFFFF", "s_addk_i32 s0, 1"], 0, 0),
            (["s_mov_b32 s0, 0x7FFFFFFF", "s_addk_i32 s0, 1"], 2147483648, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_addk_i32_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSAddkI32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_addk_i32: translator integration and negative controls."""

    def test_s_addk_i32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_addk_i32", ["s_addk_i32", "s_endpgm"])


# ==============================================================================
# 4. s_bcnt0_i32_b32: Scalar 32-bit Count Zero Bits
# ==============================================================================

class TestSBcnt0I32B32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_bcnt0_i32_b32: reference mathematical execution validation."""

    def test_s_bcnt0_i32_b32_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s1, 0xFFFFFFFE",
            "s_bcnt0_i32_b32 s0, s1",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_bcnt0_i32_b32")
        self.assertEqual(words[0], 0x1)
        self.assertEqual(words[1], 1)


class TestSBcnt0I32B32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_bcnt0_i32_b32: AST and deterministic scalar interpreter execution."""

    def test_s_bcnt0_i32_b32_ast_and_execution(self):
        ast_prog = [
            "s_bcnt0_i32_b32 s0, s2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_bcnt0_i32_b32", ast_prog)
        for needle in ["OpNot", "OpBitCount", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s1, 0xFFFFFFFE",
            "s_bcnt0_i32_b32 s0, s1",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_bcnt0_i32_b32")
        self.assertEqual(words[0], 0x1)
        self.assertEqual(words[1], 1)


class TestSBcnt0I32B32SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_bcnt0_i32_b32: boundary edge cases and status flag tracking."""

    def test_s_bcnt0_i32_b32_edge_cases(self):
        cases = [
            (["s_mov_b32 s1, 0x00000000", "s_bcnt0_i32_b32 s0, s1"], 32, 1),
            (["s_mov_b32 s1, 0xFFFFFFFF", "s_bcnt0_i32_b32 s0, s1"], 0, 0),
            (["s_mov_b32 s1, 0xFFFFFFFE", "s_bcnt0_i32_b32 s0, s1"], 1, 1),
            (["s_mov_b32 s1, 0x55555555", "s_bcnt0_i32_b32 s0, s1"], 16, 1),
            (["s_mov_b32 s1, 0xAAAAAAAA", "s_bcnt0_i32_b32 s0, s1"], 16, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_bcnt0_i32_b32_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSBcnt0I32B32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_bcnt0_i32_b32: translator integration and negative controls."""

    def test_s_bcnt0_i32_b32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_bcnt0_i32_b32", ["s_bcnt0_i32_b32", "s_endpgm"])


# ==============================================================================
# 5. s_bcnt0_i32_b64: Scalar 64-bit Count Zero Bits
# ==============================================================================

class TestSBcnt0I32B64ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_bcnt0_i32_b64: reference mathematical execution validation."""

    def test_s_bcnt0_i32_b64_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s2, 0xFFFFFFFE",
            "s_mov_b32 s3, 0xFFFFFFFF",
            "s_bcnt0_i32_b64 s0, s[2:3]",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_bcnt0_i32_b64")
        self.assertEqual(words[0], 0x1)
        self.assertEqual(words[1], 1)


class TestSBcnt0I32B64ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_bcnt0_i32_b64: AST and deterministic scalar interpreter execution."""

    def test_s_bcnt0_i32_b64_ast_and_execution(self):
        ast_prog = [
            "s_bcnt0_i32_b64 s0, s[2:3]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_bcnt0_i32_b64", ast_prog)
        for needle in ["OpNot", "OpBitCount", "OpIAdd", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s2, 0xFFFFFFFE",
            "s_mov_b32 s3, 0xFFFFFFFF",
            "s_bcnt0_i32_b64 s0, s[2:3]",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_bcnt0_i32_b64")
        self.assertEqual(words[0], 0x1)
        self.assertEqual(words[1], 1)


class TestSBcnt0I32B64SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_bcnt0_i32_b64: boundary edge cases and status flag tracking."""

    def test_s_bcnt0_i32_b64_edge_cases(self):
        cases = [
            (["s_mov_b32 s2, 0", "s_mov_b32 s3, 0", "s_bcnt0_i32_b64 s0, s[2:3]"], 64, 1),
            (["s_mov_b32 s2, 0xFFFFFFFF", "s_mov_b32 s3, 0xFFFFFFFF", "s_bcnt0_i32_b64 s0, s[2:3]"], 0, 0),
            (["s_mov_b32 s2, 0xFFFFFFFE", "s_mov_b32 s3, 0xFFFFFFFF", "s_bcnt0_i32_b64 s0, s[2:3]"], 1, 1),
            (["s_mov_b32 s2, 0x55555555", "s_mov_b32 s3, 0x55555555", "s_bcnt0_i32_b64 s0, s[2:3]"], 32, 1),
            (["s_mov_b32 s2, 0", "s_mov_b32 s3, 0xFFFFFFFF", "s_bcnt0_i32_b64 s0, s[2:3]"], 32, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_bcnt0_i32_b64_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSBcnt0I32B64TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_bcnt0_i32_b64: translator integration and negative controls."""

    def test_s_bcnt0_i32_b64_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_bcnt0_i32_b64", ["s_bcnt0_i32_b64", "s_endpgm"])


# ==============================================================================
# 6. s_bcnt1_i32_b32: Scalar 32-bit Count One Bits
# ==============================================================================

class TestSBcnt1I32B32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_bcnt1_i32_b32: reference mathematical execution validation."""

    def test_s_bcnt1_i32_b32_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s1, 7",
            "s_bcnt1_i32_b32 s0, s1",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_bcnt1_i32_b32")
        self.assertEqual(words[0], 0x3)
        self.assertEqual(words[1], 1)


class TestSBcnt1I32B32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_bcnt1_i32_b32: AST and deterministic scalar interpreter execution."""

    def test_s_bcnt1_i32_b32_ast_and_execution(self):
        ast_prog = [
            "s_bcnt1_i32_b32 s0, s2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_bcnt1_i32_b32", ast_prog)
        for needle in ["OpBitCount", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s1, 7",
            "s_bcnt1_i32_b32 s0, s1",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_bcnt1_i32_b32")
        self.assertEqual(words[0], 0x3)
        self.assertEqual(words[1], 1)


class TestSBcnt1I32B32SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_bcnt1_i32_b32: boundary edge cases and status flag tracking."""

    def test_s_bcnt1_i32_b32_edge_cases(self):
        cases = [
            (["s_mov_b32 s1, 0x00000000", "s_bcnt1_i32_b32 s0, s1"], 0, 0),
            (["s_mov_b32 s1, 0xFFFFFFFF", "s_bcnt1_i32_b32 s0, s1"], 32, 1),
            (["s_mov_b32 s1, 7", "s_bcnt1_i32_b32 s0, s1"], 3, 1),
            (["s_mov_b32 s1, 0x55555555", "s_bcnt1_i32_b32 s0, s1"], 16, 1),
            (["s_mov_b32 s1, 0xAAAAAAAA", "s_bcnt1_i32_b32 s0, s1"], 16, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_bcnt1_i32_b32_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSBcnt1I32B32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_bcnt1_i32_b32: translator integration and negative controls."""

    def test_s_bcnt1_i32_b32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_bcnt1_i32_b32", ["s_bcnt1_i32_b32", "s_endpgm"])


# ==============================================================================
# 7. s_bcnt1_i32_b64: Scalar 64-bit Count One Bits
# ==============================================================================

class TestSBcnt1I32B64ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_bcnt1_i32_b64: reference mathematical execution validation."""

    def test_s_bcnt1_i32_b64_oracle_identity(self):
        exec_prog = [
            "s_mov_b32 s2, 7",
            "s_mov_b32 s3, 1",
            "s_bcnt1_i32_b64 s0, s[2:3]",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_oracle_s_bcnt1_i32_b64")
        self.assertEqual(words[0], 0x4)
        self.assertEqual(words[1], 1)


class TestSBcnt1I32B64ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_bcnt1_i32_b64: AST and deterministic scalar interpreter execution."""

    def test_s_bcnt1_i32_b64_ast_and_execution(self):
        ast_prog = [
            "s_bcnt1_i32_b64 s0, s[2:3]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_bcnt1_i32_b64", ast_prog)
        for needle in ["OpBitCount", "OpIAdd", "%scc"]:
            self.assertIn(needle, spv)

        exec_prog = [
            "s_mov_b32 s2, 7",
            "s_mov_b32 s3, 1",
            "s_bcnt1_i32_b64 s0, s[2:3]",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_bcnt1_i32_b64")
        self.assertEqual(words[0], 0x4)
        self.assertEqual(words[1], 1)


class TestSBcnt1I32B64SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_bcnt1_i32_b64: boundary edge cases and status flag tracking."""

    def test_s_bcnt1_i32_b64_edge_cases(self):
        cases = [
            (["s_mov_b32 s2, 0", "s_mov_b32 s3, 0", "s_bcnt1_i32_b64 s0, s[2:3]"], 0, 0),
            (["s_mov_b32 s2, 0xFFFFFFFF", "s_mov_b32 s3, 0xFFFFFFFF", "s_bcnt1_i32_b64 s0, s[2:3]"], 64, 1),
            (["s_mov_b32 s2, 7", "s_mov_b32 s3, 1", "s_bcnt1_i32_b64 s0, s[2:3]"], 4, 1),
            (["s_mov_b32 s2, 0x55555555", "s_mov_b32 s3, 0x55555555", "s_bcnt1_i32_b64 s0, s[2:3]"], 32, 1),
            (["s_mov_b32 s2, 0xFFFFFFFF", "s_mov_b32 s3, 0", "s_bcnt1_i32_b64 s0, s[2:3]"], 32, 1),
        ]
        for idx, (lines, expected_val, expected_scc) in enumerate(cases):
            with self.subTest(case_idx=idx):
                words = run_salu_bundle_ps(lines, f"ps_edge_s_bcnt1_i32_b64_{idx}")
                self.assertEqual(words[0], expected_val)
                self.assertEqual(words[1], expected_scc)


class TestSBcnt1I32B64TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_bcnt1_i32_b64: translator integration and negative controls."""

    def test_s_bcnt1_i32_b64_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_bcnt1_i32_b64", ["s_bcnt1_i32_b64", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
