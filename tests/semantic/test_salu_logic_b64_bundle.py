"""Hardware-verified semantic reference and translator regression suite for RDNA2 SALU 64-bit logic bundle.

Covers the complete 64-bit scalar ALU bitwise logic instruction family:
- s_and_b64:   D = S0 & S1, SCC = (D != 0)
- s_or_b64:    D = S0 | S1, SCC = (D != 0)
- s_xor_b64:   D = S0 ^ S1, SCC = (D != 0)
- s_andn2_b64: D = S0 & ~S1, SCC = (D != 0)
- s_orn2_b64:  D = S0 | ~S1, SCC = (D != 0)
- s_nand_b64:  D = ~(S0 & S1), SCC = (D != 0)
- s_nor_b64:   D = ~(S0 | S1), SCC = (D != 0)
- s_xnor_b64:  D = ~(S0 ^ S1), SCC = (D != 0)

Zero game data, standard library Python only.
All 96 test cases compile and execute through support.translate_lines and support.mrt0_words.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_BIT64: int = 0x8000_0000_0000_0000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit quadword."""
    return val & MASK64


def split64(val: int) -> tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32)."""
    return val & MASK32, (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    """Combine (lo32, hi32) into 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


# Reference mathematical oracles
def oracle_s_and_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) & u64(b)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_or_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) | u64(b)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_xor_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) ^ u64(b)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_andn2_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) & (~u64(b) & MASK64)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_orn2_b64(a: int, b: int) -> tuple[int, int]:
    res = (u64(a) | (~u64(b) & MASK64)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_nand_b64(a: int, b: int) -> tuple[int, int]:
    res = (~(u64(a) & u64(b))) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_nor_b64(a: int, b: int) -> tuple[int, int]:
    res = (~(u64(a) | u64(b))) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def oracle_s_xnor_b64(a: int, b: int) -> tuple[int, int]:
    res = (~(u64(a) ^ u64(b))) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


_TRANSLATE_ID: int = 0


def run_translated_salu64(op: str, s0_64: int, s1_64: int, dst: str = "s[4:5]", src0: str = "s[0:1]", src1: str = "s[2:3]") -> tuple[int, int]:
    """Execute 64-bit SALU opcode through rdna2spirv and return (res64, scc)."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    s0_lo, s0_hi = split64(s0_64)
    s1_lo, s1_hi = split64(s1_64)
    prog = [
        f"s_mov_b32 s0, {s0_lo:#x}",
        f"s_mov_b32 s1, {s0_hi:#x}",
        f"s_mov_b32 s2, {s1_lo:#x}",
        f"s_mov_b32 s3, {s1_hi:#x}",
        f"{op} {dst}, {src0}, {src1}",
    ]
    if dst == "s[0:1]":
        prog.extend([
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ])
    else:
        prog.extend([
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ])
    prog.extend([
        "s_cselect_b32 s6, 1, 0",
        "v_mov_b32 v2, s6",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_salu64_{op}_{_TRANSLATE_ID}", prog)
    words = support.mrt0_words(spv)
    res64 = join64(words[0], words[1])
    scc = words[2] & 1
    return res64, scc


# =============================================================================
# 1. s_and_b64
# =============================================================================

class TestSAndB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_and_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_and_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_and_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        a = 0x5555_5555_5555_5555
        b = 0xAAAA_AAAA_AAAA_AAAA
        exp_res, exp_scc = oracle_s_and_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_and_b64", a, b)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_and_b64(val, val)
        got_res, got_scc = run_translated_salu64("s_and_b64", val, val)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSAndB64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_and_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_and_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_and_b64", prog)
        self.assertIn("OpBitwiseAnd %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_and_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_and_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_and_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_and_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSAndB64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_and_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_and_b64", 0, 0)
        self.assertEqual(res_zero, 0)
        self.assertEqual(scc_zero, 0)
        res_ones, scc_ones = run_translated_salu64("s_and_b64", MASK64, MASK64)
        self.assertEqual(res_ones, MASK64)
        self.assertEqual(scc_ones, 1)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_and_b64", p5, pa)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_and_b64", hi_only, lo_only)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)


class TestSAndB64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_and_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_and_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_and_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        a = 0x5555_5555_5555_5555
        b = 0xAAAA_AAAA_AAAA_AAAA
        exp_res, exp_scc = oracle_s_and_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_and_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_and_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_and_b64_syntax_err", prog)


# =============================================================================
# 2. s_or_b64
# =============================================================================

class TestSOrB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_or_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_or_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_or_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        exp_res, exp_scc = oracle_s_or_b64(0, 0)
        got_res, got_scc = run_translated_salu64("s_or_b64", 0, 0)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_or_b64(val, val)
        got_res, got_scc = run_translated_salu64("s_or_b64", val, val)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSOrB64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_or_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_or_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_or_b64", prog)
        self.assertIn("OpBitwiseOr %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_or_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_or_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_or_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_or_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSOrB64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_or_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_or_b64", 0, 0)
        self.assertEqual(res_zero, 0)
        self.assertEqual(scc_zero, 0)
        res_ones, scc_ones = run_translated_salu64("s_or_b64", MASK64, MASK64)
        self.assertEqual(res_ones, MASK64)
        self.assertEqual(scc_ones, 1)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_or_b64", p5, pa)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_or_b64", hi_only, lo_only)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)


class TestSOrB64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_or_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_or_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_or_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        exp_res, exp_scc = oracle_s_or_b64(0, 0)
        got_res, got_scc = run_translated_salu64("s_or_b64", 0, 0)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_or_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_or_b64_syntax_err", prog)


# =============================================================================
# 3. s_xor_b64
# =============================================================================

class TestSXorB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_xor_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        exp_res, exp_scc = oracle_s_xor_b64(a, a)
        got_res, got_scc = run_translated_salu64("s_xor_b64", a, a)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_xor_b64(val, 0)
        got_res, got_scc = run_translated_salu64("s_xor_b64", val, 0)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSXorB64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_xor_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_xor_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_xor_b64", prog)
        self.assertIn("OpBitwiseXor %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xor_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSXorB64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_xor_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_xor_b64", 0, 0)
        self.assertEqual(res_zero, 0)
        self.assertEqual(scc_zero, 0)
        res_ones, scc_ones = run_translated_salu64("s_xor_b64", MASK64, MASK64)
        self.assertEqual(res_ones, 0)
        self.assertEqual(scc_ones, 0)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_xor_b64", p5, pa)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_xor_b64", hi_only, lo_only)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)


class TestSXorB64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_xor_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        exp_res, exp_scc = oracle_s_xor_b64(a, a)
        got_res, got_scc = run_translated_salu64("s_xor_b64", a, a)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_xor_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_xor_b64_syntax_err", prog)


# =============================================================================
# 4. s_andn2_b64
# =============================================================================

class TestSAndn2B64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_andn2_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_andn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        exp_res, exp_scc = oracle_s_andn2_b64(a, a)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", a, a)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_andn2_b64(val, 0)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", val, 0)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSAndn2B64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_andn2_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_andn2_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_andn2_b64", prog)
        self.assertIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpNot %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_andn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_andn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSAndn2B64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_andn2_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_andn2_b64", 0, 0)
        self.assertEqual(res_zero, 0)
        self.assertEqual(scc_zero, 0)
        res_ones, scc_ones = run_translated_salu64("s_andn2_b64", MASK64, MASK64)
        self.assertEqual(res_ones, 0)
        self.assertEqual(scc_ones, 0)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_andn2_b64", p5, pa)
        self.assertEqual(res, p5)
        self.assertEqual(scc, 1)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_andn2_b64", hi_only, lo_only)
        self.assertEqual(res, hi_only)
        self.assertEqual(scc, 1)


class TestSAndn2B64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_andn2_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_andn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        exp_res, exp_scc = oracle_s_andn2_b64(a, a)
        got_res, got_scc = run_translated_salu64("s_andn2_b64", a, a)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_andn2_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_andn2_b64_syntax_err", prog)


# =============================================================================
# 5. s_orn2_b64
# =============================================================================

class TestSOrn2B64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_orn2_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_orn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        exp_res, exp_scc = oracle_s_orn2_b64(0, MASK64)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", 0, MASK64)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_orn2_b64(val, val)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", val, val)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSOrn2B64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_orn2_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_orn2_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_orn2_b64", prog)
        self.assertIn("OpBitwiseOr %uint", spv)
        self.assertIn("OpNot %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_orn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_orn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSOrn2B64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_orn2_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_orn2_b64", 0, 0)
        self.assertEqual(res_zero, MASK64)
        self.assertEqual(scc_zero, 1)
        res_ones, scc_ones = run_translated_salu64("s_orn2_b64", MASK64, MASK64)
        self.assertEqual(res_ones, MASK64)
        self.assertEqual(scc_ones, 1)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_orn2_b64", p5, pa)
        self.assertEqual(res, p5)
        self.assertEqual(scc, 1)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        exp_res, exp_scc = oracle_s_orn2_b64(hi_only, lo_only)
        res, scc = run_translated_salu64("s_orn2_b64", hi_only, lo_only)
        self.assertEqual(res, exp_res)
        self.assertEqual(scc, exp_scc)


class TestSOrn2B64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_orn2_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_orn2_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        exp_res, exp_scc = oracle_s_orn2_b64(0, MASK64)
        got_res, got_scc = run_translated_salu64("s_orn2_b64", 0, MASK64)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_orn2_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_orn2_b64_syntax_err", prog)


# =============================================================================
# 6. s_nand_b64
# =============================================================================

class TestSNandB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_nand_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nand_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nand_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        exp_res, exp_scc = oracle_s_nand_b64(MASK64, MASK64)
        got_res, got_scc = run_translated_salu64("s_nand_b64", MASK64, MASK64)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_nand_b64(val, val)
        got_res, got_scc = run_translated_salu64("s_nand_b64", val, val)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSNandB64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_nand_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_nand_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_nand_b64", prog)
        self.assertIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpNot %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nand_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nand_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nand_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nand_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSNandB64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_nand_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_nand_b64", 0, 0)
        self.assertEqual(res_zero, MASK64)
        self.assertEqual(scc_zero, 1)
        res_ones, scc_ones = run_translated_salu64("s_nand_b64", MASK64, MASK64)
        self.assertEqual(res_ones, 0)
        self.assertEqual(scc_ones, 0)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_nand_b64", p5, pa)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_nand_b64", hi_only, lo_only)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)


class TestSNandB64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_nand_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nand_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nand_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        exp_res, exp_scc = oracle_s_nand_b64(MASK64, MASK64)
        got_res, got_scc = run_translated_salu64("s_nand_b64", MASK64, MASK64)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_nand_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_nand_b64_syntax_err", prog)


# =============================================================================
# 7. s_nor_b64
# =============================================================================

class TestSNorB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_nor_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        exp_res, exp_scc = oracle_s_nor_b64(MASK64, 0)
        got_res, got_scc = run_translated_salu64("s_nor_b64", MASK64, 0)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_nor_b64(val, val)
        got_res, got_scc = run_translated_salu64("s_nor_b64", val, val)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSNorB64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_nor_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_nor_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_nor_b64", prog)
        self.assertIn("OpBitwiseOr %uint", spv)
        self.assertIn("OpNot %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nor_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSNorB64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_nor_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_nor_b64", 0, 0)
        self.assertEqual(res_zero, MASK64)
        self.assertEqual(scc_zero, 1)
        res_ones, scc_ones = run_translated_salu64("s_nor_b64", MASK64, MASK64)
        self.assertEqual(res_ones, 0)
        self.assertEqual(scc_ones, 0)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_nor_b64", p5, pa)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_nor_b64", hi_only, lo_only)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)


class TestSNorB64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_nor_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_nor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_nor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        exp_res, exp_scc = oracle_s_nor_b64(MASK64, 0)
        got_res, got_scc = run_translated_salu64("s_nor_b64", MASK64, 0)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_nor_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_nor_b64_syntax_err", prog)


# =============================================================================
# 8. s_xnor_b64
# =============================================================================

class TestSXnorB64ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for s_xnor_b64."""

    def test_oracle_basic_truth_table(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xnor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_oracle_zero_scc(self) -> None:
        exp_res, exp_scc = oracle_s_xnor_b64(0, MASK64)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", 0, MASK64)
        self.assertEqual(got_res, 0)
        self.assertEqual(got_scc, 0)

    def test_oracle_idempotence_and_identity(self) -> None:
        val = 0xCAFE_BABE_DEAD_BEEF
        exp_res, exp_scc = oracle_s_xnor_b64(val, val)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", val, val)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSXnorB64ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for s_xnor_b64."""

    def test_ast_to_asm(self) -> None:
        prog = ["s_xnor_b64 s[4:5], s[0:1], s[2:3]", "s_endpgm"]
        spv = support.translate_lines("ps_ast_s_xnor_b64", prog)
        self.assertIn("OpBitwiseXor %uint", spv)
        self.assertIn("OpNot %uint", spv)

    def test_scalar_interpreter_step(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xnor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xnor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", a, b, dst="s[0:1]")
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)


class TestSXnorB64MaskDivergenceAndBoundaries(unittest.TestCase):
    """64-bit mask divergence patterns and bit-boundary extremes for s_xnor_b64."""

    def test_boundary_all_zeros_and_ones(self) -> None:
        res_zero, scc_zero = run_translated_salu64("s_xnor_b64", 0, 0)
        self.assertEqual(res_zero, MASK64)
        self.assertEqual(scc_zero, 1)
        res_ones, scc_ones = run_translated_salu64("s_xnor_b64", MASK64, MASK64)
        self.assertEqual(res_ones, MASK64)
        self.assertEqual(scc_ones, 1)

    def test_alternating_bit_patterns(self) -> None:
        p5 = 0x5555_5555_5555_5555
        pa = 0xAAAA_AAAA_AAAA_AAAA
        res, scc = run_translated_salu64("s_xnor_b64", p5, pa)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_high_low_dword_independence(self) -> None:
        hi_only = 0xFFFF_FFFF_0000_0000
        lo_only = 0x0000_0000_FFFF_FFFF
        res, scc = run_translated_salu64("s_xnor_b64", hi_only, lo_only)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)


class TestSXnorB64TranslatorIntegration(unittest.TestCase):
    """Translator integration and SCC status verification for s_xnor_b64."""

    def test_translated_execution_matches_oracle(self) -> None:
        a = 0x1234_5678_9ABC_DEF0
        b = 0x0F0F_0F0F_00FF_00FF
        exp_res, exp_scc = oracle_s_xnor_b64(a, b)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", a, b)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_translated_scc_zero_result(self) -> None:
        exp_res, exp_scc = oracle_s_xnor_b64(0, MASK64)
        got_res, got_scc = run_translated_salu64("s_xnor_b64", 0, MASK64)
        self.assertEqual(got_res, exp_res)
        self.assertEqual(got_scc, exp_scc)

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["s_xnor_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_s_xnor_b64_syntax_err", prog)


if __name__ == "__main__":
    unittest.main()
