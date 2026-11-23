"""Bundle 36: SALU 32-bit Integer Add/Sub with Carry & Borrow Suite.

Tests for 6 SALU integer arithmetic instructions:
- s_add_i32: 32-bit signed addition with signed overflow in SCC
- s_sub_i32: 32-bit signed subtraction with signed overflow in SCC
- s_add_u32: 32-bit unsigned addition with carry out in SCC
- s_sub_u32: 32-bit unsigned subtraction with borrow out in SCC
- s_addc_u32: 32-bit unsigned addition with carry in from SCC, carry out in SCC
- s_subb_u32: 32-bit unsigned subtraction with borrow in from SCC, borrow out in SCC

Standard library only; zero game data.
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def run_salu_bundle(lines: list[str], name: str = "ps_salu_bundle") -> list[int]:
    """Helper to translate and run shader; words[0] = s0 result, words[1] = scc."""
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


def run_salu_64_bundle(lines: list[str], name: str = "ps_salu_64") -> list[int]:
    """Helper to translate and run 64-bit chain; words[0] = s0 (lo), words[1] = s1 (hi), words[2] = scc."""
    all_lines = list(lines) + [
        "s_cselect_b32 s15, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "v_mov_b32 v2, s15",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


# ==============================================================================
# 1. s_add_i32: Signed 32-bit Addition with Signed Overflow
# ==============================================================================

class TestSAddI32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 20",
            "s_add_i32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_add_i32_basic1")
        self.assertEqual(words1[0], 30)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, -5",
            "s_mov_b32 s2, 5",
            "s_add_i32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_add_i32_basic2")
        self.assertEqual(words2[0], 0)
        self.assertEqual(words2[1], 0)

    def test_signed_overflow(self) -> None:
        # 0x7FFFFFFF + 1 -> 0x80000000 (overflow from positive to negative)
        prog1 = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_mov_b32 s2, 1",
            "s_add_i32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_add_i32_ovf1")
        self.assertEqual(words1[0], 0x80000000)
        self.assertEqual(words1[1], 1)

        # 0x80000000 + 0x80000000 -> 0 (negative + negative = 0 -> overflow)
        prog2 = [
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 0x80000000",
            "s_add_i32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_add_i32_ovf2")
        self.assertEqual(words2[0], 0)
        self.assertEqual(words2[1], 1)


class TestSAddI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        prog = [
            "s_mov_b32 s1, 100",
            "s_mov_b32 s2, -50",
            "s_add_i32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_add_i32_ast1")
        self.assertEqual(words[0], 50)
        self.assertEqual(words[1], 0)

        prog_ovf = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_mov_b32 s2, 10",
            "s_add_i32 s0, s1, s2",
        ]
        words_ovf = run_salu_bundle(prog_ovf, "ps_add_i32_ast2")
        self.assertEqual(words_ovf[0], (0x7FFFFFFF + 10) & MASK32)
        self.assertEqual(words_ovf[1], 1)


class TestSAddI32OverflowAndSCC(unittest.TestCase):
    def test_no_overflow_opposite_signs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_mov_b32 s2, 0x80000000",
            "s_add_i32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_add_i32_opp")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 0)

    def test_boundary_values(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_mov_b32 s2, 0",
            "s_add_i32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_add_i32_bnd1")
        self.assertEqual(words1[0], 0x7FFFFFFF)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_add_i32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_add_i32_bnd2")
        self.assertEqual(words2[0], 0x7FFFFFFF)
        self.assertEqual(words2[1], 1)


class TestSAddI32Translator(unittest.TestCase):
    def test_translator_s_add_i32_execution(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_add_i32 s0, s1, 1",
        ]
        words = run_salu_bundle(prog, "ps_trans_s_add_i32")
        self.assertEqual(words[0], 0x80000000)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 2. s_sub_i32: Signed 32-bit Subtraction with Signed Overflow
# ==============================================================================

class TestSSubI32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 50",
            "s_mov_b32 s2, 20",
            "s_sub_i32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_sub_i32_basic1")
        self.assertEqual(words1[0], 30)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 10",
            "s_sub_i32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_sub_i32_basic2")
        self.assertEqual(words2[0], 0)
        self.assertEqual(words2[1], 0)

    def test_signed_overflow(self) -> None:
        # 0x80000000 - 1 -> 0x7FFFFFFF (negative - positive = positive -> overflow)
        prog1 = [
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 1",
            "s_sub_i32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_sub_i32_ovf1")
        self.assertEqual(words1[0], 0x7FFFFFFF)
        self.assertEqual(words1[1], 1)

        # 0x7FFFFFFF - (-1) -> 0x80000000 (overflow)
        prog2 = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_sub_i32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_sub_i32_ovf2")
        self.assertEqual(words2[0], 0x80000000)
        self.assertEqual(words2[1], 1)


class TestSSubI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        prog = [
            "s_mov_b32 s1, 100",
            "s_mov_b32 s2, 40",
            "s_sub_i32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_sub_i32_ast1")
        self.assertEqual(words[0], 60)
        self.assertEqual(words[1], 0)

        prog_ovf = [
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 1",
            "s_sub_i32 s0, s1, s2",
        ]
        words_ovf = run_salu_bundle(prog_ovf, "ps_sub_i32_ast2")
        self.assertEqual(words_ovf[0], 0x7FFFFFFF)
        self.assertEqual(words_ovf[1], 1)


class TestSSubI32OverflowAndSCC(unittest.TestCase):
    def test_same_signs_no_overflow(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 0x7FFFFFFF",
            "s_mov_b32 s2, 0x7FFFFFFF",
            "s_sub_i32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_sub_i32_same1")
        self.assertEqual(words1[0], 0)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 0x80000000",
            "s_sub_i32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_sub_i32_same2")
        self.assertEqual(words2[0], 0)
        self.assertEqual(words2[1], 0)

    def test_sub_zero_no_overflow(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 0",
            "s_sub_i32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_sub_i32_zero")
        self.assertEqual(words[0], 0x80000000)
        self.assertEqual(words[1], 0)


class TestSSubI32Translator(unittest.TestCase):
    def test_translator_s_sub_i32_execution(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x80000000",
            "s_sub_i32 s0, s1, 1",
        ]
        words = run_salu_bundle(prog, "ps_trans_s_sub_i32")
        self.assertEqual(words[0], 0x7FFFFFFF)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 3. s_add_u32: Unsigned 32-bit Addition with Carry Out
# ==============================================================================

class TestSAddU32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        prog = [
            "s_mov_b32 s1, 100",
            "s_mov_b32 s2, 200",
            "s_add_u32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_add_u32_basic")
        self.assertEqual(words[0], 300)
        self.assertEqual(words[1], 0)

    def test_carry_out(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 1",
            "s_add_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_add_u32_cout1")
        self.assertEqual(words1[0], 0)
        self.assertEqual(words1[1], 1)

        prog2 = [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_add_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_add_u32_cout2")
        self.assertEqual(words2[0], 0xFFFFFFFE)
        self.assertEqual(words2[1], 1)


class TestSAddU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        prog = [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 2",
            "s_add_u32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_add_u32_ast")
        self.assertEqual(words[0], 1)
        self.assertEqual(words[1], 1)


class TestSAddU32CarryAndSCC(unittest.TestCase):
    def test_no_carry_just_below_limit(self) -> None:
        prog = [
            "s_mov_b32 s1, 0xFFFFFFFE",
            "s_mov_b32 s2, 1",
            "s_add_u32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_add_u32_nocarry")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 0)

    def test_carry_with_zero(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_add_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_add_u32_zero1")
        self.assertEqual(words1[0], 0)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0",
            "s_add_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_add_u32_zero2")
        self.assertEqual(words2[0], 0xFFFFFFFF)
        self.assertEqual(words2[1], 0)


class TestSAddU32Translator(unittest.TestCase):
    def test_translator_s_add_u32_execution(self) -> None:
        prog = [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_add_u32 s0, s1, 1",
        ]
        words = run_salu_bundle(prog, "ps_trans_s_add_u32")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 4. s_sub_u32: Unsigned 32-bit Subtraction with Borrow Out
# ==============================================================================

class TestSSubU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 500",
            "s_mov_b32 s2, 200",
            "s_sub_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_sub_u32_basic1")
        self.assertEqual(words1[0], 300)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, 100",
            "s_mov_b32 s2, 100",
            "s_sub_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_sub_u32_basic2")
        self.assertEqual(words2[0], 0)
        self.assertEqual(words2[1], 0)

    def test_borrow_out(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 1",
            "s_sub_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_sub_u32_bout1")
        self.assertEqual(words1[0], 0xFFFFFFFF)
        self.assertEqual(words1[1], 1)

        prog2 = [
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 20",
            "s_sub_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_sub_u32_bout2")
        self.assertEqual(words2[0], 0xFFFFFFFF - 9)
        self.assertEqual(words2[1], 1)


class TestSSubU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        prog = [
            "s_mov_b32 s1, 1",
            "s_mov_b32 s2, 2",
            "s_sub_u32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_sub_u32_ast")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)


class TestSSubU32BorrowAndSCC(unittest.TestCase):
    def test_borrow_boundary(self) -> None:
        prog1 = [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_sub_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_sub_u32_bnd1")
        self.assertEqual(words1[0], 0)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_mov_b32 s1, 0xFFFFFFFE",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_sub_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_sub_u32_bnd2")
        self.assertEqual(words2[0], 0xFFFFFFFF)
        self.assertEqual(words2[1], 1)


class TestSSubU32Translator(unittest.TestCase):
    def test_translator_s_sub_u32_execution(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_sub_u32 s0, s1, 1",
        ]
        words = run_salu_bundle(prog, "ps_trans_s_sub_u32")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 5. s_addc_u32: Unsigned 32-bit Addition with Carry In & Carry Out
# ==============================================================================

class TestSAddcU32Oracle(unittest.TestCase):
    def test_with_carry_in_zero(self) -> None:
        prog1 = [
            "s_add_u32 s14, 0, 0",           # SCC = 0
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 20",
            "s_addc_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_addc_u32_c0_1")
        self.assertEqual(words1[0], 30)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_add_u32 s14, 0, 0",           # SCC = 0
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0",
            "s_addc_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_addc_u32_c0_2")
        self.assertEqual(words2[0], 0xFFFFFFFF)
        self.assertEqual(words2[1], 0)

    def test_with_carry_in_one(self) -> None:
        prog1 = [
            "s_add_u32 s14, 0xFFFFFFFF, 1",   # SCC = 1 (carry out)
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 20",
            "s_addc_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_addc_u32_c1_1")
        self.assertEqual(words1[0], 31)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_add_u32 s14, 0xFFFFFFFF, 1",   # SCC = 1 (carry out)
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0",
            "s_addc_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_addc_u32_c1_2")
        self.assertEqual(words2[0], 0)
        self.assertEqual(words2[1], 1)

        prog3 = [
            "s_add_u32 s14, 0xFFFFFFFF, 1",   # SCC = 1 (carry out)
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 1",
            "s_addc_u32 s0, s1, s2",
        ]
        words3 = run_salu_bundle(prog3, "ps_addc_u32_c1_3")
        self.assertEqual(words3[0], 1)
        self.assertEqual(words3[1], 1)


class TestSAddcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        prog = [
            "s_add_u32 s14, 0xFFFFFFFF, 1",   # SCC = 1
            "s_mov_b32 s1, 0x70000000",
            "s_mov_b32 s2, 0x10000000",
            "s_addc_u32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_addc_u32_ast")
        self.assertEqual(words[0], 0x80000001)
        self.assertEqual(words[1], 0)


class TestSAddcU32ChainingAndPropagation(unittest.TestCase):
    def test_64bit_add_chain(self) -> None:
        # 64-bit addition: A + B
        # A = 0x00000001_FFFFFFFF
        # B = 0x00000002_00000001
        prog = [
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_mov_b32 s3, 0x00000001",
            "s_mov_b32 s4, 0x00000001",
            "s_mov_b32 s5, 0x00000002",
            "s_add_u32 s0, s2, s4",
            "s_addc_u32 s1, s3, s5",
        ]
        words = run_salu_64_bundle(prog, "ps_addc_u32_chain")
        self.assertEqual((words[0], words[1]), (0, 4))
        self.assertEqual(words[2], 0)


class TestSAddcU32Translator(unittest.TestCase):
    def test_translator_s_addc_u32_execution(self) -> None:
        prog = [
            "s_mov_b32 s4, 0xFFFFFFFF",
            "s_add_u32 s5, s4, 1",        # produces carry = 1 in SCC
            "s_mov_b32 s6, 100",
            "s_addc_u32 s0, s6, 200",     # 100 + 200 + 1 = 301
        ]
        words = run_salu_bundle(prog, "ps_trans_s_addc_u32")
        self.assertEqual(words[0], 301)
        self.assertEqual(words[1], 0)


# ==============================================================================
# 6. s_subb_u32: Unsigned 32-bit Subtraction with Borrow In & Borrow Out
# ==============================================================================

class TestSSubbU32Oracle(unittest.TestCase):
    def test_with_borrow_in_zero(self) -> None:
        prog1 = [
            "s_sub_u32 s14, 0, 0",           # SCC = 0
            "s_mov_b32 s1, 50",
            "s_mov_b32 s2, 20",
            "s_subb_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_subb_u32_b0_1")
        self.assertEqual(words1[0], 30)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_sub_u32 s14, 0, 0",           # SCC = 0
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 20",
            "s_subb_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_subb_u32_b0_2")
        self.assertEqual(words2[0], 0xFFFFFFFF - 9)
        self.assertEqual(words2[1], 1)

    def test_with_borrow_in_one(self) -> None:
        prog1 = [
            "s_sub_u32 s14, 0, 1",           # SCC = 1 (borrow out)
            "s_mov_b32 s1, 50",
            "s_mov_b32 s2, 20",
            "s_subb_u32 s0, s1, s2",
        ]
        words1 = run_salu_bundle(prog1, "ps_subb_u32_b1_1")
        self.assertEqual(words1[0], 29)
        self.assertEqual(words1[1], 0)

        prog2 = [
            "s_sub_u32 s14, 0, 1",           # SCC = 1 (borrow out)
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_subb_u32 s0, s1, s2",
        ]
        words2 = run_salu_bundle(prog2, "ps_subb_u32_b1_2")
        self.assertEqual(words2[0], 0xFFFFFFFF)
        self.assertEqual(words2[1], 1)


class TestSSubbU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        prog = [
            "s_sub_u32 s14, 0, 1",           # SCC = 1
            "s_mov_b32 s1, 10",
            "s_mov_b32 s2, 5",
            "s_subb_u32 s0, s1, s2",
        ]
        words = run_salu_bundle(prog, "ps_subb_u32_ast")
        self.assertEqual(words[0], 4)
        self.assertEqual(words[1], 0)


class TestSSubbU32ChainingAndPropagation(unittest.TestCase):
    def test_64bit_sub_chain(self) -> None:
        # 64-bit subtraction: A - B
        # A = 0x00000005_00000000
        # B = 0x00000002_00000001
        prog = [
            "s_mov_b32 s2, 0x00000000",
            "s_mov_b32 s3, 0x00000005",
            "s_mov_b32 s4, 0x00000001",
            "s_mov_b32 s5, 0x00000002",
            "s_sub_u32 s0, s2, s4",
            "s_subb_u32 s1, s3, s5",
        ]
        words = run_salu_64_bundle(prog, "ps_subb_u32_chain")
        self.assertEqual((words[0], words[1]), (0xFFFFFFFF, 2))
        self.assertEqual(words[2], 0)


class TestSSubbU32Translator(unittest.TestCase):
    def test_translator_s_subb_u32_execution(self) -> None:
        prog = [
            "s_mov_b32 s4, 0",
            "s_sub_u32 s5, s4, 1",        # produces borrow = 1 in SCC
            "s_mov_b32 s6, 50",
            "s_subb_u32 s0, s6, 10",      # 50 - 10 - 1 = 39
        ]
        words = run_salu_bundle(prog, "ps_trans_s_subb_u32")
        self.assertEqual(words[0], 39)
        self.assertEqual(words[1], 0)


if __name__ == "__main__":
    unittest.main()
