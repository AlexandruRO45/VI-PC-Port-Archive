"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 SALU 32-bit bitwise logic operations (Bundle 3):
  - s_and_b32:   Bitwise AND
  - s_or_b32:    Bitwise OR
  - s_xor_b32:   Bitwise XOR
  - s_andn2_b32: Bitwise AND with NOT second source
  - s_orn2_b32:  Bitwise OR with NOT second source
  - s_nand_b32:  Bitwise NAND
  - s_nor_b32:   Bitwise NOR
  - s_xnor_b32:  Bitwise XNOR (Equivalence)

In AMD RDNA2 ISA (SOP2 encoding format):
  sdst, ssrc0, ssrc1
  SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def run_salu_logic_ps(lines: list[str], name: str = "ps_salu_logic") -> list[int]:
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
# 1. s_and_b32: Bitwise AND
# ==============================================================================

class TestSAndB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_and_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_and_b32")
        self.assertEqual(words[0], 0x02040608)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_and_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_and_b32")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)


class TestSAndB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_and_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_and_b32", ast_prog)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_and_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_and_b32")
        self.assertEqual(words[0], 0x02040608)
        self.assertEqual(words[1], 1)


class TestSAndB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_and_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_and_b32")
        self.assertEqual(words[0], 0x02040608)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_and_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_and_b32")
        self.assertEqual(words[0], 0x7FC00000 & 0x7F800001)
        self.assertEqual(words[1], 1)


class TestSAndB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_and_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_and_b32")
        self.assertEqual(words[0], 0x02040608)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_and_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_and_b32")
        self.assertEqual(words[0], 5 & 3)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 2. s_or_b32: Bitwise OR
# ==============================================================================

class TestSOrB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_or_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_or_b32")
        self.assertEqual(words[0], 0x1F3F5F7F)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_or_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_or_b32")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)


class TestSOrB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_or_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_or_b32", ast_prog)
        self.assertIn("OpBitwiseOr", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_or_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_or_b32")
        self.assertEqual(words[0], 0x1F3F5F7F)
        self.assertEqual(words[1], 1)


class TestSOrB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_or_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_or_b32")
        self.assertEqual(words[0], 0x1F3F5F7F)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_or_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_or_b32")
        self.assertEqual(words[0], 0x7FC00000 | 0x7F800001)
        self.assertEqual(words[1], 1)


class TestSOrB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_or_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_or_b32")
        self.assertEqual(words[0], 0x1F3F5F7F)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_or_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_or_b32")
        self.assertEqual(words[0], 5 | 3)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 3. s_xor_b32: Bitwise XOR
# ==============================================================================

class TestSXorB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_xor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_xor_b32")
        self.assertEqual(words[0], 0x1D3B5977)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_xor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_xor_b32")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)


class TestSXorB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_xor_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_xor_b32", ast_prog)
        self.assertIn("OpBitwiseXor", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_xor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_xor_b32")
        self.assertEqual(words[0], 0x1D3B5977)
        self.assertEqual(words[1], 1)


class TestSXorB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_xor_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_xor_b32")
        self.assertEqual(words[0], 0x1D3B5977)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_xor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_xor_b32")
        self.assertEqual(words[0], 0x7FC00000 ^ 0x7F800001)
        self.assertEqual(words[1], 1)


class TestSXorB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_xor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_xor_b32")
        self.assertEqual(words[0], 0x1D3B5977)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_xor_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_xor_b32")
        self.assertEqual(words[0], 5 ^ 3)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 4. s_andn2_b32: Bitwise AND with NOT second source
# ==============================================================================

class TestSAndn2B32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_andn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_andn2_b32")
        self.assertEqual(words[0], 0x10305070)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_andn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_andn2_b32")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)


class TestSAndn2B32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_andn2_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_andn2_b32", ast_prog)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_andn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_andn2_b32")
        self.assertEqual(words[0], 0x10305070)
        self.assertEqual(words[1], 1)


class TestSAndn2B32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_andn2_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_andn2_b32")
        self.assertEqual(words[0], 0x10305070)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_andn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_andn2_b32")
        self.assertEqual(words[0], 0x7FC00000 & (~0x7F800001 & MASK32))
        self.assertEqual(words[1], 1)


class TestSAndn2B32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_andn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_andn2_b32")
        self.assertEqual(words[0], 0x10305070)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_andn2_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_andn2_b32")
        self.assertEqual(words[0], 5 & (~3 & MASK32))
        self.assertEqual(words[1], 1)


# ==============================================================================
# 5. s_orn2_b32: Bitwise OR with NOT second source
# ==============================================================================

class TestSOrn2B32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_orn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_orn2_b32")
        self.assertEqual(words[0], 0xF2F4F6F8)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_orn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_orn2_b32")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)


class TestSOrn2B32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_orn2_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_orn2_b32", ast_prog)
        self.assertIn("OpBitwiseOr", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_orn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_orn2_b32")
        self.assertEqual(words[0], 0xF2F4F6F8)
        self.assertEqual(words[1], 1)


class TestSOrn2B32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_orn2_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_orn2_b32")
        self.assertEqual(words[0], 0xF2F4F6F8)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_orn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_orn2_b32")
        self.assertEqual(words[0], 0x7FC00000 | (~0x7F800001 & MASK32))
        self.assertEqual(words[1], 1)


class TestSOrn2B32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_orn2_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_orn2_b32")
        self.assertEqual(words[0], 0xF2F4F6F8)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_orn2_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_orn2_b32")
        self.assertEqual(words[0], (5 | (~3)) & MASK32)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 6. s_nand_b32: Bitwise NAND
# ==============================================================================

class TestSNandB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_nand_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_nand_b32")
        self.assertEqual(words[0], 0xFDFBF9F7)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_nand_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_nand_b32")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)


class TestSNandB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_nand_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_nand_b32", ast_prog)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_nand_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_nand_b32")
        self.assertEqual(words[0], 0xFDFBF9F7)
        self.assertEqual(words[1], 1)


class TestSNandB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_nand_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_nand_b32")
        self.assertEqual(words[0], 0xFDFBF9F7)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_nand_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_nand_b32")
        self.assertEqual(words[0], (~(0x7FC00000 & 0x7F800001)) & MASK32)
        self.assertEqual(words[1], 1)


class TestSNandB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_nand_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_nand_b32")
        self.assertEqual(words[0], 0xFDFBF9F7)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_nand_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_nand_b32")
        self.assertEqual(words[0], (~(5 & 3)) & MASK32)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 7. s_nor_b32: Bitwise NOR
# ==============================================================================

class TestSNorB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_nor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_nor_b32")
        self.assertEqual(words[0], 0xE0C0A080)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_nor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_nor_b32")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)


class TestSNorB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_nor_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_nor_b32", ast_prog)
        self.assertIn("OpBitwiseOr", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_nor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_nor_b32")
        self.assertEqual(words[0], 0xE0C0A080)
        self.assertEqual(words[1], 1)


class TestSNorB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_nor_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_nor_b32")
        self.assertEqual(words[0], 0xE0C0A080)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_nor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_nor_b32")
        self.assertEqual(words[0], (~(0x7FC00000 | 0x7F800001)) & MASK32)
        self.assertEqual(words[1], 1)


class TestSNorB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_nor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_nor_b32")
        self.assertEqual(words[0], 0xE0C0A080)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_nor_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_nor_b32")
        self.assertEqual(words[0], (~(5 | 3)) & MASK32)
        self.assertEqual(words[1], 1)


# ==============================================================================
# 8. s_xnor_b32: Bitwise XNOR (Equivalence)
# ==============================================================================

class TestSXnorB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_xnor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_oracle_s_xnor_b32")
        self.assertEqual(words[0], 0xE2C4A688)
        self.assertEqual(words[1], 1)

    def test_oracle_zero_inputs(self) -> None:
        prog = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0",
            "s_xnor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_zero_s_xnor_b32")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)


class TestSXnorB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        ast_prog = [
            "s_xnor_b32 s0, s1, s2",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_xnor_b32", ast_prog)
        self.assertIn("OpBitwiseXor", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("%scc", spv)

    def test_scalar_interpreter_step(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_xnor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_step_s_xnor_b32")
        self.assertEqual(words[0], 0xE2C4A688)
        self.assertEqual(words[1], 1)


class TestSXnorB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        prog = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_xnor_b32 s0, s0, s1",
        ]
        words = run_salu_logic_ps(prog, "ps_alias_s_xnor_b32")
        self.assertEqual(words[0], 0xE2C4A688)
        self.assertEqual(words[1], 1)

    def test_ieee754_bit_transparency(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x7FC00000",
            "s_mov_b32 s2, 0x7F800001",
            "s_xnor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_ieee_s_xnor_b32")
        self.assertEqual(words[0], (~(0x7FC00000 ^ 0x7F800001)) & MASK32)
        self.assertEqual(words[1], 1)


class TestSXnorB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        prog = [
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_xnor_b32 s0, s1, s2",
        ]
        words = run_salu_logic_ps(prog, "ps_bundle_s_xnor_b32")
        self.assertEqual(words[0], 0xE2C4A688)

    def test_inline_constant_folding(self) -> None:
        prog = [
            "s_xnor_b32 s0, 5, 3",
        ]
        words = run_salu_logic_ps(prog, "ps_fold_s_xnor_b32")
        self.assertEqual(words[0], (~(5 ^ 3)) & MASK32)
        self.assertEqual(words[1], 1)


if __name__ == "__main__":
    unittest.main()
