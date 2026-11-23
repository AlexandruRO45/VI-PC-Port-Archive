"""Hardware and translator regression tests for RDNA2 v_rcp_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  The v_rcp_f16 instruction computes the IEEE-754 binary16 half-precision reciprocal:
      D.f16 = 1.0 / S0.f16

  On GFX10, 16-bit float operations preserve the unselected half of the destination register:
  - Low half destination (default, op_sel[1]=0): preserves bits [31:16] of the destination register.
  - High half destination (op_sel[1]=1): preserves bits [15:0] of the destination register.
  - Source selection (op_sel[0]): selects low half (0) or high half (1) of the source operand.
  - Modifiers: neg (-v0), abs (|v0|), clamp [0.0, 1.0], and omod (mul:2, mul:4, div:2).

Covers:
- Basic half-precision reciprocal execution through tools/rdna2spirv.py and spirv_mini_interp
- Powers of two and reciprocal identities (1.0/2.0 = 0.5, 1.0/0.5 = 2.0, 1.0/4.0 = 0.25, 1.0/0.25 = 4.0)
- Sign handling: positive and negative unit values (1.0/1.0 = 1.0, 1.0/-1.0 = -1.0)
- Input modifiers: negation (-v0), absolute value (|v0|), combined negation and absolute value (-|v0|)
- Output modifiers: clamp to [0.0, 1.0], omod mul:2, omod mul:4, omod div:2
- Half-register operand selection (op_sel:[0,0], op_sel:[1,0], op_sel:[0,1], op_sel:[1,1])
- Unselected destination half preservation (low-half preserves high; high-half preserves low)
- In-place register aliasing (vdst == src0 for all op_sel permutations)
- IEEE-754 boundary conditions: infinities to signed zeros (1.0/+inf = +0.0, 1.0/-inf = -0.0)
- Division-by-zero behavior (OpFDiv undefined in SPIR-V, caught by spirv_mini_interp)
- Instruction syntax rejections and negative syntax controls
- Translator emission verification (OpFDiv, lack of rounding mode override on hardware)

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import spirv_mini_interp  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

# IEEE-754 binary16 constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_ONE = 0x3C00          # 1.0
F16_NEG_ONE = 0xBC00      # -1.0
F16_TWO = 0x4000          # 2.0
F16_NEG_TWO = 0xC000      # -2.0
F16_HALF = 0x3800         # 0.5
F16_NEG_HALF = 0xB800     # -0.5
F16_FOUR = 0x4400         # 4.0
F16_QUARTER = 0x3400      # 0.25


def run_ps_rcp(line: str, v0_val: int = F16_TWO, initial_dst: int = 0xDEADBEEF) -> int:
    """Translates and executes a fragment shader line using v_rcp_f16."""
    lines = [
        f"v_mov_b32 v0, {v0_val:#010x}",
        f"v_mov_b32 v1, {initial_dst:#010x}",
        line,
        "exp mrt0 v1, v1, v1, v1 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_rcp_f16", lines)
    words = mrt0_words(spv)
    return words[0]


def run_ps_rcp_inplace(line: str, v0_val: int) -> int:
    """Translates and executes in-place aliased v_rcp_f16 where vdst is also src0."""
    lines = [
        f"v_mov_b32 v0, {v0_val:#010x}",
        line,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_rcp_f16_inplace", lines)
    words = mrt0_words(spv)
    return words[0]


class TestRcpF16BasicExecution(unittest.TestCase):
    def test_reciprocal_two_gives_half(self):
        # 1.0 / 2.0 (0x4000) = 0.5 (0x3800), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_TWO)
        self.assertEqual(word, 0xDEAD3800)

    def test_reciprocal_half_gives_two(self):
        # 1.0 / 0.5 (0x3800) = 2.0 (0x4000), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_HALF)
        self.assertEqual(word, 0xDEAD4000)

    def test_reciprocal_four_gives_quarter(self):
        # 1.0 / 4.0 (0x4400) = 0.25 (0x3400), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_FOUR)
        self.assertEqual(word, 0xDEAD3400)

    def test_reciprocal_quarter_gives_four(self):
        # 1.0 / 0.25 (0x3400) = 4.0 (0x4400), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_QUARTER)
        self.assertEqual(word, 0xDEAD4400)

    def test_reciprocal_one_identity(self):
        # 1.0 / 1.0 (0x3C00) = 1.0 (0x3C00), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_ONE)
        self.assertEqual(word, 0xDEAD3C00)

    def test_reciprocal_negative_one_identity(self):
        # 1.0 / -1.0 (0xBC00) = -1.0 (0xBC00), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_NEG_ONE)
        self.assertEqual(word, 0xDEADBC00)

    def test_reciprocal_negative_two_gives_negative_half(self):
        # 1.0 / -2.0 (0xC000) = -0.5 (0xB800), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_NEG_TWO)
        self.assertEqual(word, 0xDEADB800)


class TestRcpF16Modifiers(unittest.TestCase):
    def test_input_negation(self):
        # 1.0 / -2.0 = -0.5 (0xB800)
        word = run_ps_rcp("v_rcp_f16 v1, -v0", v0_val=F16_TWO)
        self.assertEqual(word, 0xDEADB800)

    def test_input_absolute_value(self):
        # 1.0 / |-2.0| = 0.5 (0x3800)
        word = run_ps_rcp("v_rcp_f16 v1, |v0|", v0_val=F16_NEG_TWO)
        self.assertEqual(word, 0xDEAD3800)

    def test_input_negated_absolute_value(self):
        # 1.0 / -|2.0| = -0.5 (0xB800)
        word = run_ps_rcp("v_rcp_f16 v1, -|v0|", v0_val=F16_TWO)
        self.assertEqual(word, 0xDEADB800)

    def test_output_clamp_positive(self):
        # 1.0 / 0.5 = 2.0 -> clamped to 1.0 (0x3C00)
        word = run_ps_rcp("v_rcp_f16 v1, v0 clamp", v0_val=F16_HALF)
        self.assertEqual(word, 0xDEAD3C00)

    def test_output_clamp_negative(self):
        # 1.0 / -2.0 = -0.5 -> clamped to 0.0 (0x0000)
        word = run_ps_rcp("v_rcp_f16 v1, v0 clamp", v0_val=F16_NEG_TWO)
        self.assertEqual(word, 0xDEAD0000)

    def test_output_modifier_mul2(self):
        # 1.0 / 2.0 = 0.5 -> mul:2 = 1.0 (0x3C00)
        word = run_ps_rcp("v_rcp_f16 v1, v0 mul:2", v0_val=F16_TWO)
        self.assertEqual(word, 0xDEAD3C00)

    def test_output_modifier_mul4(self):
        # 1.0 / 2.0 = 0.5 -> mul:4 = 2.0 (0x4000)
        word = run_ps_rcp("v_rcp_f16 v1, v0 mul:4", v0_val=F16_TWO)
        self.assertEqual(word, 0xDEAD4000)

    def test_output_modifier_div2(self):
        # 1.0 / 2.0 = 0.5 -> div:2 = 0.25 (0x3400)
        word = run_ps_rcp("v_rcp_f16 v1, v0 div:2", v0_val=F16_TWO)
        self.assertEqual(word, 0xDEAD3400)


class TestRcpF16OpSel(unittest.TestCase):
    def test_opsel_read_low_write_low(self):
        # Read low half (2.0 = 0x4000), write low half (0.5 = 0x3800), preserve high 0xdead
        word = run_ps_rcp("v_rcp_f16 v1, v0 op_sel:[0,0]", v0_val=0x55554000, initial_dst=0xDEADBEEF)
        self.assertEqual(word, 0xDEAD3800)

    def test_opsel_read_high_write_low(self):
        # Read high half (2.0 = 0x4000), write low half (0.5 = 0x3800), preserve high 0xdead
        word = run_ps_rcp("v_rcp_f16 v1, v0 op_sel:[1,0]", v0_val=0x40005555, initial_dst=0xDEADBEEF)
        self.assertEqual(word, 0xDEAD3800)

    def test_opsel_read_low_write_high(self):
        # Read low half (2.0 = 0x4000), write high half (0.5 = 0x3800), preserve low 0xbeef
        word = run_ps_rcp("v_rcp_f16 v1, v0 op_sel:[0,1]", v0_val=0x55554000, initial_dst=0xDEADBEEF)
        self.assertEqual(word, 0x3800BEEF)

    def test_opsel_read_high_write_high(self):
        # Read high half (2.0 = 0x4000), write high half (0.5 = 0x3800), preserve low 0xbeef
        word = run_ps_rcp("v_rcp_f16 v1, v0 op_sel:[1,1]", v0_val=0x40005555, initial_dst=0xDEADBEEF)
        self.assertEqual(word, 0x3800BEEF)


class TestRcpF16RegisterAliasing(unittest.TestCase):
    def test_inplace_low_half_overwrite(self):
        # vdst == src0: write to low half, preserve high half
        word = run_ps_rcp_inplace("v_rcp_f16 v0, v0", v0_val=0x55AA4000)
        self.assertEqual(word, 0x55AA3800)

    def test_inplace_high_half_overwrite(self):
        # vdst == src0: write to high half, preserve low half
        word = run_ps_rcp_inplace("v_rcp_f16 v0, v0 op_sel:[1,1]", v0_val=0x4000BEEF)
        self.assertEqual(word, 0x3800BEEF)

    def test_inplace_high_to_low(self):
        # vdst == src0: read high half, write to low half, preserve high half
        word = run_ps_rcp_inplace("v_rcp_f16 v0, v0 op_sel:[1,0]", v0_val=0x40001234)
        self.assertEqual(word, 0x40003800)

    def test_inplace_low_to_high(self):
        # vdst == src0: read low half, write to high half, preserve low half
        word = run_ps_rcp_inplace("v_rcp_f16 v0, v0 op_sel:[0,1]", v0_val=0x12344000)
        self.assertEqual(word, 0x38004000)


class TestRcpF16SpecialValues(unittest.TestCase):
    def test_positive_infinity_gives_positive_zero(self):
        # 1.0 / +inf (0x7C00) = +0.0 (0x0000), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_POS_INF)
        self.assertEqual(word, 0xDEAD0000)

    def test_negative_infinity_gives_negative_zero(self):
        # 1.0 / -inf (0xFC00) = -0.0 (0x8000), high half 0xdead preserved
        word = run_ps_rcp("v_rcp_f16 v1, v0", v0_val=F16_NEG_INF)
        self.assertEqual(word, 0xDEAD8000)

    def test_division_by_zero_translates_and_triggers_unsupported_in_interp(self):
        # Translates successfully to valid SPIR-V OpFDiv, but interpreter halts on div-by-zero
        lines = [
            f"v_mov_b32 v0, {F16_POS_ZERO:#010x}",
            "v_mov_b32 v1, 0xdeadbeef",
            "v_rcp_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_div_zero", lines)
        self.assertIn("OpFDiv %float", spv)
        with self.assertRaises(spirv_mini_interp.UnsupportedConstruct):
            mrt0_words(spv)


class TestRcpF16NegativeSyntaxControls(unittest.TestCase):
    def test_missing_operands_rejected(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_0", ["v_rcp_f16", "s_endpgm"])

    def test_missing_source_operand_rejected(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_1", ["v_rcp_f16 v0", "s_endpgm"])

    def test_invalid_modifier_rejected(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_2", ["v_rcp_f16 v1, v0 bad_modifier", "s_endpgm"])


class TestRcpF16TranslatorEmission(unittest.TestCase):
    def test_spirv_op_fdiv_emission(self):
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_rcp_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_emission", lines)
        self.assertIn("OpFDiv %float", spv)

    def test_spirv_unselected_half_preservation_emission(self):
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_rcp_f16 v1, v0 op_sel:[0,0]",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_emission_opsel", lines)
        # Low half destination preserves high half via bitwise operations
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpBitwiseOr", spv)


if __name__ == "__main__":
    unittest.main()
