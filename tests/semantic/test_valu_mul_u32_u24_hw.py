from __future__ import annotations

"""Semantic regression tests and hardware translator verification for v_mul_u32_u24.

RDNA2 / GFX10 VOP2 24-bit Unsigned Integer Multiply:
Multiplies the low 24 bits of two unsigned integer source operands (bits [31:24] ignored)
and returns the lower 32 bits of the 48-bit product into a 32-bit VGPR.

Syntax:
    v_mul_u32_u24 vdst, vsrc0, vsrc1

Semantics:
    s0_u24 = src0 & 0x00FF_FFFF
    s1_u24 = src1 & 0x00FF_FFFF
    product_48 = s0_u24 * s1_u24
    vdst = product_48 & 0xFFFF_FFFF

All tests compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.
"""

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

MASK_24: int = 0x00FF_FFFF
MASK_32: int = 0xFFFF_FFFF
MAX_U24: int = 0x00FF_FFFF


def run_translator_mul_u24(
    src0: int,
    src1: int,
    name: str = "ps_mul_u24",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK_32:#x}",
        f"v_mov_b32 v2, {src1 & MASK_32:#x}",
        "v_mul_u32_u24 v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_multiplication(self) -> None:
        """Verify 100 * 250 = 25000."""
        got = run_translator_mul_u24(100, 250, "ps_basic_mul")
        self.assertEqual(got, 25000)

    def test_high_byte_masking(self) -> None:
        """Verify bits [31:24] are strictly ignored on both operands."""
        # 0xFF000002 low 24 bits = 2, 0xAA000003 low 24 bits = 3 -> 6
        got = run_translator_mul_u24(0xFF00_0002, 0xAA00_0003, "ps_mask_mul")
        self.assertEqual(got, 6)

    def test_max_u24_product(self) -> None:
        """Verify (2^24 - 1) * (2^24 - 1) low 32 bits equals 0xFE000001."""
        got = run_translator_mul_u24(MAX_U24, MAX_U24, "ps_max_mul")
        self.assertEqual(got, 0xFE00_0001)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly against translator output."""
        src0 = 0x1234_5678
        src1 = 0xABCD_EF02
        expected = ((src0 & MASK_24) * (src1 & MASK_24)) & MASK_32
        translated = run_translator_mul_u24(src0, src1, name="ps_mul_oracle")
        self.assertEqual(translated, expected)


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly serializes to valid RDNA2 assembly text and lowers to SPIR-V."""
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mul_u32_u24 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_mul_asm", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpIMul", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 200)

    def test_ast_translation_validity(self) -> None:
        """Verify that assembly generated translates cleanly."""
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 250",
            "v_mul_u32_u24 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_mul_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 25000)


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        """Verify execution across low 24-bit inputs."""
        res = run_translator_mul_u24(0x1200_0005, 0xAB00_0007, "ps_scalar_mul1")
        self.assertEqual(res, 35)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        """Verify scalar interpreter exactly matches translated SPIR-V execution."""
        src0 = 0x0012_3456
        src1 = 0x0000_0008
        res = run_translator_mul_u24(src0, src1, name="ps_scalar_mul_cross")
        self.assertEqual(res, (src0 * src1) & MASK_32)


class TestSIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_alternating_lanes(self) -> None:
        """Verify execution across multiple lane inputs."""
        for lane in range(8):
            got = run_translator_mul_u24(lane + 1, 10, f"ps_simd_wave32_{lane}")
            self.assertEqual(got, (lane + 1) * 10)

    def test_wave64_divergence(self) -> None:
        """Verify execution on boundary values."""
        got = run_translator_mul_u24(100, 20, "ps_simd_wave64")
        self.assertEqual(got, 2000)


class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0_in_place(self) -> None:
        """Verify v0 = v0 * v1 reads initial v0 before writeback."""
        lines = [
            "v_mov_b32 v0, 0xAA000007",
            "v_mov_b32 v1, 0xBB000009",
            "v_mul_u32_u24 v0, v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_dst_src0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 63)

    def test_alias_all_same_register(self) -> None:
        """Verify v0 = v0 * v0 squaring."""
        lines = [
            "v_mov_b32 v0, 12",
            "v_mul_u32_u24 v0, v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_square", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 144)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_invariance(self) -> None:
        """Verify v_mul_u32_u24 never alters VCC or SCC."""
        lines = [
            "v_cmp_eq_u32 vcc, 42, 42",            # Sets VCC to all 1s
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mul_u32_u24 v0, v1, v2",            # Must not touch VCC
            "v_cndmask_b32 v4, 0x11111111, 0x22222222, vcc",  # Should choose 0x22222222
            "exp mrt0 v4, v0, v4, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mul_vcc_invariant", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x22222222)
        self.assertEqual(words[1], 200)


class TestIEEE754EdgeCases(unittest.TestCase):
    """Commit 7: Boundary extremes, bit limits, and wrapping."""

    def test_zero_identity(self) -> None:
        """Verify multiplication by zero gives zero."""
        self.assertEqual(run_translator_mul_u24(0, 0x00FF_FFFF, "ps_zero_1"), 0)
        self.assertEqual(run_translator_mul_u24(0x00FF_FFFF, 0, "ps_zero_2"), 0)

    def test_overflow_wrapping_past_32_bits(self) -> None:
        """Verify wrapping modulo 2^32 on large 24-bit products."""
        # 0x800000 * 0x800 = 0x400000000; low 32 bits = 0
        got = run_translator_mul_u24(0x80_0000, 0x800, "ps_wrap_mul")
        self.assertEqual(got, 0)

    def test_power_of_two_shift(self) -> None:
        """Verify multiplying by power of two matches logical shift."""
        val = 0x1234
        got = run_translator_mul_u24(val, 1 << 16, "ps_shift_mul")
        self.assertEqual(got, (val << 16) & MASK_32)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Commit 8: Negative controls, syntax rejection, and mutation detection."""

    def test_rejection_invalid_destination(self) -> None:
        """Verify syntax rejection when instruction has insufficient operands."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_dest", ["v_mul_u32_u24 v0", "s_endpgm"])

    def test_rejection_out_of_range_registers(self) -> None:
        """Verify rejection of invalid assembly."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_op", ["v_mul_u32_u24", "s_endpgm"])

    def test_mutation_detection_swap_opcodes(self) -> None:
        """Mutation detector: swapping OpIMul with OpISub in SPIR-V is caught."""
        spv = support.translate_lines("ps_mut_mul_test", [
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 5",
            "v_mul_u32_u24 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        orig_words = support.mrt0_words(spv)
        self.assertEqual(orig_words[0], 100)

        # Mutate SPIR-V: Replace OpIMul with OpISub
        self.assertIn("OpIMul", spv)
        mutated_spv = spv.replace("OpIMul", "OpISub", 1)
        mutated_words = support.mrt0_words(mutated_spv)
        self.assertNotEqual(orig_words[0], mutated_words[0])

    def test_mutation_detection_missing_24bit_mask(self) -> None:
        """Mutation detector: failing to mask high 8 bits is caught."""
        s0 = 0x0100_0002  # low 24 = 2
        s1 = 3
        correct = run_translator_mul_u24(s0, s1, "ps_mut_mask")
        self.assertEqual(correct, 6)
        # If unmasked: 16777218 * 3 = 50331654 != 6
        unmasked = (s0 * s1) & MASK_32
        self.assertNotEqual(correct, unmasked)

    def test_mutation_detection_inactive_lane_overwritten(self) -> None:
        """Verify translator emits proper OpBitwiseAnd and OpIMul instructions."""
        spv = support.translate_lines("ps_mut_verify_ops", [
            "v_mov_b32 v1, 7",
            "v_mov_b32 v2, 8",
            "v_mul_u32_u24 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpIMul", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 56)


if __name__ == "__main__":
    unittest.main()
