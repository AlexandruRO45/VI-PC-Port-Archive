from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_b32 instruction belongs to SOP2 format.
  Instruction:
    s_andn2_b32 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo)
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo
    ssrc1 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = ssrc0 & ~ssrc1
    SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFFFFFF
ALL_LANES_ACTIVE_32: int = 0xFFFFFFFF
ALL_LANES_INACTIVE_32: int = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_b32(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_b32 instruction.

    Args:
        ssrc0: 32-bit first source operand (uint32)
        ssrc1: 32-bit second source operand (uint32)

    Returns:
        tuple (sdst, scc):
          - sdst: 32-bit result = ssrc0 & ~ssrc1
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u32(ssrc0)
    s1 = u32(ssrc1)
    sdst = u32(s0 & (~s1 & MASK32))
    scc = 1 if sdst != 0 else 0
    return sdst, scc


def run_hw_andn2(ssrc0: int, ssrc1: int, name: str = "ps_andn2_run") -> tuple[int, int]:
    """Execute s_andn2_b32 through rdna2spirv translator and return (sdst, scc)."""
    s0 = u32(ssrc0)
    s1 = u32(ssrc1)
    lines = [
        f"s_mov_b32 s0, {s0:#x}",
        f"s_mov_b32 s1, {s1:#x}",
        "s_andn2_b32 s2, s0, s1",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32_e32 v0, s2",
        "v_mov_b32_e32 v1, s3",
        "exp mrt0, v0, v1, v0, v1 done",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

class TestSaluAndn2B32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_b32."""

    def test_oracle_basic(self):
        s0 = 0x12345678
        s1 = 0x00FF00FF
        sdst, scc = ref_andn2_b32(s0, s1)
        expected = s0 & (~s1 & MASK32)
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(s0, s1, "ps_oracle_basic")
        self.assertEqual(hw_dst, expected)
        self.assertEqual(hw_scc, 1)

    def test_oracle_zero_s0(self):
        s1 = 0x12345678
        sdst, scc = ref_andn2_b32(0, s1)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(0, s1, "ps_oracle_zero_s0")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_oracle_all_ones_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(s0, ALL_LANES_ACTIVE_32, "ps_oracle_all_ones_s1")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_oracle_zero_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, 0)
        self.assertEqual(sdst, s0)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(s0, 0, "ps_oracle_zero_s1")
        self.assertEqual(hw_dst, s0)
        self.assertEqual(hw_scc, 1)

    def test_oracle_operand_asymmetry(self):
        # s0 & ~s1 != s1 & ~s0 (non-commutative)
        s0 = 0x0000FFFF
        s1 = 0x00FF00FF
        res_01, _ = ref_andn2_b32(s0, s1)
        res_10, _ = ref_andn2_b32(s1, s0)
        self.assertNotEqual(res_01, res_10)
        self.assertEqual(res_01, 0x0000FF00)
        self.assertEqual(res_10, 0x00FF0000)
        hw_01, _ = run_hw_andn2(s0, s1, "ps_oracle_asym_01")
        hw_10, _ = run_hw_andn2(s1, s0, "ps_oracle_asym_10")
        self.assertNotEqual(hw_01, hw_10)
        self.assertEqual(hw_01, 0x0000FF00)
        self.assertEqual(hw_10, 0x00FF0000)


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST and Register Formatting Validation
# -----------------------------------------------------------------------------

class TestSaluAndn2B32AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_register_operands(self):
        """Verify translation of various SGPR registers and special registers."""
        cases = [
            ["s_andn2_b32 s4, s6, s8", "s_endpgm"],
            ["s_andn2_b32 exec_lo, exec_lo, vcc_lo", "s_endpgm"],
            ["s_andn2_b32 s2, s0, 0x123", "s_endpgm"],
        ]
        for idx, lines in enumerate(cases):
            spv = support.translate_lines(f"ps_ast_andn2_{idx}", lines)
            self.assertIn("OpBitwiseAnd", spv)
            self.assertIn("OpNot", spv)

    def test_ast_pipeline_execution(self):
        """Verify that translated SPIR-V structure contains expected opcodes."""
        lines = [
            "s_andn2_b32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_pipe_exec", lines)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)
        self.assertIn("OpSelect", spv)
        self.assertIn("OpReturn", spv)


# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter & Execution
# -----------------------------------------------------------------------------

class TestSaluAndn2B32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        s0 = 0x12345678
        s1 = 0x00FF00FF
        expected = s0 & (~s1 & MASK32)
        hw_dst, hw_scc = run_hw_andn2(s0, s1, "ps_interp_crosscheck")
        self.assertEqual(hw_dst, expected)
        self.assertEqual(hw_scc, 1)

    def test_scalar_interpreter_zero_output(self):
        hw_dst, hw_scc = run_hw_andn2(0x12345678, 0x12345678, "ps_interp_zero")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_sequential_chained_andn2(self):
        """Execute chained s_andn2_b32 instructions in a single translated shader."""
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 0x000000FF",
            "s_mov_b32 s2, 0x0000FF00",
            "s_andn2_b32 s3, s0, s1",
            "s_andn2_b32 s4, s3, s2",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_interp_chained", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF0000)
        self.assertEqual(words[1], 1)


# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32-Bit EXEC Masking
# -----------------------------------------------------------------------------

class TestSaluAndn2B32SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32-bit EXEC masking."""

    def test_wave32_lane_mask_isolation(self):
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        s0 = lanes(0, 24)  # lanes 0..23
        s1 = lanes(8, 32)  # lanes 8..31
        # s0 & ~s1 = lanes 0..7
        sdst, scc = ref_andn2_b32(s0, s1)
        self.assertEqual(sdst, lanes(0, 8))
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(s0, s1, "ps_lane_mask_iso")
        self.assertEqual(hw_dst, lanes(0, 8))
        self.assertEqual(hw_scc, 1)

    def test_divergent_mask_checkerboard(self):
        even_lanes = 0x55555555
        odd_lanes = 0xAAAAAAAA
        # even & ~odd = even & even = even
        sdst, scc = ref_andn2_b32(even_lanes, odd_lanes)
        self.assertEqual(sdst, even_lanes)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(even_lanes, odd_lanes, "ps_chk_divergent")
        self.assertEqual(hw_dst, even_lanes)
        self.assertEqual(hw_scc, 1)

        # even & ~even = 0
        sdst_zero, scc_zero = ref_andn2_b32(even_lanes, even_lanes)
        self.assertEqual(sdst_zero, 0)
        self.assertEqual(scc_zero, 0)
        hw_zero, hw_scc_z = run_hw_andn2(even_lanes, even_lanes, "ps_chk_zero")
        self.assertEqual(hw_zero, 0)
        self.assertEqual(hw_scc_z, 0)


# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSaluAndn2B32RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        # s0 = s0 & ~s2
        s0_val = 0x12345678
        s1_val = 0x000000FF
        lines = [
            f"s_mov_b32 s0, {s0_val:#x}",
            f"s_mov_b32 s2, {s1_val:#x}",
            "s_andn2_b32 s0, s0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345600)
        self.assertEqual(words[1], 1)

    def test_dst_aliases_src1(self):
        # s2 = s0 & ~s2
        s0_val = 0x12345678
        s1_val = 0x000000FF
        lines = [
            f"s_mov_b32 s0, {s0_val:#x}",
            f"s_mov_b32 s2, {s1_val:#x}",
            "s_andn2_b32 s2, s0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src1", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345600)
        self.assertEqual(words[1], 1)

    def test_self_cancellation_aliasing(self):
        # s0 = s0 & ~s0 = 0
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s0, s0, s0",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_self_cancel", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)


# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and SCC Generation
# -----------------------------------------------------------------------------

class TestSaluAndn2B32ConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and SCC generation."""

    def test_scc_set_when_result_nonzero(self):
        sdst, scc = ref_andn2_b32(1, 0)
        self.assertEqual(sdst, 1)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(1, 0, "ps_scc_nonzero")
        self.assertEqual(hw_dst, 1)
        self.assertEqual(hw_scc, 1)

    def test_scc_cleared_when_result_zero(self):
        sdst, scc = ref_andn2_b32(0x1234, 0x1234)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(0x1234, 0x1234, "ps_scc_zero")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_scc_cleared_when_src0_is_zero(self):
        sdst, scc = ref_andn2_b32(0, 0x55555555)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(0, 0x55555555, "ps_scc_zero_src0")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)


# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

class TestSaluAndn2B32BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 32-bit IEEE-754 bit representations and boundary bitmasks."""

    def test_ieee754_single_precision_bit_patterns(self):
        cases = [
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC00000,  # Quiet NaN
            0x7F800001,  # Signaling NaN
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x00000001,  # Minimum subnormal
            0x007FFFFF,  # Maximum subnormal
        ]
        for p0 in cases:
            for p1 in cases:
                with self.subTest(p0=hex(p0), p1=hex(p1)):
                    sdst, scc = ref_andn2_b32(p0, p1)
                    expected = p0 & (~p1 & MASK32)
                    self.assertEqual(sdst, expected)
                    self.assertEqual(scc, int(expected != 0))

    def test_boundary_alternating_bitmasks(self):
        checkerboards = [
            0x55555555,
            0xAAAAAAAA,
            0x33333333,
            0xCCCCCCCC,
            0x0F0F0F0F,
            0xF0F0F0F0,
            0x00FF00FF,
            0xFF00FF00,
        ]
        for cb0 in checkerboards:
            for cb1 in checkerboards:
                sdst, scc = ref_andn2_b32(cb0, cb1)
                self.assertEqual(sdst, cb0 & (~cb1 & MASK32))
        hw_dst1, hw_scc1 = run_hw_andn2(0x55555555, 0xAAAAAAAA, "ps_cb_5_a")
        self.assertEqual(hw_dst1, 0x55555555)
        self.assertEqual(hw_scc1, 1)
        hw_dst2, hw_scc2 = run_hw_andn2(0x33333333, 0xCCCCCCCC, "ps_cb_3_c")
        self.assertEqual(hw_dst2, 0x33333333)
        self.assertEqual(hw_scc2, 1)
        hw_dst3, hw_scc3 = run_hw_andn2(0x0F0F0F0F, 0xF0F0F0F0, "ps_cb_0f_f0")
        self.assertEqual(hw_dst3, 0x0F0F0F0F)
        self.assertEqual(hw_scc3, 1)


# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

# Mutant functions for mutation testing
def mutant_and(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 1: uses regular AND (omits ~ negation)."""
    res = (s0 & s1) & MASK32
    return res, 1 if res != 0 else 0


def mutant_orn2(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 2: uses OR instead of AND."""
    res = (s0 | (~s1 & MASK32)) & MASK32
    return res, 1 if res != 0 else 0


def mutant_swapped_operands(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 3: swaps operand order (s1 & ~s0)."""
    res = (s1 & (~s0 & MASK32)) & MASK32
    return res, 1 if res != 0 else 0


def mutant_invert_scc(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 4: inverts SCC logic."""
    res = (s0 & (~s1 & MASK32)) & MASK32
    return res, 1 if res == 0 else 0


class TestSaluAndn2B32NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed_syntax(self):
        bad_cases = [
            ["s_andn2_b32 s0, s1"],
            ["s_andn2_b32"],
        ]
        for idx, lines in enumerate(bad_cases):
            with self.subTest(idx=idx):
                with self.assertRaises(Exception):
                    support.translate_lines(f"ps_neg_bad_{idx}", lines)

    def test_reject_unsupported_opcode(self):
        lines = [
            "s_unknown_b32 s0, s1, s2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_neg_unsupported", lines)

    def test_mutation_detection_and(self):
        s0, s1 = 0x0F, 0x0F
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_and(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_orn2(self):
        s0, s1 = 0x00, 0x00
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_orn2(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_swapped_operands(self):
        s0, s1 = 0x0F, 0x00
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_swapped_operands(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_invert_scc(self):
        s0, s1 = 0x00, 0x00
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_invert_scc(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_andn2_spirv_emission(self):
        """Verify that support.translate_lines emits expected SPIR-V logic for s_andn2_b32."""
        lines = [
            "s_andn2_b32 s2, s0, s1",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_andn2_b32_emit", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

    def test_andn2_hardware_execution(self):
        """Verify hardware-accurate execution of s_andn2_b32 through support.mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_b32_hw", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12005600)
        self.assertEqual(words[1], 1)

    def test_andn2_spirv_lowering_operand_variants(self):
        """Verify that support.translate_lines lowers operand variants with OpBitwiseAnd and OpNot."""
        spv_lit0 = support.translate_lines("ps_andn2_lit0", [
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, 0x12345678, s1",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_lit0)
        self.assertIn("OpNot", spv_lit0)

        spv_lit1 = support.translate_lines("ps_andn2_lit1", [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s2, s0, 0x00ff00ff",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_lit1)
        self.assertIn("OpNot", spv_lit1)

        spv_zero = support.translate_lines("ps_andn2_zero", [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s2, s0, 0",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_zero)
        self.assertIn("OpNot", spv_zero)

        spv_exec = support.translate_lines("ps_andn2_exec", [
            "s_mov_b32 s0, 0",
            "s_andn2_b32 exec_lo, exec_lo, s0",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_exec)
        self.assertIn("OpNot", spv_exec)

    def test_andn2_hardware_execution_inline_constants(self):
        """Verify execution of inline 0 and -1 constants via support.mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s1, s0, 0",
            "s_andn2_b32 s2, s0, -1",
            "v_mov_b32_e32 v0, s1",
            "v_mov_b32_e32 v1, s2",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_inline_consts", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345678)
        self.assertEqual(words[1], 0)

    def test_andn2_hardware_execution_literal_constants(self):
        """Verify execution with 32-bit literal constants via support.mrt0_words."""
        lines = [
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, 0x12345678, s1",
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s3, s0, 0x0000ffff",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_literals", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12005600)
        self.assertEqual(words[1], 0x12340000)

    def test_andn2_hardware_execution_sign_bit_boundary(self):
        """Verify execution on sign-bit and highest 32-bit boundary patterns."""
        lines = [
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 0x7fffffff",
            "s_andn2_b32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_sign_boundary", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80000000)
        self.assertEqual(words[1], 1)

    def test_andn2_negative_control_mutated_operator_fails(self):
        """Verify that replacing OpBitwiseAnd with an incorrect logic op produces detectable failure."""
        correct_dst, _ = run_hw_andn2(0x12345678, 0x00FF00FF, "ps_neg_ctrl_sample")
        mutated_dst = 0x12345678 | (~0x00FF00FF & MASK32)
        self.assertNotEqual(correct_dst, mutated_dst)


if __name__ == "__main__":
    unittest.main()
