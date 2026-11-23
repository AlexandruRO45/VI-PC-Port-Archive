from __future__ import annotations

"""Hardware-rule tests and bit-accurate translator execution for s_or_b64.

RDNA2 Instruction Set Architecture, SOP2:
    s_or_b64 D, S0, S1

Semantic operation:
    D.u64 = S0.u64 | S1.u64
    SCC = (D != 0)

Outputs:
    D: 64-bit destination SGPR pair (or EXEC/VCC) storing bitwise OR of S0 and S1.
    SCC: 1 if destination is non-zero, 0 if all zeros.

Standard library Python only (unittest, struct, os, sys, typing).
Strictly zero game data.
All test cases compile and execute through support.translate_lines and support.mrt0_words.
"""

import os
import struct
import sys
from typing import Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import rdna2spirv

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF


# -----------------------------------------------------------------------------
# Reference Mathematical Oracle and Hardware Execution Helpers
# -----------------------------------------------------------------------------

def s_or_b64_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical oracle for s_or_b64.

    Args:
        s0: 64-bit unsigned integer source 0.
        s1: 64-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (s0 | s1) & MASK64
            scc: 1 if d != 0 else 0
    """
    s0_u64 = s0 & MASK64
    s1_u64 = s1 & MASK64
    d = (s0_u64 | s1_u64) & MASK64
    scc = 1 if d != 0 else 0
    return d, scc


def run_salu_or_b64(s0: int, s1: int, tag: str = "run") -> Tuple[int, int]:
    """Compile and execute s_or_b64 with concrete inputs via support.translate_lines and support.mrt0_words.

    Args:
        s0: 64-bit unsigned integer source 0.
        s1: 64-bit unsigned integer source 1.
        tag: Unique identifier for shader naming.

    Returns:
        (d, scc) where d is 64-bit result and scc is 0 or 1.
    """
    s0_lo = s0 & MASK32
    s0_hi = (s0 >> 32) & MASK32
    s1_lo = s1 & MASK32
    s1_hi = (s1 >> 32) & MASK32

    lines = [
        f"s_mov_b32 s2, {s0_lo:#x}",
        f"s_mov_b32 s3, {s0_hi:#x}",
        f"s_mov_b32 s4, {s1_lo:#x}",
        f"s_mov_b32 s5, {s1_hi:#x}",
        "s_or_b64 s[6:7], s[2:3], s[4:5]",
        "s_cselect_b32 s8, 1, 0",
        "v_mov_b32 v0, s6",
        "v_mov_b32 v1, s7",
        "v_mov_b32 v2, s8",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    name = f"ps_s_or_b64_{tag}"
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    d = (words[1] << 32) | words[0]
    scc = words[2]
    return d, scc


# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Hardware Execution
# -----------------------------------------------------------------------------

class TestSOrB64Oracle(unittest.TestCase):
    """Verify reference oracle matches hardware translator execution across core algebra."""

    def test_oracle_matches_hardware_truth_table_bit_combinations(self):
        cases = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        for idx, (b0, b1) in enumerate(cases):
            with self.subTest(b0=b0, b1=b1):
                exp_d, exp_scc = s_or_b64_oracle(b0, b1)
                hw_d, hw_scc = run_salu_or_b64(b0, b1, tag=f"truth_{idx}")
                self.assertEqual(hw_d, exp_d)
                self.assertEqual(hw_scc, exp_scc)

    def test_oracle_matches_hardware_identity_zero(self):
        # x | 0 == x
        for idx, val in enumerate((0, ALL64, 0x1234_5678_9ABC_DEF0, 0x5555_5555_5555_5555)):
            with self.subTest(val=hex(val)):
                exp_d, exp_scc = s_or_b64_oracle(val, 0)
                hw_d, hw_scc = run_salu_or_b64(val, 0, tag=f"ident_{idx}")
                self.assertEqual(hw_d, exp_d)
                self.assertEqual(hw_scc, exp_scc)

    def test_oracle_matches_hardware_annihilator_all_ones(self):
        # x | ALL64 == ALL64
        for idx, val in enumerate((0, ALL64, 0x1234_5678_9ABC_DEF0, 0xAAAA_AAAA_AAAA_AAAA)):
            with self.subTest(val=hex(val)):
                exp_d, exp_scc = s_or_b64_oracle(val, ALL64)
                hw_d, hw_scc = run_salu_or_b64(val, ALL64, tag=f"annihil_{idx}")
                self.assertEqual(hw_d, exp_d)
                self.assertEqual(hw_scc, exp_scc)

    def test_oracle_matches_hardware_idempotence(self):
        # x | x == x
        for idx, val in enumerate((0, ALL64, 0xCAFE_BABE_DEAD_BEEF)):
            with self.subTest(val=hex(val)):
                exp_d, exp_scc = s_or_b64_oracle(val, val)
                hw_d, hw_scc = run_salu_or_b64(val, val, tag=f"idemp_{idx}")
                self.assertEqual(hw_d, exp_d)
                self.assertEqual(hw_scc, exp_scc)

    def test_oracle_matches_hardware_commutative(self):
        # s0 | s1 == s1 | s0
        s0, s1 = 0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321
        hw_d1, hw_scc1 = run_salu_or_b64(s0, s1, tag="comm_1")
        hw_d2, hw_scc2 = run_salu_or_b64(s1, s0, tag="comm_2")
        exp_d, exp_scc = s_or_b64_oracle(s0, s1)
        self.assertEqual(hw_d1, exp_d)
        self.assertEqual(hw_d2, exp_d)
        self.assertEqual(hw_scc1, exp_scc)
        self.assertEqual(hw_scc2, exp_scc)

    def test_translator_pipeline_execution(self):
        """Verify that translated program contains correct SPIR-V structure and opcodes."""
        lines = [
            "s_mov_b32 s2, 0x55555555",
            "s_mov_b32 s3, 0x55555555",
            "s_mov_b32 s4, 0xAAAAAAAA",
            "s_mov_b32 s5, 0xAAAAAAAA",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_salu_or_b64_spirv_check", lines)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        self.assertIn("OpSelect", spv)
        self.assertIn("OpReturn", spv)


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST and Register Formatting Validation
# -----------------------------------------------------------------------------

class TestSOrB64AST(unittest.TestCase):
    """Verify translator syntax acceptance and AST emission across register operand forms."""

    def test_ast_various_sgpr_pairs(self):
        pairs = [
            ("s[0:1]", "s[2:3]", "s[4:5]"),
            ("s[8:9]", "s[10:11]", "s[12:13]"),
            ("s[16:17]", "s[18:19]", "s[20:21]"),
        ]
        for dst, s0, s1 in pairs:
            lines = [
                f"s_or_b64 {dst}, {s0}, {s1}",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_ast_sgpr_{dst.replace('[','').replace(']','').replace(':','_')}", lines)
            self.assertIn("OpBitwiseOr %uint", spv)

    def test_ast_exec_and_vcc_special_registers(self):
        cases = [
            ["s_or_b64 exec, exec, vcc", "s_endpgm"],
            ["s_or_b64 vcc, exec, vcc", "s_endpgm"],
        ]
        for idx, lines in enumerate(cases):
            spv = support.translate_lines(f"ps_ast_special_{idx}", lines)
            self.assertIn("OpBitwiseOr %uint", spv)

    def test_ast_literal_and_inline_constants(self):
        lines = [
            "s_mov_b32 s4, 0x12345678",
            "s_mov_b32 s5, 0x9abcdef0",
            "s_or_b64 s[6:7], 0, s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_literal_const", lines)
        words = support.mrt0_words(spv)
        res = (words[1] << 32) | words[0]
        self.assertEqual(res, 0x9ABC_DEF0_1234_5678)
        self.assertEqual(words[2], 1)

    def test_ast_null_destination(self):
        lines = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0x00000000",
            "s_mov_b32 s4, 0x00000000",
            "s_mov_b32 s5, 0x00000000",
            "s_or_b64 null, s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_null_dst", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 1)  # SCC set because non-zero even though dst is null


# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter & Execution
# -----------------------------------------------------------------------------

class TestSOrB64ScalarExecution(unittest.TestCase):
    """Verify deterministic scalar instruction lowering and execution."""

    def test_execution_basic_hex_patterns(self):
        s0 = 0x00FF_0000_00FF_0000
        s1 = 0x0000_00FF_0000_00FF
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="exec_basic")
        self.assertEqual(hw_d, 0x00FF_00FF_00FF_00FF)
        self.assertEqual(hw_scc, 1)

    def test_execution_zero_output(self):
        hw_d, hw_scc = run_salu_or_b64(0, 0, tag="exec_zero")
        self.assertEqual(hw_d, 0)
        self.assertEqual(hw_scc, 0)

    def test_execution_high_dword_only(self):
        s0 = 0x1234_5678_0000_0000
        s1 = 0x0000_ABCD_0000_0000
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="exec_hi_only")
        self.assertEqual(hw_d, 0x1234_FFFD_0000_0000)
        self.assertEqual(hw_scc, 1)

    def test_execution_low_dword_only(self):
        s0 = 0x0000_0000_1234_5678
        s1 = 0x0000_0000_0000_ABCD
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="exec_lo_only")
        self.assertEqual(hw_d, 0x0000_0000_1234_FFFD)
        self.assertEqual(hw_scc, 1)

    def test_sequential_chained_or(self):
        """Execute chained s_or_b64 instructions in a single translated shader."""
        lines = [
            "s_mov_b32 s2, 0x00000001",
            "s_mov_b32 s3, 0x00000000",
            "s_mov_b32 s4, 0x00000002",
            "s_mov_b32 s5, 0x00000000",
            "s_mov_b32 s8, 0x00000004",
            "s_mov_b32 s9, 0x00000000",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_or_b64 s[6:7], s[6:7], s[8:9]",
            "s_cselect_b32 s10, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_exec_chained_or", lines)
        words = support.mrt0_words(spv)
        res = (words[1] << 32) | words[0]
        self.assertEqual(res, 0x7)
        self.assertEqual(words[2], 1)


# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 64-Bit Lane Masking
# -----------------------------------------------------------------------------

class TestSOrB64SIMDDivergence(unittest.TestCase):
    """Verify 64-bit lane mask activation and Wave64 multi-lane divergence union."""

    def test_or_lane_union_halves(self):
        # S0 covers lower 32 lanes (0..31), S1 covers upper 32 lanes (32..63)
        s0 = 0x0000_0000_FFFF_FFFF
        s1 = 0xFFFF_FFFF_0000_0000
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="simd_halves")
        self.assertEqual(hw_d, ALL64)
        self.assertEqual(hw_scc, 1)

    def test_checkerboard_lane_union(self):
        # S0 = even lanes, S1 = odd lanes
        s0 = 0x5555_5555_5555_5555
        s1 = 0xAAAA_AAAA_AAAA_AAAA
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="simd_checkerboard")
        self.assertEqual(hw_d, ALL64)
        self.assertEqual(hw_scc, 1)

    def test_sparse_boundary_lanes(self):
        s0 = (1 << 0)
        s1 = (1 << 63)
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="simd_boundary")
        self.assertEqual(hw_d, (1 << 0) | (1 << 63))
        self.assertEqual(hw_scc, 1)

    def test_empty_lane_masks(self):
        hw_d, hw_scc = run_salu_or_b64(0, 0, tag="simd_empty")
        self.assertEqual(hw_d, 0)
        self.assertEqual(hw_scc, 0)


# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSOrB64Aliasing(unittest.TestCase):
    """Verify operand aliasing and non-destructive register handling under translated hardware execution."""

    def test_alias_dst_equals_src0(self):
        """s_or_b64 s[2:3], s[2:3], s[4:5]: D aliases S0."""
        s0_val = 0x1111_1111_1111_1111
        s1_val = 0x3333_3333_3333_3333
        lines = [
            f"s_mov_b32 s2, {s0_val & MASK32:#x}",
            f"s_mov_b32 s3, {(s0_val >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s1_val & MASK32:#x}",
            f"s_mov_b32 s5, {(s1_val >> 32) & MASK32:#x}",
            "s_or_b64 s[2:3], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, s4",
            "v_mov_b32 v3, s5",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_dst_src0", lines)
        words = support.mrt0_words(spv)
        res_d = (words[1] << 32) | words[0]
        res_s1 = (words[3] << 32) | words[2]
        self.assertEqual(res_d, s0_val | s1_val)
        self.assertEqual(res_s1, s1_val)  # S1 strictly preserved

    def test_alias_dst_equals_src1(self):
        """s_or_b64 s[4:5], s[2:3], s[4:5]: D aliases S1."""
        s0_val = 0x00FF_0000_0000_0000
        s1_val = 0x0000_0000_FFFF_0000
        lines = [
            f"s_mov_b32 s2, {s0_val & MASK32:#x}",
            f"s_mov_b32 s3, {(s0_val >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s1_val & MASK32:#x}",
            f"s_mov_b32 s5, {(s1_val >> 32) & MASK32:#x}",
            "s_or_b64 s[4:5], s[2:3], s[4:5]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_dst_src1", lines)
        words = support.mrt0_words(spv)
        res_d = (words[1] << 32) | words[0]
        res_s0 = (words[3] << 32) | words[2]
        self.assertEqual(res_d, s0_val | s1_val)
        self.assertEqual(res_s0, s0_val)  # S0 strictly preserved

    def test_all_operands_aliased(self):
        """s_or_b64 s[2:3], s[2:3], s[2:3]: D, S0, and S1 all share same pair."""
        val = 0x1234_5678_9ABC_DEF0
        lines = [
            f"s_mov_b32 s2, {val & MASK32:#x}",
            f"s_mov_b32 s3, {(val >> 32) & MASK32:#x}",
            "s_or_b64 s[2:3], s[2:3], s[2:3]",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_all_three", lines)
        words = support.mrt0_words(spv)
        res_d = (words[1] << 32) | words[0]
        self.assertEqual(res_d, val)
        self.assertEqual(words[2], 1)


# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and SCC Generation
# -----------------------------------------------------------------------------

class TestSOrB64ConditionCodes(unittest.TestCase):
    """Verify hardware SCC condition code generation and state machine transitions."""

    def test_scc_cleared_strictly_when_zero(self):
        hw_d, hw_scc = run_salu_or_b64(0, 0, tag="scc_zero")
        self.assertEqual(hw_d, 0)
        self.assertEqual(hw_scc, 0)

    def test_scc_set_for_any_single_bit(self):
        for bit in (0, 1, 15, 16, 31, 32, 47, 48, 63):
            with self.subTest(bit=bit):
                hw_d, hw_scc = run_salu_or_b64(1 << bit, 0, tag=f"scc_bit_{bit}")
                self.assertEqual(hw_d, 1 << bit)
                self.assertEqual(hw_scc, 1)

    def test_scc_transitions_state_machine(self):
        """Verify sequential state machine transitions of SCC: 0 -> 0 -> 1 -> 1 -> 0."""
        lines = [
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0",
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
            # Step 1: 0 | 0 = 0 -> SCC = 0
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            # Step 2: 1 | 0 = 1 -> SCC = 1
            "s_mov_b32 s2, 1",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s9, 1, 0",
            # Step 3: ALL64 | 0 = ALL64 -> SCC = 1
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_mov_b32 s3, 0xFFFFFFFF",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s10, 1, 0",
            # Step 4: 0 | 0 = 0 -> SCC = 0
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s11, 1, 0",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scc_transitions_seq", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words, [0, 1, 1, 0])


# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Bit Patterns
# -----------------------------------------------------------------------------

class TestSOrB64IEEE754Bitwise(unittest.TestCase):
    """Verify bit-exact preservation of IEEE 754 float64 patterns under translated execution."""

    FLOAT64_PATTERNS = {
        "pos_zero": 0x0000_0000_0000_0000,
        "neg_zero": 0x8000_0000_0000_0000,
        "pos_inf": 0x7FF0_0000_0000_0000,
        "neg_inf": 0xFFF0_0000_0000_0000,
        "qnan": 0x7FF8_0000_0000_0000,
        "snan": 0x7FF0_0000_0000_0001,
        "min_subnormal": 0x0000_0000_0000_0001,
        "max_subnormal": 0x000F_FFFF_FFFF_FFFF,
        "min_normal": 0x0010_0000_0000_0000,
        "max_normal": 0x7FEF_FFFF_FFFF_FFFF,
        "one_point_zero": 0x3FF0_0000_0000_0000,
    }

    def test_float64_bitwise_or_idempotent(self):
        for name, pattern in self.FLOAT64_PATTERNS.items():
            with self.subTest(pattern_name=name):
                hw_d, hw_scc = run_salu_or_b64(pattern, pattern, tag=f"f64_{name}")
                self.assertEqual(hw_d, pattern)
                self.assertEqual(hw_scc, 1 if pattern != 0 else 0)

    def test_float64_nan_and_inf_combination(self):
        qnan = self.FLOAT64_PATTERNS["qnan"]
        pos_inf = self.FLOAT64_PATTERNS["pos_inf"]
        hw_d, hw_scc = run_salu_or_b64(qnan, pos_inf, tag="f64_qnan_inf")
        self.assertEqual(hw_d, qnan | pos_inf)
        self.assertEqual(hw_scc, 1)

    def test_float64_signed_zero_combination(self):
        pos_zero = self.FLOAT64_PATTERNS["pos_zero"]
        neg_zero = self.FLOAT64_PATTERNS["neg_zero"]
        hw_d, hw_scc = run_salu_or_b64(pos_zero, neg_zero, tag="f64_signed_zero")
        self.assertEqual(hw_d, neg_zero)
        self.assertEqual(hw_scc, 1)


# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

class TestSOrB64NegativeAndMutation(unittest.TestCase):
    """Verify negative syntax controls and mutation detection against translator execution."""

    def test_reject_vgpr_destination(self):
        lines = [
            "s_or_b64 v[0:1], s[2:3], s[4:5]",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_neg_vgpr_dst", lines)

    def test_reject_single_sgpr_destination(self):
        lines = [
            "s_or_b64 s0, s[2:3], s[4:5]",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_neg_single_sgpr_dst", lines)

    def test_reject_missing_operands(self):
        lines = [
            "s_or_b64 s[6:7], s[2:3]",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_missing_op", lines)

    def test_mutation_detector_against_translator(self):
        """Verify that mutations to ALU logic (AND, XOR, ORN2) are caught by the translator output."""
        def mutant_and(s0: int, s1: int):
            d = (s0 & s1) & MASK64
            return d, int(d != 0)

        def mutant_xor(s0: int, s1: int):
            d = (s0 ^ s1) & MASK64
            return d, int(d != 0)

        def mutant_orn2(s0: int, s1: int):
            d = (s0 | (~s1 & MASK64)) & MASK64
            return d, int(d != 0)

        s0 = 0x0000_FFFF_FFFF_0000
        s1 = 0x0000_0000_FFFF_FFFF
        hw_d, hw_scc = run_salu_or_b64(s0, s1, tag="mut_detect")
        self.assertEqual(hw_d, s0 | s1)
        self.assertEqual(hw_scc, 1)

        m_and, _ = mutant_and(s0, s1)
        self.assertNotEqual(m_and, hw_d)

        m_xor, _ = mutant_xor(s0, s1)
        self.assertNotEqual(m_xor, hw_d)

        m_orn2, _ = mutant_orn2(s0, s1)
        self.assertNotEqual(m_orn2, hw_d)


if __name__ == "__main__":
    unittest.main()
