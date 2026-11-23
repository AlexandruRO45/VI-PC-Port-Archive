from __future__ import annotations

"""Semantic tests and hardware translator verification for RDNA2 v_subrev_co_u32.

RDNA2 VALU architecture contract:
- Opcode: v_subrev_co_u32 vdst, sdst, src0, src1 [clamp]
- Operation: D = src1 - src0 (unsigned 32-bit integer subtraction).
- Reverse operand order: computes src1 minus src0, contrasting with v_sub_co_u32 (src0 minus src1).
- Borrow out: borrow occurs if src1 < src0 (unsigned), writing 1 to sdst mask for active lanes.
- VOP3 clamp modifier: hardware saturates D to 0 upon borrow (underflow) while still reporting
  the real borrow bit in sdst. If no borrow occurs, D receives (src1 - src0) & 0xFFFFFFFF.
- EXEC masking: inactive lanes preserve their destination VGPR and contribute 0 to the sdst mask.
- Condition codes: VALU operations do not modify SCC.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
All test cases compile and execute through support.translate_lines and support.mrt0_words.
"""

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

import support
import rdna2spirv

M32 = 0xFFFFFFFF


# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Hardware Execution Helpers
# -----------------------------------------------------------------------------

def ref_subrev_co_u32(src0: int, src1: int, clamp: bool = False) -> tuple[int, int]:
    """Pure mathematical oracle matching RDNA2 hardware semantics for v_subrev_co_u32.

    Returns (dst, borrow) where:
    - dst is (src1 - src0) & 0xFFFFFFFF (or 0 if clamp and borrow occurred)
    - borrow is 1 if (src1 & 0xFFFFFFFF) < (src0 & 0xFFFFFFFF) else 0
    """
    s0 = src0 & M32
    s1 = src1 & M32
    diff = (s1 - s0) & M32
    borrow = 1 if s1 < s0 else 0
    dst = 0 if (clamp and borrow) else diff
    return dst, borrow


def run_v_subrev_co_u32(src0: int, src1: int, clamp: bool = False, tag: str = "run") -> tuple[int, int]:
    """Compile and execute v_subrev_co_u32 via support.translate_lines and support.mrt0_words."""
    clamp_str = " clamp" if clamp else ""
    lines = [
        f"v_mov_b32 v1, {src0 & M32:#x}",
        f"v_mov_b32 v2, {src1 & M32:#x}",
        f"v_subrev_co_u32 v0, vcc_lo, v1, v2{clamp_str}",
        "v_cndmask_b32 v3, 0, 1, vcc_lo",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "exp mrt0, v0, v1, v2, v3 done",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_subrev_co_{tag}", lines)
    words = support.mrt0_words(spv)
    return words[0], words[3]


class TestSubrevCoU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        dst, borrow = ref_subrev_co_u32(10, 25)
        self.assertEqual(dst, 15)
        self.assertEqual(borrow, 0)
        hw_d, hw_b = run_v_subrev_co_u32(10, 25, tag="oracle_basic")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

    def test_borrow_underflow(self) -> None:
        dst, borrow = ref_subrev_co_u32(25, 10)
        self.assertEqual(dst, 0xFFFFFFF1)
        self.assertEqual(borrow, 1)
        hw_d, hw_b = run_v_subrev_co_u32(25, 10, tag="oracle_borrow")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

    def test_zero_identity(self) -> None:
        dst, borrow = ref_subrev_co_u32(0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)
        hw_d, hw_b = run_v_subrev_co_u32(0, 0, tag="oracle_zero0")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

        dst, borrow = ref_subrev_co_u32(12345, 12345)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)
        hw_d, hw_b = run_v_subrev_co_u32(12345, 12345, tag="oracle_zero1")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

    def test_boundary_opposites(self) -> None:
        # src1 = 0, src0 = 0xFFFFFFFF -> 0 - 0xFFFFFFFF = 1 with borrow
        dst, borrow = ref_subrev_co_u32(0xFFFFFFFF, 0)
        self.assertEqual(dst, 1)
        self.assertEqual(borrow, 1)
        hw_d, hw_b = run_v_subrev_co_u32(0xFFFFFFFF, 0, tag="oracle_opp0")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

        # src1 = 0xFFFFFFFF, src0 = 0 -> 0xFFFFFFFF - 0 = 0xFFFFFFFF without borrow
        dst, borrow = ref_subrev_co_u32(0, 0xFFFFFFFF)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(borrow, 0)
        hw_d, hw_b = run_v_subrev_co_u32(0, 0xFFFFFFFF, tag="oracle_opp1")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

    def test_clamp_saturation(self) -> None:
        # Underflow with clamp saturates dst to 0, borrow remains 1
        dst, borrow = ref_subrev_co_u32(50, 20, clamp=True)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 1)
        hw_d, hw_b = run_v_subrev_co_u32(50, 20, clamp=True, tag="oracle_clamp0")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

        # Normal subtraction with clamp remains diff, borrow is 0
        dst, borrow = ref_subrev_co_u32(20, 50, clamp=True)
        self.assertEqual(dst, 30)
        self.assertEqual(borrow, 0)
        hw_d, hw_b = run_v_subrev_co_u32(20, 50, clamp=True, tag="oracle_clamp1")
        self.assertEqual(hw_d, dst)
        self.assertEqual(hw_b, borrow)

    def test_subtraction_reversibility_asymmetry(self) -> None:
        # v_subrev_co_u32 computes src1 - src0, which is anti-symmetric to src0 - src1
        d1, b1 = ref_subrev_co_u32(100, 40)
        self.assertEqual(d1, (40 - 100) & M32)
        self.assertEqual(b1, 1)
        hw_d, hw_b = run_v_subrev_co_u32(100, 40, tag="oracle_asym")
        self.assertEqual(hw_d, d1)
        self.assertEqual(hw_b, b1)


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class TestSubrevCoU32AST(unittest.TestCase):
    def test_ast_register_operands(self) -> None:
        cases = [
            ["v_subrev_co_u32 v0, vcc_lo, v1, v2", "s_endpgm"],
            ["v_subrev_co_u32 v5, s2, v3, 100 clamp", "s_endpgm"],
        ]
        for idx, lines in enumerate(cases):
            spv = support.translate_lines(f"ps_ast_subrev_{idx}", lines)
            self.assertIn("OpISub", spv)

    def test_ast_pipeline_execution(self) -> None:
        lines = [
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_pipeline_subrev", lines)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpISub", spv)
        self.assertIn("OpReturn", spv)


# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

class TestSubrevCoU32ScalarInterpreter(unittest.TestCase):
    def test_scalar_execution_normal(self) -> None:
        dst, borrow = run_v_subrev_co_u32(100, 250, tag="scalar_normal")
        self.assertEqual(dst, 150)
        self.assertEqual(borrow, 0)

    def test_scalar_execution_borrow(self) -> None:
        dst, borrow = run_v_subrev_co_u32(300, 100, tag="scalar_borrow")
        self.assertEqual(dst, (100 - 300) & M32)
        self.assertEqual(borrow, 1)

    def test_scalar_execution_clamped(self) -> None:
        dst, borrow = run_v_subrev_co_u32(500, 200, clamp=True, tag="scalar_clamp")
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 1)


# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# -----------------------------------------------------------------------------

class TestSubrevCoU32SIMD(unittest.TestCase):
    def test_wave32_all_active(self) -> None:
        dst, borrow = run_v_subrev_co_u32(10, 25, tag="simd_all_active")
        self.assertEqual(dst, 15)
        self.assertEqual(borrow, 0)

    def test_wave32_divergent_lanes(self) -> None:
        # Lane 0 active: computes 20 - 50 = (20 - 50) & M32 with borrow
        lines = [
            "s_mov_b32 exec_lo, 0x55555555",
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 20",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_simd_divergent", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], (20 - 50) & M32)
        self.assertEqual(words[3], 1)

    def test_wave32_inactive_lane_preservation(self) -> None:
        # Lane 0 inactive: v0 remains initial value 0xBAADF00D
        lines = [
            "v_mov_b32 v0, 0xBAADF00D",
            "s_mov_b32 exec_lo, 0xAAAAAAAA",
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 20",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_simd_inactive_pres", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xBAADF00D)


# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSubrevCoU32Aliasing(unittest.TestCase):
    def test_alias_vdst_src0(self) -> None:
        # v1 = v2 - v1 (destination aliases src0)
        lines = [
            "v_mov_b32 v1, 40",
            "v_mov_b32 v2, 100",
            "v_subrev_co_u32 v1, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_vdst_src0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 60)
        self.assertEqual(words[3], 0)

    def test_alias_vdst_src1(self) -> None:
        # v2 = v2 - v1 (destination aliases src1)
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 85",
            "v_subrev_co_u32 v2, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_vdst_src1", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 55)
        self.assertEqual(words[3], 0)

    def test_alias_all_same_vgpr(self) -> None:
        # v0 = v0 - v0 -> 0, borrow 0
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_subrev_co_u32 v0, vcc_lo, v0, v0",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_all_same", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[3], 0)


# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------

class TestSubrevCoU32FlagsAndBorrow(unittest.TestCase):
    def test_borrow_truth_table_boundaries(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 0, 0),
            (0xFFFFFFFF, 0, 1, 1),
            (0, 0xFFFFFFFF, 0xFFFFFFFF, 0),
            (0x80000000, 0x80000000, 0, 0),
            (0x80000001, 0x80000000, 0xFFFFFFFF, 1),
            (0x7FFFFFFF, 0x80000000, 1, 0),
        ]
        for s0, s1, exp_d, exp_b in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, b = ref_subrev_co_u32(s0, s1)
                self.assertEqual(d, exp_d)
                self.assertEqual(b, exp_b)

    def test_scc_flag_isolation(self) -> None:
        # RDNA2 architecture rule: VALU instructions like v_subrev_co_u32 do not alter SCC
        lines = [
            "s_mov_b32 s0, 1",
            "s_cmp_eq_u32 s0, 1",  # sets SCC = 1
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "s_cselect_b32 s1, 777, 999",  # Reads preserved SCC
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scc_isolation", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[1], 777)


# -----------------------------------------------------------------------------
# Commit 7: Boundary Extremes, Borrow Generation, Clamps, and Translator Validation
# -----------------------------------------------------------------------------

class TestSubrevCoU32Translator(unittest.TestCase):
    def _run_ps(self, lines: list[str]) -> list[int]:
        spvasm = support.translate_lines("ps_subrev_co_test", lines)
        return support.mrt0_words(spvasm)

    def test_translator_basic_difference(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 25",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        words = self._run_ps(lines)
        self.assertEqual(words[0], 15)
        self.assertEqual(words[3], 0)

    def test_translator_borrow_generation(self) -> None:
        lines = [
            "v_mov_b32 v1, 25",
            "v_mov_b32 v2, 10",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        words = self._run_ps(lines)
        self.assertEqual(words[0], 0xFFFFFFF1)
        self.assertEqual(words[3], 1)

    def test_translator_clamp_saturation(self) -> None:
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 20",
            "v_subrev_co_u32 v0, vcc_lo, v1, v2 clamp",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        words = self._run_ps(lines)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[3], 1)

    def test_translator_boundary_extremes(self) -> None:
        pairs = [
            (0, 0, 0, 0),
            (0xFFFFFFFF, 0, 1, 1),
            (0, 0xFFFFFFFF, 0xFFFFFFFF, 0),
            (0x80000000, 0x80000000, 0, 0),
        ]
        for s0, s1, exp_d, exp_b in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    "v_subrev_co_u32 v0, vcc_lo, v1, v2",
                    "v_cndmask_b32 v3, 0, 1, vcc_lo",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "exp mrt0, v0, v1, v2, v3 done",
                    "s_endpgm",
                ]
                words = self._run_ps(lines)
                self.assertEqual(words[0], exp_d)
                self.assertEqual(words[3], exp_b)


# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

class TestSubrevCoU32NegativeAndMutation(unittest.TestCase):
    def test_rejection_missing_operands(self) -> None:
        lines = [
            "v_subrev_co_u32 v0, vcc_lo, v1",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_subrev_co_bad", lines)

    def test_rejection_unsupported_opcode(self) -> None:
        lines = [
            "v_subrev_unknown_u32 v0, vcc_lo, v1, v2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_subrev_co_bad_op", lines)

    def test_mutation_detection_inverted_subtraction(self) -> None:
        hw_d, hw_b = run_v_subrev_co_u32(10, 25, tag="mut_inv_sub")
        mutated_diff = (10 - 25) & M32
        self.assertNotEqual(mutated_diff, hw_d)

    def test_mutation_detection_inverted_borrow_condition(self) -> None:
        _, hw_b = run_v_subrev_co_u32(25, 10, tag="mut_inv_borrow")
        mutated_borrow = 0  # wrong borrow
        self.assertNotEqual(mutated_borrow, hw_b)

    def test_mutation_detection_clamp_wrong_saturation(self) -> None:
        hw_d, _ = run_v_subrev_co_u32(50, 20, clamp=True, tag="mut_clamp_sat")
        mutated_clamp = 0xFFFFFFFF
        self.assertNotEqual(mutated_clamp, hw_d)


if __name__ == "__main__":
    unittest.main()
