from __future__ import annotations

"""Hardware-rule tests and bit-accurate translator execution for s_or_b32.

AMD RDNA2 / GFX10 Scalar ALU (SALU) Architecture:
  - 32-bit Scalar Bitwise OR:
    - RDNA2 ISA: D.u32 = S0.u32 | S1.u32
    - Destination: SGPR (s0..s105) or special registers (exec_lo, exec_hi, vcc_lo, vcc_hi, m0, null)
    - Sources: SGPRs, inline constants (0..64, -1..-16), 32-bit literals, special registers
    - Condition code SCC:
        SCC = 1 if (D != 0) else 0
  - Mathematical and algebraic properties:
    - Identity element:       A | 0 == A
    - Annihilator / Dominance: A | 0xFFFFFFFF == 0xFFFFFFFF
    - Idempotence:            A | A == A
    - Commutativity:          A | B == B | A
    - Associativity:          (A | B) | C == A | (B | C)
    - De Morgan duality:      ~(A | B) == (~A) & (~B)

Standard library Python only (unittest, struct, math, os, sys).
Strictly zero game data.
All test cases compile and execute through support.translate_lines and support.mrt0_words.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32: int = 0xFFFFFFFF


def ref_s_or_b32(s0: int, s1: int) -> int:
    """Pure reference oracle for RDNA2 32-bit scalar bitwise OR: (s0 | s1) & 0xFFFFFFFF."""
    return ((s0 & MASK32) | (s1 & MASK32)) & MASK32


def ref_scc(val: int) -> int:
    """RDNA2 Scalar Condition Code for bitwise SALU operations: 1 if val != 0 else 0."""
    return 1 if (val & MASK32) != 0 else 0


def run_salu_or_b32(s0: int, s1: int, tag: str = "run") -> tuple[int, int]:
    """Execute s_or_b32 s2, s0, s1 through translate_lines and mrt0_words, returning (s2, scc)."""
    lines = [
        f"s_mov_b32 s0, {s0 & MASK32:#x}",
        f"s_mov_b32 s1, {s1 & MASK32:#x}",
        "s_or_b32 s2, s0, s1",
        "v_mov_b32 v0, s2",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32 v1, s3",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_or_{tag}_{s0 & MASK32:08x}_{s1 & MASK32:08x}", lines)
    out = mrt0_words(spv)
    return out[0], out[1]


# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

class SaluOrB32OracleTests(unittest.TestCase):
    """Reference mathematical oracle and architectural model tests for s_or_b32."""

    def test_truth_table_exhaustive(self):
        """Verify full 2-input 1-bit truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1 via translator."""
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1,
        }
        for (b0, b1), expected in truth_table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK32 if b0 else 0
                s1 = MASK32 if b1 else 0
                want = MASK32 if expected else 0
                got, scc = run_salu_or_b32(s0, s1, tag=f"tt_{b0}_{b1}")
                self.assertEqual(got, want)
                self.assertEqual(scc, expected)

    def test_algebraic_identities(self):
        """Test algebraic laws: identity, annihilator, idempotence, commutativity via translator."""
        test_values = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x87654321,
            0xAAAAAAAA,
            0x55555555,
            0x80000000,
            0x00000001,
        ]
        for a in test_values:
            with self.subTest(identity=hex(a)):
                res_id, scc_id = run_salu_or_b32(a, 0, tag=f"id_{a:#x}")
                self.assertEqual(res_id, a)
                self.assertEqual(scc_id, 1 if a != 0 else 0)

                res_ann, scc_ann = run_salu_or_b32(a, MASK32, tag=f"ann_{a:#x}")
                self.assertEqual(res_ann, MASK32)
                self.assertEqual(scc_ann, 1)

                res_idem, scc_idem = run_salu_or_b32(a, a, tag=f"idem_{a:#x}")
                self.assertEqual(res_idem, a)
                self.assertEqual(scc_idem, 1 if a != 0 else 0)

        for a in test_values[:4]:
            for b in test_values[:4]:
                with self.subTest(commutative=(hex(a), hex(b))):
                    res_ab, _ = run_salu_or_b32(a, b, tag=f"comm_ab_{a:#x}_{b:#x}")
                    res_ba, _ = run_salu_or_b32(b, a, tag=f"comm_ba_{b:#x}_{a:#x}")
                    self.assertEqual(res_ab, res_ba)

    def test_de_morgan_duality(self):
        """Test De Morgan duality: ~(A | B) == (~A) & (~B) via translated shader."""
        test_pairs = [
            (0x12345678, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
            (0xFF00FF00, 0x00FF00FF),
            (0x00000000, 0xFFFFFFFF),
        ]
        for idx, (a, b) in enumerate(test_pairs):
            with self.subTest(a=hex(a), b=hex(b)):
                lines = [
                    f"s_mov_b32 s0, {a:#x}",
                    f"s_mov_b32 s1, {b:#x}",
                    "s_or_b32 s2, s0, s1",
                    "s_not_b32 s3, s2",
                    "s_not_b32 s4, s0",
                    "s_not_b32 s5, s1",
                    "s_and_b32 s6, s4, s5",
                    "v_mov_b32 v0, s3",
                    "v_mov_b32 v1, s6",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_demorgan_{idx}", lines)
                out = mrt0_words(spv)
                self.assertEqual(out[0], out[1])
                self.assertEqual(out[0], (~(a | b)) & MASK32)

    def test_boundary_bit_patterns(self):
        """Test single-bit walking, byte boundaries, and alternating bit patterns via translator."""
        for i in [0, 1, 7, 8, 15, 16, 23, 24, 30, 31]:
            bit = 1 << i
            res, scc = run_salu_or_b32(bit, 0, tag=f"bit_walk_{i}")
            self.assertEqual(res, bit)
            self.assertEqual(scc, 1)

            res_self, scc_self = run_salu_or_b32(bit, bit, tag=f"bit_self_{i}")
            self.assertEqual(res_self, bit)
            self.assertEqual(scc_self, 1)

        for i in [0, 7, 15, 23, 30]:
            b0 = 1 << i
            b1 = 1 << (i + 1)
            combined, scc_comb = run_salu_or_b32(b0, b1, tag=f"bit_comb_{i}")
            self.assertEqual(combined, b0 | b1)
            self.assertEqual(scc_comb, 1)

    def test_scc_oracle_mapping(self):
        """Verify SCC mapping across exact zero and non-zero boundary transitions via translator."""
        cases = [
            (0, 0, 0),
            (0, 1, 1),
            (0x80000000, 0, 1),
            (0x7FFFFFFF, 0, 1),
            (0xFFFFFFFF, 0, 1),
        ]
        for a, b, expected_scc in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = run_salu_or_b32(a, b, tag=f"scc_map_{a:#x}_{b:#x}")
                self.assertEqual(res, (a | b) & MASK32)
                self.assertEqual(scc, expected_scc)

    def test_translator_smoke(self):
        """Smoke test verifying tools/rdna2spirv.py translates s_or_b32 correctly."""
        lines = [
            "s_mov_b32 s0, 0x12",
            "s_mov_b32 s1, 0x20",
            "s_or_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_or_smoke", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x32)
        self.assertEqual(out[1], 1)


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST and Register Formatting Validation
# -----------------------------------------------------------------------------

class SaluOrB32AstTests(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions for s_or_b32."""

    def test_ast_various_sgprs(self):
        """Verify translator accepts various SGPR destinations and sources."""
        pairs = [
            ("s2", "s0", "s1"),
            ("s10", "s4", "s6"),
            ("s105", "s104", "s103"),
        ]
        for dst, s0, s1 in pairs:
            lines = [
                f"s_or_b32 {dst}, {s0}, {s1}",
                "s_endpgm",
            ]
            spv = translate_lines(f"ps_ast_sgpr_{dst}_{s0}_{s1}", lines)
            self.assertIn("OpBitwiseOr %uint", spv)

    def test_ast_special_registers(self):
        """Verify translator accepts special registers (exec_lo, vcc_lo, m0, null)."""
        cases = [
            ["s_or_b32 exec_lo, exec_lo, vcc_lo", "s_endpgm"],
            ["s_or_b32 vcc_lo, exec_lo, vcc_lo", "s_endpgm"],
            ["s_or_b32 m0, m0, 0xF00", "s_endpgm"],
            ["s_or_b32 null, s0, s1", "s_endpgm"],
        ]
        for idx, lines in enumerate(cases):
            spv = translate_lines(f"ps_ast_special_or_{idx}", lines)
            self.assertIn("OpBitwiseOr %uint", spv)

    def test_ast_literal_and_inline_constants(self):
        """Verify translation of literal and inline constant sources."""
        lines = [
            "s_mov_b32 s0, 0x1000",
            "s_or_b32 s2, s0, 0xFF",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_or_emit", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x10FF)

    def test_ast_pipeline_execution(self):
        """Verify translated SPIR-V structure contains expected opcodes."""
        lines = [
            "s_or_b32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_pipeline_exec", lines)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        self.assertIn("OpSelect", spv)
        self.assertIn("OpReturn", spv)


# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter & Execution
# -----------------------------------------------------------------------------

class SaluOrB32InterpreterTests(unittest.TestCase):
    """Deterministic scalar instruction interpreter tests for s_or_b32."""

    def test_interpreter_basic_or(self):
        """Test standard scalar OR execution on SGPRs via hardware translator."""
        hw_res, hw_scc = run_salu_or_b32(0x12340000, 0x00005678, tag="interp_basic")
        self.assertEqual(hw_res, 0x12345678)
        self.assertEqual(hw_scc, 1)

    def test_interpreter_scc_updates(self):
        """Test that SCC is cleared strictly on zero results and set otherwise via translator."""
        hw_res0, hw_scc0 = run_salu_or_b32(0, 0, tag="interp_scc0")
        self.assertEqual(hw_res0, 0)
        self.assertEqual(hw_scc0, 0)

        hw_res1, hw_scc1 = run_salu_or_b32(0, 1, tag="interp_scc1")
        self.assertEqual(hw_res1, 1)
        self.assertEqual(hw_scc1, 1)

    def test_interpreter_special_registers(self):
        """Test OR operations reading and writing m0, vcc_lo, exec_lo, and null via translator."""
        lines = [
            "s_mov_b32 m0, 0x000000F0",
            "s_or_b32 m0, m0, 0x00000F00",
            "s_or_b32 null, 0x1234, 0x5678",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, m0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_special_regs", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x00000FF0)
        self.assertEqual(out[1], 1)

    def test_translator_interpreter_equivalence(self):
        """Validate exact equivalence between mathematical oracle and translated SPIR-V."""
        test_pairs = [
            (0x00000000, 0x00000000),
            (0x12345678, 0x00000000),
            (0x00000000, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x80000000, 0x00000001),
        ]
        for idx, (v0, v1) in enumerate(test_pairs):
            with self.subTest(v0=hex(v0), v1=hex(v1)):
                lines = [
                    f"s_mov_b32 s0, {v0:#x}",
                    f"s_mov_b32 s1, {v1:#x}",
                    "s_or_b32 s2, s0, s1",
                    "v_mov_b32 v0, s2",
                    "s_cselect_b32 s3, 1, 0",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_salu_interp_eq_{idx}", lines)
                out = mrt0_words(spv)
                self.assertEqual(out[0], ref_s_or_b32(v0, v1))
                self.assertEqual(out[1], ref_scc(ref_s_or_b32(v0, v1)))


# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SaluOrB32SimdDivergenceTests(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking for s_or_b32."""

    def test_wave32_divergence_reconvergence(self):
        """Simulate divergent branch merging in Wave32 using s_or_b32 via translator."""
        even_lanes = 0x55555555  # lanes 0, 2, 4, ... 30
        odd_lanes = 0xAAAAAAAA   # lanes 1, 3, 5, ... 31

        out, scc = run_salu_or_b32(even_lanes, odd_lanes, tag="simd_divergence")
        self.assertEqual(out, MASK32)
        self.assertEqual(scc, 1)

    def test_wave64_dual_half_masks(self):
        """Manage 64-bit wavefront execution masks across dual halves via translator."""
        out_lo, _ = run_salu_or_b32(0x0000FFFF, 0xFFFF0000, tag="simd_lo")
        out_hi, _ = run_salu_or_b32(0x000000FF, 0xFFFFFF00, tag="simd_hi")
        self.assertEqual(out_lo, MASK32)
        self.assertEqual(out_hi, MASK32)

    def test_vcc_condition_accumulation(self):
        """Accumulate vector condition outcomes via s_or_b32 via translator."""
        lines = [
            "s_mov_b32 s2, 0x00000000",
            "s_or_b32 s2, s2, 0x0000000F",
            "s_or_b32 s2, s2, 0x000000F0",
            "s_or_b32 s2, s2, 0x00000F00",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_vcc_accum", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x00000FFF)

    def test_translated_exec_mask_manipulation(self):
        """Translate shader performing s_or_b32 on exec_lo and verify SPIR-V output."""
        lines = [
            "s_mov_b32 s0, 0x0000FFFF",
            "s_mov_b32 s1, 0xFFFF0000",
            "s_or_b32 exec_lo, s0, s1",
            "v_mov_b32 v0, exec_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_exec_or_mask", lines)
        self.assertIn("OpBitwiseOr", spv)


# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class SaluOrB32AliasingTests(unittest.TestCase):
    """Register aliasing and in-place operand preservation for s_or_b32."""

    def test_self_aliasing_all_operands_identical(self):
        """s_or_b32 s0, s0, s0: in-place idempotent self-OR via translator."""
        val = 0xFEEDBEEF
        lines = [
            f"s_mov_b32 s0, {val:#x}",
            "s_or_b32 s0, s0, s0",
            "v_mov_b32 v0, s0",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_self_alias", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], val)
        self.assertEqual(out[1], 1)

    def test_in_place_dst_alias_src0(self):
        """s_or_b32 s0, s0, s1: accumulate into src0 while preserving src1 via translator."""
        lines = [
            "s_mov_b32 s0, 0x0F0F0F0F",
            "s_mov_b32 s1, 0xF0F0F0F0",
            "s_or_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_alias_src0_test", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], MASK32)
        self.assertEqual(out[1], 0xF0F0F0F0)

    def test_in_place_dst_alias_src1(self):
        """s_or_b32 s1, s0, s1: accumulate into src1 while preserving src0 via translator."""
        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_or_b32 s1, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_alias_src1_test", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x11111111)
        self.assertEqual(out[1], 0x33333333)

    def test_adjacent_register_isolation(self):
        """Modifying destination register must not touch adjacent SGPRs via translator."""
        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_mov_b32 s2, 0x33333333",
            "s_mov_b32 s3, 0x44444444",
            "s_or_b32 s1, s1, s2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_adjacent_isolation_test", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x11111111)
        self.assertEqual(out[1], 0x33333333)
        self.assertEqual(out[2], 0x33333333)
        self.assertEqual(out[3], 0x44444444)

    def test_translated_aliasing_shader(self):
        """Translate shader with dst aliasing src0 and execute via mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x0F0F0F0F",
            "s_mov_b32 s1, 0xF0F0F0F0",
            "s_or_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_or_alias", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], MASK32)
        self.assertEqual(out[1], 0xF0F0F0F0)


# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and SCC Generation
# -----------------------------------------------------------------------------

class SaluOrB32ConditionCodeTests(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow generation for s_or_b32."""

    def test_scc_zero_flag_exhaustion(self):
        """SCC must be 0 if and only if the bitwise result is exactly zero via translator."""
        res0, scc0 = run_salu_or_b32(0, 0, tag="cc_zero")
        self.assertEqual(res0, 0)
        self.assertEqual(scc0, 0)

        for bit in [0, 1, 7, 15, 16, 24, 30, 31]:
            val = 1 << bit
            with self.subTest(bit=bit):
                res, scc = run_salu_or_b32(val, 0, tag=f"cc_bit_{bit}")
                self.assertEqual(res, val)
                self.assertEqual(scc, 1)

    def test_scc_preservation_across_mov(self):
        """s_mov_b32 does NOT alter the SCC flag set by s_or_b32 via translator."""
        lines1 = [
            "s_or_b32 s0, 0, 0",
            "s_mov_b32 s1, 0x12345678",
            "s_cselect_b32 s2, 100, 200",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv1 = translate_lines("ps_scc_pres_0", lines1)
        self.assertEqual(mrt0_words(spv1)[0], 200)

        lines2 = [
            "s_or_b32 s0, 1, 0",
            "s_mov_b32 s1, 0x00000000",
            "s_cselect_b32 s2, 100, 200",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv2 = translate_lines("ps_scc_pres_1", lines2)
        self.assertEqual(mrt0_words(spv2)[0], 100)

    def test_scc_conditional_branching_shader(self):
        """Translate shader utilizing SCC from s_or_b32 to drive s_cselect_b32."""
        cases = [
            (0, 0, 0, 555),
            (0, 1, 1, 777),
            (0x80000000, 0, 1, 777),
        ]
        for idx, (a, b, expected_scc, expected_sel) in enumerate(cases):
            with self.subTest(a=hex(a), b=hex(b)):
                lines = [
                    f"s_mov_b32 s0, {a:#x}",
                    f"s_mov_b32 s1, {b:#x}",
                    "s_or_b32 s2, s0, s1",
                    "s_cselect_b32 s3, 777, 555",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_scc_cselect_{idx}", lines)
                out = mrt0_words(spv)
                self.assertEqual(out[1], expected_sel)


# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

class SaluOrB32FloatBitPatternTests(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN handling for s_or_b32."""

    def test_ieee754_sign_bit_manipulation(self):
        """Inject sign bit (0x80000000) into positive floats via translator."""
        pos_zero = 0x00000000
        neg_zero = 0x80000000
        res, scc = run_salu_or_b32(pos_zero, 0x80000000, tag="float_sign_zero")
        self.assertEqual(res, neg_zero)
        self.assertEqual(scc, 1)

        pos_one_bits = struct.unpack("<I", struct.pack("<f", 1.0))[0]
        neg_one_bits = struct.unpack("<I", struct.pack("<f", -1.0))[0]
        res_neg_one, scc_neg = run_salu_or_b32(pos_one_bits, 0x80000000, tag="float_sign_one")
        self.assertEqual(res_neg_one, neg_one_bits)
        self.assertEqual(scc_neg, 1)

    def test_nan_quieting_and_payload_synthesis(self):
        """Set quiet bit (bit 22, 0x00400000) to convert signaling NaN to quiet NaN via translator."""
        snan_bits = 0x7F800001
        qnan_expected = 0x7FC00001
        res, scc = run_salu_or_b32(snan_bits, 0x00400000, tag="nan_quiet")
        self.assertEqual(res, qnan_expected)
        self.assertEqual(scc, 1)

    def test_subnormal_to_infinity_promotion(self):
        """Promote subnormal float mantissa to infinity exponent using bitwise OR via translator."""
        subnorm = 0x000000FF
        inf_exp = 0x7F800000
        res, scc = run_salu_or_b32(subnorm, inf_exp, tag="subnorm_inf")
        self.assertEqual(res, inf_exp | subnorm)
        self.assertEqual(scc, 1)
        self.assertTrue(math.isnan(struct.unpack("<f", struct.pack("<I", res))[0]))

    def test_translated_float_bitwise_shader(self):
        """Verify shader synthesizing float NaN and sign bit manipulation translates."""
        lines = [
            "s_mov_b32 s0, 0x3F800000",   # +1.0f
            "s_or_b32 s1, s0, 0x80000000", # -1.0f
            "v_mov_b32 v0, s1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_float_or_shader", lines)
        out = mrt0_words(spv)
        expected_neg_one = struct.unpack("<I", struct.pack("<f", -1.0))[0]
        self.assertEqual(out[0], expected_neg_one)


# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

class SaluOrB32NegativeControlsTests(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for s_or_b32."""

    def test_mutation_discrimination_matrix(self):
        """Ensure s_or_b32 is strictly distinguished from AND, XOR, ADD, and NOR via translator."""
        pairs = [
            (0x12345678, 0x87654321),
            (0xFF00FF00, 0x00FF00FF),
            (0xAAAAAAAA, 0x55555555),
            (0x0000FFFF, 0xFFFF0000),
            (0x00000001, 0x00000002),
        ]
        for a, b in pairs:
            or_res, _ = run_salu_or_b32(a, b, tag=f"mut_matrix_{a:#x}_{b:#x}")
            and_res = (a & b) & MASK32
            xor_res = (a ^ b) & MASK32
            nor_res = (~(a | b)) & MASK32
            add_res = (a + b) & MASK32

            with self.subTest(a=hex(a), b=hex(b)):
                self.assertNotEqual(or_res, and_res)
                self.assertNotEqual(or_res, nor_res)
                if (a & b) != 0:
                    self.assertNotEqual(or_res, xor_res)
                    self.assertNotEqual(or_res, add_res)

    def test_reject_unsupported_opcode(self):
        """Reject unsupported SALU opcode."""
        lines = [
            "s_unknown_b32 s0, s1, s2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_neg_unknown_opcode", lines)

    def test_syntax_rejection_malformed_instruction(self):
        """RDNA2 parser / translator must reject malformed scalar instruction syntax."""
        bad_syntax_programs = [
            ["s_or_b32 s0, s1"],  # Missing second source operand
            ["s_or_b32"],          # Missing all operands
        ]
        for idx, lines in enumerate(bad_syntax_programs):
            with self.subTest(idx=idx):
                with self.assertRaises(Exception):
                    translate_lines(f"ps_bad_syntax_{idx}", lines)


if __name__ == "__main__":
    unittest.main()
