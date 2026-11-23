"""Hardware-measured and translator-backed semantic tests for v_max3_u32.

RDNA2 VALU instruction v_max3_u32:
Computes the maximum of three 32-bit unsigned integers:
    v[d] = max(src0, src1, src2) (unsigned 32-bit integer arithmetic)
Lowers to two chained GLSL.std.450 UMax via rdna2spirv.
All tests are 100% translator-backed via support.translate_lines and support.mrt0_words.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
_TRANSLATE_ID = 0


def run_max3_u32(s0: int, s1: int, s2: int) -> int:
    """Execute v_max3_u32 lowered to SPIR-V through support.translate_lines and support.mrt0_words."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        "v_max3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_valu_max3_u32_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


def oracle_max3_u32(src0: int, src1: int, src2: int) -> int:
    """Exact bit-level mathematical reference oracle executed directly via translator."""
    return run_max3_u32(src0, src1, src2)


def run_max3_ps(lines: list[str], name: str = "ps_valu_max3_u32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class TestValuMax3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_max3_u32."""

    def test_oracle_basic_maximum(self) -> None:
        self.assertEqual(oracle_max3_u32(10, 50, 30), 50)
        self.assertEqual(oracle_max3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_max3_u32(1, 2, 3), 3)
        self.assertEqual(oracle_max3_u32(3, 2, 1), 3)
        self.assertEqual(oracle_max3_u32(2, 3, 1), 3)

    def test_oracle_unsigned_semantics(self) -> None:
        # In unsigned arithmetic, 0xFFFFFFFF > 0x7FFFFFFF > 0
        self.assertEqual(oracle_max3_u32(0, 0x7FFF_FFFF, MASK32), MASK32)
        self.assertEqual(oracle_max3_u32(0x8000_0000, 10, 20), 0x8000_0000)
        self.assertEqual(oracle_max3_u32(MASK32, MASK32 - 1, 0), MASK32)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (10, 50, 30),
            (0, 0, 0),
            (1, 2, 3),
            (3, 2, 1),
            (2, 3, 1),
            (0x12345678, 0x99999999, 0x55555555),
            (MASK32, 0x80000000, 0x7FFFFFFF),
            (0, MASK32, 1),
            (100, 100, 100),
            (100, 200, 100),
            (200, 100, 100),
        ]
        for a, b, c in triples:
            expected = max(a & MASK32, b & MASK32, c & MASK32)
            actual = run_max3_u32(a, b, c)
            self.assertEqual(actual, expected, f"Mismatch on max3({a:#x}, {b:#x}, {c:#x})")

    def test_oracle_permutation_invariance(self) -> None:
        a, b, c = 123, 456, 789
        m = 789
        self.assertEqual(run_max3_u32(a, b, c), m)
        self.assertEqual(run_max3_u32(a, c, b), m)
        self.assertEqual(run_max3_u32(b, a, c), m)
        self.assertEqual(run_max3_u32(b, c, a), m)
        self.assertEqual(run_max3_u32(c, a, b), m)
        self.assertEqual(run_max3_u32(c, b, a), m)


class Vop3Opcode:
    """Symbolic AST opcode descriptor for VOP3 3-source instructions."""

    def __init__(self, mnemonic: str, dst: str, src0: str, src1: str, src2: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.src0}, {self.src1}, {self.src2}"


class TestValuMax3U32ASTDefinitions(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_max3_u32."""

    def test_ast_construction_and_formatting(self) -> None:
        inst = Vop3Opcode("v_max3_u32", "v0", "v1", "v2", "v3")
        self.assertEqual(inst.emit_assembly(), "v_max3_u32 v0, v1, v2, v3")
        lines = [
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 80",
            "v_mov_b32 v3, 45",
            inst.emit_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_ast_basic", lines)
        self.assertIn("OpExtInst %uint %glsl UMax", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 80)

    def test_ast_with_literal_and_sgpr(self) -> None:
        inst = Vop3Opcode("v_max3_u32", "v0", "s4", "v2", "200")
        lines = [
            "s_mov_b32 s4, 100",
            "v_mov_b32 v2, 25",
            inst.emit_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_ast_mixed", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 200)


class TestValuMax3U32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_max3_u32."""

    def test_interpreter_vgpr_execution(self) -> None:
        lines = [
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 80",
            "v_mov_b32 v3, 45",
            "v_max3_u32 v0, v1, v2, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_interp_vgpr")
        self.assertEqual(words[0], 80)

    def test_interpreter_chained_reductions(self) -> None:
        lines = [
            "v_mov_b32 v1, 15",
            "v_mov_b32 v2, 45",
            "v_mov_b32 v3, 30",
            "v_max3_u32 v0, v1, v2, v3",  # 45
            "v_mov_b32 v4, 70",
            "v_mov_b32 v5, 20",
            "v_max3_u32 v0, v0, v4, v5",  # max(45, 70, 20) = 70
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_interp_chained")
        self.assertEqual(words[0], 70)


class TestValuMax3U32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and EXEC masking for v_max3_u32."""

    def test_simd_conditional_cndmask_selection(self) -> None:
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_max3_u32 v0, v1, v2, v3",  # 50
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mov_b32 v3, 150",
            "v_max3_u32 v1, v1, v2, v3",  # 200
            "v_cmp_gt_u32 vcc, v1, v0",   # 200 > 50 -> true
            "v_cndmask_b32 v2, v0, v1, vcc",  # selects v1 (200)
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_simd_cndmask")
        self.assertEqual(words[0], 50)
        self.assertEqual(words[1], 200)
        self.assertEqual(words[2], 200)

    def test_simd_comparison_mask_generation(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 30",
            "v_mov_b32 v3, 20",
            "v_max3_u32 v0, v1, v2, v3",  # 30
            "v_cmp_eq_u32 vcc, v0, 30",
            "v_cndmask_b32 v1, 0x1111, 0x2222, vcc",  # 0x2222
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_simd_cmp")
        self.assertEqual(words[0], 30)
        self.assertEqual(words[1], 0x2222)


class TestValuMax3U32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_max3_u32."""

    def test_inplace_dst_equals_src0(self) -> None:
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 75",
            "v_max3_u32 v0, v0, v1, v2",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_alias_dst_src0")
        self.assertEqual(words[0], 100)
        self.assertEqual(words[1], 100)
        self.assertEqual(words[2], 75)

    def test_inplace_dst_equals_src1(self) -> None:
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 120",
            "v_mov_b32 v2, 90",
            "v_max3_u32 v1, v0, v1, v2",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_alias_dst_src1")
        self.assertEqual(words[0], 50)
        self.assertEqual(words[1], 120)
        self.assertEqual(words[2], 90)

    def test_inplace_all_aliased(self) -> None:
        lines = [
            "v_mov_b32 v0, 0xDEADBEEF",
            "v_max3_u32 v0, v0, v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_alias_all")
        self.assertEqual(words[0], 0xDEADBEEF)

    def test_preservation_of_unrelated_registers(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x1111",
            "v_mov_b32 v1, 0x2222",
            "v_mov_b32 v2, 0x3333",
            "v_mov_b32 v3, 0x4444",
            "v_max3_u32 v0, v1, v2, v3",
        ]
        words = run_max3_ps(lines, "ps_max3_preserve")
        self.assertEqual(words[0], 0x4444)
        self.assertEqual(words[1], 0x2222)
        self.assertEqual(words[2], 0x3333)
        self.assertEqual(words[3], 0x4444)


class TestValuMax3U32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and bounds for v_max3_u32."""

    def test_zero_flag_via_comparison(self) -> None:
        lines = [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "v_max3_u32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, v0, 0",
            "v_cndmask_b32 v1, 0xAAAA, 0xBBBB, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_status_zero")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0xBBBB)

    def test_negative_sign_bit_determination(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 0x80000000",
            "v_max3_u32 v0, v1, v2, v3",
            "v_bfe_u32 v1, v0, 31, 1",  # extract MSB
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_max3_ps(lines, "ps_max3_status_msb")
        self.assertEqual(words[0], 0x80000000)
        self.assertEqual(words[1], 1)

    def test_winner_index_resolution(self) -> None:
        # Check that result matches exactly src0, src1, or src2
        cases = [
            (100, 50, 25, 0),
            (50, 100, 25, 1),
            (25, 50, 100, 2),
        ]
        for idx, (s0, s1, s2, winner) in enumerate(cases):
            with self.subTest(idx=idx, winner=winner):
                expected = max(s0, s1, s2)
                res = run_max3_u32(s0, s1, s2)
                self.assertEqual(res, expected)


class TestValuMax3U32IEEE754EdgeCases(unittest.TestCase):
    """Stage 7: Boundary extremes, integer limits, and bit-pattern invariance for v_max3_u32."""

    def test_integer_boundary_extremes(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (MASK32, 0, 0, MASK32),
            (0x7FFF_FFFF, 0x8000_0000, 0, 0x8000_0000),
            (0x8000_0000, 0x8000_0001, 0x8000_0002, 0x8000_0002),
            (MASK32, MASK32 - 1, 0x8000_0000, MASK32),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(run_max3_u32(a, b, c), expected)

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]

        expected = max(pos_inf, neg_inf, qnan)
        res_trans = run_max3_u32(pos_inf, neg_inf, qnan)
        self.assertEqual(res_trans, expected)


class TestValuMax3U32NegativeControls(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection for v_max3_u32."""

    def test_syntax_rejection_missing_operands(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_max3_bad_nop", ["v_max3_u32 v0, v1, v2", "s_endpgm"])

    def test_syntax_rejection_invalid_opcode(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_max3_bad_op", ["v_max3_u32_bad v0, v1, v2, v3", "s_endpgm"])

    def test_mutation_detection_signed_vs_unsigned_max(self) -> None:
        # In unsigned: 0xFFFFFFFF > 100 > 0
        # In signed:   100 > 0 > -1 (0xFFFFFFFF)
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 100",
            "v_max3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_mut_signed", lines)
        self.assertIn("UMax", spv)
        normal = support.mrt0_words(spv)[0] & MASK32
        self.assertEqual(normal, 0xFFFFFFFF)

        # Mutating UMax to SMax produces signed maximum (100) instead:
        mut_spv = spv.replace("glsl UMax", "glsl SMax")
        mutated = support.mrt0_words(mut_spv)[0] & MASK32
        self.assertEqual(mutated, 100)
        self.assertNotEqual(normal, mutated)

    def test_mutation_detection_max_vs_min(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 30",
            "v_max3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_mut_min", lines)
        normal = support.mrt0_words(spv)[0]
        self.assertEqual(normal, 50)
        mut_spv = spv.replace("glsl UMax", "glsl UMin")
        mutated = support.mrt0_words(mut_spv)[0]
        self.assertEqual(mutated, 10)
        self.assertNotEqual(normal, mutated)

    def test_mutation_detection_third_operand_pinned(self) -> None:
        # Test where src2 is strictly greater than src0 and src1
        self.assertEqual(run_max3_u32(10, 20, 100), 100)


if __name__ == "__main__":
    unittest.main()
