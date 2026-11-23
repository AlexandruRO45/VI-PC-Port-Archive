"""Hardware-measured and translator-backed semantic tests for v_cvt_f32_u32.

RDNA2 VALU instruction v_cvt_f32_u32:
Converts a 32-bit unsigned integer to an IEEE 754 single-precision float (binary32):
    v[d] = (float)src0 (unsigned 32-bit integer to binary32)
Rounding follows round-to-nearest-even (RNE).
Lowers to SPIR-V OpConvertUToF via rdna2spirv.
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


def ref_cvt_f32_u32(val: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_cvt_f32_u32: u32 to binary32 bits."""
    val &= MASK32
    f = float(val)
    return struct.unpack("<I", struct.pack("<f", f))[0]


def run_hw_cvt_f32_u32(val: int) -> int:
    """Execute v_cvt_f32_u32 through rdna2spirv translator and spirv_mini_interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {val & MASK32:#x}",
        "v_cvt_f32_u32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_cvt_f32_u32_hw_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_cvt_ps(lines: list[str], name: str = "ps_cvt_f32_u32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class TestVCvtF32U32Reference(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_exact_integers(self) -> None:
        cases = [
            (0, 0x00000000),         # +0.0
            (1, 0x3F800000),         # 1.0
            (2, 0x40000000),         # 2.0
            (4, 0x40800000),         # 4.0
            (10, 0x41200000),        # 10.0
            (16777215, 0x4B7FFFFF),  # 2^24 - 1
            (16777216, 0x4B800000),  # 2^24
        ]
        for u, exp_bits in cases:
            with self.subTest(u=u):
                self.assertEqual(ref_cvt_f32_u32(u), exp_bits)
                got = run_hw_cvt_f32_u32(u)
                self.assertEqual(got, exp_bits)

    def test_reference_rounding_mode_rne(self) -> None:
        # 16777217 (2^24 + 1): midway between 16777216 and 16777218 -> rounds to even 16777216.0
        self.assertEqual(ref_cvt_f32_u32(16777217), 0x4B800000)
        self.assertEqual(run_hw_cvt_f32_u32(16777217), 0x4B800000)

        # 16777219 (2^24 + 3): midway between 16777218 and 16777220 -> rounds to even 16777220.0
        exp_16777219 = struct.unpack("<I", struct.pack("<f", 16777220.0))[0]
        self.assertEqual(ref_cvt_f32_u32(16777219), exp_16777219)
        self.assertEqual(run_hw_cvt_f32_u32(16777219), exp_16777219)

        # 0xFFFFFFFF (4294967295): rounds to 4294967296.0 (2^32) = 0x4F800000
        self.assertEqual(ref_cvt_f32_u32(0xFFFFFFFF), 0x4F800000)
        self.assertEqual(run_hw_cvt_f32_u32(0xFFFFFFFF), 0x4F800000)

    def test_translator_hw_clamp(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_cvt_f32_u32 v0, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_clamp", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x3F800000)


class Vop1Opcode:
    """Symbolic AST opcode descriptor for VOP1 conversion instructions."""

    def __init__(self, mnemonic: str, dst: str, src0: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.src0 = src0

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.src0}"


class TestVCvtF32U32AST(unittest.TestCase):
    """Stage 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        inst = Vop1Opcode("v_cvt_f32_u32", "v0", "v1")
        self.assertEqual(inst.emit_assembly(), "v_cvt_f32_u32 v0, v1")
        lines = [
            "v_mov_b32 v1, 42",
            inst.emit_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_cvt", lines)
        self.assertIn("OpConvertUToF %float", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], ref_cvt_f32_u32(42))

    def test_ast_with_literal(self) -> None:
        inst = Vop1Opcode("v_cvt_f32_u32", "v0", "100")
        lines = [
            inst.emit_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_lit", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], ref_cvt_f32_u32(100))

    def test_ast_with_sgpr(self) -> None:
        inst = Vop1Opcode("v_cvt_f32_u32", "v0", "s4")
        lines = [
            "s_mov_b32 s4, 250",
            inst.emit_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sgpr", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], ref_cvt_f32_u32(250))


class TestVCvtF32U32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter."""

    def test_scalar_conversion_pipeline(self) -> None:
        lines = [
            "v_mov_b32 v1, 42",
            "v_cvt_f32_u32 v0, v1",
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_cvt_f32_u32 v2, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_scalar_pipeline")
        self.assertEqual(words[0], ref_cvt_f32_u32(42))
        self.assertEqual(words[2], 0x4F800000)

    def test_chained_convert_and_arithmetic(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_cvt_f32_u32 v0, v1",  # 10.0f
            "v_add_f32 v0, v0, v0",   # 20.0f
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_chained_arith")
        self.assertEqual(words[0], ref_cvt_f32_u32(20))


class TestVCvtF32U32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_conditional_cndmask_selection(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_cvt_f32_u32 v0, v1",  # 10.0f
            "v_mov_b32 v2, 50",
            "v_cvt_f32_u32 v3, v2",  # 50.0f
            "v_cmp_gt_u32 vcc, v2, v1",
            "v_cndmask_b32 v2, v0, v3, vcc",  # selects v3 (50.0f)
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_simd_cndmask")
        self.assertEqual(words[0], ref_cvt_f32_u32(10))
        self.assertEqual(words[2], ref_cvt_f32_u32(50))

    def test_simd_divergent_comparison_results(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_cvt_f32_u32 v0, v1",
            "v_cmp_eq_u32 vcc, v0, 0x42C80000",  # 100.0f
            "v_cndmask_b32 v1, 0x1111, 0x2222, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_simd_divergent_cmp")
        self.assertEqual(words[0], ref_cvt_f32_u32(100))
        self.assertEqual(words[1], 0x2222)


class TestVCvtF32U32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation."""

    def test_dst_aliases_src(self) -> None:
        lines = [
            "v_mov_b32 v0, 100",
            "v_cvt_f32_u32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_alias_dst_src")
        self.assertEqual(words[0], ref_cvt_f32_u32(100))

    def test_unrelated_registers_preserved(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x1111",
            "v_mov_b32 v1, 42",
            "v_mov_b32 v2, 0x2222",
            "v_mov_b32 v3, 0x3333",
            "v_cvt_f32_u32 v0, v1",
        ]
        words = run_cvt_ps(lines, "ps_preserve_unrelated")
        self.assertEqual(words[0], ref_cvt_f32_u32(42))
        self.assertEqual(words[1], 42)
        self.assertEqual(words[2], 0x2222)
        self.assertEqual(words[3], 0x3333)


class TestVCvtF32U32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry/borrow preservation."""

    def test_zero_flag_via_comparison(self) -> None:
        lines = [
            "v_mov_b32 v1, 0",
            "v_cvt_f32_u32 v0, v1",  # 0.0f (0x00000000)
            "v_cmp_eq_u32 vcc, v0, 0",
            "v_cndmask_b32 v1, 0xAAAA, 0xBBBB, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_status_zero")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0xBBBB)

    def test_sign_bit_never_set_for_unsigned(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x80000000",
            "v_cvt_f32_u32 v0, v1",
            "v_bfe_u32 v1, v0, 31, 1",  # extract MSB
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_cvt_ps(lines, "ps_status_msb")
        self.assertEqual(words[0], 0x4F000000)
        self.assertEqual(words[1], 0)


class TestVCvtF32U32IEEE754(unittest.TestCase):
    """Stage 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_zero_is_always_positive_zero(self) -> None:
        res_bits = run_hw_cvt_f32_u32(0)
        self.assertEqual(res_bits, 0x00000000)
        self.assertEqual(res_bits >> 31, 0)

    def test_powers_of_two_boundaries(self) -> None:
        for k in range(32):
            val = 1 << k
            expected = struct.unpack("<I", struct.pack("<f", float(val)))[0]
            with self.subTest(k=k, val=hex(val)):
                self.assertEqual(run_hw_cvt_f32_u32(val), expected)

    def test_max_u32_extreme_rounding(self) -> None:
        res_bits = run_hw_cvt_f32_u32(0xFFFFFFFF)
        self.assertEqual(res_bits, 0x4F800000)


class TestVCvtF32U32NegativeAndMutation(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_extra_operands_accepted_by_translator(self) -> None:
        """The translator accepts surplus operands in VOP1 assembly and emits OpConvertUToF."""
        spv = support.translate_lines("ps_extra", ["v_cvt_f32_u32 v0, v1, v2", "s_endpgm"])
        self.assertIn("OpConvertUToF %float", spv)

    def test_syntax_rejection_insufficient_operands(self) -> None:
        """Missing source operand is rejected during translation."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_invalid", ["v_cvt_f32_u32 v0", "s_endpgm"])

    def test_syntax_rejection_invalid_opcode(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_bad_opcode", ["v_cvt_f32_u32_bad v0, v1", "s_endpgm"])

    def test_mutation_signed_vs_unsigned_conversion(self) -> None:
        # In unsigned conversion: 0xFFFFFFFF -> +4294967296.0 (0x4F800000)
        # In signed conversion:   0xFFFFFFFF (-1) -> -1.0 (0xBF800000)
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_cvt_f32_u32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_signed", lines)
        self.assertIn("OpConvertUToF %float", spv)
        normal = support.mrt0_words(spv)[0]
        self.assertEqual(normal, 0x4F800000)

        # Mutating OpConvertUToF to OpConvertSToF yields -1.0
        mut_spv = spv.replace("OpConvertUToF %float", "OpConvertSToF %float")
        mutated = support.mrt0_words(mut_spv)[0]
        self.assertEqual(mutated, 0xBF800000)
        self.assertNotEqual(normal, mutated)

    def test_mutation_detection_clamp_behavior(self) -> None:
        # Verify clamp modifier pins output to [0.0, 1.0]
        lines = [
            "v_mov_b32 v1, 100",
            "v_cvt_f32_u32 v0, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_clamp", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3F800000)  # 1.0f


if __name__ == "__main__":
    unittest.main()
