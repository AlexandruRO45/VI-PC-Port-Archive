from __future__ import annotations

"""Architectural test suite for AMD RDNA2 v_cvt_f32_i32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
    v_cvt_f32_i32 D, S0

Semantic operation:
    D.f32 = float(S0.i32)

Bit-exact binary32 hardware semantics:
    - Converts 32-bit two's-complement signed integer to IEEE 754 binary32 float.
    - Rounding Mode: IEEE 754 round to nearest even (RNE).
    - Integers in [-2^24, 2^24] are converted with zero precision loss.
    - Values with magnitude > 2^24 round to nearest even representation.
    - VOP3 supports output modifiers: clamp ([0.0, 1.0]), mul:2, div:2.
    - VALU conversions preserve VCC and SCC condition codes.

Standard library Python only (unittest, struct, math, os, sys).
Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import support
import rdna2spirv


def _build_cvt_lines(s0_val: int, mod: str = "") -> list[str]:
    modifier_str = f" {mod}" if mod else ""
    return [
        f"v_mov_b32_e32 v0, {s0_val & 0xFFFFFFFF:#x}",
        f"v_cvt_f32_i32 v1, v0{modifier_str}",
        "v_mov_b32_e32 v2, 0",
        "exp mrt0 v1, v2, v2, v2 done vm",
        "s_endpgm",
    ]


class TestVCvtF32I32BasicConversions(unittest.TestCase):
    """Basic integer to float conversions and boundary extrema."""

    def test_zero_to_float(self):
        """0 converts to +0.0f (0x0000_0000)."""
        lines = _build_cvt_lines(0)
        spv = support.translate_lines("ps_cvt_zero", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x0000_0000)

    def test_positive_integers(self):
        """Positive integers: 1 -> 1.0f, 2 -> 2.0f, 42 -> 42.0f, 100 -> 100.0f."""
        cases = [
            (1, 0x3F80_0000),   # 1.0f
            (2, 0x4000_0000),   # 2.0f
            (42, 0x4228_0000),  # 42.0f
            (100, 0x42C8_0000), # 100.0f
        ]
        for val, expected in cases:
            lines = _build_cvt_lines(val)
            spv = support.translate_lines(f"ps_cvt_pos_{val}", lines)
            out = support.mrt0_words(spv)[0]
            self.assertEqual(out, expected)

    def test_negative_integers(self):
        """Negative integers: -1 -> -1.0f, -2 -> -2.0f, -100 -> -100.0f."""
        cases = [
            (-1, 0xBF80_0000),   # -1.0f
            (-2, 0xC000_0000),   # -2.0f
            (-100, 0xC2C8_0000), # -100.0f
        ]
        for val, expected in cases:
            lines = _build_cvt_lines(val)
            spv = support.translate_lines(f"ps_cvt_neg_{abs(val)}", lines)
            out = support.mrt0_words(spv)[0]
            self.assertEqual(out, expected)

    def test_extrema_int32_limits(self):
        """32-bit signed integer limits: INT_MAX and INT_MIN."""
        # 2147483647 (0x7FFFFFFF) -> 2147483648.0f (0x4F00_0000) due to RNE round-up to 2^31
        lines_max = _build_cvt_lines(0x7FFF_FFFF)
        spv_max = support.translate_lines("ps_cvt_int_max", lines_max)
        self.assertEqual(support.mrt0_words(spv_max)[0], 0x4F00_0000)

        # -2147483648 (0x80000000) -> -2147483648.0f (0xCF00_0000) exact power of 2 (-2^31)
        lines_min = _build_cvt_lines(0x8000_0000)
        spv_min = support.translate_lines("ps_cvt_int_min", lines_min)
        self.assertEqual(support.mrt0_words(spv_min)[0], 0xCF00_0000)


class TestVCvtF32I32RoundingPrecision(unittest.TestCase):
    """IEEE 754 Round-to-Nearest-Even (RNE) behavior at the 24-bit precision boundary."""

    def test_rne_exact_at_2_pow_24(self):
        """2^24 = 16777216 is represented exactly in float32."""
        lines = _build_cvt_lines(16777216)
        spv = support.translate_lines("ps_cvt_rne_2pow24", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4B80_0000)

    def test_rne_round_to_even_at_2_pow_24_plus_1(self):
        """2^24 + 1 = 16777217 is exactly halfway and rounds down to even (16777216.0f)."""
        lines = _build_cvt_lines(16777217)
        spv = support.translate_lines("ps_cvt_rne_halfway_down", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4B80_0000)

    def test_rne_exact_at_2_pow_24_plus_2(self):
        """2^24 + 2 = 16777218 is represented exactly in float32."""
        lines = _build_cvt_lines(16777218)
        spv = support.translate_lines("ps_cvt_rne_exact_plus2", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4B80_0001)

    def test_rne_round_to_even_at_2_pow_24_plus_3(self):
        """2^24 + 3 = 16777219 is exactly halfway and rounds up to even (16777220.0f)."""
        lines = _build_cvt_lines(16777219)
        spv = support.translate_lines("ps_cvt_rne_halfway_up", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4B80_0002)

    def test_rne_negative_at_2_pow_24_plus_1(self):
        """-(2^24 + 1) = -16777217 rounds to even -16777216.0f (0xCB80_0000)."""
        lines = _build_cvt_lines(-16777217)
        spv = support.translate_lines("ps_cvt_rne_neg_halfway", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0xCB80_0000)


class TestVCvtF32I32RegisterOperandsAndAliasing(unittest.TestCase):
    """Register encoding flexibility, operand sourcing, and aliasing."""

    def test_vgpr_source(self):
        """Source operand from VGPR: v_cvt_f32_i32 v0, v1."""
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_cvt_f32_i32 v0, v1",
            "v_mov_b32_e32 v2, 0",
            "exp mrt0 v0, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_vgpr_src", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x42C8_0000)  # 100.0f

    def test_sgpr_source(self):
        """Source operand from SGPR: v_cvt_f32_i32 v0, s0."""
        lines = [
            "s_mov_b32 s0, 10",
            "v_cvt_f32_i32 v0, s0",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_sgpr_src", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4120_0000)  # 10.0f

    def test_inplace_aliasing_dst_src(self):
        """In-place conversion where destination aliases source: v_cvt_f32_i32 v0, v0."""
        lines = [
            "v_mov_b32_e32 v0, 50",
            "v_cvt_f32_i32 v0, v0",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_inplace_alias", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4248_0000)  # 50.0f

    def test_unrelated_registers_preserved(self):
        """Unrelated registers v0 and v2 remain unaltered across v_cvt_f32_i32 v1, v0."""
        lines = [
            "v_mov_b32_e32 v0, 10",
            "v_mov_b32_e32 v2, 30",
            "v_cvt_f32_i32 v1, v0",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_unrelated_reg", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 10)           # v0 unaltered
        self.assertEqual(out[1], 0x4120_0000)  # v1 = 10.0f
        self.assertEqual(out[2], 30)           # v2 unaltered

    def test_sgpr_source_preserved(self):
        """Scalar register s4 remains unaltered after being used as source."""
        lines = [
            "s_mov_b32 s4, 0xFFFFFFE7",  # -25
            "v_cvt_f32_i32 v0, s4",
            "v_mov_b32_e32 v1, s4",
            "v_mov_b32_e32 v2, 0",
            "exp mrt0 v0, v1, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_sgpr_preserve", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xC1C8_0000)  # -25.0f
        self.assertEqual(out[1], 0xFFFF_FFE7)  # s4 preserved


class TestVCvtF32I32ModifiersAndSpirv(unittest.TestCase):
    """VOP3 output modifiers (clamp, mul:2, div:2) and emitted SPIR-V instruction validation."""

    def test_vop3_clamp_modifier_negative(self):
        """Negative value with clamp saturates to 0.0f."""
        lines = _build_cvt_lines(-100, mod="clamp")
        spv = support.translate_lines("ps_cvt_clamp_neg", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x0000_0000)

    def test_vop3_clamp_modifier_positive_saturate(self):
        """Value >= 1.0f with clamp saturates to 1.0f (0x3F80_0000)."""
        lines = _build_cvt_lines(5, mod="clamp")
        spv = support.translate_lines("ps_cvt_clamp_pos", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x3F80_0000)

    def test_vop3_clamp_modifier_in_range(self):
        """0.0f is within [0.0, 1.0] and remains 0.0f."""
        lines = _build_cvt_lines(0, mod="clamp")
        spv = support.translate_lines("ps_cvt_clamp_zero", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x0000_0000)

    def test_vop3_mul2_modifier(self):
        """Conversion with mul:2 modifier: 10 -> 20.0f (0x41A0_0000)."""
        lines = _build_cvt_lines(10, mod="mul:2")
        spv = support.translate_lines("ps_cvt_mul2", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x41A0_0000)

    def test_vop3_div2_modifier(self):
        """Conversion with div:2 modifier: 10 -> 5.0f (0x40A0_0000)."""
        lines = _build_cvt_lines(10, mod="div:2")
        spv = support.translate_lines("ps_cvt_div2", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x40A0_0000)

    def test_spirv_opcode_structure(self):
        """Verify emitted SPIR-V contains OpConvertSToF %float instruction."""
        lines = _build_cvt_lines(42)
        spv = support.translate_lines("ps_cvt_spirv_ops", lines)
        self.assertIn("OpConvertSToF %float", spv)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4228_0000)


class TestVCvtF32I32FlagsAndExecutionControl(unittest.TestCase):
    """Condition codes (VCC, SCC) preservation and conditional EXEC simulation."""

    def test_vcc_flag_preservation(self):
        """VALU conversion does not alter or clobber VCC condition code."""
        lines = [
            "v_mov_b32_e32 v0, 0x3F800000",
            "v_cmp_gt_f32_e32 vcc, 0, v0",  # 0 > 1.0f is False: VCC = 0
            "v_mov_b32_e32 v1, -1",
            "v_cvt_f32_i32 v2, v1",          # v_cvt_f32_i32 executes
            "v_cndmask_b32_e32 v3, 0x1111, 0x2222, vcc",  # Picks 0x1111 because VCC is 0
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_vcc_preserve", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xBF80_0000)  # -1.0f
        self.assertEqual(out[1], 0x1111)       # VCC preserved False

    def test_scc_flag_preservation(self):
        """Scalar condition code (SCC) is preserved across v_cvt_f32_i32."""
        lines = [
            "s_cmp_eq_u32 5, 5",             # SCC = 1
            "v_mov_b32_e32 v0, 10",
            "v_cvt_f32_i32 v1, v0",          # VALU op executes
            "s_cselect_b32 s0, 0xAAAA, 0xBBBB",  # Picks 0xAAAA because SCC is 1
            "v_mov_b32_e32 v2, s0",
            "exp mrt0 v1, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_scc_preserve", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x4120_0000)  # 10.0f
        self.assertEqual(out[1], 0xAAAA)       # SCC preserved

    def test_exec_masking_simulation(self):
        """Simulate divergent execution where active condition selects converted output."""
        lines = [
            "v_mov_b32_e32 v0, 42",
            "v_cmp_gt_u32_e64 vcc, 1, 0",    # Active condition True
            "v_cvt_f32_i32 v1, v0",          # 42 -> 42.0f
            "v_cndmask_b32_e32 v2, 0, v1, vcc",  # Active picks v1
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_exec_sim", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4228_0000)  # 42.0f


class TestVCvtF32I32ReviewerMutationsAndRejection(unittest.TestCase):
    """Pin reviewer mutations (signed vs unsigned conversion) and negative syntax controls."""

    def test_mutation_pin_signed_vs_unsigned_conversion(self):
        """Pin signed OpConvertSToF vs unsigned OpConvertUToF.

        If the translator mistakenly emits OpConvertUToF instead of OpConvertSToF,
        -1 (0xFFFFFFFF = 4294967295) converts to 4294967296.0f (0x4F80_0000).
        With OpConvertSToF, it evaluates correctly to -1.0f (0xBF80_0000).
        """
        lines = _build_cvt_lines(-1)
        spv = support.translate_lines("ps_cvt_pin_signed_neg1", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0xBF80_0000)
        self.assertNotEqual(out, 0x4F80_0000)

    def test_mutation_pin_min_int32_signed_conversion(self):
        """Pin signed conversion on -2147483648 (0x80000000).

        Unsigned conversion would yield +2147483648.0f (0x4F00_0000).
        Signed conversion must yield -2147483648.0f (0xCF00_0000).
        """
        lines = _build_cvt_lines(0x8000_0000)
        spv = support.translate_lines("ps_cvt_pin_min_int32", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0xCF00_0000)
        self.assertNotEqual(out, 0x4F00_0000)

    def test_syntax_rejection_bare_opcode(self):
        """Reject bare opcode without operands."""
        lines = ["v_cvt_f32_i32", "s_endpgm"]
        with self.assertRaises(IndexError):
            support.translate_lines("ps_cvt_bad_bare", lines)

    def test_syntax_rejection_missing_operand(self):
        """Reject incomplete instruction with missing source operand."""
        lines = ["v_cvt_f32_i32 v0", "s_endpgm"]
        with self.assertRaises(IndexError):
            support.translate_lines("ps_cvt_bad_missing", lines)

    def test_syntax_rejection_invalid_opcode(self):
        """Reject unrecognized opcode variant."""
        lines = ["v_cvt_f32_i32_invalid v0, v0", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_cvt_bad_opcode", lines)

    def test_syntax_rejection_invalid_modifier(self):
        """Reject unrecognized output modifier."""
        lines = ["v_cvt_f32_i32 v0, v0 invalid_mod", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_cvt_bad_mod", lines)


if __name__ == "__main__":
    unittest.main()
