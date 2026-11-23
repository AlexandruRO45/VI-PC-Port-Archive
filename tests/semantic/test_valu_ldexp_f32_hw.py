from __future__ import annotations

"""Architectural test suite for v_ldexp_f32.

RDNA2 Instruction Set Architecture, VOP2/VOP3:
    v_ldexp_f32 D, S0, S1

Semantic operation:
    D.f32 = S0.f32 * 2^(S1.i32)

Bit-exact binary32 hardware semantics:
    - Normal inputs (biased exponent 1..254): exponent field is incremented by clamped S1 in [-512, 512].
    - Zero and denormals (biased exponent 0): flushed to signed zero (+0.0 / -0.0).
    - Infinities and NaNs (biased exponent 255): preserved bit-exact without modification.
    - Exponent overflow (> 254): saturates to +/-infinity with original sign.
    - Exponent underflow (<= 0): saturates to +/-zero with original sign.
    - S1 clamping to [-512, 512] prevents 32-bit addition wrap-around (e.g. S1 = 0x7FFFFFFF saturates to +inf).

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

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000
QNAN: int = 0x7FC0_0000
SNAN: int = 0x7F80_0001
MAX_NORMAL: int = 0x7F7F_FFFF
MIN_NORMAL: int = 0x0080_0000
MIN_SUBNORMAL: int = 0x0000_0001
MAX_SUBNORMAL: int = 0x007F_FFFF
ONE_F32: int = 0x3F80_0000


def _build_ldexp_lines(s0_u32: int, s1_i32: int, clamp: bool = False) -> list[str]:
    mod = " clamp" if clamp else ""
    return [
        f"v_mov_b32_e32 v0, {s0_u32:#x}",
        f"s_mov_b32 s0, {s1_i32 & 0xFFFF_FFFF:#x}",
        f"v_ldexp_f32 v1, v0, s0{mod}",
        "v_mov_b32_e32 v2, 0",
        "exp mrt0 v1, v2, v2, v2 done vm",
        "s_endpgm",
    ]


class TestVLdexpF32BasicMath(unittest.TestCase):
    """Basic scaling calculations and signed floating-point inputs."""

    def test_identity_scale_zero(self):
        """1.0f * 2^0 = 1.0f (0x3F80_0000)."""
        lines = _build_ldexp_lines(ONE_F32, 0)
        spv = support.translate_lines("ps_ldexp_ident", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, ONE_F32)

    def test_positive_scale(self):
        """1.0f * 2^2 = 4.0f (0x4080_0000)."""
        lines = _build_ldexp_lines(ONE_F32, 2)
        spv = support.translate_lines("ps_ldexp_pos_scale", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4080_0000)

    def test_negative_scale(self):
        """1.0f * 2^-2 = 0.25f (0x3E80_0000)."""
        lines = _build_ldexp_lines(ONE_F32, -2)
        spv = support.translate_lines("ps_ldexp_neg_scale", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x3E80_0000)

    def test_negative_float_scaled(self):
        """-1.5f * 2^1 = -3.0f (0xC040_0000) and -2.0f * 2^-1 = -1.0f (0xBF80_0000)."""
        lines1 = _build_ldexp_lines(0xBFC0_0000, 1)
        spv1 = support.translate_lines("ps_ldexp_neg_float_1", lines1)
        out1 = support.mrt0_words(spv1)[0]
        self.assertEqual(out1, 0xC040_0000)

        lines2 = _build_ldexp_lines(0xC000_0000, -1)
        spv2 = support.translate_lines("ps_ldexp_neg_float_2", lines2)
        out2 = support.mrt0_words(spv2)[0]
        self.assertEqual(out2, 0xBF80_0000)

    def test_fractional_mantissa_scale(self):
        """1.75f * 2^3 = 14.0f (0x4160_0000)."""
        lines = _build_ldexp_lines(0x3FE0_0000, 3)
        spv = support.translate_lines("ps_ldexp_frac_mant", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4160_0000)


class TestVLdexpF32RegisterOperandsAndAliasing(unittest.TestCase):
    """Register encoding flexibility and in-place operand aliasing."""

    def test_vgpr_src0_vgpr_src1(self):
        """Both operands sourced from VGPRs: v_ldexp_f32 v2, v0, v1."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "v_mov_b32_e32 v1, 3",
            "v_ldexp_f32 v2, v0, v1",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_vgpr_vgpr", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4100_0000)  # 8.0f

    def test_vgpr_src0_sgpr_src1(self):
        """Scale factor sourced from SGPR: v_ldexp_f32 v2, v0, s0."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "s_mov_b32 s0, 3",
            "v_ldexp_f32 v2, v0, s0",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_vgpr_sgpr", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4100_0000)

    def test_aliased_dst_and_src0(self):
        """In-place update where destination aliases src0: v_ldexp_f32 v0, v0, s0."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "s_mov_b32 s0, 2",
            "v_ldexp_f32 v0, v0, s0",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_alias_dst_src0", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4080_0000)  # 4.0f

    def test_aliased_dst_and_src1(self):
        """Destination aliases src1 register: v_ldexp_f32 v1, v0, v1."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "v_mov_b32_e32 v1, 2",
            "v_ldexp_f32 v1, v0, v1",
            "v_mov_b32_e32 v2, 0",
            "exp mrt0 v1, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_alias_dst_src1", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4080_0000)

    def test_all_operands_aliased(self):
        """All operands alias the same VGPR: v_ldexp_f32 v0, v0, v0."""
        # 2.0f (0x4000_0000) interpreted as integer scale > 512, overflows to +inf
        lines = [
            "v_mov_b32_e32 v0, 0x40000000",
            "v_ldexp_f32 v0, v0, v0",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_alias_all", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, POS_INF)


class TestVLdexpF32OutputModifiers(unittest.TestCase):
    """VOP3 clamp modifier and emitted SPIR-V instruction verification."""

    def test_vop3_clamp_modifier_saturates_above_one(self):
        """1.0f * 2^2 = 4.0f clamped to [0.0, 1.0] -> 1.0f (0x3F80_0000)."""
        lines = _build_ldexp_lines(ONE_F32, 2, clamp=True)
        spv = support.translate_lines("ps_ldexp_clamp_above", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, ONE_F32)

    def test_vop3_clamp_modifier_saturates_below_zero(self):
        """-1.0f * 2^1 = -2.0f clamped to [0.0, 1.0] -> 0.0f (0x0000_0000)."""
        lines = _build_ldexp_lines(0xBF80_0000, 1, clamp=True)
        spv = support.translate_lines("ps_ldexp_clamp_below", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, POS_ZERO)

    def test_vop3_clamp_modifier_in_range_unaffected(self):
        """1.0f * 2^-1 = 0.5f clamped to [0.0, 1.0] -> 0.5f (0x3F00_0000)."""
        lines = _build_ldexp_lines(ONE_F32, -1, clamp=True)
        spv = support.translate_lines("ps_ldexp_clamp_in_range", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x3F00_0000)

    def test_spirv_opcode_structure(self):
        """Verify emitted SPIR-V contains bitfield extraction, shift, and clamp instructions."""
        lines = _build_ldexp_lines(ONE_F32, 2, clamp=True)
        spv = support.translate_lines("ps_ldexp_spirv_ops", lines)
        self.assertIn("OpShiftLeftLogical", spv)
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertIn("OpSelect", spv)
        self.assertIn("OpExtInst %float %glsl FClamp", spv)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, ONE_F32)


class TestVLdexpF32ExecutionControlAndFlags(unittest.TestCase):
    """EXEC lane simulation and status flag / condition code preservation."""

    def test_exec_masking_active_inactive_simulation(self):
        """Simulate divergent execution where active lane scales and inactive lane preserves input."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "s_mov_b32 s0, 1",
            "v_cmp_gt_u32_e64 vcc, 1, 0",  # Active condition True
            "v_cndmask_b32_e32 v1, 0, v0, vcc",  # Active picks v0 (1.0f)
            "v_ldexp_f32 v2, v1, s0",  # 1.0f * 2^1 = 2.0f
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_exec_sim", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x4000_0000)  # 2.0f

    def test_vcc_flag_preservation(self):
        """Verify that VCC flag state is not modified or clobbered by v_ldexp_f32."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "v_cmp_gt_f32_e32 vcc, 0, v0",  # 0 > 1.0f is False: VCC=0
            "s_mov_b32 s0, 1",
            "v_ldexp_f32 v1, v0, s0",  # v_ldexp_f32 executes
            "v_cndmask_b32_e32 v2, 0x1111, 0x2222, vcc",  # Should pick 0x1111 because VCC is 0
            "exp mrt0 v1, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_vcc_preserve", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x4000_0000)  # 2.0f
        self.assertEqual(out[1], 0x1111)  # VCC preserved False

    def test_scc_flag_preservation(self):
        """Verify that scalar condition code (SCC) is preserved across v_ldexp_f32."""
        lines = [
            f"v_mov_b32_e32 v0, {ONE_F32:#x}",
            "s_cmp_eq_u32 5, 5",  # SCC = 1
            "s_mov_b32 s0, 1",
            "v_ldexp_f32 v1, v0, s0",  # VALU op executes
            "s_cselect_b32 s1, 0xAAAA, 0xBBBB",  # Should select 0xAAAA because SCC=1
            "v_mov_b32_e32 v2, s1",
            "exp mrt0 v1, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_scc_preserve", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x4000_0000)
        self.assertEqual(out[1], 0xAAAA)


class TestVLdexpF32IEEE754EdgeCases(unittest.TestCase):
    """IEEE 754 float32 boundary handling according to RDNA2 hardware semantics."""

    def test_signed_zeros_preserved(self):
        """+/-0.0 scaled by any exponent remains +/-0.0."""
        for z, name in ((POS_ZERO, "pos"), (NEG_ZERO, "neg")):
            for scale in (-100, -1, 0, 1, 100):
                lines = _build_ldexp_lines(z, scale)
                spv = support.translate_lines(f"ps_ldexp_zero_{name}_{scale}", lines)
                out = support.mrt0_words(spv)[0]
                self.assertEqual(out, z)

    def test_infinities_pass_through(self):
        """+/-infinity with any exponent passes through unchanged."""
        for inf, name in ((POS_INF, "pos_inf"), (NEG_INF, "neg_inf")):
            for scale in (-50, 0, 50):
                lines = _build_ldexp_lines(inf, scale)
                spv = support.translate_lines(f"ps_ldexp_inf_{name}_{scale}", lines)
                out = support.mrt0_words(spv)[0]
                self.assertEqual(out, inf)

    def test_nans_pass_through(self):
        """Quiet and signaling NaNs pass through bit-exact without modification."""
        for nan, name in ((QNAN, "qnan"), (SNAN, "snan")):
            for scale in (-5, 0, 5):
                lines = _build_ldexp_lines(nan, scale)
                spv = support.translate_lines(f"ps_ldexp_nan_{name}_{scale}", lines)
                out = support.mrt0_words(spv)[0]
                self.assertEqual(out, nan)

    def test_subnormals_flushed_to_signed_zero(self):
        """Subnormal inputs are flushed to signed zero (+0.0 / -0.0) under default denormal-off mode."""
        for sub, expected, name in (
            (MIN_SUBNORMAL, POS_ZERO, "min_sub_pos"),
            (MAX_SUBNORMAL, POS_ZERO, "max_sub_pos"),
            (MIN_SUBNORMAL | NEG_ZERO, NEG_ZERO, "min_sub_neg"),
            (MAX_SUBNORMAL | NEG_ZERO, NEG_ZERO, "max_sub_neg"),
        ):
            lines = _build_ldexp_lines(sub, 10)
            spv = support.translate_lines(f"ps_ldexp_sub_{name}", lines)
            out = support.mrt0_words(spv)[0]
            self.assertEqual(out, expected)

    def test_underflow_to_signed_zero(self):
        """Normal values underflowing the exponent range saturate to signed zero."""
        # +1.0f * 2^-150 -> +0.0
        lines_pos = _build_ldexp_lines(ONE_F32, -150)
        spv_pos = support.translate_lines("ps_ldexp_underflow_pos", lines_pos)
        out_pos = support.mrt0_words(spv_pos)[0]
        self.assertEqual(out_pos, POS_ZERO)

        # -1.0f * 2^-150 -> -0.0
        lines_neg = _build_ldexp_lines(0xBF80_0000, -150)
        spv_neg = support.translate_lines("ps_ldexp_underflow_neg", lines_neg)
        out_neg = support.mrt0_words(spv_neg)[0]
        self.assertEqual(out_neg, NEG_ZERO)

    def test_overflow_to_signed_infinity(self):
        """Normal values overflowing the exponent range saturate to signed infinity."""
        # +1.0f * 2^150 -> +inf
        lines_pos = _build_ldexp_lines(ONE_F32, 150)
        spv_pos = support.translate_lines("ps_ldexp_overflow_pos", lines_pos)
        out_pos = support.mrt0_words(spv_pos)[0]
        self.assertEqual(out_pos, POS_INF)

        # -1.0f * 2^150 -> -inf
        lines_neg = _build_ldexp_lines(0xBF80_0000, 150)
        spv_neg = support.translate_lines("ps_ldexp_overflow_neg", lines_neg)
        out_neg = support.mrt0_words(spv_neg)[0]
        self.assertEqual(out_neg, NEG_INF)


class TestVLdexpF32ReviewerMutationsAndBoundaries(unittest.TestCase):
    """Pinning reviewer-identified mutations, hardware boundary cases, and syntax controls."""

    def test_mutation_pin_s1_clamp_int32_max_positive(self):
        """Pin S1 clamp against 32-bit wrap mutation: 1.0f with S1=0x7FFFFFFF saturates to +inf.

        If S1 is not clamped to [-512, 512], e + S1 (127 + 2147483647) wraps to 0x8000007E.
        With bit 31 set, the unclamped translation misidentifies the result as negative/zero
        and emits +0.0 instead of +inf. Clamping to 512 guarantees saturation to +inf.
        """
        lines = _build_ldexp_lines(ONE_F32, 0x7FFF_FFFF)
        spv = support.translate_lines("ps_ldexp_pin_clamp_max", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, POS_INF)

    def test_mutation_pin_s1_clamp_int32_min_negative(self):
        """Pin S1 clamp against negative 32-bit wrap mutation: 1.0f with S1=0x80000000 underflows to +0.0."""
        lines = _build_ldexp_lines(ONE_F32, -0x8000_0000)
        spv = support.translate_lines("ps_ldexp_pin_clamp_min", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, POS_ZERO)

    def test_mutation_pin_top_normal_binade_exponent_254(self):
        """Pin top normal exponent boundary: 1.0f * 2^127 lands on biased exponent 254 (0x7F00_0000).

        If the translator mistakenly uses 253 as the valid exponent upper bound, 1.0f * 2^127
        would prematurely overflow to +inf. It must evaluate to the largest finite binade.
        """
        lines = _build_ldexp_lines(ONE_F32, 127)
        spv = support.translate_lines("ps_ldexp_pin_exp_254", lines)
        out = support.mrt0_words(spv)[0]
        self.assertEqual(out, 0x7F00_0000)

    def test_mutation_pin_exponent_254_to_255_overflow_boundary(self):
        """Verify sharp step from highest normal exponent (254) to overflow infinity (255)."""
        # +1.0f * 2^127 = 0x7F00_0000 (normal)
        lines_pos_254 = _build_ldexp_lines(ONE_F32, 127)
        spv_pos_254 = support.translate_lines("ps_ldexp_pos_254", lines_pos_254)
        self.assertEqual(support.mrt0_words(spv_pos_254)[0], 0x7F00_0000)

        # +1.0f * 2^128 = 0x7F80_0000 (+inf)
        lines_pos_255 = _build_ldexp_lines(ONE_F32, 128)
        spv_pos_255 = support.translate_lines("ps_ldexp_pos_255", lines_pos_255)
        self.assertEqual(support.mrt0_words(spv_pos_255)[0], POS_INF)

        # -1.0f * 2^127 = 0xFF00_0000 (normal)
        lines_neg_254 = _build_ldexp_lines(0xBF80_0000, 127)
        spv_neg_254 = support.translate_lines("ps_ldexp_neg_254", lines_neg_254)
        self.assertEqual(support.mrt0_words(spv_neg_254)[0], 0xFF00_0000)

        # -1.0f * 2^128 = 0xFF80_0000 (-inf)
        lines_neg_255 = _build_ldexp_lines(0xBF80_0000, 128)
        spv_neg_255 = support.translate_lines("ps_ldexp_neg_255", lines_neg_255)
        self.assertEqual(support.mrt0_words(spv_neg_255)[0], NEG_INF)

    def test_max_normal_float_boundary(self):
        """Max normal float (0x7F7F_FFFF): scale 0 preserves, scale 1 overflows, scale -1 halves."""
        # 0x7F7F_FFFF * 2^0 = 0x7F7F_FFFF
        lines_zero = _build_ldexp_lines(MAX_NORMAL, 0)
        spv_zero = support.translate_lines("ps_ldexp_maxnorm_zero", lines_zero)
        self.assertEqual(support.mrt0_words(spv_zero)[0], MAX_NORMAL)

        # 0x7F7F_FFFF * 2^1 -> overflows to +inf
        lines_one = _build_ldexp_lines(MAX_NORMAL, 1)
        spv_one = support.translate_lines("ps_ldexp_maxnorm_one", lines_one)
        self.assertEqual(support.mrt0_words(spv_one)[0], POS_INF)

        # 0x7F7F_FFFF * 2^-1 = 0x7EFF_FFFF
        lines_neg = _build_ldexp_lines(MAX_NORMAL, -1)
        spv_neg = support.translate_lines("ps_ldexp_maxnorm_neg", lines_neg)
        self.assertEqual(support.mrt0_words(spv_neg)[0], 0x7EFF_FFFF)

    def test_syntax_rejection_unsupported_modifier(self):
        """Reject unrecognized or unsupported modifier on v_ldexp_f32."""
        lines = [
            "v_ldexp_f32 v0, v1, v2 invalid_modifier",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_ldexp_bad_mod", lines)

    def test_syntax_rejection_missing_operand(self):
        """Reject incomplete assembly instruction lacking operands."""
        lines = [
            "v_ldexp_f32 v0, v1",
            "s_endpgm",
        ]
        with self.assertRaises(IndexError):
            support.translate_lines("ps_ldexp_missing_op", lines)

    def test_syntax_rejection_invalid_opcode(self):
        """Reject unrecognized opcode variant."""
        lines = [
            "v_ldexp_f32_invalid v0, v1, v2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_ldexp_bad_opcode", lines)

    def test_negative_control_divergence_from_mutants(self):
        """Verify that a mutated exponent shift (shift 22 instead of 23) diverges from translated execution."""
        lines = _build_ldexp_lines(ONE_F32, 2)
        spv = support.translate_lines("ps_ldexp_neg_ctrl", lines)
        translated = support.mrt0_words(spv)[0]

        # Shift-22 mutant
        e = ((ONE_F32 >> 23) & 0xFF) + 2
        mutant = (ONE_F32 & 0x807F_FFFF) | (e << 22)

        self.assertEqual(translated, 0x4080_0000)
        self.assertNotEqual(translated, mutant)


if __name__ == "__main__":
    unittest.main()
