from __future__ import annotations

"""Hardware-measured semantic tests for RDNA2 v_cvt_f16_f32.

Tests the vector 32-bit float to 16-bit half-precision float conversion:
    v_cvt_f16_f32 vdst, vsrc
Lowered semantics:
    Converts a binary32 float in src to a binary16 half-precision float in dst[15:0]
    using IEEE-754 round to nearest, ties to even (RNE).
    On GFX10 / RDNA2, this is a 16-bit partial write: bits [31:16] of destination
    VGPR are strictly preserved.

VOP3 modifiers:
    - Source modifiers: absolute value (|src|), negation (-src), negated abs (-|src|).
    - Output modifiers: omod (mul:2, div:2) and clamp ([0.0, 1.0]).
    - Execution order: round, omod, clamp, preserving the destination high half.

Translated and verified through tools/rdna2spirv.py via tests/semantic/support.py.
Standard library Python only: unittest, struct, math. Zero game data.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

OLD_HIGH = 0xDEAD0000


def run_cvt(
    src_bits: int,
    old_dst: int = OLD_HIGH,
    opcode: str = "v_cvt_f16_f32",
    src_operand: str = "v1",
    mod_suffix: str = "",
) -> int:
    """Translate and run a fragment executing a single v_cvt_f16_f32 conversion."""
    suffix = f" {mod_suffix}" if mod_suffix else ""
    lines = [
        f"v_mov_b32 v0, {old_dst:#010x}",
        f"v_mov_b32 v1, {src_bits:#010x}",
        f"{opcode} v0, {src_operand}{suffix}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cvt_f16_f32", lines)
    return support.mrt0_words(spv)[0]


def run_cvt_quad(cases: list[tuple[int, int, str]]) -> list[int]:
    """Translate and run 4 simultaneous conversions exporting to mrt0 [v0, v1, v2, v3]."""
    assert len(cases) == 4
    lines = []
    for i, (src_bits, old_dst, insn_tail) in enumerate(cases):
        lines.append(f"v_mov_b32 v{i}, {old_dst:#010x}")
        lines.append(f"v_mov_b32 v{10 + i}, {src_bits:#010x}")
        lines.append(f"v_cvt_f16_f32_e64 v{i}, v{10 + i}{' ' + insn_tail if insn_tail else ''}")
    lines.append("exp mrt0 v0, v1, v2, v3 done vm")
    lines.append("s_endpgm")
    spv = support.translate_lines("ps_cvt_f16_f32_quad", lines)
    return support.mrt0_words(spv)


class TestValuCvtF16F32NormalConversions(unittest.TestCase):
    """Test normal IEEE-754 binary32 to binary16 conversions."""

    def test_normal_finite_values(self):
        cases = [
            (0x3F800000, 0x3C00),  # +1.0
            (0xBF800000, 0xBC00),  # -1.0
            (0x3F000000, 0x3800),  # +0.5
            (0xBF000000, 0xB800),  # -0.5
            (0x40000000, 0x4000),  # +2.0
            (0xC0000000, 0xC000),  # -2.0
            (0x3FC00000, 0x3E00),  # +1.5
            (0xBFC00000, 0xBE00),  # -1.5
            (0x40200000, 0x4100),  # +2.5
            (0x3E000000, 0x3000),  # +0.125
            (0x42280000, 0x5140),  # +42.0
            (0x477FE000, 0x7BFF),  # +65504.0 (max normal f16)
            (0xC77FE000, 0xFBFF),  # -65504.0
        ]
        for src_bits, exp_h in cases:
            with self.subTest(src=hex(src_bits), exp=hex(exp_h)):
                got = run_cvt(src_bits, old_dst=OLD_HIGH)
                self.assertEqual(got & 0xFFFF, exp_h)
                self.assertEqual(got & 0xFFFF0000, OLD_HIGH)

    def test_signed_zeros(self):
        # +0.0 -> +0.0 (0x0000)
        got_pos = run_cvt(0x00000000, old_dst=OLD_HIGH)
        self.assertEqual(got_pos, OLD_HIGH | 0x0000)

        # -0.0 -> -0.0 (0x8000)
        got_neg = run_cvt(0x80000000, old_dst=OLD_HIGH)
        self.assertEqual(got_neg, OLD_HIGH | 0x8000)

    def test_signed_infinities(self):
        # +inf -> +inf (0x7C00)
        got_pos = run_cvt(0x7F800000, old_dst=OLD_HIGH)
        self.assertEqual(got_pos, OLD_HIGH | 0x7C00)

        # -inf -> -inf (0xFC00)
        got_neg = run_cvt(0xFF800000, old_dst=OLD_HIGH)
        self.assertEqual(got_neg, OLD_HIGH | 0xFC00)

    def test_nans(self):
        # Quiet NaN -> quiet NaN (0x7E00)
        got_qnan = run_cvt(0x7FC00000, old_dst=OLD_HIGH)
        self.assertEqual(got_qnan, OLD_HIGH | 0x7E00)

        # Negative quiet NaN -> negative quiet NaN (0xFE00)
        got_neg_qnan = run_cvt(0xFFC00000, old_dst=OLD_HIGH)
        self.assertEqual(got_neg_qnan, OLD_HIGH | 0xFE00)

        # Signalling NaN -> quieted NaN with preserved payload bits (0x7F00)
        got_snan = run_cvt(0x7FA00000, old_dst=OLD_HIGH)
        self.assertEqual(got_snan, OLD_HIGH | 0x7F00)

    def test_subnormals_and_underflow(self):
        # Min normal f16: 2^-14 (0x38800000) -> 0x0400
        got_min_norm = run_cvt(0x38800000, old_dst=OLD_HIGH)
        self.assertEqual(got_min_norm, OLD_HIGH | 0x0400)

        # Max subnormal f16: (0x387FC000) -> 0x03FF
        got_max_sub = run_cvt(0x387FC000, old_dst=OLD_HIGH)
        self.assertEqual(got_max_sub, OLD_HIGH | 0x03FF)

        # Smallest subnormal f16: 2^-24 (0x33800000) -> 0x0001
        got_smallest_sub = run_cvt(0x33800000, old_dst=OLD_HIGH)
        self.assertEqual(got_smallest_sub, OLD_HIGH | 0x0001)

        # Underflow below smallest subnormal: 2^-25 (0x33000000) -> +0.0
        got_underflow = run_cvt(0x33000000, old_dst=OLD_HIGH)
        self.assertEqual(got_underflow, OLD_HIGH | 0x0000)

        # Negative underflow: -(2^-25) (0xB3000000) -> -0.0
        got_neg_underflow = run_cvt(0xB3000000, old_dst=OLD_HIGH)
        self.assertEqual(got_neg_underflow, OLD_HIGH | 0x8000)

    def test_rne_halfway_ties(self):
        # Round to nearest even ties
        # 1.0 (0x3F800000) -> 0x3C00
        self.assertEqual(run_cvt(0x3F800000) & 0xFFFF, 0x3C00)
        # 1.0 + 2^-12 (0x3F801000) is midpoint between 0x3C00 and 0x3C01 -> rounds to even 0x3C00
        self.assertEqual(run_cvt(0x3F801000) & 0xFFFF, 0x3C00)
        # 1.0 + 3*2^-12 (0x3F803000) is midpoint between 0x3C01 and 0x3C02 -> rounds to even 0x3C02
        self.assertEqual(run_cvt(0x3F803000) & 0xFFFF, 0x3C02)


class TestValuCvtF16F32DestinationPreservation(unittest.TestCase):
    """Test GFX10 16-bit partial register write destination preservation."""

    def test_high_half_preserved(self):
        seeds = [0x00000000, 0x12340000, 0x55AA0000, 0xDEAD0000, 0xFFFF0000, 0x80000000]
        for seed in seeds:
            with self.subTest(seed=hex(seed)):
                got = run_cvt(0x3F800000, old_dst=seed)  # 1.0 -> 0x3C00
                self.assertEqual(got, seed | 0x3C00)


class TestValuCvtF16F32VOP3Modifiers(unittest.TestCase):
    """Test VOP3 source and output modifiers."""

    def test_source_abs(self):
        # |-1.0| -> +1.0 (0x3C00)
        got = run_cvt(0xBF800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="|v1|")
        self.assertEqual(got, OLD_HIGH | 0x3C00)

    def test_source_neg(self):
        # -(+1.0) -> -1.0 (0xBC00)
        got = run_cvt(0x3F800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="-v1")
        self.assertEqual(got, OLD_HIGH | 0xBC00)

        # -(-1.0) -> +1.0 (0x3C00)
        got2 = run_cvt(0xBF800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="-v1")
        self.assertEqual(got2, OLD_HIGH | 0x3C00)

    def test_source_neg_abs(self):
        # -|+1.0| -> -1.0 (0xBC00)
        got = run_cvt(0x3F800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="-|v1|")
        self.assertEqual(got, OLD_HIGH | 0xBC00)

    def test_output_clamp(self):
        # Value > 1.0 (2.0 = 0x40000000) clamped to 1.0 (0x3C00)
        got_pos = run_cvt(0x40000000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="v1", mod_suffix="clamp")
        self.assertEqual(got_pos, OLD_HIGH | 0x3C00)

        # Value < 0.0 (-1.0 = 0xBF800000) clamped to 0.0 (0x0000)
        got_neg = run_cvt(0xBF800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="v1", mod_suffix="clamp")
        self.assertEqual(got_neg, OLD_HIGH | 0x0000)

    def test_output_omod(self):
        # 1.0 with mul:2 -> 2.0 (0x4000)
        got_mul2 = run_cvt(0x3F800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="v1", mod_suffix="mul:2")
        self.assertEqual(got_mul2, OLD_HIGH | 0x4000)

        # 1.0 with div:2 -> 0.5 (0x3800)
        got_div2 = run_cvt(0x3F800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="v1", mod_suffix="div:2")
        self.assertEqual(got_div2, OLD_HIGH | 0x3800)

        # 1.0 with mul:2 clamp -> 2.0 clamped to 1.0 (0x3C00)
        got_mul_clamp = run_cvt(0x3F800000, old_dst=OLD_HIGH, opcode="v_cvt_f16_f32_e64", src_operand="v1", mod_suffix="mul:2 clamp")
        self.assertEqual(got_mul_clamp, OLD_HIGH | 0x3C00)


class TestValuCvtF16F32InstructionForms(unittest.TestCase):
    """Test assembler forms, SGPR operands, and inline constants."""

    def test_e32_vgpr_form(self):
        lines = [
            "v_mov_b32 v0, 0x11110000",
            "v_mov_b32 v1, 0x3F800000",
            "v_cvt_f16_f32_e32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_e32", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x11113C00)

    def test_e32_sgpr_source(self):
        lines = [
            "v_mov_b32 v0, 0x22220000",
            "s_mov_b32 s4, 0x40000000",  # 2.0
            "v_cvt_f16_f32_e32 v0, s4",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_sgpr", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x22224000)

    def test_inline_constants(self):
        cases = [
            ("1.0", 0x3C00),
            ("0.5", 0x3800),
            ("-1.0", 0xBC00),
            ("0", 0x0000),
        ]
        for const_str, exp_h in cases:
            with self.subTest(const=const_str):
                lines = [
                    "v_mov_b32 v0, 0x33330000",
                    f"v_cvt_f16_f32 v0, {const_str}",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_cvt_const_{const_str.replace('.', '_')}", lines)
                self.assertEqual(support.mrt0_words(spv)[0], 0x33330000 | exp_h)

    def test_quad_pipeline(self):
        cases = [
            (0x3F800000, 0xAAAA0000, ""),             # 1.0 -> 0x3C00
            (0x40000000, 0xBBBB0000, "clamp"),        # 2.0 clamped -> 0x3C00
            (0x3F000000, 0xCCCC0000, "mul:2"),        # 0.5 * 2 = 1.0 -> 0x3C00
            (0xBF800000, 0xDDDD0000, ""),             # -1.0 -> 0xBC00
        ]
        words = run_cvt_quad(cases)
        self.assertEqual(words, [0xAAAA3C00, 0xBBBB3C00, 0xCCCC3C00, 0xDDDDBC00])


if __name__ == "__main__":
    unittest.main()
