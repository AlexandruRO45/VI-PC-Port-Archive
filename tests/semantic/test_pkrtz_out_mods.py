"""v_cvt_pkrtz_f16_f32 VOP3 output modifiers (standard library only; no GPU, no LLVM, no game files).

llvm-mc 23 for gfx1030 encodes `clamp` and `mul:2`/`mul:4`/`div:2` on v_cvt_pkrtz_f16_f32_e64. The translator dropped
both until 2026-11-23, so pkrtz(1.5, -2.0) clamp gave 0xc0003e00 instead of 0x00003c00. They now apply to each f16
half after the toward-zero narrowing, omod before clamp, through the helper v_cvt_f16_f32 uses (Gen.cvt_f16_out_mods;
docs/design.md, "v_cvt_pkrtz_f16_f32 output modifiers"). The expected words are worked out by hand below.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "pkrtz_out_mods.asm")


def run_one(s0, s1, mods):
    """mrt0.x of v_cvt_pkrtz_f16_f32_e64 v0, v1, v2 <mods> with v1 = s0, v2 = s1 (f32 bit patterns)."""
    lines = [f"v_mov_b32 v1, {s0:#010x}", f"v_mov_b32 v2, {s1:#010x}", "v_mov_b32 v0, 0x5a5a5a5a",
             f"v_cvt_pkrtz_f16_f32_e64 v0, v1, v2 {mods}".rstrip(), "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_pkrtz_mods", lines))[0]


class PkrtzOutputModifiers(unittest.TestCase):
    def test_fixture_words(self):
        with open(FIXTURE, "rb") as f:
            text = support.translate_listing("ps_pkrtz_out_mods", f.read())
        self.assertEqual(support.mrt0_words(text), [
            0x00003C00,   # clamp: 1.5 -> 1.0 (0x3c00), -2.0 -> +0.0; dropped, 0xc0003e00
            0x3E004600,   # mul:2: 3.0 -> 6.0 (0x4600), 0.75 -> 1.5 (0x3e00); dropped, 0x3a004200
            0xB8003E00,   # div:2: 3.0 -> 1.5, -1.0 -> -0.5 (0xb800); dropped, 0xbc004200
            0x38003C00,   # mul:4 then clamp: 0.375 -> 1.5 -> 1.0, 0.125 -> 0.5 (0x3800); clamp first gives 0x38003a00
        ])

    def test_clamp_per_half(self):
        cases = [
            (0x3F800000, 0xBF800000, 0x00003C00),   # 1.0 stays, -1.0 -> +0
            (0x7F7FFFFF, 0x00000000, 0x00003C00),   # FLT_MAX narrows to 65504 toward zero, then clamps to 1.0
            (0x3EAAAAAB, 0x80000000, 0x00003555),   # 1/3 toward zero is 0x3555 (inside [0, 1]); -0.0 -> +0
            (0xFF800000, 0x7F800000, 0x3C000000),   # -inf -> +0, +inf -> 1.0
        ]
        for s0, s1, want in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(run_one(s0, s1, "clamp"), want)

    def test_omod_per_half(self):
        cases = [
            (0x3F800000, 0x40000000, "mul:4", 0x48004400),   # 4.0, 8.0
            (0x477FE000, 0x3F800000, "mul:2", 0x40007C00),   # 65504 * 2 overflows to +inf (RNE MODE), 1.0 -> 2.0
            (0x38800000, 0xBF800000, "div:2", 0xB8000000),   # 2^-14 (e == 1) div:2 -> +0 (sign kept: +), -1.0 -> -0.5
            (0x33800000, 0x3F800000, "mul:2", 0x40000000),   # 2^-24 is an f16 denormal: omod flushes it to +0
        ]
        for s0, s1, mods, want in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), mods=mods):
                self.assertEqual(run_one(s0, s1, mods), want)

    def test_e32_form_unchanged(self):
        # The VOP2 form has no modifiers; toward zero still decides each half (1/3 -> 0x3555, -65520 -> -65504).
        lines = ["v_mov_b32 v1, 0x3eaaaaab", "v_mov_b32 v2, 0xc77ff000",
                 "v_cvt_pkrtz_f16_f32_e32 v0, v1, v2", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_pkrtz_e32", lines))[0], 0xFBFF3555)


if __name__ == "__main__":
    unittest.main()
