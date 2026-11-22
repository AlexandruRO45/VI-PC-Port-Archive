"""Semantic interpreter tests for RDNA2 (GFX10.3) v_fma_legacy_f32 and v_fmac_legacy_f32.

Both compute single-rounded FMA with the DX9 legacy zero rule (0 * anything = +0.0, including inf/NaN),
without flushing f32 denormals:
  - v_fma_legacy_f32  vdst, src0, src1, src2   (VOP3)
  - v_fmac_legacy_f32 vdst, src0, src1         (VOP2/VOP3, vdst as addend)

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
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

import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
ONE = 0x3F800000
NEG_ONE = 0xBF800000
TWO = 0x40000000
HALF = 0x3F000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SUB_2M127 = 0x00400000  # 2^-127 (f32 subnormal)


def f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_bits(bits: int) -> bool:
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def run_ps(lines) -> tuple:
    spv = translate_lines("ps_fma_legacy", list(lines) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    return tuple(mrt0_words(spv))


class FmaLegacyF32Tests(unittest.TestCase):
    def test_single_rounding_differs_from_double_rounded(self):
        # (1 + 2^-23) * (1 + 2^-23) - (1 + 2^-22) = 2^-46 (exact FMA); double-rounded gives 0.0.
        a = f32_bits(1.0 + 2.0 ** -23)
        c = f32_bits(-(1.0 + 2.0 ** -22))
        want = f32_bits(2.0 ** -46)
        got = run_ps([
            f"v_mov_b32 v1, {a:#x}",
            f"v_mov_b32 v2, {a:#x}",
            f"v_mov_b32 v3, {c:#x}",
            "v_fma_legacy_f32 v0, v1, v2, v3",
            f"v_mov_b32 v3, {c:#x}",
            "v_fmac_legacy_f32 v3, v1, v2",
        ])
        self.assertEqual(got[0], want)
        self.assertEqual(got[3], want)

    def test_legacy_zero_rules_and_nan_inf(self):
        # (+0 * +inf + 2.0) == 2.0, (-0 * -inf + 2.0) == 2.0, (+0 * NaN + 2.0) == 2.0, (-0 * NaN + -0.0) == +0.0
        got = run_ps([
            f"v_mov_b32 v4, {POS_ZERO:#x}",
            f"v_mov_b32 v5, {NEG_ZERO:#x}",
            f"v_mov_b32 v6, {POS_INF:#x}",
            f"v_mov_b32 v7, {NEG_INF:#x}",
            f"v_mov_b32 v8, {QNAN:#x}",
            f"v_mov_b32 v9, {TWO:#x}",
            "v_fma_legacy_f32 v0, v4, v6, v9",
            "v_fma_legacy_f32 v1, v5, v7, v9",
            "v_fma_legacy_f32 v2, v4, v8, v9",
            "v_fma_legacy_f32 v3, v5, v8, v5",
        ])
        self.assertEqual(got, (TWO, TWO, TWO, POS_ZERO))

        # (-0.0) * 1.0 + (-0.0) == +0.0, non_zero * NaN + 2.0 == NaN, non_zero * inf + (-inf) == NaN
        got2 = run_ps([
            f"v_mov_b32 v4, {NEG_ZERO:#x}",
            f"v_mov_b32 v5, {ONE:#x}",
            f"v_mov_b32 v6, {QNAN:#x}",
            f"v_mov_b32 v7, {POS_INF:#x}",
            f"v_mov_b32 v8, {NEG_INF:#x}",
            f"v_mov_b32 v9, {TWO:#x}",
            "v_fma_legacy_f32 v0, v4, v5, v4",
            "v_fma_legacy_f32 v1, v5, v6, v9",
            "v_fma_legacy_f32 v2, v5, v7, v8",
            f"v_mov_b32 v3, {NEG_ZERO:#x}",
            "v_fmac_legacy_f32 v3, v4, v6",
        ])
        self.assertEqual(got2[0], POS_ZERO)
        self.assertTrue(is_nan_bits(got2[1]))
        self.assertTrue(is_nan_bits(got2[2]))
        self.assertEqual(got2[3], POS_ZERO)

    def test_denormals_flushed_gfx1036(self):
        # 7 native RDNA2 (gfx1036) denormal cases measured by @TheMrBeez: inputs and result flush f32 denormals.
        cases = [
            (0x00000003, ONE, POS_ZERO, 0x00000000),
            (0x80000003, ONE, POS_ZERO, 0x00000000),
            (TWO, 0x00000003, POS_ZERO, 0x00000000),
            (POS_ZERO, ONE, 0x00000003, 0x00000000),
            (0x00800000, HALF, POS_ZERO, 0x00000000),
            (0x80800000, HALF, POS_ZERO, 0x80000000),
            (ONE, 0x00800000, 0x80000003, 0x00800000),
        ]
        for x, y, z, want in cases:
            with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                got = run_ps([
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    f"v_mov_b32 v3, {z:#x}",
                    "v_fma_legacy_f32 v0, v1, v2, v3",
                    "v_fmac_legacy_f32 v3, v1, v2",
                ])
                self.assertEqual(got[0], want)
                self.assertEqual(got[3], want)

    def test_source_and_output_modifiers_and_operands(self):
        # Source abs/neg, output mul:2, div:2, clamp, SGPR + inline constants
        got = run_ps([
            f"s_mov_b32 s4, {TWO:#x}",
            f"v_mov_b32 v4, {NEG_ONE:#x}",
            f"v_mov_b32 v5, {HALF:#x}",
            # |-1.0| * 2.0 + (-0.5) = 1.5, then * 2 = 3.0
            "v_fma_legacy_f32 v0, |v4|, s4, -v5 mul:2",
            # 2.0 * 1.0 + 2.0 = 4.0, then / 2 = 2.0
            "v_fma_legacy_f32 v1, s4, 1.0, 2.0 div:2",
            # 2.0 * 2.0 + 1.0 = 5.0, clamped to 1.0
            "v_fma_legacy_f32 v2, s4, 2.0, 1.0 clamp",
            # v_fmac_legacy_f32_e32 and _e64 clamp
            f"v_mov_b32 v3, {HALF:#x}",
            "v_fmac_legacy_f32_e32 v3, 2.0, v5",  # 2.0 * 0.5 + 0.5 = 1.5
            "v_fmac_legacy_f32_e64 v3, 1.0, v5 clamp",  # 1.0 * 0.5 + 1.5 = 2.0 -> clamp 1.0
        ])
        self.assertEqual(got, (f32_bits(3.0), TWO, ONE, ONE))

    def test_fmac_legacy_rmw_prepass(self):
        insns = [shader_ir.parse_line(ln) for ln in (
            "v_fmac_legacy_f32 v7, v1, v2",
            "exp mrt0 v7, v7, v7, v7 done vm",
            "s_endpgm",
        )]
        rbw = rdna2spirv.read_before_write_vgprs(insns)
        rr, ww = rdna2spirv.reg_reads_writes(insns)
        self.assertIn(7, rbw)
        self.assertIn(7, rr)
        self.assertIn(7, ww)


if __name__ == "__main__":
    unittest.main()
