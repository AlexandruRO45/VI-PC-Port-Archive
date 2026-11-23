"""Semantic and structural tests for RDNA2 v_cvt_i32_f64 and v_cvt_u32_f64.

Note: On upstream/main before #2532 merges, spirv_mini_interp.py narrows Trunc/FClamp
results through f32(...), so runtime mrt0_words tests use values exact in binary32, while structural SPIR-V
assertions verify %double types, dconst saturation bounds (-2147483648.0/2147483647.0 and 0.0/4294967295.0),
OpConvertFToS/U, abs/neg, high-word hex literal placement, and Unsupported rejections.

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
TESTS_DIR = os.path.join(ROOT, "tests")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_interp  # noqa: E402
from support import translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def f64_words(val: float) -> tuple:
    bits = struct.unpack("<Q", struct.pack("<d", val))[0]
    return bits & MASK32, bits >> 32


def run_ps(lines) -> tuple:
    spv = translate_lines("ps_f64_rc", list(lines) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    return tuple(spirv_interp.run(spv).vgpr[:4]), spv


class F64CvtToIntTests(unittest.TestCase):
    def test_cvt_i32_and_u32_runtime_values(self):
        i32_cases = [
            (-42.0, (-42) & MASK32),
            (-2.5, (-2) & MASK32),
            (-1.5, (-1) & MASK32),
            (-0.5, 0),
            (-0.0, 0),
            (0.0, 0),
            (0.5, 0),
            (1.5, 1),
            (2.5, 2),
            (42.0, 42),
            (1000.0, 1000),
            (16777215.0, 16777215),
            (-2147483648.0, 0x80000000),
            (2147483647.0, 0x7FFFFFFF),
            (1e20, 0x7FFFFFFF),
            (math.inf, 0x7FFFFFFF),
            (-1e20, 0x80000000),
            (-math.inf, 0x80000000),
            (math.nan, 0),
        ]
        for val, want in i32_cases:
            lo, hi = f64_words(val)
            with self.subTest(op="v_cvt_i32_f64", val=val):
                got, _ = run_ps([
                    f"v_mov_b32 v2, {lo:#x}",
                    f"v_mov_b32 v3, {hi:#x}",
                    "v_mov_b32 v1, 0",
                    "v_cvt_i32_f64 v0, v[2:3]",
                ])
                self.assertEqual(got[0], want)

        u32_cases = [
            (-42.0, 0),
            (-2.5, 0),
            (-0.5, 0),
            (-0.0, 0),
            (0.0, 0),
            (0.5, 0),
            (1.5, 1),
            (2.5, 2),
            (42.0, 42),
            (1000.0, 1000),
            (16777215.0, 16777215),
            (4294967295.0, 0xFFFFFFFF),
            (1e20, 0xFFFFFFFF),
            (math.inf, 0xFFFFFFFF),
            (-1e20, 0),
            (-math.inf, 0),
            (math.nan, 0),
        ]
        for val, want in u32_cases:
            lo, hi = f64_words(val)
            with self.subTest(op="v_cvt_u32_f64", val=val):
                got, _ = run_ps([
                    f"v_mov_b32 v2, {lo:#x}",
                    f"v_mov_b32 v3, {hi:#x}",
                    "v_mov_b32 v1, 0",
                    "v_cvt_u32_f64 v0, v[2:3]",
                ])
                self.assertEqual(got[0], want)

    def test_cvt_spirv_structure_and_saturation_bounds(self):
        lo, hi = f64_words(1e20)
        spv_i = translate_lines("ps_cvt_i32", [
            f"s_mov_b32 s4, {lo:#x}",
            f"s_mov_b32 s5, {hi:#x}",
            "v_cvt_i32_f64 v0, -|s[4:5]|",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpIsNan %bool", spv_i)
        self.assertIn("OpExtInst %double %glsl Trunc", spv_i)
        self.assertIn("OpExtInst %double %glsl FClamp", spv_i)
        self.assertIn("OpConvertFToS %int", spv_i)
        self.assertIn("%cc1e00000", spv_i)   # -2147483648.0 hi word
        self.assertIn("%cffc00000", spv_i)   # 2147483647.0 lo word
        self.assertIn("%c41dfffff", spv_i)   # 2147483647.0 hi word
        self.assertIn("OpExtInst %double %glsl FAbs", spv_i)
        self.assertIn("OpFNegate %double", spv_i)

        spv_u = translate_lines("ps_cvt_u32", [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_cvt_u32_f64 v0, v[2:3]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpIsNan %bool", spv_u)
        self.assertIn("OpExtInst %double %glsl Trunc", spv_u)
        self.assertIn("OpExtInst %double %glsl FClamp", spv_u)
        self.assertIn("OpConvertFToU %uint", spv_u)
        self.assertIn("%cffe00000", spv_u)   # 4294967295.0 lo word
        self.assertIn("%c41efffff", spv_u)   # 4294967295.0 hi word

    def test_high_word_hex_literals(self):
        cases = [
            ("v_cvt_i32_f64_e32 v0, 0x3ff80000", 1, "%c3ff80000"),
            ("v_cvt_i32_f64_e64 v0, 0x40040000", 2, "%c40040000"),
            ("v_cvt_i32_f64_e32 v0, 0x80000000", 0, "%c80000000"),
            ("v_cvt_i32_f64_e32 v0, 0xc0040000", (-2) & MASK32, "%cc0040000"),
            ("v_cvt_u32_f64_e32 v0, 0xc0040000", 0, "%cc0040000"),
            ("v_cvt_u32_f64_e32 v0, 0x40040000", 2, "%c40040000"),
        ]
        for ins, want, hi_const in cases:
            with self.subTest(ins=ins):
                got, spv = run_ps([
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    ins,
                ])
                self.assertEqual(got[0], want)
                self.assertIn(f"OpCompositeConstruct %v2uint_d %c0 {hi_const}", spv)

    def test_unsupported_operands_and_modifiers(self):
        bad_lines = [
            "v_cvt_i32_f64 v0, v[2:3] clamp",
            "v_cvt_u32_f64 v0, v[2:3] mul:2",
            "v_cvt_i32_f64 v0, v[2:3] div:2",
            "v_cvt_u32_f64 v0, v2",
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_f64_bad", [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
