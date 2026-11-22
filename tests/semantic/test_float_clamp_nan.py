"""The VOP3 ``clamp`` bit on float results with a NaN, a -0 and an infinity, translated and executed (SHADER-186).

Measured on RDNA2 hardware (docs/shader_validation.md, the gfx1036 runs of `f32_fuzz.py --oracle hw` and of the f16 /
mix ops; the gfx1150 run agrees): clamp is min(max(x, 0), 1), and in the MODE the translator assumes for the game
(IEEE 0, DX10_CLAMP 1) a NaN result clamps to +0. A -0 result clamps to +0 in every MODE. An infinity clamps to 0 or
1 like any other value. The rule is the same for f32 results, for f16 results (scalar and each half of a packed op) and
for the mix ops; for f64 it is the same clamp field, unmeasured.

GLSL.std.450 FClamp, which the translator emitted before, leaves the NaN result undefined (the interpreter refuses
it) and the sign of a -0 result open. Inputs come from memory, so no driver can fold them.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402  (puts LEONIDA_TOOLS_DIR or tools/ on the path)
from test_float_nan_fract import LOAD, NAN_A, NAN_B, PUSH, run_op, translate_text  # noqa: E402

PZ, NZ, ONE, HALF, TWO, M25 = 0x00000000, 0x80000000, 0x3F800000, 0x3F000000, 0x40000000, 0xC0200000
PINF, NINF = 0x7F800000, 0xFF800000
# (input, clamped result) for f32
F32_ROWS = [(NAN_A, PZ), (NAN_B, PZ), (NZ, PZ), (PINF, ONE), (NINF, PZ), (HALF, HALF), (TWO, ONE), (M25, PZ),
            (PZ, PZ), (ONE, ONE)]
# (f16 input, clamped f16 result)
F16_ROWS = [(0x7E11, 0x0000), (0xFE22, 0x0000), (0x8000, 0x0000), (0x7C00, 0x3C00), (0xFC00, 0x0000),
            (0x3800, 0x3800), (0x4000, 0x3C00), (0xC100, 0x0000)]


class ClampF32(unittest.TestCase):
    def test_v_mul_f32_clamp_rows(self):
        for x, want in F32_ROWS:
            with self.subTest(x=hex(x)):
                got = run_op("v_mul_f32_e64 v0, v1, 1.0 clamp", (x,))
                self.assertEqual(got, want, f"clamp({x:#010x}): got {got:#010x}, want {want:#010x}")

    def test_omod_then_clamp(self):
        # omod is applied first: NaN * 2 is still a NaN and clamps to +0; 0.75 * 2 = 1.5 clamps to 1.0.
        self.assertEqual(run_op("v_add_f32_e64 v0, v1, v2 mul:2 clamp", (NAN_A, ONE)), PZ)
        self.assertEqual(run_op("v_add_f32_e64 v0, v1, v2 mul:2 clamp", (0x3F400000, PZ)), ONE)
        self.assertEqual(run_op("v_add_f32_e64 v0, v1, v2 div:2 clamp", (NZ, NZ)), PZ)

    def test_v_mad_f32_clamp(self):
        self.assertEqual(run_op("v_mad_f32 v0, v1, v2, v3 clamp", (NAN_A, ONE, ONE)), PZ)
        self.assertEqual(run_op("v_mad_f32 v0, v1, v2, v3 clamp", (NZ, ONE, NZ)), PZ)

    def test_v_fma_mix_f32_clamp(self):
        # op_sel_hi clear: all three operands are f32.
        self.assertEqual(run_op("v_fma_mix_f32 v0, v1, v2, v3 clamp", (NAN_A, ONE, PZ)), PZ)
        self.assertEqual(run_op("v_fma_mix_f32 v0, v1, v2, v3 clamp", (TWO, ONE, PZ)), ONE)

    def test_no_fclamp_for_the_clamp_bit(self):
        spv = translate_text(LOAD + "v_mul_f32_e64 v0, v1, 1.0 clamp\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        self.assertNotIn("FClamp", spv)


class ClampF16(unittest.TestCase):
    def test_v_mul_f16_clamp_rows(self):
        # The high half of the destination (v0 = 0) is kept.
        for x, want in F16_ROWS:
            with self.subTest(x=hex(x)):
                got = run_op("v_mul_f16_e64 v0, v1, 1.0 clamp", (x,))
                self.assertEqual(got, want, f"clamp({x:#06x}): got {got:#010x}, want {want:#06x}")

    def test_v_fmac_f16_clamp(self):
        self.assertEqual(run_op("v_fmac_f16_e64 v0, v1, v2 clamp", (0x7E11, 0x3C00)), 0x0000)

    def test_v_pk_mul_f16_clamp(self):
        # Each half separately: (NaN, -0) -> (+0, +0); (NaN, 0.5) -> (+0, 0.5); (inf, -inf) -> (1.0, +0).
        for x, want in ((0x80007E11, 0x00000000), (0x38007E11, 0x38000000), (0xFC007C00, 0x00003C00)):
            with self.subTest(x=hex(x)):
                got = run_op("v_pk_mul_f16 v0, v1, v2 clamp", (x, 0x3C003C00))
                self.assertEqual(got, want, f"clamp({x:#010x}): got {got:#010x}, want {want:#010x}")


class ClampF64(unittest.TestCase):
    """The interpreters run no 64-bit floats, so the f64 clamp is checked as SPIR-V: the same compare-and-select
    sequence on the %double sum as for f32 (runtime/test/shaders/ps_f64_mods.asm runs the values that are not NaN)."""

    def test_v_add_f64_clamp_is_compare_and_select(self):
        spv = translate_text(LOAD + "v_add_f64 v[6:7], v[1:2], v[3:4] clamp\n"
                             "exp mrt0 v6, v7, v6, v7 done vm\ns_endpgm")
        self.assertNotIn("FClamp", spv)
        add = re.search(r"(?m)^ *(%\w+) = OpFAdd %double ", spv).group(1)
        pos = re.search(rf"(?m)^ *(%\w+) = OpFOrdGreaterThan %bool {add} (%\w+)$", spv)
        self.assertIsNotNone(pos, "no ordered v > 0 on the f64 sum")
        lo = re.search(rf"(?m)^ *(%\w+) = OpSelect %double {pos.group(1)} {add} {pos.group(2)}$", spv)
        self.assertIsNotNone(lo, "no select of +0 when v > 0 is false (NaN, -0)")
        below = re.search(rf"(?m)^ *(%\w+) = OpFOrdLessThan %bool {lo.group(1)} (%\w+)$", spv)
        self.assertIsNotNone(below)
        self.assertRegex(spv, rf"(?m)^ *%\w+ = OpSelect %double {below.group(1)} {lo.group(1)} {below.group(2)}$")


class GpuFixture(unittest.TestCase):
    """runtime/test/shaders/ps_fnan_clamp.asm with the heap words check.py and run_gpu_tests.py seed."""

    def test_ps_fnan_clamp(self):
        heap = (NAN_A, NZ, 0x80007E11, 0x3C003C00, PINF)
        spv = support.translate_fixture("ps_fnan_clamp")
        words = support.mrt0_words(spv, push_constants=PUSH,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(heap)})
        self.assertEqual(words, [PZ, PZ, PZ, ONE])


if __name__ == "__main__":
    unittest.main()
