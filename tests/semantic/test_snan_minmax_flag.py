"""--snan-minmax: f32 min/max/min3/max3/med3 stay right on a driver whose NMin/NMax mishandle a signalling NaN.

Standard library only; no GPU, no LLVM, no game files.

GLSL.std.450 NMin/NMax return the other operand when one operand is a NaN, and the contract row "Float min/max/med3
with NaN" in docs/design.md relies on it: one NaN operand gives the other operand, both NaN give src0, med3 with a
NaN is min3. The AMD Windows driver (RX 7700 XT, 26.11.2, LLPC) returns a quiet NaN instead when the *first* operand
is a signalling NaN. Measured with vk_compute_test on hand-written listings translated by main, subgroup size 32 and
64 alike: v_min_f32(0x7f800001, 1.0) = 0x7fc00001, v_max_f32(0x7fa00000, 2.0) = 0x7fe00000,
v_med3_f32(0x7f800001, 0x03b3bba2, 0x7a3234ad) = 0x7a3234ad; a quiet NaN, or the signalling one second, is fine.
With --snan-minmax the same listings give 0x3f800000, 0x40000000 and 0x03b3bba2 on that GPU.

The CPU interpreter follows the SPIR-V rule, so AMD_NMIN below models the driver (a signalling NaN first gives that
NaN, quietened) and the tests run the translated code under both: the measured driver words come back without the
flag, the contract's words with it, and the flag changes nothing under the SPIR-V rule.
"""
import contextlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402  (puts tools/ on the path)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

SNAN, SNAN_NEG, SNAN2, QNAN = 0x7F800001, 0xFF800001, 0x7FA00000, 0x7FC00000
ONE, TWO, THREE, MINUS_ONE = 0x3F800000, 0x40000000, 0x40400000, 0xBF800000
TINY, BIG = 0x03B3BBA2, 0x7A3234AD

# (instruction, operand words, the contract's result, the AMD driver's word without the flag where it was measured)
CASES = [
    ("v_min_f32", (SNAN, ONE), ONE, 0x7FC00001),
    ("v_max_f32", (SNAN2, TWO), TWO, 0x7FE00000),
    ("v_min_f32", (SNAN_NEG, MINUS_ONE), MINUS_ONE, 0xFFC00001),
    ("v_min_f32", (ONE, SNAN), ONE, ONE),
    ("v_min_f32", (QNAN, ONE), ONE, ONE),
    ("v_med3_f32", (SNAN, TINY, BIG), TINY, BIG),
    ("v_med3_f32", (QNAN, TINY, BIG), TINY, TINY),
    ("v_min3_f32", (SNAN, THREE, TWO), TWO, None),
    ("v_max3_f32", (SNAN, ONE, TWO), TWO, None),
    ("v_med3_f32", (TINY, SNAN, BIG), TINY, None),
    ("v_min_f32", (SNAN, QNAN), SNAN, None),        # both NaN: src0, bits unchanged
]


def is_nan_bits(b):
    return (b & 0x7F800000) == 0x7F800000 and b & 0x7FFFFF


def amd_nmin(original):
    """The driver as measured: a signalling NaN as the first operand gives that NaN, quietened."""
    def nmin_nmax(name, x, y):
        if isinstance(x, float) and math.isnan(x):
            b = si.f32_bits(x)
            if is_nan_bits(b) and not b & 0x400000:
                return si.bits_f32(b | 0x400000)
        return original(name, x, y)
    return nmin_nmax


@contextlib.contextmanager
def amd_driver():
    original = si._nmin_nmax
    si._nmin_nmax = amd_nmin(original)
    try:
        yield
    finally:
        si._nmin_nmax = original


def translate(op, words, flag):
    lines = [f"v_mov_b32 v{i}, {w:#x}" for i, w in enumerate(words)]
    lines += [f"{op} v3, " + ", ".join(f"v{i}" for i in range(len(words))), "exp mrt0 v3, v3, v3, v3 done vm",
              "s_endpgm"]
    saved = rdna2spirv.SNAN_MINMAX
    rdna2spirv.SNAN_MINMAX = flag
    try:
        return support.translate_lines("ps_snan_minmax", lines)
    finally:
        rdna2spirv.SNAN_MINMAX = saved


def result(spv):
    return support.mrt0_words(spv)[0] & 0xFFFFFFFF


class SnanMinmaxFlag(unittest.TestCase):
    def test_flag_is_off_by_default(self):
        self.assertFalse(rdna2spirv.SNAN_MINMAX)

    def test_driver_model_gives_the_measured_words_without_the_flag(self):
        with amd_driver():
            for op, words, _, measured in CASES:
                if measured is None:
                    continue
                with self.subTest(op=op, words=[hex(w) for w in words]):
                    self.assertEqual(result(translate(op, words, False)), measured)

    def test_flag_gives_the_contract_words_on_the_driver_model(self):
        with amd_driver():
            for op, words, want, _ in CASES:
                with self.subTest(op=op, words=[hex(w) for w in words]):
                    self.assertEqual(result(translate(op, words, True)), want)

    def test_flag_changes_nothing_under_the_spirv_rule(self):
        for op, words, want, _ in CASES:
            with self.subTest(op=op, words=[hex(w) for w in words]):
                self.assertEqual(result(translate(op, words, False)), want)
                self.assertEqual(result(translate(op, words, True)), want)

    def test_flag_adds_one_select_per_first_operand(self):
        off, on = translate("v_min_f32", (SNAN, ONE), False), translate("v_min_f32", (SNAN, ONE), True)
        self.assertEqual(on.count("OpIsNan") - off.count("OpIsNan"), 1)
        self.assertEqual(on.count("OpSelect %float") - off.count("OpSelect %float"), 1)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "SPIRV-Tools not on PATH")
    def test_flagged_modules_validate(self):
        for op, words, _, _ in CASES:
            with self.subTest(op=op), tempfile.TemporaryDirectory() as tmp:
                sa, sv = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                with open(sa, "w", encoding="utf-8", newline="\n") as f:
                    f.write(translate(op, words, True))
                subprocess.run(["spirv-as", "--target-env", "vulkan1.2", sa, "-o", sv], check=True)
                subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv], check=True)


if __name__ == "__main__":
    unittest.main()
