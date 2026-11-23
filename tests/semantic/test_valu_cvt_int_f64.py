"""v_cvt_i32_f64 / v_cvt_u32_f64 through the translator: truncate toward zero, saturate, NaN -> 0."""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

INF, NAN = float("inf"), float("nan")
I32 = [(5.75, 5), (-5.75, -5), (-0.5, 0), (2147483647.9, 2147483647), (3e9, 2147483647),
       (-3e9, -2147483648), (INF, 2147483647), (-INF, -2147483648), (NAN, 0)]
U32 = [(5.75, 5), (-0.5, 0), (-7.0, 0), (4294967295.5, 4294967295), (5e9, 4294967295), (INF, 4294967295),
       (-INF, 0), (NAN, 0)]


def convert(op, value, mods=""):
    low, high = struct.unpack("<II", struct.pack("<d", value))
    lines = [f"v_mov_b32 v0, {low:#x}", f"v_mov_b32 v1, {high:#x}", f"{op} v2, {mods or 'v[0:1]'}",
             "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_cvt_int_f64", lines))[0]


class CvtIntF64(unittest.TestCase):
    def test_i32(self):
        for value, want in I32:
            with self.subTest(value=value):
                self.assertEqual(convert("v_cvt_i32_f64", value), want & 0xFFFFFFFF)

    def test_u32(self):
        for value, want in U32:
            with self.subTest(value=value):
                self.assertEqual(convert("v_cvt_u32_f64", value), want)

    def test_abs_neg(self):
        self.assertEqual(convert("v_cvt_i32_f64_e64", 7.5, "-|v[0:1]|"), (-7) & 0xFFFFFFFF)
        self.assertEqual(convert("v_cvt_u32_f64_e64", -7.5, "|v[0:1]|"), 7)


if __name__ == "__main__":
    unittest.main()
