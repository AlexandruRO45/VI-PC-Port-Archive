"""Translator rules for v_mul_f64, v_fma_f64, v_min_f64, v_max_f64, v_cvt_f64_{f32,i32,u32} and
v_cvt_{i32,u32}_f64, executed with tests/spirv_interp.py.

Expected values are written by hand from the RDNA2 ISA descriptions: the products and sums below are exact in
binary64, so no rounding question arises; the f64 -> integer conversions truncate toward zero, saturate out-of-range
values and infinities, and give 0 for NaN. Standard library only, no GPU, no game data.
"""

import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

from listing import translate_listing  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_interp  # noqa: E402

INF = 0x7FF0000000000000
NINF = 0xFFF0000000000000
QNAN = 0x7FF8000000000000


def d2u(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def f2u32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def movs(reg, bits):
    return [f"v_mov_b32 v{reg}, {bits & 0xFFFFFFFF:#x}", f"v_mov_b32 v{reg + 1}, {bits >> 32 & 0xFFFFFFFF:#x}"]


class F64AluRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._dir.name)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def word(self, lines, reg):
        """Run ``lines`` in a pixel shader and return the 32-bit word left in v<reg> (exported through v3)."""
        type(self).n += 1
        spv = translate_listing(self.tmp, "\n".join(lines + [f"v_mov_b32 v3, v{reg}"]), name=f"ps_f64_{self.n}")
        return f2u32(spirv_interp.run(spv).outputs["out_mrt0"][0])

    def pair(self, lines, reg=10):
        return self.word(lines, reg) | self.word(lines, reg + 1) << 32

    def test_mul(self):
        cases = ((1.5, 2.25, 3.375), (-0.75, 4.0, -3.0), (2.0 ** 600, 2.0 ** 600, math.inf), (0.0, -2.0, -0.0))
        for x, y, want in cases:
            with self.subTest(x=x, y=y):
                body = movs(4, d2u(x)) + movs(6, d2u(y)) + ["v_mul_f64 v[10:11], v[4:5], v[6:7]"]
                self.assertEqual(self.pair(body), d2u(want))

    def test_mul_modifiers(self):
        body = movs(4, d2u(1.5)) + movs(6, d2u(-2.0)) + ["v_mul_f64 v[10:11], -v[4:5], |v[6:7]| mul:2"]
        self.assertEqual(self.pair(body), d2u(-6.0))

    def test_fma_is_fused(self):
        # (1 + 2^-30)^2 - (1 + 2^-29) = 2^-60: exact only with a single rounding.
        a = 1.0 + 2.0 ** -30
        body = (movs(4, d2u(a)) + movs(6, d2u(a)) + movs(8, d2u(-(1.0 + 2.0 ** -29)))
                + ["v_fma_f64 v[10:11], v[4:5], v[6:7], v[8:9]"])
        self.assertEqual(self.pair(body), d2u(2.0 ** -60))

    def test_min_max(self):
        cases = (
            ("v_min_f64", d2u(1.0), d2u(-3.5), d2u(-3.5)),
            ("v_max_f64", d2u(1.0), d2u(-3.5), d2u(1.0)),
            ("v_min_f64", QNAN, d2u(2.0), d2u(2.0)),
            ("v_max_f64", d2u(2.0), QNAN, d2u(2.0)),
            ("v_max_f64", NINF, d2u(-1e300), d2u(-1e300)),
        )
        for op, x, y, want in cases:
            with self.subTest(op=op, x=hex(x), y=hex(y)):
                body = movs(4, x) + movs(6, y) + [f"{op} v[10:11], v[4:5], v[6:7]"]
                self.assertEqual(self.pair(body), want)

    def test_cvt_to_f64(self):
        cases = (
            ("v_cvt_f64_f32", f2u32(0.1), d2u(struct.unpack("<f", struct.pack("<f", 0.1))[0])),
            ("v_cvt_f64_f32", 0xFF800000, NINF),
            ("v_cvt_f64_i32", 0xFFFFFFFF, d2u(-1.0)),
            ("v_cvt_f64_i32", 0x80000000, d2u(-2147483648.0)),
            ("v_cvt_f64_u32", 0xFFFFFFFF, d2u(4294967295.0)),
        )
        for op, src, want in cases:
            for enc in ("_e32", "_e64"):
                with self.subTest(op=op + enc, src=hex(src)):
                    self.assertEqual(self.pair([f"v_mov_b32 v4, {src:#x}", f"{op}{enc} v[10:11], v4"]), want)

    def test_cvt_from_f64(self):
        cases = (
            ("v_cvt_i32_f64", d2u(-7.9), 0xFFFFFFF9),
            ("v_cvt_i32_f64", d2u(2147483647.9), 0x7FFFFFFF),
            ("v_cvt_i32_f64", d2u(-2147483648.9), 0x80000000),
            ("v_cvt_i32_f64", d2u(1e20), 0x7FFFFFFF),
            ("v_cvt_i32_f64", NINF, 0x80000000),
            ("v_cvt_i32_f64", QNAN, 0),
            ("v_cvt_u32_f64", d2u(4294967295.5), 0xFFFFFFFF),
            ("v_cvt_u32_f64", d2u(-0.9), 0),
            ("v_cvt_u32_f64", d2u(-5.0), 0),
            ("v_cvt_u32_f64", INF, 0xFFFFFFFF),
            ("v_cvt_u32_f64", QNAN, 0),
        )
        for op, src, want in cases:
            with self.subTest(op=op, src=hex(src)):
                self.assertEqual(self.word(movs(4, src) + [f"{op} v10, v[4:5]"], 10), want)

    def test_cvt_from_f64_refuses_output_modifiers(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            self.word(movs(4, d2u(1.0)) + ["v_cvt_i32_f64_e64 v10, v[4:5] clamp"], 10)


if __name__ == "__main__":
    unittest.main()
