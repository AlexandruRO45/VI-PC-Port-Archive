"""test_alu_f64_hw.py: v_mul/fma/min/max_f64 against words measured on RDNA2 gfx1036 by @Vonsek (#1976)."""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402


def D(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def r64(op, a, b, c=None):
    lines = [f"v_mov_b32 v2, {a & 0xffffffff:#x}", f"v_mov_b32 v3, {a >> 32:#x}",
             f"v_mov_b32 v4, {b & 0xffffffff:#x}", f"v_mov_b32 v5, {b >> 32:#x}"]
    if c is not None:
        lines += [f"v_mov_b32 v6, {c & 0xffffffff:#x}", f"v_mov_b32 v7, {c >> 32:#x}"]
    lines += [op, "v_mov_b32 v0, v8", "v_mov_b32 v1, v9", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0",
              "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    w = mrt0_words(translate_lines("ps_f64_hw", lines))
    return w[0] | (w[1] << 32)


QN=0x7ff8000000000001; QB=0xfff8000000000002; SN=0x7ff0000000000001
cases=[
    ("v_mul_f64 v[8:9], v[2:3], v[4:5]",D(1.5),D(-2),None,0xc008000000000000),
    ("v_fma_f64 v[8:9], v[2:3], v[2:3], v[6:7]",D(1+2**-30),0,D(-(1+2**-29)),0x3c30000000000000),
    ("v_mul_f64 v[8:9], v[2:3], v[4:5]",D(2**-1022),D(0.5),None,0x0008000000000000),
    ("v_fma_f64 v[8:9], v[2:3], v[4:5], v[6:7]",D(2**-1022),D(0.5),D(2**-1022),0x0018000000000000),
    ("v_min_f64 v[8:9], v[2:3], v[4:5]",QN,D(3),None,0x4008000000000000),
    ("v_max_f64 v[8:9], v[2:3], v[4:5]",D(3),QN,None,0x4008000000000000),
    ("v_min_f64 v[8:9], v[2:3], v[4:5]",QN,QB,None,QN),
    ("v_min_f64 v[8:9], v[2:3], v[4:5]",D(-0.0),0,None,0x8000000000000000),
    ("v_max_f64 v[8:9], v[2:3], v[4:5]",0,D(-0.0),None,0),
    ("v_min_f64 v[8:9], v[2:3], v[4:5]",0,D(-0.0),None,0x8000000000000000),
    ("v_min_f64 v[8:9], v[2:3], v[4:5]",SN,D(1),None,D(1)),
    ("v_max_f64 v[8:9], v[2:3], v[4:5]",1,0,None,1),
    ("v_min_f64 v[8:9], v[2:3], v[4:5]",SN,QN,None,SN),
    ("v_min_f64 v[8:9], v[2:3], v[4:5] clamp",D(5),D(3),None,0x3ff0000000000000),
]


class F64HwWords(unittest.TestCase):
    def test_measured_words(self):
        for op, a, b, c, want in cases:
            with self.subTest(op=op, a=hex(a), b=hex(b)):
                self.assertEqual(r64(op, a, b, c), want)

    def test_fma_f64_rounds_to_binary64(self):
        self.assertEqual(si.fma_f64(2.0 ** -1022, 0.5, 2.0 ** -1022), 1.5 * 2.0 ** -1022)
        self.assertEqual(si.fma_f64(1 + 2.0 ** -30, 1 + 2.0 ** -30, -(1 + 2.0 ** -29)), 2.0 ** -60)


if __name__ == "__main__":
    unittest.main()
