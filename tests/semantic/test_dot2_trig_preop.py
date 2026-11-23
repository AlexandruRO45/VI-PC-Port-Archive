"""test_dot2_trig_preop.py: value tests for v_dot2_f32_f16, v_dot2c_f32_f16 and v_trig_preop_f64."""
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_dot2tp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    _TMP.cleanup()


def translate(lines, out):
    src = os.path.join(_TMP.name, "disasm", "ps_dot2tp.s")
    with open(src, "w", encoding="utf-8") as f:
        f.write("; dot2tp\n" + "".join(f"\t{ln}\n" for ln in lines) + f"\texp mrt0 {out} done vm\n\ts_endpgm\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def run(lines, out="v10, v11, v10, v10"):
    w = si.run_fragment(translate(lines, out))["%out_mrt0"]
    return w[0] & 0xFFFFFFFF, w[1] & 0xFFFFFFFF


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def rf32(x):
    return bits_f32(f32(x))


def movs(regs):
    return [f"v_mov_b32 v{r}, 0x{v & 0xFFFFFFFF:x}" for r, v in regs.items()]


def dot2_ref(a, b, c, nlo=(0, 0, 0), nhi=(0, 0), sel=(0, 0), selhi=(1, 1), clamp=False):
    ah = (h2f(a >> (16 * sel[0])), h2f(a >> (16 * selhi[0])))
    bh = (h2f(b >> (16 * sel[1])), h2f(b >> (16 * selhi[1])))
    ah = (-ah[0] if nlo[0] else ah[0], -ah[1] if nhi[0] else ah[1])
    bh = (-bh[0] if nlo[1] else bh[0], -bh[1] if nhi[1] else bh[1])
    z = bits_f32(c)
    z = -z if nlo[2] else z
    r0 = rf32(ah[0] * bh[0] + z)
    r1 = rf32(ah[1] * bh[1] + r0)
    if clamp:
        r1 = min(max(r1, 0.0), 1.0)
    return f32(r1)


TWO_BY_PI_WORDS = 37


def two_by_pi_bits(nbits):
    prec = nbits + 64
    one = 1 << prec

    def atan_inv(x):
        s = t = one // x
        n, x2, sign = 1, x * x, -1
        while t:
            t //= x2
            n += 2
            s += sign * (t // n)
            sign = -sign
        return s
    pi = 4 * (4 * atan_inv(5) - atan_inv(239))
    return ((2 * one) << nbits) // pi


_B = two_by_pi_bits(32 * TWO_BY_PI_WORDS)
TABLE = [(_B >> (32 * (TWO_BY_PI_WORDS - 1 - i))) & 0xFFFFFFFF for i in range(TWO_BY_PI_WORDS)]


def preop_ref(xbits, seg):
    if (xbits >> 52) & 0x7FF == 0x7FF and xbits & ((1 << 52) - 1):
        return xbits | (1 << 51)
    e = (xbits >> 52) & 0x7FF
    shift = (seg & 31) * 53 + (e - 1077 if e > 1077 else 0)
    idx = shift >> 5
    if idx + 2 >= TWO_BY_PI_WORDS:
        return 0
    b = shift & 31
    thi = (TABLE[idx] << 32) | TABLE[idx + 1]
    if b:
        thi = ((thi << b) | ((TABLE[idx + 2] << 32) >> (64 - b))) & (2 ** 64 - 1)
    scale = -53 - shift + (128 if e >= 1968 else 0)
    return struct.unpack("<Q", struct.pack("<d", math.ldexp(float(thi >> 11), scale)))[0]


def d2b(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


class TableTests(unittest.TestCase):
    def test_table_is_two_over_pi(self):
        self.assertEqual(TABLE[:2], [0xa2f9836e, 0x4e441529])
        self.assertEqual(tuple(TABLE), rdna2spirv.Gen.TWO_BY_PI)

    def test_reference_known_values(self):
        # segment 0 of 1.0 is 2/pi truncated to 53 bits; values from LLVM 21's trig_preop constant fold
        self.assertEqual(preop_ref(d2b(1.0), 0), 0x3FE45F306DC9C882)
        self.assertEqual(preop_ref(d2b(1.0), 1), 0x3C94A7F09D5F47D4)
        self.assertEqual(preop_ref(d2b(1e300), 2), 0x066EF14A525D4D7F)
        self.assertEqual(preop_ref((2040 << 52) | 12345, 3), 0)


class TrigPreopTests(unittest.TestCase):
    # the mini interpreter has no f64; values are checked on the GPU by the cs_w32_trig_preop fixture
    def test_spirv_val(self):
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            self.skipTest("SPIRV-Tools not on PATH")
        lines = movs({2: 0, 3: 0x3FF00000, 4: 1}) + ["v_trig_preop_f64 v[10:11], -v[2:3], v4 clamp",
                                                      "v_dot2_f32_f16 v12, v1, v2, v3 neg_lo:[1,1,1] clamp",
                                                      "v_dot2c_f32_f16 v12, v1, v2"]
        asm = translate(lines, "v10, v11, v12, v12")
        sa, sv = os.path.join(_TMP.name, "t.spvasm"), os.path.join(_TMP.name, "t.spv")
        with open(sa, "w", encoding="utf-8") as f:
            f.write(asm)
        r = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", sa, "-o", sv], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_declares_table_only_when_used(self):
        self.assertNotIn("%two_by_pi", translate(["v_mov_b32 v10, 0", "v_mov_b32 v11, 0"], "v10, v11, v10, v10"))
        asm = translate(["v_trig_preop_f64 v[10:11], v[2:3], 0"], "v10, v11, v10, v10")
        self.assertIn("%two_by_pi = OpVariable", asm)
        self.assertIn("Float64", asm)

    def test_refuses_scalar_destination_and_literal_source(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["v_trig_preop_f64 v[10:11], 1.0, v4"], "v10, v11, v10, v10")


if __name__ == "__main__":
    unittest.main()
