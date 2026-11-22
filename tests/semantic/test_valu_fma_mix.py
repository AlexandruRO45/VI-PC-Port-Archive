"""Semantic interpreter tests for RDNA2 v_fma_mix_f32, v_fma_mixlo_f16 and v_fma_mixhi_f16.

Each VOP3P mixed-precision FMA computes fma(src0, src1, src2) in f32:
- op_sel_hi[i] == 0: src[i] is an f32 word.
- op_sel_hi[i] == 1: src[i] is an f16 packed in the low half (op_sel[i] == 0) or high half (op_sel[i] == 1),
  widened to f32 before applying neg/abs modifiers.
- v_fma_mix_f32 writes the f32 result; v_fma_mixlo_f16 narrows to f16 in the low half (keeping bits [31:16]);
  v_fma_mixhi_f16 narrows to f16 in the high half (keeping bits [15:0]).

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported word with a plain-Python reference.
"""
from fractions import Fraction
import itertools
import math
import os
import re
import struct
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
OLD = 0xCAFE55AA


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def pack_h(lo: float, hi: float) -> int:
    return (f2h(hi) << 16) | f2h(lo)


def flush_f32_bits(b: int) -> int:
    """Flush f32 denormals to signed zero under denorm_mode_32 = 0 (SHADER-298)."""
    return (b & 0x80000000) if (b & 0x7FFFFFFF) < 0x00800000 else b


def unpack_src(word: int, is_f16: int, sel_hi_half: int, neg: bool = False, abs_: bool = False) -> float:
    if is_f16:
        v = h2f((word >> 16) if sel_hi_half else word)
    else:
        v = u2f(flush_f32_bits(word))
    if abs_:
        v = math.copysign(v, 1.0)
    if neg:
        v = -v
    return v


def ref_fma(w0: int, w1: int, w2: int, sel_hi, sel, mods_src=((False, False),) * 3, clamp: bool = False) -> float:
    x = unpack_src(w0, sel_hi[0], sel[0], *mods_src[0])
    y = unpack_src(w1, sel_hi[1], sel[1], *mods_src[1])
    z = unpack_src(w2, sel_hi[2], sel[2], *mods_src[2])
    r = u2f(f2u(float(Fraction(x) * Fraction(y) + Fraction(z))))
    if clamp:
        r = min(max(r, 0.0), 1.0)
    return r


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_fma_mix.s")
    with open(os.path.join(_TMP_DIR.name, "ps_fma_mix.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_and_run(line: str, v1: int, v2: int, v3: int, old: int = OLD):
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"v_mov_b32 v3, {v3:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ]
    asm = (
        "; test_valu_fma_mix\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return spv, si.run_fragment(spv)["%out_mrt0"][0]


def run(line: str, v1: int, v2: int, v3: int, old: int = OLD) -> int:
    return translate_and_run(line, v1, v2, v3, old=old)[1]


class FmaMixF32Tests(unittest.TestCase):
    """v_fma_mix_f32 across op_sel_hi/op_sel combinations, source modifiers and clamp."""

    # Each word is valid both as a finite f32 and as two distinct finite f16 halves.
    W0 = pack_h(1.5, -3.25)    # f16: (1.5, -3.25), also finite f32
    W1 = pack_h(2.25, 0.75)    # f16: (2.25, 0.75)
    W2 = pack_h(-0.5, 4.5)     # f16: (-0.5, 4.5)

    def test_all_f32_default_no_modifiers(self):
        a, b, c = f2u(2.5), f2u(-1.25), f2u(0.75)
        got = run("v_fma_mix_f32 v0, v1, v2, v3", a, b, c)
        want = f2u(ref_fma(a, b, c, (0, 0, 0), (0, 0, 0)))
        self.assertEqual(got, want)

    def test_all_64_op_sel_and_op_sel_hi_combinations(self):
        for sel_hi in itertools.product((0, 1), repeat=3):
            for sel in itertools.product((0, 1), repeat=3):
                # When op_sel_hi[i] == 0, supply a clean f32; when 1, supply packed f16 lo/hi.
                w0 = self.W0 if sel_hi[0] else f2u(1.75)
                w1 = self.W1 if sel_hi[1] else f2u(-2.5)
                w2 = self.W2 if sel_hi[2] else f2u(0.375)
                line = (
                    f"v_fma_mix_f32 v0, v1, v2, v3 "
                    f"op_sel:[{sel[0]},{sel[1]},{sel[2]}] "
                    f"op_sel_hi:[{sel_hi[0]},{sel_hi[1]},{sel_hi[2]}]"
                )
                with self.subTest(sel_hi=sel_hi, sel=sel):
                    got = run(line, w0, w1, w2)
                    want = f2u(ref_fma(w0, w1, w2, sel_hi, sel))
                    self.assertEqual(got, want)

    def test_source_neg_abs_modifiers(self):
        w0 = pack_h(-2.0, 3.0)
        w1 = f2u(-1.5)
        w2 = pack_h(0.25, -4.0)
        line = "v_fma_mix_f32 v0, -|v1|, -v2, |v3| op_sel:[0,0,1] op_sel_hi:[1,0,1]"
        got = run(line, w0, w1, w2)
        want = f2u(ref_fma(w0, w1, w2, (1, 0, 1), (0, 0, 1), mods_src=((True, True), (True, False), (False, True))))
        self.assertEqual(got, want)

    def test_clamp_modifier(self):
        cases = [
            (pack_h(2.0, 0.0), f2u(3.0), pack_h(1.0, 0.0), 1.0),     # > 1.0 -> 1.0
            (pack_h(-2.0, 0.0), f2u(3.0), pack_h(-1.0, 0.0), 0.0),   # < 0.0 -> 0.0
            (pack_h(0.5, 0.0), f2u(0.5), pack_h(0.25, 0.0), 0.5),    # in [0, 1] -> unchanged
        ]
        for w0, w1, w2, expected in cases:
            with self.subTest(expected=expected):
                got = run("v_fma_mix_f32 v0, v1, v2, v3 op_sel_hi:[1,0,1] clamp", w0, w1, w2)
                self.assertEqual(got, f2u(expected))


class FmaMixF16Tests(unittest.TestCase):
    """v_fma_mixlo_f16 and v_fma_mixhi_f16: half preservation, mixed inputs, modifiers, clamp and aliasing."""

    def test_mixlo_and_mixhi_preserve_untouched_half(self):
        w0 = pack_h(1.5, -2.5)
        w1 = f2u(2.0)
        w2 = pack_h(0.25, 3.0)
        for op, dst_hi in (("v_fma_mixlo_f16", 0), ("v_fma_mixhi_f16", 1)):
            for sel in ((0, 0, 0), (1, 0, 1), (1, 0, 0), (0, 0, 1)):
                line = f"{op} v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]}] op_sel_hi:[1,0,1]"
                with self.subTest(op=op, sel=sel):
                    got = run(line, w0, w1, w2, old=OLD)
                    h = f2h(ref_fma(w0, w1, w2, (1, 0, 1), sel))
                    want = ((h << 16) | (OLD & 0xFFFF)) if dst_hi else ((OLD & 0xFFFF0000) | h)
                    self.assertEqual(got, want)

    def test_mixlo_and_mixhi_neg_abs_and_clamp(self):
        w0 = pack_h(-3.0, 1.0)
        w1 = pack_h(2.0, -0.5)
        w2 = f2u(0.5)
        # -|-3.0| * (-0.5) + 0.5 = -3.0 * -0.5 + 0.5 = 2.0 -> clamped to 1.0 (0x3c00)
        got_lo = run("v_fma_mixlo_f16 v0, -|v1|, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] clamp", w0, w1, w2, old=OLD)
        self.assertEqual(got_lo, (OLD & 0xFFFF0000) | f2h(1.0))

        # -|-3.0| * 2.0 + 0.5 = -5.5 -> clamped to 0.0 (0x0000)
        got_hi = run("v_fma_mixhi_f16 v0, -|v1|, v2, v3 op_sel:[0,0,0] op_sel_hi:[1,1,0] clamp", w0, w1, w2, old=OLD)
        self.assertEqual(got_hi, (f2h(0.0) << 16) | (OLD & 0xFFFF))

    def test_destination_aliases_source(self):
        # v0 starts as pack_h(1.5, 4.0); compute mixhi using v0.lo (1.5) * 2.0 + 0.5 = 3.5 -> keep lo (1.5)
        old = pack_h(1.5, 4.0)
        got = run("v_fma_mixhi_f16 v0, v0, v1, v2 op_sel:[0,0,0] op_sel_hi:[1,0,1]", f2u(2.0), pack_h(0.5, 9.0), 0, old=old)
        self.assertEqual(got, pack_h(1.5, 3.5))


class FmaMixDenormalFlushTests(unittest.TestCase):
    """v_fma_mix* flushes f32 denormal operands to signed zero under denorm_mode_32 = 0 (SHADER-298).

    On gfx1036 (RDNA2), reading an f32 denormal operand (such as 0x00000001, 0x807fffff, or integer
    constants 1/64 used as f32) reads as signed zero in the game's MODE (found by @Sanonymous01).
    """

    SUBNORM_POS_MIN = 0x00000001       # 2^-149 (smallest positive subnormal)
    SUBNORM_POS_MAX = 0x007FFFFF       # largest positive subnormal (just below 2^-126)
    SUBNORM_NEG_MIN = 0x80000001       # -2^-149 (smallest negative subnormal)
    SUBNORM_NEG_MAX = 0x807FFFFF       # largest negative subnormal
    FLT_MIN = 0x00800000               # 2^-126 (normal float: must NOT flush)
    FLT_MIN_NEG = 0x80800000           # -2^-126 (normal float: must NOT flush)
    POS_ZERO = 0x00000000
    NEG_ZERO = 0x80000000

    def test_f32_denormal_operands_flush_to_signed_zero(self):
        # fma(src0, src1, src2) = src0 * src1 + src2
        # If an operand is denormal, it is flushed to +0.0 or -0.0 depending on sign.
        two = f2u(2.0)
        three = f2u(3.0)
        five = f2u(5.0)

        cases = [
            # (src0, src1, src2, expected_result)
            # src0 is positive denormal -> +0 * 2.0 + 5.0 = 5.0
            (self.SUBNORM_POS_MIN, two, five, five),
            (self.SUBNORM_POS_MAX, two, five, five),
            # src0 is negative denormal -> -0 * 2.0 + 5.0 = 5.0
            (self.SUBNORM_NEG_MIN, two, five, five),
            (self.SUBNORM_NEG_MAX, two, five, five),
            # src1 is positive denormal -> 2.0 * +0 + 5.0 = 5.0
            (two, self.SUBNORM_POS_MIN, five, five),
            (two, self.SUBNORM_POS_MAX, five, five),
            # src1 is negative denormal -> 2.0 * -0 + 5.0 = 5.0
            (two, self.SUBNORM_NEG_MIN, five, five),
            (two, self.SUBNORM_NEG_MAX, five, five),
            # src2 is positive denormal -> 2.0 * 3.0 + 0 = 6.0
            (two, three, self.SUBNORM_POS_MIN, f2u(6.0)),
            (two, three, self.SUBNORM_POS_MAX, f2u(6.0)),
            # src2 is negative denormal -> 2.0 * 3.0 + (-0) = 6.0
            (two, three, self.SUBNORM_NEG_MIN, f2u(6.0)),
            (two, three, self.SUBNORM_NEG_MAX, f2u(6.0)),
            # Flush to zero multiplying to zero: denorm * 1.0 + 0.0 -> 0.0
            (self.SUBNORM_POS_MIN, f2u(1.0), self.POS_ZERO, self.POS_ZERO),
            (self.SUBNORM_POS_MAX, f2u(1.0), self.POS_ZERO, self.POS_ZERO),
        ]
        for s0, s1, s2, want in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = run("v_fma_mix_f32 v0, v1, v2, v3", s0, s1, s2)
                self.assertEqual(got, want)

    def test_signed_zero_preserved_when_flushing(self):
        # 1.0 * (-denorm) + (-0.0) -> 1.0 * (-0.0) + (-0.0) = -0.0
        got = run("v_fma_mix_f32 v0, v1, v2, v3", f2u(1.0), self.SUBNORM_NEG_MAX, self.NEG_ZERO)
        self.assertEqual(got, self.NEG_ZERO)

        # -1.0 * (+denorm) + (-0.0) -> -1.0 * (+0.0) + (-0.0) = -0.0
        got = run("v_fma_mix_f32 v0, v1, v2, v3", f2u(-1.0), self.SUBNORM_POS_MIN, self.NEG_ZERO)
        self.assertEqual(got, self.NEG_ZERO)

        # -1.0 * (-denorm) + (+0.0) -> -1.0 * (-0.0) + (+0.0) = +0.0
        got = run("v_fma_mix_f32 v0, v1, v2, v3", f2u(-1.0), self.SUBNORM_NEG_MAX, self.POS_ZERO)
        self.assertEqual(got, self.POS_ZERO)

    def test_integer_inline_constants_used_as_f32(self):
        # Integer constants 1 (0x00000001) and 64 (0x00000040) used as f32 are subnormals on RDNA2.
        # Under denorm_mode_32 = 0 they read as +0.0.
        two = f2u(2.0)
        three = f2u(3.0)
        seven = f2u(7.0)
        # 1 * 2.0 + 7.0 -> 0.0 * 2.0 + 7.0 = 7.0
        got = run("v_fma_mix_f32 v0, 1, v2, v3", 0, two, seven)
        self.assertEqual(got, seven)
        # 64 * 2.0 + 7.0 -> 0.0 * 2.0 + 7.0 = 7.0
        got = run("v_fma_mix_f32 v0, 64, v2, v3", 0, two, seven)
        self.assertEqual(got, seven)
        # 2.0 * 1 + 7.0 -> 2.0 * 0.0 + 7.0 = 7.0
        got = run("v_fma_mix_f32 v0, v1, 1, v3", two, 0, seven)
        self.assertEqual(got, seven)
        # 2.0 * 3.0 + 1 -> 6.0 + 0.0 = 6.0
        got = run("v_fma_mix_f32 v0, v1, v2, 1", two, three, 0)
        self.assertEqual(got, f2u(6.0))

    def test_flt_min_boundary_not_flushed(self):
        # FLT_MIN (0x00800000, 2^-126) is the smallest normal float and must NOT be flushed.
        got = run("v_fma_mix_f32 v0, v1, v2, v3", self.FLT_MIN, f2u(1.0), self.POS_ZERO)
        self.assertEqual(got, self.FLT_MIN)

        got = run("v_fma_mix_f32 v0, v1, v2, v3", self.FLT_MIN_NEG, f2u(1.0), self.POS_ZERO)
        self.assertEqual(got, self.FLT_MIN_NEG)

    def test_fma_mix_f16_with_f32_denormal_operand(self):
        # v_fma_mixlo_f16 / v_fma_mixhi_f16: when an operand is f32 (op_sel_hi[i] == 0),
        # an f32 denormal must flush to signed zero.
        h_1_5 = pack_h(1.5, 0.0)
        h_2_0 = pack_h(2.0, 0.0)
        got = run(
            "v_fma_mixlo_f16 v0, v1, v2, v3 op_sel:[0,0,0] op_sel_hi:[0,1,1]",
            self.SUBNORM_POS_MIN, h_1_5, h_2_0, old=OLD
        )
        self.assertEqual(got, (OLD & 0xFFFF0000) | f2h(2.0))

        got_hi = run(
            "v_fma_mixhi_f16 v0, v1, v2, v3 op_sel:[0,0,0] op_sel_hi:[0,1,1]",
            self.SUBNORM_NEG_MAX, h_1_5, h_2_0, old=OLD
        )
        self.assertEqual(got_hi, (f2h(2.0) << 16) | (OLD & 0xFFFF))


class NegativeControls(unittest.TestCase):
    """Sabotaging the translated SPIR-V must change the observed output."""

    def test_sabotage_mixhi_half_mask_detected(self):
        w0, w1, w2 = pack_h(1.5, 2.0), f2u(2.0), pack_h(0.5, 1.0)
        spv, good = translate_and_run("v_fma_mixhi_f16 v0, v1, v2, v3 op_sel_hi:[1,0,1]", w0, w1, w2, old=OLD)
        self.assertEqual(good, (f2h(3.5) << 16) | (OLD & 0xFFFF))
        # Replacing the preserved-low-half mask 0xFFFF with 0 wipes OLD's low half.
        bad, n = re.subn(r"(OpConstant %uint )65535\b", r"\g<1>0", spv)
        self.assertGreaterEqual(n, 1)
        self.assertNotEqual(si.run_fragment(bad)["%out_mrt0"][0], good)

    def test_sabotage_f32_denormal_flush_detected(self):
        # Without flush_f32, 2^-149 * 1.0 + 0.0 yields 2^-149 (0x00000001); with flush it yields 0.
        spv, good = translate_and_run(
            "v_fma_mix_f32 v0, v1, v2, v3",
            0x00000001, f2u(1.0), 0
        )
        self.assertEqual(good, 0)
        # Sabotage: replace the OpSelect in flush_f32 with the unfiltered float operand
        bad, n = re.subn(r"(%\w+) = OpSelect %float (%\w+) (%\w+) (%\w+)", r"\1 = OpCopyObject %float \4", spv)
        self.assertGreaterEqual(n, 1)
        res = si.run_fragment(bad)["%out_mrt0"][0]
        self.assertEqual(res, 0x00000001)
        self.assertNotEqual(res, good)


if __name__ == "__main__":
    unittest.main()
