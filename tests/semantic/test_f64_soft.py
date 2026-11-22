"""--f64-soft: binary64 ops as 32-bit integer arithmetic (tools/soft_f64.py), for devices without shaderFloat64.

Hand-written programs run through the translator with --f64-soft and through spirv_mini_interp.py; every exported
word is compared with an exact reference written here with fractions.Fraction (it shares no code with
tools/soft_f64.py): IEEE-754 binary64 rounding in MODE.FP_ROUND's four modes, f64 denormals kept, -0, overflow,
NaN (an operand NaN comes back quiet, src0 first; inf - inf is 0x7ff80000_00000000), the VOP3 neg/abs input
modifiers on the sign bit, omod (rounded to nearest even, as the native OpFMul) and clamp (+0 for NaN and -0).
A few words are also written down by hand, to pin the reference itself.

    python -I tests/semantic/test_f64_soft.py
"""
import contextlib
import os
import random
import struct
import sys
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))

import rdna2spirv  # noqa: E402
from support import mrt0_words, translate_fixture, translate_lines  # noqa: E402

M64 = (1 << 64) - 1
SIGN = 1 << 63
QUIET = 1 << 51
DEFAULT_NAN = 0x7FF8000000000000


@contextlib.contextmanager
def soft():
    old = rdna2spirv.F64_SOFT
    rdna2spirv.F64_SOFT = True
    try:
        yield
    finally:
        rdna2spirv.F64_SOFT = old


# ---- the reference --------------------------------------------------------------------------------------------
def is_nan(b):
    return (b >> 52) & 0x7FF == 0x7FF and b & ((1 << 52) - 1) != 0


def is_inf(b):
    return (b >> 52) & 0x7FF == 0x7FF and b & ((1 << 52) - 1) == 0


def value(b):
    s = -1 if b >> 63 else 1
    e, m = (b >> 52) & 0x7FF, b & ((1 << 52) - 1)
    if e == 0:
        return s * Fraction(m, 1 << 1074)
    return s * Fraction(m | 1 << 52) * Fraction(2) ** (e - 1075)


def round64(q, mode):
    """A nonzero Fraction -> binary64 bits: mode 0 nearest even, 1 toward +inf, 2 toward -inf, 3 toward zero."""
    neg = q < 0
    a = -q if neg else q
    e = 1                                   # the significand a / 2**(e - 1075) lies in [2**52, 2**53) when normal
    while a >= Fraction(2) ** (e - 1022):
        e += 1
    scaled = a / Fraction(2) ** (e - 1075)
    m, rem = divmod(scaled.numerator, scaled.denominator)
    rem = Fraction(rem, scaled.denominator)
    up = {0: rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m & 1),
          1: rem > 0 and not neg, 2: rem > 0 and neg, 3: False}[mode]
    m += up
    if m == 1 << 53:
        m, e = 1 << 52, e + 1
    s = SIGN if neg else 0
    if e >= 0x7FF:
        to_max = mode == 3 or (mode == 1 and neg) or (mode == 2 and not neg)
        return s | (0x7FEFFFFFFFFFFFFF if to_max else 0x7FF0000000000000)
    return s | ((e << 52) + (m - (1 << 52)) if m >= 1 << 52 else m)


def add(x, y, mode=0):
    if is_nan(x):
        return x | QUIET
    if is_nan(y):
        return y | QUIET
    if is_inf(x) and is_inf(y):
        return x if x >> 63 == y >> 63 else DEFAULT_NAN
    if is_inf(x):
        return x
    if is_inf(y):
        return y
    q = value(x) + value(y)
    if q != 0:
        return round64(q, mode)
    if x >> 63 == y >> 63:                  # both zero, same sign
        return x & SIGN
    return SIGN if mode == 2 else 0


def scale(r, k):
    if is_nan(r):
        return r | QUIET
    if is_inf(r) or r & ~SIGN == 0:
        return r
    return round64(value(r) * Fraction(2) ** k, 0)


def clamp(r):
    if is_nan(r) or r >> 63 or r == 0:
        return 0
    return min(r, 0x3FF0000000000000)


def to_f32(b):
    s = (b >> 63) << 31
    if is_nan(b):
        return s | 0x7FC00000 | ((b >> 29) & 0x7FFFFF)
    if is_inf(b):
        return s | 0x7F800000
    return round32(abs(value(b)), s)


def value32(w):
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    q = Fraction(m, 1 << 149) if e == 0 else Fraction(m | 1 << 23) * Fraction(2) ** (e - 150)
    return -q if w >> 31 else q


def scale32(w, k):
    """VOP3 omod on binary32 bits: w * 2**k rounded to nearest even, f32 denormals kept, ±0 and ±inf kept, a NaN
    quiet with its payload."""
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    if e == 0xFF:
        return w | 0x400000 if m else w
    return round32(abs(value32(w)) * Fraction(2) ** k, w & 0x80000000)


def clamp32(w):
    if (w & 0x7FFFFFFF) > 0x7F800000 or w >> 31 or w == 0:
        return 0
    return min(w, 0x3F800000)


def round32(q, s):
    """A Fraction q >= 0 -> binary32 bits with sign bit s, nearest even, denormals kept, overflow to infinity."""
    if q == 0:
        return s
    e = 1                                   # binary32: significand in [2**23, 2**24), denormal scale 2**-149
    while q >= Fraction(2) ** (e - 126):
        e += 1
    scaled = q / Fraction(2) ** (e - 150)
    m, rem = divmod(scaled.numerator, scaled.denominator)
    rem = Fraction(rem, scaled.denominator)
    m += rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m & 1)
    if m == 1 << 24:
        m, e = 1 << 23, e + 1
    if e >= 0xFF:
        return s | 0x7F800000
    return s | ((e << 23) + (m - (1 << 23)) if m >= 1 << 23 else m)


# ---- programs ------------------------------------------------------------------------------------------------------
def movs(reg, bits):
    return [f"v_mov_b32 v{reg}, {bits & 0xFFFFFFFF:#x}", f"v_mov_b32 v{reg + 1}, {bits >> 32:#x}"]


def run(lines):
    with soft():
        text = translate_lines("ps_f64_soft_case", lines + ["s_endpgm"])
    assert "Float64" not in text and "OpTypeFloat 64" not in text and "Int64" not in text, "soft output uses f64"
    return mrt0_words(text)


def add_pair(x1, y1, x2, y2, suffix1="", suffix2="", mode=None, src1=("v[0:1]", "v[2:3]"), src2=("v[4:5]", "v[6:7]")):
    """Two v_add_f64 results (each (lo, hi)) from one program; `mode` writes MODE.FP_ROUND[3:2] first."""
    lines = movs(0, x1) + movs(2, y1) + movs(4, x2) + movs(6, y2)
    if mode is not None:
        lines.append(f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}")
    lines += [f"v_add_f64 v[8:9], {src1[0]}, {src1[1]} {suffix1}".strip(),
              f"v_add_f64 v[10:11], {src2[0]}, {src2[1]} {suffix2}".strip(),
              "exp mrt0 v8, v9, v10, v11 done vm"]
    w = run(lines)
    return w[0] | w[1] << 32, w[2] | w[3] << 32


def h(b):
    return f"{b:#018x}"


DMAX = 0x7FEFFFFFFFFFFFFF
DMIN_NORMAL = 0x0010000000000000
DENORM_MAX = 0x000FFFFFFFFFFFFF
ONE = 0x3FF0000000000000
INF = 0x7FF0000000000000


class ReferenceAnchors(unittest.TestCase):
    """Words worked out by hand, so the Fraction reference is pinned to something outside itself."""

    def test_anchor_words(self):
        self.assertEqual(add(ONE, 0x3CB8000000000000), 0x3FF0000000000002)           # 1 + 1.5 ulp: tie to even
        self.assertEqual(add(ONE, 0x3CA0000000000000), ONE)                          # 1 + 0.5 ulp: tie to even
        self.assertEqual(add(0x3FF0000000000002, 0x3CAFFFFFFFFFFFFF, 3), 0x3FF0000000000002)   # toward zero
        self.assertEqual(add(0x3FF0000000000002, 0x3CAFFFFFFFFFFFFF, 0), 0x3FF0000000000003)
        self.assertEqual(add(DMAX, DMAX), INF)
        self.assertEqual(add(DMAX, DMAX, 3), DMAX)
        self.assertEqual(add(1, 1), 2)                                               # denormals kept
        self.assertEqual(add(DENORM_MAX, 1), DMIN_NORMAL)                            # denormal sum becomes normal
        self.assertEqual(add(ONE, ONE | SIGN), 0)
        self.assertEqual(add(ONE, ONE | SIGN, 2), SIGN)
        self.assertEqual(add(SIGN, SIGN), SIGN)
        self.assertEqual(add(INF, INF | SIGN), DEFAULT_NAN)
        self.assertEqual(scale(DMIN_NORMAL | 1, -1), 0x0008000000000000)             # 0.5 ulp of a denormal: even
        self.assertEqual(scale(DMIN_NORMAL | 3, -1), 0x0008000000000002)
        self.assertEqual(to_f32(0x3FF0000010000000), 0x3F800000)
        self.assertEqual(to_f32(0x380FFFFFE0000000), 0x00800000)                     # rounds up to FLT_MIN
        self.assertEqual(to_f32(0x36A0000000000000), 0x00000001)                     # 2**-149
        self.assertEqual(to_f32(0x3690000000000000), 0x00000000)                     # 2**-150: tie to even, 0


class SoftF64Fixtures(unittest.TestCase):
    """The GPU fixtures that use f64, translated with --f64-soft: the words their .asm headers give."""

    def fixture(self, name):
        with soft():
            text = translate_fixture(name)
        self.assertNotIn("Float64", text)
        self.assertNotIn("OpTypeFloat 64", text)
        return mrt0_words(text)

    def test_ps_f64_mods(self):
        self.assertEqual(self.fixture("ps_f64_mods"), [0x40040000, 0x3FF00000, 0x3FE40000, 0x3FF00000])

    def test_ps_f64_ties(self):
        self.assertEqual(self.fixture("ps_f64_ties"), [0x00000002, 0x80000000, 0x00000002, 0x007FEFFF])

    def test_ps_f64_round(self):
        """main's toward-zero extremes near DBL_MAX (#2891's TwoSum case), the words its .asm header gives."""
        self.assertEqual(self.fixture("ps_f64_round"), [0xC12AF665, 0x3FE7FFFF, 0xBFFFFFFF, 0xC12AF666])

    def test_ps_f64_cvt(self):
        self.assertEqual(self.fixture("ps_f64_cvt"), [0x3F800000, 0x3F800001, 0xFF800000, 0xBF400000])

    def test_ps_f64_cvt_mods(self):
        self.assertEqual(self.fixture("ps_f64_cvt_mods"), [0x3FC00000, 0x3F800000, 0x00000000, 0x7F800000])

    def test_ps_f64_frexp_lit(self):
        self.assertEqual(self.fixture("ps_f64_frexp_lit"), [0x00000000, 0x3FF80000, 0x00000001, 0x00000002])

    def test_without_the_flag_the_output_is_native(self):
        text = translate_fixture("ps_f64_mods")
        self.assertIn("OpCapability Float64", text)
        self.assertIn("OpFAdd %double", text)
        self.assertNotIn("%f64s_", text)

    def test_an_op_without_a_soft_form_is_refused(self):
        """Under --f64-soft a module never declares Float64: native double code is Unsupported, not emitted."""
        with soft():
            gen_op = rdna2spirv.Gen.f64_soft_op
            try:
                rdna2spirv.Gen.f64_soft_op = lambda g, op, a, mods: setattr(g, "uses_f64", True)
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_fixture("ps_f64_mods")
            finally:
                rdna2spirv.Gen.f64_soft_op = gen_op

    def test_function_types_are_distinct(self):
        """spirv-val rejects two OpTypeFunction with the same signature: scale_f32 shares to_f32's type."""
        with soft():
            text = translate_fixture("ps_f64_cvt_mods")
        sigs = [ln.split("OpTypeFunction", 1)[1].split() for ln in text.splitlines() if "OpTypeFunction" in ln]
        self.assertEqual(len(sigs), len({tuple(x) for x in sigs}), sigs)
        self.assertIn("%f64s_scale_f32 = OpFunction %uint Pure %fn_f64s_to_f32", text)

    def test_one_helper_per_module(self):
        """Every v_add_f64 calls the same function; it is emitted once."""
        with soft():
            text = translate_fixture("ps_f64_mods")
        self.assertEqual(text.count("%f64s_add = OpFunction"), 1)
        self.assertEqual(text.count("OpFunctionCall %v2uint_d %f64s_add"), 4)
        self.assertEqual(text.count("%f64s_scale = OpFunction"), 1)


class SoftF64Add(unittest.TestCase):
    """v_add_f64 edge cases: each pair is one program, both results checked against the reference."""

    def check(self, x1, y1, x2, y2, mode=None, **kw):
        m = 0 if mode is None else mode
        got = add_pair(x1, y1, x2, y2, mode=mode, **kw)
        want = (add(x1, y1, m), add(x2, y2, m))
        self.assertEqual([h(g) for g in got], [h(w) for w in want], f"{h(x1)} + {h(y1)}, {h(x2)} + {h(y2)}, mode {m}")

    def test_ties_to_even(self):
        self.check(ONE, 0x3CB8000000000000, ONE, 0x3CA0000000000000)
        self.check(0x3FF0000000000001, 0x3CA0000000000000, 0xBFF0000000000001, 0xBCA0000000000000)

    def test_denormals(self):
        self.check(1, 1, DENORM_MAX, 1)
        self.check(DMIN_NORMAL, 1 | SIGN, DENORM_MAX | SIGN, DMIN_NORMAL)
        self.check(0x0000000080000001, 0x0000000000000003, 0x0010000000000001, 0x8010000000000000)

    def test_cancellation_and_signed_zero(self):
        self.check(ONE, ONE | SIGN, SIGN, SIGN)
        self.check(0, SIGN, 0x4330000000000001, 0xC330000000000000)
        self.check(ONE, ONE | SIGN, 0, SIGN, mode=3)

    def test_overflow(self):
        self.check(DMAX, DMAX, DMAX | SIGN, DMAX | SIGN)
        self.check(DMAX, DMAX, DMAX | SIGN, DMAX | SIGN, mode=3)
        self.check(DMAX, 0x7C90000000000000, DMAX, 0x7C9FFFFFFFFFFFFF)               # 0.5 ulp, just below

    def test_toward_zero(self):
        self.check(0x3FF0000000000002, 0x3CAFFFFFFFFFFFFF, 0xBFF0000000000002, 0xBCAFFFFFFFFFFFFF, mode=3)
        self.check(ONE, 0xBC90000000000000, 0x7FDD9BB47DAA1333, 0xFFEFFFFFFFFFFFFF, mode=3)

    def test_directed_rounding(self):
        """Modes 1 and 2: the translator refuses a MODE write of them today, but the function takes any mode."""
        import soft_f64

        class G:
            def __init__(self):
                self.body, self.n, self.consts = [], 0, {}

            def tmp(self):
                self.n += 1
                return f"%t{self.n}"

            def cu(self, v):
                return self.consts.setdefault(v & 0xFFFFFFFF, f"%c{v & 0xFFFFFFFF:x}")

            def emit(self, s):
                self.body.append(s)

        sf = soft_f64.SoftF64(G())
        sf.add_f64(("%a0", "%a1"), ("%b0", "%b1"), "%m")
        fn = sf.fns["%f64s_add"]
        for x, y in ((ONE, 0x3C90000000000000), (ONE | SIGN, 0xBC90000000000000), (DMAX, DMAX), (1, 1 | SIGN)):
            for mode in (1, 2):
                got = eval_fn(fn, sf.g.consts, [x & 0xFFFFFFFF, x >> 32, y & 0xFFFFFFFF, y >> 32, mode])
                self.assertEqual(h(got[0] | got[1] << 32), h(add(x, y, mode)), f"{h(x)} + {h(y)} mode {mode}")

    def test_infinity_and_nan(self):
        self.check(INF, INF | SIGN, INF, ONE)
        self.check(0x7FF0000000000001, ONE, ONE, 0xFFF4000000001234)                 # signalling NaNs come back quiet
        self.check(0x7FF8000000000005, 0xFFF8000000000006, INF | SIGN, 0x7FF0000000000009)   # src0's NaN first

    def test_source_modifiers(self):
        """neg/abs flip or clear the sign bit, NaNs included."""
        x, y = 0x400921FB54442D18, 0xC005BF0A8B145769
        got = add_pair(x, y, 0x7FF0000000000001, ONE, src1=("-|v[0:1]|", "-v[2:3]"), src2=("-v[4:5]", "|v[6:7]|"))
        self.assertEqual([h(g) for g in got], [h(add(x | SIGN, y ^ SIGN)), h(0xFFF8000000000001)])

    def test_output_modifiers(self):
        cases = [(DMIN_NORMAL | 1, 0, "div:2"), (DMIN_NORMAL | 3, 0, "div:2"), (0x7FD0000000000000, 0x7FD0000000000000, "mul:4"),
                 (0x3FE0000000000000, 0x3FD0000000000000, "mul:2"), (1, 0, "mul:4"), (0x7FF0000000000001, 0, "mul:2")]
        for (x1, y1, m1), (x2, y2, m2) in zip(cases[0::2], cases[1::2]):
            got = add_pair(x1, y1, x2, y2, suffix1=m1, suffix2=m2)
            k = {"mul:2": 1, "mul:4": 2, "div:2": -1}
            want = (scale(add(x1, y1), k[m1]), scale(add(x2, y2), k[m2]))
            self.assertEqual([h(g) for g in got], [h(w) for w in want], (m1, m2))

    def test_clamp(self):
        rows = [(0x7FF0000000000001, 0), (SIGN, SIGN), (INF, 0), (0x3FE0000000000000, 0x3FE0000000000001),
                (0xBFF0000000000000, 0x3FE0000000000000), (1, 0), (DMAX, DMAX), (INF | SIGN, ONE)]
        for (x1, y1), (x2, y2) in zip(rows[0::2], rows[1::2]):
            got = add_pair(x1, y1, x2, y2, suffix1="clamp", suffix2="clamp div:2")
            want = (clamp(add(x1, y1)), clamp(scale(add(x2, y2), -1)))
            self.assertEqual([h(g) for g in got], [h(w) for w in want], (h(x1), h(y1), h(x2), h(y2)))

    def test_toward_zero_then_modifiers(self):
        got = add_pair(0x3FF0000000000002, 0x3CAFFFFFFFFFFFFF, DMAX, DMAX, suffix1="mul:2", suffix2="div:2", mode=3)
        self.assertEqual([h(g) for g in got], [h(scale(add(0x3FF0000000000002, 0x3CAFFFFFFFFFFFFF, 3), 1)),
                                               h(scale(add(DMAX, DMAX, 3), -1))])

    def test_inline_constants(self):
        """An inline float constant is its binary64 value (as on the native path)."""
        lines = movs(0, 0x3FF8000000000000) + ["v_add_f64 v[2:3], v[0:1], 0.5", "v_add_f64 v[4:5], -4.0, v[0:1]",
                                               "exp mrt0 v2, v3, v4, v5 done vm"]
        w = run(lines)
        self.assertEqual([h(w[0] | w[1] << 32), h(w[2] | w[3] << 32)],
                         [h(add(0x3FF8000000000000, 0x3FE0000000000000)), h(add(0xC010000000000000, 0x3FF8000000000000))])

    def test_random_pairs(self):
        rng = random.Random(64)
        for _ in range(12):
            vals = []
            for _ in range(4):
                e = rng.choice((rng.randrange(0, 2047), rng.randrange(1000, 1050), rng.randrange(0, 3)))
                vals.append(rng.getrandbits(1) << 63 | e << 52 | rng.getrandbits(52))
            if rng.random() < 0.5:                    # nearby exponents, so the sum rounds
                vals[1] = vals[1] & ~(0x7FF << 52) | (((vals[0] >> 52) & 0x7FF) ^ rng.randrange(4)) << 52
            self.check(*vals, mode=rng.choice((None, 3)))


class SoftF64Operands(unittest.TestCase):
    def test_literal_and_inline_source_words(self):
        """fixtures/f64_literals.asm (test_vop3_const_mods): a 32-bit literal is the upper word, an inline integer is
        sign-extended to 64 bits, an inline float is its binary64 value; the soft path reads the same words."""
        path = os.path.join(HERE, "fixtures", "f64_literals.asm")
        with open(path, encoding="utf-8") as f:
            lines = [ln.split(";")[0] for ln in f.read().splitlines() if ln and not ln.startswith(";")]
        w = run(lines + ["exp mrt0 v0, v1, v2, v3 done vm"])
        self.assertEqual([hex(x) for x in w], [hex(to_f32(b)) for b in
                                               (0x3FF8000000000000, 0xFFF0000000000000, 64, 0xFFFFFFFFFFFFFFFF)])
        w = run(lines + ["exp mrt0 v4, v5, v6, v7 done vm"])
        self.assertEqual([h(w[0] | w[1] << 32), h(w[2] | w[3] << 32)],
                         [h(add(0x0000000100000000, 1)), h(add(ONE, 0xBFF8000000000000))])

    def test_frexp_mant_output_modifiers(self):
        """v_frexp_mant_f64 is integer code; its omod/clamp go through the same soft functions as v_add_f64."""
        pi = 0x400921FB54442D18
        lines = movs(0, pi) + movs(2, pi | SIGN) + ["v_frexp_mant_f64 v[4:5], v[0:1] mul:2",
                                                    "v_frexp_mant_f64 v[6:7], v[2:3] clamp",
                                                    "exp mrt0 v4, v5, v6, v7 done vm"]
        w = run(lines)
        self.assertEqual([h(w[0] | w[1] << 32), h(w[2] | w[3] << 32)], [h(0x3FF921FB54442D18), h(0)])

    def test_ops_without_a_soft_form_are_refused(self):
        for op in ("v_rcp_f64", "v_rsq_f64", "v_sqrt_f64"):
            with self.subTest(op=op), soft(), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines("ps_f64_soft_case", movs(0, ONE) + [f"{op} v[2:3], v[0:1]",
                                                                    "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])


class SoftF64Frexp(unittest.TestCase):
    """v_frexp_mant_f64 / v_frexp_exp_i32_f64 with a 32-bit literal: llvm-mc encodes it as bits 63:32 of the f64
    source, bits 31:0 zero (as for v_add_f64), so 0x3ff80000 is 1.5 = 0.75 * 2**1, not a denormal."""

    LINES = ["v_frexp_mant_f64_e64 v[0:1], 0x3ff80000 mul:2", "v_frexp_mant_f64_e64 v[2:3], 0x3ff80000",
             "v_frexp_exp_i32_f64_e64 v4, 0x3ff80000", "v_frexp_exp_i32_f64_e64 v5, 0xc0080000"]

    def test_literal_is_the_upper_word_soft(self):
        w = run(self.LINES + ["exp mrt0 v0, v1, v4, v5 done vm"])
        self.assertEqual([h(w[0] | w[1] << 32), w[2], w[3]], [h(0x3FF8000000000000), 1, 2])     # 1.5; 1.5, -3.0
        w = run(self.LINES + ["exp mrt0 v2, v3, v4, v4 done vm"])
        self.assertEqual(h(w[0] | w[1] << 32), h(0x3FE8000000000000))                          # 0.75

    def test_literal_is_the_upper_word_native(self):
        """The source decode is integer code shared by both paths: without the flag the words are the same."""
        text = translate_lines("ps_f64_soft_case", self.LINES[1:] + ["exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"])
        self.assertNotIn("Float64", text)
        w = mrt0_words(text)
        self.assertEqual([h(w[0] | w[1] << 32), w[2], w[3]], [h(0x3FE8000000000000), 1, 2])


class SoftF64Capabilities(unittest.TestCase):
    """The soft lowering adds neither Float64 nor Int64; a feature that declares Int64 on its own still may."""

    def test_pure_f64_shader_declares_neither(self):
        with soft():
            text = translate_fixture("ps_f64_mods")
        self.assertNotIn("OpCapability Float64", text)
        self.assertNotIn("OpCapability Int64", text)
        self.assertNotIn("OpTypeInt 64", text)

    def test_scalar_buffer_load_keeps_its_own_int64(self):
        """s_load_dword addresses memory through a 64-bit physical pointer and declares Int64 with or without the
        flag; v_add_f64 next to it adds no Float64 under --f64-soft."""
        lines = ["s_load_dword s4, s[0:1], 0x0", "s_waitcnt lgkmcnt(0)", "v_mov_b32 v0, s4", "v_mov_b32 v1, 0x3ff00000",
                 "v_add_f64 v[2:3], v[0:1], v[0:1]", "exp mrt0 v2, v3, v0, v1 done vm", "s_endpgm"]
        native = translate_lines("ps_f64_soft_case", lines)
        with soft():
            text = translate_lines("ps_f64_soft_case", lines)
        self.assertIn("OpCapability Float64", native)
        self.assertIn("OpCapability Int64", native)
        self.assertNotIn("OpCapability Float64", text)
        self.assertNotIn("OpTypeFloat 64", text)
        self.assertIn("OpCapability Int64", text)
        self.assertIn("%f64s_add = OpFunction", text)


class SoftF64Convert(unittest.TestCase):
    def test_cvt_f32_f64(self):
        rows = [0x3FF0000010000000, 0x3FF0000010000001, 0x380FFFFFE0000000, 0x36A0000000000000, 0x3690000000000001,
                0x47EFFFFFF0000000, 0x47EFFFFFEFFFFFFF, 1, SIGN, INF | SIGN, 0x7FF0000000000001, 0xFFF8000123456789]
        for i in range(0, len(rows), 4):
            lines = []
            for j, b in enumerate(rows[i:i + 4]):
                lines += movs(2 * j, b)
            lines += [f"v_cvt_f32_f64 v{8 + j}, v[{2 * j}:{2 * j + 1}]" for j in range(4)]
            lines.append("exp mrt0 v8, v9, v10, v11 done vm")
            self.assertEqual([hex(w) for w in run(lines)], [hex(to_f32(b)) for b in rows[i:i + 4]])

    def test_cvt_modifiers(self):
        lines = movs(0, 0xC000000000000000) + movs(2, 0x3FE0000000000001) + [
            "v_cvt_f32_f64 v4, -|v[0:1]|", "v_cvt_f32_f64 v5, -v[2:3]", "v_cvt_f32_f64 v6, v[0:1] clamp",
            "v_cvt_f32_f64 v7, v[2:3] mul:2", "exp mrt0 v4, v5, v6, v7 done vm"]
        self.assertEqual([hex(w) for w in run(lines)], ["0xc0000000", "0xbf000000", "0x0", "0x3f800000"])

    def test_cvt_output_modifiers_on_special_values(self):
        """omod then clamp on the f32 result, in integer code: clamp gives +0 for NaN and -0 (GLSL FClamp leaves NaN
        undefined, and the interpreter refuses it), omod keeps ±0 and ±inf, quiets a NaN, rounds an f32 denormal to
        nearest even and overflows to ±inf."""
        nan_s, nan_q = 0x7FF0000000000001, 0xFFF8000123456789
        rows = [(nan_s, "clamp"), (nan_q, "clamp mul:2"), (SIGN, "clamp"), (0, "clamp"),
                (SIGN, "mul:2"), (SIGN, "div:2"), (0, "mul:4"), (nan_q, "mul:2"),
                (nan_s, "div:2"), (INF | SIGN, "mul:4"), (INF, "clamp"), (INF | SIGN, "clamp"),
                (0x36A8000000000000, "div:2"), (0x36A0000000000000, "div:2"), (0x36B8000000000000, "div:2"),
                (0x36A8000000000000, "mul:2"),
                (0x380FFFFFC0000000, "mul:2"), (0x3810000000000000, "div:2"), (0x36A0000000000000, "clamp"),
                (0x36A0000000000000 | SIGN, "clamp mul:4"),
                (0x47EFFFFFE0000000, "mul:2"), (0xC7EFFFFFE0000000, "mul:4"), (0x47E0000000000000, "div:2"),
                (0x4000000000000000, "clamp"),
                (0xC008000000000000, "clamp"), (0x3FE0000000000000, "clamp mul:4"), (0x3FD0000000000000, "clamp mul:2"),
                (0x3FEFFFFFF0000000, "clamp mul:2"),
                (0x7FEFFFFFFFFFFFFF, "clamp"), (0x7FEFFFFFFFFFFFFF, "div:2"), (0x3FF0000000000001, "clamp div:2"),
                (0x3FF8000010000000, "mul:4")]
        k = {"mul:2": 1, "mul:4": 2, "div:2": -1}
        for i in range(0, len(rows), 4):
            lines, want = [], []
            for j, (b, mods) in enumerate(rows[i:i + 4]):
                lines += movs(2 * j, b)
                w = to_f32(b)
                for m in mods.split():
                    if m in k:
                        w = scale32(w, k[m])
                want.append(clamp32(w) if "clamp" in mods else w)
            lines += [f"v_cvt_f32_f64 v{8 + j}, v[{2 * j}:{2 * j + 1}] {r[1]}" for j, r in enumerate(rows[i:i + 4])]
            lines.append("exp mrt0 v8, v9, v10, v11 done vm")
            with self.subTest(rows=[(h(b), m) for b, m in rows[i:i + 4]]):
                self.assertEqual([hex(w) for w in run(lines)], [hex(w) for w in want])

    def test_cvt_output_modifier_anchors(self):
        """A few of those words by hand, so scale32/clamp32 are pinned too."""
        self.assertEqual(clamp32(0x7FC00000), 0)
        self.assertEqual(clamp32(0x80000000), 0)
        self.assertEqual(clamp32(0x7F800000), 0x3F800000)
        self.assertEqual(scale32(0x80000000, 1), 0x80000000)                          # -0 kept, as OpFMul gives
        self.assertEqual(scale32(3, -1), 2)                                           # 1.5 * 2**-149: tie to even
        self.assertEqual(scale32(1, -1), 0)                                           # 0.5 * 2**-149: tie to even, +0
        self.assertEqual(scale32(0x80000001, -1), 0x80000000)
        self.assertEqual(scale32(0x007FFFFF, 1), 0x00FFFFFE)                          # denormal * 2: exact, normal
        self.assertEqual(scale32(0x7F7FFFFF, 1), 0x7F800000)
        self.assertEqual(scale32(0x7F800001, 1), 0x7FC00001)


# ---- a direct evaluator for one helper function (straight-line uint code) -----------------------------------------
def eval_fn(lines, consts, args, fns=None):
    """Evaluate a soft_f64 helper's body: the handful of 32-bit opcodes it uses, on Python ints. `fns` holds the
    helpers it may call."""
    env = {name: v for v, name in consts.items()}
    params = [ln.split(" = ")[0] for ln in lines if "OpFunctionParameter" in ln]
    env.update(zip(params, args))
    u = 0xFFFFFFFF

    def s32(x):
        return x - (1 << 32) if x & 0x80000000 else x
    for ln in lines:
        ln = ln.strip()
        if " = " not in ln:
            if ln.startswith("OpReturnValue"):
                return env[ln.split()[1]]
            continue
        res, rhs = ln.split(" = ", 1)
        op, *rest = rhs.split()
        if op in ("OpFunction", "OpFunctionParameter", "OpLabel"):
            continue
        a = [env.get(x, x) for x in rest[1:]]
        if op == "OpFunctionCall":                   # %ret %fn args...
            env[res] = eval_fn(fns[rest[1]], consts, [env.get(x, x) for x in rest[2:]], fns)
            continue
        if op == "OpExtInst":                        # %glsl FindUMsb x
            env[res] = (a[2].bit_length() - 1) & u
            continue
        f = {"OpIAdd": lambda x, y: (x + y) & u, "OpISub": lambda x, y: (x - y) & u,
             "OpShiftLeftLogical": lambda x, y: (x << y) & u, "OpShiftRightLogical": lambda x, y: x >> y,
             "OpBitwiseAnd": lambda x, y: x & y, "OpBitwiseOr": lambda x, y: x | y, "OpBitwiseXor": lambda x, y: x ^ y,
             "OpULessThan": lambda x, y: x < y, "OpUGreaterThan": lambda x, y: x > y,
             "OpUGreaterThanEqual": lambda x, y: x >= y, "OpIEqual": lambda x, y: x == y,
             "OpINotEqual": lambda x, y: x != y, "OpSLessThan": lambda x, y: s32(x) < s32(y),
             "OpSGreaterThan": lambda x, y: s32(x) > s32(y), "OpLogicalAnd": lambda x, y: x and y,
             "OpLogicalOr": lambda x, y: x or y, "OpLogicalNot": lambda x: not x,
             "OpSelect": lambda c, x, y: x if c else y, "OpCompositeConstruct": lambda *xs: tuple(xs)}[op]
        if op.startswith("OpShift"):
            assert a[1] < 32, f"shift by {a[1]} in {ln}"
        env[res] = f(*a)
    raise AssertionError("no OpReturnValue")


class SoftF64Exhaustive(unittest.TestCase):
    """Many more inputs than the translated programs can take, straight through the add / scale / convert bodies."""

    def setUp(self):
        import soft_f64

        class G:
            def __init__(self):
                self.body, self.n, self.consts = [], 0, {}

            def tmp(self):
                self.n += 1
                return f"%t{self.n}"

            def cu(self, v):
                return self.consts.setdefault(v & 0xFFFFFFFF, f"%c{v & 0xFFFFFFFF:x}")

            def emit(self, s):
                self.body.append(s)

        self.sf = soft_f64.SoftF64(G())
        self.sf.add_f64(("%a0", "%a1"), ("%b0", "%b1"), "%m")
        self.sf.scale_f64(("%a0", "%a1"), 1)
        self.sf.to_f32(("%a0", "%a1"))
        self.sf.scale_f32("%a0", 1)
        self.sf.g.body.clear()
        r = self.sf.clamp01_f32("%x")
        self.clamp_lines = (["%x = OpFunctionParameter %uint"] + self.sf.g.body + [f"OpReturnValue {r}"])

    def values(self, rng, n):
        special = [0, SIGN, 1, 1 | SIGN, DENORM_MAX, DMIN_NORMAL, DMAX, DMAX | SIGN, INF, INF | SIGN, ONE,
                   0x7FF8000000000000, 0x7FF0000000000001, 0x3CA0000000000000, 0x3FEFFFFFFFFFFFFF]
        out = []
        for _ in range(n):
            k = rng.randrange(4)
            if k == 0:
                out.append(rng.choice(special))
            elif k == 1:
                out.append(rng.getrandbits(64))
            else:
                e = rng.choice((rng.randrange(0, 60), rng.randrange(990, 1060), rng.randrange(1990, 2047)))
                out.append(rng.getrandbits(1) << 63 | e << 52 | rng.getrandbits(52))
        return out

    def test_add_all_modes(self):
        rng = random.Random(1)
        fn, consts = self.sf.fns["%f64s_add"], self.sf.g.consts
        xs, ys = self.values(rng, 1500), self.values(rng, 1500)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i % 3 == 0:                            # nearby exponents and cancellation
                y = (x ^ rng.choice((0, SIGN))) + rng.randrange(-3, 4) & M64
            mode = i % 4
            got = eval_fn(fn, consts, [x & 0xFFFFFFFF, x >> 32, y & 0xFFFFFFFF, y >> 32, mode])
            self.assertEqual(h(got[0] | got[1] << 32), h(add(x, y, mode)), f"{h(x)} + {h(y)} mode {mode}")

    def test_scale_and_convert(self):
        rng = random.Random(2)
        sfn, cfn, consts = self.sf.fns["%f64s_scale"], self.sf.fns["%f64s_to_f32"], self.sf.g.consts
        for x in self.values(rng, 1500):
            for k in (1, 2, -1):
                got = eval_fn(sfn, consts, [x & 0xFFFFFFFF, x >> 32, k & 0xFFFFFFFF])
                self.assertEqual(h(got[0] | got[1] << 32), h(scale(x, k)), f"{h(x)} * 2**{k}")
            self.assertEqual(hex(eval_fn(cfn, consts, [x & 0xFFFFFFFF, x >> 32])), hex(to_f32(x)), h(x))

    def test_f32_omod_and_clamp(self):
        """v_cvt_f32_f64's omod and clamp, on binary32 words: every class of word, f32 denormals and their ties."""
        rng = random.Random(3)
        fns, consts = self.sf.fns, self.sf.g.consts
        special = [0, 0x80000000, 1, 0x80000001, 3, 0x007FFFFF, 0x00800000, 0x00800001, 0x80800000, 0x3F800000,
                   0x3F7FFFFF, 0x3F800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7E800000, 0x7F000000, 0x7F800000, 0xFF800000,
                   0x7F800001, 0xFFC00000, 0x7FC12345, 0xBF000000, 0x3F000000]
        words = special + [rng.getrandbits(32) for _ in range(1200)] + \
            [rng.getrandbits(1) << 31 | rng.choice((0, 1, 2, 253, 254)) << 23 | rng.getrandbits(23) for _ in range(600)]
        for w in words:
            for k in (1, 2, -1):
                got = eval_fn(fns["%f64s_scale_f32"], consts, [w, k & 0xFFFFFFFF], fns)
                self.assertEqual(hex(got), hex(scale32(w, k)), f"{w:#010x} * 2**{k}")
            self.assertEqual(hex(eval_fn(self.clamp_lines, consts, [w])), hex(clamp32(w)), f"clamp {w:#010x}")


if __name__ == "__main__":
    unittest.main()
