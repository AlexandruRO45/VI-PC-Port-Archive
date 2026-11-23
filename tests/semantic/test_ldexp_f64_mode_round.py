"""v_ldexp_f64 after a MODE.FP_ROUND[3:2] write (f16/f64 rounding), run in tests/semantic/spirv_mini_interp.py.

FP_ROUND[3:2] selects nearest even, +inf, -inf or toward zero. For v_ldexp_f64 (D = S0 * 2^S1), it affects
overflow and the rounding of inexact results below the normal range. The lowering uses 32-bit integer operations.
The expected values include RX 6600 XT measurements and an independent exact rational reference.
"""
import os
import random
import sys
import unittest
from fractions import Fraction as Q

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from support import mrt0_words, rdna2spirv, translate_fixture, translate_lines  # noqa: E402

RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"
RUP = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1"
RDN = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2"
RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
ROUND_MODE = {None: 0, RNE: 0, RUP: 1, RDN: 2, RTZ: 3}
DBL_MAX = 0x7FEFFFFFFFFFFFFF


def run(mode_line, ops):
    """ops: one or two (x, n) pairs, x a binary64 bit pattern; returns v_ldexp_f64's 64-bit result for each."""
    lines = [mode_line] if mode_line else []
    for i, (x, n) in enumerate(ops):
        lines += [f"v_mov_b32 v{4 * i}, 0x{x & 0xFFFFFFFF:08x}", f"v_mov_b32 v{4 * i + 1}, 0x{x >> 32:08x}",
                  f"v_mov_b32 v{4 * i + 2}, 0x{n & 0xFFFFFFFF:x}",
                  f"v_ldexp_f64 v[{8 + 2 * i}:{9 + 2 * i}], v[{4 * i}:{4 * i + 1}], v{4 * i + 2}"]
    lines += ["exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm"]
    w = mrt0_words(translate_lines("ps_ldexp_f64_mode", lines))
    return [w[2 * i] | w[2 * i + 1] << 32 for i in range(len(ops))]


def double_value(b):
    e, f = b >> 52 & 0x7FF, b & ((1 << 52) - 1)
    q = Q(f, 1 << 1074) if e == 0 else Q((1 << 52) | f, 1 << 52) * Q(2) ** (e - 1023)
    return -q if b >> 63 else q


def exponent(a):
    """The integer E with 2^E <= a < 2^(E+1), for a > 0."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while a >= Q(2) ** (e + 1):
        e += 1
    while a < Q(2) ** e:
        e -= 1
    return e


def ref_ldexp(x, n, mode):
    """x a finite non-zero binary64, n an integer; x * 2^n rounded to binary64 in the selected MODE."""
    q = double_value(x) * Q(2) ** n
    sign, a = (1 << 63 if q < 0 else 0), abs(q)
    toward_larger_magnitude = (mode == 1 and q > 0) or (mode == 2 and q < 0)
    ulp = Q(2) ** (max(exponent(a), -1022) - 52)
    k, r = divmod(a, ulp)
    nearest_up = r > ulp / 2 or (r == ulp / 2 and k % 2)
    if (mode == 0 and nearest_up) or (toward_larger_magnitude and r):
        k += 1
    v = k * ulp
    if v >= Q(2) ** 1024:
        return sign | (0x7FF0000000000000 if mode == 0 or toward_larger_magnitude else DBL_MAX)
    if v < Q(2) ** -1022:
        return sign | int(v * Q(2) ** 1074)
    e = exponent(v)
    return sign | (e + 1023) << 52 | int(v / Q(2) ** (e - 52)) - (1 << 52)


class LdexpF64ModeRound(unittest.TestCase):
    # (mode line, x, n, want)
    CASES = (
        (None, DBL_MAX, 1, 0x7FF0000000000000),                 # DBL_MAX * 2 is inf under nearest even (no MODE write)
        (RTZ, DBL_MAX, 1, DBL_MAX),                             # and saturates to DBL_MAX toward zero
        (RTZ, DBL_MAX | 1 << 63, 1, DBL_MAX | 1 << 63),
        (RTZ, 0x3FF0000000000000, 5000, DBL_MAX),               # n clamped to 4096: still a finite overflow
        (RTZ, 0x7FF0000000000000, 1, 0x7FF0000000000000),      # inf stays inf
        (RTZ, 0x7FF8000000000001, 1, 0x7FF8000000000001),      # a NaN passes unchanged
        (RTZ, 1 << 63, 5, 1 << 63),                             # a signed zero stays one
        (None, 0x3FF0000000000003, -1023, 0x0008000000000002),  # (1 + 3 * 2^-52) * 2^-1023: 1.5 ulp, rounds up
        (RTZ, 0x3FF0000000000003, -1023, 0x0008000000000001),   # toward zero truncates
        (RTZ, 0xBFF0000000000003, -1023, 0x8008000000000001),
        (RTZ, 0x3FF0000000000001, -1023, 0x0008000000000000),   # 0.5 ulp: a tie both modes take to the even word
        (None, 0x3FEFFFFFFFFFFFFF, -1022, 0x0010000000000000),  # (1 - 2^-53) * 2^-1022: nearest even carries into
        (RTZ, 0x3FEFFFFFFFFFFFFF, -1022, 0x000FFFFFFFFFFFFF),   # the smallest normal, toward zero stays subnormal
        (None, 0x3FF8000000000000, -1075, 0x0000000000000001),  # 0.75 * 2^-1074
        (RTZ, 0x3FF8000000000000, -1075, 0x0000000000000000),
        (RTZ, 0x0000000000000003, 3, 0x0000000000000018),      # exact: a subnormal operand, a subnormal result
        (RTZ, 0x3FF123456789ABCD, 100, 0x463123456789ABCD),     # exact: a normal result
        (RNE, DBL_MAX, 1, 0x7FF0000000000000),                  # back to nearest even
    )

    def test_cases(self):
        for mode, x, n, want in self.CASES:
            with self.subTest(mode=mode, x=hex(x), n=n):
                self.assertEqual(run(mode, [(x, n)])[0], want)

    def test_reference_agrees_with_the_cases(self):
        for mode, x, n, want in self.CASES:
            if x >> 52 & 0x7FF == 0x7FF or not x & ~(1 << 63):
                continue
            with self.subTest(mode=mode, x=hex(x), n=n):
                self.assertEqual(ref_ldexp(x, max(-4096, min(4096, n)), ROUND_MODE[mode]), want)

    def test_directed_modes_at_range_boundaries(self):
        # Native RX 6600 XT words for overflow, a half-ULP tie, carry into min normal, and tiny/exact results.
        cases = (
            (DBL_MAX, 1, 0x7FF0000000000000, DBL_MAX),
            (DBL_MAX | 1 << 63, 1, 0xFFEFFFFFFFFFFFFF, 0xFFF0000000000000),
            (0x3FF0000000000003, -1023, 0x0008000000000002, 0x0008000000000001),
            (0xBFF0000000000003, -1023, 0x8008000000000001, 0x8008000000000002),
            (0x3FEFFFFFFFFFFFFF, -1022, 0x0010000000000000, 0x000FFFFFFFFFFFFF),
            (0xBFEFFFFFFFFFFFFF, -1022, 0x800FFFFFFFFFFFFF, 0x8010000000000000),
            (0x3FF8000000000000, -1075, 1, 0),
            (0xBFF8000000000000, -1075, 1 << 63, (1 << 63) | 1),
            (0x3FF0000000000000, -1074, 1, 1),
            (0xBFF0000000000000, -1074, (1 << 63) | 1, (1 << 63) | 1),
        )
        for x, n, up, down in cases:
            for mode, expected in ((RUP, up), (RDN, down)):
                with self.subTest(mode=mode, x=hex(x), n=n):
                    self.assertEqual(ref_ldexp(x, n, ROUND_MODE[mode]), expected)
                    self.assertEqual(run(mode, [(x, n)])[0], expected)

    def test_directed_rounding_checks_sticky_bits_and_tiny_nonzero_values(self):
        for mode in (RUP, RDN):
            for sign in (0, 1 << 63):
                # sh=31/32/33 exercise every split-word shift path. A discarded low bit is below a half ULP.
                for n in (-1053, -1054, -1055, -1076, -4096):
                    for lsb in (0, 1):
                        x = sign | 0x3FF0000000000000 | lsb
                        with self.subTest(mode=mode, x=hex(x), n=n):
                            self.assertEqual(run(mode, [(x, n)])[0], ref_ldexp(x, n, ROUND_MODE[mode]))

    def test_s_round_mode_switch_uses_modified_sign(self):
        # Toward +inf saturates on negative overflow, then toward -inf goes to negative infinity.
        w = mrt0_words(translate_fixture("ps_ldexp_f64_directed"))
        self.assertEqual(w, [0xFFFFFFFF, 0xFFEFFFFF, 0x00000000, 0xFFF00000])

    def test_directed_write_with_an_unmodelled_f16_result_is_rejected(self):
        lines = [RUP, "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3ff00000", "v_mov_b32 v2, 0",
                 "v_ldexp_f64 v[8:9], v[0:1], v2", "v_cvt_f16_f32 v10, v0",
                 "exp mrt0 v8, v9, v10, v10 done vm", "s_endpgm"]
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "directed rounding"):
            translate_lines("ps_directed_mixed", lines)

    def test_mode_switches_back_and_source_neg(self):
        lines = ["v_mov_b32 v0, -1", "v_mov_b32 v1, 0x7fefffff", "v_mov_b32 v2, 1", RTZ,
                 "v_ldexp_f64 v[8:9], -v[0:1], v2", RNE, "v_ldexp_f64 v[10:11], -v[0:1], v2",
                 "exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm"]
        w = mrt0_words(translate_lines("ps_ldexp_f64_switch", lines))
        self.assertEqual(w, [0xFFFFFFFF, 0xFFEFFFFF, 0x00000000, 0xFFF00000])

    def test_clamp_after_toward_zero_overflow(self):
        # DX10 clamp after the rounding: +DBL_MAX clamps to 1.0, as +inf does under nearest even.
        lines = ["v_mov_b32 v0, -1", "v_mov_b32 v1, 0x7fefffff", "v_mov_b32 v2, 1", RTZ,
                 "v_ldexp_f64 v[8:9], v[0:1], v2 clamp", "exp mrt0 v8, v9, v8, v9 done vm", "s_endpgm"]
        self.assertEqual(mrt0_words(translate_lines("ps_ldexp_f64_clamp", lines))[:2], [0, 0x3FF00000])

    def test_seeded_sweep_against_exact_reference(self):
        rng = random.Random(4064)
        for mode in (RNE, RUP, RDN, RTZ):
            for _ in range(8):
                ops = []
                for _ in range(2):
                    k = rng.random()
                    if k < 0.3:                                   # near the overflow edge
                        x, n = rng.getrandbits(1) << 63 | rng.randint(2040, 2046) << 52 | rng.getrandbits(52), \
                            rng.randint(0, 8)
                    elif k < 0.8:                                 # into the subnormal range
                        x, n = rng.getrandbits(1) << 63 | rng.randint(1, 2046) << 52 | rng.getrandbits(52), \
                            rng.randint(-1100, -1)
                    else:                                         # a subnormal operand
                        x, n = rng.getrandbits(1) << 63 | rng.randint(1, (1 << 52) - 1), rng.randint(-8, 60)
                    ops.append((x, n))
                got = run(mode, ops)
                for (x, n), g in zip(ops, got):
                    with self.subTest(mode=mode, x=hex(x), n=n):
                        self.assertEqual(g, ref_ldexp(x, max(-4096, min(4096, n)), ROUND_MODE[mode]))

    def test_probe_cases_match_the_translator(self):
        """tools/mode_round_probe.py's v_ldexp_f64 words in both modes are the translator's, so a run of the probe
        on an RDNA2 GPU checks this rule."""
        import mode_round_probe as probe
        cases = [c for c in probe.CASES if "v_ldexp_f64" in c[2]]
        self.assertEqual(len(cases), 2)
        for name, setup, op, is_64, want_rne, want_rtz in cases:
            self.assertTrue(is_64)
            for mode, want in ((RNE, want_rne), (RTZ, want_rtz)):
                with self.subTest(case=name, mode=mode):
                    lines = list(setup) + [mode, op, "exp mrt0 v3, v4, v3, v4 done vm", "s_endpgm"]
                    self.assertEqual(tuple(mrt0_words(translate_lines("ps_mode_probe", lines))[:2]), want)


if __name__ == "__main__":
    unittest.main()
