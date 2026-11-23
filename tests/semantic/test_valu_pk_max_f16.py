"""v_pk_max_f16 (RDNA2 VOP3P): D.f16[i] = max(S0.f16[i], S1.f16[i]) for both halves; a NaN operand loses to the other one.

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py. The expected
words are written down by hand (from the ISA) next to each case; the plain-Python binary16 reference in
tests/semantic/f16_ref.py rechecks them lane by lane and drives seeded random operands and modifier combinations.
Covers per-lane results, op_sel / op_sel_hi half selection, neg_lo / neg_hi, clamp and SGPR sources.
Standard library only: no GPU, no LLVM, no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
from f16_ref import clamp01, f16_value, is_nan, same_f16  # noqa: E402

ONE, TWO, THREE, HALF, QUARTER = 0x3C00, 0x4000, 0x4200, 0x3800, 0x3400
MAXH, INF, NINF, QNAN = 0x7BFF, 0x7C00, 0xFC00, 0x7E00
MIN_NORMAL, MIN_SUB, MAX_SUB = 0x0400, 0x0001, 0x03FF


def ref_max(a, b):
    """v_max_f16: a NaN operand yields the other operand (both NaN: a NaN); +0 > -0."""
    if is_nan(a) or is_nan(b):
        return QNAN if is_nan(a) and is_nan(b) else (b if is_nan(a) else a)
    if (a | b) & 0x7FFF == 0:
        return a & b
    return a if f16_value(a) > f16_value(b) else b


def P(lo, hi):
    """A dword holding two binary16 values."""
    return (hi << 16) | lo


def run(line, seeds, old=0xDEADBEEF):
    """Run one instruction on a one-lane pixel shader and return v0 as exported to mrt0 (seeds map reg -> dword)."""
    prog = [f"v_mov_b32 v0, {old:#x}"]
    for reg, w in seeds.items():
        prog.append(f"{'s' if reg[0] == 's' else 'v'}_mov_b32 {reg}, {w:#x}")
    prog += [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_pk_max_f16", prog))[0]


def half(w, i):
    return (w >> (16 * i)) & 0xFFFF


def pk_ref(fn, srcs, op_sel=None, op_sel_hi=None, neg_lo=None, neg_hi=None, clamp=False):
    """Reference for one packed op: the low result reads half op_sel[i] of source i (negated by neg_lo[i]), the high
    result half op_sel_hi[i] (negated by neg_hi[i]); defaults are op_sel 0, op_sel_hi 1, no negation."""
    n = len(srcs)
    op_sel, op_sel_hi = op_sel or [0] * n, op_sel_hi or [1] * n
    neg_lo, neg_hi = neg_lo or [0] * n, neg_hi or [0] * n
    out = []
    for sel, ng in ((op_sel, neg_lo), (op_sel_hi, neg_hi)):
        r = fn(*[half(s, h) ^ (0x8000 if g else 0) for s, h, g in zip(srcs, sel, ng)])
        out.append(clamp01(r) if clamp else r)
    return P(out[0], out[1])


def same_pk(a, b):
    return same_f16(a & 0xFFFF, b & 0xFFFF) and same_f16(a >> 16, b >> 16)


def mods_text(op_sel=None, op_sel_hi=None, neg_lo=None, neg_hi=None, clamp=False):
    t = ""
    for name, v in (("op_sel", op_sel), ("op_sel_hi", op_sel_hi), ("neg_lo", neg_lo), ("neg_hi", neg_hi)):
        if v is not None:
            t += f" {name}:[{','.join(map(str, v))}]"
    return t + (" clamp" if clamp else "")


def regs(n):
    return ", ".join(f"v{i}" for i in range(1, n + 1))


# mnemonic -> (source count, reference for one lane)
OPS = {"v_pk_max_f16": (2, ref_max)}

# rows of (source dwords..., D), written by hand; lane 0 is the low half
CASES = [
    (P(ONE, THREE), P(TWO, HALF), P(TWO, THREE)),                  # (max(1, 2), max(3, 0.5))
    (P(0xBC00, 0x0000), P(ONE, 0x8000), P(ONE, 0x0000)),           # (max(-1, 1), max(+0, -0) = +0)
    (P(0x8000, 0x8000), P(0x0000, 0x8000), P(0x0000, 0x8000)),     # (max(-0, +0) = +0, max(-0, -0) = -0)
    (P(QNAN, ONE), P(ONE, QNAN), P(ONE, ONE)),                     # a NaN operand yields the other one
    (P(QNAN, INF), P(QNAN, MAXH), P(QNAN, INF)),                   # (both NaN: NaN, max(inf, 65504))
    (P(NINF, MIN_SUB), P(0xFBFF, 0x0000), P(0xFBFF, MIN_SUB)),     # (max(-inf, -65504), max(2^-24, 0))
    (P(0x8001, 0x3C01), P(MIN_SUB, ONE), P(MIN_SUB, 0x3C01)),      # (max(-2^-24, 2^-24), max(1 + 2^-10, 1))
    (P(MAXH, NINF), P(INF, MAXH), P(INF, MAXH)),                   # (max(65504, inf), max(-inf, 65504))
    (P(0x0000, 0x0000), P(0x8000, 0x8000), P(0x0000, 0x0000)),     # (max(+0, -0), max(+0, -0))
    (P(0x8000, 0x0000), P(0x8000, 0x0000), P(0x8000, 0x0000)),     # identical signs
]

# (sources, modifier text, D), by hand. S0 = (1.0, 3.0), S1 = (0.5, -0.25) as (low, high) halves.
S0, S1 = P(ONE, THREE), P(HALF, 0xB400)
MODS = [
    ((S0, S1), "", P(ONE, THREE)),                                       # (max(1, 0.5), max(3, -0.25))
    ((S0, S1), " op_sel:[1,0]", P(THREE, THREE)),                        # low: max(3, 0.5)
    ((S0, S1), " op_sel:[0,1] op_sel_hi:[0,0]", P(ONE, ONE)),            # low: max(1, -0.25); high: max(1, 0.5)
    ((S0, S1), " op_sel:[1,1] op_sel_hi:[0,0]", P(THREE, ONE)),          # low: max(3, -0.25); high: max(1, 0.5)
    ((S0, S1), " neg_lo:[1,0]", P(HALF, THREE)),                         # low: max(-1, 0.5)
    ((S0, S1), " neg_hi:[0,1]", P(ONE, THREE)),                          # high: max(3, 0.25)
    ((S0, S1), " neg_lo:[1,1] neg_hi:[1,1]", P(0xB800, 0x3400)),         # low: max(-1, -0.5) = -0.5; high: max(-3, 0.25) = 0.25
    ((P(0x3E00, 0xB800), P(TWO, 0xBC00)), " clamp", P(ONE, 0x0000)),     # (max(1.5, 2.0)=2.0 -> 1.0, max(-0.5, -1.0)=-0.5 -> 0.0)
    ((P(HALF, QUARTER), P(0x3A00, ONE)), " clamp", P(0x3A00, ONE)),      # inside [0, 1]: unchanged
]


class PkMaxF16(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for *src, want in CASES:
            with self.subTest(src=[hex(s) for s in src]):
                self.assertTrue(same_pk(pk_ref(ref_max, src), want), hex(pk_ref(ref_max, src)))

    def test_per_lane_results(self):
        for *src, want in CASES:
            with self.subTest(src=[hex(s) for s in src]):
                got = run(f"v_pk_max_f16 v0, {regs(len(src))}", {f"v{i}": s for i, s in enumerate(src, 1)})
                self.assertTrue(same_pk(got, want), hex(got))

    def test_sgpr_sources(self):
        for *src, want in CASES[:6]:
            n = len(src)
            forms = [
                ("all sgpr", ", ".join(f"s{i}" for i in range(1, n + 1)), {f"s{i}": s for i, s in enumerate(src, 1)}),
                ("sgpr src0", ", ".join(["s1"] + [f"v{i}" for i in range(2, n + 1)]),
                 {"s1": src[0], **{f"v{i}": s for i, s in enumerate(src[1:], 2)}}),
                ("sgpr src1", ", ".join(["v1", "s2"]),
                 {"v1": src[0], "s2": src[1]}),
            ]
            for form, text, seeds in forms:
                with self.subTest(src=[hex(s) for s in src], form=form):
                    self.assertTrue(same_pk(run(f"v_pk_max_f16 v0, {text}", seeds), want))

    def test_hand_written_modifier_cases(self):
        for srcs, mods, want in MODS:
            with self.subTest(mods=mods):
                got = run(f"v_pk_max_f16 v0, {regs(len(srcs))}{mods}", {f"v{i}": s for i, s in enumerate(srcs, 1)})
                self.assertTrue(same_pk(got, want), hex(got))

    def test_every_op_sel_and_op_sel_hi_combination(self):
        a, b = P(0x3E00, 0xC400), P(0x4900, 0x3A00)  # distinct halves: 1.5/-4, 10/.75
        srcs = [a, b]
        for sel in range(4):
            for selhi in range(4):
                os_ = [(sel >> i) & 1 for i in range(2)]
                oh = [(selhi >> i) & 1 for i in range(2)]
                with self.subTest(op_sel=os_, op_sel_hi=oh):
                    got = run(f"v_pk_max_f16 v0, {regs(2)}{mods_text(os_, oh)}", {f"v{i}": s for i, s in enumerate(srcs, 1)})
                    self.assertTrue(same_pk(got, pk_ref(ref_max, srcs, os_, oh)), hex(got))

    def test_negation_combinations(self):
        a, b = P(ONE, TWO), P(HALF, THREE)
        srcs = [a, b]
        for nlo_mask in range(4):
            for nhi_mask in range(4):
                nl = [(nlo_mask >> i) & 1 for i in range(2)]
                nh = [(nhi_mask >> i) & 1 for i in range(2)]
                with self.subTest(neg_lo=nl, neg_hi=nh):
                    got = run(f"v_pk_max_f16 v0, {regs(2)}{mods_text(neg_lo=nl, neg_hi=nh)}",
                              {f"v{i}": s for i, s in enumerate(srcs, 1)})
                    self.assertTrue(same_pk(got, pk_ref(ref_max, srcs, neg_lo=nl, neg_hi=nh)), hex(got))

    def test_clamp_combinations(self):
        # Clamping behavior on defined (non-NaN) values: below 0 -> 0, above 1 -> 1, within [0, 1] unchanged
        test_values = [
            (P(0xBC00, 0x4200), P(0xB800, 0x4000), P(0x0000, ONE)),   # max(-1.0, -0.5)=-0.5 -> 0.0, max(3.0, 2.0)=3.0 -> 1.0
            (P(0x0000, 0x3800), P(0x8000, QUARTER), P(0x0000, HALF)),  # zeros -> 0.0, max(0.5, 0.25)=0.5 -> 0.5
            (P(NINF, INF), P(0xFBFF, MAXH), P(0x0000, ONE)),           # infinities clamped to [0, 1]
        ]
        for s0, s1, want in test_values:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run(f"v_pk_max_f16 v0, v1, v2 clamp", {"v1": s0, "v2": s1})
                self.assertTrue(same_pk(got, want), f"got {hex(got)}, want {hex(want)}")

    def test_random_operands_and_modifiers(self):
        rng = random.Random(0x4D4158)
        pool = [
            0x0000, 0x8000, ONE, 0xBC00, TWO, HALF, THREE, QUARTER, MAXH, 0xFBFF,
            INF, NINF, QNAN, MIN_NORMAL, MIN_SUB, 0x8001, MAX_SUB, 0x3C01, 0x6800,
            0x6801, 0x4C00, 0x1000, 0x9000,
        ]
        for _ in range(120):
            srcs = [P(*[rng.choice(pool) if rng.random() < 0.6 else rng.getrandbits(16) for _ in range(2)])
                    for _ in range(2)]
            os_ = [rng.getrandbits(1) for _ in range(2)]
            oh = [rng.getrandbits(1) for _ in range(2)]
            nl = [rng.getrandbits(1) for _ in range(2)]
            nh = [rng.getrandbits(1) for _ in range(2)]
            with self.subTest(src=[hex(s) for s in srcs], mods=(os_, oh, nl, nh)):
                got = run(f"v_pk_max_f16 v0, {regs(2)}{mods_text(os_, oh, nl, nh)}",
                          {f"v{i}": s for i, s in enumerate(srcs, 1)})
                self.assertTrue(same_pk(got, pk_ref(ref_max, srcs, os_, oh, nl, nh)), hex(got))


if __name__ == "__main__":
    unittest.main()
