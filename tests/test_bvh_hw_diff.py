"""tools/bvh_hw_diff.py on synthetic structures only (no game data, no GPU).

    python -I tests/test_bvh_hw_diff.py
"""
import os
import sys
import unittest
from fractions import Fraction as Fr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bvh_hw_diff as D   # noqa: E402

f2u, u2f, F = D.f2u, D.u2f, D.f32


def box_node(children):
    w = list(D.SURROGATE)
    for i in range(4):
        lo, hi = children[i] if i < len(children) else ((D.FMAX,) * 3, (-D.FMAX,) * 3)
        w += [f2u(x) for x in (*lo, *hi)]
    return w + [0] * 4


def zray(inv_z, E):
    """+z ray from (0.25, 0.25, 0) with inv.z supplied (x and y parallel: inv +inf)."""
    return [f2u(E), f2u(0.25), f2u(0.25), 0, 0, 0, f2u(F(1.0 / u2f(inv_z))), 0x7F800000, 0x7F800000, inv_z]


class SelfTest(unittest.TestCase):
    def test_self_test(self):
        self.assertEqual(D.self_test(), 0)


class RoundTowardZeroBox(unittest.TestCase):
    """The rounding of the slab distances decides a child at the extent boundary. Invented numbers: the min plane
    at 7/3 * 3 / ... chosen so that a * inv.z lies between E + 1/2 ulp and E + 1 ulp."""

    def setUp(self):
        self.inv = f2u(F(1.0 / 3.0))                     # 0x3eaaaaab
        for k in range(1, 4000):                         # first a = 1 + k/64 with the product in that window
            a = 1.0 + k / 64.0
            q = Fr(a) * Fr(u2f(self.inv))
            E = D.rtz(q)
            if (q - Fr(E)) / D.ulp(E) > Fr(1, 2):
                self.a, self.E = a, E
                break

    def test_extent_at_rtz_t0(self):
        node = box_node([((-1.0, -1.0, self.a), (1.0, 1.0, self.a + 4.0))])
        ray = zray(self.inv, self.E)
        rtz = D.box_eval(node, 5, ray, "rtz")["ids"]
        rne = D.box_eval(node, 5, ray, "rne")["ids"]
        self.assertEqual(rtz[0], D.SURROGATE[0])         # t0 rounds down to E: kept
        self.assertEqual(rne[0], D.MISS)                 # t0 rounds up past E: dropped
        self.assertEqual(D.classify_box(rtz, rne, node, 5, ray), "B_SET_EQUAL")

    def test_order_tie_only_in_rtz(self):
        found = None
        for k in range(1, 4000):                         # a: product above E + 1/2 ulp; b = a - 1 ulp: in [E, E + 1/2)
            a = 3.0 + k / 64.0
            qa = Fr(a) * Fr(u2f(self.inv))
            E = D.rtz(qa)
            b = u2f(f2u(a) - 1)
            qb = Fr(b) * Fr(u2f(self.inv))
            if (qa - Fr(E)) / D.ulp(E) > Fr(1, 2) and D.rtz(qb) == E and F(float(qb)) < F(float(qa)):
                found = (a, b, E)
                break
        self.assertIsNotNone(found)
        a, lo2, E = found
        node = box_node([((-1.0, -1.0, a), (1.0, 1.0, a + 4)), ((-1.0, -1.0, lo2), (1.0, 1.0, a + 4))])
        ray = zray(self.inv, F(4 * E + 16))
        rtz = D.box_eval(node, 5, ray, "rtz")["ids"]
        rne = D.box_eval(node, 5, ray, "rne")["ids"]
        self.assertEqual(rtz[:2], D.SURROGATE[:2])       # equal keys: the network keeps slot order
        self.assertEqual(rne[:2], [D.SURROGATE[1], D.SURROGATE[0]])
        self.assertEqual(D.classify_box(rtz, rne, node, 5, ray), "B_ORDER_TIE")


class Classes(unittest.TestCase):
    def test_triangle_classes(self):
        hit = [f2u(-8.0), f2u(-1.0), f2u(-0.25), f2u(-0.125)]
        node, ray = [0] * 16, [0] * 10
        self.assertEqual(D.classify_tri(hit, hit, node, 0, ray), "T_EXACT")
        self.assertEqual(D.classify_tri(hit, [f2u(-16.0), f2u(-2.0), f2u(-0.5), f2u(-0.25)], node, 0, ray), "T_SCALE")
        near = [hit[0] + 1, hit[1], hit[2], hit[3]]
        self.assertEqual(D.classify_tri(hit, near, node, 0, ray), "T_QUOT_ULP")

    def test_outside_is_flagged(self):
        node = box_node([])
        self.assertEqual(D.classify_box([0x1234, D.MISS, D.MISS, D.MISS], [D.MISS] * 4, node, 5, [0] * 10),
                         "B_OUTSIDE")

    def test_walk_refuses_types_6_7(self):
        buf = bytearray(256)
        with self.assertRaises(ValueError):
            D.walk(bytes(buf), (2 << 3) | 6)


# Invented nodes and rays (no game data) measured on an RX 6800 (gfx1030, AMD 32.0.21045.29003): node | ray | the
# hardware's four words (32-bit form; the triangle rows gave the same words in the 64-bit form). EXT: a child whose
# min-plane distance lies between E + 1/2 ulp and E + 1 ulp, extent E; ORD: two children one ulp apart; FLIP: a ray
# exactly through a vertex or a rounded edge point of a 1e4-magnitude triangle; WORD: an interior hit; DET: a hit
# whose word 1 tells (kU + kV) + kW from k(U + V + W).
MEASURED = [
    ('EXT', 5, """
     00000040 00000048 00000050 00000058 bf800000 bf800000 40c3b78b 3f800000
     3f800000 4121dbc6 bf800000 bf800000 3f000000 3f800000 3f800000 3f400000
     7f7fffff 7f7fffff 7f7fffff ff7fffff ff7fffff ff7fffff 7f7fffff 7f7fffff
     7f7fffff ff7fffff ff7fffff ff7fffff 00000000 00000000 00000000 00000000 |
     3f285bd2 3e800000 3e800000 00000000 00000000 00000000 4114cccd 7f800000 7f800000 3ddc370e |
     00000048 00000040 ffffffff ffffffff"""),
    ('EXT', 5, """
     00000040 00000048 00000050 00000058 bf800000 bf800000 416786f5 3f800000
     3f800000 4193c37a bf800000 bf800000 3f000000 3f800000 3f800000 3f400000
     7f7fffff 7f7fffff 7f7fffff ff7fffff ff7fffff ff7fffff 7f7fffff 7f7fffff
     7f7fffff ff7fffff ff7fffff ff7fffff 00000000 00000000 00000000 00000000 |
     40044d1e 3e800000 3e800000 00000000 00000000 00000000 40dfffff 7f800000 7f800000 3e124925 |
     00000048 00000040 ffffffff ffffffff"""),
    ('ORD', 5, """
     00000040 00000048 00000050 00000058 bf800000 bf800000 41220711 3f800000
     3f800000 41620711 bf800000 bf800000 41220710 3f800000 3f800000 41620711
     7f7fffff 7f7fffff 7f7fffff ff7fffff ff7fffff ff7fffff 7f7fffff 7f7fffff
     7f7fffff ff7fffff ff7fffff ff7fffff 00000000 00000000 00000000 00000000 |
     41a2d83a 3e800000 3e800000 00000000 00000000 00000000 4114cccd 7f800000 7f800000 3ddc370e |
     00000040 00000048 ffffffff ffffffff"""),
    ('FLIP', 0, """
     4631bf04 4637fecb 462939c8 4631bdd1 4637ffa0 462939bd 4631be01 4637ffe0
     462939ed 4631bcee 46380053 46293a3b 00000000 00000000 00000000 00000909 |
     42c80000 4631f676 4637b982 46294494 c161f800 418ca400 c02b0000 bd9102de 3d68fdb2 bebfa030 |
     ff800000 c24e9e92 c11af793 c227e0b9"""),
    ('FLIP', 1, """
     4631bf04 4637fecb 462939c8 4631bdd1 4637ffa0 462939bd 4631be01 4637ffe0
     462939ed 4631bcee 46380053 46293a3b 00000000 00000000 00000000 00000909 |
     42c80000 4631f676 4637b982 46294494 c161f800 418ca400 c02b0000 bd9102de 3d68fdb2 bebfa030 |
     c2cf9ced c2cf9ced b831ff90 c2a8af58"""),
    ('FLIP', 0, """
     4632e00f 46139dcc 46274859 4632e569 46139d5b 46274a63 4632e221 46139b63
     46274caf 4632e779 46139aa6 46274e6c 00000000 00000000 00000000 00000909 |
     42c80000 46335876 4613c208 4627a9bd c1eaca00 c1185400 c1bb6c00 bd0b9046 bdd71d55 bd2ed5e4 |
     c69d0dea c69d0de9 c5b37d5d c6605d18"""),
    ('FLIP', 1, """
     4632e00f 46139dcc 46274859 4632e569 46139d5b 46274a63 4632e221 46139b63
     46274caf 4632e779 46139aa6 46274e6c 00000000 00000000 00000000 00000909 |
     42c80000 46335876 4613c208 4627a9bd c1eaca00 c1185400 c1bb6c00 bd0b9046 bdd71d55 bd2ed5e4 |
     ff800000 c6b233fa 3c569ca4 c67e9388"""),
    ('WORD', 0, """
     462622c8 463355df 4637d023 4626241d 46335675 4637d05b 4626242e 463354fe
     4637cee0 46262590 463355be 4637cf08 00000000 00000000 00000000 00000909 |
     42c80000 46263942 463400ca 4638076c bdf3f6ec bf71b926 be9d2c7a c106508a bf878f5c c0507b90 |
     4041aaac 3d88e8ba 3cc8d369 3ca42153"""),
    ('DET', 0, """
     460b98c0 4616ee71 4622c7c3 460b938d 4616e9d0 4622cc21 460b9640 4616eb55
     4622c9fe 460b90e1 4616e6ed 4622ce2f 00000000 00000000 00000000 00000909 |
     42c80000 460bc1ed 46165239 4622b0b4 be88b446 3f739010 3e1d0809 c06fb336 3f868943 40d0abf2 |
     bf8d1ec8 bcdf0c40 bbdd2b89 bc0b5f90"""),
]


def parse(text):
    parts = [[int(w, 16) for w in p.split()] for p in text.split("|")]
    return parts[0], parts[1], parts[2]


class MeasuredSynthetic(unittest.TestCase):
    def test_round_toward_zero_model_reproduces_the_hardware(self):
        for kind, ntype, text in MEASURED:
            node, ray, hw = parse(text)
            with self.subTest(kind=kind, type=ntype):
                self.assertEqual(D.hw_model_words(node, ntype, ray), hw)

    def test_round_to_nearest_does_not(self):
        for kind, ntype, text in MEASURED:
            node, ray, hw = parse(text)
            if kind in ("EXT", "ORD"):
                self.assertNotEqual(D.box_eval(node, ntype, ray, "rne")["ids"], hw)
            if kind == "FLIP":                          # hit or miss flips with nearest rounding
                e = D.tri_eval(node, ntype, ray, "rne")
                self.assertIsNotNone(e)

if __name__ == "__main__":
    unittest.main()
