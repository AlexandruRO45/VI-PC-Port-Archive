"""tex_export.detile_offsets and tex_mode_probe.offsets_for give the same offsets as the per-bit parity grid they replaced.

    python -I tests/test_tex_export_detile.py
    python -m pytest tests/test_tex_export_detile.py

The equations are hand-built (no addrlib header, no game data): an S layout (interleaved x/y bits) and layouts where the
address bits above the element size are XORed with coordinate bits beyond the tile, for every element size.
"""
import os
import random
import sys
import unittest

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

try:
    import numpy as np
    import tex_export as T
    import tex_mode_probe as P
except ModuleNotFoundError as e:
    if e.name != "numpy":
        raise
    np = T = P = None


def make_equation(bpe, bits, xor):
    """One (x mask, y mask) per address bit; the bits below the element size address bytes inside an element."""
    low = bpe.bit_length() - 1
    eq, xi, yi = [], 0, 0
    for bit in range(bits):
        if bit < low:
            eq.append((0, 0))
        elif (bit - low) % 2 == 0:
            eq.append((1 << xi, 0))
            xi += 1
        else:
            eq.append((0, 1 << yi))
            yi += 1
    if xor:
        for bit in range(low + 2, bits):
            mx, my = eq[bit]
            eq[bit] = (mx ^ (1 << (xi + bit % 3)) ^ (1 << (xi + 1)), my ^ (1 << (yi + bit % 2)) ^ (1 << (yi + 2)))
    return eq


def reference_export(eq, bpe, we, he):
    """tex_export.detile_offsets as it was: per address bit, a parity over the whole he x we grid."""
    elems = 4096 // bpe
    lb = elems.bit_length() - 1
    bw = 1 << ((lb + 1) // 2)
    bh = elems // bw
    y, x = np.mgrid[0:he, 0:we].astype(np.int64)
    o = np.zeros_like(x)
    for bit, (mx, my) in enumerate(eq):
        if mx or my:
            o |= (T._parity(x & mx) ^ T._parity(y & my)) << bit
    nbx = (we + bw - 1) // bw
    return ((o + ((y // bh) * nbx + x // bw) * 4096) // bpe).reshape(-1), bw, bh


def reference_probe(eq, bb, bpe, wb, hb):
    """tex_mode_probe.offsets_for as it was."""
    bw, bh = P._tile_shape(1 << bb, bpe)
    y, x = np.mgrid[0:hb, 0:wb].astype(np.int64)
    o = np.zeros_like(x)
    for bit, (mx, my) in enumerate(eq):
        if mx or my:
            o |= (P._parity(x & mx) ^ P._parity(y & my)) << bit
    nbx = (wb + bw - 1) // bw
    return (o + ((y // bh) * nbx + x // bw) * (1 << bb)) // bpe


# Narrow, tall, single-row/column and sizes that are not multiples of any tile extent.
SIZES = [(1, 1), (1, 70), (70, 1), (3, 5), (37, 5), (5, 37), (64, 64), (100, 37), (129, 65), (130, 3), (257, 513)]
# (name, bytes per element, pixels per element along an axis): BC1/BC7 are 4x4-pixel elements, RGBA8 one pixel.
REALISTIC = [("BC1", 8, 4), ("BC7", 16, 4), ("RGBA8", 4, 1)]


@unittest.skipIf(np is None, "numpy is not installed")
class DetileOffsetsMatchReference(unittest.TestCase):
    def check(self, eq, bpe, we, he):
        T._EQ[bpe] = eq
        try:
            want, wbw, wbh = reference_export(eq, bpe, we, he)
            got, gbw, gbh = T.detile_offsets(bpe, we, he)
        finally:
            T._EQ.pop(bpe, None)
        label = (bpe, we, he)
        self.assertEqual((gbw, gbh), (wbw, wbh), label)
        self.assertEqual(got.shape, (we * he,), label)
        self.assertEqual(got.dtype, np.int64, label)
        np.testing.assert_array_equal(got, want, str(label))

    def test_every_element_size_equation_and_shape(self):
        for bpe in (1, 2, 4, 8, 16):
            for xor in (False, True):
                eq = make_equation(bpe, 12, xor)
                for we, he in SIZES:
                    with self.subTest(bpe=bpe, xor=xor, size=(we, he)):
                        self.check(eq, bpe, we, he)

    def test_realistic_texture_sizes(self):
        for _, bpe, per_element in REALISTIC:
            for px in (512, 1024):
                n = px // per_element
                for xor in (False, True):
                    with self.subTest(bpe=bpe, px=px, xor=xor):
                        self.check(make_equation(bpe, 12, xor), bpe, n, n)

    def test_random_equations(self):
        rng = random.Random(20261123)
        for _ in range(60):
            bpe = rng.choice((1, 2, 4, 8, 16))
            eq = [(rng.getrandbits(14) if rng.random() < 0.8 else 0, rng.getrandbits(14) if rng.random() < 0.8 else 0)
                  for _ in range(12)]
            we, he = rng.randint(1, 200), rng.randint(1, 200)
            with self.subTest(bpe=bpe, size=(we, he)):
                self.check(eq, bpe, we, he)

    def test_element_size_that_is_not_a_power_of_two(self):
        # eq4k only has equations for 1/2/4/8/16, but the division by bpe must stay exact for any size.
        eq = make_equation(4, 12, True)
        for bpe in (3, 6, 12):
            for we, he in ((1, 1), (7, 9), (40, 33)):
                with self.subTest(bpe=bpe, size=(we, he)):
                    self.check(eq, bpe, we, he)

    def test_whole_element_gather_matches_byte_rows(self):
        # export_one reads the tiled bytes as one void item per element; the bytes must equal the (-1, bpe) row gather.
        rng = np.random.default_rng(3)
        for bpe in (1, 2, 4, 8, 16):
            T._EQ[bpe] = make_equation(bpe, 12, True)
            try:
                offs, _, _ = T.detile_offsets(bpe, 45, 29)
            finally:
                T._EQ.pop(bpe, None)
            need = (int(offs.max()) + 1) * bpe
            data = rng.integers(0, 256, need + 5, dtype=np.uint8).tobytes()
            want = np.frombuffer(data[:need], np.uint8).reshape(-1, bpe)[offs].tobytes()
            got = np.frombuffer(memoryview(data)[:need], f"V{bpe}")[offs].tobytes()
            self.assertEqual(got, want, bpe)


@unittest.skipIf(np is None, "numpy is not installed")
class ProbeOffsetsMatchReference(unittest.TestCase):
    def check(self, eq, bb, bpe, wb, hb):
        want = reference_probe(eq, bb, bpe, wb, hb)
        got = P.offsets_for(eq, bb, bpe, wb, hb)
        label = (bb, bpe, wb, hb)
        self.assertEqual(got.shape, (hb, wb), label)
        self.assertEqual(got.dtype, np.int64, label)
        np.testing.assert_array_equal(got, want, str(label))

    def test_tile_sizes_element_sizes_and_shapes(self):
        for bb in (12, 16):
            for bpe in (8, 16):
                for xor in (False, True):
                    eq = make_equation(bpe, bb, xor)
                    for wb, hb in SIZES:
                        with self.subTest(bb=bb, bpe=bpe, xor=xor, size=(wb, hb)):
                            self.check(eq, bb, bpe, wb, hb)

    def test_realistic_texture_sizes(self):
        for _, bpe, per_element in REALISTIC[:2]:
            for px in (512, 1024):
                n = px // per_element
                for bb in (12, 16):
                    with self.subTest(bpe=bpe, px=px, bb=bb):
                        self.check(make_equation(bpe, bb, True), bb, bpe, n, n)

    def test_other_element_sizes(self):
        eq = make_equation(4, 12, True)
        for bpe in (1, 2, 4, 3, 6):
            for wb, hb in ((1, 1), (7, 9), (40, 33)):
                with self.subTest(bpe=bpe, size=(wb, hb)):
                    self.check(eq, 12, bpe, wb, hb)

    def test_random_equations(self):
        rng = random.Random(20261124)
        for _ in range(40):
            bpe = rng.choice((8, 16))
            bb = rng.choice((12, 16))
            eq = [(rng.getrandbits(14) if rng.random() < 0.8 else 0, rng.getrandbits(14) if rng.random() < 0.8 else 0)
                  for _ in range(bb)]
            wb, hb = rng.randint(1, 150), rng.randint(1, 150)
            with self.subTest(bpe=bpe, bb=bb, size=(wb, hb)):
                self.check(eq, bb, bpe, wb, hb)


if __name__ == "__main__":
    unittest.main()
