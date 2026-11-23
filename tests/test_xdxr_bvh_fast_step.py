"""The one-unpack traversal step of tools/xdxr_bvh.py answers exactly what the field-by-field step answers.

``intersect_node`` reads a node's bytes in one ``struct`` call (``Memory._span``) and tests the four child boxes
unrolled; it must return the very values the field-by-field reads (``_intersect_node_reads``, through a memory that has
only read/u32/f32) return: NaN, infinities, clamped half infinities, zero direction components, truncated and
overlapping regions included. No game data is read.

    python -I tests/test_xdxr_bvh_fast_step.py
"""
import math
import os
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import xdxr_bvh as X  # noqa: E402

INF = math.inf


def bits(x):
    """Exact identity of a float or a tuple of them: NaN equals NaN, 0.0 differs from -0.0."""
    if isinstance(x, (tuple, list)):
        return tuple(bits(c) for c in x)
    return struct.pack("<d", x) if isinstance(x, float) else x


class ReadsOnly:
    """A memory with only read/u32/f32: what a caller's own memory object offers, so no one-unpack path."""

    def __init__(self, mem):
        self.mem = mem

    def read(self, va, n):
        return self.mem.read(va, n)

    def u32(self, va, n=1):
        return self.mem.u32(va, n)

    def f32(self, va, n=1):
        return self.mem.f32(va, n)


def box_node(kind, children, boxes, tail=b""):
    """A box node record: ``children`` pointers and ``boxes`` of ((lo), (hi)), BOX16 ones as half-float bit words."""
    out = struct.pack("<4I", *children)
    for lo, hi in boxes:
        out += struct.pack("<6f", *lo, *hi) if kind == X.BOX32 else struct.pack("<3I", *X.encode_box16(lo, hi))
    return out + tail


class BoxStep(unittest.TestCase):
    def kids(self, mem, kind, org, d, extent, sort=True):
        inv = X._inverse_dir(d)
        fast = X.intersect_node(mem, 0, X.ptr(0, kind), org, d, inv, extent, sort)
        slow = X.intersect_node(ReadsOnly(mem), 0, X.ptr(0, kind), org, d, inv, extent, sort)
        self.assertEqual(fast, slow, (kind, org, d, extent, sort))
        return fast

    def test_random_nodes_match_the_field_by_field_step(self):
        rng = random.Random(21)
        hits = 0
        for kind in (X.BOX32, X.BOX16):
            for _ in range(1500):
                boxes = []
                for _ in range(4):
                    c = [rng.uniform(-3, 3) for _ in range(3)]
                    h = [rng.uniform(0.0, 1.5) for _ in range(3)]
                    lo, hi = [c[k] - h[k] for k in range(3)], [c[k] + h[k] for k in range(3)]
                    if rng.random() < 0.2:
                        lo, hi = [INF] * 3, [-INF] * 3                         # an empty slot
                    elif rng.random() < 0.1:
                        lo[rng.randrange(3)], hi[rng.randrange(3)] = -INF, INF   # beyond the half range
                    boxes.append((lo, hi))
                children = [X.ptr(64 * (i + 1), X.TRI0) if rng.random() < 0.9 else X.INVALID for i in range(4)]
                mem = X.Memory()
                mem.place(0, box_node(kind, children, boxes))
                o = [rng.uniform(-5, 5) for _ in range(3)]
                d = [rng.uniform(-1, 1) for _ in range(3)]
                for c in range(3):
                    if rng.random() < 0.2:
                        d[c] = 0.0                                              # the slab test's d == 0 branch
                for extent in (INF, 4.0, 0.5):
                    for sort in (True, False):
                        hits += sum(k != X.INVALID for k in self.kids(mem, kind, o, d, extent, sort))
        self.assertGreater(hits, 1000)

    def test_infinite_halves_clamp_to_flt_max(self):
        # a BOX16 slot holding +-inf halves (0x7C00 / 0xFC00) is read as +-FLT_MAX: a ray far from the origin is inside
        words = (0xFC00 | 0xFC00 << 16, 0xFC00 | 0x7C00 << 16, 0x7C00 | 0x7C00 << 16)
        data = struct.pack("<4I", *[X.ptr(64, X.TRI0)] * 4) + struct.pack("<3I", *words) * 4
        mem = X.Memory()
        mem.place(0, data)
        kids = self.kids(mem, X.BOX16, [1e30, -1e30, 5.0], [0.0, 0.0, 1.0], INF)
        self.assertEqual(kids, [X.ptr(64, X.TRI0)] * 4)
        box = X.decode_box16(words)
        self.assertEqual(box[0][0], -X.FLT_MAX)
        self.assertEqual(box[1][2], X.FLT_MAX)

    def test_nan_and_tiny_directions_match(self):
        # (lo - org) * inf is NaN when a box face is at the ray origin and the direction is denormal
        boxes = [([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), ([0.0, -1.0, -1.0], [2.0, 1.0, 1.0]),
                 ([-2.0, 0.0, -1.0], [0.0, 3.0, 1.0]), ([-1.0, -1.0, 1.0], [1.0, 1.0, 2.0])]
        for kind in (X.BOX32, X.BOX16):
            mem = X.Memory()
            mem.place(0, box_node(kind, [X.ptr(64 * (i + 1), X.TRI0) for i in range(4)], boxes))
            for d in ([5e-324, 0.0, 1.0], [5e-324, 5e-324, 5e-324], [-5e-324, 1.0, 0.0], [0.0, 0.0, 0.0]):
                self.kids(mem, kind, [0.0, 0.0, 0.0], d, INF)
                self.kids(mem, kind, [0.0, 0.0, 0.0], d, 3.0)

    def test_other_node_types_are_none(self):
        mem = X.Memory()
        mem.place(0, bytes(256))
        for kind in (X.INSTANCE, X.AABB):
            self.assertIsNone(X.intersect_node(mem, 0, X.ptr(0, kind), [0, 0, 0], [0, 0, 1], [1, 1, 1], INF))
            self.assertIsNone(X.intersect_node(ReadsOnly(mem), 0, X.ptr(0, kind), [0, 0, 0], [0, 0, 1], [1, 1, 1], INF))

    def test_leaf_steps_match_for_every_triangle_node_type(self):
        rng = random.Random(33)
        for _ in range(300):
            verts = [rng.uniform(-2, 2) for _ in range(15)]
            remap = rng.getrandbits(32)
            data = struct.pack("<15fI", *verts, remap)
            mem = X.Memory()
            mem.place(0x40, data)
            o = [rng.uniform(-2, 2) for _ in range(3)]
            d = [rng.uniform(-1, 1) for _ in range(3)]
            for kind in (X.TRI0, X.TRI1, X.TRI2, X.TRI3):
                p = X.ptr(0, kind)
                a = X.intersect_node(mem, 0x40, p, o, d, X._inverse_dir(d), INF)
                b = X.intersect_node(ReadsOnly(mem), 0x40, p, o, d, X._inverse_dir(d), INF)
                self.assertEqual(bits(a), bits(b))


class Regions(unittest.TestCase):
    def test_span_answers_from_the_first_region_that_holds_the_bytes(self):
        mem = X.Memory()
        mem.place(0x1000, b"A" * 256)
        mem.place(0x1000, b"B" * 512)                   # an in-place update's second dump at the same address
        data, off = mem._span(0x1040, 64)
        self.assertEqual((data[off:off + 64], off), (b"A" * 64, 0x40))
        self.assertEqual(mem.read(0x1040, 64), b"A" * 64)

    def test_span_declines_when_an_earlier_region_only_overlaps(self):
        mem = X.Memory()
        mem.place(0x1000, b"A" * 0x50)                  # ends inside the node at 0x1040
        mem.place(0x1000, b"B" * 0x200)
        self.assertIsNone(mem._span(0x1040, 64))
        self.assertIsNone(mem._span(0x3000, 64))        # nothing holds it
        # and a node there reads exactly as before: the fields come from whichever region holds each one
        node = struct.pack("<4I", *[X.INVALID] * 4) + struct.pack("<6f", -1, -1, -1, 1, 1, 1) * 4
        mem = X.Memory()
        mem.place(0x1000, bytes(0x40) + node[:0x20])    # a truncated first copy of the node at 0x1040
        mem.place(0x1000, bytes(0x40) + node)
        o, d = [0.0, 0.0, -3.0], [0.0, 0.0, 1.0]
        got = X.intersect_node(mem, 0x1000, X.ptr(0x40, X.BOX32), o, d, X._inverse_dir(d), INF)
        want = X.intersect_node(ReadsOnly(mem), 0x1000, X.ptr(0x40, X.BOX32), o, d, X._inverse_dir(d), INF)
        self.assertEqual(got, want)

    def test_reads_outside_every_region_still_raise(self):
        mem = X.Memory()
        mem.place(0, bytes(100))                        # shorter than a node
        o, d = [0.0, 0.0, -3.0], [0.0, 0.0, 1.0]
        for kind in (X.BOX32, X.BOX16, X.TRI0):
            with self.assertRaises(IndexError):
                X.intersect_node(mem, 0, X.ptr(64, kind), o, d, X._inverse_dir(d), INF)


class Traces(unittest.TestCase):
    def scene(self, box16, pair):
        rng = random.Random(3)
        tris = X.grid_mesh(8) if pair else X.random_scene(rng, 150)
        data, info = X.build_blas(tris, box16=box16, pair=pair)
        return tris, data, info

    def test_blas_trace_matches_the_field_by_field_memory(self):
        rng = random.Random(4)
        for box16 in (False, True):
            for pair in (False, True):
                tris, data, info = self.scene(box16, pair)
                mem = X.Memory()
                mem.place(0x10000, data)
                slow = ReadsOnly(mem)
                hits = 0
                for _ in range(150):
                    o = [rng.uniform(-6, 6) for _ in range(3)]
                    t = [rng.uniform(-3, 3) for _ in range(3)]
                    d = [t[c] - o[c] for c in range(3)]
                    s1, s2 = {"steps": 0}, {"steps": 0}
                    a = X.trace(mem, 0x10000, o, d, stats=s1)
                    b = X.trace(slow, 0x10000, o, d, stats=s2)
                    self.assertEqual(a, b)
                    self.assertEqual(s1, s2)
                    hits += a is not None
                self.assertGreater(hits, 10)

    def test_tlas_trace_matches_the_field_by_field_memory(self):
        rng = random.Random(8)
        mem = X.Memory()
        insts = []
        for i in range(4):
            tris = X.random_scene(rng, 40)
            data, info = X.build_blas(tris, box16=bool(i % 2))
            va = 0x100000 * (i + 1)
            mem.place(va, data)
            m = [[1, 0, 0, 2.0 * i], [0, 1, 0, 0.0], [0, 0, 1, -1.0 * i]]
            insts.append(X.Instance(m, va, info["root"], info["bounds"], instance_id=i))
        for box16 in (False, True):
            tdata, _ = X.build_tlas(insts, box16=box16)
            va = 0x4000000 + box16 * 0x100000
            mem.place(va, tdata)
            slow = ReadsOnly(mem)
            for _ in range(150):
                o = [rng.uniform(-12, 12) for _ in range(3)]
                t = [rng.uniform(-6, 6) for _ in range(3)]
                d = [t[c] - o[c] for c in range(3)]
                self.assertEqual(X.trace(mem, va, o, d), X.trace(slow, va, o, d))

    def test_tmax_limits_hits_the_same_way(self):
        tris, data, info = self.scene(False, False)
        mem = X.Memory()
        mem.place(0, data)
        slow = ReadsOnly(mem)
        rng = random.Random(6)
        checked = 0
        for _ in range(300):
            o = [rng.uniform(-6, 6) for _ in range(3)]
            t = [rng.uniform(-3, 3) for _ in range(3)]
            d = [t[c] - o[c] for c in range(3)]
            full = X.trace(mem, 0, o, d)
            if full is None:
                continue
            checked += 1
            self.assertEqual(X.trace(mem, 0, o, d, tmax=full["t"]), full)
            self.assertEqual(X.trace(mem, 0, o, d, tmax=full["t"] * 0.5), X.trace(slow, 0, o, d, tmax=full["t"] * 0.5))
            self.assertIsNone(X.trace(mem, 0, o, d, tmax=full["t"] * 0.5))
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
