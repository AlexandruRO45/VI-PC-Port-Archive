"""Parity of the batched tools/bvh_rehydrate.py with the one-value-at-a-time algorithm it replaced, which is kept here
as an inline reference. Every check compares the tool with that reference on the same input in the same process, so
nothing depends on the platform, the Python build or a stored hash. No game data is read.

NaN is only fed to steps that do no arithmetic on it: the sign of the NaN an add or multiply of two NaNs returns
follows the operand order the interpreter happened to compile, so the tests keep arithmetic on finite values.

    python -I tests/test_bvh_rehydrate_parity.py
"""
import math
import os
import random
import struct
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bvh_rehydrate as R  # noqa: E402
from test_bvh_rehydrate import BLAS_DEHYDRATED, header, leaf_chunk, level_table, ptr  # noqa: E402

M32 = 0xFFFFFFFF
NAN = struct.unpack("<f", struct.pack("<I", 0x7FC00000))[0]
INF = float("inf")


# --- the reference: the scalar algorithm, one float32 rounding per operation -----------------------------------------

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits32(b):
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def half_to_float(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def float_to_half(x):
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return R.HALF_NEG_INF if x < 0 else R.HALF_POS_INF


def min_num(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a <= b else b


def max_num(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


def half_down(x):
    if x != x:
        return float_to_half(x)
    b = bits32(x)
    if x < 0.0:
        y = min_num(from_bits32((b + 0x1FFF) & ~0x1FFF), -R.HALF_MIN_NORMAL)
    else:
        y = from_bits32(b & ~0x1FFF)
        y = 0.0 if y < R.HALF_MIN_NORMAL else y
    return float_to_half(y)


def half_up(x):
    if x != x:
        return float_to_half(x)
    b = bits32(x)
    if x > 0.0:
        y = max_num(from_bits32((b + 0x1FFF) & ~0x1FFF), R.HALF_MIN_NORMAL)
    else:
        y = from_bits32(b & ~0x1FFF)
        y = 0.0 if y > -R.HALF_MIN_NORMAL else y
    return float_to_half(y)


def frame_apply(frame, v):
    if not frame.on:
        return v
    out = []
    for row in range(3):
        acc = f32(frame.r[row][0] * v[0])
        acc = f32(frame.r[row][1] * v[1] + acc)
        acc = f32(frame.r[row][2] * v[2] + acc)
        out.append(f32(f32(acc - frame.t[row]) * frame.s[row]))
    return tuple(out)


def geometry_vertex(g, i):
    v = struct.unpack_from("<3f", g.vertices, i * g.stride)
    m = g.transform
    if m is None:
        return v
    out = []
    for row in range(3):
        acc = f32(m[row][0] * v[0])
        acc = f32(m[row][1] * v[1] + acc)
        acc = f32(m[row][2] * v[2] + acc)
        out.append(f32(acc + m[row][3]))
    return tuple(out)


def read_u24(buf, at):
    return buf[at] | buf[at + 1] << 8 | buf[at + 2] << 16


def leaf_records(src, hdr):
    remaining, chunk = hdr.num_elements, 0
    while remaining > 0:
        c = min(R.LEAVES_PER_CHUNK, remaining)
        base = hdr.offset_to_elements + chunk * R.CHUNK_STRIDE
        for i in range(c):
            prim1 = read_u24(src, base + 7 * c + 3 * i)
            yield (base + 64 * i, read_u24(src, base + 3 * i), src[base + 3 * c + i], read_u24(src, base + 4 * c + 3 * i),
                   None if prim1 == 0xFFFFFF else prim1, src[base + 10 * c + i])
        remaining -= c
        chunk += 1


def node_bounds(dst, p):
    kind, at = p & 7, (p >> 3) << 6
    if kind == R.TRI0:
        v = struct.unpack_from("<12f", dst, at)
        pts = [v[0:3], v[3:6], v[6:9]] + ([v[9:12]] if struct.unpack_from("<I", dst, at + 48)[0] & 0x80 else [])
    elif kind == R.TRI1:
        v = struct.unpack_from("<9f", dst, at + 12)
        pts = [v[0:3], v[3:6], v[6:9]]
    elif kind == R.BOX32:
        f = struct.unpack_from("<24f", dst, at + 16)
        lo = [f[a] for a in range(3)]
        hi = [f[3 + a] for a in range(3)]
        for s in range(1, 4):
            for a in range(3):
                lo[a] = min_num(lo[a], f[6 * s + a])
                hi[a] = max_num(hi[a], f[6 * s + 3 + a])
        return tuple(lo) + tuple(hi)
    elif kind == R.BOX16:
        h = struct.unpack_from("<24H", dst, at + 16)
        out = []
        for c in range(6):
            best = h[c]
            for s in range(1, 4):
                cand = h[6 * s + c]
                x, y = half_to_float(best), half_to_float(cand)
                if x != x or (y == y and ((y < x) if c < 3 else (y > x))):
                    best = cand
            out.append(half_to_float(best))
        return tuple(out)
    else:
        return (R.FLT_MAX,) * 3 + (-R.FLT_MAX,) * 3
    lo, hi = list(pts[0]), list(pts[0])
    for q in pts[1:]:
        for a in range(3):
            lo[a] = min_num(lo[a], q[a])
            hi[a] = max_num(hi[a], q[a])
    return tuple(lo) + tuple(hi)


def write_leaves(src, dst, hdr, geometries, frame):
    for out, gi, flags, prim0, prim1, order in leaf_records(src, hdr):
        g = geometries[gi]
        tri = [frame_apply(frame, geometry_vertex(g, i)) for i in g.triangle(prim0)]
        if prim1 is None:
            fourth = (R.NAN32,) * 3
        else:
            j = g.triangle(prim1)
            fourth = frame_apply(frame, geometry_vertex(g, j[0] if order & 0x20 else j[1] if order & 0x80 else j[2]))
        a, b = order & 3, (order >> 2) & 3
        slots = [None, None, None, fourth]
        for which, value in zip((a, b, (a ^ 3) - b), tri):
            slots[R.SLOT_OF_ORDER[which]] = value
        struct.pack_into("<12f4I", dst, out, *slots[0], *slots[1], *slots[2], *slots[3],
                         (gi << 8 | flags) & M32, prim0, R.INVALID if prim1 is None else prim1,
                         (order & 0xF) | ((order << 4) & 0xF00))


def write_box_groups(src, dst, hdr, start, count, wide):
    node_size, group_bytes = (128, 1 << 13) if wide else (64, 1 << 12)
    for g in range((count + R.GROUP - 1) // R.GROUP):
        n = min(R.GROUP, count - g * R.GROUP)
        base = hdr.offset_to_nodes + start + g * group_bytes
        words = struct.unpack_from("<%dI" % n, src, base)
        total_children = sum(w >> 24 for w in words)
        child_at = base + 4 * n
        value_at = base + 4 * (n + total_children)
        for t, word in enumerate(words):
            kids, mask = word >> 24, word & 0xFFFFFF
            children = list(struct.unpack_from("<%dI" % kids, src, child_at)) + [R.INVALID] * (4 - kids)
            child_at += 4 * kids
            comps = []
            for slot in range(4):
                if slot < kids:
                    b = node_bounds(dst, children[slot])
                    comps += list(b) if wide else [half_down(b[0]), half_down(b[1]), half_down(b[2]),
                                                   half_up(b[3]), half_up(b[4]), half_up(b[5])]
                else:
                    comps += [R.FLT_MAX] * 3 + [-R.FLT_MAX] * 3 if wide else [R.HALF_POS_INF] * 3 + [R.HALF_NEG_INF] * 3
            for bit in range(24):
                if mask >> bit & 1:
                    if wide:
                        comps[bit] = struct.unpack_from("<f", src, value_at)[0]
                        value_at += 4
                    else:
                        comps[bit] = struct.unpack_from("<H", src, value_at)[0]
                        value_at += 2
            at = base + t * node_size
            if wide:
                struct.pack_into("<4I24f4I", dst, at, *children, *comps, 0, 0, 0, 0)
            else:
                struct.pack_into("<4I24H", dst, at, *children, *comps)
            me = (at >> 6) << 3 | (R.BOX32 if wide else R.BOX16)
            for child in children:
                if (child & 6) == 4 or ((child & 7) == R.TRI0 and hdr.flags & R.FLAG_UPDATABLE):
                    entry = hdr.stored_parent_entry(((child >> 3) << 6) & M32, len(dst))
                    if entry is not None:
                        struct.pack_into("<I", dst, entry, me)


def rehydrate(dehydrated, geometries, header_offset=0):
    hdr = R.Header(dehydrated, header_offset)
    src = bytes(dehydrated[header_offset:header_offset + hdr.total_bytes])
    dst = bytearray(hdr.total_bytes)
    kept = hdr.kept_header_bytes()
    dst[:kept] = src[:kept]
    struct.pack_into("<I", dst, 0, hdr.flags & ~R.FLAG_DEHYDRATED & M32)
    root_entry = hdr.stored_parent_entry(((hdr.root >> 3) << 6) & M32, len(dst))
    if root_entry is not None:
        struct.pack_into("<I", dst, root_entry, R.INVALID)
    write_leaves(src, dst, hdr, list(geometries), R.LeafFrame(hdr))
    if hdr.root & 7 in (R.BOX16, R.BOX32):
        for b32_start, b32_count, b16_start, b16_count in R.levels(src, hdr):
            write_box_groups(src, dst, hdr, b32_start, b32_count, True)
            write_box_groups(src, dst, hdr, b16_start, b16_count, False)
    return dst


# --- inputs ----------------------------------------------------------------------------------------------------------

ORDERS = (0x09, 0x06, 0x02, 0x08, 0x01, 0x04)        # the six distinct (first, second) pairs of vertex slots
SCALE_ONLY = 0x80000000 | (127 << 16) | (129 << 8) | 126
TRANSLATE_SCALE = 0xC0000000 | (125 << 16) | (127 << 8) | 130
ROTATED = 0xC0000000 | (37 << 24) | (127 << 16) | (127 << 8) | 127
FRAMES = (R.NO_TRANSFORM, SCALE_ONLY, TRANSLATE_SCALE, ROTATED)
MATRIX = ((0.75, 0.125, -0.5, 3.25), (-0.0625, 0.9, 0.3, -1.5), (0.2, -0.35, 1.125, 0.001))


def grid(rng, quads, index16=False, stride=12, wild=False):
    """A quads x quads grid of two triangles per quad: (vertex bytes, index bytes, triangle count)."""
    w = quads + 1
    pad = b"\0" * (stride - 12)
    vb = b"".join(struct.pack("<3f", x * 0.37 + rng.random() * 0.1 - 2, y * 0.41 + rng.random() * 0.1,
                              rng.uniform(-300, 300) if wild else rng.random() * 3) + pad
                  for y in range(w) for x in range(w))
    idx = []
    for y in range(quads):
        for x in range(quads):
            a, b, c, d = y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1
            idx += [a, b, c, c, b, d]
    return vb, struct.pack(("<%dH" if index16 else "<%dI") % len(idx), *idx), len(idx) // 3


def tree(rng, n_leaves, n_tris, compression=R.NO_TRANSFORM, flags=BLAS_DEHYDRATED, parents=0x40, fanout=4,
         n_geoms=1, bounds=(10, 20, -300, 500, 600, -100), overrides=0.08, special=False):
    """A dehydrated BLAS with n_leaves leaves in 2,048-leaf chunks under levels of BOX32 and BOX16 nodes (alternating,
    `fanout` children each) with random override masks; special adds an infinity and a NaN override per level."""
    chunks, leaf_ptrs, p = [], [], 0
    for first in range(0, n_leaves, R.LEAVES_PER_CHUNK):
        leaves = []
        for _ in range(min(R.LEAVES_PER_CHUNK, n_leaves - first)):
            p1 = (p + 1) % n_tris if rng.random() < 0.7 else None
            leaves.append((rng.randrange(n_geoms), rng.randrange(256), p % n_tris, p1,
                           rng.choice(ORDERS) | rng.choice((0, 0x20, 0x80))))
            p += 2 if p1 is not None else 1
        chunks.append(leaves)
    shapes, n, wide = [], n_leaves, True
    while True:
        n = (n + fanout - 1) // fanout
        shapes.append((wide, n))
        if n == 1:
            break
        wide = not wide
    starts, pos = [], 0
    for w, c in shapes:                       # a level starts where the previous one's nodes end (unpadded)
        starts.append(pos)
        pos += c * (128 if w else 64)
    nodes_size = max(s + ((c + R.GROUP - 1) // R.GROUP) * ((1 << 13) if w else (1 << 12))
                     for (w, c), s in zip(shapes, starts))
    leaf_bytes = (len(chunks) - 1) * R.CHUNK_STRIDE + 64 * len(chunks[-1])
    table = level_table([(s + c * 128, s + c * 128) if w else (s, s + c * 64) for (w, c), s in zip(shapes, starts)])
    elements = (0x40 + 4 * ((leaf_bytes + nodes_size) // 64 + 16) + 63) // 64 * 64
    nodes = (elements + leaf_bytes + 63) // 64 * 64
    total = (nodes + nodes_size + len(table) + 63) // 64 * 64
    src = bytearray(total)
    for ci, leaves in enumerate(chunks):
        data = leaf_chunk(leaves)
        at = elements + ci * R.CHUNK_STRIDE
        src[at:at + len(data)] = data
    child_ptrs = [ptr(elements + ci * R.CHUNK_STRIDE + 64 * i, R.TRI0) for ci, lv in enumerate(chunks) for i in range(len(lv))]
    for (w, c), s in zip(shapes, starts):
        size, stride = (128, 1 << 13) if w else (64, 1 << 12)
        made = []
        for gi in range((c + R.GROUP - 1) // R.GROUP):
            base = nodes + s + gi * stride
            group = []
            for t in range(min(R.GROUP, c - gi * R.GROUP)):
                first = (gi * R.GROUP + t) * fanout
                over = {}
                for bit in range(24):
                    if rng.random() < overrides:
                        over[bit] = rng.uniform(-50, 600) if w else rng.randrange(0x2C00, 0x5800)
                if special and over:
                    bit = rng.choice(sorted(over))
                    over[bit] = rng.choice((NAN, INF, -INF)) if w else rng.choice((0x7E00, 0x7C00, 0xFC00))
                group.append((child_ptrs[first:first + fanout], over))
                made.append(ptr(base + t * size, R.BOX32 if w else R.BOX16))
            data = _group(group, w)
            src[base:base + len(data)] = data
        child_ptrs = made
    src[total - len(table):] = table
    src[0:64] = header(total, n_leaves, bounds, nodes, child_ptrs[0], elements, parents, compression, flags)
    return bytes(src)


def _group(nodes, wide):
    words, kids, values = [], [], []
    for children, overrides in nodes:
        mask = 0
        for bit in sorted(overrides):
            mask |= 1 << bit
            values.append(overrides[bit])
        words.append(len(children) << 24 | mask)
        kids += children
    body = struct.pack("<%dI" % len(words), *words) + struct.pack("<%dI" % len(kids), *kids)
    return body + (struct.pack("<%df" % len(values), *values) if wide else struct.pack("<%dH" % len(values), *values))


def same_float(a, b):
    """Equal bit for bit, with every NaN equal to every other."""
    if a != a or b != b:
        return a != a and b != b
    return struct.pack("<d", a) == struct.pack("<d", b)


def same_floats(a, b):
    return len(a) == len(b) and all(same_float(x, y) for x, y in zip(a, b))


# --- tests -----------------------------------------------------------------------------------------------------------

class TestRoundingHelpers(unittest.TestCase):
    def test_half_down_and_up_match_the_float_arithmetic_version(self):
        rng = random.Random(11)
        patterns = [s << 31 | e << 23 | m for s in (0, 1) for e in range(256)
                    for m in (0, 1, 0x1000, 0x1FFF, 0x2000, 0x2001, 0x3FFF, 0x400000, 0x7FDFFF, 0x7FE000, 0x7FFFFF)]
        patterns += [rng.getrandbits(32) for _ in range(30000)]
        for bits in patterns:
            x = struct.unpack("<f", struct.pack("<I", bits))[0]
            self.assertEqual(R.half_down(x), half_down(x), hex(bits))
            self.assertEqual(R.half_up(x), half_up(x), hex(bits))

    def test_half_helpers_agree_on_doubles_between_float32_values(self):
        for x in (0.1, -0.1, 1e-50, -1e-50, 65519.9, -65519.9, 65520.0, 1e10, -1e10, 5.96e-8, 6.1e-5, -6.1e-5):
            self.assertEqual(R.half_down(x), half_down(x), x)
            self.assertEqual(R.half_up(x), half_up(x), x)
        for fn_new, fn_ref in ((R.half_down, half_down), (R.half_up, half_up)):
            with self.assertRaises(OverflowError):
                fn_new(1e39)
            with self.assertRaises(OverflowError):
                fn_ref(1e39)


class TestNodeBounds(unittest.TestCase):
    def _nodes(self, rng, specials):
        """A buffer of 64-byte nodes: TRI0 with and without the fourth vertex, TRI1, BOX32 and BOX16, 40 of each."""
        buf = bytearray()
        pool = [0.0, -0.0, 1.0, -1.0, 7.5, -7.5, INF, -INF, 6.1e-5, R.FLT_MAX]
        if specials:
            pool += [NAN, NAN]

        def value():
            return rng.choice(pool) if rng.random() < 0.3 else rng.uniform(-100, 100)
        for _ in range(40):
            for flag in (0, 0x80):
                buf += struct.pack("<12f4I", *[value() for _ in range(12)], flag, 0, 0, 0)
            buf += struct.pack("<I", 0) * 3 + struct.pack("<9f", *[value() for _ in range(9)]) + b"\0" * 25
            buf += struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<24f", *[value() for _ in range(24)]) + b"\0" * 16
            buf += struct.pack("<4I", 0, 0, 0, 0) + b"".join(
                struct.pack("<H", rng.choice((0x7C00, 0xFC00, 0x7E00, 0, 0x8000)) if rng.random() < 0.2 and specials
                            else R.float_to_half(rng.uniform(-60, 60)) if rng.random() < 0.9 else 0x7C00)
                for _ in range(24)) + b"\0" * 0
            buf += b"\0" * (-len(buf) % 64)
        return buf

    def test_bounds_match_the_pairwise_fold(self):
        for specials in (False, True):
            rng = random.Random(5 + specials)
            for _ in range(8):
                buf = self._nodes(rng, specials)
                for at in range(0, len(buf) - 63, 64):
                    for kind in (R.TRI0, R.TRI1, R.BOX16, R.BOX32, 2, 3, 6, 7):
                        p = (at >> 6) << 3 | kind
                        try:
                            want = node_bounds(buf, p)
                        except struct.error:
                            with self.assertRaises(struct.error):
                                R.node_bounds(buf, p)
                            continue
                        got = R.node_bounds(buf, p)
                        self.assertTrue(same_floats(got, want), (specials, at, kind, got, want))

    def test_a_leading_nan_loses_and_zero_ties_keep_the_first(self):
        # (NaN, 3, 2) must give 2 and 3, and a -0.0 before a +0.0 stays -0.0 (the builtin min keeps the first of equals)
        leaf = struct.pack("<12f4I", NAN, -0.0, 5.0, 3.0, 0.0, 6.0, 2.0, 0.0, 4.0, 0, 0, 0, 0, 0, 0, 0)
        got = R.node_bounds(leaf, R.TRI0)
        self.assertTrue(same_floats(got, node_bounds(leaf, R.TRI0)))
        self.assertEqual(got[:2], (2.0, -0.0))
        self.assertEqual(math.copysign(1.0, got[1]), -1.0)


class TestBatchedTransforms(unittest.TestCase):
    def _points(self, rng, count):
        return [tuple(f32(rng.uniform(-400, 400)) for _ in range(3)) for _ in range(count)]

    def test_apply_many_matches_apply_for_every_frame_and_point_count(self):
        rng = random.Random(21)
        for word in FRAMES + (0x40000000 | (127 << 16) | (127 << 8) | 127, 0x00000000):
            hdr = R.Header(header(0x300, 1, (-3, 2, -1, 4, 9, 7), 0x200, ptr(0x200, R.BOX32), 0x100, 0x40, word))
            frame = R.LeafFrame(hdr)
            for count in (1, 2, 3, 4, 5, 7):
                for _ in range(20):
                    pts = self._points(rng, count)
                    got = frame.apply_many(pts)
                    self.assertEqual([struct.pack("<3f", *v) for v in got],
                                     [struct.pack("<3f", *frame_apply(frame, v)) for v in pts], (hex(word), count))

    def test_vertex_batch_matches_vertex_with_and_without_a_transform(self):
        rng = random.Random(22)
        for transform in (None, MATRIX, ((1, 0, 0, 0.5), (0, 1, 0, 0), (0, 0, 1, 0))):
            for index16, stride in ((False, 12), (True, 16), (False, 32)):
                vb, ib, n_tris = grid(rng, 6, index16, stride, wild=True)
                g = R.Geometry(vb, stride, ib, index16, transform)
                for count in (1, 3, 4, 6):
                    for _ in range(10):
                        picks = tuple(rng.randrange(49) for _ in range(count))
                        self.assertEqual([struct.pack("<3f", *v) for v in g.vertex_batch(picks)],
                                         [struct.pack("<3f", *geometry_vertex(g, i)) for i in picks], (transform, picks))

    def test_a_transform_replaced_after_use_is_not_stale(self):
        rng = random.Random(23)
        vb, ib, _ = grid(rng, 3)
        g = R.Geometry(vb, 12, ib, False, MATRIX)
        g.vertex_batch((0, 1, 2))
        g.transform = ((2, 0, 0, 1), (0, 2, 0, 1), (0, 0, 2, 1))
        self.assertEqual([struct.pack("<3f", *v) for v in g.vertex_batch((0, 1, 2))],
                         [struct.pack("<3f", *geometry_vertex(g, i)) for i in (0, 1, 2)])

    def test_the_batch_tables_stay_bounded(self):
        rng = random.Random(24)
        hdr = R.Header(header(0x300, 1, (-3, 2, -1, 4, 9, 7), 0x200, ptr(0x200, R.BOX32), 0x100, 0x40, TRANSLATE_SCALE))
        frame = R.LeafFrame(hdr)
        vb, ib, _ = grid(rng, 4)
        g = R.Geometry(vb, 12, ib, False, MATRIX)
        for count in range(1, 40):
            frame.apply_many(self._points(rng, count))
            g.vertex_batch(tuple(i % 25 for i in range(count)))
        self.assertLessEqual(len(frame._consts), len(R.BATCH_POINTS))
        self.assertLessEqual(len(g._consts), len(R.BATCH_POINTS))
        self.assertLessEqual(len(R._ROUNDERS), 4096)


class TestLeafRecords(unittest.TestCase):
    def test_records_match_the_per_leaf_reader_across_block_and_chunk_edges(self):
        rng = random.Random(31)
        for n in (1, 2, R.LEAF_BLOCK - 1, R.LEAF_BLOCK, R.LEAF_BLOCK + 1, 2 * R.LEAF_BLOCK + 3, R.LEAVES_PER_CHUNK,
                  R.LEAVES_PER_CHUNK + 1, R.LEAVES_PER_CHUNK + R.LEAF_BLOCK + 5):
            src = tree(rng, n, 50, fanout=4)
            hdr = R.Header(src)
            self.assertEqual(list(R.leaf_records(src, hdr)), list(leaf_records(src, hdr)), n)


class TestRehydrateParity(unittest.TestCase):
    def _check(self, seed, n_leaves, compression, transform=None, **kw):
        rng = random.Random(seed)
        n_geoms = kw.get("n_geoms", 1)
        index16 = rng.random() < 0.5
        stride = rng.choice((12, 16, 24))
        vb, ib, n_tris = grid(rng, 12, index16, stride, wild=transform is not None or compression != R.NO_TRANSFORM)
        src = tree(rng, n_leaves, n_tris, compression, **kw)
        geoms = [R.Geometry(vb, stride, ib, index16, transform) for _ in range(n_geoms)]
        got = R.rehydrate(src, geoms)
        want = rehydrate(src, geoms)
        self.assertEqual(len(got), len(want))
        self.assertTrue(got == want, "rehydrated bytes differ at %s" % next((i for i in range(len(got)) if got[i] != want[i]), None))
        return got

    def test_every_leaf_frame(self):
        for k, word in enumerate(FRAMES):
            self._check(40 + k, 300, word, flags=BLAS_DEHYDRATED | R.FLAG_UPDATABLE)

    def test_geometry_transform_with_and_without_a_frame(self):
        self._check(50, 250, R.NO_TRANSFORM, MATRIX)
        self._check(51, 250, TRANSLATE_SCALE, MATRIX, n_geoms=2)

    def test_more_than_one_chunk_and_group(self):
        self._check(60, R.LEAVES_PER_CHUNK + 190, SCALE_ONLY, flags=BLAS_DEHYDRATED | R.FLAG_UPDATABLE)
        self._check(61, 4 * R.GROUP * 3 + 5, R.NO_TRANSFORM, fanout=3)

    def test_fanouts_and_few_leaves(self):
        for k, (n, fanout) in enumerate(((1, 4), (2, 2), (3, 4), (5, 3), (17, 2), (65, 3))):
            self._check(70 + k, n, TRANSLATE_SCALE if k % 2 else R.NO_TRANSFORM, fanout=fanout)

    def test_parent_table_forms(self):
        for parents in (0x40, 0):          # a console structure keeps offsetToParents at 0
            for flags in (BLAS_DEHYDRATED, BLAS_DEHYDRATED | R.FLAG_UPDATABLE):
                self._check(80, 400, R.NO_TRANSFORM, parents=parents, flags=flags)

    def test_infinite_and_nan_overrides_pass_through_the_box_levels(self):
        for seed in range(90, 96):
            self._check(seed, 500, R.NO_TRANSFORM, special=True, overrides=0.15)

    def test_a_nan_vertex_without_arithmetic(self):
        rng = random.Random(100)
        vb, ib, n_tris = grid(rng, 6)
        vb = bytearray(vb)
        for at in range(0, len(vb), 12 * 5):
            vb[at:at + 4] = struct.pack("<f", NAN)
        src = tree(rng, 120, n_tris, R.NO_TRANSFORM)
        geoms = [R.Geometry(bytes(vb), 12, ib)]
        self.assertTrue(R.rehydrate(src, geoms) == rehydrate(src, geoms))

    def test_header_offset_and_bytearray_input(self):
        rng = random.Random(110)
        vb, ib, n_tris = grid(rng, 8)
        src = tree(rng, 200, n_tris, TRANSLATE_SCALE)
        geoms = [R.Geometry(vb, 12, ib)]
        padded = bytearray(b"\x5a" * 256) + src
        self.assertTrue(R.rehydrate(padded, geoms, 256) == rehydrate(padded, geoms, 256))
        self.assertTrue(R.rehydrate(padded, geoms, 256) == R.rehydrate(src, geoms))


class TestPeakMemory(unittest.TestCase):
    """The tool must not need more memory than the algorithm it replaced: a dense tree and a sparse one."""

    def _peak(self, fn, src, geoms):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            fn(src, geoms)
            return tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()

    def test_dense_and_sparse_peaks_stay_with_the_reference(self):
        rng = random.Random(120)
        vb, ib, n_tris = grid(rng, 20, wild=True)
        for n_leaves in (R.LEAVES_PER_CHUNK + 300, 3, 1):
            src = tree(rng, n_leaves, n_tris, TRANSLATE_SCALE, flags=BLAS_DEHYDRATED | R.FLAG_UPDATABLE)
            geoms = [R.Geometry(vb, 12, ib, False, MATRIX)]
            self.assertTrue(R.rehydrate(src, geoms) == rehydrate(src, geoms))
            new_peak = self._peak(R.rehydrate, src, geoms)
            ref_peak = self._peak(rehydrate, src, geoms)
            self.assertLessEqual(new_peak, ref_peak + 8192 + ref_peak // 50, (n_leaves, new_peak, ref_peak))


if __name__ == "__main__":
    unittest.main()
