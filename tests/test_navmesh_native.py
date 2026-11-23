"""Parity of the optional Rust kernels (tools/navmesh_native) with the Python reference in navmesh_decode.py.

Everything here is synthetic and seeded; no game file is read. The native tests skip when the library is not
built (``cargo build --release --manifest-path tools/navmesh_native/Cargo.toml``); the fallback tests always run.
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import _navmesh_native  # noqa: E402
import navmesh_decode as nd  # noqa: E402
import test_navmesh_decode as fixtures  # noqa: E402

BASE = nd.BASE
HAVE_NATIVE = _navmesh_native.native() is not None
needs_native = unittest.skipUnless(HAVE_NATIVE, "tools/navmesh_native is not built")


class _Off:
    """Stands in for the adapter: every kernel declines, so navmesh_decode runs its Python reference."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def reference(fn, *args):
    saved, nd.native = nd.native, _Off()
    try:
        return fn(*args)
    finally:
        nd.native = saved


def canon(value):
    """Plain, order-preserving form of a stats tree, so insertion order is compared too."""
    if isinstance(value, dict):
        return [type(value).__name__] + [(k, canon(v)) for k, v in value.items()]
    if isinstance(value, set):
        return ["set"] + sorted(value)
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def ptr(rng, offset):
    """A pointer to `offset`, sometimes broken in one of the ways offset_of refuses."""
    r = rng.random()
    if r < 0.01:
        return BASE - 1 - rng.randrange(64)
    if r < 0.02:
        return (1 << 32) | (BASE + offset)
    if r < 0.03:
        return BASE + 0x10000000 + rng.randrange(64)
    return BASE + offset


class Heap:
    def __init__(self, rng, noise=False):
        self.rng = rng
        self.b = bytearray(rng.randbytes(0x40) if noise else bytes(0x40))
        self.noise = noise

    def alloc(self, size, align=4):
        at = (len(self.b) + align - 1) & ~(align - 1)
        fill = self.rng.randbytes(at + size - len(self.b)) if self.noise else bytes(at + size - len(self.b))
        self.b += fill
        return at

    def u64(self, at, value):
        struct.pack_into("<Q", self.b, at, value & 0xFFFFFFFFFFFFFFFF)


def gen_quadtree(rng):
    heap = Heap(rng, noise=rng.random() < 0.05)
    n_polys = rng.choice((0, 1, 3, rng.randrange(1, 300)))
    count = rng.choice((1, 2, rng.randrange(1, 40), rng.randrange(1, 400)))
    nodes = [heap.alloc(0x60) for _ in range(count)]
    for o in nodes:
        heap.b[o + 0x54:o + 0x60] = bytes(12)
        if rng.random() < 0.01:
            heap.b[o + rng.randrange(0x54, 0x60)] = rng.randrange(1, 256)
    leaves = []
    for _ in range(rng.randrange(0, count + 1)):
        lo = heap.alloc(0x20)
        npl = rng.randrange(0, n_polys + 1) if n_polys else 0
        if rng.random() < 0.02:
            npl = n_polys + rng.randrange(1, 4)
        po = heap.alloc(2 * npl, align=2)
        for k in range(npl):
            v = rng.randrange(n_polys) if n_polys else 0
            if rng.random() < 0.005:
                v = n_polys + rng.randrange(0, 3)
            struct.pack_into("<H", heap.b, po + 2 * k, v)
        heap.u64(lo, 0 if rng.random() > 0.01 else 1)
        heap.u64(lo + 8, ptr(rng, po) if npl or rng.random() < 0.5 else 0)
        heap.u64(lo + 16, 0 if rng.random() > 0.01 else 7)
        heap.u64(lo + 24, npl)
        leaves.append(lo)
    # A tree in allocation order (child index > parent), with rare back edges, shared children and wild pointers.
    for i, o in enumerate(nodes):
        if leaves and rng.random() < 0.5:
            heap.u64(o + 0x2C, ptr(rng, rng.choice(leaves)))
        for c in range(4):
            r = rng.random()
            if r < 0.5 and i + 1 < count:
                target = rng.randrange(i + 1, count)
            elif r < 0.52:
                target = rng.choice(nodes)
            elif r < 0.53:
                target = len(heap.b) - rng.randrange(0, 0x60)
            else:
                continue
            heap.u64(o + 0x34 + 8 * c, ptr(rng, target))
    if rng.random() < 0.03:
        del heap.b[rng.randrange(0x40, len(heap.b)):]
    return bytes(heap.b), nodes[0], n_polys


def gen_polys(rng):
    heap = Heap(rng, noise=rng.random() < 0.1)
    owner = rng.randrange(1 << 18)
    blocks, nxt = [], 0
    water = False
    for _ in range(rng.randrange(0, 4)):
        n = rng.randrange(0, 200)
        o = heap.alloc(44 * n)
        for i in range(n):
            at = o + 44 * i
            run = rng.randrange(0, 16)
            start = nxt if rng.random() > 0.003 else nxt + 1
            mesh = owner if rng.random() > 0.003 else owner ^ 1
            struct.pack_into("<II", heap.b, at, mesh, (start << 4) | run)
            pf18 = rng.getrandbits(32) if rng.random() < 0.05 else 0
            struct.pack_into("<I", heap.b, at + 0x18, pf18)
            water = water or bool(pf18 & 0x80000)
            nxt += run
        blocks.append((o, n))
    pool = nxt if rng.random() > 0.1 else nxt + rng.randrange(-3, 4)
    return bytes(heap.b), blocks, owner, max(pool, 0)


def gen_volumes(rng):
    heap = Heap(rng)
    dc = 2 * rng.randrange(1, 40)
    vol = heap.alloc(96 * dc)
    for p in range(dc // 2):
        o0 = vol + 192 * p
        o1 = o0 + 96
        head = struct.pack("<Q3f", rng.getrandbits(64), *(rng.uniform(-4096, 4096) for _ in range(3)))
        heap.b[o0:o0 + 20] = head
        heap.b[o1:o1 + 20] = head
        vc0 = rng.choice((4, 8))
        heap.b[o0 + 0x14], heap.b[o0 + 0x15] = rng.choice((0, 4)), vc0
        heap.b[o1 + 0x14], heap.b[o1 + 0x15] = 8, 4
        heap.b[o0 + 0x16:o0 + 0x16 + 4 * vc0] = rng.randbytes(4 * vc0)
        heap.b[o1 + 0x16:o1 + 0x16 + 16] = rng.randbytes(16)
        heap.b[o0 + 0x56:o0 + 0x5A] = rng.randbytes(4)
        heap.b[o1 + 0x56:o1 + 0x5A] = rng.randbytes(4)
        r = rng.random()
        if r < 0.01:
            heap.b[o1 + rng.randrange(20)] ^= 1 << rng.randrange(8)
        elif r < 0.02:
            struct.pack_into("<f", heap.b, o0 + 8 + 4 * rng.randrange(3), rng.choice((float("nan"), float("inf"))))
        elif r < 0.03:
            heap.b[rng.choice((o0 + 0x14, o0 + 0x15, o1 + 0x14, o1 + 0x15))] = rng.randrange(256)
        elif r < 0.04:
            o = rng.choice((o0, o1))
            heap.b[o + rng.choice((0x50, 0x55, 0x5A, 0x5F))] = rng.randrange(1, 256)
    return bytes(heap.b), vol, dc


def gen_grid(rng):
    """A grid resource with a random octree in a random 16-way table, and rare corruptions of each rule."""
    groups = 1 << rng.randrange(0, 7)
    entries = rng.randrange(0, 16 * groups + 1)
    keys = [1]
    while len(keys) < entries:
        parent = rng.choice(keys)
        if parent < 1 << 60:
            child = (parent << 3) | rng.randrange(8)
            if child not in keys:
                keys.append(child)
    keys = keys[:entries]
    r = rng.random()
    if keys and r < 0.03:
        keys[rng.randrange(len(keys))] = rng.choice((0, 2, 16, 72, rng.getrandbits(64), keys[0]))
    elif keys and r < 0.04:
        keys[0] = 8
    parents = {k >> 3 for k in keys if k != 1}
    words = [0x0000FF00 if k in parents else rng.choice((0x0000FF00, 0x00020001, 0x7803, 2, rng.getrandbits(32)))
             for k in keys]
    if keys and rng.random() < 0.03:
        words[rng.randrange(len(words))] = rng.getrandbits(32)
    values = list(range(entries))
    rng.shuffle(values)
    if entries > 1 and rng.random() < 0.03:
        values[rng.randrange(entries)] = rng.choice((values[0], entries, rng.getrandbits(64)))
    slots = rng.sample(range(16 * groups), entries)
    parts = fixtures.build_grid(groups=groups, entries=entries, keys=keys or None, words=words or None,
                                declared_entries=entries)
    b = bytearray(parts[0])
    for g in range(groups):
        b[0x60 + nd.GRID_GROUP * g:0x60 + nd.GRID_GROUP * g + 16] = bytes([0x80] * 16)
        b[0x60 + nd.GRID_GROUP * g + 16:0x60 + nd.GRID_GROUP * (g + 1)] = rng.randbytes(256)
    for k, slot in enumerate(slots):
        g, s = divmod(slot, 16)
        base = 0x60 + nd.GRID_GROUP * g
        b[base + s] = rng.randrange(0x80)
        struct.pack_into("<QQ", b, base + 16 + 16 * s, keys[k], values[k] & 0xFFFFFFFFFFFFFFFF)
    payload = struct.unpack_from("<Q", b, 0x20)[0] - BASE
    for k, value in enumerate(values):
        if 0 <= value < entries:
            struct.pack_into("<I", b, payload + 4 * value, words[k])
    r = rng.random()
    if r < 0.02:
        b[0x60 + rng.randrange(16)] = rng.randrange(0x81, 0x100)
    elif r < 0.04:
        struct.pack_into("<I", b, 0x14, entries + rng.choice((-1, 1)))
    elif r < 0.06:
        struct.pack_into("<Q", b, 0x20, BASE + len(b) - rng.randrange(0, 8))
    return bytes(b)


@needs_native
class KernelParityTest(unittest.TestCase):
    CASES = 1500

    def test_quadtree(self):
        rng = random.Random(0x5154)
        outcomes = Counter()
        for i in range(self.CASES):
            b, root, n_polys = gen_quadtree(rng)
            limits = (rng.choice((0, 1, 3)), rng.choice((1, 5, 50))) if i % 10 == 0 else None
            saved = nd.MAX_QT_DEPTH, nd.MAX_QT_NODES
            if limits:
                nd.MAX_QT_DEPTH, nd.MAX_QT_NODES = limits
            try:
                want = reference(nd._walk_quadtree, b, root, n_polys)
                got = nd._walk_quadtree(b, root, n_polys)
                fast = _navmesh_native.walk_quadtree(b, root, n_polys, nd.MAX_QT_DEPTH, nd.MAX_QT_NODES)
            finally:
                nd.MAX_QT_DEPTH, nd.MAX_QT_NODES = saved
            self.assertIsNotNone(fast, i)
            self.assertEqual(got, want, i)
            self.assertEqual(tuple(map(type, got)), tuple(map(type, want)), i)
            outcomes[want[0]] += 1
        self.assertGreater(outcomes[True], self.CASES // 10)
        self.assertGreater(outcomes[False], self.CASES // 10)

    def test_poly_chain(self):
        rng = random.Random(0x9017)
        outcomes = Counter()
        for i in range(self.CASES):
            b, blocks, owner, pool = gen_polys(rng)
            want = reference(nd._poly_chain, b, blocks, owner, pool)
            self.assertIsNotNone(_navmesh_native.poly_chain(b, blocks, owner, pool, 0x80000), i)
            self.assertEqual(nd._poly_chain(b, blocks, owner, pool), want, i)
            outcomes[want] += 1
        self.assertGreater(outcomes[(True, False)] + outcomes[(True, True)], self.CASES // 10)
        self.assertGreater(outcomes[(False, False)] + outcomes[(False, True)], self.CASES // 10)
        self.assertGreater(outcomes[(True, True)], 10)

    def test_volumes(self):
        rng = random.Random(0x7011)
        outcomes = Counter()
        want_stats, got_stats = nd.new_stats()["navmesh"], nd.new_stats()["navmesh"]
        for i in range(self.CASES):
            b, vol, dc = gen_volumes(rng)
            self.assertIsNotNone(_navmesh_native.walk_volumes(b, vol, dc), i)
            want = reference(nd._walk_volumes, b, vol, dc, want_stats)
            self.assertEqual(nd._walk_volumes(b, vol, dc, got_stats), want, i)
            self.assertEqual(canon(got_stats), canon(want_stats), i)
            outcomes[want] += 1
            if i % 100 == 0:  # restart the counters so a later first-seen value can land in a fresh order
                want_stats, got_stats = nd.new_stats()["navmesh"], nd.new_stats()["navmesh"]
        self.assertGreater(outcomes[True], self.CASES // 10)
        self.assertGreater(outcomes[False], self.CASES // 10)

    def test_grid(self):
        rng = random.Random(0x6121)
        outcomes = Counter()
        want_stats, got_stats = nd.new_stats()["grid"], nd.new_stats()["grid"]
        for i in range(self.CASES):
            b = gen_grid(rng)
            want = reference(nd.decode_grid, b, 1, want_stats, True)
            got = nd.decode_grid(b, 1, got_stats, True)
            self.assertEqual(canon(got), canon(want), i)
            self.assertEqual(canon(got_stats), canon(want_stats), i)
            outcomes[tuple(want[1].get(k) for k in ("values_bijective", "keys_octree", "inner_payload_ff00"))] += 1
            if i % 50 == 0:
                want_stats, got_stats = nd.new_stats()["grid"], nd.new_stats()["grid"]
        self.assertGreater(outcomes[(True, True, True)], self.CASES // 10)
        for partial in ((False, None, None), (True, False, None), (True, True, False), (True, None, None)):
            self.assertGreater(outcomes[partial], 5, (partial, outcomes))

    def test_edge_cases(self):
        nat = _navmesh_native
        self.assertIsNone(nat.walk_quadtree(bytearray(0x60), 0, 1, 32, 1 << 20))  # not bytes: reference
        self.assertIsNone(nat.walk_quadtree(bytes(0x60), -1, 1, 32, 1 << 20))
        self.assertEqual(nat.walk_quadtree(bytes(0x60), 0, 1, 32, 1 << 20), (True, 1, 0, 0))
        self.assertEqual(nat.walk_quadtree(b"", 0, 1, 32, 1 << 20), (False, 0, 0, 0))
        self.assertEqual(nat.walk_quadtree(bytes(0x60), 0x60, 1, 32, 1 << 20), (False, 0, 0, 0))
        # A block past the buffer is the reference's to handle; the kernel declines it.
        self.assertIsNone(nat.poly_chain(bytes(44), [(0, 2)], 0, 0, 0x80000))
        self.assertIsNone(nat.poly_chain(bytes(44), [(1 << 63, 1 << 62)], 0, 0, 0x80000))
        self.assertEqual(nat.poly_chain(bytes(44), [], 0, 0, 0x80000), (True, False))
        self.assertEqual(nat.poly_chain(bytes(44), [(0, 1)], 0, 0, 0x80000), (True, False))
        self.assertEqual(nat.poly_chain(bytes(44), [(0, 0)], 0, 1, 0x80000), (False, False))
        self.assertIsNone(nat.walk_volumes(bytes(96), 0, 1))  # odd count: the reference refuses first
        self.assertIsNone(nat.walk_volumes(bytes(96), 0, 2))  # past the buffer
        self.assertIsNone(nat.grid_deep(bytes(0x60), 1, 0, None))  # table past the buffer
        self.assertIsNone(nat.grid_deep(bytes(0x60 + 272), 1, 1, None))  # occupied != entries
        self.assertEqual(nat.grid_deep(bytes(0x60 + 272), 1, 16, None),
                         (False, None, None, 0, [], []))  # 16 slots all point at value 0
        for parts in (fixtures.build_grid(entries=0), fixtures.build_grid(groups=1, entries=16)):
            want_stats, got_stats = nd.new_stats()["grid"], nd.new_stats()["grid"]
            want = reference(nd.decode_grid, parts[0], 1, want_stats, True)
            self.assertEqual(canon(nd.decode_grid(parts[0], 1, got_stats, True)), canon(want))
            self.assertEqual(canon(got_stats), canon(want_stats))


def summary_without_time(doc):
    doc = dict(doc)
    doc.pop("elapsed_s", None)
    doc.pop("elapsed", None)
    return json.loads(json.dumps(doc, sort_keys=False, default=str))


@needs_native
class ScanParityTest(unittest.TestCase):
    def test_deep_scan_reports_match(self):
        rng = random.Random(0x5CA9)
        sector = (260 << 9) | 250
        streams = []
        streams += fixtures.build_0drp(total_size=0x2000, node_count=4)
        for k in range(24):
            streams += fixtures.build_navmesh(flags=rng.choice((1, 3, 9, 11)), sector=sector + k,
                                              n_verts=rng.randrange(4, 400),
                                              poly_sizes=tuple(rng.randrange(3, 9) for _ in range(rng.randrange(1, 500))),
                                              water_poly=rng.random() < 0.5, corrupt_chain=rng.random() < 0.2,
                                              qt_cycle=rng.random() < 0.2, n_volumes=rng.choice((0, 2)))
        for _ in range(40):
            streams.append(gen_grid(rng))
        streams += fixtures.build_psin(version=256)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "synthetic.rpf")
            fixtures.make_rpf(arc, streams, pad_between=8)
            rows_native, rows_ref = [], []
            got = nd.scan(arc, deep=True, rows=rows_native)
            want = reference(nd.scan, arc, None, None, True, rows_ref)
            carved = os.path.join(tmp, "carved")
            os.mkdir(carved)
            for i, raw in enumerate(streams):
                with open(os.path.join(carved, "%08x_part.bin" % (0x1000 * i)), "wb") as f:
                    f.write(raw)
            got_dir = nd.scan(carved, deep=True)
            want_dir = reference(nd.scan, carved, None, None, True, None)
        self.assertEqual(summary_without_time(got), summary_without_time(want))
        self.assertEqual(rows_native, rows_ref)
        self.assertEqual(summary_without_time(got_dir), summary_without_time(want_dir))
        self.assertGreater(want["resources"]["grid"]["resources"], 30)


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.saved_path, self.saved_cache = _navmesh_native.library_path, list(_navmesh_native._NATIVE)

    def tearDown(self):
        _navmesh_native.library_path = self.saved_path
        _navmesh_native._NATIVE[:] = self.saved_cache

    def _scan_with(self, path):
        _navmesh_native.library_path = lambda: path
        _navmesh_native._NATIVE.clear()
        self.assertIsNone(_navmesh_native.native())
        self.assertIsNone(_navmesh_native.walk_quadtree(bytes(0x60), 0, 1, 32, 1 << 20))
        self.assertIsNone(_navmesh_native.grid_deep(bytes(0x60 + 272), 1, 16, None))
        streams = fixtures.build_navmesh(flags=11, water_poly=True) + fixtures.build_grid(entries=9)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "a.rpf")
            fixtures.make_rpf(arc, streams)
            doc = nd.scan(arc, deep=True)
        self.assertEqual(doc["navmesh"]["quadtree"], {"nodes": 2, "leaves": 1, "max_depth": 1})
        self.assertEqual(doc["navmesh"]["water_bit3"], {"b3_1_w_1": 1})
        self.assertEqual(doc["grid"]["inner_nodes"], 1)

    def test_missing_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._scan_with(os.path.join(tmp, "absent.dll"))

    def test_unloadable_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "navmesh_native.dll")
            with open(bad, "wb") as f:
                f.write(b"not a shared library")
            self._scan_with(bad)

    def test_env_opt_out(self):
        code = ("import sys; sys.path.insert(0, %r); import _navmesh_native as n; "
                "assert n.native() is None; assert n.poly_chain(bytes(44), [], 0, 0, 0x80000) is None; print('off')"
                % os.path.join(ROOT, "tools"))
        env = dict(os.environ, LEONIDA_NAVMESH_NATIVE="0")
        proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "off")

    def test_declined_call_runs_the_reference(self):
        class Declines:
            calls = 0

            def __getattr__(self, name):
                def decline(*args):
                    Declines.calls += 1
                    return None
                return decline

        saved, nd.native = nd.native, Declines()
        try:
            stats = nd.new_stats()
            row, chk = nd.decode_grid(fixtures.build_grid(entries=9)[0], 1, stats["grid"], True)
            parts = fixtures.build_navmesh(flags=3, n_volumes=2)
            _, nav = nd.decode_navmesh(parts[0], 1, stats["navmesh"], True)
        finally:
            nd.native = saved
        self.assertTrue(chk["inner_payload_ff00"])
        self.assertTrue(all(nav.values()), nav)
        self.assertEqual(Declines.calls, 4)
        self.assertEqual(stats["navmesh"]["volumes"]["pairs"], 1)


@needs_native
class BenchmarkTest(unittest.TestCase):
    """Kernel timings on large synthetic inputs; prints a table and checks the native path is not slower."""

    def _time(self, fn, repeat=5):
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            result = fn()
            best = min(best, time.perf_counter() - t0)
        return best, result

    def test_benchmark(self):
        rng = random.Random(0xBE4C)
        # One wide quadtree: 4-way, depth 7 (21 845 nodes), every node with a leaf of 8 polygons.
        heap = Heap(rng)
        n_polys = 5000
        level, nodes = [heap.alloc(0x60)], 1
        root = level[0]
        for _ in range(7):
            nxt = []
            for o in level:
                for c in range(4):
                    child = heap.alloc(0x60)
                    heap.u64(o + 0x34 + 8 * c, BASE + child)
                    nxt.append(child)
            level, nodes = nxt, nodes + len(nxt)
        qt_nodes = [root]
        seen = 0
        while seen < len(qt_nodes):
            o = qt_nodes[seen]
            seen += 1
            lo = heap.alloc(0x20)
            po = heap.alloc(16, align=2)
            struct.pack_into("<8H", heap.b, po, *(rng.randrange(n_polys) for _ in range(8)))
            heap.u64(lo + 8, BASE + po)
            heap.u64(lo + 24, 8)
            heap.u64(o + 0x2C, BASE + lo)
            qt_nodes += [struct.unpack_from("<Q", heap.b, o + 0x34 + 8 * c)[0] - BASE for c in range(4)
                         if struct.unpack_from("<Q", heap.b, o + 0x34 + 8 * c)[0]]
        qt = bytes(heap.b)

        polys = bytearray(44 * 200_000)
        nxt = 0
        for i in range(200_000):
            struct.pack_into("<II", polys, 44 * i, 7, (nxt << 4) | 5)
            nxt += 5
        polys = bytes(polys)
        per = nd.SPLIT_BLOCK // 44
        blocks = [(44 * s, min(per, 200_000 - s)) for s in range(0, 200_000, per)]

        vols = []
        for _ in range(5000):
            head = struct.pack("<Q3f", rng.getrandbits(64), 1.0, 2.0, 3.0)
            a = bytearray(96)
            c = bytearray(96)
            a[:20] = c[:20] = head
            a[0x14], a[0x15], c[0x14], c[0x15] = 0, 8, 8, 4
            vols += (a, c)
        vol = b"".join(vols)

        keys = [1]
        frontier = [1]
        while len(keys) < 60_000:
            parent = frontier.pop(0)
            for octant in range(8):
                keys.append((parent << 3) | octant)
                frontier.append(keys[-1])
        keys = keys[:60_000]
        grid = fixtures.build_grid(groups=4096, entries=len(keys), keys=keys)[0]

        cases = (
            ("quadtree (%d nodes)" % nodes, lambda: nd._walk_quadtree(qt, root, n_polys)),
            ("polygon chain (200 000 polygons)", lambda: nd._poly_chain(polys, blocks, 7, 5 * 200_000)),
            ("volumes (5 000 pairs)", lambda: nd._walk_volumes(vol, 0, 10_000, nd.new_stats()["navmesh"])),
            ("grid --deep (60 000 slots)", lambda: nd.decode_grid(grid, 1, nd.new_stats()["grid"], True)[1]),
        )
        lines = []
        for name, fn in cases:
            slow, want = self._time(lambda: reference(fn), repeat=2)
            fast, got = self._time(fn)
            self.assertEqual(got, want, name)
            lines.append("%-34s python %9.2f ms  native %8.3f ms  %7.1fx" % (name, slow * 1e3, fast * 1e3,
                                                                             slow / fast))
            self.assertLess(fast, slow, name)
        print("\n" + "\n".join(lines), file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
