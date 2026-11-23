"""Parity of the optional Rust traversal (tools/xdxr_bvh_native) with the Python path of tools/xdxr_bvh.py.

Every case runs ``trace`` or ``brute_force`` twice, once natively and once with LEONIDA_XDXR_BVH_NATIVE=0, and
compares the results bit for bit (floats by their float64 bits, dict keys in order, the stats step count) or the
exception type and message. The scenes are random synthetic triangles and instances; no game data is read. Without
the built library the parity cases are skipped and the fallback cases still run.

    cargo build --release --manifest-path tools/xdxr_bvh_native/Cargo.toml
    python -I tests/test_xdxr_bvh_native.py
"""
import contextlib
import math
import os
import random
import struct
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import xdxr_bvh as X  # noqa: E402

NATIVE = X._native()
ENV = "LEONIDA_XDXR_BVH_NATIVE"
needs_native = unittest.skipIf(NATIVE is None or os.environ.get(ENV) == "0",
                               "tools/xdxr_bvh_native is not built, or LEONIDA_XDXR_BVH_NATIVE=0")


@contextlib.contextmanager
def python_only():
    old = os.environ.get(ENV)
    os.environ[ENV] = "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ[ENV]
        else:
            os.environ[ENV] = old


def canon(x):
    """A comparable form that tells apart -0.0, NaN payloads, bool from int and tuple from list."""
    if type(x) is float:
        return ("f", struct.pack("<d", x).hex())
    if type(x) is dict:
        return ("d", tuple((k, canon(v)) for k, v in x.items()))
    if type(x) in (list, tuple):
        return (type(x).__name__, tuple(canon(v) for v in x))
    return (type(x).__name__, x)


def outcome(fn, *args, **kw):
    try:
        return ("ok", canon(fn(*args, **kw)))
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__, str(e))


def both(fn, *args, **kw):
    native = outcome(fn, *args, **kw)
    with python_only():
        python = outcome(fn, *args, **kw)
    return native, python


def traced(mem, va, o, d, **kw):
    """trace() plus its stats, so the step count is compared too."""
    stats = {"steps": 0}
    hit = X.trace(mem, va, o, d, stats=stats, **kw)
    return hit, stats["steps"]


# ---- synthetic scenes ----
def coord(rng):
    r = rng.random()
    if r < 0.03:
        return rng.choice([0.0, -0.0, 1.0, -1.0, 0.5])
    if r < 0.05:
        return rng.uniform(-1, 1) * 70000.0      # past the half range: BOX16 bounds clamp
    if r < 0.07:
        return rng.uniform(-1, 1) * 1e-6
    return rng.uniform(-4, 4)


def random_tris(rng, n):
    kind = rng.random()
    if kind < 0.3:
        return X.grid_mesh(rng.randint(1, 6), rng.uniform(1, 8))
    tris = []
    for _ in range(n):
        if rng.random() < 0.05:                   # shared or repeated corners, degenerate triangles
            p = [coord(rng) for _ in range(3)]
            tris.append([p, list(p), [coord(rng) for _ in range(3)]])
            continue
        c = [coord(rng) for _ in range(3)]
        s = rng.choice([0.01, 0.6, 2.0])
        tris.append([[c[k] + rng.uniform(-s, s) for k in range(3)] for _ in range(3)])
    return tris


def random_ray(rng, tris=None):
    r = rng.random()
    if tris and r < 0.25:                         # aimed at a vertex or an edge midpoint: the tie rules
        t = rng.choice(tris)
        a, b = rng.sample(range(3), 2)
        target = t[a] if rng.random() < 0.5 else [(t[a][c] + t[b][c]) * 0.5 for c in range(3)]
        o = [target[c] + rng.choice([0.0, rng.uniform(-5, 5)]) for c in range(3)]
        d = [target[c] - o[c] for c in range(3)]
        if d == [0.0, 0.0, 0.0]:
            d = [0.0, 0.0, -1.0]
            o[2] += 3.0
        return o, d
    o = [rng.uniform(-8, 8) for _ in range(3)]
    if r < 0.4:                                   # axis-aligned, with zero and negative-zero components
        d = [rng.choice([0.0, -0.0]) for _ in range(3)]
        d[rng.randrange(3)] = rng.choice([1.0, -1.0, 3.5, -0.25])
    elif r < 0.45:
        d = [rng.randint(-3, 3) for _ in range(3)]  # ints convert exactly
        o = [rng.randint(-8, 8) for _ in range(3)]
        if d == [0, 0, 0]:
            d = [1, 0, 0]
    elif r < 0.5:
        d = [rng.uniform(-1, 1) * rng.choice([1e-30, 1e30, 1.0]) for _ in range(3)]
    elif r < 0.52:
        d = [math.nan, 1.0, 0.0]
    elif r < 0.54:
        d = [math.inf, -1.0, 0.5]
    else:
        target = [rng.uniform(-4, 4) for _ in range(3)]
        d = [target[c] - o[c] for c in range(3)]
    return o, d


def random_tmax(rng):
    return rng.choice([X.INF, X.INF, X.INF, rng.uniform(0.1, 3.0), 2, 0.0])


def corrupt(rng, data, info, box16):
    """Flip bytes of vertex and box data (never a pointer, which could make a cycle)."""
    buf = bytearray(data)
    node_size = X.NODE16_SIZE if box16 else X.NODE32_SIZE
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.5:
            leaf = info["elements"] + X.LEAF_SIZE * rng.randrange(info["leaves"])
            off = leaf + rng.choice([rng.randrange(48), 48, 60, 61])   # vertices, geometry flags, remap byte
        else:
            node = info["node_offset"] + node_size * rng.randrange(info["nodes"])
            off = node + 16 + rng.randrange(node_size - 16)
        buf[off] = rng.randrange(256)
    return bytes(buf)


def random_transform(rng):
    if rng.random() < 0.2:
        return [[1.0, 0.0, 0.0, rng.uniform(-3, 3)], [0.0, 1.0, 0.0, rng.uniform(-3, 3)],
                [0.0, 0.0, 1.0, rng.uniform(-3, 3)]]
    a, b = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    s = [rng.choice([1.0, rng.uniform(0.3, 2.5)]) for _ in range(3)]
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    rot = [[ca * cb, -sa, ca * sb], [sa * cb, ca, sa * sb], [-sb, 0.0, cb]]
    return [[rot[r][c] * s[c] for c in range(3)] + [rng.uniform(-4, 4)] for r in range(3)]


@needs_native
class Parity(unittest.TestCase):
    def check(self, fn, *args, **kw):
        native, python = both(fn, *args, **kw)
        self.assertEqual(native, python, (args, kw))
        return native

    def test_native_path_runs(self):
        data, _ = X.build_blas(X.grid_mesh(3))
        mem = X.Memory()
        mem.place(0x10000, data)
        done, hit = NATIVE.try_trace(mem, 0x10000, [0.1, 0.2, 5.0], [0.0, 0.0, -1.0], X.INF, 0xFF, 0, None,
                                     X.Memory)
        self.assertTrue(done)
        self.assertIsNotNone(hit)
        done, _ = NATIVE.try_brute_force([(None, 0xFF, X.grid_mesh(3))], [0.1, 0.2, 5.0], [0.0, 0.0, -1.0], X.INF,
                                         0xFF, X.invert_affine, X._transform_point, X._transform_dir)
        self.assertTrue(done)

    def test_blas_random(self):
        rng = random.Random(1)
        calls = hits = 0
        for case in range(300):
            tris = random_tris(rng, rng.randint(1, 80))
            box16, pair = rng.random() < 0.5, rng.random() < 0.5
            data, info = X.build_blas(tris, geometry_index=rng.randrange(1 << 20), box16=box16, pair=pair,
                                      opaque=rng.random() < 0.8)
            if case % 4 == 3:
                data = corrupt(rng, data, info, box16)
            mem = X.Memory()
            va = rng.choice([0, 0x10000, 0x7FFF_0000_0000])
            mem.place(va, data)
            for _ in range(6):
                o, d = random_ray(rng, tris)
                r = self.check(traced, mem, va, o, d, tmax=random_tmax(rng))
                hits += r[0] == "ok" and r[1][1][0][0] == "d"
                self.check(X.brute_force, [(None, 0xFF, tris)], o, d, tmax=random_tmax(rng))
                calls += 2
        self.assertGreater(calls, 3000)
        self.assertGreater(hits, 300)

    def test_tlas_random(self):
        rng = random.Random(2)
        calls = 0
        for case in range(80):
            mem = X.Memory()
            bias = rng.choice([0, 0, 0x100000, -0x2000])
            blases, meshes = [], []
            for k in range(rng.randint(1, 3)):
                tris = random_tris(rng, rng.randint(1, 40))
                data, info = X.build_blas(tris, geometry_index=k, box16=rng.random() < 0.5, pair=rng.random() < 0.5)
                app_va = 0x100000 * (k + 2)
                mem.place(app_va + bias, data)
                blases.append((app_va, info))
                meshes.append(tris)
            insts, scene = [], []
            for i in range(rng.randint(1, 12)):
                k = rng.randrange(len(blases))
                m = random_transform(rng)
                mask = rng.choice([0xFF, 0x01, 0x80, 0x00, rng.randrange(256)])
                app_va, info = blases[k]
                if rng.random() < 0.08:                 # an instance with no BLAS
                    insts.append(X.Instance(m, 0, info["root"], info["bounds"], i, mask))
                    scene.append((m, mask, []))
                    continue
                insts.append(X.Instance(m, app_va, info["root"], info["bounds"], instance_id=i, mask=mask))
                scene.append((m, mask, meshes[k]))
            data, _ = X.build_tlas(insts, box16=rng.random() < 0.5)
            mem.place(0x1000000, data)
            for _ in range(15):
                o, d = random_ray(rng)
                ray_mask = rng.choice([0xFF, 0xFF, 0x01, 0x80, -1, 0x1FF, 0])
                self.check(traced, mem, 0x1000000, o, d, tmax=random_tmax(rng), mask=ray_mask, bias=bias)
                self.check(X.trace, mem, 0x1000000, o, d, mask=ray_mask, bias=bias)
                self.check(X.brute_force, scene, o, d, tmax=random_tmax(rng), mask=ray_mask)
                calls += 3
        self.assertGreater(calls, 1000)

    def test_nan_and_infinite_geometry(self):
        rng = random.Random(3)
        for _ in range(60):
            tris = random_tris(rng, 30)
            for t in rng.sample(tris, min(3, len(tris))):
                t[rng.randrange(3)][rng.randrange(3)] = rng.choice([math.nan, math.inf, -math.inf])
            mem = X.Memory()
            try:
                data, _ = X.build_blas(tris, box16=rng.random() < 0.5)
            except (ValueError, OverflowError):
                data = None
            if data is not None:
                mem.place(0x10000, data)
            for _ in range(8):
                o, d = random_ray(rng, tris)
                if data is not None:
                    self.check(traced, mem, 0x10000, o, d)
                self.check(X.brute_force, [(None, 0xFF, tris)], o, d)

    def test_errors_match(self):
        data, info = X.build_blas(X.grid_mesh(2), box16=True)
        mem = X.Memory()
        mem.place(0x10000, data)
        o, d = [0.1, 0.2, 5.0], [0.0, 0.0, -1.0]
        self.check(traced, mem, 0x90000, o, d)                     # nothing mapped there
        short = X.Memory()
        short.place(0x10000, data[:len(data) - 64])                # the last node is cut off
        self.check(traced, short, 0x10000, o, d)
        # a TLAS that names a BLAS root the BLAS header does not hold, and one whose BLAS is not mapped
        inst = X.Instance([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], 0x10000, info["root"] + 8,
                          info["bounds"])
        tlas, _ = X.build_tlas([inst])
        mem.place(0x200000, tlas)
        self.check(traced, mem, 0x200000, o, d)
        self.check(traced, mem, 0x200000, o, d, bias=0x1000)
        # a triangle node in a top-level tree: the root pointer swapped for a TRI0 pointer
        bad = bytearray(tlas)
        struct.pack_into("<I", bad, 48, X.ptr(X.HEADER_SIZE + 64, X.TRI0))
        mem2 = X.Memory()
        mem2.place(0x200000, bytes(bad) + bytes(256))
        self.check(traced, mem2, 0x200000, o, d)
        # brute force: a coordinate float32 cannot hold, a singular transform, a 4-point triangle, a string
        tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        for scene in ([(None, 0xFF, [[[1e39, 0, 0], tri[1], tri[2]]])],
                      [([[0.0] * 4] * 3, 0xFF, [tri])],
                      [(None, 0xFF, [tri + [tri[0]]])],
                      [(None, 0xFF, [[["x", 0, 0], tri[1], tri[2]]])],
                      [(None, 0xFF, [tri]), (None,)]):
            self.check(X.brute_force, scene, [0.2, 0.2, 1.0], [0.0, 0.0, -1.0])

    def test_inputs_the_native_path_declines(self):
        """Python handles these; the result is still the same."""
        tris = X.grid_mesh(3)
        data, _ = X.build_blas(tris, flags=0x80, pair=True)       # a leaf with QUAD_PRIM set: Python's to model
        mem = X.Memory()
        mem.place(0x10000, data)
        o, d = [0.1, 0.2, 5.0], [0.0, 0.0, -1.0]
        done, _ = NATIVE.try_trace(mem, 0x10000, o, d, X.INF, 0xFF, 0, None, X.Memory)
        self.assertFalse(done)
        self.check(traced, mem, 0x10000, o, d)

        class Sub(X.Memory):
            pass
        sub = Sub()
        sub.place(0x10000, X.build_blas(tris)[0])
        self.assertFalse(NATIVE.try_trace(sub, 0x10000, o, d, X.INF, 0xFF, 0, None, X.Memory)[0])
        self.check(traced, sub, 0x10000, o, d)
        self.check(traced, sub, 0x10000, o + [9.0], d + [0.0])     # 4 components
        self.check(traced, sub, 0x10000, [2 ** 60, 0, 1], d)       # an int past 2**53
        self.check(X.brute_force, [(None, 0xFF, tris)], o, d + [7.0])
        self.check(X.brute_force, [(None, 0xFF, [[tuple(p) for p in t] for t in tris])], tuple(o), tuple(d))
        self.assertEqual(outcome(X.brute_force, [(None, 0xFF, iter(tris))], o, d),
                         both(X.brute_force, [(None, 0xFF, tris)], o, d)[1])

    def test_region_cache_follows_the_memory(self):
        data, _ = X.build_blas(X.grid_mesh(3))
        other, _ = X.build_blas(X.random_scene(random.Random(5)))
        mem = X.Memory()
        mem.place(0x10000, data)
        o, d = [0.1, 0.2, 5.0], [0.0, 0.0, -1.0]
        first = self.check(traced, mem, 0x10000, o, d)
        mem.regions[0] = (0x10000, other)                         # replaced in place
        second = self.check(traced, mem, 0x10000, o, d)
        self.assertNotEqual(first, second)
        mem.regions.insert(0, (0x10000, data))                    # an earlier region answers first
        self.assertEqual(self.check(traced, mem, 0x10000, o, d), first)


class Fallback(unittest.TestCase):
    def test_switched_off(self):
        with python_only():
            mem = X.Memory()
            mem.place(0, X.build_blas(X.grid_mesh(2))[0])
            if NATIVE is not None:
                self.assertEqual(NATIVE.try_trace(mem, 0, [0, 0, 1.0], [0, 0, -1.0], X.INF, 0xFF, 0, None,
                                                  X.Memory), (False, None))
            self.assertEqual(X.selftest(seed=4, rays=40), 0)

    def test_library_missing(self):
        saved = X._NATIVE
        X._NATIVE = None
        try:
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
                self.assertEqual(X.selftest(seed=4, rays=40), 0)
        finally:
            X._NATIVE = saved

    def test_loader_without_a_build(self):
        import importlib.util
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "__init__.py")
            shutil.copy(os.path.join(ROOT, "tools", "xdxr_bvh_native", "__init__.py"), dst)
            spec = importlib.util.spec_from_file_location("xdxr_bvh_native_copy", dst)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIsNone(module._load())
            self.assertEqual(module.try_brute_force([], [0, 0, 0], [1, 0, 0], X.INF, 0xFF, X.invert_affine,
                                                    X._transform_point, X._transform_dir), (False, None))


@needs_native
class Speed(unittest.TestCase):
    def bench(self, fn, args_list):
        start = time.perf_counter()
        native = [fn(*a) for a in args_list]
        t_native = time.perf_counter() - start
        with python_only():
            start = time.perf_counter()
            python = [fn(*a) for a in args_list]
            t_python = time.perf_counter() - start
        self.assertEqual([canon(x) for x in native], [canon(x) for x in python])
        return t_python, t_native

    def test_speedup(self):
        rng = random.Random(6)
        tris = X.random_scene(rng, n_tris=2000, spread=8.0)
        data, info = X.build_blas(tris, box16=True)
        mem = X.Memory()
        mem.place(0x10000, data)
        rays = [random_ray(rng) for _ in range(200)]
        tp, tn = self.bench(lambda o, d: X.trace(mem, 0x10000, o, d), rays)
        bp, bn = self.bench(lambda o, d: X.brute_force([(None, 0xFF, tris)], o, d), rays[:40])
        print(f"\n  trace: {len(rays)} rays, {info['triangles']} triangles: Python {tp * 1e3:.1f} ms, "
              f"native {tn * 1e3:.2f} ms ({tp / tn:.0f}x)"
              f"\n  brute_force: 40 rays x {len(tris)} triangles: Python {bp * 1e3:.1f} ms, "
              f"native {bn * 1e3:.2f} ms ({bp / bn:.0f}x)", file=sys.stderr)
        self.assertGreater(tp / tn, 5)
        self.assertGreater(bp / bn, 5)


if __name__ == "__main__":
    unittest.main()
