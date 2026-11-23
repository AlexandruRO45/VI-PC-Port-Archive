"""The vectorised paths of glb_writer give the bytes of the one-vertex-at-a-time code, in less memory.

``pack_primitive`` / ``build_glb`` / ``build_instanced_glb`` build their float columns, normals, bounds and
instance bounds in C-level passes. Three kinds of check keep that honest, all on synthetic meshes:

* whole GLBs against the GLBs of a scalar reference (the one-vertex-at-a-time code these paths replaced,
  restated below), byte for byte, on the machine the test runs on;
* the fast helpers against straightforward reference implementations on awkward values (-0.0, subnormals,
  NaN, infinities, degenerate triangles, short positions);
* tracemalloc peaks, which must stay within 5% of the scalar reference's peak on the same machine.

Nothing here pins a hash or a byte count: float formatting, libm and the allocator differ between platforms
and Python versions, so only a comparison with the reference run on the same interpreter is portable.

    python -I tests/test_glb_writer_fast_paths.py
"""
from __future__ import annotations

import array
import hashlib
import importlib.util
import itertools
import math
import os
import struct
import sys
import tracemalloc
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_writer  # noqa: E402

INF = float("inf")
NAN = float("nan")


class _Lcg:
    """Deterministic floats that do not depend on the random module's version."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def unit(self) -> float:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0


def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    rows = b"\x00" + b"\xff\x00\x00" * 2 + b"\x00" + b"\x00\xff\x00" * 2
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _grid(n: int, seed: int, *, uvs: bool = True, normals: bool = False, name: str = "grid") -> dict:
    rnd = _Lcg(seed)
    positions, uv_list, normal_list = [], [], []
    for j in range(n):
        for i in range(n):
            positions.append((i * 0.1 + rnd.unit() * 0.01, rnd.unit() * 0.5, j * 0.1 - rnd.unit() * 0.01))
            uv_list.append((i / n, j / n))
            normal_list.append((rnd.unit() - 0.5, 1.0, rnd.unit() - 0.5))
    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b, c = a + 1, a + n
            indices += [a, c, b, b, c, c + 1]
    prim = {"positions": positions, "indices": indices, "material": "m", "name": name}
    if uvs:
        prim["uvs"] = uv_list
    if normals:
        prim["normals"] = normal_list
    return prim


def _awkward_prim() -> dict:
    """Signed zeros, subnormals, degenerate and repeated-vertex triangles, normals that cannot be scaled."""
    positions = [
        (0.0, -0.0, 0.0), (1.0, 0.0, -0.0), (0.0, 1.0, 0.0), (-0.0, -0.0, 1.0),
        (1e-30, -1e-30, 0.0), (3.0e38, 0.0, 0.0), (0.0, 3.0e38, 0.0), (1e-320, 0.0, 0.0),
    ]
    indices = [0, 1, 2, 0, 2, 3, 0, 0, 1, 1, 1, 1, 4, 4, 4, 0, 5, 6, 7, 0, 1, 2, 3, 7]
    normals = [
        (0.0, 0.0, 0.0), (-0.0, -0.0, 5.0), (1e-320, -1e-320, 0.0), (3.0e38, 3.0e38, 3.0e38),
        (1.0, 1.0, 1.0), (NAN, 1.0, 0.0), (INF, 0.0, 0.0), (-1e-300, 1e-300, 0.0),
    ]
    uvs = [(-0.0, 0.0), (NAN, INF), (-INF, -0.0), (0.5, -0.0), (1e-320, -1e-320), (2.0, 3.0), (-0.0, -0.0), (1.0, 1.0)]
    return {"positions": positions, "indices": indices, "normals": normals, "uvs": uvs, "material": "m"}


def _instances(count: int, seed: int) -> list:
    rnd = _Lcg(seed)
    out = []
    for k in range(count):
        a = rnd.unit() * 6.28
        translation = (rnd.unit() * 100, rnd.unit() * 10, rnd.unit() * 100)
        rotation = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
        scale = (1 + rnd.unit(), 1 + rnd.unit(), 1 + rnd.unit())
        if k % 7 == 0:
            translation = (0.0, -0.0, 0.0)
        if k % 11 == 0:
            rotation = (0.0, 0.0, 0.0, 1.0)
        if k % 13 == 0:
            scale = (1.0, 1.0, 1.0)
        out.append((translation, rotation, scale))
    return out


def _instanced_meshes(mod) -> list:
    packed = [mod.pack_primitive(_grid(12, seed, uvs=bool(seed % 2))) for seed in range(4)]
    return [{"name": f"m{i}", "primitives": [packed[i]], "instances": _instances(300, 40 + i)} for i in range(4)]


def _scalar_reference_module():
    """A second copy of glb_writer whose vectorised helpers are replaced by the scalar code they replaced.

    The copy is loaded from the same source file, so everything the optimisation did not touch (JSON, chunk
    layout, index packing, the node and accessor bookkeeping) is shared, and only the float columns, normals,
    bounds and instance bounds go through the one-vertex-at-a-time arithmetic below.
    """
    spec = importlib.util.spec_from_file_location("glb_writer_scalar_reference", glb_writer.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def compute_smooth_normals(positions, indices):
        ax, ay, az = mod._smooth_normal_sums_scalar(positions, indices)
        return [mod._normalize3(ax[i], ay[i], az[i]) for i in range(len(positions))]

    def pack_f32(values, components):
        if any(len(item) != components for item in values):
            raise mod.GlbError(f"expected {components} floats per vertex")
        arr = array.array(mod._FLOAT32_CODE, itertools.chain.from_iterable(values))
        if not all(map(math.isfinite, arr)):
            raise mod.GlbError("non-finite float32 vertex attribute")
        if mod._NEEDS_BYTESWAP:
            swapped = array.array(mod._FLOAT32_CODE, arr)
            swapped.byteswap()
            return swapped.tobytes(), arr
        return arr.tobytes(), arr

    def pack_primitive(prim):
        positions = list(prim["positions"])
        indices = list(prim["indices"])
        vert_count = len(positions)
        pos_bytes, pos_f32 = pack_f32(positions, 3)
        idx_bytes, idx_comp_type = mod._pack_indices(indices, vert_count)
        normals_in = prim.get("normals")
        if normals_in is None or len(normals_in) == 0:
            normals = compute_smooth_normals(positions, indices)
        elif len(normals_in) == vert_count:
            normals = [mod._normalize3(float(n[0]), float(n[1]), float(n[2])) for n in normals_in]
        else:
            raise mod.GlbError("normals length does not match positions length")
        xs, ys, zs = pos_f32[0::3], pos_f32[1::3], pos_f32[2::3]
        nrm_bytes, _ = pack_f32(normals, 3)
        uv_bytes = b""
        uvs_in = prim.get("uvs")
        if uvs_in is not None and len(uvs_in) > 0:
            if len(uvs_in) != vert_count:
                raise mod.GlbError("uvs length does not match positions length")
            clean_uvs = [
                (mod._clean_zero(float(uv[0])) if math.isfinite(float(uv[0])) else 0.0,
                 mod._clean_zero(float(uv[1])) if math.isfinite(float(uv[1])) else 0.0)
                for uv in uvs_in
            ]
            uv_bytes, _ = pack_f32(clean_uvs, 2)
        material = prim.get("material", "")
        return {
            "positions": pos_bytes, "normals": nrm_bytes, "uvs": uv_bytes, "indices": idx_bytes,
            "index_type": idx_comp_type, "vertices": vert_count, "index_count": len(indices),
            "min": [mod._clean_zero(min(xs)), mod._clean_zero(min(ys)), mod._clean_zero(min(zs))],
            "max": [mod._clean_zero(max(xs)), mod._clean_zero(max(ys)), mod._clean_zero(max(zs))],
            "material": material if isinstance(material, (str, int)) and not isinstance(material, bool) else "",
            "material_index": prim.get("material_index"),
        }

    def compute_bounds(primitives):
        lo, hi, found = [INF] * 3, [-INF] * 3, False
        for prim in primitives:
            for pos in prim.get("positions") or ():
                if len(pos) < 3:
                    continue
                xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
                if not all(math.isfinite(v) for v in xyz):
                    continue
                found = True
                for k in range(3):
                    lo[k] = min(lo[k], xyz[k])
                    hi[k] = max(hi[k], xyz[k])
        return [mod._clean_zero(v) for v in lo + hi] if found else None

    def instance_bounds(box_min, box_max, translation, rotation, scale):
        r = mod._quat_matrix(rotation)
        m = [[r[i][j] * float(scale[j]) for j in range(3)] for i in range(3)]
        c = [(float(box_min[j]) + float(box_max[j])) * 0.5 for j in range(3)]
        h = [(float(box_max[j]) - float(box_min[j])) * 0.5 for j in range(3)]
        out_min, out_max = [], []
        for i in range(3):
            ci = m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2] + float(translation[i])
            hv = abs(m[i][0]) * h[0] + abs(m[i][1]) * h[1] + abs(m[i][2]) * h[2]
            out_min.append(ci - hv)
            out_max.append(ci + hv)
        return out_min + out_max

    def grow_instance_bounds(box_min, box_max, transforms, lo, hi):
        for t, q, sc in transforms:
            b = instance_bounds(box_min, box_max, t, q, sc)
            for k in range(3):
                lo[k] = min(lo[k], b[k])
                hi[k] = max(hi[k], b[k + 3])

    mod.pack_primitive = pack_primitive
    mod.compute_bounds = compute_bounds
    mod.instance_bounds = instance_bounds
    mod._grow_instance_bounds = grow_instance_bounds
    return mod


_REFERENCE = _scalar_reference_module()


MATERIALS = {"m": {"png_bytes": _png()}}

# Each case builds GLB bytes with the glb_writer module it is given.
CASES = {
    "mesh_smooth_normals_uint16": lambda mod: mod.build_glb([_grid(70, 1)], MATERIALS),
    "mesh_smooth_normals_uint32": lambda mod: mod.build_glb([_grid(257, 2)], MATERIALS),
    "mesh_given_normals_no_uvs": lambda mod: mod.build_glb([_grid(40, 3, uvs=False, normals=True)]),
    "mesh_three_primitives": lambda mod: mod.build_glb(
        [_grid(30, 4), _grid(25, 5, normals=True), _grid(20, 6, uvs=False)], MATERIALS),
    "mesh_awkward_values": lambda mod: mod.build_glb([_awkward_prim()], MATERIALS),
    "instanced_gpu": lambda mod: mod.build_instanced_glb(_instanced_meshes(mod), MATERIALS, mode="gpu")[0],
    "instanced_nodes": lambda mod: mod.build_instanced_glb(_instanced_meshes(mod), MATERIALS, mode="nodes")[0],
}

# Cases whose peak memory is compared with the reference's (the large ones, where the saving is clear).
PEAK_CASES = ("mesh_smooth_normals_uint16", "mesh_smooth_normals_uint32", "instanced_gpu", "instanced_nodes")


def _peak_bytes(build) -> int:
    tracemalloc.start()
    try:
        build()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


class TestByteIdenticalOutput(unittest.TestCase):
    def test_glbs_match_the_scalar_reference(self):
        for name, build in CASES.items():
            with self.subTest(case=name):
                got, want = build(glb_writer), build(_REFERENCE)
                self.assertEqual(hashlib.sha256(got).hexdigest(), hashlib.sha256(want).hexdigest())
                self.assertEqual(got, want)

    def test_exact_zero_normals_and_uvs_are_not_negative_zero(self):
        prim = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "indices": [0, 1, 2],
            "normals": [(-0.0, 1.0, -0.0)] * 3,
            "uvs": [(-0.0, -0.0)] * 3,
        }
        packed = glb_writer.pack_primitive(prim)
        self.assertEqual(packed["uvs"], bytes(24))
        self.assertEqual(packed["normals"], struct.pack("<9f", *([0.0, 1.0, 0.0] * 3)))


class TestHelpersAgainstReferences(unittest.TestCase):
    TRICKY = [0.0, -0.0, 1.0, -1.0, 0.5, 1e-320, -1e-320, 5e-324, 1e-200, -1e-200, 1e-160, 1e150, -1e150,
              3.0e38, 1e300, -1e300, 1.7e308, -1.7e308, 2.2250738585072014e-308, INF, -INF, NAN]

    def _triples(self):
        values = self.TRICKY
        out = []
        for i, x in enumerate(values):
            for j, y in enumerate(values):
                out.append((x, y, values[(i * 7 + j * 3) % len(values)]))
        return out

    def test_normalize_columns_matches_normalize3_per_vertex(self):
        triples = self._triples()
        # the finite subset takes the vectorised route, the whole set takes the per-vertex route
        finite = [t for t in triples if all(math.isfinite(v) for v in t)]
        for rows in (finite, triples, finite[:1], []):
            with self.subTest(rows=len(rows)):
                got = glb_writer._normalize_columns([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
                want = [glb_writer._normalize3(*r) for r in rows]
                self.assertEqual([tuple(map(repr, row)) for row in zip(*got)], [tuple(map(repr, w)) for w in want])

    def test_smooth_normal_sums_match_the_scalar_loop(self):
        rnd = _Lcg(9)
        positions = [(rnd.unit() - 0.5, rnd.unit() - 0.5, rnd.unit() - 0.5) for _ in range(40)]
        positions[3] = (-0.0, 0.0, -0.0)
        positions[4] = positions[3]
        indices = [int(rnd.unit() * 40) for _ in range(300)] + [3, 4, 5, 7, 7, 7]
        for finite_positions in (False, True):
            with self.subTest(finite_positions=finite_positions):
                got = glb_writer._smooth_normal_sums(positions, indices, finite_positions=finite_positions)
                want = glb_writer._smooth_normal_sums_scalar(positions, indices)
                self.assertEqual([[repr(v) for v in col] for col in got], [[repr(v) for v in col] for col in want])
        # a triangle whose cross product is not finite is left out, a trailing partial triangle is ignored
        spiky = [(0.0, 0.0, 0.0), (1e200, 0.0, 0.0), (0.0, 1e200, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
        indices = [0, 1, 2, 0, 3, 4, 3, 4]
        self.assertEqual(glb_writer._smooth_normal_sums(spiky, indices), glb_writer._smooth_normal_sums_scalar(spiky, indices))
        # indices that only make an iterator
        self.assertEqual(glb_writer._smooth_normal_sums(spiky, iter(indices)),
                         glb_writer._smooth_normal_sums_scalar(spiky, indices))

    def test_unused_malformed_position_still_works(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (5.0, 5.0)]
        want = glb_writer._smooth_normal_sums_scalar(positions, [0, 1, 2])
        self.assertEqual(glb_writer._smooth_normal_sums(positions, [0, 1, 2]), want)
        with self.assertRaises(IndexError):
            glb_writer._smooth_normal_sums(positions, [0, 1, 3])

    def test_compute_smooth_normals_keeps_its_tuple_rows(self):
        rows = glb_writer._compute_smooth_normals([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [0, 1, 2])
        self.assertEqual(rows, [(0.0, 0.0, 1.0)] * 3)
        self.assertTrue(all(type(row) is tuple and all(type(v) is float for v in row) for row in rows))

    def test_compute_bounds_matches_a_plain_loop(self):
        def reference(primitives):
            lo = [INF] * 3
            hi = [-INF] * 3
            found = False
            for prim in primitives:
                for pos in prim.get("positions") or ():
                    if len(pos) < 3:
                        continue
                    xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
                    if not all(math.isfinite(v) for v in xyz):
                        continue
                    found = True
                    for k in range(3):
                        lo[k] = min(lo[k], xyz[k])
                        hi[k] = max(hi[k], xyz[k])
            return [repr(0.0 if v == 0.0 else v) for v in lo + hi] if found else None

        cases = [
            lambda: [],
            lambda: [{"positions": []}],
            lambda: [{"positions": [(1, 2, 3), (-4.5, 0.0, 7), (1e300, -1e300, -0.0)]}],
            lambda: [{"positions": [(1.0, 2.0, 3.0), (NAN, 0.0, 0.0), (INF, 1.0, 1.0), (2.0, 2.0)]},
                     {"positions": [(-1.0, 5.0, 0.0, 9.0)]}],
            lambda: [{"positions": [(NAN, 0.0, 0.0), (0.0, INF, 0.0)]}],
            lambda: [{"positions": [("1.5", "2", "-3")]}],
            lambda: [{"positions": iter([(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)])}],
            lambda: [{"positions": [[0.0, -0.0, 5.0]]}, {"positions": [(-0.0, 0.0, 5.0)]}],
        ]
        for index, make in enumerate(cases):
            with self.subTest(case=index):
                got = glb_writer.compute_bounds(make())
                self.assertEqual(None if got is None else [repr(v) for v in got], reference(make()))

    def test_instance_bounds_follow_the_quaternion_matrix_form(self):
        def reference(box_min, box_max, translation, rotation, scale):
            r = glb_writer._quat_matrix(rotation)
            m = [[r[i][j] * float(scale[j]) for j in range(3)] for i in range(3)]
            c = [(float(box_min[j]) + float(box_max[j])) * 0.5 for j in range(3)]
            h = [(float(box_max[j]) - float(box_min[j])) * 0.5 for j in range(3)]
            lo, hi = [], []
            for i in range(3):
                ci = m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2] + float(translation[i])
                hv = abs(m[i][0]) * h[0] + abs(m[i][1]) * h[1] + abs(m[i][2]) * h[2]
                lo.append(ci - hv)
                hi.append(ci + hv)
            return lo + hi

        box_min, box_max = [-0.5, 0.0, -1.5], [2.0, 0.25, 1.0]
        instances = _instances(400, 77)
        instances.append(((1, 2, 3), (0, 0, 0.6, 0.8), (1, 2, 3)))  # ints
        for translation, rotation, scale in instances:
            self.assertEqual(
                [repr(v) for v in glb_writer.instance_bounds(box_min, box_max, translation, rotation, scale)],
                [repr(v) for v in reference(box_min, box_max, translation, rotation, scale)])
        lo = [INF] * 3
        hi = [-INF] * 3
        glb_writer._grow_instance_bounds(box_min, box_max, instances[:-1], lo, hi)
        want_lo, want_hi = [INF] * 3, [-INF] * 3
        for translation, rotation, scale in instances[:-1]:
            b = reference(box_min, box_max, translation, rotation, scale)
            for k in range(3):
                want_lo[k] = min(want_lo[k], b[k])
                want_hi[k] = max(want_hi[k], b[k + 3])
        self.assertEqual([repr(v) for v in lo + hi], [repr(v) for v in want_lo + want_hi])

    def test_malformed_rotation_and_scale_still_raise(self):
        for rotation, scale in (((0, 0, 1), (1, 1, 1)), ((0, 0, 0, 1, 1), (1, 1, 1)), ((0, 0, 0, 1), (1, 1))):
            with self.subTest(rotation=rotation, scale=scale), self.assertRaises((ValueError, IndexError)):
                glb_writer.instance_bounds([0, 0, 0], [1, 1, 1], (0, 0, 0), rotation, scale)


class TestPeakMemory(unittest.TestCase):
    def test_peak_memory_stays_within_the_scalar_reference(self):
        for name in PEAK_CASES:
            with self.subTest(case=name):
                build = CASES[name]
                # The reference runs before and after, so warm-up and run order favour neither side.
                reference_first = _peak_bytes(lambda: build(_REFERENCE))
                peak = _peak_bytes(lambda: build(glb_writer))
                reference = max(reference_first, _peak_bytes(lambda: build(_REFERENCE)))
                # 5% for allocator noise; on large meshes the saving is far beyond that.
                self.assertLessEqual(peak, reference * 1.05, f"{name}: {peak} bytes vs {reference} from the reference")

    def test_pack_primitive_peak_is_below_its_inputs_times_a_small_factor(self):
        prim = _grid(120, 8)
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            glb_writer.pack_primitive(prim)
            peak = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        # a 14 400 vertex primitive: a few hundred bytes per vertex at most
        self.assertLess(peak / 14400, 400)


if __name__ == "__main__":
    unittest.main()
