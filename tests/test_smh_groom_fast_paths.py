"""Parity tests for the fast paths of tools/smh_groom.py (synthetic data only, stdlib only).

The GLB export builds tubes, ribbons and lines, converts them to the glTF Y-up frame, normalises the normals
and packs everything as float32. Those steps are written as fused, call-free loops; every test here compares
them with the plain per-element code they replaced (kept below as ``reference_*``), value for value and byte
for byte, including the -0.0, NaN and infinity edge cases.

Run with:
    python -I tests/test_smh_groom_fast_paths.py
"""
from __future__ import annotations

import gc
import itertools
import math
import os
import random
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("tools", "tests"):
    _path = os.path.join(ROOT, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import glb_writer  # noqa: E402
import smh_groom  # noqa: E402
from test_smh_groom import build_synthetic_groom_buffers  # noqa: E402

SPECIAL = (0.0, -0.0, 1e-310, -1e-310, 5e-324, 1.0, -1.0, 3e38, -3e38, 1e300, math.inf, -math.inf, math.nan)


# --- Reference implementations: the straightforward code the fast paths replaced ---------------------------


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def reference_convert_up(points, up_axis="z"):
    if up_axis == "y":
        return [(_clean_zero(x), _clean_zero(y), _clean_zero(z)) for x, y, z in points]
    return [(_clean_zero(x), _clean_zero(z), _clean_zero(-y)) for x, y, z in points]


def reference_convert_normals(normals, up_axis="z"):
    return [glb_writer._normalize3(*n) for n in reference_convert_up(normals, up_axis)]


def reference_polyline_frames(poly):
    k = len(poly)
    sides, ups = [], []
    hypot = math.hypot
    for j in range(k):
        if k == 1:
            tx, ty, tz = 0.0, 0.0, 1.0
        elif j == 0:
            tx = poly[1][0] - poly[0][0]
            ty = poly[1][1] - poly[0][1]
            tz = poly[1][2] - poly[0][2]
        elif j == k - 1:
            tx = poly[j][0] - poly[j - 1][0]
            ty = poly[j][1] - poly[j - 1][1]
            tz = poly[j][2] - poly[j - 1][2]
        else:
            tx = 0.5 * (poly[j + 1][0] - poly[j - 1][0])
            ty = 0.5 * (poly[j + 1][1] - poly[j - 1][1])
            tz = 0.5 * (poly[j + 1][2] - poly[j - 1][2])
        norm = hypot(tx, ty, tz)
        if norm > 1e-9:
            inv = 1.0 / norm
            tx, ty, tz = tx * inv, ty * inv, tz * inv
        else:
            tx, ty, tz = 0.0, 0.0, 1.0
        if abs(tz) < 0.9:
            sx, sy, sz = ty, -tx, 0.0
        else:
            sx, sy, sz = 0.0, tz, -ty
        sn = hypot(sx, sy, sz)
        if sn > 1e-9:
            inv_s = 1.0 / sn
            sx, sy, sz = sx * inv_s, sy * inv_s, sz * inv_s
        else:
            sx, sy, sz = 1.0, 0.0, 0.0
        ux = ty * sz - tz * sy
        uy = tz * sx - tx * sz
        uz = tx * sy - ty * sx
        sides.append((sx, sy, sz))
        ups.append((ux, uy, uz))
    return sides, ups


def _seg_limit(live, idx, k):
    seg_limit = (live[idx] - 1) if live is not None else (k - 1)
    return max(min(seg_limit, k - 1), 0)


def reference_build_tubes(polylines, live=None, *, radius=0.0004, tip=0.25, sides=3):
    if not polylines or sides < 3:
        return [], [], []
    k = len(polylines[0])
    if k < 2:
        return [], [], []
    ring_trig = [
        (math.cos(s * (2.0 * math.pi / sides)), math.sin(s * (2.0 * math.pi / sides))) for s in range(sides)
    ]
    denom = float(k - 1)
    tapers = [(1.0 - (1.0 - tip) * (j / denom)) * radius for j in range(k)]
    verts, normals, indices = [], [], []
    for idx, poly in enumerate(polylines):
        base_poly = idx * k * sides
        side_vecs, up_vecs = reference_polyline_frames(poly)
        for j in range(k):
            px, py, pz = poly[j]
            sx, sy, sz = side_vecs[j]
            ux, uy, uz = up_vecs[j]
            rad = tapers[j]
            for ca, sa in ring_trig:
                rx = ca * sx + sa * ux
                ry = ca * sy + sa * uy
                rz = ca * sz + sa * uz
                verts.append((px + rx * rad, py + ry * rad, pz + rz * rad))
                normals.append((rx, ry, rz))
        for j in range(_seg_limit(live, idx, k)):
            ring0 = base_poly + j * sides
            ring1 = ring0 + sides
            for s in range(sides):
                s_next = (s + 1) % sides
                a, b, c, d = ring0 + s, ring0 + s_next, ring1 + s, ring1 + s_next
                indices.extend((a, b, c, b, d, c))
    return verts, normals, indices


def reference_build_ribbons(polylines, live=None, *, width=0.0008, tip=0.25):
    if not polylines:
        return [], [], []
    k = len(polylines[0])
    if k < 2:
        return [], [], []
    half_w = 0.5 * width
    denom = float(k - 1)
    half_tapers = [(1.0 - (1.0 - tip) * (j / denom)) * half_w for j in range(k)]
    verts, normals, indices = [], [], []
    for idx, poly in enumerate(polylines):
        base_poly = idx * k * 2
        side_vecs, up_vecs = reference_polyline_frames(poly)
        for j in range(k):
            px, py, pz = poly[j]
            sx, sy, sz = side_vecs[j]
            up = up_vecs[j]
            hw = half_tapers[j]
            verts.append((px - sx * hw, py - sy * hw, pz - sz * hw))
            normals.append(up)
            verts.append((px + sx * hw, py + sy * hw, pz + sz * hw))
            normals.append(up)
        for j in range(_seg_limit(live, idx, k)):
            r0 = base_poly + j * 2
            r1 = r0 + 2
            indices.extend((r0, r0 + 1, r1, r0 + 1, r1 + 1, r1))
    return verts, normals, indices


def reference_build_lines(polylines, live=None):
    if not polylines:
        return [], []
    k = len(polylines[0])
    verts, indices = [], []
    for idx, poly in enumerate(polylines):
        base = idx * k
        verts.extend(poly)
        for j in range(_seg_limit(live, idx, k)):
            indices.extend((base + j, base + j + 1))
    return verts, indices


def reference_generate_strands(groom, count, strand_class=0, region_mask=None, max_strands=None):
    """The per-node ``generate_strands`` loop, evaluated for every node even after a strand is fully trimmed."""
    roots = smh_groom.scatter(groom, count, strand_class, None, region_mask, max_strands)
    positions, triangles, lengths, uvs = groom.positions, groom.triangles, groom.lengths, groom.uvs
    ppg = groom.nodes_per_guide
    amount, low = groom.length_random
    out = {"points": [], "live": [], "length": [], "uv": [], "tri": [], "bary": []}
    for i in range(len(roots["tri"])):
        if roots["culled"][i]:
            continue
        if max_strands is not None and len(out["points"]) >= max_strands:
            break
        tri_i = roots["tri"][i]
        b0, b1, b2 = roots["bary"][i]
        g0, g1, g2 = triangles[tri_i]
        guide_len = b0 * lengths[g0] + b1 * lengths[g1] + b2 * lengths[g2]
        factor = low + smh_groom.pcg_float(roots["seed"][i]) * (1.0 - low)
        target = guide_len * (1.0 + (factor - 1.0) * amount)
        if target < 0.0:
            target = 0.0
        blend = [
            tuple(
                b0 * positions[g0][k][c] + b1 * positions[g1][k][c] + b2 * positions[g2][k][c]
                for c in range(3)
            )
            for k in range(ppg)
        ]
        prev = cur = blend[0]
        pts = [cur]
        remaining, used, live_nodes = target, 0.0, 1
        for k in range(1, ppg):
            nxt = blend[k]
            dx, dy, dz = nxt[0] - prev[0], nxt[1] - prev[1], nxt[2] - prev[2]
            prev = nxt
            seg_len = max(math.hypot(dx, dy, dz), 2.0 ** -23)
            if remaining > 0.0:
                use = seg_len if remaining >= seg_len else remaining
                if use < 1e-9:
                    use = 0.0
                if use > 0.0:
                    scale = use / seg_len
                    cur = (cur[0] + dx * scale, cur[1] + dy * scale, cur[2] + dz * scale)
                    used += use
                    live_nodes += 1
                remaining -= seg_len
            pts.append(cur)
        u0, v0 = uvs[g0]
        u1, v1 = uvs[g1]
        u2, v2 = uvs[g2]
        out["points"].append(pts)
        out["live"].append(live_nodes)
        out["length"].append(used)
        out["uv"].append((b0 * u0 + b1 * u1 + b2 * u2, b0 * v0 + b1 * v1 + b2 * v2))
        out["tri"].append(tri_i)
        out["bary"].append((b0, b1, b2))
    return out


# --- Synthetic data ----------------------------------------------------------------------------------------


def make_groom(seed: int, *, ppg: int = 6, guides: int = 12, tris: int = 20, length_random=(0.4, 0.5)):
    rng = random.Random(seed)
    guide_pos = []
    for _ in range(guides):
        ox, oy, oz = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.7 + rng.uniform(0.0, 0.05)
        pts = []
        for k in range(ppg):
            if k and rng.random() < 0.15:
                pts.append(pts[-1])  # zero-length segment
            else:
                pts.append((
                    ox + 0.004 * k * rng.uniform(0.5, 1.5),
                    oy + 0.003 * k * rng.uniform(0.5, 1.5),
                    oz + 0.012 * k * rng.uniform(0.5, 1.5),
                ))
        guide_pos.append(pts)
    sysd, gpu = build_synthetic_groom_buffers(
        guides_pos=guide_pos,
        triangles=[(rng.randrange(guides), rng.randrange(guides), rng.randrange(guides)) for _ in range(tris)],
        tri_count=[(rng.randrange(1, 30), rng.randrange(1, 30), 0, 0) for _ in range(tris)],
        tri_region=[rng.choice((0, 0, 1)) for _ in range(tris)],
        uvs=[(rng.random(), rng.random()) for _ in range(guides)],
        length_random=length_random,
    )
    return smh_groom.parse(sysd, gpu)


def random_polylines(rng: random.Random, count: int, k: int, *, special: bool = False):
    def coord():
        if special and rng.random() < 0.4:
            return rng.choice(SPECIAL)
        return rng.uniform(-1.0, 1.0)

    return [[(coord(), coord(), coord()) for _ in range(k)] for _ in range(count)]


def flat(rows):
    return list(itertools.chain.from_iterable(rows))


def same(testcase: unittest.TestCase, got, want, msg: str = "") -> None:
    # ``repr`` tells -0.0 from 0.0 and ``==`` alone would not match NaN with NaN.
    testcase.assertEqual(repr(got), repr(want), msg)


class ConversionTests(unittest.TestCase):
    def test_convert_up_matches_clean_zero_per_component(self):
        pts = [(a, b, c) for a in SPECIAL for b in SPECIAL[:6] for c in SPECIAL[:6]]
        for up in ("z", "y"):
            same(self, smh_groom._convert_up(pts, up), reference_convert_up(pts, up), up)

    def test_convert_up_returns_the_input_float_objects_for_nonzero_values(self):
        pts = [(0.25, 0.5, 0.75)]
        out_y = smh_groom._convert_up(pts, "y")[0]
        self.assertTrue(all(a is b for a, b in zip(out_y, pts[0])))  # no per-float allocation, as before
        self.assertEqual(repr(smh_groom._convert_up([(-0.0, -0.0, -0.0)], "z")), "[(0.0, 0.0, 0.0)]")

    def test_convert_normals_matches_normalize3_for_every_class_of_vector(self):
        rng = random.Random(3)
        vecs = [tuple(rng.choice(SPECIAL) for _ in range(3)) for _ in range(3000)]
        vecs += [tuple(rng.uniform(-1.0, 1.0) * s for _ in range(3)) for s in (1e-300, 1e-160, 1.0, 1e150, 1e300)
                 for _ in range(300)]
        vecs += [(0.0, 0.0, 0.0), (-0.0, -0.0, 0.0), (5e-324, 0.0, 0.0), (1e-310, 1e-310, 1e-310)]
        for up in ("z", "y"):
            same(self, smh_groom._convert_normals(vecs, up), reference_convert_normals(vecs, up), up)

    def test_pack_vec3_matches_glb_writer_bytes_bounds_and_errors(self):
        rng = random.Random(4)
        pts = [tuple(rng.uniform(-5.0, 5.0) for _ in range(3)) for _ in range(500)] + [(-0.0, 0.0, -0.0)]
        raw, vmin, vmax = smh_groom._pack_vec3(pts)
        want_raw, floats = glb_writer._pack_f32(pts, 3)
        self.assertEqual(raw, want_raw)
        self.assertEqual(vmin, [_clean_zero(min(floats[i::3])) for i in range(3)])
        self.assertEqual(vmax, [_clean_zero(max(floats[i::3])) for i in range(3)])
        for bad in ([(1.0, math.nan, 0.0)], [(math.inf, 0.0, 0.0)], [(0.0, 0.0, -math.inf)], [(1e300, 0.0, 0.0)]):
            with self.assertRaises(glb_writer.GlbError):
                smh_groom._pack_vec3(bad)
            with self.assertRaises(glb_writer.GlbError):
                glb_writer._pack_f32(bad, 3)
        with self.assertRaises(glb_writer.GlbError):
            smh_groom._pack_vec3([(1.0, 2.0)])


class BuilderTests(unittest.TestCase):
    def test_polyline_frames_match_for_every_length_and_degenerate_shape(self):
        rng = random.Random(5)
        for k in range(0, 9):
            polys = random_polylines(rng, 20, k) + random_polylines(rng, 10, k, special=True)
            polys += [[(0.0, 0.0, 0.0)] * k, [(0.0, 0.0, float(j)) for j in range(k)],
                      [(float(j), 0.0, 0.0) for j in range(k)], [(0.0, 0.0, -float(j)) for j in range(k)]]
            for poly in polys:
                same(self, smh_groom._polyline_frames(poly), reference_polyline_frames(poly), (k, poly))

    def test_builders_match_with_every_live_shape_and_side_count(self):
        rng = random.Random(6)
        for k in (2, 3, 7):
            polys = random_polylines(rng, 9, k)
            lives = (
                None,
                [rng.randint(1, k) for _ in polys],
                [rng.randint(-2, k + 4) for _ in polys],  # clamped to 0..k-1 segments
                [0] * len(polys),
                [k] * len(polys),
            )
            for live in lives:
                for sides in (3, 4, 7):
                    same(self, smh_groom.build_tubes(polys, live, radius=0.003, tip=0.4, sides=sides),
                         reference_build_tubes(polys, live, radius=0.003, tip=0.4, sides=sides), (k, live, sides))
                same(self, smh_groom.build_ribbons(polys, live, width=0.01, tip=0.1),
                     reference_build_ribbons(polys, live, width=0.01, tip=0.1), (k, live))
                same(self, smh_groom.build_lines(polys, live), reference_build_lines(polys, live), (k, live))

    def test_builders_keep_their_empty_and_degenerate_returns(self):
        for fn, empty in ((smh_groom.build_tubes, ([], [], [])), (smh_groom.build_ribbons, ([], [], [])),
                          (smh_groom.build_lines, ([], []))):
            self.assertEqual(fn([]), empty)
        self.assertEqual(smh_groom.build_tubes([[(0.0, 0.0, 0.0)]]), ([], [], []))
        self.assertEqual(smh_groom.build_ribbons([[(0.0, 0.0, 0.0)]]), ([], [], []))
        self.assertEqual(smh_groom.build_tubes([[(0.0, 0.0, 0.0)] * 3], sides=2), ([], [], []))

    def test_fused_float_builders_equal_build_convert_normalise(self):
        rng = random.Random(7)
        for special in (False, True):
            for k in (2, 5):
                polys = random_polylines(rng, 8, k, special=special)
                live = [rng.randint(0, k + 1) for _ in polys]
                for up in ("z", "y"):
                    for sides in (3, 5):
                        verts, nrms, idx = reference_build_tubes(polys, live, radius=0.002, sides=sides)
                        got = smh_groom._tube_floats(polys, live, radius=0.002, sides=sides, z_up=up == "z")
                        want = (flat(reference_convert_up(verts, up)), flat(reference_convert_normals(nrms, up)), idx)
                        same(self, got, want, ("tubes", special, k, up, sides))
                    verts, nrms, idx = reference_build_ribbons(polys, live, width=0.004)
                    got = smh_groom._ribbon_floats(polys, live, width=0.004, z_up=up == "z")
                    want = (flat(reference_convert_up(verts, up)), flat(reference_convert_normals(nrms, up)), idx)
                    same(self, got, want, ("ribbons", special, k, up))

    def test_fused_float_builders_keep_the_empty_returns(self):
        self.assertEqual(smh_groom._tube_floats([]), ([], [], []))
        self.assertEqual(smh_groom._tube_floats([[(0.0, 0.0, 0.0)]]), ([], [], []))
        self.assertEqual(smh_groom._tube_floats([[(0.0,) * 3] * 3], sides=2), ([], [], []))
        self.assertEqual(smh_groom._ribbon_floats([]), ([], [], []))
        self.assertEqual(smh_groom._ribbon_floats([[(0.0, 0.0, 0.0)]]), ([], [], []))


class GenerateStrandsTests(unittest.TestCase):
    def test_generate_strands_matches_the_per_node_reference(self):
        # (amount, low): untrimmed, mildly trimmed, strongly trimmed, and amounts that drive targets to zero.
        for seed, length_random in enumerate(((0.0, 1.0), (0.4, 0.5), (1.0, 0.0), (2.0, 0.0), (1.0, 0.01))):
            groom = make_groom(seed, ppg=7, length_random=length_random)
            for cls in (0, 1):
                for mask, cap in ((None, None), ({1}, None), ({1}, 9), (None, 1)):
                    got = smh_groom.generate_strands(groom, count=240, strand_class=cls, region_mask=mask,
                                                     max_strands=cap)
                    want = reference_generate_strands(groom, 240, cls, mask, cap)
                    same(self, got, want, (length_random, cls, mask, cap))
                    for pts, n_live in zip(got["points"], got["live"]):
                        self.assertEqual(len(pts), groom.nodes_per_guide)
                        self.assertTrue(all(p == pts[n_live - 1] for p in pts[n_live - 1:]))  # tip repeats

    def test_two_node_and_long_guides(self):
        for ppg in (2, 3, 24):
            groom = make_groom(40 + ppg, ppg=ppg, guides=5, tris=6, length_random=(0.7, 0.3))
            same(self, smh_groom.generate_strands(groom, count=60), reference_generate_strands(groom, 60), ppg)


class GlbExportTests(unittest.TestCase):
    """``to_glb`` rebuilt from the reference builders must give the same bytes as the fused export."""

    @staticmethod
    def reference_to_glb(groom, geometry, up, classes, max_strands, scalp=True, guides=True, strands=True):
        mod = smh_groom
        saved = (mod._convert_up, mod._convert_normals, mod._tube_floats, mod._ribbon_floats,
                 mod.build_tubes, mod.build_ribbons, mod.build_lines, mod._polyline_frames,
                 mod.generate_strands)

        def tube_floats(polylines, live=None, *, radius, z_up=True, **_):
            v, n, i = reference_build_tubes(polylines, live, radius=radius)
            u = "z" if z_up else "y"
            return flat(reference_convert_up(v, u)), flat(reference_convert_normals(n, u)), i

        def ribbon_floats(polylines, live=None, *, width, z_up=True, **_):
            v, n, i = reference_build_ribbons(polylines, live, width=width)
            u = "z" if z_up else "y"
            return flat(reference_convert_up(v, u)), flat(reference_convert_normals(n, u)), i

        try:
            mod._convert_up = reference_convert_up
            mod._convert_normals = reference_convert_normals
            mod._tube_floats = tube_floats
            mod._ribbon_floats = ribbon_floats
            mod.build_lines = reference_build_lines
            if strands:
                mod.generate_strands = lambda g, count=None, strand_class=0, density_scale=None, \
                    region_mask=None, max_strands=None: reference_generate_strands(
                        g, count, strand_class, region_mask, max_strands)
            return mod.to_glb(groom, geometry=geometry, up_axis=up, classes=classes, max_strands=max_strands,
                              scalp=scalp, guides=guides)
        finally:
            (mod._convert_up, mod._convert_normals, mod._tube_floats, mod._ribbon_floats,
             mod.build_tubes, mod.build_ribbons, mod.build_lines, mod._polyline_frames,
             mod.generate_strands) = saved

    def test_to_glb_bytes_match_the_reference_pipeline(self):
        groom = make_groom(11, ppg=5, guides=10, tris=18)
        for geometry in smh_groom.GEOMETRY_MODES:
            for up in ("z", "y"):
                for classes, cap in (((0, 1), 25), ((1,), 7), ((0,), 6000)):
                    got, meta = smh_groom.to_glb(groom, geometry=geometry, up_axis=up, classes=classes,
                                                 max_strands=cap)
                    want, want_meta = self.reference_to_glb(groom, geometry, up, classes, cap)
                    self.assertEqual(got, want, (geometry, up, classes, cap))
                    self.assertEqual(meta, want_meta)

    def test_to_glb_guides_and_scalp_only_match(self):
        groom = make_groom(12, ppg=4, guides=8, tris=10)
        for geometry in smh_groom.GEOMETRY_MODES:
            got, _ = smh_groom.to_glb(groom, geometry=geometry, classes=(), scalp=False)
            want, _ = self.reference_to_glb(groom, geometry, "z", (), 10, scalp=False)
            self.assertEqual(got, want, geometry)

    def test_to_glb_refuses_non_finite_positions_like_before(self):
        groom = make_groom(13, ppg=4, guides=6, tris=8)
        groom.positions[0][1] = (math.nan, 0.0, 0.0)
        for geometry in smh_groom.GEOMETRY_MODES:
            with self.assertRaises(glb_writer.GlbError, msg=geometry):
                smh_groom.to_glb(groom, geometry=geometry, scalp=False, classes=())

    def test_fused_tube_export_peaks_below_the_tuple_pipeline(self):
        groom = make_groom(14, ppg=12, guides=30, tris=60)

        def peak(build):
            gc.collect()
            tracemalloc.start()
            try:
                build()
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        fused = peak(lambda: smh_groom.to_glb(groom, geometry="tubes", max_strands=3000))
        reference = peak(lambda: self.reference_to_glb(groom, "tubes", "z", (0, 1), 3000, strands=False))
        self.assertLessEqual(fused, reference)


if __name__ == "__main__":
    unittest.main()
