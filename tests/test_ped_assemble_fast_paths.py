"""Exact-parity tests for the fast paths in tools/ped_assemble.py (zero game files).

    python -I tests/test_ped_assemble_fast_paths.py

The assembler's speedups are all supposed to leave every output bit-identical, so each
test compares a fast path against the plain definition it replaced: the full
`find_coplanar_overlap_indices` query, the row-by-row `struct` packers, the slot-by-slot
joint remap, and the separate shift-then-convert coordinate passes. The meshes are
synthetic and deliberately sit on the tolerance boundaries.
"""
from __future__ import annotations

import math
import os
import random
import struct
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import ped_assemble as pa  # noqa: E402


def _grid(name, z0, nu, nv, *, radius=0.2, height=0.2, jitter=0.0, seed=0, translation=(0.0, 0.0, 0.0),
          coord_space="z-up", role=""):
    rnd = random.Random(seed)
    positions = []
    for iv in range(nv + 1):
        for iu in range(nu):
            a = 2.0 * math.pi * iu / nu
            positions.append((
                radius * math.cos(a) + rnd.uniform(-jitter, jitter),
                radius * math.sin(a) + rnd.uniform(-jitter, jitter),
                z0 + height * iv / nv + rnd.uniform(-jitter, jitter),
            ))
    tris = []
    for iv in range(nv):
        for iu in range(nu):
            a = iv * nu + iu
            b = iv * nu + (iu + 1) % nu
            c = (iv + 1) * nu + iu
            d = (iv + 1) * nu + (iu + 1) % nu
            tris += [(a, b, c), (b, d, c)]
    return pa.MeshPart(name=name, positions=positions, triangles=tris, translation=translation,
                       coord_space=coord_space, role=role)


class OverlapQueryParity(unittest.TestCase):
    """`_overlap_query` and the callers built on it return what the full query returns."""

    def assert_same(self, target, refs, **kw):
        full_set, full_stats = pa.find_coplanar_overlap_indices(target, refs, **kw)
        fast_set, fast_stats = pa._overlap_query(
            target, refs, centroid_tol=kw.get("centroid_tol", pa.DEFAULT_WAIST_CENTROID_TOL),
            plane_tol=kw.get("plane_tol"), normal_cos_min=kw.get("normal_cos_min", pa.DEFAULT_NORMAL_COS_MIN),
            vertex_tol=kw.get("vertex_tol", pa.DEFAULT_VERTEX_QUANT_TOL))
        self.assertEqual(fast_set, full_set)
        self.assertEqual(fast_stats, full_stats)
        self.assertEqual(list(fast_stats), list(full_stats))  # same key order, it is serialised
        return full_set

    def test_partial_overlap_band(self):
        upper = _grid("hand_000_nude_cull", 0.0, 40, 30, height=1.0, seed=1)
        # Same ring spacing, shifted so the lower shell's top ring sits 1 mm above the upper's second ring.
        lower = _grid("feet_000_nude_cull", -1.0 + 1.0 / 30 + 0.001, 40, 30, height=1.0, seed=2)
        culled = self.assert_same(upper, [lower], centroid_tol=0.004)
        self.assertTrue(culled)
        self.assertLess(len(culled), len(upper.triangles))

    def test_jittered_near_misses_on_tolerance_boundary(self):
        for seed in range(6):
            tol = 0.003
            base = _grid("a", 0.0, 24, 12, jitter=0.0, seed=seed)
            # The second shell sits right at, a hair under and a hair over the centroid tolerance.
            for lift in (tol * 0.5, tol * 0.999, tol * 1.001, tol * 2.0):
                other = _grid("b", lift, 24, 12, jitter=tol * 0.2, seed=seed + 100)
                self.assert_same(base, [other], centroid_tol=tol)
                self.assert_same(other, [base], centroid_tol=tol, normal_cos_min=0.0)

    def test_exact_key_tier_with_shuffled_corner_order(self):
        base = _grid("a", 0.0, 16, 8, seed=3)
        shuffled = pa.MeshPart(
            name="b", positions=list(base.positions),
            triangles=[(t[2], t[0], t[1]) if i % 2 else (t[1], t[2], t[0]) for i, t in enumerate(base.triangles)],
            role="")
        culled = self.assert_same(base, [shuffled])
        self.assertEqual(len(culled), len(base.triangles))

    def test_translation_plane_tol_and_tolerance_variants(self):
        a = _grid("a", 0.0, 20, 10, seed=4, translation=(0.5, -0.25, 0.125))
        b = _grid("b", 0.0, 20, 10, seed=4, translation=(0.5, -0.25, 0.125 + 0.002))
        for kw in ({}, {"centroid_tol": 0.001}, {"centroid_tol": 0.01, "plane_tol": 0.0005},
                   {"vertex_tol": 1e-3}, {"normal_cos_min": 0.99}, {"centroid_tol": -0.004},
                   {"centroid_tol": 0.0}, {"centroid_tol": 5e-5}):
            self.assert_same(a, [b], **kw)

    def test_far_apart_and_empty_parts(self):
        a = _grid("a", 0.0, 12, 6, seed=5)
        far = _grid("b", 50.0, 12, 6, seed=6)
        empty = pa.MeshPart(name="e", positions=[], triangles=[], role="x")
        self.assertEqual(self.assert_same(a, [far]), set())
        self.assert_same(a, [empty])
        self.assert_same(empty, [a])
        self.assert_same(a, [])
        self.assert_same(a, [far, a])
        self.assert_same(a, [empty, far, _grid("c", 0.001, 12, 6, seed=5)])

    def test_multiple_refs(self):
        t = _grid("t", 0.0, 24, 20, height=1.0, seed=7)
        refs = [_grid("r1", 0.2, 24, 6, height=0.1, seed=7), _grid("r2", 0.7, 24, 6, height=0.1, seed=7),
                _grid("r3", 3.0, 24, 6, seed=8)]
        self.assert_same(t, refs)

    def test_degenerate_triangles_and_non_finite_values(self):
        a = _grid("a", 0.0, 10, 4, seed=9)
        a.triangles.append((0, 0, 0))
        a.triangles.append((0, 1, 0))
        b = _grid("b", 0.0, 10, 4, seed=9)
        self.assert_same(a, [b])
        self.assert_same(b, [a])
        bad = _grid("n", 0.0, 10, 4, seed=9)
        bad.positions[3] = (float("nan"), 0.0, 0.0)
        for target, ref in ((bad, b), (b, bad)):
            outcomes = []
            for fn in (
                lambda: pa.find_coplanar_overlap_indices(target, [ref]),
                lambda: pa._overlap_query(target, [ref], centroid_tol=pa.DEFAULT_WAIST_CENTROID_TOL, plane_tol=None,
                                          normal_cos_min=pa.DEFAULT_NORMAL_COS_MIN,
                                          vertex_tol=pa.DEFAULT_VERTEX_QUANT_TOL),
            ):
                try:
                    outcomes.append(("ok", fn()))
                except Exception as exc:  # noqa: BLE001 - the type and message are the comparison
                    outcomes.append((type(exc), str(exc)))
            self.assertEqual(outcomes[0], outcomes[1])

    def test_coord_space_mismatch_still_raises(self):
        a = _grid("a", 0.0, 8, 4, seed=1)
        b = _grid("b", 100.0, 8, 4, seed=1, coord_space="y-up")
        with self.assertRaisesRegex(pa.PedAssembleError, "Coordinate space mismatch"):
            pa.cull_part_overlap(a, [b])
        with self.assertRaisesRegex(pa.PedAssembleError, "Coordinate space mismatch"):
            pa.inspect_seams([a, b])

    def test_cull_part_overlap_matches_full_query(self):
        upper = _grid("hand_000_nude_cull", 0.0, 30, 20, height=1.0, seed=11)
        lower = _grid("feet_000_nude_cull", -1.0, 30, 20, height=1.0 + 0.002, seed=12)
        for compact in (False, True):
            got, stats = pa.cull_part_overlap(upper, [lower], compact_vertices=compact)
            full_set, full_stats = pa.find_coplanar_overlap_indices(upper, [lower])
            want = [t for i, t in enumerate(upper.triangles) if i not in full_set]
            if compact:
                want_part = pa.compact_part_vertices(pa.MeshPart(
                    name=upper.name, positions=list(upper.positions), triangles=want, role=upper.role))
                self.assertEqual(got.positions, want_part.positions)
                self.assertEqual(got.triangles, want_part.triangles)
            else:
                self.assertEqual(got.triangles, want)
            self.assertEqual(stats, full_stats)

    def test_standard_culling_report_matches_unnarrowed_run(self):
        parts = [
            _grid("hand_000_nude_cull", 0.0, 30, 20, height=1.0, seed=21),
            _grid("feet_000_nude_cull", -1.0, 30, 20, height=1.004, seed=22),
            _grid("bodynipplesgen_fr1_0", 0.4, 12, 4, height=0.05, seed=21),
            _grid("bodygenitalsgen_fr1_0", -0.5, 12, 4, height=0.05, seed=22),
        ]
        out, reports = pa.apply_standard_ped_culling(parts)
        with mock.patch.object(pa, "_overlap_query", side_effect=lambda t, r, **kw: pa.find_coplanar_overlap_indices(
                t, r, centroid_tol=kw["centroid_tol"], plane_tol=kw["plane_tol"],
                normal_cos_min=kw["normal_cos_min"], vertex_tol=kw["vertex_tol"])):
            want_out, want_reports = pa.apply_standard_ped_culling(parts)
        self.assertEqual(reports, want_reports)
        self.assertEqual([p.triangles for p in out], [p.triangles for p in want_out])

    def test_inspect_seams_matches_unnarrowed_run(self):
        parts = [
            _grid("hand_000_nude_cull", 0.0, 30, 20, height=1.0, seed=31),
            _grid("feet_000_nude_cull", -1.0, 30, 20, height=1.003, seed=32),
            _grid("shirt", 0.001, 30, 20, height=1.0, radius=0.2, seed=31, translation=(0.0, 0.0, 0.001)),
            _grid("far", 9.0, 10, 4, seed=33),
            pa.MeshPart(name="empty", positions=[], triangles=[], role="x"),
        ]
        got = pa.inspect_seams(parts)
        with mock.patch.object(pa, "_overlap_query", side_effect=lambda t, r, **kw: pa.find_coplanar_overlap_indices(
                t, r, centroid_tol=kw["centroid_tol"], plane_tol=kw["plane_tol"],
                normal_cos_min=kw["normal_cos_min"], vertex_tol=kw["vertex_tol"])):
            want = pa.inspect_seams(parts)
        self.assertEqual(got, want)
        self.assertTrue(got["pairwise_overlaps"])


class HelperParity(unittest.TestCase):
    def test_triangle_helpers_match_their_definitions(self):
        rnd = random.Random(1)
        for _ in range(500):
            pts = [tuple(rnd.uniform(-3, 3) for _ in range(3)) for _ in range(3)]
            off = (rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))
            tol = rnd.choice([1e-4, 4e-3, 1e-12, 0.5])
            p = [(q[0] + off[0], q[1] + off[1], q[2] + off[2]) for q in pts]
            cent = tuple((p[0][k] + p[1][k] + p[2][k]) / 3.0 for k in range(3))
            norm = pa._vec_norm(pa._vec_cross(pa._vec_sub(p[1], p[0]), pa._vec_sub(p[2], p[0])))
            self.assertEqual(pa._triangle_geometry(pts, (0, 1, 2), off), (p[0], p[1], p[2], cent, norm))
            inv = 1.0 / max(tol, 1e-12)
            q = [tuple(int(round(c * inv)) for c in pt) for pt in p]
            self.assertEqual(pa._quantize_point(p[0], tol), q[0])
            self.assertEqual(pa._canonical_tri_key(p[0], p[1], p[2], tol), tuple(sorted(q)))

    def test_degenerate_triangle_normal_is_none(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        self.assertIsNone(pa._triangle_geometry(pts, (0, 1, 2))[4])

    def test_align_matches_shift_then_convert(self):
        rnd = random.Random(2)
        positions = [tuple(rnd.uniform(-2, 2) for _ in range(3)) for _ in range(50)]
        normals = [tuple(rnd.uniform(-1, 1) for _ in range(3)) for _ in range(50)]
        for space, target in (("z-up", "y-up"), ("y-up", "z-up"), ("z-up", "z-up"), ("y-up", "y-up")):
            for apply_offset in (True, False):
                part = pa.MeshPart(name="p", positions=positions, triangles=[(0, 1, 2)], coord_space=space,
                                   translation=(0.1, -0.2, 0.3), normals=normals)
                got = pa.align_part_to_standing_space(part, standing_offset=0.75, target_space=target,
                                                      apply_offset=apply_offset)
                ox, oy, oz = pa.standing_offset_vector(0.75, space) if apply_offset else (0.0, 0.0, 0.0)
                tx, ty, tz = part.translation
                shifted = [(p[0] + tx + ox, p[1] + ty + oy, p[2] + tz + oz) for p in positions]
                want_n = list(normals)
                if space != target:
                    convert = pa.rage_to_yup_vec3 if target == "y-up" else pa.yup_to_rage_vec3
                    shifted = [convert(p) for p in shifted]
                    want_n = [convert(n) for n in want_n]
                self.assertEqual(got.positions, shifted)
                self.assertEqual(got.normals, want_n)
                self.assertEqual(got.coord_space, target)

    def test_negative_zero_survives_alignment(self):
        part = pa.MeshPart(name="p", positions=[(-0.0, -0.0, -0.0)], triangles=[(0, 0, 0)], coord_space="z-up")
        got = pa.align_part_to_standing_space(part, standing_offset=0.0, target_space="y-up", apply_offset=False)
        self.assertEqual([math.copysign(1.0, c) for c in got.positions[0]], [1.0, 1.0, -1.0])


class PackParity(unittest.TestCase):
    @staticmethod
    def reference(rows, comps, code):
        fmt = f"<{comps}{code}"
        stride = struct.calcsize(fmt)
        buf = bytearray(len(rows) * stride)
        for i, row in enumerate(rows):
            struct.pack_into(fmt, buf, i * stride, *row[:comps])
        return bytes(buf)

    def test_f32_rows_across_chunk_boundaries(self):
        rnd = random.Random(3)
        for n in (0, 1, 7, pa._PACK_CHUNK_ROWS - 1, pa._PACK_CHUNK_ROWS, pa._PACK_CHUNK_ROWS + 1, 3 * pa._PACK_CHUNK_ROWS + 5):
            for comps in (2, 3, 4):
                rows = [tuple(rnd.uniform(-1e3, 1e3) for _ in range(comps)) for _ in range(n)]
                self.assertEqual(pa._pack_f32_vec(rows, comps), self.reference(rows, comps, "f"))

    def test_u16_rows(self):
        rnd = random.Random(4)
        rows = [tuple(rnd.randrange(65536) for _ in range(4)) for _ in range(9000)]
        self.assertEqual(pa._pack_u16_vec4(rows), self.reference(rows, 4, "H"))
        self.assertEqual(pa._pack_u16_vec4([]), b"")

    def test_longer_rows_are_truncated_like_before(self):
        rows = [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]
        self.assertEqual(pa._pack_f32_vec(rows, 3), self.reference(rows, 3, "f"))

    def test_bad_input_raises_the_same_error_type(self):
        for rows, comps, code in (
            ([(1.0, 2.0), (3.0,)], 2, "f"),
            ([(1.0, 2.0, 3.0)] * 3 + [(1e40, 0.0, 0.0)], 3, "f"),
            ([(1.0, "x", 3.0)], 3, "f"),
            ([(70000, 0, 0, 0)], 4, "H"),
            ([(-1, 0, 0, 0)], 4, "H"),
        ):
            with self.assertRaises((struct.error, OverflowError, TypeError)) as want:
                self.reference(rows, comps, code)
            fn = (lambda r: pa._pack_f32_vec(r, comps)) if code == "f" else pa._pack_u16_vec4
            with self.assertRaises(type(want.exception)):
                fn(rows)

    def test_mixed_row_lengths_that_sum_correctly_are_not_packed_silently(self):
        rows = [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0)]
        with self.assertRaises(struct.error):
            self.reference(rows, 3, "f")
        with self.assertRaises(struct.error):
            pa._pack_f32_vec(rows, 3)


class RemapParity(unittest.TestCase):
    @staticmethod
    def slow(part, skeleton):
        with mock.patch.object(pa, "_remap_joint_rows_fast", return_value=None):
            return pa.remap_part_joints(part, skeleton)

    @staticmethod
    def outcome(fn):
        try:
            return ("ok", fn())
        except Exception as exc:  # noqa: BLE001
            return (type(exc), str(exc))

    def skeleton(self, names):
        return pa.SkeletonSpec(name="rig", joints=[pa.SkeletonJoint(name=n) for n in names])

    def part(self, joints, weights, names, joints_1=None, weights_1=None):
        n = len(joints)
        return pa.MeshPart(name="p", positions=[(0.0, 0.0, 0.0)] * n, triangles=[(0, 0, 0)] if n else [],
                           joints_0=joints, weights_0=weights, joints_1=joints_1, weights_1=weights_1,
                           joint_names=names)

    def assert_same(self, part, skeleton):
        self.assertEqual(self.outcome(lambda: pa.remap_part_joints(part, skeleton)),
                         self.outcome(lambda: self.slow(part, skeleton)))

    def test_named_remap_reordered_skeleton(self):
        rnd = random.Random(5)
        src = [f"J{i}" for i in range(12)]
        dst = list(reversed(src)) + ["extra"]
        joints = [tuple(rnd.randrange(12) for _ in range(4)) for _ in range(300)]
        weights = [tuple(rnd.choice([0.0, 0.25, 0.5]) for _ in range(4)) for _ in range(300)]
        part = self.part(joints, weights, src, joints, weights)
        self.assert_same(part, self.skeleton(dst))
        fast = pa._remap_joint_rows_fast(joints, weights, [(True, dst.index(n)) for n in src], len(dst))
        self.assertIsNotNone(fast)  # the clean case really takes the fast path

    def test_raw_index_space(self):
        rnd = random.Random(6)
        joints = [tuple(rnd.randrange(8) for _ in range(4)) for _ in range(200)]
        weights = [(0.5, 0.5, 0.0, 0.0)] * 200
        part = self.part(joints, weights, None)
        self.assert_same(part, self.skeleton([f"J{i}" for i in range(8)]))

    def test_error_cases_report_the_same_message(self):
        names = ["a", "b", "c"]
        good = [(0, 1, 2, 0)]
        w = [(0.5, 0.5, 0.0, 0.0)]
        cases = [
            self.part([(0, 1, 5, 0)], [(0.5, 0.5, 0.5, 0.0)], names),          # active out of range
            self.part([(0, 1, 5, 0)], w, names),                                  # inactive out of range
            self.part([(0, 1, -1, 0)], [(0.5, 0.5, 0.5, 0.0)], names),
            self.part(good, w, ["a", "b", "zzz"]),                                # missing target joint
            self.part([(0, 1, 2, 0)], [(0.0, 0.0, 0.0, 0.0)], ["a", "b", "zzz"]),  # unmapped padding slot
            self.part(good, w, ["a", "a", "c"]),                                  # ambiguous source
            self.part(good, w, ["a", "", "c"]),                                   # unnamed source slot
            self.part([(0, 1, 2)], [(0.5, 0.5, 0.0, 0.0)], names),                # ragged joints
            self.part(good, [(0.5, 0.5, 0.0)], names),                            # ragged weights
            self.part([(0.0, 1, 2, 0)], w, names),                                # float joint index
            self.part([(True, 1, 2, 0)], w, names),                               # bool joint index
            self.part([], [], names),
        ]
        for case in cases:
            for skel_names in (["a", "b", "c", "d"], ["c", "b", "a"], ["a", "b", "c", "c"]):
                self.assert_same(case, self.skeleton(skel_names))
        raw = [self.part([(0, 1, 9, 0)], w, None), self.part([(0, 1, 9, 0)], [(0.5, 0.5, 0.5, 0.5)], None),
               self.part([(0, 1, 2, 3)], w, None), self.part([(-1, 1, 2, 3)], w, None)]
        for case in raw:
            self.assert_same(case, self.skeleton(["a", "b", "c", "d"]))


class BuildParity(unittest.TestCase):
    def test_build_matches_with_every_fast_path_disabled(self):
        rnd = random.Random(7)
        parts = []
        for k in range(3):
            part = _grid(f"part{k}", 0.3 * k, 18, 10, seed=k)
            n = len(part.positions)
            part.uvs = [(rnd.random(), rnd.random()) for _ in range(n)]
            part.colors = [(rnd.random(), 0.5, 0.25, 1.0) for _ in range(n)]
            if k != 1:
                part.normals = [(0.0, 0.0, 1.0)] * n
            part.joint_names = [f"J{i}" for i in range(6)]
            part.joints_0 = [tuple(rnd.randrange(6) for _ in range(4)) for _ in range(n)]
            part.weights_0 = [(0.5, 0.25, 0.25, 0.0)] * n
            part.joints_1 = [(i % 6, 0, 0, 0) for i in range(n)]
            part.weights_1 = [(0.0, 0.0, 0.0, 0.0)] * n
            parts.append(part)
        parts.append(_grid("static", 5.0, 9, 5, seed=9))
        skel = pa.SkeletonSpec(name="rig", joints=[
            pa.SkeletonJoint(name=f"J{i}", parent=i - 1, translation=(0.0, 0.0, 0.01 * i)) for i in range(6)])

        got = pa.build_assembled_glb(parts, skeleton=skel)

        def pack_f32(rows, comps):
            return PackParity.reference(rows, comps, "f")

        with mock.patch.object(pa, "_remap_joint_rows_fast", return_value=None), \
                mock.patch.object(pa, "_pack_rows_fast", return_value=None):
            want = pa.build_assembled_glb(parts, skeleton=skel)
        self.assertEqual(got[0], want[0])
        self.assertEqual(got[1], want[1])
        self.assertEqual(pa._pack_f32_vec(parts[0].positions, 3), pack_f32(parts[0].positions, 3))


if __name__ == "__main__":
    unittest.main()
