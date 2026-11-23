"""Bit-for-bit comparisons for the optional groom kernel, using synthetic inputs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import struct
import sys
import unittest
from unittest import mock

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "tools")]

import _smh_groom_native as native
import glb_writer
import smh_groom as groom
from test_smh_groom import build_synthetic_groom_buffers


def synthetic(guides=6, nodes=8, seed=0, threads=12, triangle_count=None):
    rng = random.Random(seed)
    points = []
    for _ in range(guides):
        x, y, z = (rng.uniform(-0.25, 0.25) for _ in range(3))
        poly = [(x, y, z)]
        for _ in range(1, nodes):
            x += rng.uniform(-0.04, 0.04)
            y += rng.uniform(-0.04, 0.04)
            z += rng.uniform(0.0, 0.07)
            poly.append((x, y, z))
        points.append(poly)
    tris = [tuple(rng.sample(range(guides), 3)) for _ in range(triangle_count or guides - 2)]
    sys_page, gpu = build_synthetic_groom_buffers(
        guides_pos=points,
        triangles=tris,
        tri_count=[(threads, max(1, threads // 3), 0, 0)] * len(tris),
        length_random=(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)),
    )
    return groom.parse(sys_page, gpu)


def python_geometry(polylines, live, mode, radius, up, *, sides=3, tip=0.25):
    normals = None
    if mode == "tubes":
        points, normals, indices = groom.build_tubes(polylines, live, radius=radius, sides=sides, tip=tip)
    elif mode == "ribbons":
        points, normals, indices = groom.build_ribbons(polylines, live, width=radius * 2.0, tip=tip)
    elif mode == "curves":
        points, indices = groom.build_lines(polylines, live)
    else:
        points, strips = [], []
        for i, poly in enumerate(polylines):
            count = min(live[i], len(poly)) if live is not None else len(poly)
            if count >= 2:
                strips.append(range(len(points), len(points) + count))
                points.extend(poly[:count])
        indices = [index for strip in strips for index in strip]

    positions, low, high = groom._pack_vec3(groom._convert_up(points, up))
    if normals is not None:
        unit = [glb_writer._normalize3(*n) for n in groom._convert_up(normals, up)]
        normals = glb_writer._pack_f32(unit, 3)[0]
    index_bytes, component = groom._pack_scalar_indices(indices, len(points))
    return positions, normals, index_bytes, component, low, high


def outcome(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except (ValueError, IndexError, OverflowError, TypeError) as error:
        return type(error), str(error)


def float_bits(value):
    if isinstance(value, float):
        return struct.pack("=d", value)
    if isinstance(value, (tuple, list)):
        return tuple(map(float_bits, value))
    if isinstance(value, dict):
        return {key: float_bits(item) for key, item in value.items()}
    return value


class FallbackTests(unittest.TestCase):
    def test_missing_library_matches_explicit_disable_for_the_public_apis(self):
        model = synthetic()
        with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "0"}):
            expected = [groom.to_glb(model, geometry=mode) for mode in groom.GEOMETRY_MODES]
            expected_strands = groom.generate_strands(model)
        with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "1"}), mock.patch.object(native, "_load", return_value=None):
            actual = [groom.to_glb(model, geometry=mode) for mode in groom.GEOMETRY_MODES]
            self.assertEqual(actual, expected)
            self.assertEqual(float_bits(groom.generate_strands(model)), float_bits(expected_strands))

    def test_disable_bypasses_library_loading(self):
        with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "0"}), mock.patch.object(native, "_load") as load:
            self.assertIsNone(native.native())
        load.assert_not_called()


@unittest.skipUnless(native.native(), "optional groom library unavailable")
class NativeTests(unittest.TestCase):
    def assert_geometry(self, polylines, live, mode, radius, up, **kwargs):
        packed = native.pack_geometry(polylines, live, mode, radius, up, **kwargs)
        self.assertIsNotNone(packed)
        positions, normals, index_bytes, component, low, high = python_geometry(
            polylines, live, mode, radius, up, **kwargs,
        )
        self.assertEqual(bytes(packed.positions), positions)
        self.assertEqual(bytes(packed.normals) if packed.normals is not None else None, normals)
        self.assertEqual(bytes(packed.indices), index_bytes)
        self.assertEqual(b"".join(chunk for chunk, _ in packed.index_groups()), index_bytes)
        self.assertEqual(packed.index_width, 2 if component == glb_writer.COMPONENT_UNSIGNED_SHORT else 4)
        self.assertEqual(float_bits((packed.lo, packed.hi)), float_bits((low, high)))

    def test_randomized_geometry_streams(self):
        rng = random.Random(39228)
        for case in range(100):
            nodes = rng.randint(2, 20)
            sides = rng.choice((3, 4, 5, 7))
            radius = rng.uniform(-0.01, 0.01)
            tip = rng.uniform(-0.5, 1.5)
            magnitude = rng.choice((1.0, 1e-9, 1e-40, 1e-315, 1e9))
            points = [
                [tuple(rng.uniform(-magnitude, magnitude) for _ in range(3)) for _ in range(nodes)]
                for _ in range(rng.randint(2, 8))
            ]
            points[0][0] = (-0.0, 0.0, -0.0)
            points[0][1] = points[0][0]
            live = None if case % 4 == 0 else [nodes] + [
                rng.randint(-3, nodes + 3) for _ in range(len(points))
            ]
            for mode in groom.GEOMETRY_MODES:
                for up in ("y", "z"):
                    with self.subTest(case=case, mode=mode, up=up):
                        self.assert_geometry(points, live, mode, radius, up, sides=sides, tip=tip)

    def test_index_width_boundary(self):
        for n in (65535, 65536):
            points = [[(float(j), -0.0, 0.0) for j in range(n)]]
            for mode in ("curves", "line-strip"):
                with self.subTest(count=n, mode=mode):
                    self.assert_geometry(points, None, mode, 0.0, "z")

    def test_seeded_strands_match_binary64(self):
        for seed in range(40):
            model = synthetic(guides=3 + seed % 9, nodes=2 + seed % 13, seed=seed)
            model.positions[0][0] = (-0.0, 0.0, -0.0)
            if seed % 5 == 0:
                model.positions[1][1] = model.positions[1][0]
                model.lengths[0] = 0.0
            for cls in (0, 1):
                with self.subTest(seed=seed, cls=cls):
                    roots = groom.scatter(model, strand_class=cls, max_accepted=31)
                    self.assertIsNotNone(native.blend_strands(model, roots, 31, groom.pcg_float))
                    actual = groom.generate_strands(model, strand_class=cls, max_strands=31)
                    with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "0"}):
                        expected = groom.generate_strands(model, strand_class=cls, max_strands=31)
                    self.assertEqual(float_bits(actual), float_bits(expected))

    def test_glbs_match_and_use_native_geometry(self):
        for seed in range(14):
            model = synthetic(guides=3 + seed % 7, nodes=2 + seed % 11, seed=seed)
            model.positions[0][0] = (-0.0, 0.0, -0.0)
            for mode in groom.GEOMETRY_MODES:
                for up in ("y", "z"):
                    opts = dict(geometry=mode, up_axis=up, scalp=bool(seed % 2), max_strands=15)
                    with self.subTest(seed=seed, mode=mode, up=up):
                        accepted = []

                        def pack(*args):
                            result = native_pack(*args)
                            if result is not None:
                                accepted.append(result)
                            return result

                        native_pack = native.pack_geometry
                        with mock.patch.object(native, "pack_geometry", side_effect=pack):
                            actual = groom.to_glb(model, **opts)
                        self.assertTrue(accepted)
                        with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "0"}):
                            expected = groom.to_glb(model, **opts)
                        self.assertEqual(actual, expected)

    def test_non_finite_and_float32_overflow_preserve_fallback_results(self):
        for value in (math.nan, math.inf, -math.inf, 1e100):
            model = synthetic()
            model.positions[0][1] = (value, 0.0, 0.0)
            for mode in groom.GEOMETRY_MODES:
                for up in ("y", "z"):
                    with self.subTest(value=value, mode=mode, up=up):
                        self.assertIsNone(native.pack_geometry(model.positions, None, mode, 0.001, up))
                        opts = dict(geometry=mode, up_axis=up, scalp=False, classes=())
                        actual = outcome(groom.to_glb, model, **opts)
                        with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "0"}):
                            expected = outcome(groom.to_glb, model, **opts)
                        self.assertEqual(actual, expected)
            if not math.isfinite(value):
                roots = groom.scatter(model)
                self.assertIsNone(native.blend_strands(model, roots, None, groom.pcg_float))
                actual = groom.generate_strands(model)
                with mock.patch.dict(os.environ, {"LEONIDA_GROOM_NATIVE": "0"}):
                    expected = groom.generate_strands(model)
                self.assertEqual(float_bits(actual), float_bits(expected))

    def test_unsupported_shapes_fall_back(self):
        points = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)]]
        self.assertIsNone(native.pack_geometry(points, None, "curves", 0.0, "y"))
        points[1].append((3.0, 0.0, 0.0))
        self.assertIsNone(native.pack_geometry(points, [2], "curves", 0.0, "y"))
        self.assertIsNone(native.pack_geometry(points, [0, -3], "tubes", 0.01, "y"))
        self.assertIsNone(native.pack_geometry(points, None, "tubes", math.nan, "y"))
        self.assertIsNone(native.pack_geometry(points, None, "ribbons", 1e308, "y"))


if __name__ == "__main__":
    unittest.main()
