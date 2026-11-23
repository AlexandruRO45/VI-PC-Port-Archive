"""tools/ms_sweep.py on a synthetic index and synthetic slots and images: no GPU, game data or Vulkan tools needed.

    python -I tests/test_ms_sweep.py
"""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import ms_sweep  # noqa: E402

# A small hand-made slot layout: 4 vertices of position + one param, 3 primitives (rdna2spirv.mesh_compute_layout shape).
LAY = {"slot_bytes": 16 + 4 * 32 + 3 * 16, "vertices": 16, "vertex_stride": 32, "primitives": 16 + 4 * 32,
       "primitive_stride": 16, "max_vertices": 4, "max_primitives": 3, "params": {"0": 16}}
NPX = ms_sweep.W * ms_sweep.H


def meta(params=(), bindings=(), clip=False):
    caps = ["Shader", "MeshShadingEXT"] + (["ClipDistance"] if clip else [])
    return {"stage": "ms", "params": list(params), "bindings": list(bindings), "capabilities": caps}


def synthetic_index():
    index = {}
    for i in range(90):
        index[f"a{i}"] = meta()
    for i in range(60):
        index[f"b{i}"] = meta(params=range(8), bindings=(0, 1))
    for i in range(30):
        index[f"c{i}"] = meta(params=range(3), clip=True)
    for i in range(3):
        index[f"d{i}"] = meta(params=(0,), bindings=(2,), clip=True)
    index["v0"] = {"stage": "vs"}
    return index


def sizes_for(index):
    return {k: 100 + 7 * i for i, k in enumerate(sorted(index))}


def slot(nv, npr, verts=(), prims=(), total=LAY["slot_bytes"]):
    """One slot: counts, then vertices (4 floats position, 4 floats param) and primitives (3 indices, cull flag)."""
    b = bytearray(total)
    struct.pack_into("<II", b, 0, nv, npr)
    for k, v in enumerate(verts):
        struct.pack_into("<4f", b, LAY["vertices"] + k * 32, *v)
    for k, p in enumerate(prims):
        struct.pack_into("<4I", b, LAY["primitives"] + k * 16, *p)
    return bytes(b)


def image(covered=(), value=0.5):
    px = [-1.0] * (NPX * 4)
    for p in covered:
        px[4 * p: 4 * p + 4] = [value, value, value, 1.0]
    return px


class Sampling(unittest.TestCase):
    def test_strata_floor_and_whole_small_strata(self):
        index = synthetic_index()
        ms = {k: v for k, v in index.items() if v["stage"] == "ms"}
        self.assertEqual(len(ms), 183)
        picked, per = ms_sweep.pick_sample(ms, sizes_for(index), 60, 7)
        self.assertEqual(len(picked), len(set(picked)))
        self.assertTrue(all(name in ms for name in picked))
        self.assertTrue(all(sampled >= min(6, size) for size, sampled in per.values()))
        self.assertEqual(sum(sampled for _, sampled in per.values()), len(picked))
        # a stratum of one kind of shader (reads buffers, clip distances) is represented
        self.assertTrue(any(s[2] and s[3] for s in per))

    def test_the_sample_has_n_shaders_when_the_population_holds_n(self):
        # 60 candidates in three groups of 20: 19 * 20 / 60 rounds down to 6 in each group, 18 in all, so the
        # nineteenth must be handed out; the strata here are size terciles of one kind of shader.
        index = {f"m{i:02}": meta() for i in range(60)}
        ms = dict(index)
        for n in (18, 19, 20, 21, 59, 60):
            picked, per = ms_sweep.pick_sample(ms, sizes_for(index), n, 3)
            self.assertEqual((n, len(picked), len(set(picked))), (n, n, n))
            self.assertEqual(sum(sampled for _, sampled in per.values()), n)
        # more than the population is the whole population
        self.assertEqual(len(ms_sweep.pick_sample(ms, sizes_for(index), 99, 3)[0]), 60)
        # the same inputs give the same allocation
        self.assertEqual(ms_sweep.pick_sample(ms, sizes_for(index), 19, 3)[1],
                         ms_sweep.pick_sample(ms, sizes_for(index), 19, 3)[1])

    def test_fixed_seed_is_stable_and_seed_matters(self):
        index = synthetic_index()
        ms = {k: v for k, v in index.items() if v["stage"] == "ms"}
        first = ms_sweep.pick_sample(ms, sizes_for(index), 60, 7)[0]
        self.assertEqual(first, ms_sweep.pick_sample(ms, sizes_for(index), 60, 7)[0])
        self.assertNotEqual(first, ms_sweep.pick_sample(ms, sizes_for(index), 60, 8)[0])

    def test_size_bins_are_terciles(self):
        bins = ms_sweep.size_bins({f"s{i}": i for i in range(9)})
        self.assertEqual(sorted(bins.values()), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(bins["s0"], 0)
        self.assertEqual(bins["s8"], 2)

    def test_sample_command_prints_numbers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            spirv, out = os.path.join(tmp, "spirv"), os.path.join(tmp, "out")
            os.makedirs(spirv)
            index = synthetic_index()
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(index, f)
            for k, n in sizes_for(index).items():
                with open(os.path.join(spirv, k + ".spv"), "wb") as f:
                    f.write(b"\0" * n)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ms_sweep.main(["sample", "--spirv", spirv, "--out", out, "--n", "60"])
            self.assertNotIn("a1 ", buf.getvalue())
            self.assertIn("population 183", buf.getvalue())
            with open(os.path.join(out, "sample.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), int(buf.getvalue().split("sampled ")[1].split()[0]))


class PixelShader(unittest.TestCase):
    def test_inputs_follow_the_params_and_alpha_is_one(self):
        src = ms_sweep.ps_glsl([0, 3, 9])
        for loc in (0, 3, 9):
            self.assertIn(f"layout(location={loc}) in vec4 i{loc};", src)
        self.assertNotIn("location=1)", src)
        self.assertIn("o = vec4(s.rgb, 1.0);", src)

    def test_no_params_is_a_valid_constant_shader(self):
        src = ms_sweep.ps_glsl([])
        self.assertNotIn(" in vec4", src)
        self.assertIn("main", src)


class Slots(unittest.TestCase):
    GOOD = [(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1)]

    def test_counts_and_clean_geometry(self):
        data = slot(4, 2, self.GOOD, [(0, 1, 2, 0), (1, 2, 3, 0)]) * ms_sweep.GROUPS
        r = ms_sweep.analyse_slots(data, LAY)
        self.assertEqual((r["verts"], r["prims"], r["culled"], r["bad_pos"], r["bad_index"], r["over_max"]),
                         (16, 8, 0, 0, 0, 0))
        self.assertEqual(r["groups"], ms_sweep.GROUPS)

    def test_nan_inf_culled_and_out_of_range_index(self):
        verts = [(float("nan"), 0, 0, 1), (float("inf"), 0, 0, 1), (0, 0, 0, 1), (1, 1, 0, 1)]
        data = slot(4, 3, verts, [(0, 1, 2, 0), (1, 2, 3, 1), (0, 1, 7, 0)]) + slot(0, 0) * 3
        r = ms_sweep.analyse_slots(data, LAY)
        self.assertEqual((r["verts"], r["prims"], r["culled"], r["bad_pos"], r["bad_index"]), (4, 3, 1, 2, 1))
        self.assertEqual(r["groups"], 1)

    def test_a_count_above_the_maximum_is_clamped_and_counted(self):
        data = slot(99, 99, self.GOOD, []) + slot(0, 0) * 3
        r = ms_sweep.analyse_slots(data, LAY)
        self.assertEqual((r["verts"], r["prims"], r["over_max"]), (4, 3, 1))

    def test_short_output_is_none(self):
        self.assertIsNone(ms_sweep.analyse_slots(slot(0, 0) * 3, LAY))


class Images(unittest.TestCase):
    def test_equal_images(self):
        a = image([5, 6, 7])
        self.assertEqual(ms_sweep.compare_images(a, list(a)), (3, 3, 0))

    def test_coverage_difference_counts_the_pixel(self):
        self.assertEqual(ms_sweep.compare_images(image([5, 6]), image([5, 6, 7])), (2, 3, 1))

    def test_colour_beyond_tolerance_and_nan(self):
        a, b = image([5, 6]), image([5, 6], 0.5)
        b[4 * 6] = 0.6
        self.assertEqual(ms_sweep.compare_images(a, b)[2], 1)
        b = image([5, 6])
        b[4 * 5] = float("nan")
        self.assertEqual(ms_sweep.compare_images(a, b)[2], 1)
        a[4 * 5] = float("nan")
        self.assertEqual(ms_sweep.compare_images(a, b)[2], 0)   # NaN against NaN agrees
        b[4 * 5] = float("inf")
        self.assertEqual(ms_sweep.compare_images(a, b)[2], 1)   # NaN against infinity does not

    def test_within_tolerance(self):
        a, b = image([5]), image([5], 0.5)
        b[4 * 5] = 0.5 + 1e-5
        self.assertEqual(ms_sweep.compare_images(a, b)[2], 0)


class Classify(unittest.TestCase):
    def write(self, tmp, name, img, slots=None):
        path = os.path.join(tmp, name)
        data = struct.pack(f"<{len(img)}f", *img)
        if slots is not None:
            data += slots
        with open(path, "wb") as f:
            f.write(data)
        return path

    def rec(self, code=0, prims=None):
        return {"code": code, "prims": prims, "reason": ""}

    def run_case(self, rn, rc, img_n, img_c, slots):
        with tempfile.TemporaryDirectory() as tmp:
            return ms_sweep.classify(rn, rc, self.write(tmp, "n.raw", img_n), self.write(tmp, "c.raw", img_c, slots), LAY)

    def test_failures_by_mode(self):
        for rn, rc, expect in (
                (self.rec(3), self.rec(), ("timeout", "native")),
                (self.rec(), self.rec(3), ("timeout", "fallback")),
                (self.rec(4), self.rec(), ("device lost", "native")),
                (self.rec(4), self.rec(4), ("device lost", "both")),
                (self.rec(77), self.rec(), ("fails in one mode", "native: missing device feature")),
                (self.rec(), self.rec(5), ("fails in one mode", "fallback: pipeline creation")),
                (self.rec(5), self.rec(77), ("fails in both modes", "5/77"))):
            self.assertEqual(ms_sweep.classify(rn, rc, "x", "y", LAY)[:2], expect)

    def test_a_short_capture_is_a_failure_not_an_agreement(self):
        empty = slot(0, 0) * ms_sweep.GROUPS
        with tempfile.TemporaryDirectory() as tmp:
            good = self.write(tmp, "c.raw", image([1]), empty)
            native = os.path.join(tmp, "n.raw")
            for size, expect in ((0, "native: short output"), (NPX * 16 - 4, "native: short output")):
                with open(native, "wb") as f:
                    f.write(bytes(size))
                self.assertEqual(ms_sweep.classify(self.rec(prims=0), self.rec(), native, good, LAY)[:2],
                                 ("fails in one mode", expect))
            whole = self.write(tmp, "n2.raw", image([1]))
            short = os.path.join(tmp, "c2.raw")
            with open(short, "wb") as f:
                f.write(bytes(NPX * 16 - 4))
            self.assertEqual(ms_sweep.classify(self.rec(prims=0), self.rec(), whole, short, LAY)[:2],
                             ("fails in one mode", "fallback: short output"))

    def test_agree_with_and_without_geometry(self):
        empty = slot(0, 0) * ms_sweep.GROUPS
        self.assertEqual(self.run_case(self.rec(prims=0), self.rec(), image(), image(), empty)[:2],
                         ("agree, empty", ""))
        geo = slot(4, 2, Slots.GOOD, [(0, 1, 2, 0), (1, 2, 3, 0)]) * ms_sweep.GROUPS
        self.assertEqual(self.run_case(self.rec(prims=8), self.rec(), image([1, 2]), image([1, 2]), geo)[:2],
                         ("agree, geometry", ""))
        # the native count may or may not include culled primitives
        culled = slot(4, 2, Slots.GOOD, [(0, 1, 2, 0), (1, 2, 3, 1)]) * ms_sweep.GROUPS
        for native in (8, 4):
            self.assertEqual(self.run_case(self.rec(prims=native), self.rec(), image([1]), image([1]), culled)[0],
                             "agree, geometry")

    def test_primitive_count_and_image_differences(self):
        geo = slot(4, 2, Slots.GOOD, [(0, 1, 2, 0), (1, 2, 3, 0)]) * ms_sweep.GROUPS
        c, _, facts = self.run_case(self.rec(prims=5), self.rec(), image([1]), image([1]), geo)
        self.assertEqual((c, facts["prims"]), ("primitive count differs", 8))
        self.assertEqual(self.run_case(self.rec(prims=8), self.rec(), image([1, 2, 3, 4]), image([1]), geo)[:2],
                         ("image differs", "coverage"))
        self.assertEqual(self.run_case(self.rec(prims=8), self.rec(), image([1, 2, 3, 4], 0.4), image([1, 2, 3, 4]), geo)[:2],
                         ("image differs", "colour"))
        # a pixel or two on an edge is not a difference
        self.assertEqual(self.run_case(self.rec(prims=8), self.rec(), image([1, 2, 3]), image([1, 2]), geo)[0],
                         "agree, geometry")

    def test_unavailable_native_count_is_not_a_difference_and_nan_is_flagged(self):
        nan = slot(4, 1, [(float("nan"), 0, 0, 1)] + Slots.GOOD[1:], [(0, 1, 2, 0)]) * ms_sweep.GROUPS
        self.assertEqual(self.run_case(self.rec(prims=None), self.rec(), image(), image(), nan)[:2],
                         ("agree, geometry", "NaN or Inf position"))

    def test_short_fallback_output(self):
        self.assertEqual(self.run_case(self.rec(), self.rec(), image(), image(), b"")[:2],
                         ("fails in one mode", "fallback: short output"))


if __name__ == "__main__":
    unittest.main()
