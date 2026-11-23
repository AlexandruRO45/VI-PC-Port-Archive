"""Tests for tools/ms_pass.py (TEST-362): every piece but the GPU draw, on synthetic inputs.

    python -I tests/test_ms_pass.py
    LEONIDA_MS_PASS_DIR=<work>/ms_pass/out python -I tests/test_ms_pass.py    # also re-check a real draw
"""
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ms_pass as mp  # noqa: E402

SPEC = next(iter(mp.SHADERS.values()))


def frame(covered, width, height, colour=mp.COLOR):
    px = [mp.CLEAR] * (width * height * 4)
    for i in covered:
        px[4 * i:4 * i + 4] = colour
    return px


class Descriptors(unittest.TestCase):
    def test_raw_vsharp_words(self):
        self.assertEqual(mp.vsharp(0x400, 124), [0x400, 0, 124, 0x30000000])
        for bad in ((-4, 16), (1 << 32, 16), (0x400, 0)):
            with self.assertRaises(ValueError):
                mp.vsharp(*bad)

    def test_heap_layout_table_and_matrices(self):
        vblob, gblob = bytes(range(256)) * 2 + b"\x42" * 96, b"\x80" * 124
        view = [float(i) for i in range(16)]
        heap, layout = mp.build_heap(SPEC, vblob, gblob, view)
        self.assertEqual(len(heap), mp.HEAP_BYTES)
        for name in ("meshlet_group", "vertex_block", "object", "view", "heap_base", "table"):
            self.assertEqual(layout[name] % 0x100, 0, name)
        regions = sorted((layout[n], n) for n in ("meshlet_group", "vertex_block", "object", "view", "heap_base"))
        self.assertGreater(regions[0][0], layout["table"] + max(SPEC["slots"].values()) + 15)
        self.assertEqual(heap[layout["vertex_block"]:layout["vertex_block"] + len(vblob)], vblob)
        self.assertEqual(heap[layout["meshlet_group"]:layout["meshlet_group"] + len(gblob)], gblob)
        self.assertEqual(struct.unpack_from("<16f", heap, layout["object"]),
                         tuple(1.0 if r == c else 0.0 for r in range(4) for c in range(4)))
        self.assertEqual(struct.unpack_from("<16f", heap, layout["view"] + SPEC["view_proj"]), tuple(view))
        sizes = {"meshlet_group": len(gblob), "object": SPEC["object_bytes"], "view": SPEC["view_bytes"]}
        for name, slot in SPEC["slots"].items():
            self.assertEqual(list(struct.unpack_from("<4I", heap, layout["table"] + slot)),
                             mp.vsharp(layout[name], sizes[name]), name)

    def test_heap_that_does_not_fit_is_refused(self):
        with self.assertRaises(ValueError):
            mp.build_heap(SPEC, b"\x01" * 0x8000, b"\x02" * 0x8000, [0.0] * 16)

    def test_heap_env_round_trips_and_is_bounded(self):
        heap, _ = mp.build_heap(SPEC, b"\x07" * 608, b"\x09" * 124, [0.5] * 16)
        text = mp.heap_env(heap)
        self.assertEqual(mp.parse_heap_env(text), bytes(heap))
        self.assertNotIn("=0x0,", text + ",")
        with self.assertRaises(ValueError):
            mp.heap_env(b"\xff" * mp.HEAP_BYTES)

    def test_sgprs(self):
        layout = {"table": 0x100, "vertex_block": 0x300, "heap_base": 0x900}
        s = mp.sgpr_words(SPEC, layout)
        self.assertEqual(len(s), 32)
        self.assertEqual((s[17], s[19], s[20], s[23]), (0x100, 0x300, 0, 0x900))
        self.assertEqual(s[21], 1 << 28)
        self.assertEqual(sum(1 for w in s if w), 4)


class Camera(unittest.TestCase):
    LO, HI = (-1.0, -0.5, -0.25), (1.0, 0.5, 0.25)

    def corners(self):
        return [(x, y, z) for x in (self.LO[0], self.HI[0]) for y in (self.LO[1], self.HI[1])
                for z in (self.LO[2], self.HI[2])]

    def test_every_corner_projects_inside_the_frame(self):
        for yaw, pitch in ((35, 25), (-60, 40), (180, -10)):
            m = mp.camera(self.LO, self.HI, yaw, pitch)
            for p in self.corners():
                x, y = mp.project(m, p, 128, 128)
                self.assertTrue(0 <= x <= 128 and 0 <= y <= 128, (yaw, pitch, p))

    def test_centre_lands_on_the_frame_centre_and_up_is_up(self):
        m = mp.camera(self.LO, self.HI, 0, 0)
        x, y = mp.project(m, (0.0, 0.0, 0.0), 128, 128)
        self.assertAlmostEqual(x, 64.0, places=6)
        self.assertAlmostEqual(y, 64.0, places=6)
        self.assertLess(mp.project(m, (0.0, 0.4, 0.0), 128, 128)[1], 64.0)    # +y is up: a smaller row

    def test_a_clipped_vertex_is_refused(self):
        m = mp.camera(self.LO, self.HI, 35, 25)
        with self.assertRaises(mp.CameraError):
            mp.project(m, (500.0, 0.0, 0.0), 128, 128)
        with self.assertRaises(mp.CameraError):
            mp.camera((0, 0, 0), (0, 0, 0), 0, 90)

    def test_view_proj_words_are_the_transpose_row_by_row(self):
        m = [[r * 4 + c for c in range(4)] for r in range(4)]
        self.assertEqual(mp.view_proj_words(m), [float(c * 4 + r) for r in range(4) for c in range(4)])


class Raster(unittest.TestCase):
    def test_square_from_two_triangles_either_winding(self):
        a, b, c, d = (2.0, 2.0), (10.0, 2.0), (10.0, 10.0), (2.0, 10.0)
        for tris in ([(a, b, c), (a, c, d)], [(a, c, b), (a, d, c)]):
            covered, edge = mp.rasterize(tris, 16, 16)
            self.assertEqual(covered, {y * 16 + x for y in range(2, 10) for x in range(2, 10)})
            self.assertIn(2 * 16 + 2, edge)
            self.assertNotIn(6 * 16 + 6, edge)

    def test_degenerate_and_offscreen(self):
        self.assertEqual(mp.rasterize([((1, 1), (5, 5), (9, 9))], 16, 16), (set(), set()))
        covered, _ = mp.rasterize([((-50, -50), (-40, -50), (-50, -40))], 16, 16)
        self.assertEqual(covered, set())


class Compare(unittest.TestCase):
    def setUp(self):
        self.tris = [((4.0, 4.0), (28.0, 6.0), (8.0, 27.0))]
        self.cpu, self.edge = mp.rasterize(self.tris, 32, 32)

    def test_exact_match_passes(self):
        r = mp.compare(frame(self.cpu, 32, 32), self.cpu, self.edge)
        self.assertEqual((r["iou"], r["missing"], r["extra"], r["off_edge"]), (1.0, 0, 0, 0))
        self.assertTrue(mp.passed(r))

    def test_edge_disagreement_alone_passes_but_interior_does_not(self):
        on_edge = next(iter(self.edge & self.cpu))
        r = mp.compare(frame(self.cpu - {on_edge}, 32, 32), self.cpu, self.edge)
        self.assertEqual((r["missing"], r["off_edge"]), (1, 0))
        self.assertTrue(mp.passed(r))
        inside = next(iter(self.cpu - self.edge))
        r = mp.compare(frame(self.cpu - {inside}, 32, 32), self.cpu, self.edge)
        self.assertEqual(r["off_edge"], 1)
        self.assertFalse(mp.passed(r))

    def test_shifted_empty_full_and_wrong_colour_fail(self):
        shifted = {i + 3 for i in self.cpu if (i % 32) < 28}
        empty, full = set(), set(range(32 * 32))
        for gpu in (shifted, empty, full):
            self.assertFalse(mp.passed(mp.compare(frame(gpu, 32, 32), self.cpu, self.edge)))
        r = mp.compare(frame(self.cpu, 32, 32, colour=(1.0, 0.5, 0.25, 0.5)), self.cpu, self.edge)
        self.assertEqual(r["wrong_colour"], len(self.cpu))
        self.assertFalse(mp.passed(r))

    def test_raw_of_the_wrong_length_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "image.raw")
            with open(path, "wb") as f:
                f.write(b"\0" * (32 * 32 * 16 - 4))
            with self.assertRaises(ValueError):
                mp.read_raw(path, 32, 32)


class Inflate(unittest.TestCase):
    @staticmethod
    def blob(*payloads, pad=b""):
        out = bytearray(b"\xAA" * 16)            # whatever precedes the resource
        for p in payloads:
            out += zlib.compress(p)
            out += b"\0" * ((16 - len(out) % 16) % 16)
        return bytes(out + pad)

    def test_blocks_on_16_byte_boundaries(self):
        data = self.blob(b"system" * 100, b"graphics" * 300, pad=b"\x01trailing")
        self.assertEqual(mp.inflate_chain(io.BytesIO(data), 16), [b"system" * 100, b"graphics" * 300])

    def test_cap_is_enforced(self):
        data = self.blob(b"\0" * (1 << 20))
        with self.assertRaises(ValueError):
            mp.inflate_chain(io.BytesIO(data), 16, cap=1 << 16)

    def test_no_zlib_header_gives_nothing(self):
        self.assertEqual(mp.inflate_chain(io.BytesIO(b"\x00" * 64), 0), [])


class Wave64Mode(unittest.TestCase):
    def test_explicit_choices_do_not_probe_the_gpu(self):
        self.assertTrue(mp.wave64_mode("on"))
        self.assertFalse(mp.wave64_mode("off"))

    def test_auto_follows_the_probe_and_emulates_when_it_cannot_tell(self):
        from unittest import mock
        answers = (('{"native64": true}', False), ('{"native64": false}', True), ("not json", True), ("", True))
        for stdout, emulate in answers:
            with self.subTest(stdout=stdout), mock.patch.object(mp.subprocess, "run") as run:
                run.return_value = mock.Mock(stdout=stdout)
                self.assertEqual(mp.wave64_mode("auto"), emulate)


class FixtureShader(unittest.TestCase):
    def test_fragment_shader_writes_the_colour(self):
        for value in mp.COLOR:
            self.assertIn(f"OpConstant %float {value!r}", mp.FS_ASM)
        self.assertNotIn("Input", mp.FS_ASM)          # no inputs: the position-only mesh shader exports none


@unittest.skipUnless(os.environ.get("LEONIDA_MS_PASS_DIR"), "LEONIDA_MS_PASS_DIR is not set (run tools/ms_pass.py first)")
class RealDraw(unittest.TestCase):
    """A draw tools/ms_pass.py made from the user's own package: the CPU projection must match it."""

    def test_draw_matches_the_cpu_projection(self):
        out = os.environ["LEONIDA_MS_PASS_DIR"]
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        self.assertIn(man["fssm_sha256"], mp.SHADERS)
        r = mp.check_folder(out, man)
        self.assertTrue(mp.passed(r), mp.summary(r))
        self.assertGreater(r["gpu"], 0.02 * man["width"] * man["height"])


if __name__ == "__main__":
    sys.exit(unittest.main())
