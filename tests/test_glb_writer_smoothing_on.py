"""OBJ smoothing on/off directives, using hand-written geometry only.

    python -I tests/test_glb_writer_smoothing_on.py
"""
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glb_writer as G


VERTICES = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
FACES = "f 1 2 3\nf 1 4 2\n"


def parse(directives, faces=FACES, normals=""):
    return G.parse_obj(VERTICES + normals + directives + faces, is_path=False, up_axis="y")


class SmoothingOnTest(unittest.TestCase):
    def test_on_matches_group_one_after_another_setting(self):
        for previous in ("off", "0", "2"):
            with self.subTest(previous=previous):
                primitives = parse(f"s {previous}\ns on\n")
                self.assertEqual(primitives, parse(f"s {previous}\ns 1\n"))
                primitive, = primitives
                self.assertEqual(len(primitive["positions"]), 4)
                # Equal-area perpendicular triangles share their edge normal.
                packed = G.pack_primitive(primitive)
                normal = struct.unpack_from("<3f", packed["normals"])
                for actual, expected in zip(normal, (0.0, 1 / math.sqrt(2), 1 / math.sqrt(2))):
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_off_and_zero_still_disable_smoothing_after_on(self):
        for off in ("off", "0"):
            with self.subTest(off=off):
                primitive, = parse(f"s on\ns {off}\n")
                self.assertEqual(len(primitive["positions"]), 6)
                self.assertEqual(primitive["normals"], [(0.0, 0.0, 1.0)] * 3 + [(0.0, 1.0, 0.0)] * 3)

    def test_on_rejoins_numbered_group_one(self):
        source = VERTICES + "s 1\nf 1 2 3\ns 2\ns on\nf 1 4 2\n"
        primitive, = G.parse_obj(source, is_path=False, up_axis="y")
        self.assertEqual(len(primitive["positions"]), 4)
        self.assertAlmostEqual(primitive["normals"][0][1], 1 / math.sqrt(2))
        self.assertAlmostEqual(primitive["normals"][0][2], 1 / math.sqrt(2))

    def test_explicit_normals_are_preserved(self):
        faces = "f 1//1 2//1 3//1\nf 1//1 4//1 2//1\n"
        primitive, = parse("s off\ns on\n", faces, "vn 1 0 0\n")
        self.assertEqual(len(primitive["positions"]), 4)
        self.assertEqual(primitive["normals"], [(1.0, 0.0, 0.0)] * 4)


if __name__ == "__main__":
    unittest.main()
