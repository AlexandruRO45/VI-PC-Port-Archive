"""A --entities/--placements file that is not UTF-8 text gets an error naming the file, not a codec traceback.

    python -I tests/test_world_build_encoding.py

The rows are written here; no game file is used.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import world_build  # noqa: E402

ROW = {"archetype": 123, "position": [0, 0, 0], "rotation": [0, 0, 0, 1], "lod_level": 0}


class PlacementEncodingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.utf16 = os.path.join(self.tmp.name, "utf16.jsonl")
        with open(self.utf16, "wb") as f:   # what PowerShell's Out-File writes by default on Windows
            f.write((json.dumps(ROW) + "\r\n").encode("utf-16"))
        self.bom = os.path.join(self.tmp.name, "bom.jsonl")
        with open(self.bom, "wb") as f:     # out-file -encoding utf8 on PowerShell 5.1
            f.write(b"\xef\xbb\xbf" + (json.dumps(ROW) + "\n").encode("utf-8"))

    def test_load_placements_names_the_file(self):
        """A UnicodeDecodeError used to escape as a traceback, with no word about which file it was."""
        with self.assertRaises(ValueError) as caught:
            world_build.load_placements(self.utf16)
        self.assertIn(self.utf16, str(caught.exception))
        self.assertIn("not UTF-8", str(caught.exception))

    def test_iter_placements_names_the_file(self):
        with self.assertRaises(ValueError) as caught:
            list(world_build.iter_placements(self.utf16, world_build._placement_counts()))
        self.assertIn(self.utf16, str(caught.exception))

    def test_utf8_bom_still_loads(self):
        rows, counts = world_build.load_placements(self.bom)
        self.assertEqual([row["_arch"] for row in rows], [123])
        self.assertEqual(counts["malformed"], 0)
        self.assertEqual(counts["selected"], 1)


if __name__ == "__main__":
    unittest.main()
