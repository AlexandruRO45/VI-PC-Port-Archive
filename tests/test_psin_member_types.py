"""Tests for the 0x5a f32 grid and the 0x76 / 0x77 hash sets of tools/psin_data.py (no game data).

    python -I tests/test_psin_member_types.py
"""
import collections
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psin_data as D  # noqa: E402
from test_psin_data import make_container, sample  # noqa: E402
from test_psin_index import joaat  # noqa: E402


class TestMemberTypes(unittest.TestCase):
    """The synthetic RootThing container carries one of each; see test_psin_data.sample()."""

    def setUp(self):
        self.stats = collections.Counter()
        self.tree = D.Walker(D.Container(sample()), self.stats, keep=True).run()

    def test_hash_set_of_u32_hashes(self):
        self.assertEqual(self.tree[f"{joaat('hashes'):08x}"], ["deadbeef", "12345678"])
        self.assertEqual((self.stats["set 76 slots"], self.stats["set 76 entries"]), (4, 2))
        self.assertEqual(self.stats["set 76 resolved"], 1)

    def test_hash_set_of_buckets(self):
        self.assertEqual(self.tree[f"{joaat('buckets'):08x}"],
                         [{f"{joaat('h'):08x}": f"{joaat('first'):016x}"},
                          {f"{joaat('h'):08x}": f"{joaat('second'):016x}"}])
        self.assertEqual((self.stats["set 77 slots"], self.stats["set 77 entries"]), (2, 3))

    def test_empty_hash_set(self):
        self.assertEqual(self.tree[f"{joaat('odd'):08x}"], [])
        self.assertEqual(self.stats["set 77 empty"], 1)

    def test_f32_grid(self):
        self.assertEqual(self.tree[f"{joaat('grid'):08x}"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(self.stats["grid 5a count == rows * cols"], 1)
        self.assertEqual(self.stats["grid 5a block is f32"], 1)

    def test_hash_set_counts_only_follows_bucket_pointers(self):
        r, b, i = (joaat(n) for n in ("Root", "Bucket", "Item"))
        schema = [
            (r, 1, 16, 3, [(1, 0, 0x77, 0, 0, 0)]),
            (b, 2, 8, 3, [(2, 0, 0x07, 0, 0, 0)]),
            (i, 3, 4, 2, [(3, 0, 0x15, 0, 0, 0)]),
        ]
        blocks = [
            (r, struct.pack("<IIQ", 1, 1, 2)),
            (b, struct.pack("<Q", 3)),
            (i, struct.pack("<I", 42)),
        ]
        container = D.Container(make_container(blocks, schema))
        counts_only = collections.Counter()
        with_values = collections.Counter()
        self.assertIsNone(D.Walker(container, counts_only, keep=False).run())
        tree = D.Walker(container, with_values, keep=True).run()
        self.assertEqual(tree[f"{1:08x}"], [{f"{2:08x}": {f"{3:08x}": 42}}])
        for key in ("blocks reached", "bytes reached", "structs"):
            self.assertEqual((key, counts_only[key]), (key, with_values[key]))
        self.assertEqual((counts_only["blocks reached"], counts_only["bytes reached"], counts_only["structs"]),
                         (3, 28, 3))


if __name__ == "__main__":
    unittest.main()
