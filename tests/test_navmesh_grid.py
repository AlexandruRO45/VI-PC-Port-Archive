"""Synthetic tests for tools/navmesh_grid.py. No game files are read or needed."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import navmesh_decode as nd  # noqa: E402
import navmesh_grid as ng  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "navmesh_grid.py")
BASE = nd.BASE


def build_grid(items, *, extent=128, sector=(256, 256), origin=(0.0, 0.0, 0.0)):
    """Build a synthetic Band-3 ``grid`` resource from ``[(key, word), ...]``."""
    entries = len(items)
    groups = 1
    while 16 * groups <= entries:
        groups *= 2
    ht_off = 0x60
    pay_off = ht_off + groups * nd.GRID_GROUP
    total = pay_off + 4 * entries
    buf = bytearray(total)
    struct.pack_into("<Q", buf, 0x08, BASE | total)
    struct.pack_into("<II", buf, 0x10, groups, entries)
    struct.pack_into("<QQ", buf, 0x18, BASE | ht_off, BASE | pay_off)
    struct.pack_into("<3I", buf, 0x28, extent, extent, extent)
    struct.pack_into("<I", buf, 0x34, sector[0] | (sector[1] << 9))
    struct.pack_into("<3f", buf, 0x40, *origin)
    struct.pack_into("<I", buf, 0x50, entries)

    # Fill all SwissTable control bytes with 0x80 (empty), then populate sequentially.
    for g in range(groups):
        base = ht_off + g * nd.GRID_GROUP
        buf[base:base + 16] = b"\x80" * 16

    for idx, (key, word) in enumerate(items):
        g, slot = divmod(idx, 16)
        base = ht_off + g * nd.GRID_GROUP
        buf[base + slot] = 0x01
        struct.pack_into("<QQ", buf, base + 16 + slot * 16, key, idx)
        struct.pack_into("<I", buf, pay_off + 4 * idx, word)
    return bytes(buf)


def make_rpf(path, streams, data_start=0x40):
    buf = bytearray(data_start)
    buf[0:4] = b"9FPR"
    struct.pack_into("<I", buf, 4, data_start)
    for raw in streams:
        buf += zlib.compress(raw)
    with open(path, "wb") as fh:
        fh.write(buf)


class TestNavmeshGrid(unittest.TestCase):
    def test_key_coords_inverted_axes(self):
        # Octant 7 (bits 2,1,0 = 1,1,1) maps to inverted (0, 0, 0) at depth 1.
        self.assertEqual(ng.grid_key_coords(1), (0, 0, 0, 0))
        self.assertEqual(ng.grid_key_coords((1 << 3) | 7), (1, 0, 0, 0))
        # Octant 0 (bits 2,1,0 = 0,0,0) maps to inverted (1, 1, 1) at depth 1.
        self.assertEqual(ng.grid_key_coords((1 << 3) | 0), (1, 1, 1, 1))
        # Bit 1 = ~X, bit 0 = ~Y, bit 2 = ~Z: octant 0b101 (X_bit=0->1, Y_bit=1->0, Z_bit=1->0).
        self.assertEqual(ng.grid_key_coords((1 << 3) | 5), (1, 1, 0, 0))
        lo, hi = ng.grid_cell_bounds((1 << 3) | 7, 256, (-8192.0, 4096.0, -128.0))
        self.assertEqual(lo, (-8192.0, 4096.0, -128.0))
        self.assertEqual(hi, (-8064.0, 4224.0, 0.0))

    def test_valid_leaves_and_histograms(self):
        # Depth 0..5 chain for extent=128 (depth 5 -> 4 m cell, depth 4 -> 8 m cell).
        # At depth 1, octant 3 (Z_bit=0 -> ~Z=1 -> z in [0, 64] when origin_z=-64 -> wait: z0=0).
        items = [
            (1, ng.GRID_INNER),
            (0b1_011, ng.GRID_INNER),           # d=1: iz=1 -> z in [0, 64] for origin=(0,0,-64), ext=128
            (0b1_011_000, ng.GRID_INNER),       # d=2
            (0b1_011_000_000, ng.GRID_INNER),   # d=3
            (0b1_011_000_000_000, 0x00007903),  # d=4 (8 m): 0x03 b8=1, nib=15
            (0b1_011_000_000_001, ng.GRID_INNER),  # d=4
            (0b1_011_000_000_001_000, 0x00005003),  # d=5 (4 m): 0x03 b8=0, nib=10
            (0b1_011_000_000_001_001, 0x00040F01),  # d=5: 0x01 b8_11=0xf (z0 >= 0), b17_18=2
            (0b1_011_000_000_001_010, ng.GRID_INNER),  # d=5 leaf with ff00
        ]
        blob = build_grid(items, extent=128, origin=(0.0, 0.0, -64.0))
        stats = ng.new_stats()
        chk = ng.verify_grid_resource(blob, stats)
        self.assertTrue(all(chk.values()), chk)
        self.assertEqual(stats["leaves_total"], 4)
        self.assertEqual(dict(stats["leaf_01_b8_11"]), {"0xf": 1})
        self.assertEqual(dict(stats["leaf_01_b17_18"]), {"2": 1})
        self.assertEqual(dict(stats["leaf_03_cell_size"]), {"4": 1, "8": 1})
        self.assertEqual(dict(stats["leaf_03_b8"]), {"0": 1, "1": 1})
        self.assertEqual(dict(stats["leaf_03_b11_14"]), {"10": 1, "15": 1})

    def test_rejects_bad_masks_cell_size_and_bit11_z0(self):
        # Bad 0x01 mask (bit 16 set)
        bad_01 = build_grid([(1, ng.GRID_INNER), (8, 0x00010001)])
        self.assertFalse(ng.verify_grid_resource(bad_01, ng.new_stats())["leaf_bitmasks_ok"])

        # Bad 0x03 mask (bit 9 set) at valid 4 m depth 5
        k_d5 = 1 << 15
        chain = [(1 << (3 * d), ng.GRID_INNER) for d in range(5)] + [(k_d5, 0x00000203)]
        bad_03 = build_grid(chain, extent=128)
        self.assertFalse(ng.verify_grid_resource(bad_03, ng.new_stats())["leaf_bitmasks_ok"])

        # Bad 0x03 cell size (depth 1 -> 64 m)
        bad_sz = build_grid([(1, ng.GRID_INNER), (8, 0x00000103)], extent=128)
        self.assertFalse(ng.verify_grid_resource(bad_sz, ng.new_stats())["leaf_03_cell_size"])

        # Bad 0x01 bit 11 with z0 < 0.0 (octant 7 -> iz=0 -> z0 = -64.0)
        bad_z = build_grid([(1, ng.GRID_INNER), (0b1_111, 0x00000F01)], extent=128, origin=(0.0, 0.0, -64.0))
        self.assertFalse(ng.verify_grid_resource(bad_z, ng.new_stats())["leaf_01_bit11_z0"])

    def test_cli_round_trip(self):
        items = [(1, ng.GRID_INNER), (0b1_011, 0x00020F01)]
        blob = build_grid(items, extent=128, origin=(0.0, 0.0, -64.0))
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "test.rpf")
            out = os.path.join(tmp, "out.json")
            make_rpf(rpf, [blob])
            res = subprocess.run([sys.executable, "-I", TOOL, rpf, "--out", out, "--quiet"], check=False)
            self.assertEqual(res.returncode, 0)
            with open(out, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["grids"], 1)
            self.assertEqual(data["checks_failed"], {})
            self.assertEqual(data["leaf_01_b8_11"], {"0xf": 1})


if __name__ == "__main__":
    unittest.main()
