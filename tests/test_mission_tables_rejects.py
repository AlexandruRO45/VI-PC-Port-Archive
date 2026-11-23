"""mission_tables.py must refuse what is not a record, element, chunk or pool, at the exact limits, without game data.

    python -I tests/test_mission_tables_rejects.py

Every fixture is synthetic. Most limits are tested from both sides: the last accepted value and the first refused one.
Numbers are written out, not read from the tool's constants, so a changed constant fails here.
"""
import math
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mission_tables as M  # noqa: E402

KEY = 0x12345678


def record(key=KEY, col_b=1, kind=3, offset=0x780, band=0, count=2, **words):
    """One 32-byte record; `w<index>` replaces that dword."""
    out = [key, 0, col_b, 0, kind | offset << 16, band, count << 16 | count, 0]
    for name, value in words.items():
        out[int(name[1:])] = value
    return struct.pack("<8I", *out)


def element(floats=None, **words):
    """One 64-byte element; `floats` replaces f32 dwords by index, `w<index>` replaces u32 dwords."""
    out = bytearray(struct.pack("<4I8f4I", KEY, 0, 0, 0, 1.5, -2.5, 3.5, 1.5, 0.25, 0.5, -0.75, 0.25, KEY, 7, 0, 0))
    for index, value in (floats or {}).items():
        struct.pack_into("<f", out, 4 * index, value)
    for name, value in words.items():
        struct.pack_into("<I", out, 4 * int(name[1:]), value)
    return bytes(out)


def row(offset=0, count=2, kind=1, band=0):
    """A decoded record whose block pointer is the byte `offset` (below 4096)."""
    return {"kind": kind, "band": band, "poolOffset": offset << 4, "count": count}


class RecordTests(unittest.TestCase):
    def test_accepts_the_layout_and_its_limits(self):
        self.assertEqual(M.decode_record(record(col_b=9, band=5), 0),
                         {"offset": 0, "key": KEY, "colB": 9, "kind": 3, "poolOffset": 0x780, "band": 5, "count": 2})
        self.assertEqual(M.decode_record(b"\0" * 4 + record(), 4)["offset"], 4)
        for label, data in (("lowest key", record(key=0x10000)), ("highest count", record(count=4096)),
                            ("no entities", record(count=0))):
            with self.subTest(label):
                self.assertIsNotNone(M.decode_record(data, 0))

    def test_rejects(self):
        for label, data, at in (
            ("key below a hash", record(key=0xFFFF), 0),
            ("dword +4 set", record(w1=1), 0),
            ("dword +12 set", record(w3=1), 0),
            ("dword +28 set", record(w7=1), 0),
            ("counts differ", record(w6=2 << 16 | 3), 0),
            ("count past the limit", record(count=4097), 0),
            ("before the blob", record() * 2, -1),
            ("one byte short", record()[:31], 0),
            ("runs past the blob", record(), 4),
            ("empty blob", b"", 0),
        ):
            with self.subTest(label):
                self.assertIsNone(M.decode_record(data, at))


class ElementTests(unittest.TestCase):
    def test_accepts_the_layout_and_its_limits(self):
        self.assertEqual(M.decode_element(b"\0" * 4 + element(w12=5, w13=6, w14=8), 4),
                         {"offset": 4, "archetype": KEY, "pos": [1.5, -2.5, 3.5], "rot": [0.25, 0.5, -0.75],
                          "hash": 5, "link": 6, "extra": 8})
        self.assertIsNotNone(M.decode_element(element(w15=1), 0), "the last dword is not required to be zero")
        for limit in (1e7, -1e7):
            for same in ((4, 7), (5,), (6,), (8, 11), (9,), (10,)):
                with self.subTest(limit=limit, dwords=same):
                    self.assertIsNotNone(M.decode_element(element(dict.fromkeys(same, limit)), 0))

    def test_rejects_a_component_that_is_no_world_coordinate(self):
        for value in (10000001.0, -10000001.0, math.inf, -math.inf, math.nan):
            for same in ((4, 7), (5,), (6,), (8, 11), (9,), (10,)):     # a first component is stored twice
                with self.subTest(value=value, dwords=same):
                    self.assertIsNone(M.decode_element(element(dict.fromkeys(same, value)), 0))

    def test_rejects(self):
        for label, data, at in (
            ("no archetype", element(w0=0), 0),
            ("dword +4 set", element(w1=1), 0),
            ("dword +8 set", element(w2=1), 0),
            ("dword +12 set", element(w3=1), 0),
            ("position x not repeated", element({7: 1.25}), 0),
            ("rotation x not repeated", element({11: 0.5}), 0),
            ("x repeated with the other zero", element({4: 0.0, 7: -0.0}), 0),
            ("before the blob", element() * 2, -1),
            ("one byte short", element()[:63], 0),
            ("runs past the blob", element(), 4),
        ):
            with self.subTest(label):
                self.assertIsNone(M.decode_element(data, at))

    def test_scan_stops_at_the_end_and_steps_by_four(self):
        blob = b"\0" * 4 + element()
        self.assertEqual(list(M.scan_elements(blob, 68)), [4])
        self.assertEqual(M.scan_elements(blob, 67), {})
        self.assertEqual(M.scan_elements(element(), 63), {})
        self.assertEqual(M.scan_elements(element(), 0), {})
        self.assertEqual(M.scan_elements(b"\0" * 2 + element(), 66), {})


class BlockPointerTests(unittest.TestCase):
    BLOCKS = [(0x11, 0x100, 0x40), (0x22, 0x140, 0x80)]

    def test_block_numbers_start_at_one(self):
        for kind, want in ((0, (None, None)), (1, (0x100, 0x40)), (2, (0x140, 0x80)), (3, (None, None))):
            with self.subTest(kind=kind):
                self.assertEqual(M.block_at(self.BLOCKS, row(kind=kind)), want)
        for blocks in ([], None):
            self.assertEqual(M.block_at(blocks, row()), (None, None))

    def test_band_holds_the_bits_above_the_offset(self):
        self.assertEqual(M.pool_byte_offset({"band": 2, "poolOffset": 0xFFF0}), 0x2FFF)
        self.assertEqual(M.pool_byte_offset({"band": 0, "poolOffset": 0x000F}), 0)


class ChunkStrideTests(unittest.TestCase):
    def stride(self, *rows, size=0x1000):
        return M.chunk_stride(list(rows), [(0x11, 0x100, size), (0x22, 0x100, size)])

    def test_accepts_slices_that_tile_the_block(self):
        self.assertEqual(self.stride(row(0), row(128)), 64)
        self.assertEqual(self.stride(row(0), row(136)), 68)
        self.assertEqual(self.stride(row(0, 1), row(112, 3), row(448, 1)), 112)
        self.assertEqual(self.stride(row(0), row(128), size=256), 64, "the last slice may end at the block's end")
        self.assertEqual(self.stride(row(0), row(128, 3), size=320), 64)
        self.assertEqual(self.stride(row(0), row(128, 1), size=192), 64)

    def test_rejects(self):
        for label, rows, size in (
            ("no records", (), 0x1000),
            ("one slice proves no stride", (row(0),), 0x1000),
            ("two kinds", (row(0), row(128, kind=2)), 0x1000),
            ("block 0", (row(0, kind=0), row(128, kind=0)), 0x1000),
            ("block past the table", (row(0, kind=3), row(128, kind=3)), 0x1000),
            ("descending slices", (row(128), row(0)), 0x1000),
            ("the same slice twice", (row(0), row(0)), 0x1000),
            ("a slice without slots", (row(0, 0), row(128)), 0x1000),
            ("a gap the count does not divide", (row(0, 3), row(128)), 0x1000),
            ("slices that disagree", (row(0), row(128), row(264)), 0x1000),
            ("stride below an element", (row(0), row(120)), 0x1000),
            ("stride not a multiple of 4", (row(0), row(132)), 0x1000),
            ("last slice one byte past the block", (row(0), row(128)), 255),
            ("a longer last slice one byte past the block", (row(0), row(128, 3)), 319),
        ):
            with self.subTest(label):
                self.assertIsNone(self.stride(*rows, size=size))


class FindChunksTests(unittest.TestCase):
    def chunks(self, blob, end=None, elements=()):
        found = M.find_chunks(blob, len(blob) if end is None else end, dict.fromkeys(elements))
        return [(chunk["offset"], [at for at, _record in chunk["records"]]) for chunk in found]

    def test_a_run_is_two_or_more_records_of_one_kind(self):
        self.assertEqual(self.chunks(record() * 2), [(0, [0, 32])])
        self.assertEqual(self.chunks(b"\0" * 4 + record() * 3), [(4, [4, 36, 68])])
        self.assertEqual(self.chunks(record(kind=1) * 2 + record(kind=2) * 2), [(0, [0, 32]), (64, [64, 96])])
        self.assertEqual(self.chunks(record(kind=1) + record(kind=2) * 2), [(32, [32, 64])])
        self.assertEqual(self.chunks(record() * 2 + b"\0" * 8 + record() * 2), [(0, [0, 32]), (72, [72, 104])])

    def test_no_run(self):
        self.assertEqual(self.chunks(record()), [])
        self.assertEqual(self.chunks(record(kind=1) + record(kind=2)), [])
        self.assertEqual(self.chunks(b"\0" * 2 + record() * 2), [], "records start on a multiple of 4")
        self.assertEqual(self.chunks(record() * 2, end=63), [], "the second record crosses the end")
        self.assertEqual(self.chunks(b""), [])

    def test_a_lone_record_must_name_an_element(self):
        accepted = record(key=0x1000000, col_b=4, offset=0x780)
        self.assertEqual(self.chunks(accepted, elements=[0x780]), [(0, [0])])
        for label, data, elements in (
            ("no element there", accepted, [0x784]),
            ("key too small for a lone record", record(key=0xFFFFFF, col_b=4), [0x780]),
            ("colB too large for a lone record", record(key=0x1000000, col_b=5), [0x780]),
        ):
            with self.subTest(label):
                self.assertEqual(self.chunks(data, elements=elements), [])


class RecordBlocksTests(unittest.TestCase):
    def test_only_a_whole_table_of_one_kind_counts(self):
        two, mixed = record() * 2, record(kind=1) + record(kind=2)
        for label, data, size, want in (
            ("two records", two, 64, [0]),
            ("three records", record() * 3, 96, [0]),
            ("one record", two, 32, []),
            ("part of a record", record() * 3, 80, []),
            ("first is no record", bytes(32) + record(), 64, []),
            ("last is no record", two + bytes(32), 96, []),
            ("two kinds", mixed, 64, []),
        ):
            with self.subTest(label):
                self.assertEqual(M.record_blocks(data, [(0x11, 0, size)]), want)

    def test_indices_are_positions_in_the_block_table(self):
        blob = bytes(96) + record() * 4
        blocks = [(0x11, 0, 32), (0x22, 32, 64), (0x33, 96, 64), (0x44, 160, 64)]
        self.assertEqual(M.record_blocks(blob, blocks), [2, 3])


class LinkedPoolTests(unittest.TestCase):
    # records, 63 bytes of metadata, more records, then the pool
    BLOCKS = [(0, 0, 64), (0, 64, 63), (0, 128, 64), (0, 192, 64)]
    TABLES = [0, 2]

    def pool(self, at, elements=(192,), blocks=None, tables=None):
        return M.linked_pool(b"", self.BLOCKS if blocks is None else blocks,
                             self.TABLES if tables is None else tables, at, dict.fromkeys(elements))

    def test_steps_over_metadata_and_other_tables(self):
        for at in (0, 63, 128, 191):
            with self.subTest(at=at):
                self.assertEqual(self.pool(at), 192)

    def test_no_pool(self):
        for label, at, options in (
            ("no block table", 0, {"blocks": []}),
            ("chunk in a block that is no table", 64, {}),
            ("chunk between blocks", 127, {}),
            ("chunk past every block", 256, {}),
            ("the pool starts with no element", 0, {"elements": ()}),
            ("a 64-byte block is not metadata", 0, {"blocks": [(0, 0, 64), (0, 64, 64), (0, 128, 64), (0, 192, 64)]}),
            ("the pool comes before the table", 128,
             {"blocks": [(0, 0, 64), (0, 64, 128)], "tables": [1], "elements": (0,)}),
        ):
            with self.subTest(label):
                self.assertIsNone(self.pool(at, **options))
        self.assertIsNone(M.linked_pool(b"", None, [], 0, {}))


class ResolveTests(unittest.TestCase):
    """Absolute addresses are 0x400+, the pool is at 0x2000, block 1 is at 0x3000 with room for two slots."""

    BLOCKS = [(0x11, 0x3000, 128)]

    def resolve(self, at, record=None, pool=0x2000, blocks=BLOCKS, stride=None):
        elements = {address: f"element {address:#x}" for address in at}
        found, mode, skipped = M.resolve(record or row(0), elements, pool, blocks, stride)
        self.assertEqual([elements[address] for address, _element in found], [e for _address, e in found])
        return [address for address, _element in found], mode, skipped

    def test_nothing_to_resolve(self):
        self.assertEqual(self.resolve([0, 0x2000, 0x3000], row(0, count=0)), ([], "empty", 0))
        self.assertEqual(self.resolve([]), ([], "unresolved", 2))
        self.assertEqual(self.resolve([0x2000, 0x3000], pool=None, blocks=None), ([], "unresolved", 2))

    def test_each_reading_alone(self):
        at = row(0x40)                                          # poolOffset 0x400
        self.assertEqual(self.resolve([0x400, 0x440], at), ([0x400, 0x440], "absolute", 0))
        self.assertEqual(self.resolve([0x2040, 0x2080], at), ([0x2040, 0x2080], "pool", 0))
        self.assertEqual(self.resolve([0x3000]), ([0x3000], "block", 1))
        self.assertEqual(self.resolve([0x3040]), ([0x3040], "block", 1))
        self.assertEqual(self.resolve([0x3040, 0x3080], at, blocks=[(0x11, 0x3000, 192)]),
                         ([0x3040, 0x3080], "block", 0))

    def test_a_tie_keeps_the_earlier_reading_and_more_slots_win(self):
        self.assertEqual(self.resolve([0, 0x2000, 0x3000])[1], "absolute")
        self.assertEqual(self.resolve([0x2000, 0x3000])[1], "pool")
        self.assertEqual(self.resolve([0, 0x2000, 0x2040, 0x3000]), ([0x2000, 0x2040], "pool", 0))
        self.assertEqual(self.resolve([0, 0x2000, 0x3000, 0x3040]), ([0x3000, 0x3040], "block", 0))

    def test_the_block_must_hold_every_slot(self):
        self.assertEqual(self.resolve([0x3000])[1], "block")
        for label, options in (("one byte short", {"blocks": [(0x11, 0x3000, 127)]}),
                               ("third slot", {"record": row(0, count=3)}),
                               ("offset moves the last slot out", {"record": row(1)}),
                               ("wider slots", {"stride": 68}),
                               ("block number past the table", {"record": row(0, kind=2)})):
            with self.subTest(label):
                self.assertEqual(self.resolve([0x3000, 0x3001], **options)[1], "unresolved")

    def test_the_stride_moves_only_block_slots(self):
        blocks = [(0x11, 0x3000, 224)]
        self.assertEqual(self.resolve([0x3000, 0x3070], blocks=blocks, stride=112), ([0x3000, 0x3070], "block", 0))
        self.assertEqual(self.resolve([0, 0x40], blocks=blocks, stride=112), ([0, 0x40], "absolute", 0))
        self.assertEqual(self.resolve([0x2000, 0x2040], blocks=blocks, stride=112), ([0x2000, 0x2040], "pool", 0))


if __name__ == "__main__":
    unittest.main()
