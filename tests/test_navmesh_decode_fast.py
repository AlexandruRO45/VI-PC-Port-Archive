"""Exact-parity tests for the fast paths in tools/navmesh_decode.py. No game files are read or needed.

The grid walkers and ``inflate_at`` were rewritten to build fewer Python objects and to copy less of the mapped
archive. Each ``ref_*`` function below is the previous implementation, kept verbatim as the oracle; every case runs
both and compares the whole result, so a change in what is accepted, counted or returned fails here.
"""
import mmap
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib
from collections import Counter
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import navmesh_decode as nd  # noqa: E402
from test_navmesh_decode import build_grid  # noqa: E402


# ------------------------------------------------------------------------------------- previous implementations

def ref_inflate_at(m, start, size, cap=nd.MAX_INFLATE):
    d = zlib.decompressobj()
    parts, total, pos, pending = [], 0, start, b""
    try:
        while not d.eof:
            if not pending:
                if pos >= size:
                    return None
                pending = m[pos:min(pos + nd.CHUNK, size)]
                pos += len(pending)
            out = d.decompress(pending, cap + 1 - total)
            pending = d.unconsumed_tail
            total += len(out)
            if total > cap:
                return None
            parts.append(out)
    except zlib.error:
        return None
    return b"".join(parts), pos - start - len(d.unused_data) - len(pending)


def ref_grid_values(b, groups, entries):
    seen = bytearray(entries)
    for g in range(groups):
        base = 0x60 + nd.GRID_GROUP * g
        for k, c in enumerate(b[base:base + 16]):
            if c < 0x80:
                value = nd._u64(b, base + 16 + 16 * k + 8)
                if value is None or value >= entries or seen[value]:
                    return False
                seen[value] = 1
    return all(seen)


def ref_grid_slots(b, groups):
    keys = {}
    for g in range(groups):
        base = 0x60 + nd.GRID_GROUP * g
        for k, c in enumerate(b[base:base + 16]):
            if c < 0x80:
                keys[nd._u64(b, base + 16 + 16 * k)] = nd._u64(b, base + 16 + 16 * k + 8)
    return keys


def ref_grid_octree(keys):
    if 1 not in keys:
        return False
    for key in keys:
        if key == 0 or (key.bit_length() - 1) % 3 or (key != 1 and key >> 3 not in keys):
            return False
    return True


def ref_grid_payloads(b, payload, keys, stats):
    inner = {key >> 3 for key in keys if key != 1}
    words = {key: nd._u32(b, payload + 4 * value) for key, value in keys.items()}
    if any(words[key] != nd.GRID_INNER for key in inner):
        return False
    stats["inner_nodes"] += len(inner)
    for key, word in words.items():
        stats["key_depth"][(key.bit_length() - 1) // 3] += 1
        if key in inner:
            continue
        kind = "ff00" if word == nd.GRID_INNER else nd.GRID_LEAF_KIND.get(word & 0xFF, "other")
        stats["leaf_payload"][kind] += 1
    return True


# ------------------------------------------------------------------------------------------------------- helpers

def octree(depth):
    keys, level = [1], [1]
    for _ in range(depth):
        level = [(k << 3) | o for k in level for o in range(8)]
        keys += level
    return keys


def payload_stats():
    return {"inner_nodes": 0, "key_depth": Counter(), "leaf_payload": Counter()}


def grid_with_words(keys, groups, words=None, **kw):
    """One-stream grid resource (the bytes of the single stream) and its payload offset."""
    entries = len(keys)
    (raw,) = build_grid(groups=groups, entries=entries, keys=keys, words=words, **kw)
    payload = nd.offset_of(struct.unpack_from("<Q", raw, 0x20)[0])
    return raw, payload


class GridParityTest(unittest.TestCase):
    def check_grid(self, raw, groups, entries, payload, expect_values=None):
        got = nd._grid_values(raw, groups, entries)
        self.assertEqual(got, ref_grid_values(raw, groups, entries))
        if expect_values is not None:
            self.assertEqual(got, expect_values)
        if not got:
            return None
        keys = nd._grid_slots(raw, groups)
        self.assertEqual(keys, ref_grid_slots(raw, groups))
        self.assertEqual(list(keys), list(ref_grid_slots(raw, groups)))        # same insertion order
        self.assertEqual(nd._grid_octree(keys), ref_grid_octree(keys))
        if len(keys) == entries and ref_grid_octree(keys):
            a, b = payload_stats(), payload_stats()
            self.assertEqual(nd._grid_payloads(raw, payload, keys, a), ref_grid_payloads(raw, payload, keys, b))
            self.assertEqual(a, b)
        return keys

    def test_octrees_of_every_depth(self):
        for depth in range(5):
            keys = octree(depth)
            groups = 1
            while groups * 16 < len(keys):
                groups *= 2
            raw, payload = grid_with_words(keys, groups)
            with self.subTest(depth=depth):
                got = self.check_grid(raw, groups, len(keys), payload, expect_values=True)
                self.assertEqual(len(got), len(keys))
                self.assertTrue(nd._grid_octree(got))

    def test_payload_words_and_depth_counts_match(self):
        keys = octree(3)
        parents = {k >> 3 for k in keys if k != 1}
        kinds = (0x00020001, 0x00020003, 0x00020002, 0xFFFFFF00, 0x0000FF00)
        words = [0x0000FF00 if k in parents else kinds[i % len(kinds)] for i, k in enumerate(keys)]
        raw, payload = grid_with_words(keys, 64, words)
        keys_by_slot = nd._grid_slots(raw, 64)
        got, ref = payload_stats(), payload_stats()
        self.assertTrue(nd._grid_payloads(raw, payload, keys_by_slot, got))
        self.assertTrue(ref_grid_payloads(raw, payload, keys_by_slot, ref))
        self.assertEqual(got, ref)
        self.assertEqual(sum(got["key_depth"].values()), len(keys))
        self.assertEqual(set(got["leaf_payload"]), {"low_byte_01", "low_byte_03", "other", "ff00"})

    def test_inner_node_without_the_inner_word_is_refused_with_untouched_stats(self):
        keys = octree(2)
        parents = {k >> 3 for k in keys if k != 1}
        words = [0x0000FF00 if k in parents else 0x00020001 for k in keys]
        words[keys.index(1)] = 0x00020001
        raw, payload = grid_with_words(keys, 8, words)
        slots = nd._grid_slots(raw, 8)
        a, b = payload_stats(), payload_stats()
        self.assertFalse(nd._grid_payloads(raw, payload, slots, a))
        self.assertFalse(ref_grid_payloads(raw, payload, slots, b))
        self.assertEqual(a, b)
        self.assertEqual(a, payload_stats())

    def test_full_groups_and_partly_empty_groups(self):
        for entries in (16, 17, 32, 40, 73):
            keys = (octree(3))[:entries]
            groups = 8
            raw, payload = grid_with_words(keys, groups)
            with self.subTest(entries=entries):
                self.check_grid(raw, groups, entries, payload)

    def test_duplicate_and_out_of_range_values_are_refused(self):
        keys = octree(2)
        for kw in ({"dup_value": True}, {"bad_ctrl": True}):
            raw, payload = grid_with_words(keys, 8, **kw)
            with self.subTest(**kw):
                self.check_grid(raw, 8, len(keys), payload)
        raw, payload = grid_with_words(keys, 8)
        for entries in (len(keys) - 1, len(keys) + 1):     # a value past the count / a value never used
            with self.subTest(entries=entries):
                self.assertFalse(nd._grid_values(raw, 8, entries))
                self.assertEqual(nd._grid_values(raw, 8, entries), ref_grid_values(raw, 8, entries))

    def test_values_are_read_only_from_occupied_slots(self):
        keys = octree(2)
        raw, payload = grid_with_words(keys, 8)
        buf = bytearray(raw)
        # An empty slot's key and value words hold garbage that must not be looked at.
        for g in range(8):
            base = 0x60 + nd.GRID_GROUP * g
            for k, c in enumerate(buf[base:base + 16]):
                if c >= 0x80:
                    struct.pack_into("<QQ", buf, base + 16 + 16 * k, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
        raw = bytes(buf)
        self.assertTrue(nd._grid_values(raw, 8, len(keys)))
        self.assertEqual(nd._grid_slots(raw, 8), ref_grid_slots(raw, 8))
        self.assertEqual(list(nd._grid_slots(raw, 8)), list(ref_grid_slots(raw, 8)))

    def test_two_slots_with_one_key_keep_the_later_value(self):
        raw, _ = grid_with_words([1, 8, 9], 8)
        buf = bytearray(raw)
        base = 0x60
        struct.pack_into("<Q", buf, base + 16 + 16 * 2, 8)        # slot 2 now repeats slot 1's key
        raw = bytes(buf)
        self.assertEqual(nd._grid_slots(raw, 8), ref_grid_slots(raw, 8))
        self.assertEqual(nd._grid_slots(raw, 8)[8], 2)

    def test_octree_key_shapes(self):
        cases = {
            "root only": {1: 0},
            "no root": {8: 0, 9: 1},
            "zero key": {1: 0, 0: 1},
            "orphan child": {1: 0, 64: 1},
            "bad bit length (2 bits)": {1: 0, 2: 1},
            "bad bit length (3 bits)": {1: 0, 5: 1},
            "bad bit length (5 bits)": {1: 0, 16: 1},
            "child of a child": {1: 0, 8: 1, 64: 2},
            "sibling without parent": {1: 0, 8: 1, 65: 2, 512: 3},
            "two levels full": dict.fromkeys(octree(2), 0),
            "wide key": {1: 0, 8: 1, 1 << 62: 2},
            "64-bit keys": {1: 0, (1 << 63) | 0: 1},
            "empty": {},
        }
        for name, keys in cases.items():
            with self.subTest(name):
                self.assertEqual(nd._grid_octree(keys), ref_grid_octree(keys))

    def test_random_octrees_with_one_corrupted_key(self):
        rnd = random.Random(1234)
        for _ in range(200):
            keys = {1: 0}
            frontier = [1]
            for _ in range(rnd.randrange(1, 4)):
                nxt = []
                for k in frontier:
                    for o in rnd.sample(range(8), rnd.randrange(0, 8)):
                        keys[(k << 3) | o] = len(keys)
                        nxt.append((k << 3) | o)
                frontier = nxt
            if rnd.random() < 0.7 and len(keys) > 1:
                victim = rnd.choice(list(keys))
                keys[rnd.choice((victim ^ (1 << rnd.randrange(1, victim.bit_length() + 1)), victim + 1, victim >> 1))] = 0
            self.assertEqual(nd._grid_octree(keys), ref_grid_octree(keys), keys)

    def test_payload_table_is_a_native_view_on_little_endian_and_a_tuple_otherwise(self):
        data = bytes(range(32)) + struct.pack("<4I", 1, 0xFFFFFFFF, 0x0000FF00, 7) + b"tail"
        want = (1, 0xFFFFFFFF, 0x0000FF00, 7)
        little = nd._u32_table(data, 32, 4)
        self.assertEqual(tuple(little), want)
        little.release() if isinstance(little, memoryview) else None
        with mock.patch.object(nd.sys, "byteorder", "big"):
            self.assertEqual(tuple(nd._u32_table(data, 32, 4)), want)
        # Payload words reach _grid_payloads through the same helper on a big-endian host.
        keys = octree(2)
        raw, payload = grid_with_words(keys, 8)
        slots = nd._grid_slots(raw, 8)
        got, ref = payload_stats(), payload_stats()
        with mock.patch.object(nd.sys, "byteorder", "big"):
            self.assertTrue(nd._grid_payloads(raw, payload, slots, got))
        self.assertTrue(ref_grid_payloads(raw, payload, slots, ref))
        self.assertEqual(got, ref)

    def test_payload_view_does_not_pin_a_bytearray(self):
        keys = octree(2)
        raw, payload = grid_with_words(keys, 8)
        buf = bytearray(raw)
        slots = nd._grid_slots(buf, 8)
        self.assertTrue(nd._grid_payloads(buf, payload, slots, payload_stats()))
        buf.extend(b"\0")                                         # raises BufferError if a view is still held


class InflateParityTest(unittest.TestCase):
    @staticmethod
    def stream(n, seed=0, noisy=True):
        rnd = random.Random(seed)
        raw = rnd.randbytes(n) if noisy else bytes(n)
        return raw, zlib.compress(raw)

    def both(self, m, start, size, cap=nd.MAX_INFLATE):
        got = nd.inflate_at(m, start, size, cap)
        want = ref_inflate_at(m, start, size, cap)
        self.assertEqual(got, want)
        return got

    def test_streams_around_every_window_boundary(self):
        sizes = [0, 1, 100, nd.FIRST_CHUNK - 40, nd.FIRST_CHUNK, nd.FIRST_CHUNK + 40, 2 * nd.FIRST_CHUNK,
                 4 * nd.FIRST_CHUNK + 5, 100_000, nd.CHUNK - 40, nd.CHUNK, nd.CHUNK + 40, 3 * nd.CHUNK + 17]
        for n in sizes:
            for noisy in (True, False):
                raw, z = self.stream(n, n, noisy)
                for pad in (b"", b"\x78\x9c junk", bytes(70000)):
                    blob = b"HEAD" + z + pad
                    with self.subTest(n=n, noisy=noisy, pad=len(pad)):
                        got = self.both(blob, 4, len(blob))
                        self.assertEqual(got, (raw, len(z)))

    def test_truncated_and_corrupt_streams(self):
        raw, z = self.stream(200_000, 3)
        for cut in (2, 10, nd.FIRST_CHUNK, nd.FIRST_CHUNK + 1, len(z) // 2, len(z) - 1, len(z) - 4):
            with self.subTest(cut=cut):
                self.assertIsNone(self.both(z[:cut], 0, cut))
        bad = bytearray(z)
        bad[len(bad) // 2] ^= 0xFF
        self.both(bytes(bad), 0, len(bad))
        self.assertIsNone(self.both(b"\x00" * 100, 0, 100))
        self.assertIsNone(self.both(z, len(z), len(z)))

    def test_size_bound_inside_and_beyond_the_stream(self):
        raw, z = self.stream(90_000, 5)
        blob = z + bytes(1000)
        for size in (len(z) - 1, len(z), len(z) + 1, len(blob)):
            with self.subTest(size=size):
                self.both(blob, 0, size)

    def test_output_cap(self):
        raw, z = self.stream(300_000, 6, noisy=False)
        for cap in (1, 1000, nd.FIRST_CHUNK, 299_999, 300_000, 300_001):
            with self.subTest(cap=cap):
                got = self.both(z, 0, len(z), cap)
                self.assertEqual(got is None, cap < 300_000)

    def test_reading_from_a_memory_map_matches(self):
        parts = [self.stream(n, n)[1] for n in (50, 70_000, nd.CHUNK + 10, 5000)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.bin")
            offsets = []
            with open(path, "wb") as f:
                for p in parts:
                    offsets.append(f.tell())
                    f.write(p)
                    f.write(b"\xAA" * 33)
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    size = len(m)
                    for off in offsets:
                        self.both(m, off, size)
                    self.assertEqual(
                        [(s, u, len(d)) for s, u, d in nd.walk(m, 0, size)],
                        [(s, u, len(d)) for s, u, d in
                         ((o, ref_inflate_at(m, o, size)[1], ref_inflate_at(m, o, size)[0]) for o in offsets)])

    def test_window_doubles_up_to_the_old_size_and_never_past_it(self):
        seen = []

        class Spy:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                seen.append(key.stop - key.start)
                return self.data[key]

        raw, z = self.stream(5 * nd.CHUNK, 9)
        spy = Spy(z)
        got = nd.inflate_at(spy, 0, len(z))
        self.assertEqual(got, (raw, len(z)))
        self.assertEqual(seen[0], nd.FIRST_CHUNK)
        self.assertTrue(all(b == min(a * 2, nd.CHUNK) for a, b in zip(seen, seen[1:-1])), seen)
        self.assertLessEqual(max(seen), nd.CHUNK)


if __name__ == "__main__":
    unittest.main()
