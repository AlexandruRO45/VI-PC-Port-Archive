"""Parity tests for the blockwise table decoders of tools/rsc8_audio.py.

    python -I tests/test_rsc8_audio_tables.py

`parse_adat_header_bytes`, `decode_grid20_table`, `decode_index_runs`, `decode_marker_records` and the region
early-out of `pair_marker_regions` read their words a block at a time. Each test here compares them with a
per-word reference written the plain way (one struct read per word, one loop iteration per record) on synthetic
bytes: sizes either side of the block edge, tables that break each invariant, and truncated tables. Every byte
is synthesized by the test; no game file, name or audio is used.
"""
import os
import random
import struct
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rsc8_audio  # noqa: E402

START, END = rsc8_audio.MARKER_VALUE_START, rsc8_audio.MARKER_VALUE_END


def ref_grid20(blob):
    out = {"size_ok": False, "magic_ok": None, "entries": None, "position_first": None, "position_last": None,
           "position_increasing": None, "tail_f32": None, "p_multiple_of_32": None}
    raw = bytes(blob)
    if len(raw) < 8 or len(raw) > rsc8_audio.MAX_GRID20_BYTES or (len(raw) - 8) % 20:
        return out
    out["size_ok"] = True
    out["magic_ok"] = struct.unpack_from("<I", raw, 0)[0] == rsc8_audio.GRID20_MAGIC
    n = (len(raw) - 8) // 20
    out["entries"] = n
    out["tail_f32"], = struct.unpack_from("<f", raw, len(raw) - 4)
    prev, p_ok, increasing = None, True, True
    for i in range(n):
        at = 4 + 20 * i
        position, = struct.unpack_from("<I", raw, at)
        p, = struct.unpack_from("<H", raw, at + 8)
        if p % 32:
            p_ok = False
        if prev is None:
            out["position_first"] = position
        elif position <= prev:
            increasing = False
        prev = position
    out["position_last"] = prev
    out["position_increasing"] = increasing
    out["p_multiple_of_32"] = p_ok
    return out


def ref_index_runs(blob, grid_entries=None):
    out = {"size_ok": False, "group_count_matches": None, "marker_ok": None, "hash_single": None, "hashes": None,
           "groups": None, "values": None, "indices_ascending": None, "index_max": None, "indices_within_grid": None}
    raw = bytes(blob)
    if len(raw) < 4 or len(raw) % 4 or len(raw) > rsc8_audio.MAX_GRID20_BYTES:
        return out
    out["size_ok"] = True
    vals = struct.unpack_from(f"<{len(raw) // 4}I", raw, 0)
    declared, at = vals[0], 1
    groups, hashes, values, index_max, ascending, marker_ok = 0, set(), 0, None, True, True
    while at < len(vals):
        if at + 3 > len(vals) or groups >= rsc8_audio.MAX_INDEXRUN_GROUPS:
            out["size_ok"] = False
            return out
        marker, length, h = vals[at], vals[at + 1], vals[at + 2]
        if marker != 2:
            marker_ok = False
        if length > len(vals) - at - 3:
            out["size_ok"] = False
            return out
        hashes.add(h)
        run = vals[at + 3 : at + 3 + length]
        if any(run[i] >= run[i + 1] for i in range(len(run) - 1)):
            ascending = False
        if run:
            run_max = max(run)
            index_max = run_max if index_max is None else max(index_max, run_max)
        values += length
        at += 3 + length
        groups += 1
    out["group_count_matches"] = groups == declared
    out["marker_ok"] = marker_ok
    out["hash_single"] = len(hashes) == 1
    out["hashes"] = sorted(hashes)
    out["groups"] = groups
    out["values"] = values
    out["indices_ascending"] = ascending
    out["index_max"] = index_max
    out["indices_within_grid"] = None if grid_entries is None else (True if index_max is None else index_max < grid_entries)
    return out


def ref_markers(blob, sample_count=None):
    out = {"size_ok": False, "records": None, "names": None, "first_name": None, "offsets_ascending": None,
           "offset_max": None, "offsets_within_samples": None, "extra_max": None}
    raw = bytes(blob)
    if not raw or len(raw) % 16 or len(raw) > rsc8_audio.MAX_MARKER_BYTES:
        return out
    out["size_ok"] = True
    n = len(raw) // 16
    names, ascending, prev, offset_max, extra_max = {}, True, None, 0, 0
    for i in range(n):
        name, _value, offset, extra = struct.unpack_from("<IIII", raw, 16 * i)
        names[name] = names.get(name, 0) + 1
        if prev is not None and offset < prev:
            ascending = False
        prev = offset
        offset_max = max(offset_max, offset)
        extra_max = max(extra_max, extra)
    out["records"] = n
    out["names"] = names
    out["first_name"] = struct.unpack_from("<I", raw, 0)[0]
    out["offsets_ascending"] = ascending
    out["offset_max"] = offset_max
    out["extra_max"] = extra_max
    if sample_count is not None:
        out["offsets_within_samples"] = offset_max <= sample_count
    return out


def ref_adat(hb, bank_offset, file_size):
    if bank_offset < 0 or len(hb) < 16 or hb[:4] != b"ADAT":
        return None
    flags, count, hdr_sz = struct.unpack_from("<III", hb, 4)
    if (count > rsc8_audio.MAX_BANK_STREAMS or hdr_sz < 16 or hdr_sz > rsc8_audio.MAX_HEADER_BYTES
            or hdr_sz > len(hb) or bank_offset + hdr_sz > file_size):
        return None
    pos = 16
    if flags & 0x00400000:
        pos += 4
    if flags & 0x00010000:
        pos += count * 2
    if flags & 0x00200000:
        pos += (count + 1) * 4
    if count == 0:
        if pos != hdr_sz:
            return None
    elif pos + count * 4 > hdr_sz:
        return None
    streams, total_tags = [], 0
    for _ in range(count):
        w, = struct.unpack_from("<I", hb, pos)
        pos += 4
        streams.append((w >> 29, w & 0x1FFFFFFF))
        total_tags += w >> 29
    if pos + total_tags * 8 > hdr_sz:
        return None
    max_end, entries = hdr_sz, []
    for tc, sid in streams:
        tags = []
        for _ in range(tc):
            q, = struct.unpack_from("<Q", hb, pos)
            pos += 8
            tt, tsz, toff = (q >> 56) & 0xFF, (q >> 28) & 0x0FFFFFFF, q & 0x0FFFFFFF
            if bank_offset + toff + tsz > file_size:
                return None
            max_end = max(max_end, toff + tsz)
            tags.append((tt, toff, tsz))
        entries.append((sid, tags))
    return {"offset": bank_offset, "flags": flags, "count": count, "header_size": hdr_sz,
            "end": bank_offset + max_end, "entries": entries}


def ref_pair_regions_early_out(records):
    """The region check pair_marker_regions() started with: no record carries a start/end value word."""
    return not any(value in rsc8_audio._REGION_VALUES for _name, value, _offset, _extra in records)


def grid_blob(positions, p_values=None, magic=rsc8_audio.GRID20_MAGIC, tail=2 / 3):
    p_values = p_values or [512] * len(positions)
    body = b"".join(struct.pack("<IfHHHHHH", pos, 5.0, p, 0, 1, 2, 3, 4) for pos, p in zip(positions, p_values))
    return struct.pack("<I", magic) + body + struct.pack("<f", tail)


def marker_blob(records):
    return b"".join(struct.pack("<IIII", *r) for r in records)


def index_blob(groups, declared=None, marker=2, hash_=0x1234):
    out = struct.pack("<I", len(groups) if declared is None else declared)
    for run in groups:
        out += struct.pack("<III", marker, len(run), hash_) + b"".join(struct.pack("<I", v) for v in run)
    return out


class GridTest(unittest.TestCase):
    def test_sizes_around_the_block_edge(self):
        block = rsc8_audio.GRID20_CHUNK_ENTRIES
        for n in (0, 1, 2, block - 1, block, block + 1, 2 * block, 2 * block + 1, 3 * block - 1):
            blob = grid_blob([10 * (i + 1) for i in range(n)])
            self.assertEqual(rsc8_audio.decode_grid20_table(blob), ref_grid20(blob), n)

    def test_each_invariant_failing_at_each_block_edge(self):
        block = rsc8_audio.GRID20_CHUNK_ENTRIES
        n = 2 * block + 3
        for at in (1, block - 1, block, block + 1, 2 * block, n - 1):
            positions = [10 * (i + 1) for i in range(n)]
            positions[at] = positions[at - 1]  # equal: not strictly increasing
            blob = grid_blob(positions)
            got = rsc8_audio.decode_grid20_table(blob)
            self.assertFalse(got["position_increasing"], at)
            self.assertEqual(got, ref_grid20(blob), at)
            positions[at] = positions[at - 1] - 1  # decreasing
            blob = grid_blob(positions)
            self.assertEqual(rsc8_audio.decode_grid20_table(blob), ref_grid20(blob), at)
            for bad in (513, 16, 48):  # not a multiple of 32
                ps = [512] * n
                ps[at] = bad
                blob = grid_blob([10 * (i + 1) for i in range(n)], ps)
                got = rsc8_audio.decode_grid20_table(blob)
                self.assertFalse(got["p_multiple_of_32"], (at, bad))
                self.assertEqual(got, ref_grid20(blob), (at, bad))

    def test_only_the_low_half_of_the_word_is_p(self):
        # word 2 of an entry is p | z << 16; a z that is not a multiple of 32 must not fail p
        raw = struct.pack("<I", rsc8_audio.GRID20_MAGIC) + struct.pack("<IfHHHHHH", 1, 1.0, 512, 7, 0, 0, 0, 0)
        raw += struct.pack("<f", 0.5)
        got = rsc8_audio.decode_grid20_table(raw)
        self.assertTrue(got["p_multiple_of_32"])
        self.assertEqual(got, ref_grid20(raw))

    def test_bad_magic_size_and_random_tables(self):
        self.assertEqual(rsc8_audio.decode_grid20_table(grid_blob([1, 2], magic=7)), ref_grid20(grid_blob([1, 2], magic=7)))
        for raw in (b"", b"\0" * 7, b"\0" * 8, b"\0" * 9, b"\0" * 27):
            self.assertEqual(rsc8_audio.decode_grid20_table(raw), ref_grid20(raw))
        rnd = random.Random(1)
        for _ in range(200):
            n = rnd.randint(0, 600)
            positions = [rnd.randint(0, 2**32 - 1) for _ in range(n)]
            if rnd.random() < 0.5:
                positions.sort()
            ps = [rnd.choice([0, 512, 31, 33, 0xFFE0]) for _ in range(n)]
            blob = grid_blob(positions, ps)
            self.assertEqual(rsc8_audio.decode_grid20_table(blob), ref_grid20(blob))


class IndexRunsTest(unittest.TestCase):
    def test_groups_and_violations(self):
        cases = [
            [], [[]], [[1, 2, 3]], [[3, 2]], [[1, 1]], [[0], [5, 9], [], [7]], [[1, 2], [2, 1], [3, 4]],
            [[0xFFFFFFFF], [1, 0xFFFFFFFF]],
        ]
        for groups in cases:
            for declared in (None, 99):
                blob = index_blob(groups, declared)
                for grid in (None, 0, 3, 10**9):
                    self.assertEqual(rsc8_audio.decode_index_runs(blob, grid), ref_index_runs(blob, grid), (groups, declared, grid))
        for marker in (2, 3):
            blob = index_blob([[1, 2]], marker=marker)
            self.assertEqual(rsc8_audio.decode_index_runs(blob), ref_index_runs(blob))

    def test_descent_after_an_ascending_prefix_still_reports_and_keeps_the_max(self):
        blob = index_blob([[1, 2, 3], [9, 4], [5, 6, 100]])
        got = rsc8_audio.decode_index_runs(blob, 50)
        self.assertFalse(got["indices_ascending"])
        self.assertEqual(got["index_max"], 100)
        self.assertFalse(got["indices_within_grid"])
        self.assertEqual(got, ref_index_runs(blob, 50))

    def test_truncated_and_random_tables(self):
        blob = index_blob([[1, 2, 3], [4, 5]])
        for cut in range(len(blob) + 1):
            self.assertEqual(rsc8_audio.decode_index_runs(blob[:cut]), ref_index_runs(blob[:cut]), cut)
        rnd = random.Random(2)
        for _ in range(300):
            groups = [sorted(rnd.sample(range(40), rnd.randint(0, 6))) if rnd.random() < 0.7
                      else [rnd.randint(0, 40) for _ in range(rnd.randint(0, 6))] for _ in range(rnd.randint(0, 5))]
            blob = index_blob(groups, declared=rnd.choice([None, None, 1, 7]), hash_=rnd.choice([1, 1, 2]))
            grid = rnd.choice([None, 5, 50])
            self.assertEqual(rsc8_audio.decode_index_runs(blob, grid), ref_index_runs(blob, grid))


class MarkerTest(unittest.TestCase):
    def check(self, records, trailing=b""):
        blob = marker_blob(records) + trailing
        for sample_count in (None, 0, 100, 10**9):
            got = rsc8_audio.decode_marker_records(blob, sample_count)
            want = ref_markers(blob, sample_count)
            self.assertEqual(got, want)
            self.assertEqual(list(got["names"] or ()), list(want["names"] or ()))  # first-seen order
        self.assertEqual(rsc8_audio.marker_records(blob), list(struct.iter_unpack("<IIII", blob[: len(blob) - len(blob) % 16])))
        return blob

    def test_sizes_around_the_block_edge(self):
        block = rsc8_audio.MARKER_CHUNK_RECORDS
        for n in (0, 1, block - 1, block, block + 1, 2 * block, 2 * block + 1, 3 * block - 1):
            self.check([(0xE5C74063 + (i % 3), 0x3F800000, 7 * i, i % 7) for i in range(n)])

    def test_a_descent_at_each_block_edge_and_a_late_maximum(self):
        block = rsc8_audio.MARKER_CHUNK_RECORDS
        n = 2 * block + 2
        for at in (1, block - 1, block, block + 1, 2 * block, n - 1):
            records = [(5, 0, 1000 + 10 * i, 0) for i in range(n)]
            records[at] = (5, 0, 1000 + 10 * (at - 1) - 1, 0)  # one sample behind its predecessor
            got = rsc8_audio.decode_marker_records(self.check(records))
            self.assertFalse(got["offsets_ascending"], at)
        records = [(1, 0, 3, 0)] * (n - 1) + [(2, 0, 3, 9)]  # equal offsets stay ascending; the max is in the last block
        got = rsc8_audio.decode_marker_records(self.check(records))
        self.assertTrue(got["offsets_ascending"])
        self.assertEqual((got["extra_max"], got["names"]), (9, {1: n - 1, 2: 1}))

    def test_not_a_whole_number_of_records_and_empty(self):
        self.check([(1, 2, 3, 4)], trailing=b"\1")
        self.assertFalse(rsc8_audio.decode_marker_records(b"")["size_ok"])
        self.assertEqual(rsc8_audio.decode_marker_records(b"\0" * 15), ref_markers(b"\0" * 15))
        self.assertEqual(rsc8_audio.decode_marker_records(memoryview(marker_blob([(1, 2, 3, 4)]))),
                         ref_markers(marker_blob([(1, 2, 3, 4)])))

    def test_random_cue_lists(self):
        rnd = random.Random(3)
        names = [rnd.getrandbits(32) for _ in range(5)]
        for _ in range(150):
            off, records = 0, []
            for _ in range(rnd.choice([0, 3, 40, 700])):
                off = max(0, off + rnd.randint(-3, 30) if rnd.random() < 0.3 else off + rnd.randint(0, 30))
                records.append((rnd.choice(names), rnd.choice([START, END, 0x3F800000, 0xFFFFFFFF]), off, rnd.randint(0, 6)))
            self.check(records)

    def test_region_flag_matches_the_value_words(self):
        block = rsc8_audio.MARKER_CHUNK_RECORDS
        for words, want in (((0x3F800000,), False), ((START,), True), ((END,), True), ((START, END), True), ((), False)):
            for at in (0, block - 1, block, 2 * block + 1):
                records = [(1, 0x3F800000, i, 0) for i in range(2 * block + 3)]
                for k, value in enumerate(words):
                    records[at + k] = (1, value, at + k, 0)
                blob = marker_blob(records)
                self.assertEqual(rsc8_audio._decode_marker_records(blob)[1], want, (words, at))
                self.assertEqual(want, not ref_pair_regions_early_out(rsc8_audio.marker_records(blob)))

    def test_pair_marker_regions_still_returns_nothing_without_region_words(self):
        records = [(1, 0x3F800000, 5, 0), (1, 7, 6, 0)]
        self.assertEqual(rsc8_audio.pair_marker_regions(records), {"regions": [], "exceptions": [], "names": {}})
        paired = rsc8_audio.pair_marker_regions([(1, START, 5, 0), (1, END, 9, 1)])
        self.assertEqual([(r["start"], r["end"], r["extra_differs"]) for r in paired["regions"]], [(5, 9, True)])
        # a list of lists (not tuples) takes the same path
        self.assertEqual(rsc8_audio.pair_marker_regions([[1, START, 5, 0], [1, END, 9, 1]]), paired)

    def test_tally_path_matches_the_records_it_pairs(self):
        records = [(0xAA, START, 0, 0), (0xAA, END, 10, 0), (0xBB, 0x3F800000, 12, 0), (0xCC, END, 20, 1)]
        tally = rsc8_audio._new_grid_tally()
        blob = marker_blob(records)
        fmt = struct.pack("<I", 100) + bytes(28)
        tmap = {0xBD: [(0, len(blob))], 0x76: [(len(blob), len(fmt))]}
        data = blob + fmt
        rsc8_audio._tally_undocumented_tables(lambda off, n: data[off:off + n], 0, tmap, tally, 9)
        regions = tally["marker_regions"]
        self.assertEqual((tally["marker_streams"], tally["marker_records"], tally["marker_invalid"]), (1, 4, 0))
        self.assertEqual((regions["tags"], regions["tags_all_paired"], regions["tags_with_sample_count"]), (1, 0, 1))
        self.assertEqual(sum(c["pairs"] for c in regions["names"].values()), 1)
        self.assertEqual(regions["exceptions_total"], 1)
        # no region word: the pairing is never run
        plain = marker_blob([(1, 0x3F800000, 0, 0), (2, 0x3F800000, 1, 0)])
        tally = rsc8_audio._new_grid_tally()
        rsc8_audio._tally_undocumented_tables(lambda off, n: plain[off:off + n], 0, {0xBD: [(0, len(plain))]}, tally, 9)
        self.assertEqual((tally["marker_records"], tally["marker_regions"]["tags"]), (2, 0))


class AdatHeaderTest(unittest.TestCase):
    @staticmethod
    def bank(streams, flags=0xFF020001, hdr_extra=0, count=None):
        """`streams` is [(id, [(tag, offset, size), ...])]; header only, so a tag's data need not exist."""
        n = len(streams) if count is None else count
        pos = 16 + (4 if flags & 0x400000 else 0) + (n * 2 if flags & 0x10000 else 0) + ((n + 1) * 4 if flags & 0x200000 else 0)
        rows = sum(len(t) for _, t in streams)
        hdr_sz = pos + 4 * len(streams) + 8 * rows + hdr_extra
        hb = bytearray(struct.pack("<4sIII", b"ADAT", flags, n, hdr_sz)) + bytes(pos - 16)
        hb += b"".join(struct.pack("<I", (len(t) << 29) | sid) for sid, t in streams)
        for _, tags in streams:
            hb += b"".join(struct.pack("<Q", (tt << 56) | (tsz << 28) | toff) for tt, toff, tsz in tags)
        return bytes(hb).ljust(hdr_sz, b"\0")

    def check(self, hb, bank_offset=0, file_size=None):
        file_size = len(hb) + 10**6 if file_size is None else file_size
        got = rsc8_audio.parse_adat_header_bytes(hb, bank_offset, file_size)
        self.assertEqual(got, ref_adat(hb, bank_offset, file_size))
        return got

    def test_streams_with_and_without_tags(self):
        streams = [(0x11, [(0x55, 100, 20), (0x76, 120, 32)]), (0x22, []), (0x33, [(0xBD, 200, 16)] * 7), (0x44, [(0x5C, 300, 4)])]
        got = self.check(self.bank(streams))
        self.assertEqual([sid for sid, _ in got["entries"]], [0x11, 0x22, 0x33, 0x44])
        self.assertEqual(got["entries"][1][1], [])
        self.assertEqual(got["end"], 304)

    def test_flag_selected_tables_before_the_stream_words(self):
        streams = [(1, [(0x55, 100, 20)]), (2, [(0x76, 130, 32)])]
        for flags in (0, 0x400000, 0x10000, 0x200000, 0x610000, 0xFF020001):
            self.assertIsNotNone(self.check(self.bank(streams, flags)))

    def test_zero_stream_banks_and_header_slop(self):
        self.assertIsNotNone(self.check(self.bank([], 0xFF210001)))
        self.assertIsNone(self.check(self.bank([], 0xFF210001, hdr_extra=4)))
        self.assertIsNotNone(self.check(self.bank([], 0)))

    def test_tag_past_the_end_of_the_file_and_the_bank_offset(self):
        hb = self.bank([(1, [(0x55, 1000, 24)])])
        self.assertIsNotNone(self.check(hb, 0, 1024))
        self.assertIsNone(self.check(hb, 0, 1023))
        self.assertIsNotNone(self.check(hb, 500, 1524))
        self.assertIsNone(self.check(hb, 500, 1523))
        # a later stream's tag is the one that runs out
        hb = self.bank([(1, [(0x55, 100, 20)]), (2, [(0x55, 5000, 20)])])
        self.assertIsNone(self.check(hb, 0, 4000))

    def test_declared_counts_that_do_not_fit(self):
        good = self.bank([(1, [(0x55, 100, 20)])])
        self.assertIsNone(self.check(good[:20]))  # shorter than the declared header
        bad_count = bytearray(good)
        struct.pack_into("<I", bad_count, 8, 3)  # three streams declared, one present
        self.assertEqual(self.check(bytes(bad_count)), None)
        tags_over = bytearray(good)
        struct.pack_into("<I", tags_over, 16, (7 << 29) | 1)  # seven tags declared, one row present
        self.assertIsNone(self.check(bytes(tags_over)))
        self.assertIsNone(self.check(b"ADAT" + bytes(8)))
        self.assertIsNone(self.check(b"XXXX" + good[4:]))
        self.assertIsNone(rsc8_audio.parse_adat_header_bytes(good, -1, 10**6))

    def test_random_headers_match_the_reference(self):
        rnd = random.Random(4)
        valid = 0
        for _ in range(1500):
            streams = [(rnd.getrandbits(29), [(rnd.randint(0, 255), rnd.randint(0, 600), rnd.randint(0, 300))
                                              for _ in range(rnd.randint(0, 7))]) for _ in range(rnd.randint(0, 6))]
            hb = self.bank(streams, rnd.choice([0xFF020001, 0, 0x610000]), hdr_extra=rnd.choice([0, 0, 0, 4, -4]))
            if rnd.random() < 0.15:
                hb = hb[: rnd.randint(0, len(hb))]
            file_size = rnd.choice([len(hb) + 1000, 700, 10**6])
            valid += self.check(hb, rnd.choice([0, 0, 40]), file_size) is not None
        self.assertGreater(valid, 100)


class MemoryTest(unittest.TestCase):
    def peak(self, fn, *args):
        tracemalloc.start()
        try:
            fn(*args)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_large_tags_are_read_a_block_at_a_time(self):
        n = 200_000
        markers = marker_blob([(1 + (i & 3), START if i & 1 else END, i, i & 3) for i in range(n)])
        grid = grid_blob(list(range(1, n + 1)))
        # the transient is one block of words, not the table's
        self.assertLess(self.peak(rsc8_audio.decode_marker_records, markers), 256 * 1024)
        self.assertLess(self.peak(rsc8_audio.decode_grid20_table, grid), 256 * 1024)


if __name__ == "__main__":
    unittest.main()
