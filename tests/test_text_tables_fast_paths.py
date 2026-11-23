"""Parity tests for the fast paths of tools/text_tables.py, against inline reference implementations.

The reference functions below are the straightforward readings of the format (a regex scan over every byte, one
``struct.unpack_from`` per hash entry, a ``csv.DictWriter``). The fast paths must give the same results on valid,
sparse, dense and randomly damaged resources. Nothing here depends on the platform or on pinned byte counts or hashes.

    python -I tests/test_text_tables_fast_paths.py
"""
import csv
import importlib.util
import os
import random
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = _load("text_tables_fast_paths_under_test", os.path.join(ROOT, "tools", "text_tables.py"))
H = _load("test_text_tables_helpers", os.path.join(ROOT, "tests", "test_text_tables.py"))


def reference_string_tables(buf):
    """string_tables as one unpack per hash entry, over every DESCRIPTOR.finditer match."""
    for m in T.DESCRIPTOR.finditer(buf):
        index = T._ptr(m.group(1))
        hash_bytes = struct.unpack("<I", m.group(2))[0]
        hashes = T._ptr(m.group(3))
        chunks = T.blob_chunks(buf, m.end() - 16)
        if not (0 < hash_bytes < len(buf) and hashes + hash_bytes <= len(buf) and chunks):
            continue
        out = {}
        for o in range(hashes, hashes + hash_bytes - 15, 16):
            h, zero = struct.unpack_from("<II", buf, o)
            if not h or zero or buf[o + 8] != 1:
                continue
            i = int.from_bytes(buf[o + 11:o + 14], "little")
            at = index + 16 * i + 8
            if at + 8 > len(buf):
                continue
            ref = struct.unpack_from("<Q", buf, at)[0]
            if ref & T.CHUNK_MASK >= len(chunks):
                continue
            lo, hi = chunks[ref & T.CHUNK_MASK]
            start = lo + (ref >> 20)
            end = buf.find(b"\0", start, hi)
            if lo <= start <= end:
                out[h] = buf[start:end].decode("utf-8", "replace")
        if out:
            yield out


def reference_write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        n = 0
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def spans(matches):
    return [(m.start(), m.end(), m.groups()) for m in matches]


def sample_resources():
    words = "the you and to is what it this that hello street".split()
    rnd = random.Random(5)
    big = {f"ZZ_{i}": " ".join(rnd.choice(words) for _ in range(rnd.randint(1, 9))) for i in range(500)}
    return [H.make_resource(H.EN), H.make_resource(H.DE, pad=40), H.make_resource(big),
            H.make_chunked_resource([{"A": "aa", "B": "bb"}, {"C": "cc"}, {"D": "dd", "E": "e"}]),
            H.make_chunked_resource([{"A": "aa"}], between=[(0x20, 5)]),
            H.make_resource(H.EN) + bytes(100) + H.make_resource(H.DE)]


class TestStringTablesParity(unittest.TestCase):
    def test_valid_resources(self):
        for data in sample_resources():
            self.assertEqual(list(T.string_tables(data)), list(reference_string_tables(data)))
            self.assertTrue(list(T.string_tables(data)))

    def test_descriptor_search_matches_finditer(self):
        for data in sample_resources():
            self.assertEqual(spans(T.descriptors(data)), spans(T.DESCRIPTOR.finditer(data)))

    def test_descriptor_flags_sit_at_the_documented_offset(self):
        data = H.make_resource(H.EN)
        m = next(T.DESCRIPTOR.finditer(data))
        self.assertEqual(data.index(T.DESCRIPTOR_FLAGS), m.start() + T.DESCRIPTOR_FLAGS_AT)

    def test_no_descriptor_and_short_buffers(self):
        for data in (b"", b"\x01\x01\x00\x00", bytes(64), T.DESCRIPTOR_FLAGS * 50):
            self.assertEqual(list(T.descriptors(data)), [])
            self.assertEqual(list(T.string_tables(data)), [])

    def test_a_flags_literal_before_the_descriptor_start_is_ignored(self):
        data = T.DESCRIPTOR_FLAGS + H.make_resource(H.EN)[4:]
        self.assertEqual(spans(T.descriptors(data)), spans(T.DESCRIPTOR.finditer(data)))

    def test_back_to_back_descriptors_do_not_overlap(self):
        data = H.make_resource(H.EN)
        m = next(T.DESCRIPTOR.finditer(data))
        block = data[m.start():m.end()]
        stacked = block * 3 + data
        self.assertEqual(spans(T.descriptors(stacked)), spans(T.DESCRIPTOR.finditer(stacked)))

    def test_hash_table_not_a_multiple_of_16(self):
        data = bytearray(H.make_resource(H.EN))
        at = data.index(T.DESCRIPTOR_FLAGS) + 4
        for extra in (1, 7, 15):
            struct.pack_into("<I", data, at, 16 * len(H.EN) + extra)
            self.assertEqual(list(T.string_tables(bytes(data))), list(reference_string_tables(bytes(data))))
        struct.pack_into("<I", data, at, 15)
        self.assertEqual(list(T.string_tables(bytes(data))), [])

    def test_damaged_resources(self):
        rnd = random.Random(20240601)
        bases = sample_resources()
        for _ in range(3000):
            data = bytearray(rnd.choice(bases))
            for _ in range(rnd.randint(0, 6)):
                data[rnd.randrange(len(data))] = rnd.choice([0, 1, 0x10, 0x50, 0xFF, rnd.randrange(256)])
            if rnd.random() < 0.2:
                data += data[rnd.randrange(len(data)):][:rnd.randint(0, 80)]
            if rnd.random() < 0.1:
                del data[rnd.randrange(len(data)):]
            data = bytes(data)
            self.assertEqual(list(T.string_tables(data)), list(reference_string_tables(data)))
            self.assertEqual(spans(T.descriptors(data)), spans(T.DESCRIPTOR.finditer(data)))

    def test_dense_and_sparse_tables(self):
        dense = H.make_resource({f"L_{i}": f"text {i}" for i in range(20000)})
        sparse = bytes(1 << 20) + H.make_resource(H.EN) + bytes(1 << 20)
        for data in (dense, sparse):
            self.assertEqual(list(T.string_tables(data)), list(reference_string_tables(data)))


class TestCsvParity(unittest.TestCase):
    FIELDS = ["source", "offset", "lang", "hash", "label", "text"]

    def rows(self):
        return [{"source": "s", "offset": 1, "lang": "en", "hash": "0000000a", "label": "", "text": "a,b"},
                {"source": "s\"q", "offset": 2, "lang": "de", "hash": "0000000b", "label": "L", "text": ""},
                {"source": "ü", "offset": 3, "lang": "ja", "hash": "0000000c", "label": "X", "text": "こんにちは"}]

    def test_bytes_match_dictwriter(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
            self.assertEqual(T._write_csv(a, self.FIELDS, iter(self.rows())),
                             reference_write_csv(b, self.FIELDS, iter(self.rows())))
            with open(a, "rb") as fa, open(b, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())

    def test_single_field_and_no_rows(self):
        for fields, rows in ((["x"], [{"x": "1"}, {"x": "a,b"}]), (self.FIELDS, [])):
            with tempfile.TemporaryDirectory() as d:
                a, b = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
                self.assertEqual(T._write_csv(a, fields, iter(rows)), reference_write_csv(b, fields, iter(rows)))
                with open(a, "rb") as fa, open(b, "rb") as fb:
                    self.assertEqual(fa.read(), fb.read())


if __name__ == "__main__":
    unittest.main()
