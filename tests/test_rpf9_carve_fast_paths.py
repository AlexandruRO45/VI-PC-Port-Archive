"""rpf9_carve's fast paths give what the straightforward code gives.

Each test compares the tool with a reference written out here (the regex search, the 1 MiB input blocks of the
inflate loop, a byte-counting zero check, and a whole carve over a small archive built in memory), so nothing is
pinned to a byte string or a hash and the result is the same on every platform. No game data is used.
"""
import collections
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_carve.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rpf9_carve  # noqa: E402

REF_HEADER = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")


def ref_all_zero(buf, a, b):
    return buf[a:b].count(0) == b - a


def ref_inflate(buf, s, end):
    """The inflate loop of the tool before the fast path: 1 MiB input blocks, to the end of the window."""
    d = zlib.decompressobj()
    chunks, p = [], s
    while not d.eof and p < end:
        chunk = buf[p:min(p + (1 << 20), end)]
        if not chunk:
            break
        chunks.append(d.decompress(chunk))
        p += len(chunk)
        if d.unused_data or d.eof:
            break
    out = b"".join(chunks) + d.flush()
    return (out, p - s - len(d.unused_data)) if d.eof else (None, 0)


def ref_entropy(sample):
    n = len(sample)
    return -sum(c / n * math.log2(c / n) for c in collections.Counter(sample).values()) if n else 0.0


def ref_carve(data, base, end):
    """A whole carve of data[base:end]: the rows, zero padding and unread ranges the tool must report."""
    rows, unread_ranges, zero, last, pos = [], [], 0, base, base

    def account(a, b):
        nonlocal zero
        if ref_all_zero(data, a, b):
            zero += b - a
            return
        sample = data[a:min(b, a + rpf9_carve.ENTROPY_SAMPLE)]
        unread_ranges.append({"off": a, "size": b - a, "entropy": round(ref_entropy(sample), 3),
                              "sampled": len(sample)})

    while pos < end:
        mt = REF_HEADER.search(data, pos, end)
        if not mt:
            break
        s = mt.start()
        if data[s:s + 3] == b"KB2":
            if s + 0x1C > end:
                pos = s + 1
                continue
            blen = struct.unpack_from("<I", data, s + 4)[0] + 8
            w, h = struct.unpack_from("<II", data, s + 0x14)
            if not (64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192):
                pos = s + 1
                continue
            account(last, s)
            rows.append((s, blen, blen, "bink2"))
            last = pos = s + blen
            continue
        try:
            out, used = ref_inflate(data, s, end)
        except zlib.error:
            pos = s + 1
            continue
        if out is None or len(out) < 64:
            pos = s + 1
            continue
        account(last, s)
        rows.append((s, used, len(out), None))
        last = pos = s + used
    account(last, end)
    return rows, zero, unread_ranges


def noisy(rnd, n, headers=0.02):
    """n random bytes with zlib header pairs and `KB2x` scattered in."""
    buf = bytearray(rnd.randbytes(n))
    for _ in range(int(n * headers)):
        k = rnd.randrange(max(1, n - 4))
        buf[k:k + 4] = rnd.choice((b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda", b"KB2a", b"KB2\x00",
                                   b"KB2", b"KB2{", b"KB2`", b"\x78\x9d")) + b"\x00\x00"
    return bytes(buf)


class AllZeroTest(unittest.TestCase):
    def test_matches_counting_every_byte(self):
        rnd = random.Random(1)
        for chunk in (1, 3, 16, 100, rpf9_carve.GAP_CHUNK):
            for size in (0, 1, 15, 16, 17, 99, 100, 101, 250):
                for _ in range(8):
                    data = bytearray(size)
                    if size and rnd.random() < 0.6:
                        data[rnd.randrange(size)] = rnd.randrange(1, 256)
                    a = rnd.randrange(size + 1)
                    b = rnd.randrange(a, size + 1)
                    self.assertEqual(rpf9_carve.all_zero(bytes(data), a, b, chunk), ref_all_zero(bytes(data), a, b),
                                     (chunk, size, a, b))

    def test_a_nonzero_byte_in_the_last_short_piece(self):
        data = bytearray(1000)
        data[999] = 1
        self.assertFalse(rpf9_carve.all_zero(bytes(data), 0, 1000, 64))
        self.assertTrue(rpf9_carve.all_zero(bytes(data), 0, 999, 64))


class InflateWindowTest(unittest.TestCase):
    def check(self, data, s, end):
        try:
            want = ref_inflate(data, s, end)
        except zlib.error:
            with self.assertRaises(zlib.error):
                rpf9_carve.inflate_window(data, s, end)
            return
        got = rpf9_carve.inflate_window(data, s, end)
        self.assertEqual(got, want, (s, end))

    def test_streams_ending_inside_and_outside_the_window(self):
        rnd = random.Random(2)
        for first, block in ((1, 1 << 20), (7, 64), (16, 16), (1 << 13, 1 << 20)):
            with mock.patch.object(rpf9_carve, "INFLATE_FIRST", first), \
                    mock.patch.object(rpf9_carve, "INFLATE_BLOCK", block):
                for n in (0, 1, 50, 1000, 40000, 300000):
                    raw = rnd.randbytes(n // 3) * 3 if n % 2 else rnd.randbytes(n)
                    z = zlib.compress(raw, rnd.choice((0, 1, 6, 9)))
                    tail = rnd.randbytes(rnd.choice((0, 5, 5000)))
                    data = b"\0\0\0" + z + tail
                    for end in (len(data), len(data) - len(tail), 3 + len(z) - 1, 3 + len(z) // 2, 4, 3):
                        self.check(data, 3, end)

    def test_garbage_and_truncated_candidates(self):
        rnd = random.Random(3)
        for first, block in ((1, 1 << 20), (5, 32), (1 << 13, 1 << 20)):
            with mock.patch.object(rpf9_carve, "INFLATE_FIRST", first), \
                    mock.patch.object(rpf9_carve, "INFLATE_BLOCK", block):
                for _ in range(300):
                    data = bytes([0x78, rnd.choice((0x01, 0x5E, 0x9C, 0xDA))]) + rnd.randbytes(rnd.randrange(0, 400))
                    self.check(data, 0, len(data))
                    self.check(data, 0, rnd.randrange(len(data) + 1))

    def test_the_result_does_not_depend_on_the_block_sizes(self):
        z = zlib.compress(random.Random(4).randbytes(5000) * 7, 1)
        data = b"\x01" * 11 + z + b"\xff" * 300
        results = set()
        for first in (1, 2, 3, 100, 1 << 13, 1 << 30):
            with mock.patch.object(rpf9_carve, "INFLATE_FIRST", first):
                out, used = rpf9_carve.inflate_window(data, 11, len(data))
                results.add((zlib.crc32(out), len(out), used))
        self.assertEqual(len(results), 1)
        self.assertEqual(results.pop()[2], len(z))


class HeaderScanTest(unittest.TestCase):
    def walk(self, data, end, steps, rnd):
        scan = rpf9_carve.HeaderScan(data, end)
        pos = 0
        for _ in range(steps):
            mt = REF_HEADER.search(data, pos, end)
            want = mt.start() if mt else -1
            self.assertEqual(scan.search(pos), want, (pos, end))
            if want < 0:
                break
            pos = want + rnd.choice((1, 1, 2, 3, 40, 1000, 70000))
            if pos >= end:
                break

    def test_the_same_headers_as_the_regex_in_noise(self):
        rnd = random.Random(5)
        for window, wmax in ((1 << 16, 1 << 24), (7, 64), (1, 1), (64, 64)):
            with mock.patch.object(rpf9_carve, "HEADER_WINDOW", window), \
                    mock.patch.object(rpf9_carve, "HEADER_WINDOW_MAX", wmax):
                for size in (0, 1, 2, 3, 4, 5, 50, 500, 5000, 100000):
                    data = noisy(rnd, size, rnd.choice((0.0, 0.002, 0.02)))
                    for end in {size, max(0, size - 1), max(0, size - 3), size // 2}:
                        self.walk(data, end, 400, rnd)

    def test_headers_in_zero_padding_and_at_the_edges(self):
        rnd = random.Random(6)
        for window, wmax in ((1 << 16, 1 << 24), (16, 16), (3, 100)):
            with mock.patch.object(rpf9_carve, "HEADER_WINDOW", window), \
                    mock.patch.object(rpf9_carve, "HEADER_WINDOW_MAX", wmax):
                for _ in range(40):
                    n = rnd.randrange(1, 3000)
                    data = bytearray(n)
                    for _ in range(rnd.randrange(0, 6)):
                        k = rnd.randrange(n)
                        data[k:k + 4] = rnd.choice((b"\x78\x9c", b"\x78\xda", b"KB2z", b"KB2A", b"KB2{", b"KB2`", b"\x78\x01"))
                    data = bytes(data)
                    for end in (n, n - 1, max(0, n - 2), max(0, n - 4)):
                        self.walk(data, end, 50, rnd)

    def test_kb2_needs_a_lowercase_letter_inside_the_window(self):
        data = b"\0" * 8 + b"KB2z" + b"\0" * 8
        scan = rpf9_carve.HeaderScan(data, 12)
        self.assertEqual(scan.search(0), 8)
        self.assertEqual(rpf9_carve.HeaderScan(data, 11).search(0), -1)   # the letter is outside the window
        self.assertEqual(rpf9_carve.HeaderScan(b"KB2A\0\0", 6).search(0), -1)
        self.assertEqual(REF_HEADER.search(data, 0, 11), None)
        for byte in b"`{@[\x00\xe1":   # just outside a-z
            edge = b"\0" * 4 + b"KB2" + bytes([byte]) + b"\0" * 4
            self.assertEqual(rpf9_carve.HeaderScan(edge, len(edge)).search(0), -1, byte)
            self.assertIsNone(REF_HEADER.search(edge))

    def test_a_smaller_position_searches_afresh(self):
        data = b"\0" * 100 + b"\x78\x9c" + b"\0" * 100 + b"\x78\xda" + b"\0" * 10
        scan = rpf9_carve.HeaderScan(data, len(data))
        self.assertEqual(scan.search(101), 202)
        self.assertEqual(scan.search(0), 100)
        self.assertEqual(scan.search(101), 202)

    def test_it_does_not_read_ahead_over_streams_it_will_skip(self):
        """A needle with no hit in the data is searched for only up to the first header, not to the end."""
        data = b"\x78\x9c" + b"\0" * (1 << 20)
        calls = []

        class Spy(bytes):
            def find(self, sub, start=0, stop=None):
                calls.append((sub, start, stop))
                return bytes.find(self, sub, start, stop)

        scan = rpf9_carve.HeaderScan(Spy(data), len(data))
        self.assertEqual(scan.search(0), 0)
        self.assertTrue(all(stop <= 2 + rpf9_carve.HEADER_WINDOW for _, _, stop in calls), calls)


def build_archive(rnd, parts):
    """An RPF9 file: header, then `parts` laid end to end (bytes, or ('zero', n) / ('noise', n))."""
    out = bytearray(b"9FPR" + struct.pack("<I", 0x40))
    out += bytes(0x40 - len(out))
    for part in parts:
        if isinstance(part, tuple):
            kind, n = part
            out += bytes(n) if kind == "zero" else noisy(rnd, n)
        else:
            out += part
    return bytes(out)


def bink(blen, w=64, h=64):
    head = b"KB2i" + struct.pack("<I", blen - 8) + struct.pack("<I", 7) + bytes(8) + struct.pack("<II", w, h)
    return head + bytes(0x1C - len(head)) + bytes(blen - 0x1C)


class WholeCarveTest(unittest.TestCase):
    def carve(self, data, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            arc, out = os.path.join(tmp, "x.rpf"), os.path.join(tmp, "out")
            with open(arc, "wb") as f:
                f.write(data)
            p = subprocess.run([sys.executable, "-I", TOOL, arc, out, "--index-only", *extra],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                return json.load(f)

    def compare(self, data, base=0x40, end=None, *extra):
        end = len(data) if end is None else end
        doc = self.carve(data, *extra)
        rows, zero, unread_ranges = ref_carve(data, base, end)
        self.assertEqual([(r["off"], r["stored"], r["size"], r.get("type")) for r in doc["files"]], rows)
        self.assertEqual(doc["summary"]["zero_padding"], zero)
        self.assertEqual(doc["unread"], unread_ranges)
        self.assertEqual(doc["summary"]["unread_bytes"], sum(r["size"] for r in unread_ranges))
        return doc

    def test_streams_padding_noise_and_videos(self):
        rnd = random.Random(7)
        parts = []
        for i in range(60):
            kind = rnd.random()
            if kind < 0.45:
                raw = rnd.randbytes(rnd.choice((10, 70, 3000))) * rnd.choice((1, 4, 40))
                parts.append(zlib.compress(raw, rnd.choice((1, 6))))
            elif kind < 0.6:
                parts.append(bink(rnd.choice((64, 200, 5000))))
            elif kind < 0.75:
                parts.append(("noise", rnd.choice((30, 700, 9000))))
            else:
                parts.append(("zero", rnd.choice((1, 15, 16, 4096, 70000))))
        parts.append(zlib.compress(b"tail" * 500)[:-3])   # a stream cut off by the end of the file
        doc = self.compare(build_archive(rnd, parts))
        self.assertGreater(doc["summary"]["files"], 10)
        self.assertGreater(doc["summary"]["bink2_files"], 0)
        self.assertGreater(len(doc["unread"]), 0)

    def test_with_small_blocks_and_windows(self):
        """The same carve with the block and window sizes forced small, in this process, against the reference."""
        rnd = random.Random(8)
        parts = [zlib.compress(rnd.randbytes(2000) * 3), ("noise", 5000), bink(300), ("zero", 9000),
                 zlib.compress(b"z" * 100000), ("noise", 700), zlib.compress(rnd.randbytes(70000))]
        data = build_archive(rnd, parts)
        for first, block, window, wmax in ((1, 1, 1, 1), (5, 64, 9, 90), (1 << 13, 1 << 20, 1 << 16, 1 << 24)):
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(rpf9_carve, "INFLATE_FIRST", first), \
                    mock.patch.object(rpf9_carve, "INFLATE_BLOCK", block), \
                    mock.patch.object(rpf9_carve, "HEADER_WINDOW", window), \
                    mock.patch.object(rpf9_carve, "HEADER_WINDOW_MAX", wmax):
                arc, out = os.path.join(tmp, "x.rpf"), os.path.join(tmp, "out")
                with open(arc, "wb") as f:
                    f.write(data)
                with mock.patch("sys.stdout"):
                    self.assertEqual(rpf9_carve.main([arc, out, "--index-only"]), 0)
                with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                    doc = json.load(f)
            rows, zero, unread_ranges = ref_carve(data, 0x40, len(data))
            self.assertEqual([(r["off"], r["stored"], r["size"], r.get("type")) for r in doc["files"]], rows,
                             (first, block, window, wmax))
            self.assertEqual(doc["summary"]["zero_padding"], zero)
            self.assertEqual(doc["unread"], unread_ranges)

    def test_limit_and_start_windows(self):
        rnd = random.Random(9)
        a, b, c = zlib.compress(rnd.randbytes(3000)), zlib.compress(b"b" * 9000), zlib.compress(b"c" * 5000)
        data = build_archive(rnd, [a, ("zero", 100), b, ("noise", 500), c, ("zero", 300)])
        off_b = 0x40 + len(a) + 100
        for limit in (len(a) + 50, len(a) + 100 + len(b) - 1, len(a) + 100 + len(b) + 3, len(data)):
            doc = self.carve(data, "--limit", str(limit))
            rows, zero, unread_ranges = ref_carve(data, 0x40, min(len(data), 0x40 + limit))
            self.assertEqual([(r["off"], r["stored"], r["size"]) for r in doc["files"]],
                             [(o, st, sz) for o, st, sz, _ in rows], limit)
            self.assertEqual(doc["summary"]["zero_padding"], zero, limit)
            self.assertEqual(doc["unread"], unread_ranges, limit)
        doc = self.carve(data, "--start", str(off_b))
        rows, zero, unread_ranges = ref_carve(data, off_b, len(data))
        self.assertEqual([(r["off"], r["stored"], r["size"]) for r in doc["files"]],
                         [(o, st, sz) for o, st, sz, _ in rows])
        self.assertEqual(doc["unread"], unread_ranges)

    def test_an_archive_of_zero_padding_only(self):
        doc = self.carve(build_archive(random.Random(10), [("zero", 5000)]))
        self.assertEqual(doc["summary"]["files"], 0)
        self.assertEqual(doc["summary"]["zero_padding"], 5000)
        self.assertEqual(doc["unread"], [])


if __name__ == "__main__":
    unittest.main()
