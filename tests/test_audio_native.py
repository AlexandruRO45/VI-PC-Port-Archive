"""Parity, fallback and timing tests for the optional rsc8_audio kernels (tools/audio_native).

    cargo build --release --manifest-path tools/audio_native/Cargo.toml
    python -I tests/test_audio_native.py

Every input is synthesized from a seeded generator; no game file is read. Without the built library the parity
tests skip and the fallback tests still run. Set LEONIDA_AUDIO_NATIVE_BENCH=1 to print kernel timings.
"""
import json
import os
import random
import struct
import sys
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import _audio_native  # noqa: E402
import rsc8_audio  # noqa: E402
import test_rsc8_audio as synth  # noqa: E402

CASES = 1500
HAVE_NATIVE = _audio_native.native() is not None


def reference(fn, *args, **kwargs):
    """`fn` with every kernel switched off."""
    with mock.patch.object(_audio_native, "native", return_value=None):
        return fn(*args, **kwargs)


class Calls:
    """Counts the kernel calls that answered, so a parity test proves the native path ran."""

    def __init__(self, name):
        self.name = name
        self.answered = 0

    def __enter__(self):
        real = getattr(_audio_native, self.name)

        def counted(*args):
            got = real(*args)
            self.answered += got is not None
            return got

        self._patch = mock.patch.object(_audio_native, self.name, counted)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def random_adat(rng):
    """Bytes of a random ADAT header, valid or broken in one of several ways, and a file size."""
    flags = rng.choice((0, 0x00400000, 0x00010000, 0x00200000, 0x00610000, 0xFF610001, rng.getrandbits(32)))
    count = rng.choice((0, 1, 2, rng.randrange(1, 40), rng.randrange(40, 400)))
    tables = 16
    if flags & 0x00400000:
        tables += 4
    if flags & 0x00010000:
        tables += 2 * count
    if flags & 0x00200000:
        tables += 4 * (count + 1)
    tag_counts = [rng.choice((0, 1, 3, 7)) if rng.random() < 0.9 else rng.randrange(8) for _ in range(count)]
    ntags = sum(tag_counts)
    hdr = tables + 4 * count + 8 * ntags
    data_room = rng.choice((0, 64, 4096, 1 << 20))
    rows = bytearray()
    for _ in range(ntags):
        off = rng.randrange(hdr, hdr + data_room + 1) if rng.random() < 0.95 else rng.getrandbits(28)
        size = rng.randrange(0, 1 + max(0, hdr + data_room - off)) if off <= hdr + data_room else rng.getrandbits(28)
        rows += struct.pack("<Q", (rng.getrandbits(8) << 56) | ((size & 0x0FFFFFFF) << 28) | (off & 0x0FFFFFFF))
    sids = b"".join(struct.pack("<I", (tc << 29) | rng.getrandbits(29)) for tc in tag_counts)
    pad = rng.choice((0, 0, 4, rng.randrange(256)))
    hdr_sz = hdr + pad
    b = bytearray(b"ADAT" + struct.pack("<III", flags, count, hdr_sz))
    b += bytes(rng.getrandbits(8) for _ in range(tables - 16))
    b += sids + rows + bytes(pad)
    b += bytes(rng.choice((0, 32, 256)))
    file_size = len(b) + data_room
    damage = rng.random()
    if damage < 0.05:
        struct.pack_into("<I", b, 12, rng.getrandbits(32))
    elif damage < 0.10:
        struct.pack_into("<I", b, 8, rng.getrandbits(rng.choice((8, 18, 32))))
    elif damage < 0.15:
        b = b[: rng.randrange(len(b) + 1)]
    elif damage < 0.18:
        b[rng.randrange(4)] ^= 0x20
    elif damage < 0.25:
        file_size = rng.randrange(file_size + 1)
    elif damage < 0.30 and len(b) > 16:
        at = rng.randrange(16, len(b))
        b[at] = rng.getrandbits(8)
    return bytes(b), file_size


def random_grid20(rng):
    n = rng.choice((0, 1, 2, 7, 8, 9, rng.randrange(10, 300), rng.randrange(300, 3000)))
    pos = rng.randrange(1 << 16)
    entries = []
    sorted_ = rng.random() < 0.7
    for _ in range(n):
        pos = (pos + rng.randrange(1, 5000)) & 0xFFFFFFFF if sorted_ else rng.getrandbits(32)
        p = rng.randrange(0, 64) * 512 if rng.random() < 0.97 else rng.getrandbits(16)
        entries.append((pos, rng.uniform(3, 150), p, rng.randrange(17), rng.getrandbits(16), rng.getrandbits(16),
                        rng.getrandbits(16), rng.getrandbits(16)))
    if n and rng.random() < 0.1:
        k = rng.randrange(n)
        entries[k] = (entries[k - 1][0],) + entries[k][1:]
    raw = struct.pack("<I", rng.choice((rsc8_audio.GRID20_MAGIC, rsc8_audio.GRID20_MAGIC, rng.getrandbits(32))))
    raw += b"".join(struct.pack("<IfHHHHHH", *e) for e in entries)
    raw += struct.pack("<f", rng.choice((2 / 3, rng.uniform(-1, 1), float("nan"))))
    if rng.random() < 0.08:
        raw = raw[: rng.randrange(len(raw) + 1)]
    return raw


def random_index_runs(rng, grid=None):
    groups = rng.choice((0, 1, 2, rng.randrange(3, 40), rng.randrange(40, 300)))
    one_hash = rng.getrandbits(32)
    words = [groups if rng.random() < 0.9 else rng.getrandbits(8)]
    for _ in range(groups):
        length = rng.choice((0, 1, 2, rng.randrange(3, 60)))
        start = rng.randrange(1000)
        if rng.random() < 0.85:
            run = sorted(rng.sample(range(start, start + 4 * length + 4), length))
        else:
            run = [rng.randrange(start, start + 50) for _ in range(length)]
        h = one_hash if rng.random() < 0.95 else rng.getrandbits(32)
        words += [2 if rng.random() < 0.97 else rng.getrandbits(4), length, h] + run
    if rng.random() < 0.1 and len(words) > 1:
        words[rng.randrange(1, len(words))] = rng.getrandbits(rng.choice((4, 16, 32)))
    raw = struct.pack(f"<{len(words)}I", *words)
    if rng.random() < 0.08:
        raw = raw[: rng.randrange(len(raw) + 1)]
    return raw


def random_markers(rng):
    n = rng.choice((0, 1, 2, 7, 8, 9, rng.randrange(10, 200), rng.randrange(200, 3000)))
    pool = [rng.getrandbits(32) for _ in range(rng.randrange(1, 30))]
    offset = 0
    rows = []
    region = (rsc8_audio.MARKER_VALUE_START, rsc8_audio.MARKER_VALUE_END)
    for _ in range(n):
        offset = offset + rng.randrange(0, 4000) if rng.random() < 0.98 else rng.getrandbits(32)
        offset &= 0xFFFFFFFF
        value = rng.choice(region) if rng.random() < 0.05 else rng.getrandbits(32)
        rows.append((rng.choice(pool), value, offset, rng.choice((0, 1, 2, 6, rng.getrandbits(32)))))
    raw = b"".join(struct.pack("<IIII", *r) for r in rows)
    if rng.random() < 0.05:
        raw = raw[: rng.randrange(len(raw) + 1)]
    return raw


def build_bank(streams):
    """An ADAT bank of `streams` [(stream id, {tag: payload})], at most 7 tags a stream."""
    ntags = sum(len(tags) for _sid, tags in streams)
    hdr_sz = 16 + 4 * len(streams) + 8 * ntags
    hdr = bytearray(struct.pack("<4sIII", b"ADAT", 0xFF020001, len(streams), hdr_sz))
    rows = bytearray()
    body = bytearray()
    for sid, tags in streams:
        hdr += struct.pack("<I", (len(tags) << 29) | sid)
        for tt, payload in sorted(tags.items()):
            rows += struct.pack("<Q", (tt << 56) | (len(payload) << 28) | (hdr_sz + len(body)))
            body += payload + bytes(-len(payload) % 16)
    return bytes(hdr + rows + body)


@unittest.skipUnless(HAVE_NATIVE, "tools/audio_native is not built")
class ParityTest(unittest.TestCase):
    def check(self, name, fn, inputs):
        with Calls(name) as calls:
            for args in inputs:
                fast = fn(*args)
                ref = reference(fn, *args)
                self.assertEqual(repr(fast), repr(ref), args[:1] if len(repr(args)) < 400 else name)
        return calls.answered

    def test_adat_header(self):
        rng = random.Random(0xADA7)
        inputs = []
        for _ in range(CASES):
            hb, file_size = random_adat(rng)
            bank_offset = rng.choice((0, 0, 16, rng.getrandbits(20)))
            inputs.append((hb, bank_offset, file_size + bank_offset * (rng.random() < 0.8)))
        answered = self.check("adat_header", rsc8_audio.parse_adat_header_bytes, inputs)
        valid = sum(rsc8_audio.parse_adat_header_bytes(*a) is not None for a in inputs)
        self.assertGreater(answered, CASES // 3)
        self.assertGreater(valid, CASES // 5)
        self.assertLess(valid, CASES - CASES // 10)

    def test_grid20(self):
        rng = random.Random(0x5A)
        answered = self.check("grid20", rsc8_audio.decode_grid20_table, [(random_grid20(rng),) for _ in range(CASES)])
        self.assertGreater(answered, CASES // 3)

    def test_index_runs(self):
        rng = random.Random(0xD9)
        inputs = [(random_index_runs(rng), rng.choice((None, 0, 10, 500, 5000))) for _ in range(CASES)]
        answered = self.check("index_runs", rsc8_audio.decode_index_runs, inputs)
        self.assertGreater(answered, CASES // 3)

    def test_index_runs_group_cap(self):
        rng = random.Random(0xD9D9)
        with mock.patch.object(rsc8_audio, "MAX_INDEXRUN_GROUPS", 5):
            inputs = [(random_index_runs(rng), 100) for _ in range(300)]
            self.check("index_runs", rsc8_audio.decode_index_runs, inputs)
            capped = sum(not rsc8_audio.decode_index_runs(*a)["size_ok"] for a in inputs)
        self.assertGreater(capped, 50)

    def test_markers(self):
        rng = random.Random(0xBD)
        inputs = [(random_markers(rng), rng.choice((None, 0, 1000, 1 << 31))) for _ in range(CASES)]
        answered = self.check("markers", rsc8_audio.decode_marker_records, inputs)
        self.assertGreater(answered, CASES // 3)

    def test_marker_has_value(self):
        rng = random.Random(0xBDBD)
        start, end = rsc8_audio.MARKER_VALUE_START, rsc8_audio.MARKER_VALUE_END
        answered = 0
        for _ in range(CASES):
            raw = random_markers(rng) + bytes(rng.choice((0, 0, 5, 15)))
            fast = _audio_native.marker_has_value(raw, start, end)
            if fast is None:
                continue
            answered += 1
            ref = any(v in rsc8_audio._REGION_VALUES for _n, v, _o, _e in rsc8_audio.marker_records(raw))
            self.assertEqual(fast, ref)
        self.assertGreater(answered, CASES // 3)

    def test_edges(self):
        rsc = synth.build_synthetic_rsc8()
        # Exactly at and around every MIN_* threshold, and the bounds of every word.
        for n in range(0, 12):
            self.check("grid20", rsc8_audio.decode_grid20_table, [(synth.build_synthetic_grid20(
                [(i * 7, 100.0, 512 * i, 0, 0, 0, 0, 0) for i in range(n)]),)])
            self.check("markers", rsc8_audio.decode_marker_records, [(synth.build_synthetic_markers(
                [(0xFFFFFFFF, 0, 0xFFFFFFFF - n, 0xFFFFFFFF)] * n), 0xFFFFFFFF)])
        top = [(0xFFFFFFFF, 2.0, 0xFFFF, 0, 0xFFFF, 0, 0, 0)] * 9
        self.check("grid20", rsc8_audio.decode_grid20_table, [(synth.build_synthetic_grid20(top),)])
        self.check("index_runs", rsc8_audio.decode_index_runs, [
            (struct.pack("<18I", 1, 2, 14, 0xFFFFFFFF, *range(0xFFFFFFF0, 0xFFFFFFFE)), 0xFFFFFFFF),
            (struct.pack("<16I", 0, *([2, 0, 7] * 5)), None),
            (struct.pack("<16I", 5, 2, 0xFFFFFFFF, 1, *range(12)), 3),
        ])
        bank = synth.build_synthetic_awc([{"id": i, "rsc8": rsc} for i in range(8)])
        self.check("adat_header", rsc8_audio.parse_adat_header_bytes, [
            (bytes(bank), 0, len(bank)), (bytes(bank), (1 << 63), (1 << 64) - 1), (bytes(bank), 1 << 64, 1 << 65),
            (bytes(bank), 0, len(bank) - 1), (bytes(bank), -1, len(bank)),
        ])

    def test_whole_bank(self):
        """decode_awc_bank and scan_dir give the same JSON with and without the kernels."""
        rng = random.Random(0xA3C)
        rsc = synth.build_synthetic_rsc8()
        for trial in range(30):
            streams = []
            for i in range(rng.randrange(1, 12)):
                fmt = bytearray(32)
                struct.pack_into("<IH", fmt, 0, rng.randrange(1 << 20), 0)
                struct.pack_into("<H", fmt, 8, 48000)
                fmt[28], fmt[29] = 0x0B, 1
                tags = {0x55: bytes(64), 0x5C: rsc, 0x76: bytes(fmt),
                        0xBD: random_markers(rng) or struct.pack("<4I", 1, 2, 3, 4)}
                for tt, payload in ((0x5A, random_grid20(rng)), (0xD9, random_index_runs(rng))):
                    if payload:
                        tags[tt] = payload
                streams.append((0x100 + i, tags))
            data = build_bank(streams)
            fast = rsc8_audio.decode_awc_bank(data)
            ref = reference(rsc8_audio.decode_awc_bank, data)
            self.assertEqual(json.dumps(fast, sort_keys=False, default=repr),
                             json.dumps(ref, sort_keys=False, default=repr), trial)


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self._saved = list(_audio_native._NATIVE)
        _audio_native._NATIVE.clear()

    def tearDown(self):
        _audio_native._NATIVE[:] = self._saved

    def test_opt_out(self):
        with mock.patch.dict(os.environ, {"LEONIDA_AUDIO_NATIVE": "0"}):
            self.assertIsNone(_audio_native.native())
            raw = struct.pack("<IIII", 1, 2, 3, 4) * 64
            self.assertIsNone(_audio_native.markers(raw))
            self.assertEqual(rsc8_audio.decode_marker_records(raw)["records"], 64)

    def test_missing_library(self):
        with mock.patch.object(_audio_native, "library_path", return_value=os.path.join(ROOT, "no", "such.dll")):
            self.assertIsNone(_audio_native.native())
        raw = synth.build_synthetic_grid20([(i, 1.0, 0, 0, 0, 0, 0, 0) for i in range(50)])
        with mock.patch.object(_audio_native, "native", return_value=None):
            self.assertEqual(rsc8_audio.decode_grid20_table(raw)["position_last"], 49)

    def test_unloadable_or_other_abi(self):
        import ctypes

        class OtherAbi:
            def audio_native_abi(self):
                return _audio_native.ABI + 1

        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch.object(ctypes, "CDLL", side_effect=OSError("bad image")):
                self.assertIsNone(_audio_native._load())
            with mock.patch.object(ctypes, "CDLL", return_value=OtherAbi()):
                self.assertIsNone(_audio_native._load())

    @unittest.skipUnless(HAVE_NATIVE, "tools/audio_native is not built")
    def test_declines(self):
        raw = struct.pack("<IIII", 1, 2, 3, 4) * 64
        self.assertIsNone(_audio_native.markers(bytearray(raw)))  # only bytes are passed by pointer
        self.assertIsNone(_audio_native.markers(raw[:16]))  # below MIN_MARKER_BYTES
        self.assertIsNone(_audio_native.adat_header(b"ADAT" + bytes(200), -1, 300, 10, 1000))
        self.assertIsNone(_audio_native.index_runs(bytes(64), -1))

    @unittest.skipUnless(HAVE_NATIVE, "tools/audio_native is not built")
    def test_kernel_refusal_falls_back(self):
        lib = list(_audio_native.native() or _audio_native._load())
        refuse = lambda *a: 2  # noqa: E731
        raw = struct.pack("<IIII", 7, 2, 3, 4) * 64
        for index in (1, 2, 3, 4):
            patched = list(lib)
            patched[index] = refuse
            with mock.patch.object(_audio_native, "native", return_value=tuple(patched)):
                self.assertIsNone(_audio_native.markers(raw) if index == 4 else None)
                self.assertEqual(rsc8_audio.decode_marker_records(raw), reference(rsc8_audio.decode_marker_records, raw))
                grid = synth.build_synthetic_grid20([(i, 1.0, 0, 0, 0, 0, 0, 0) for i in range(50)])
                self.assertEqual(rsc8_audio.decode_grid20_table(grid), reference(rsc8_audio.decode_grid20_table, grid))
                runs = struct.pack("<21I", 1, 2, 17, 9, *range(17))
                self.assertEqual(rsc8_audio.decode_index_runs(runs, 20), reference(rsc8_audio.decode_index_runs, runs, 20))
                bank = bytes(synth.build_synthetic_awc([{"id": 1, "rsc8": synth.build_synthetic_rsc8()}] * 4))
                self.assertEqual(rsc8_audio.parse_adat_header_bytes(bank, 0, len(bank)),
                                 reference(rsc8_audio.parse_adat_header_bytes, bank, 0, len(bank)))


def _best(fn, repeat=7):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


@unittest.skipUnless(HAVE_NATIVE and os.environ.get("LEONIDA_AUDIO_NATIVE_BENCH") == "1",
                     "set LEONIDA_AUDIO_NATIVE_BENCH=1 with the library built")
class BenchTest(unittest.TestCase):
    def test_bench(self):
        rng = random.Random(7)
        grid = synth.build_synthetic_grid20([(i * 300, 100.0, 512, 0, 0, 0, 0, 0) for i in range(100_000)])
        runs = struct.pack("<I", 1000) + b"".join(
            struct.pack("<3I", 2, 100, 0x4C633D07) + struct.pack("<100I", *range(i, i + 100)) for i in range(1000))
        marks = b"".join(struct.pack("<IIII", rng.choice((1, 2, 3, 4)), 0, i * 10, 1) for i in range(100_000))
        streams = [(i, [(0x55, 100 + i, 10), (0x76, 200 + i, 32), (0xBD, 300 + i, 16)]) for i in range(50_000)]
        hdr = 16 + 4 * len(streams) + 8 * 3 * len(streams)
        hb = b"ADAT" + struct.pack("<III", 0, len(streams), hdr)
        hb += b"".join(struct.pack("<I", (3 << 29) | sid) for sid, _ in streams)
        hb += b"".join(struct.pack("<Q", (tt << 56) | (sz << 28) | (off + hdr)) for _, tags in streams
                       for tt, off, sz in tags)
        rows = [
            ("parse_adat_header_bytes, 50,000 streams", lambda: rsc8_audio.parse_adat_header_bytes(hb, 0, 1 << 30)),
            ("decode_grid20_table, 100,000 entries", lambda: rsc8_audio.decode_grid20_table(grid)),
            ("decode_index_runs, 1,000 x 100", lambda: rsc8_audio.decode_index_runs(runs, 2000)),
            ("decode_marker_records, 100,000 records", lambda: rsc8_audio.decode_marker_records(marks)),
        ]
        for label, fn in rows:
            self.assertEqual(repr(fn()), repr(reference(fn)))
            py = _best(lambda: reference(fn), 3)
            nat = _best(fn)
            print(f"\n{label}: python {py * 1e3:.2f} ms, native {nat * 1e3:.3f} ms, {py / nat:.1f}x", end="")
        print()


if __name__ == "__main__":
    unittest.main()
