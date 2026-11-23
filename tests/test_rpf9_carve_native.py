"""The optional Rust scan of tools/rpf9_carve.py (tools/rpf9_carve_native) against the pure-Python reference.

Everything is synthetic, built here from seeded random data: zlib streams of every level, strategy and window
size, deflate streams written by a small encoder below with codes zlib never emits itself (one code of length
1, an empty distance code, the repeat codes), Bink 2-like headers, zero padding and noise, then the same with
flipped bits and cut-off ends. Nothing is pinned to bytes or hashes.

The comparisons skip when the library is not built (`cargo build --release --locked --offline --manifest-path
tools/rpf9_carve_native/Cargo.toml`). The fallback tests run either way.
"""
import contextlib
import io
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_carve.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rpf9_carve  # noqa: E402


def native_lib():
    rpf9_carve._NATIVE.clear()
    with mock.patch.dict(os.environ, {"LEONIDA_RPF9_CARVE_NATIVE": "1"}):
        return rpf9_carve._native_lib()


BUILT = native_lib() is not None


class Pages:
    """Stand-in for ModelSystemPages with a fixed `remaining`."""

    def __init__(self, remaining):
        self.remaining = remaining

    def minimum_size(self, adjacent):
        return 1 if adjacent and self.remaining == 1 else 64


# ---- a small deflate encoder for codes zlib's own compressor does not produce ----

LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195,
         227, 258]
LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
         4097, 6145, 8193, 12289, 16385, 24577]
DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def bits(self, value, count):
        self.acc |= value << self.n
        self.n += count
        while self.n >= 8:
            self.out.append(self.acc & 0xFF)
            self.acc >>= 8
            self.n -= 8

    def code(self, code, length):   # Huffman codes go most significant bit first
        self.bits(int(format(code, f"0{length}b")[::-1], 2), length)

    def flush(self):
        if self.n:
            self.out.append(self.acc & 0xFF)
        self.acc = self.n = 0
        return bytes(self.out)


def random_lengths(rng, symbols, size, max_len=15):
    """Lengths of a random complete prefix code over `symbols` (one symbol: length 1, an incomplete code)."""
    if not symbols:
        return [0] * size
    depths = [1] if len(symbols) == 1 else [1, 1]
    while len(depths) < len(symbols):
        i = rng.randrange(len(depths))
        if depths[i] < max_len:
            d = depths.pop(i)
            depths += [d + 1, d + 1]
    rng.shuffle(depths)
    lens = [0] * size
    for sym, d in zip(symbols, depths):
        lens[sym] = d
    return lens


def canonical(lens):
    codes, code = {}, 0
    for length in range(1, 16):
        for sym, ln in enumerate(lens):
            if ln == length:
                codes[sym] = (code, length)
                code += 1
        code <<= 1
    return codes


def length_symbol(n):
    i = max(k for k in range(29) if LBASE[k] <= n)
    return 257 + i, n - LBASE[i], LEXT[i]


def dist_symbol(d):
    i = max(k for k in range(30) if DBASE[k] <= d)
    return i, d - DBASE[i], DEXT[i]


def random_tokens(rng, allow_matches):
    toks, have = [], 0
    for _ in range(rng.randrange(0, 200)):
        if allow_matches and have and rng.random() < 0.4:
            d = rng.randint(1, min(have, 32768))
            n = rng.randint(3, 258)
            toks.append((n, d))
            have += n
        else:
            toks.append(rng.choice(b"abcdefgh\x00\xff"))
            have += 1
    return toks


def expand(toks):
    out = bytearray()
    for t in toks:
        if isinstance(t, int):
            out.append(t)
        else:
            n, d = t
            for _ in range(n):
                out.append(out[-d])
    return bytes(out)


def dynamic_block(w, rng, toks, final):
    lit_syms = sorted({t if isinstance(t, int) else length_symbol(t[0])[0] for t in toks} | {256})
    dist_syms = sorted({dist_symbol(t[1])[0] for t in toks if not isinstance(t, int)})
    nlen = max(257, max(lit_syms) + 1 + rng.randrange(3))
    nlen = min(nlen, 286)
    ndist = max(1, (max(dist_syms) + 1) if dist_syms else rng.randint(1, 4))
    lit = random_lengths(rng, lit_syms, nlen)
    dist = random_lengths(rng, dist_syms, ndist)
    seq = lit + dist
    # Encode the lengths, using 16/17/18 where they fit.
    cl_tokens, i = [], 0
    while i < len(seq):
        run = 1
        while i + run < len(seq) and seq[i + run] == seq[i]:
            run += 1
        v = seq[i]
        if v == 0 and run >= 11 and rng.random() < 0.7:
            n = min(run, 138)
            cl_tokens.append((18, n - 11, 7))
            i += n
        elif v == 0 and run >= 3 and rng.random() < 0.7:
            n = min(run, 10)
            cl_tokens.append((17, n - 3, 3))
            i += n
        elif i > 0 and seq[i - 1] == v and run >= 3 and rng.random() < 0.7:
            n = min(run, 6)
            cl_tokens.append((16, n - 3, 2))
            i += n
        else:
            cl_tokens.append((v, 0, 0))
            i += 1
    cl_syms = sorted({t[0] for t in cl_tokens})
    if len(cl_syms) == 1 and rng.random() < 0.8:   # one CL code is incomplete, which zlib refuses: add a spare
        cl_syms.append(next(x for x in range(19) if x not in cl_syms))
    cl = random_lengths(rng, cl_syms, 19, max_len=7)
    ncode = max(4, max(k + 1 for k in range(19) if cl[CL_ORDER[k]]))
    w.bits(final, 1)
    w.bits(2, 2)
    w.bits(nlen - 257, 5)
    w.bits(ndist - 1, 5)
    w.bits(ncode - 4, 4)
    for k in range(ncode):
        w.bits(cl[CL_ORDER[k]], 3)
    clc = canonical(cl)
    for sym, extra, nbits in cl_tokens:
        w.code(*clc[sym])
        if nbits:
            w.bits(extra, nbits)
    lc, dc = canonical(lit), canonical(dist)
    for t in toks:
        if isinstance(t, int):
            w.code(*lc[t])
        else:
            sym, extra, nbits = length_symbol(t[0])
            w.code(*lc[sym])
            if nbits:
                w.bits(extra, nbits)
            sym, extra, nbits = dist_symbol(t[1])
            w.code(*dc[sym])
            if nbits:
                w.bits(extra, nbits)
    w.code(*lc[256])


def handmade_stream(rng):
    """A zlib stream of one to three hand-encoded blocks (dynamic, fixed or stored) and its payload."""
    w, payload = BitWriter(), b""
    nblocks = rng.randint(1, 3)
    for b in range(nblocks):
        final = int(b == nblocks - 1)
        kind = rng.choice(("dyn", "dyn", "dyn", "fixed", "stored"))
        toks = random_tokens(rng, allow_matches=kind != "stored" and rng.random() < 0.7)
        data = expand(list(payload) + toks)[len(payload):]
        if kind == "stored":
            data = bytes(t for t in toks if isinstance(t, int))
            w.bits(final, 1)
            w.bits(0, 2)
            w.flush()   # to the byte boundary
            w.out += struct.pack("<HH", len(data), len(data) ^ 0xFFFF) + data
        elif kind == "fixed":
            w.bits(final, 1)
            w.bits(1, 2)
            fl = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
            lc, dc = canonical(fl), canonical([5] * 30)
            for t in toks:
                if isinstance(t, int):
                    w.code(*lc[t])
                else:
                    sym, extra, nbits = length_symbol(t[0])
                    w.code(*lc[sym])
                    if nbits:
                        w.bits(extra, nbits)
                    sym, extra, nbits = dist_symbol(t[1])
                    w.code(*dc[sym])
                    if nbits:
                        w.bits(extra, nbits)
            w.code(*lc[256])
        else:
            dynamic_block(w, rng, toks, final)
        payload += data
    body = w.flush()
    return b"\x78\x01" + body + struct.pack(">I", zlib.adler32(payload)), payload


def zlib_stream(rng):
    n = rng.choice((0, 1, 10, 63, 64, 65, 500, 5000, 70000))
    alphabet = bytes(rng.sample(range(256), rng.randint(1, 256)))
    data = bytes(rng.choice(alphabet) for _ in range(n)) if n < 5000 else (alphabet * (n // len(alphabet) + 1))[:n]
    c = zlib.compressobj(rng.choice((0, 1, 6, 9)), zlib.DEFLATED, rng.randint(9, 15), rng.randint(1, 9),
                         rng.choice((zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE,
                                     zlib.Z_FIXED)))
    s = c.compress(data) + c.flush()
    if s[1] not in (0x01, 0x5E, 0x9C, 0xDA):   # a header ZHDR does not match is never scanned
        s = b"\x78\x9c" + s[2:]
    return s


def mutate(rng, s):
    s = bytearray(s)
    r = rng.random()
    if r < 0.4 and len(s) > 2:
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(2, len(s))
            s[i] ^= 1 << rng.randrange(8)
    elif r < 0.7:
        s = s[:rng.randrange(len(s) + 1)]
    else:
        s[2:] = bytes(rng.randrange(256) for _ in range(rng.randint(0, 64)))
    return bytes(s)


def bink_header(rng):
    h = bytearray(b"KB2" + bytes([rng.choice(b"abcdgA")]) + bytes(0x1C))
    struct.pack_into("<I", h, 4, rng.choice((0, 50, 56, 200, 5000)))
    struct.pack_into("<I", h, 8, rng.randrange(1000))
    struct.pack_into("<II", h, 0x14, rng.choice((8, 16, 640, 8192, 9000)), rng.choice((8, 16, 360, 8193)))
    return bytes(h) + bytes(rng.randrange(256) for _ in range(rng.randrange(64)))


def random_buffer(rng):
    parts = []
    for _ in range(rng.randint(1, 8)):
        r = rng.random()
        if r < 0.3:
            parts.append(zlib_stream(rng))
        elif r < 0.5:
            parts.append(handmade_stream(rng)[0])
        elif r < 0.65:
            parts.append(mutate(rng, zlib_stream(rng) if rng.random() < 0.5 else handmade_stream(rng)[0]))
        elif r < 0.75:
            parts.append(bink_header(rng))
        elif r < 0.85:
            parts.append(bytes(rng.randrange(0, 33)))
        else:
            noise = bytearray(rng.randrange(256) for _ in range(rng.randrange(1, 300)))
            for _ in range(rng.randrange(4)):
                if noise:
                    i = rng.randrange(len(noise))
                    noise[i:i + 2] = rng.choice((b"\x78\x9c", b"\x78\xda", b"\x78\x01", b"KB2a"))
            parts.append(bytes(noise))
    return b"".join(parts)


def all_hits(step, buf, start, end, last, small):
    """Every hit from `start`, restarting one byte past each (so candidates inside kept streams count too)."""
    hits, pos = [], start
    while pos < end:
        h = step(pos)
        if h is False:
            break
        hits.append(h)
        pos = h[0] + 1
    return hits


@unittest.skipUnless(BUILT, "tools/rpf9_carve_native is not built")
class NativeScanParityTest(unittest.TestCase):
    def scan(self, buf):
        lib = rpf9_carve._native_lib()
        return rpf9_carve._NativeScan(*lib, buf)

    def compare(self, buf, rng):
        n = len(buf)
        end = n if rng.random() < 0.7 else rng.randint(0, n)
        start = rng.randint(0, end) if rng.random() < 0.3 else 0
        last = rng.randint(max(0, start - 20), start) if rng.random() < 0.5 else start
        small = rng.random() < 0.5
        pages = Pages(1 if small else rng.choice((0, 2)))
        native = self.scan(buf)
        try:
            got = all_hits(lambda p: native.next(p, end, last, small), buf, start, end, last, small)
        finally:
            native.close()
        want = all_hits(lambda p: rpf9_carve._next_hit(buf, p, end, last, pages), buf, start, end, last, small)
        self.assertEqual(got, want, (start, end, last, small))
        return got

    def test_random_buffers(self):
        rng = random.Random(0x5EED)
        kept = 0
        for _ in range(1500):
            kept += len(self.compare(random_buffer(rng), rng))
        self.assertGreater(kept, 1000)   # the comparison is about streams that are kept, not only rejections

    def test_handmade_streams_alone(self):
        rng = random.Random(1234)
        accepted = full = 0
        for _ in range(1500):
            s, payload = handmade_stream(rng)
            for buf in (s, mutate(rng, s)):
                hits = self.compare(buf, rng)
                accepted += any(h[0] == 0 for h in hits)
            if payload:   # the whole stream in a full window: kept by both, with its payload
                n = self.scan(s)
                try:
                    got = n.next(0, len(s), 0, True)
                finally:
                    n.close()
                self.assertEqual(got, (0, len(s), payload))
                self.assertEqual(rpf9_carve._next_hit(s, 0, len(s), 0, Pages(1)), got)
                full += 1
        self.assertGreater(accepted, 500)   # random windows and mutations cut many of them
        self.assertGreater(full, 1400)

    def test_every_truncation_and_bit_flip_of_one_stream(self):
        rng = random.Random(7)
        s = zlib.compress(bytes(rng.randrange(40) for _ in range(3000)), 9)
        native = self.scan(b"")
        native.close()
        for cut in range(len(s) + 1):
            buf = s[:cut]
            want = rpf9_carve._next_hit(buf, 0, len(buf), 0, Pages(0))
            n = self.scan(buf)
            try:
                self.assertEqual(n.next(0, len(buf), 0, False), want, cut)
            finally:
                n.close()
        for i in range(2, len(s)):
            for bit in (0, 3, 7):
                buf = bytearray(s)
                buf[i] ^= 1 << bit
                buf = bytes(buf)
                want = rpf9_carve._next_hit(buf, 0, len(buf), 0, Pages(0))
                n = self.scan(buf)
                try:
                    self.assertEqual(n.next(0, len(buf), 0, False), want, (i, bit))
                finally:
                    n.close()

    def test_gap_checks(self):
        rng = random.Random(99)
        for _ in range(600):
            size = rng.randrange(0, 3000)
            buf = bytearray(size)
            for _ in range(rng.randrange(3)):
                if size:
                    buf[rng.randrange(size)] = rng.randrange(1, 256)
            if rng.random() < 0.3:
                buf = bytearray(rng.randrange(rng.randint(1, 256)) for _ in range(size))
            buf = bytes(buf)
            a = rng.randint(0, size)
            b = rng.randint(a, size)
            n = self.scan(buf)
            try:
                self.assertEqual(n.all_zero(a, b), rpf9_carve.all_zero(buf, a, b))
                self.assertEqual(n.entropy(a, b), rpf9_carve.entropy(buf[a:b]))   # the same float, not close
                self.assertIsNone(n.all_zero(a, size + 1))
                self.assertIsNone(n.entropy(0, size + 1))
            finally:
                n.close()

    def test_bad_arguments_are_refused(self):
        buf = b"\x78\x9c" + bytes(100)
        n = self.scan(buf)
        try:
            self.assertIsNone(n.next(0, len(buf) + 1, 0, False))   # end past the buffer
            self.assertIsNone(n.next(10, 5, 0, False))             # pos past end
            self.assertIsNone(n.next(5, 50, 6, False))             # last past pos
        finally:
            n.close()


def synthetic_archive(rng, size_hint):
    """An RPF9-like file: header, then streams, Bink 2 headers, padding and noise with false headers."""
    out = bytearray(b"9FPR" + struct.pack("<I", 0x800) + bytes(0x7F8))
    while len(out) < size_hint:
        r = rng.random()
        if r < 0.55:
            data = bytes(rng.randrange(rng.randint(2, 256)) for _ in range(rng.randint(10, 4000)))
            if rng.random() < 0.05:
                data = bytes(65536)
            out += zlib.compress(data, rng.choice((1, 6, 9)))
            out += bytes((-len(out)) % 16) if rng.random() < 0.7 else b""
        elif r < 0.6:
            h = bytearray(b"KB2g" + bytes(0x60))
            struct.pack_into("<I", h, 4, len(h) - 8)
            struct.pack_into("<II", h, 0x14, 640, 360)
            out += h
        elif r < 0.75:
            out += bytes(rng.randrange(1, 5000))
        else:
            noise = bytearray(rng.randrange(256) for _ in range(rng.randrange(1, 3000)))
            for _ in range(rng.randrange(8)):
                i = rng.randrange(len(noise))
                noise[i:i + 2] = rng.choice((b"\x78\x9c", b"\x78\xda", b"KB2a"))
            out += noise
    return bytes(out)


def run_carve(arc, out, native, *extra):
    env = dict(os.environ, LEONIDA_RPF9_CARVE_NATIVE="1" if native else "0")
    p = subprocess.run([sys.executable, "-I", TOOL, arc, out, *extra], capture_output=True, text=True, env=env)
    return p


def folder(path):
    files = {}
    for name in sorted(os.listdir(path)):
        with open(os.path.join(path, name), "rb") as f:
            data = f.read()
        if name == "index.json":
            doc = json.loads(data)
            doc["summary"].pop("archive")
            data = json.dumps(doc).encode()
        files[name] = data
    return files


@unittest.skipUnless(BUILT, "tools/rpf9_carve_native is not built")
class WholeCarveParityTest(unittest.TestCase):
    def test_carves_match(self):
        rng = random.Random(42)
        with tempfile.TemporaryDirectory() as tmp:
            for case in range(6):
                arc = os.path.join(tmp, f"a{case}.rpf")
                with open(arc, "wb") as f:
                    f.write(synthetic_archive(rng, rng.choice((3000, 60000, 400000))))
                size = os.path.getsize(arc)
                options = [[], ["--index-only"], ["--limit", str(rng.randint(1, size))],
                           ["--start", str(rng.randrange(size)), "--index-only"]]
                for k, extra in enumerate(options):
                    outs = []
                    for native in (False, True):
                        out = os.path.join(tmp, f"o{case}_{k}_{int(native)}")
                        p = run_carve(arc, out, native, *extra)
                        self.assertEqual(p.returncode, 0, p.stderr)
                        outs.append((p.stdout, folder(out)))
                    self.assertEqual(outs[0], outs[1], (case, extra))

    def test_refusals_fall_back_step_by_step(self):
        """A refusal in the middle of a scan hands that step to Python; the index does not change."""
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            with open(arc, "wb") as f:
                f.write(synthetic_archive(rng, 200000))
            docs = []
            real_next = rpf9_carve._NativeScan.next
            real_zero = rpf9_carve._NativeScan.all_zero
            calls = [0]

            def flaky_next(self, *args):
                calls[0] += 1
                return None if calls[0] % 3 == 0 else real_next(self, *args)

            def flaky_zero(self, a, b):
                return None if (a + b) % 2 else real_zero(self, a, b)

            for patch in (False, True):
                out = os.path.join(tmp, f"o{int(patch)}")
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if patch:
                        stack.enter_context(mock.patch.object(rpf9_carve._NativeScan, "next", flaky_next))
                        stack.enter_context(mock.patch.object(rpf9_carve._NativeScan, "all_zero", flaky_zero))
                    else:
                        stack.enter_context(mock.patch.object(rpf9_carve, "_native_scan", lambda m: None))
                    self.assertEqual(rpf9_carve.main([arc, out, "--index-only"]), 0)
                docs.append(folder(out))
            self.assertGreater(calls[0], 10)
            self.assertEqual(docs[0], docs[1])

    def test_benchmark(self):
        """Timing only (printed, not asserted): the same carve with and without the library."""
        rng = random.Random(11)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "bench.rpf")
            with open(arc, "wb") as f:
                f.write(synthetic_archive(rng, 4 << 20))
            times = []
            for native in (False, True):
                out = os.path.join(tmp, f"b{int(native)}")
                t = time.perf_counter()
                p = run_carve(arc, out, native, "--index-only")
                times.append(time.perf_counter() - t)
                self.assertEqual(p.returncode, 0, p.stderr)
            print(f"\nrpf9_carve 4 MiB synthetic archive: Python {times[0]:.2f} s, native {times[1]:.2f} s "
                  f"({times[0] / max(times[1], 1e-9):.1f}x, both including interpreter start-up)", file=sys.stderr)


class FallbackTest(unittest.TestCase):
    def tearDown(self):
        rpf9_carve._NATIVE.clear()

    def test_opt_out(self):
        rpf9_carve._NATIVE.clear()
        with mock.patch.dict(os.environ, {"LEONIDA_RPF9_CARVE_NATIVE": "0"}):
            self.assertIsNone(rpf9_carve._native_lib())
            self.assertIsNone(rpf9_carve._native_scan(b"\x78\x9c"))

    def test_missing_library(self):
        rpf9_carve._NATIVE.clear()
        with mock.patch.dict(os.environ, {"LEONIDA_RPF9_CARVE_NATIVE": "1"}), \
                mock.patch.object(rpf9_carve.os.path, "isfile", return_value=False):
            self.assertIsNone(rpf9_carve._native_lib())

    def test_library_that_fails_to_load(self):
        rpf9_carve._NATIVE.clear()
        import ctypes
        with mock.patch.dict(os.environ, {"LEONIDA_RPF9_CARVE_NATIVE": "1"}), \
                mock.patch.object(rpf9_carve.os.path, "isfile", return_value=True), \
                mock.patch.object(ctypes, "CDLL", side_effect=OSError("bad image")):
            self.assertIsNone(rpf9_carve._native_lib())

    def test_wrong_abi_is_not_used(self):
        rpf9_carve._NATIVE.clear()
        import ctypes

        class Lib:
            class rpf9_abi:   # noqa: N801 (stands in for a ctypes function)
                def __call__(self):
                    return 2
            rpf9_abi = rpf9_abi()

        with mock.patch.dict(os.environ, {"LEONIDA_RPF9_CARVE_NATIVE": "1"}), \
                mock.patch.object(rpf9_carve.os.path, "isfile", return_value=True), \
                mock.patch.object(ctypes, "CDLL", return_value=Lib()):
            self.assertIsNone(rpf9_carve._native_lib())

    def test_carve_without_library_matches_tool_tests(self):
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            with open(arc, "wb") as f:
                f.write(synthetic_archive(rng, 50000))
            out = os.path.join(tmp, "o")
            p = run_carve(arc, out, False, "--index-only")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                self.assertGreater(json.load(f)["summary"]["files"], 0)

    @unittest.skipUnless(BUILT, "tools/rpf9_carve_native is not built")
    def test_buffer_is_released_so_the_mmap_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "x.rpf")
            with open(arc, "wb") as f:
                f.write(synthetic_archive(random.Random(8), 20000))
            with mock.patch.dict(os.environ, {"LEONIDA_RPF9_CARVE_NATIVE": "1"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                rpf9_carve._NATIVE.clear()
                self.assertEqual(rpf9_carve.main([arc, os.path.join(tmp, "o"), "--index-only"]), 0)
            os.remove(arc)   # an mmap left open would keep the file in use on Windows


if __name__ == "__main__":
    unittest.main()
