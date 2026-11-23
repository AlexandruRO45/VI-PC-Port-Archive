"""Carve the zlib-compressed files out of an RPF9 archive without its (encrypted) table of contents.

The data area is mostly zlib streams laid end to end, separated by zero padding. This walks the archive, inflates
every stream that decodes cleanly to the end, and writes each one as <offset>.bin. It also records which byte
ranges it couldn't read, so coverage is measurable.

index.json holds the summary (totals), `files` (one row per carved stream) and `unread`: one row per gap between
streams (or after the last one) that is not all zero bytes, with its offset, size and the Shannon entropy in
bits/byte of its first ENTROPY_SAMPLE bytes (`sampled`). An all-zero gap is padding and only counts toward
`zero_padding`. The bytes are counted, never decoded.

Usage:
    python -I rpf9_carve.py <archive.rpf> <out_dir> [--start N] [--limit BYTES] [--until-adat] [--index-only]

--until-adat ends the scan at the first raw AWC bank (a little-endian `ADAT` whose header parses as one,
rsc8_audio.parse_adat_header_bytes), so the zlib-compressed audio config at the start of audio.rpf is carved without
touching the banks and without a build-specific --limit (docs/audio_config.md). The bytes `ADAT` alone are not
enough: they also occur inside `METADATA` and in compressed data. index.json's summary records the boundary.
"""
import argparse
import collections
import json
import math
import mmap
import os
import re
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsc8_audio import parse_adat_header_bytes  # noqa: E402
from restofworld_model_join import ModelSystemPages  # noqa: E402

# zlib stream start, or a Bink 2 video ('KB2' + revision letter, then u32 size-8).
# RPF9 magic as it sits on disk: the tag "RPF9" stored little-endian (docs/rpf.md, docs/bvh.md).
MAGIC = b"9FPR"
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
ENTROPY_SAMPLE = 1 << 20   # bytes of an unread range used for its entropy
GAP_CHUNK = 1 << 20        # a gap is checked this many bytes at a time: one can be gigabytes long


def all_zero(buf, a, b, chunk=GAP_CHUNK):
    """Whether buf[a:b] is all zero bytes, reading at most `chunk` bytes at a time.

    Slicing a whole gap of a multi-gigabyte archive at once ran out of memory (a 20 GB archive in this package).
    """
    for p in range(a, b, chunk):
        piece = buf[p:min(p + chunk, b)]
        if piece.count(0) != len(piece):
            return False
    return True


def entropy(sample):
    """Shannon entropy of a byte string in bits per byte (0.0 for an empty one)."""
    if not sample:
        return 0.0
    n = len(sample)
    return -sum(c / n * math.log2(c / n) for c in collections.Counter(sample).values())


def below_normal():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        # GetCurrentProcess returns a pointer-sized pseudo-handle.
        k.GetCurrentProcess.argtypes = ()
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k.SetPriorityClass.restype = wintypes.BOOL
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


def offset(text):
    """--start: 0 or more, decimal or 0x hex."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 0 or more") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def byte_count(text):
    """--limit: at least 1; 0 used to mean no limit and a negative value silently scanned less."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


ADAT_HEADER_READ = 1 << 20   # enough for every bank header the parser accepts in practice; it rejects longer ones


def first_adat_bank(m, start, end):
    """Offset of the first raw AWC bank in m[start:end] whose header parses (not just the four bytes), or None."""
    size = len(m)
    pos = start
    while pos < end:
        at = m.find(b"ADAT", pos, end)
        if at < 0:
            return None
        if parse_adat_header_bytes(m[at:min(size, at + ADAT_HEADER_READ)], at, size) is not None:
            return at
        pos = at + 1
    return None


_NATIVE = []


class _NativeScan:
    """The optional Rust candidate scan of `tools/rpf9_carve_native` (built with Cargo, see its README.md).

    `next` finds what `_next_hit` finds (the same header search, Bink 2 check and inflate, with zlib's
    accept/reject rules), and `all_zero` / `entropy` replace the gap checks. None from any of them means
    "refused": the caller runs the Python code for that step, so the output does not depend on the library.
    """

    def __init__(self, lib, ctypes, buffer_type, m):
        self._c = ctypes
        self._lib = lib
        self._view = buffer_type()
        api = ctypes.pythonapi
        if api.PyObject_GetBuffer(m, ctypes.byref(self._view), 0) != 0:   # PyBUF_SIMPLE: contiguous bytes
            raise OSError("the mapping has no buffer")
        self._held = True
        self._buf, self._size = self._view.buf, self._view.len
        if self._view.itemsize != 1 or self._size != len(m):
            self.close()
            raise OSError("unexpected buffer layout")
        self._ctx = lib.rpf9_ctx_new()
        if not self._ctx:
            self.close()
            raise OSError("no scan context")
        self._res = (ctypes.c_uint64 * 4)()
        self._counts = (ctypes.c_uint64 * 256)()
        self._order = (ctypes.c_uint8 * 256)()

    def close(self):
        """Release the mmap's buffer (the mmap cannot close while it is held) and the scan context."""
        if self._held:
            if getattr(self, "_ctx", None):
                self._lib.rpf9_ctx_free(self._ctx)
                self._ctx = None
            self._c.pythonapi.PyBuffer_Release(self._c.byref(self._view))
            self._held = False

    def next(self, pos, end, last, small_ok):
        """(offset, stored size, inflated bytes or None for Bink 2), False for no header left, None to refuse."""
        r = self._res
        kind = self._lib.rpf9_next(self._ctx, self._buf, self._size, pos, end, last, 1 if small_ok else 0, r)
        if kind == 0:
            return False
        if kind == 1:
            return r[0], r[1], self._c.string_at(r[3], r[2])
        if kind == 2:
            return r[0], r[1], None
        return None

    def all_zero(self, a, b):
        z = self._lib.rpf9_all_zero(self._buf, self._size, a, b)
        return None if z < 0 else z == 1

    def entropy(self, a, b):
        """`entropy(m[a:b])` from native byte counts, summed in Counter's (first-seen) order: the same float."""
        k = self._lib.rpf9_byte_counts(self._buf, self._size, a, b, self._counts, self._order)
        if k < 0:
            return None
        if a == b:
            return 0.0
        n, counts = b - a, self._counts
        return -sum(c / n * math.log2(c / n) for c in (counts[v] for v in self._order[:k]))


def _native_lib():
    """(library, ctypes, Py_buffer type) of `rpf9_carve_native`, or None: not built, fails to load, or
    LEONIDA_RPF9_CARVE_NATIVE=0. The pure-Python scan below is the reference either way."""
    if not _NATIVE:
        found = None
        if os.environ.get("LEONIDA_RPF9_CARVE_NATIVE", "1") != "0":
            name = {"win32": "rpf9_carve_native.dll",
                    "darwin": "librpf9_carve_native.dylib"}.get(sys.platform, "librpf9_carve_native.so")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rpf9_carve_native", "target", "release", name)
            if os.path.isfile(path):
                try:
                    import ctypes

                    class PyBuffer(ctypes.Structure):
                        """CPython's Py_buffer: the bytes of the read-only mmap without a copy."""
                        _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p), ("len", ctypes.c_ssize_t),
                                    ("itemsize", ctypes.c_ssize_t), ("readonly", ctypes.c_int),
                                    ("ndim", ctypes.c_int), ("format", ctypes.c_char_p),
                                    ("shape", ctypes.c_void_p), ("strides", ctypes.c_void_p),
                                    ("suboffsets", ctypes.c_void_p), ("internal", ctypes.c_void_p)]

                    lib = ctypes.CDLL(path)
                    lib.rpf9_abi.argtypes, lib.rpf9_abi.restype = (), ctypes.c_uint32
                    if lib.rpf9_abi() == 1:
                        sz, vp = ctypes.c_size_t, ctypes.c_void_p
                        lib.rpf9_ctx_new.argtypes, lib.rpf9_ctx_new.restype = (), vp
                        lib.rpf9_ctx_free.argtypes, lib.rpf9_ctx_free.restype = (vp,), None
                        lib.rpf9_next.argtypes = (vp, vp, sz, sz, sz, sz, ctypes.c_int, vp)
                        lib.rpf9_next.restype = ctypes.c_int
                        lib.rpf9_all_zero.argtypes = (vp, sz, sz, sz)
                        lib.rpf9_all_zero.restype = ctypes.c_int
                        lib.rpf9_byte_counts.argtypes = (vp, sz, sz, sz, vp, vp)
                        lib.rpf9_byte_counts.restype = ctypes.c_int
                        api = ctypes.pythonapi
                        api.PyObject_GetBuffer.argtypes = (ctypes.py_object, ctypes.POINTER(PyBuffer), ctypes.c_int)
                        api.PyObject_GetBuffer.restype = ctypes.c_int
                        api.PyBuffer_Release.argtypes = (ctypes.POINTER(PyBuffer),)
                        api.PyBuffer_Release.restype = None
                        found = lib, ctypes, PyBuffer
                except (ImportError, OSError, AttributeError):
                    found = None
        _NATIVE.append(found)
    return _NATIVE[0]


def _native_scan(m):
    """A `_NativeScan` over the mmap `m`, or None to run the pure-Python scan."""
    native = _native_lib()
    if native is None:
        return None
    try:
        return _NativeScan(*native, m)
    except (OSError, AttributeError, TypeError, ValueError, BufferError):
        return None


def _next_hit(m, pos, end, last, system_pages):
    """The pure-Python reference of the scan: from `pos`, the next stream the carve keeps.

    (offset, stored size, inflated bytes) for a zlib stream, (offset, size, None) for a Bink 2 video, or False
    when no header is left before `end`.
    """
    while pos < end:
        mt = ZHDR.search(m, pos, end)
        if not mt:
            return False
        s = mt.start()
        if m[s:s + 3] == b"KB2":
            if s + 0x1C > end:   # header cut off by the end of the file or of --limit's window: not a video
                pos = s + 1
                continue
            blen = struct.unpack_from("<I", m, s + 4)[0] + 8
            w, h = struct.unpack_from("<II", m, s + 0x14)
            if not (64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192):
                pos = s + 1
                continue
            return s, blen, None
        d = zlib.decompressobj()
        try:
            chunks, p = [], s
            # Inflate only inside the scan window (all of the file without --limit): a stream that runs past
            # its end is left out, so readable + zero + unread add up to the span, as in story_index (#712).
            while not d.eof and p < end:
                chunk = m[p:min(p + (1 << 20), end)]
                if not chunk:
                    break
                chunks.append(d.decompress(chunk))
                p += len(chunk)   # the slice can be short at EOF; advancing by 1 MiB overran it
                if d.unused_data or d.eof:
                    break
            out = b"".join(chunks) + d.flush()
        except zlib.error:
            pos = s + 1
            continue
        adjacent = s == ((last + 15) & ~15) and not any(m[last:s])
        if not d.eof or len(out) < system_pages.minimum_size(adjacent):
            pos = s + 1
            continue
        return s, p - s - len(d.unused_data), out
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--start", type=offset)
    ap.add_argument("--limit", type=byte_count)
    ap.add_argument("--until-adat", action="store_true",
                    help="end the scan at the first raw AWC (ADAT) bank, e.g. the audio config at the start of "
                         "audio.rpf")
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args(argv)
    below_normal()
    if os.path.isdir(a.archive):
        sys.exit(f"{a.archive}: is a directory, not an RPF9 archive")
    try:
        size = os.path.getsize(a.archive)
    except OSError as e:
        sys.exit(f"{a.archive}: {e.strerror or e}")
    if size == 0:
        sys.exit(f"{a.archive}: empty file")
    if size < 8:
        sys.exit(f"{a.archive}: too short for an RPF9 header ({size} bytes)")
    try:
        with open(a.archive, "rb") as fh:
            hdr = fh.read(8)
    except OSError as e:
        sys.exit(f"{a.archive}: {e.strerror or e}")
    if hdr[:4] != MAGIC:
        sys.exit(f"{a.archive}: not an RPF9 archive (expected the magic {MAGIC.decode()} at offset 0)")
    if a.start is not None and not 0 <= a.start < size:
        sys.exit(f"{a.archive}: --start {a.start} is outside the file (0..{size - 1})")
    if a.start is None:
        base = struct.unpack_from("<I", hdr, 4)[0]
        if base > size:   # base == size is an empty data area, which is fine
            sys.exit(f"{a.archive}: header data offset {base} is past the end of the file ({size} bytes)")
    else:
        base = a.start
    if os.path.exists(a.out) and not os.path.isdir(a.out):
        sys.exit(f"{a.out}: output path is a file, not a directory")
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"{a.out}: cannot create output directory: {e.strerror or e}")
    with open(a.archive, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        pos = base
        end = min(size, pos + a.limit) if a.limit else size
        boundary = None
        if a.until_adat:
            boundary = first_adat_bank(m, pos, end)
            if boundary is None:
                print(f"--until-adat: no AWC bank between {pos:#x} and {end:#x}; scanning all of it", flush=True)
            else:
                end = boundary
                print(f"--until-adat: first AWC bank at {boundary:#x}; scanning {pos:#x}..{boundary:#x}", flush=True)
        rows, last, unread, zero = [], pos, 0, 0
        unread_ranges = []
        system_pages = ModelSystemPages()

        native = _native_scan(m)

        def account(a, b):
            """Count the gap m[a:b] as zero padding, or as an unread range that is also listed."""
            nonlocal zero, unread
            zero_gap = native.all_zero(a, b) if native else None
            if zero_gap is None:
                zero_gap = all_zero(m, a, b)
            if zero_gap:
                zero += b - a
                return
            unread += b - a
            sampled = min(b, a + ENTROPY_SAMPLE) - a
            bits = native.entropy(a, a + sampled) if native else None
            if bits is None:
                bits = entropy(m[a:a + sampled])
            unread_ranges.append({"off": a, "size": b - a, "entropy": round(bits, 3), "sampled": sampled})

        t0 = time.time()
        try:
            while pos < end:
                hit = native.next(pos, end, last, system_pages.remaining == 1) if native else None
                if hit is None:
                    hit = _next_hit(m, pos, end, last, system_pages)
                if hit is False:
                    break
                s, used, out = hit
                if out is None:   # Bink 2 video
                    blen = used
                    w, h = struct.unpack_from("<II", m, s + 0x14)
                    account(last, s)
                    rows.append({"off": s, "stored": blen, "size": blen, "type": "bink2", "w": w, "h": h,
                                 "frames": struct.unpack_from("<I", m, s + 8)[0], "pre": "",
                                 "head": m[s:s + 16].hex()})
                    if not a.index_only:
                        out_file = os.path.join(a.out, f"{s:012x}.bk2")
                        try:
                            with open(out_file, "wb") as f:
                                f.write(m[s:s + blen])
                        except OSError as e:
                            sys.exit(f"{out_file}: cannot write: {e.strerror or e}")
                    last = pos = s + blen
                    system_pages.reset()
                    continue
                adjacent = s == ((last + 15) & ~15) and not any(m[last:s])
                system_pages.accept(out, adjacent=adjacent,
                                    root_framed=(s & 0xFFF) == 0x10 and s >= 16 and m[s - 16:s] == b"\0" * 16)
                account(last, s)
                rows.append({"off": s, "stored": used, "size": len(out), "pre": m[max(0, s - 16):s].hex(),
                             "head": out[:16].hex()})
                if not a.index_only:
                    out_file = os.path.join(a.out, f"{s:012x}.bin")
                    try:
                        with open(out_file, "wb") as f:
                            f.write(out)
                    except OSError as e:
                        sys.exit(f"{out_file}: cannot write: {e.strerror or e}")
                last = pos = s + used
                if len(rows) % 2000 == 0:
                    done = (pos - (a.start or 0)) / max(1, end - (a.start or 0))
                    print(f"[{done:6.1%}] {len(rows)} files, {time.time() - t0:.0f}s", flush=True)
            account(last, end)
        finally:
            if native:
                native.close()
    total = sum(r["stored"] for r in rows)
    bink = sum(r["stored"] for r in rows if r.get("type") == "bink2")
    span = end - base
    summary = {"archive": a.archive, "files": len(rows), "compressed_bytes": total,
               "inflated_bytes": sum(r["size"] for r in rows), "bink2_bytes": bink,
               "bink2_files": sum(1 for r in rows if r.get("type") == "bink2"),
               "zero_padding": zero, "unread_bytes": unread, "unread_ranges": len(unread_ranges),
               "span": span}
    if a.until_adat:
        summary["first_adat_bank"] = boundary   # None: no bank in the window, the whole window was scanned
    index_path = os.path.join(a.out, "index.json")
    try:
        with open(index_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": rows, "unread": unread_ranges}, f)
    except OSError as e:
        sys.exit(f"{index_path}: cannot write: {e.strerror or e}")
    print(json.dumps(summary, indent=1))
    span = max(1, span)   # an empty data area (or --start at the end) has nothing to divide by
    print(f"readable: {total / span:.1%}, zero padding {zero / span:.1%}, unread {unread / span:.1%}")
    if unread_ranges:
        print(f"unread: {len(unread_ranges)} range(s), largest {max(r['size'] for r in unread_ranges)} bytes "
              f"(listed in index.json)")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
