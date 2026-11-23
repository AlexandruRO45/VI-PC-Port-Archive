"""ctypes adapter for the optional rsc8_audio kernels in tools/audio_native.

Every function returns None when the library is missing, fails to load, reports another ABI, declines the call,
or LEONIDA_AUDIO_NATIVE=0 is set; rsc8_audio.py then runs its pure-Python reference. Build the library with
``cargo build --release --manifest-path tools/audio_native/Cargo.toml``.

A call costs about a microsecond of ctypes overhead, so a table shorter than the ``MIN_*`` sizes below is left to
the reference, which is as fast there; the result is the same either way.
"""
import os
import struct
import sys
from itertools import islice

ABI = 1
U64 = (1 << 64) - 1
DONE, INVALID = 0, 1
_NATIVE = []

# Below these sizes (bytes) the Python loop is about as fast as the call overhead.
MIN_HEADER_BYTES = 128
MIN_GRID20_BYTES = 8 + 20 * 8
MIN_INDEXRUN_BYTES = 64
MIN_MARKER_BYTES = 16 * 8


def library_path():
    name = {"win32": "audio_native.dll", "darwin": "libaudio_native.dylib"}.get(sys.platform, "libaudio_native.so")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_native", "target", "release", name)


def _load():
    path = library_path()
    if not os.path.isfile(path):
        return None
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        if lib.audio_native_abi() != ABI:
            return None
        header = lib.audio_adat_header_v1
        grid = lib.audio_grid20_v1
        runs = lib.audio_index_runs_v1
        marks = lib.audio_markers_v1
        has_value = lib.audio_marker_has_value_v1
    except (ImportError, OSError, AttributeError):
        return None
    c_u32, c_u64, c_size, c_ptr = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_char_p
    p_u32, p_u64 = ctypes.POINTER(c_u32), ctypes.POINTER(c_u64)
    header.argtypes = (c_ptr, c_size, c_u64, c_u64, c_u64, c_u64, p_u32, c_size, p_u32, c_size, p_u64)
    grid.argtypes = (c_ptr, c_size, p_u64)
    runs.argtypes = (c_ptr, c_size, c_u64, p_u32, c_size, p_u64)
    marks.argtypes = (c_ptr, c_size, p_u32, p_u64, c_size, p_u64)
    has_value.argtypes = (c_ptr, c_size, c_u32, c_u32)
    for fn in (header, grid, runs, marks, has_value):
        fn.restype = ctypes.c_int32
    return ctypes, header, grid, runs, marks, has_value


def native():
    """The loaded kernels, or None when they are unavailable or switched off."""
    if os.environ.get("LEONIDA_AUDIO_NATIVE", "1") == "0" or sys.implementation.name != "cpython":
        return None
    if not _NATIVE:
        _NATIVE.append(_load())
    return _NATIVE[0]


def _word(v):
    return type(v) is int and 0 <= v <= U64


def adat_header(hb, bank_offset, file_size, max_streams, max_header):
    """None to fall back, else ``(ok, bank)`` with ``bank`` as parse_adat_header_bytes() returns it (None if not ok)."""
    if type(hb) is not bytes or len(hb) < MIN_HEADER_BYTES or not (
            _word(bank_offset) and _word(file_size) and _word(max_streams) and _word(max_header)):
        return None
    lib = native()
    if lib is None:
        return None
    ctypes, header = lib[0], lib[1]
    count, hdr_sz = struct.unpack_from("<II", hb, 8)
    if count > max_streams or hdr_sz > len(hb) or hdr_sz > max_header:
        return False, None
    # The kernel writes into bytearrays so the words come back as tuples through struct at C speed.
    st_buf = bytearray(4 * max(2 * count, 1))
    tag_cap = 3 * (hdr_sz // 8)
    tag_buf = bytearray(4 * max(tag_cap, 1))
    head = (ctypes.c_uint64 * 5)()
    rc = header(hb, len(hb), bank_offset, file_size, max_streams, max_header,
                (ctypes.c_uint32 * (len(st_buf) // 4)).from_buffer(st_buf), 2 * count,
                (ctypes.c_uint32 * (len(tag_buf) // 4)).from_buffer(tag_buf), tag_cap, head)
    if rc == INVALID:
        return False, None
    if rc != DONE:
        return None
    flags, count, hdr_sz, end, ntags = head
    triples = struct.iter_unpack("=III", memoryview(tag_buf)[: 12 * ntags])
    entries = [(sid, list(islice(triples, tc))) for sid, tc in struct.iter_unpack("=II", memoryview(st_buf)[: 8 * count])]
    return True, {
        "offset": bank_offset,
        "flags": flags,
        "count": count,
        "header_size": hdr_sz,
        "end": end,
        "entries": entries,
    }


def grid20(raw):
    """None, or (position_first, position_last, position_increasing, p_multiple_of_32) of a size-checked table."""
    if type(raw) is not bytes or len(raw) < MIN_GRID20_BYTES:
        return None
    lib = native()
    if lib is None:
        return None
    ctypes, grid = lib[0], lib[2]
    out = (ctypes.c_uint64 * 5)()
    if grid(raw, len(raw), out) != DONE:
        return None
    has, first, last, increasing, p_ok = out
    return (first if has else None), (last if has else None), bool(increasing), bool(p_ok)


def index_runs(raw, max_groups):
    """None to fall back, False when the reference finds the groups overrun the table (size_ok False), else
    (groups, group_count_matches, marker_ok, sorted hashes, values, indices_ascending, index_max)."""
    if type(raw) is not bytes or len(raw) < MIN_INDEXRUN_BYTES or not _word(max_groups):
        return None
    lib = native()
    if lib is None:
        return None
    ctypes, runs = lib[0], lib[3]
    cap = (len(raw) // 4 - 1) // 3 + 1
    hashes = (ctypes.c_uint32 * cap)()
    out = (ctypes.c_uint64 * 8)()
    rc = runs(raw, len(raw), max_groups, hashes, cap, out)
    if rc == INVALID:
        return False
    if rc != DONE:
        return None
    groups, matches, marker_ok, nhash, values, ascending, has_max, index_max = out
    return (groups, bool(matches), bool(marker_ok), hashes[:nhash], values, bool(ascending),
            index_max if has_max else None)


def markers(raw):
    """None, or (names dict in first-seen order, offsets_ascending, offset_max, extra_max) of a size-checked tag."""
    if type(raw) is not bytes or len(raw) < MIN_MARKER_BYTES:
        return None
    lib = native()
    if lib is None:
        return None
    ctypes, marks = lib[0], lib[4]
    cap = len(raw) // 16
    names = (ctypes.c_uint32 * cap)()
    counts = (ctypes.c_uint64 * cap)()
    out = (ctypes.c_uint64 * 4)()
    if marks(raw, len(raw), names, counts, cap, out) != DONE:
        return None
    k, ascending, offset_max, extra_max = out
    return dict(zip(names[:k], counts[:k])), bool(ascending), offset_max, extra_max


def marker_has_value(raw, a, b):
    """None, or whether a whole 16-byte record's value word is `a` or `b`."""
    if type(raw) is not bytes or len(raw) < MIN_MARKER_BYTES:
        return None
    lib = native()
    if lib is None:
        return None
    return bool(lib[5](raw, len(raw), a, b))
