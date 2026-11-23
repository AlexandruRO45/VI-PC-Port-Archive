"""Optional Rust kernels for the offline BCPack decoder."""
import ctypes
from functools import lru_cache
from pathlib import Path
import sys


CRATE = Path(__file__).with_name("bcpack_calibration")
LIBRARY = CRATE / "target/release" / {
    "win32": "bcpack_calibration.dll",
    "darwin": "libbcpack_calibration.dylib",
}.get(sys.platform, "libbcpack_calibration.so")

_BYTE_POINTER = ctypes.c_char_p
_BUFFER = ctypes.c_void_p
_SIZE = ctypes.c_size_t
_MAX_NATIVE_BYTES = 16 << 20


@lru_cache(maxsize=1)
def library():
    return ctypes.CDLL(str(LIBRARY))


@lru_cache(maxsize=None)
def kernel(name, result, *args):
    try:
        func = getattr(library(), name)
    except (OSError, AttributeError):
        return None
    func.argtypes = args
    func.restype = result
    return func


def unpack_lanes(raw, widths, count):
    if not 0 < count * 8 <= _MAX_NATIVE_BYTES:
        return None
    decode = kernel("unpack_lanes", ctypes.c_ssize_t,
                    _BYTE_POINTER, _SIZE, _BYTE_POINTER, _SIZE, _SIZE, _BUFFER)
    if decode is None:
        return None
    output = ctypes.create_string_buffer(count * len(widths))
    used = decode(raw, len(raw), bytes(widths), len(widths), count, output)
    if used < 0:
        return None
    lanes = output.raw
    return [lanes[i:i + count] for i in range(0, len(lanes), count)], used


@lru_cache(maxsize=1)
def strict_bc1_tables():
    from bcpack_tables import BC1_HISTORY_CONFIRMED, BC1_QUADRANT

    endpoints = ((ctypes.c_int32 * 6) * 256)()
    for index, row in enumerate(endpoints):
        row[:] = BC1_HISTORY_CONFIRMED.get(index, (-1,) * 6)
    selectors = ((ctypes.c_uint32 * 256) * 4)()
    for row, (low, high) in zip(selectors, ((0, 8), (4, 12), (16, 24), (20, 28))):
        row[:] = [(q & 15) << low | (q >> 4) << high for q in BC1_QUADRANT]
    return endpoints, selectors


def decode_strict_bc1(raw, size):
    if not 0 < size <= _MAX_NATIVE_BYTES or size % 8:
        return None
    decode = kernel("decode_strict_bc1", ctypes.c_bool,
                    _BYTE_POINTER, _SIZE, _SIZE, _BUFFER, _BUFFER, _BUFFER)
    if decode is None:
        return None
    output = ctypes.create_string_buffer(size)
    if decode(raw, len(raw), size // 8, *strict_bc1_tables(), output):
        return output.raw
    return None


def decode_strict_mode6(raw, size, signature):
    if not 0 < size <= _MAX_NATIVE_BYTES or size % 16:
        return None
    decode = kernel("decode_strict_mode6", ctypes.c_bool,
                    _BYTE_POINTER, _SIZE, _SIZE, ctypes.c_uint8, ctypes.c_uint8, _BUFFER)
    if decode is None:
        return None
    output = ctypes.create_string_buffer(size)
    if decode(raw, len(raw), size // 16, *signature, output):
        return output.raw
    return None


def rebuild_calibrated_bc1(symbols, endpoints, selectors):
    if symbols.dtype.name != "uint8" or symbols.ndim != 2 or symbols.shape[1] != 12:
        return None
    if not symbols.flags.c_contiguous:
        return None
    decode = kernel("rebuild_calibrated_bc1", ctypes.c_bool,
                    _BUFFER, _SIZE, _BUFFER, _BUFFER, _BUFFER)
    if decode is None:
        return None
    count = len(symbols)
    output = ctypes.create_string_buffer(count * 8)
    if decode(symbols.ctypes.data, count, endpoints.ctypes.data, selectors.ctypes.data, output):
        return output.raw
    raise ValueError("Missing BC1 endpoint history")
