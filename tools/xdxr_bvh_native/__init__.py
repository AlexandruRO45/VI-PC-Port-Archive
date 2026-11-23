"""Optional ctypes binding for the Rust traversal of tools/xdxr_bvh.py (``trace`` and ``brute_force``).

Each ``try_*`` returns ``(True, result)`` when the native library produced the result, and ``(False, None)`` when the
caller must run its Python path: the library is missing or switched off (``LEONIDA_XDXR_BVH_NATIVE=0``), an input
is not one the native path takes as is (a Memory subclass, a generator, a 4-component vector, an int past 2**53),
or the library refused (a read outside every buffer, any other error, a leaf with internal flag bits). A refusal
changes nothing, so the Python path then returns its own result or raises its own exception.
"""

import functools
import math
import os
import struct
import sys
import weakref
from itertools import chain

try:
    import ctypes
except ImportError:                 # an interpreter built without ctypes: the Python path runs
    ctypes = None

_ENV = "LEONIDA_XDXR_BVH_NATIVE"
_EXACT = 2 ** 53            # ints up to this convert to float64 exactly, as Python's mixed arithmetic does
_ADDRESS = 2 ** 62          # addresses and biases the native side holds in an i64 without overflow
_REFUSED = (False, None)
_SEQ = frozenset((list, tuple))
_THREE = frozenset((3,))


@functools.lru_cache(maxsize=1)
def _load():
    """(trace, brute_force) entry points of the built library, or None."""
    if sys.platform == "win32":
        name = "xdxr_bvh_native.dll"
    elif sys.platform == "darwin":
        name = "libxdxr_bvh_native.dylib"
    else:
        name = "libxdxr_bvh_native.so"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target", "release", name)
    try:
        if ctypes is None or ctypes.sizeof(ctypes.c_size_t) != 8:
            return None
        double, u64 = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64)
        lib = ctypes.CDLL(path)
        trace = lib.xdxr_trace_v1
        brute = lib.xdxr_brute_force_v1
        trace.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_char_p),
                          ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int64, double,
                          ctypes.c_uint32, ctypes.c_int64, double, u64]
        trace.restype = ctypes.c_int
        brute.argtypes = [ctypes.c_size_t, double, ctypes.POINTER(ctypes.c_size_t),
                          ctypes.POINTER(ctypes.c_void_p), ctypes.c_double, double, u64]
        brute.restype = ctypes.c_int
    except (ImportError, OSError, AttributeError):
        return None
    return trace, brute


def enabled():
    return os.environ.get(_ENV) != "0" and _load() is not None


def _scalar(x):
    """A float or exactly convertible int, or None."""
    t = type(x)
    if t is float:
        return x
    if t is int and -_EXACT <= x <= _EXACT:
        return float(x)
    return None


def _vec3(v):
    if type(v) not in _SEQ or len(v) != 3:
        return None
    out = [_scalar(c) for c in v]
    return None if None in out else out


def _address(x):
    return type(x) is int and -_ADDRESS <= x < _ADDRESS


# Memory -> (the region tuples the arrays were built from, bases, pointers, lengths). The tuples keep the bytes the
# pointers refer to alive; a changed region list builds new arrays.
_regions = weakref.WeakKeyDictionary()


def _region_arrays(mem):
    regions = mem.regions
    if type(regions) is not list:
        return None
    cached = _regions.get(mem)
    if cached is not None:
        snap = cached[0]
        if len(snap) == len(regions) and all(a is b for a, b in zip(snap, regions)):
            return cached[1:]
    snap = tuple(regions)
    for r in snap:
        if type(r) is not tuple or len(r) != 2 or not _address(r[0]) or type(r[1]) is not bytes:
            return None
    n = len(snap)
    arrays = ((ctypes.c_int64 * n)(*[r[0] for r in snap]), (ctypes.c_char_p * n)(*[r[1] for r in snap]),
              (ctypes.c_size_t * n)(*[len(r[1]) for r in snap]))
    _regions[mem] = (snap,) + arrays
    return arrays


def try_trace(mem, bvh_va, org, d, tmax, mask, bias, stats, memory_class):
    if os.environ.get(_ENV) == "0":
        return _REFUSED
    lib = _load()
    if lib is None:
        return _REFUSED
    try:
        # a subclass or an instance with its own read/u32/f32 keeps the Python path
        if type(mem) is not memory_class or any(k in vars(mem) for k in ("read", "u32", "f32")):
            return _REFUSED
        if stats is not None and (type(stats) is not dict or type(stats.get("steps")) is not int):
            return _REFUSED
        o, v, tm = _vec3(org), _vec3(d), _scalar(tmax)
        if o is None or v is None or tm is None or type(mask) is not int or not _address(bvh_va) \
                or not _address(bias):
            return _REFUSED
        arrays = _region_arrays(mem)
        if arrays is None:
            return _REFUSED
        bases, ptrs, lens = arrays
        ray = (ctypes.c_double * 7)(*o, *v, tm)
        out_f = (ctypes.c_double * 3)()
        out_i = (ctypes.c_uint64 * 8)()
        if lib[0](bases, ptrs, lens, len(bases), bvh_va, ray, mask & 0xFF, bias, out_f, out_i) != 0:
            return _REFUSED
    except Exception:       # noqa: BLE001 - the Python rerun raises its own exception
        return _REFUSED
    if stats is not None:
        stats["steps"] += out_i[7]
    if not out_i[0]:
        return True, None
    return True, {"t": out_f[0], "instance": out_i[2] if out_i[1] else None, "geometry": out_i[3],
                  "primitive": out_i[4], "leaf": out_i[5], "bary": (out_f[1], out_f[2]), "front": bool(out_i[6])}


# id(triangle list) -> (the list, a copy of its nesting, the signs of its zero coordinates, the float32 array). The
# copy holds the same coordinate objects, so `tris == copy` is a C-speed check that nothing in the list changed
# except, possibly, a zero's sign (0.0 == -0.0), which the stored signs catch. The entry keeps the list alive, so its
# id is not reused while cached.
_TRI_CACHE = {}
_TRI_CACHE_SIZE = 32
_MAX_ZEROS = 64


def _packed(tris):
    """A ctypes float32 array of the triangles' coordinates (f32() of each), None for no triangles, or a refusal
    (False) for anything brute_force would not read as three points of three coordinates."""
    entry = _TRI_CACHE.get(id(tris))
    if entry is not None and entry[0] is tris and tris == entry[1] and \
            all(math.copysign(1.0, tris[a][b][c]) == sign for a, b, c, sign in entry[2]):
        return entry[3]
    if not _SEQ.issuperset(map(type, tris)) or not _THREE.issuperset(map(len, tris)):
        return False
    points = list(chain.from_iterable(tris))
    if not _SEQ.issuperset(map(type, points)) or not _THREE.issuperset(map(len, points)):
        return False
    flat = list(chain.from_iterable(points))
    n = len(flat)
    if n != 9 * len(tris):
        return False
    # struct "<f" is the conversion f32() makes, with the same OverflowError and TypeError
    buf = (ctypes.c_float * n).from_buffer_copy(struct.pack(f"<{n}f", *flat)) if n else None
    if flat.count(0.0) <= _MAX_ZEROS:
        zeros, i = [], -1
        for _ in range(flat.count(0.0)):
            i = flat.index(0.0, i + 1)
            zeros.append((i // 9, i // 3 % 3, i % 3, math.copysign(1.0, flat[i])))
        if len(_TRI_CACHE) >= _TRI_CACHE_SIZE:
            del _TRI_CACHE[next(iter(_TRI_CACHE))]
        _TRI_CACHE[id(tris)] = (tris, type(tris)(type(t)(p[:] for p in t) for t in tris), zeros, buf)
    return buf


def try_brute_force(scene, org, d, tmax, mask, invert_affine, transform_point, transform_dir):
    if os.environ.get(_ENV) == "0":
        return _REFUSED
    lib = _load()
    if lib is None:
        return _REFUSED
    try:
        if type(scene) not in _SEQ:
            return _REFUSED
        tm = _scalar(tmax)
        if tm is None:
            return _REFUSED
        picked, rays, counts, bufs = [], [], [], []
        # the same per-instance steps as brute_force, so whatever they raise is raised again by the Python rerun
        for idx, entry in enumerate(scene):
            if type(entry) not in _SEQ:                 # unpacking an iterator would use it up
                return _REFUSED
            m, imask, tris = entry
            if m is not None and not imask & mask:
                continue
            if m is None:
                lo, ld = org, d
            else:
                inv = invert_affine(m)
                lo, ld = transform_point(inv, org), transform_dir(inv, d)
            lo, ld = _vec3(lo), _vec3(ld)
            if lo is None or ld is None or type(tris) not in _SEQ:
                return _REFUSED
            buf = _packed(tris)
            if buf is False:
                return _REFUSED
            picked.append(idx if m is not None else None)
            rays += lo
            rays += ld
            counts.append(len(tris))
            bufs.append(buf)
        k = len(counts)
        ray_buf = (ctypes.c_double * len(rays))(*rays) if rays else None
        count_buf = (ctypes.c_size_t * k)(*counts) if k else None
        ptr_buf = (ctypes.c_void_p * k)(*[ctypes.addressof(b) if b is not None else None for b in bufs]) if k \
            else None
        out_f = (ctypes.c_double * 3)()
        out_i = (ctypes.c_uint64 * 3)()
        if lib[1](k, ray_buf, count_buf, ptr_buf, tm, out_f, out_i) != 0:
            return _REFUSED
    except Exception:       # noqa: BLE001 - the Python rerun raises its own exception
        return _REFUSED
    if not out_i[0]:
        return True, None
    return True, {"t": out_f[0], "instance": picked[out_i[1]], "primitive": out_i[2],
                  "bary": (out_f[1], out_f[2])}
