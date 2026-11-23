"""ctypes adapter for the optional navmesh_decode --deep kernels in tools/navmesh_native.

Every function returns None when the library is missing, fails to load, reports another ABI, declines the call,
or LEONIDA_NAVMESH_NATIVE=0 is set; navmesh_decode.py then runs its pure-Python reference. Build the library
with ``cargo build --release --manifest-path tools/navmesh_native/Cargo.toml``.
"""
import os
import sys

ABI = 1
U64 = (1 << 64) - 1
_NATIVE = []
GRID_KINDS = ("ff00", "low_byte_01", "low_byte_03", "other")


def library_path():
    name = {"win32": "navmesh_native.dll", "darwin": "libnavmesh_native.dylib"}.get(
        sys.platform, "libnavmesh_native.so")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "navmesh_native", "target", "release", name)


def _load():
    path = library_path()
    if not os.path.isfile(path):
        return None
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        if lib.navmesh_native_abi() != ABI:
            return None
        quadtree = lib.navmesh_quadtree_v1
        chain = lib.navmesh_poly_chain_v1
        volumes = lib.navmesh_volumes_v1
        grid = lib.navmesh_grid_deep_v1
    except (ImportError, OSError, AttributeError):
        return None
    c_u64, c_size, c_ptr = ctypes.c_uint64, ctypes.c_size_t, ctypes.c_char_p
    p_u64 = ctypes.POINTER(c_u64)
    quadtree.argtypes = (c_ptr, c_size, c_u64, c_u64, c_u64, c_u64, p_u64)
    chain.argtypes = (c_ptr, c_size, p_u64, c_size, c_u64, c_u64, ctypes.c_uint32, p_u64)
    volumes.argtypes = (c_ptr, c_size, c_u64, c_u64, p_u64)
    grid.argtypes = (c_ptr, c_size, c_u64, c_u64, ctypes.c_int32, c_u64, p_u64)
    for fn in (quadtree, chain, volumes, grid):
        fn.restype = ctypes.c_int32
    return ctypes, quadtree, chain, volumes, grid


def native():
    """The loaded kernels, or None when they are unavailable or switched off."""
    if os.environ.get("LEONIDA_NAVMESH_NATIVE", "1") == "0" or sys.implementation.name != "cpython":
        return None
    if not _NATIVE:
        _NATIVE.append(_load())
    return _NATIVE[0]


def _words(*values):
    return all(type(v) is int and 0 <= v <= U64 for v in values)


def walk_quadtree(b, root, n_polys, max_depth, max_nodes):
    """(ok, nodes, leaves, max_depth) as _walk_quadtree returns them, or None."""
    lib = native()
    if lib is None or type(b) is not bytes or not _words(root, n_polys, max_depth, max_nodes):
        return None
    ctypes, quadtree = lib[0], lib[1]
    out = (ctypes.c_uint64 * 4)()
    if quadtree(b, len(b), root, n_polys, max_depth, max_nodes, out):
        return None
    return bool(out[0]), out[1], out[2], out[3]


def poly_chain(b, blocks, owner, pool, water_mask):
    """(chain_ok, has_water) as _poly_chain returns them, or None."""
    lib = native()
    if lib is None or type(b) is not bytes or not _words(owner, pool) or not 0 <= water_mask <= 0xFFFFFFFF:
        return None
    flat = []
    for o, n in blocks:
        if not _words(o, n):
            return None
        flat += (o, n)
    ctypes, chain = lib[0], lib[2]
    pairs = (ctypes.c_uint64 * max(len(flat), 1))(*flat)
    out = (ctypes.c_uint64 * 2)()
    if chain(b, len(b), pairs, len(flat) // 2, owner, pool, water_mask, out):
        return None
    return bool(out[0]), bool(out[1])


def _ordered(out, at):
    return [(out[at + 1 + 2 * i], out[at + 2 + 2 * i]) for i in range(out[at])]


def walk_volumes(b, vol, dc):
    """None, or (ok, pairs, [(flag, count)], [(vertex_count, count)]) in first-seen order."""
    lib = native()
    if lib is None or type(b) is not bytes or not _words(vol, dc):
        return None
    ctypes, volumes = lib[0], lib[3]
    out = (ctypes.c_uint64 * 16)()
    if volumes(b, len(b), vol, dc, out):
        return None
    return bool(out[0]), out[1], _ordered(out, 2), _ordered(out, 9)


def grid_deep(b, groups, entries, payload):
    """None, or (values_bijective, keys_octree, inner_payload_ff00, inner_nodes, [(depth, n)], [(kind, n)]).

    keys_octree and inner_payload_ff00 are None when the reference would not compute them; `payload` is the
    payload offset when it is in bounds, else None."""
    lib = native()
    if lib is None or type(b) is not bytes or not _words(groups, entries) or not (
            payload is None or _words(payload)):
        return None
    ctypes, grid = lib[0], lib[4]
    out = (ctypes.c_uint64 * 64)()
    if grid(b, len(b), groups, entries, payload is not None, payload or 0, out):
        return None
    state = (None, False, True)
    kinds = [(GRID_KINDS[k], n) for k, n in _ordered(out, 49)]
    return bool(out[0]), state[out[1]], state[out[2]], out[3], _ordered(out, 4), kinds
