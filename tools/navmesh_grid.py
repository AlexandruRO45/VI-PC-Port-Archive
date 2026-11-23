"""Decode and verify Band-3 ``grid`` octree leaf payloads and coordinate mapping in ``navmeshes.rpf``.

Each ``grid`` resource stores a 16-way SwissTable of sentinel-bit linear octree keys (``1`` = root,
``(parent << 3) | octant``) and a ``u32[N]`` payload array:

- Octant bits map ``(X, Y, Z)`` inverted relative to ``origin + extent``:
  ``bit 1`` = ``~X``, ``bit 0`` = ``~Y``, ``bit 2`` = ``~Z``. When ``extent > 128``, this places the
  ``128 m x 128 m`` sector at ``[0, 128)`` from ``(origin_x, origin_y)`` after inversion.
- Inner nodes (keys with at least one child) always hold ``0x0000ff00``.
- Leaf nodes hold ``0x0000ff00`` (empty/uniform), ``0x01``-low-byte volume cells (bitwise OR
  ``0x00060f01``: ``bits 8..11`` in ``{0, 5, 7, 9, B, D, F}`` with bit 11 requiring ``z0 >= 0.0``,
  ``bits 17..18`` in ``0..3``), or ``0x03``-low-byte surface cells (bitwise OR ``0x00007903``:
  ``bit 8`` flag, ``bits 11..14`` 4-bit sub-cell Z height ``0..15``, cell size ``extent >> depth`` in
  ``{4, 8}`` m).

The tool reads headers, tables and counts only. It writes aggregate metadata only: no payload bytes,
names or keys.

Usage:
    python -I navmesh_grid.py <archive.rpf|dir> [--out FILE] [--start N] [--limit BYTES] [--quiet]
"""
import argparse
import json
import mmap
import os
import struct
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import navmesh_decode as nd  # noqa: E402

GRID_INNER = 0x0000FF00
MASK_01 = 0x00060F01
MASK_03 = 0x00007903


def grid_key_coords(key):
    """Decode a sentinel-bit linear octree key into ``(depth, ix, iy, iz)`` with inverted axes.

    Octant bit 1 = ``~X``, bit 0 = ``~Y``, bit 2 = ``~Z``.
    """
    if key <= 0 or (key.bit_length() - 1) % 3:
        raise ValueError("invalid linear octree key: %r" % (key,))
    depth = (key.bit_length() - 1) // 3
    ix = iy = iz = 0
    for level in range(depth):
        octant = (key >> (3 * (depth - 1 - level))) & 7
        ix = (ix << 1) | (((octant >> 1) & 1) ^ 1)
        iy = (iy << 1) | ((octant & 1) ^ 1)
        iz = (iz << 1) | (((octant >> 2) & 1) ^ 1)
    return depth, ix, iy, iz


def grid_cell_bounds(key, extent, origin):
    """Return ``((x0, y0, z0), (x1, y1, z1))`` world bounds for ``key`` inside ``origin + extent``."""
    depth, ix, iy, iz = grid_key_coords(key)
    size = extent / (1 << depth)
    ox, oy, oz = origin
    x0, y0, z0 = ox + ix * size, oy + iy * size, oz + iz * size
    return (x0, y0, z0), (x0 + size, y0 + size, z0 + size)


def _grid_slots(b, groups):
    """Return ``{key: value}`` for every occupied SwissTable slot."""
    keys = {}
    for g in range(groups):
        base = 0x60 + nd.GRID_GROUP * g
        for k, c in enumerate(b[base:base + 16]):
            if c < 0x80:
                key = nd._u64(b, base + 16 + 16 * k)
                val = nd._u64(b, base + 16 + 16 * k + 8)
                keys[key] = val
    return keys


def _grid_octree_ok(keys):
    if 1 not in keys:
        return False
    for key in keys:
        if key == 0 or (key.bit_length() - 1) % 3 or (key != 1 and (key >> 3) not in keys):
            return False
    return True


def verify_grid_resource(b, stats):
    """Validate one Band-3 ``grid`` resource's octree leaves and update ``stats``."""
    groups, entries = nd._u32(b, 0x10), nd._u32(b, 0x14)
    payload = nd.offset_of(nd._u64(b, 0x20))
    ex = struct.unpack_from("<3I", b, 0x28)
    origin = nd._f32s(b, 0x40, 3)

    chk = {
        "table_in_bounds": 0x60 + nd.GRID_GROUP * groups <= len(b),
        "payload_in_bounds": payload is not None and payload + 4 * entries <= len(b),
        "cubic_extent": ex[0] == ex[1] == ex[2] and ex[0] > 0,
        "origin_finite": origin is not None,
    }
    if not all(chk.values()):
        return chk

    ctrl = b"".join(b[0x60 + nd.GRID_GROUP * g:0x60 + nd.GRID_GROUP * g + 16] for g in range(groups))
    valid_ctrl = max(ctrl) <= 0x80 if ctrl else True
    used = len(ctrl) - ctrl.count(0x80) if valid_ctrl else -1
    chk["values_bijective"] = valid_ctrl and used == entries and nd._grid_values(b, groups, entries)
    if not chk["values_bijective"]:
        return chk

    keys = _grid_slots(b, groups)
    chk["keys_octree"] = len(keys) == entries and _grid_octree_ok(keys)
    if not chk["keys_octree"]:
        return chk

    extent = ex[0]
    oz = origin[2]
    inner = {k >> 3 for k in keys if k != 1}
    inner_ok = True
    masks_ok = True
    cell_03_ok = True
    bit11_z0_ok = True

    stats["grids"] += 1
    stats["entries_total"] += entries
    stats["inner_nodes"] += len(inner)
    stats["extents"][str(extent)] += 1

    for k, val_idx in keys.items():
        word = nd._u32(b, payload + 4 * val_idx)
        if k in inner:
            if word != GRID_INNER:
                inner_ok = False
            continue
        stats["leaves_total"] += 1
        low = word & 0xFF
        stats["leaf_low_bytes"]["0x%02x" % low] += 1
        if word == GRID_INNER:
            continue
        if low == 0x01:
            if word & ~MASK_01:
                masks_ok = False
            b8_11 = (word >> 8) & 0xF
            b17_18 = (word >> 17) & 3
            stats["leaf_01_b8_11"]["0x%x" % b8_11] += 1
            stats["leaf_01_b17_18"][str(b17_18)] += 1
            if (b8_11 & 0x8) and oz is not None:
                (_, _, z0), _ = grid_cell_bounds(k, extent, origin)
                if z0 < 0.0:
                    bit11_z0_ok = False
        elif low == 0x03:
            if word & ~MASK_03:
                masks_ok = False
            depth = (k.bit_length() - 1) // 3
            cell_sz = extent >> depth
            if cell_sz not in (4, 8):
                cell_03_ok = False
            stats["leaf_03_cell_size"][str(cell_sz)] += 1
            stats["leaf_03_b8"][str((word >> 8) & 1)] += 1
            stats["leaf_03_b11_14"][str((word >> 11) & 0xF)] += 1
        else:
            masks_ok = False

    chk["inner_payload_ff00"] = inner_ok
    chk["leaf_bitmasks_ok"] = masks_ok
    chk["leaf_03_cell_size"] = cell_03_ok
    chk["leaf_01_bit11_z0"] = bit11_z0_ok
    return chk


def new_stats():
    return {
        "grids": 0,
        "entries_total": 0,
        "inner_nodes": 0,
        "leaves_total": 0,
        "extents": Counter(),
        "leaf_low_bytes": Counter(),
        "leaf_01_b8_11": Counter(),
        "leaf_01_b17_18": Counter(),
        "leaf_03_b8": Counter(),
        "leaf_03_b11_14": Counter(),
        "leaf_03_cell_size": Counter(),
        "checks_failed": Counter(),
    }


def _walk_resources(stream, end=None):
    pending = next(stream, None)
    while pending is not None:
        s, _, head = pending
        if end is not None and s >= end:
            break
        pending = next(stream, None)
        if len(head) < nd.MIN_STREAM:
            continue
        cls = nd.classify(head)
        parts = 1
        buf = bytearray(head)
        while pending is not None and nd.wants_more(cls, buf, parts):
            if len(buf) + len(pending[2]) > nd.MAX_INFLATE:
                break
            buf.extend(pending[2])
            parts += 1
            pending = next(stream, None)
        yield s, cls, bytes(buf)


def scan(path, start=None, limit=None):
    """Scan ``path`` (RPF9 archive or carved directory) and return the Band-3 ``grid`` summary."""
    t0 = time.time()
    stats = new_stats()
    if os.path.isdir(path):
        for _, cls, data in _walk_resources(nd.walk_dir(path)):
            if cls == "grid":
                chk = verify_grid_resource(data, stats)
                for k, ok in chk.items():
                    if not ok:
                        stats["checks_failed"][k] += 1
    else:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size < 8:
                raise ValueError("%s is %d bytes, too small for an RPF header" % (path, size))
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
                if m[1:4] != b"FPR" or m[0:1] not in (b"7", b"8", b"9"):
                    raise ValueError("%s: not an RPF7/8/9 archive" % path)
                data_start = struct.unpack_from("<I", m, 4)[0]
                if start is None:
                    start = data_start
                if not 0 <= start < size:
                    raise ValueError("start %d is outside %s (%d bytes)" % (start, path, size))
                end = size if limit is None else min(size, start + limit)
                for _, cls, data in _walk_resources(nd.walk(m, start, size), end=end):
                    if cls == "grid":
                        chk = verify_grid_resource(data, stats)
                        for k, ok in chk.items():
                            if not ok:
                                stats["checks_failed"][k] += 1
    out = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Decode and verify Band-3 grid octree leaf payloads.")
    ap.add_argument("input", help="navmeshes.rpf or carved stream directory")
    ap.add_argument("--out", help="JSON summary output path")
    ap.add_argument("--start", type=int, default=None, help="first byte offset in archive")
    ap.add_argument("--limit", type=int, default=None, help="maximum bytes to scan from --start")
    ap.add_argument("--quiet", action="store_true", help="suppress stdout summary")
    args = ap.parse_args(argv)

    if args.start is not None and args.start < 0:
        ap.error("--start must be >= 0")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    if not os.path.exists(args.input):
        ap.error("input path does not exist: %s" % args.input)
    if args.out and os.path.isdir(args.out):
        ap.error("--out must name a file, not a directory: %s" % args.out)

    summary = scan(args.input, start=args.start, limit=args.limit)
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["checks_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
