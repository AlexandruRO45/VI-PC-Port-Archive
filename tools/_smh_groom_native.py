"""Optional ctypes bridge for the CPU-only groom geometry and blend kernels."""
from __future__ import annotations

import array
import ctypes
import math
import os
import sys
from functools import lru_cache
from itertools import chain
from pathlib import Path
from struct import Struct
from typing import NamedTuple

_MODES = {"curves": 1, "line-strip": 2, "ribbons": 3, "tubes": 4}
_XYZ = Struct("=ddd")
_STATS = Struct("=dddd")


@lru_cache(maxsize=1)
def _load() -> ctypes.CDLL | None:
    if sys.platform == "win32":
        name = "smh_groom_native.dll"
    elif sys.platform == "darwin":
        name = "libsmh_groom_native.dylib"
    else:
        name = "libsmh_groom_native.so"
    path = Path(__file__).with_name("smh_groom_native") / "target" / "release" / name
    try:
        lib = ctypes.CDLL(str(path))
        lib.smh_groom_abi_version.argtypes = []
        lib.smh_groom_abi_version.restype = ctypes.c_uint32
        if lib.smh_groom_abi_version() != 1:
            return None
        ptr = ctypes.c_void_p
        size = ctypes.c_size_t
        double = ctypes.c_double
        lib.smh_groom_geometry.argtypes = [
            ptr, ptr, size, size, ctypes.c_uint32, ptr, size,
            double, double, ctypes.c_bool, ptr, size, ptr,
        ]
        lib.smh_groom_geometry.restype = ctypes.c_int
        lib.smh_groom_strands.argtypes = [
            ptr, ptr, size, size, ptr, ptr, size, double, double, ptr,
        ]
        lib.smh_groom_strands.restype = ctypes.c_int
    except (AttributeError, OSError):
        return None
    return lib


def native() -> ctypes.CDLL | None:
    if os.environ.get("LEONIDA_GROOM_NATIVE") == "0":
        return None
    # The kernel matches this implementation of math.hypot and these C widths.
    if (sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] <= (3, 14)
            or array.array("I").itemsize != 4 or array.array("d").itemsize != 8):
        return None
    return _load()


def _point_buffer(polylines, nodes):
    if any(type(poly) is not list or len(poly) != nodes for poly in polylines):
        raise ValueError("expected fixed-width polylines")
    triples = list(chain.from_iterable(polylines))
    if set(map(type, triples)) != {tuple} or set(map(len, triples)) != {3}:
        raise ValueError("expected float triples")
    values = list(chain.from_iterable(triples))
    if set(map(type, values)) != {float}:
        raise ValueError("expected float coordinates")
    return array.array("d", values)

def _address(buffer):
    return buffer.buffer_info()[0]


class PackedGeometry(NamedTuple):
    positions: memoryview
    normals: memoryview | None
    indices: memoryview
    vertex_count: int
    index_width: int
    counts: array.array
    lo: list[float]
    hi: list[float]

    def index_groups(self):
        if not self.counts:
            yield self.indices, len(self.indices) // self.index_width
            return
        offset = 0
        for count in self.counts:
            if count:
                end = offset + count * self.index_width
                yield self.indices[offset:end], count
                offset = end


def pack_geometry(polylines, live, geometry, radius, up_axis, *, sides=3, tip=0.25):
    lib = native()
    if lib is None or type(polylines) is not list or not polylines or type(geometry) is not str:
        return None
    try:
        first = polylines[0]
        if type(first) is not list or len(first) < 2:
            return None
        nodes = len(first)
        rows = len(polylines)
        mode = _MODES[geometry]
        strip = geometry == "line-strip"
        mesh = geometry in ("tubes", "ribbons")
        if mesh and (type(radius) is not float or type(tip) is not float):
            return None
        if geometry == "tubes" and (type(sides) is not int or sides < 3):
            return None
        if live is None:
            counts = array.array("I", [nodes]) * rows
        elif type(live) is list and len(live) >= rows:
            counts = array.array("I")
            for n in live[:rows]:
                if type(n) is not int:
                    return None
                counts.append(min(nodes, n) if n >= 2 else 0)
        else:
            return None

        per_node, per_segment = {
            "tubes": (sides, 6 * sides), "ribbons": (2, 6),
            "curves": (1, 2), "line-strip": (1, 1),
        }[geometry]
        if strip:
            vertex_count = index_count = sum(counts)
        else:
            vertex_count = rows * nodes * per_node
            index_count = sum(n - 1 for n in counts if n >= 2) * per_segment
        if not index_count or vertex_count > 0xffffffff:
            return None

        trig = array.array("d")
        if geometry == "tubes":
            step = 2.0 * math.pi / sides
            trig.extend(chain.from_iterable(
                (math.cos(s * step), math.sin(s * step)) for s in range(sides)
            ))
        elif geometry == "ribbons":
            # The Python caller supplies width=radius * 2.0.
            radius = 0.5 * (radius * 2.0)
        else:
            radius = 0.0
        points = _point_buffer(polylines, nodes)
        if len(points) != rows * nodes * 3:
            return None
        pos_bytes = 12 * vertex_count
        idx_start = pos_bytes * (2 if mesh else 1)
        width = 2 if vertex_count <= 65535 else 4
        payload = bytearray(idx_start + index_count * width)
        dest = (ctypes.c_uint8 * len(payload)).from_buffer(payload)
        bounds = (ctypes.c_float * 6)()
        status = lib.smh_groom_geometry(
            _address(points), _address(counts), rows, nodes, mode,
            _address(trig), len(trig) // 2, radius, tip, up_axis == "z",
            dest, len(payload), bounds,
        )
        if status:
            return None
    except (IndexError, KeyError, OverflowError, TypeError, ValueError):
        return None

    view = memoryview(payload)
    return PackedGeometry(
        view[:pos_bytes], view[pos_bytes:idx_start] if mesh else None, view[idx_start:],
        vertex_count, width, counts if strip else array.array("I"), bounds[:3], bounds[3:],
    )


def blend_strands(groom, roots, max_strands, random_float):
    lib = native()
    if lib is None:
        return None
    try:
        nodes = groom.nodes_per_guide
        positions = groom.positions
        length = groom.lengths
        uvs = groom.uvs
        amount, low = groom.length_random
        if (type(nodes) is not int or nodes <= 0 or type(positions) is not list
                or type(amount) is not float or type(low) is not float
                or type(length) is not list or type(uvs) is not list
                or not len(positions) == len(length) == len(uvs)
                or not positions
                or (max_strands is not None and type(max_strands) is not int)):
            return None

        tris, bary = [], []
        triangles = groom.triangles
        if type(triangles) is not list:
            return None
        indices = array.array("I")
        weights = array.array("d")
        for i, culled in enumerate(roots["culled"]):
            if culled:
                continue
            if max_strands is not None and len(tris) >= max_strands:
                break
            tri = roots["tri"][i]
            w = roots["bary"][i]
            guides = triangles[tri]
            if (type(guides) is not tuple or len(guides) != 3
                    or not type(guides[0]) is type(guides[1]) is type(guides[2]) is int):
                return None
            indices.extend(guides)
            weights.extend((*w, random_float(roots["seed"][i])))
            tris.append(tri)
            bary.append(w)
        if not tris:
            return None

        guide_count = len(positions)
        points = _point_buffer(positions, nodes)
        attributes = array.array("d")
        for guide_len, uv in zip(length, uvs):
            if type(uv) is not tuple or len(uv) != 2 or not type(guide_len) is type(uv[0]) is type(uv[1]) is float:
                return None
            attributes.extend((guide_len, *uv))
        if (len(points) != guide_count * nodes * 3 or len(attributes) != guide_count * 3
                or len(weights) != len(tris) * 4):
            return None
        stride = nodes * 3 + 4
        output = array.array("d", [0.0]) * (len(tris) * stride)
        if lib.smh_groom_strands(
            _address(points), _address(attributes), guide_count, nodes,
            _address(indices), _address(weights), len(tris), amount, low, _address(output),
        ):
            return None
    except (IndexError, OverflowError, TypeError, ValueError):
        return None

    view = memoryview(output).cast("B")
    coords, live, lengths, uv = [], [], [], []
    pos_size = nodes * _XYZ.size
    stride = pos_size + _STATS.size
    for start in range(0, len(view), stride):
        end = start + pos_size
        coords.append(list(_XYZ.iter_unpack(view[start:end])))
        n, strand_length, u, v = _STATS.unpack_from(view, end)
        live.append(int(n))
        lengths.append(strand_length)
        uv.append((u, v))
    return {"points": coords, "live": live, "length": lengths, "uv": uv, "tri": tris, "bary": bary}
