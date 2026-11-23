"""Synthetic dehydrated XDXR BLASes for the GPU rehydration test (vk_xdxr_rehydrate_test), with the bytes
tools/bvh_rehydrate.py gives for each: the model that matches XboxBvhBuilder.dll byte for byte (docs/bvh.md §4a).

    python -I runtime/test/xdxr_rehydrate_cases.py <out dir>

For every case it writes <case>.src.bin (the dehydrated structure), <case>.expected.bin, one vertex file (and an
index file when the geometry has indices) per geometry, and <case>.geom.txt with one line per geometry:

    <vertices file> <stride> <indices file or -> <index type 0/1/2> <12 transform floats, or ->

The trees are random but well formed: several levels bottom-up, BOX32 and BOX16 nodes mixed in a level, more than
64 nodes in a level, more than 2,048 leaves (two chunks), stored override components, paired leaves with every
vertex order, u16/u32/no indices, a vertex stride, a geometry transform, the leaf frame (scale, translation and a
rotation) and the updatable flag (TRI0 parents). Standard library only; nothing is read from the package.
"""
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bvh_rehydrate as R  # noqa: E402

BLAS_DEHYDRATED = 0x80000000 | R.FLAG_DEHYDRATED | 0x00020006
ORDERS = [(a, b) for a in range(3) for b in range(3) if a != b]


def align(x, a):
    return (x + a - 1) // a * a


def ptr(offset, kind):
    return (offset >> 6) << 3 | kind


class Mesh:
    """One geometry: float3 vertices with a stride, and u16/u32/no indices, plus an optional transform."""

    def __init__(self, rng, triangles, index_type, stride, transform):
        self.index_type, self.stride, self.transform = index_type, stride, transform
        if index_type == 0:
            count = 3 * triangles
            self.indices = None
        else:
            count = triangles + 2
            self.indices = [rng.randrange(count) for _ in range(3 * triangles)]
        self.vertices = bytearray()
        for _ in range(count):
            v = [rng.uniform(-50.0, 50.0) for _ in range(3)]
            self.vertices += struct.pack("<3f", *v) + b"\xee" * (stride - 12)
        self.triangles = triangles

    def geometry(self):
        if self.indices is None:
            ib, index16 = None, False
        elif self.index_type == 1:
            ib, index16 = struct.pack("<%dH" % len(self.indices), *self.indices), True
        else:
            ib, index16 = struct.pack("<%dI" % len(self.indices), *self.indices), False
        m = None if self.transform is None else [self.transform[0:4], self.transform[4:8], self.transform[8:12]]
        return R.Geometry(bytes(self.vertices), self.stride, ib, index16, m)

    def index_bytes(self):
        if self.indices is None:
            return None
        fmt = "<%dH" if self.index_type == 1 else "<%dI"
        return struct.pack(fmt % len(self.indices), *self.indices)


def half_bits(rng):
    """A finite half, as a stored BOX16 override."""
    while True:
        h = rng.randrange(0x10000)
        if (h >> 10) & 0x1F != 0x1F:
            return h


def build(rng, meshes, leaves, compression=R.NO_TRANSFORM, updatable=False, box16_share=0.5, override_share=0.2,
          bounds=(-60.0, -60.0, -60.0, 60.0, 60.0, 60.0), single_leaf=False):
    """A dehydrated BLAS over `leaves` leaves of `meshes`. Returns the bytes."""
    n = leaves
    # Leaves: geometry, flags, prim0, prim1 (None: unpaired), vertex order.
    recs = []
    for _ in range(n):
        g = rng.randrange(len(meshes))
        prim0 = rng.randrange(meshes[g].triangles)
        prim1 = None if rng.random() < 0.4 else rng.randrange(meshes[g].triangles)
        a, b = rng.choice(ORDERS)
        order = a | b << 2 | rng.choice((0, 0x20, 0x80, 0xa0)) | rng.randrange(4) << 4 & 0x50
        recs.append((g, rng.randrange(256), prim0, prim1, order))

    chunks = (n + R.LEAVES_PER_CHUNK - 1) // R.LEAVES_PER_CHUNK
    leaves_bytes = (chunks - 1) * R.CHUNK_STRIDE + 64 * (n - (chunks - 1) * R.LEAVES_PER_CHUNK)

    # The tree, bottom-up: each level groups the level below into nodes of 2-4 children, BOX32 or BOX16.
    levels = []          # per level: [(wide, [child index into the level below])]
    width = n
    if not single_leaf:
        while True:
            nodes, i = [], 0
            while i < width:
                k = min(width - i, rng.randint(2, 4))
                nodes.append((rng.random() >= box16_share, list(range(i, i + k))))
                i += k
            # BOX32 nodes come first in a level.
            nodes.sort(key=lambda node: not node[0])
            levels.append(nodes)
            width = len(nodes)
            if width == 1:
                break

    # Layout: header, parent table, leaves, nodes, level table at the very end.
    nodes_bytes = 0
    for nodes in levels:
        n32 = sum(1 for wide, _ in nodes if wide)
        nodes_bytes += 128 * n32 + 64 * (len(nodes) - n32)
    body = leaves_bytes + 64 + nodes_bytes + 8 * len(levels) + 4 + 64
    elements = align(64 + 4 * (body // 64 + 2), 64)
    parents = 64
    nodes_at = align(elements + leaves_bytes, 64)
    total = align(nodes_at + nodes_bytes + 8 * len(levels) + 4, 64)
    src = bytearray(total)

    # Leaf chunks: five arrays per chunk.
    leaf_ptrs = []
    for c in range(chunks):
        part = recs[c * R.LEAVES_PER_CHUNK:(c + 1) * R.LEAVES_PER_CHUNK]
        base = elements + c * R.CHUNK_STRIDE
        out = bytearray()
        for g, *_ in part:
            out += g.to_bytes(3, "little")
        for _, fl, *_ in part:
            out.append(fl)
        for _, _, p0, *_ in part:
            out += p0.to_bytes(3, "little")
        for _, _, _, p1, _ in part:
            out += (0xFFFFFF if p1 is None else p1).to_bytes(3, "little")
        for *_, order in part:
            out.append(order)
        src[base:base + len(out)] = out
        leaf_ptrs += [ptr(base + 64 * i, R.TRI0) for i in range(len(part))]

    # Box levels: each level's BOX32 region, then its BOX16 region, groups of 64 nodes.
    below = leaf_ptrs
    table, at = [], 0
    for nodes in levels:
        wide_nodes = [node for node in nodes if node[0]]
        narrow_nodes = [node for node in nodes if not node[0]]
        regions = []
        level_ptrs = []
        for wide, group_nodes in ((True, wide_nodes), (False, narrow_nodes)):
            size = 128 if wide else 64
            start = at
            for g0 in range(0, len(group_nodes), R.GROUP):
                group = group_nodes[g0:g0 + R.GROUP]
                words, kids, values = [], [], []
                for _, children in group:
                    mask = 0
                    kid_count = len(children)
                    for bit in range(6 * kid_count):
                        if rng.random() < override_share:
                            mask |= 1 << bit
                            values.append(rng.uniform(-80.0, 80.0) if wide else half_bits(rng))
                    words.append(kid_count << 24 | mask)
                    kids += [below[c] for c in children]
                packed = struct.pack("<%dI" % len(words), *words) + struct.pack("<%dI" % len(kids), *kids)
                packed += struct.pack("<%df" % len(values), *values) if wide else struct.pack("<%dH" % len(values), *values)
                group_at = nodes_at + start + (g0 // R.GROUP) * R.GROUP * size
                assert len(packed) <= len(group) * size
                src[group_at:group_at + len(packed)] = packed
            for k in range(len(group_nodes)):
                level_ptrs.append(ptr(nodes_at + start + k * size, R.BOX32 if wide else R.BOX16))
            at += size * len(group_nodes)
            regions.append(at)
        table.append(tuple(regions))
        below = level_ptrs
    root = below[0] if levels else leaf_ptrs[0]

    lv = b"".join(struct.pack("<2I", a, b) for a, b in table) + struct.pack("<I", len(table))
    src[total - len(lv):] = lv
    flags = BLAS_DEHYDRATED | (R.FLAG_UPDATABLE if updatable else 0)
    src[0:64] = struct.pack("<4I6f6I", flags, total // 64 - 1, 0x1234, n, *bounds, nodes_at, 0, root, elements,
                            parents, compression)
    return bytes(src)


def write_case(out, name, src, meshes):
    with open(os.path.join(out, name + ".src.bin"), "wb") as f:
        f.write(src)
    expected = R.rehydrate(src, [m.geometry() for m in meshes])
    with open(os.path.join(out, name + ".expected.bin"), "wb") as f:
        f.write(expected)
    lines = []
    for g, m in enumerate(meshes):
        vname = "%s.g%d.vertices.bin" % (name, g)
        with open(os.path.join(out, vname), "wb") as f:
            f.write(m.vertices)
        ib = m.index_bytes()
        iname = "-"
        if ib is not None:
            iname = "%s.g%d.indices.bin" % (name, g)
            with open(os.path.join(out, iname), "wb") as f:
                f.write(ib)
        xf = "-" if m.transform is None else " ".join("%.9g" % x for x in m.transform)   # exact for float32
        lines.append("%s %d %s %d %s" % (vname, m.stride, iname, m.index_type if ib is not None else 0, xf))
    with open(os.path.join(out, name + ".geom.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(expected)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    out = argv[1]
    os.makedirs(out, exist_ok=True)
    for entry in os.listdir(out):
        if entry.endswith((".bin", ".txt")):
            os.remove(os.path.join(out, entry))
    rng = random.Random(7)
    # The transform is float32 in GPU memory, so the model gets the float32 values too.
    rot45 = [R.f32(x) for x in (0.70710677, 0.0, 0.70710677, 1.5, 0.0, 1.0, 0.0, -2.0, -0.70710677, 0.0, 0.70710677, 0.25)]
    cases = []
    # One BOX32 root over a triangle list.
    meshes = [Mesh(rng, 12, 0, 12, None)]
    cases.append(("small", build(rng, meshes, 4, box16_share=0.0, override_share=0.0), meshes))
    # Several levels, both widths, >64 nodes a level, two leaf chunks, overrides, u32 indices, a transform, TRI0
    # parents (updatable).
    meshes = [Mesh(rng, 900, 2, 12, rot45), Mesh(rng, 700, 0, 16, None)]
    cases.append(("mixed", build(rng, meshes, 3000, updatable=True), meshes))
    # BOX16 mostly, u16 indices with a 20-byte stride, the leaf frame: scale on, translation on, rotation index 5.
    compression = 1 << 31 | 1 << 30 | 5 << 24 | 0x80 << 16 | 0x7e << 8 | 0x81
    meshes = [Mesh(rng, 400, 1, 20, None)]
    cases.append(("box16_frame", build(rng, meshes, 600, compression=compression, box16_share=0.9,
                                       bounds=(5.0, -40.0, -45.0, 55.0, 41.0, -2.0)), meshes))
    # The leaf frame with scale only and a rotation of 3 * pi/8 about every axis.
    compression = 1 << 31 | 0x3f << 24 | 0x7f << 16 | 0x80 << 8 | 0x7e
    meshes = [Mesh(rng, 300, 2, 12, None), Mesh(rng, 50, 1, 12, rot45)]
    cases.append(("frame_rotated", build(rng, meshes, 500, compression=compression), meshes))
    # A single TRI0 leaf with no box levels.
    meshes = [Mesh(rng, 3, 0, 12, None)]
    cases.append(("root_leaf", build(rng, meshes, 1, single_leaf=True), meshes))
    for name, src, meshes in cases:
        size = write_case(out, name, src, meshes)
        print("%s: %d bytes" % (name, size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
