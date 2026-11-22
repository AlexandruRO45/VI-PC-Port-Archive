"""Census of the vertex layouts of non-skinned geometries in an RPF9 archive, and a measured NORMAL slot.

Reuses tools/skin_streams.py's TOC-less drawable walk. For every non-skinned geometry it records the set of
present FVF semantic slots with their (element size, format byte), prints the histogram of distinct layouts,
and for each non-position slot tries the plausible unit-vector decodings on sampled vertices, reporting the
fraction whose length is within 1% of 1. Counts and format bytes only; no package data is stored or emitted.

Usage:
    python -I tools/fvf_layouts.py <archive.rpf> [...] [--top N] [--sample-verts N] [--json out.json]
"""
import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin_streams as ss  # noqa: E402

SKIN = ss.WEIGHT_SLOTS + ss.INDEX_SLOTS


def _f16(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def _s(v, bits):
    return max(v / ((1 << (bits - 1)) - 1), -1.0)


def _p10(w):
    out = []
    for sh in (0, 10, 20):
        v = (w >> sh) & 0x3FF
        out.append(_s(v - 1024 if v & 0x200 else v, 10))
    return out


# element size -> [(decoding name, function(bytes) -> xyz)]
DECODERS = {
    12: [("f32x3", lambda b: struct.unpack("<3f", b))],
    16: [("f32x4.xyz", lambda b: struct.unpack("<3f", b[:12]))],
    8: [("snorm16x4.xyz", lambda b: [_s(v, 16) for v in struct.unpack("<3h", b[:6])]),
        ("f16x4.xyz", lambda b: [_f16(v) for v in struct.unpack("<3H", b[:6])])],
    4: [("unorm10x3*2-1", lambda b: [((struct.unpack("<I", b)[0] >> s) & 0x3FF) / 1023 * 2 - 1
                                      for s in (0, 10, 20)]),
        ("unorm8x4.xyz*2-1", lambda b: [v / 255 * 2 - 1 for v in b[:3]]),
        ("snorm8x4.xyz", lambda b: [_s(v, 8) for v in struct.unpack("<3b", b[:3])]),
        ("snorm10x3", lambda b: _p10(struct.unpack("<I", b)[0]))],
}


class Census:
    def __init__(self, sample_verts, max_meshes_per_slot=400):
        self.layouts = {}
        self.sample_verts = sample_verts
        self.max_meshes = max_meshes_per_slot
        self.normals = {}      # (slot, size, fmt, decoding) -> [vectors, unit, meshes]
        self.slot_meshes = {}  # (slot, size, fmt) -> meshes sampled
        self.meshes = self.skinned = 0

    def on_mesh(self, mesh, gfx):
        self.meshes += 1
        if any(i in mesh.slots for i in SKIN):
            self.skinned += 1
            return
        key = tuple(sorted((i, s, f) for i, (s, f, _o) in mesh.slots.items()))
        self.layouts[key] = self.layouts.get(key, 0) + 1
        data = gfx[mesh.gfx_at:mesh.gfx_at + mesh.gfx_len]
        if len(data) < mesh.gfx_len:
            return
        n = mesh.vertices
        step = max(1, n // self.sample_verts)
        for slot, (size, fmt, off) in mesh.slots.items():
            if slot == 0 or size not in DECODERS:
                continue
            sk = (slot, size, fmt)
            if self.slot_meshes.get(sk, 0) >= self.max_meshes:
                continue
            self.slot_meshes[sk] = self.slot_meshes.get(sk, 0) + 1
            for name, fn in DECODERS[size]:
                acc = self.normals.setdefault(sk + (name,), [0, 0])
                for v in range(0, n, step):
                    x = fn(data[off + v * size:off + (v + 1) * size])
                    length = math.sqrt(sum(c * c for c in x))
                    acc[0] += 1
                    if length == length and abs(length - 1) <= 0.01:
                        acc[1] += 1


def fmt_name(size, fmt):
    return f"{size}b/{ss.FORMATS[fmt][0]}" if fmt in ss.FORMATS else f"{size}b/fmt{fmt}?"


def layout_text(key):
    return " ".join(f"{i}:{fmt_name(s, f)}" for i, s, f in key)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="RPF9 archives to scan")
    ap.add_argument("--top", type=int, default=10, help="layouts to print")
    ap.add_argument("--sample-verts", type=int, default=32, help="vertices sampled per mesh and slot")
    ap.add_argument("--json", help="write the tallies to this file")
    a = ap.parse_args(argv)
    for path in a.inputs:
        if not os.path.isfile(path):
            print(f"error: {path} is not a file", file=sys.stderr)
            return 2
    report = {}
    for path in a.inputs:
        c = Census(a.sample_verts)
        ss.scan_archive(path, 1, lambda t: print(t, flush=True), on_mesh=c.on_mesh)
        total = sum(c.layouts.values())
        print(f"{os.path.basename(path)}: {c.meshes} meshes, {c.skinned} skinned, {total} non-skinned, "
              f"{len(c.layouts)} distinct layouts")
        ranked = sorted(c.layouts.items(), key=lambda kv: -kv[1])
        for key, count in ranked[:a.top]:
            print(f"  {count:8d}  {layout_text(key)}")
        print("  unit-length share (|n| within 1% of 1) per non-position slot and decoding:")
        for (slot, size, fmt, name), (n, unit) in sorted(c.normals.items()):
            if n:
                print(f"    slot {slot:2d} {fmt_name(size, fmt):24s} {name:18s} {unit / n:6.1%}  "
                      f"({n} vectors, {c.slot_meshes[(slot, size, fmt)]} meshes)")
        report[path] = {"layouts": [{"count": n, "slots": [list(s) for s in k]} for k, n in ranked],
                        "normals": [{"slot": k[0], "size": k[1], "fmt": k[2], "decoding": k[3],
                                     "vectors": v[0], "unit": v[1]} for k, v in sorted(c.normals.items())]}
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
