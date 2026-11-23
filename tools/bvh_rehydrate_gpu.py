"""Run XboxBvhBuilder.dll's three rehydration programs on the GPU and compare with tools/bvh_rehydrate.py (TEST-376).

tools/bvh_gpu_smoke.py runs every builder program once with a synthetic root table; the rehydration programs need a
real one. This tool takes a dehydrated BLAS and its geometry (the inputs tools/bvh_rehydrate.py takes), lays them out in
one heap image the way vk_compute_test hands memory to a translated program (root table at heap VA 0, s[0:1] pointing
at it, every descriptor and address a heap VA), runs the dispatches in the order the data needs them, and compares the
structure the GPU wrote with the CPU reference, counting records and bytes only.

    python -I tools/bvh_rehydrate_gpu.py <dehydrated.bin> --geom VB[,IB] [--geom ...] [--header-offset 256]
                                         [--transform 12 floats] [--stride N] [--index16]
                                         --spirv <translated folder> [--build runtime/build | --vkc <exe>]
                                         [--partial pad|raw] [--timeout 120] [--no-validation]
                                         [--work DIR] [--plan-only] [--json report.json]

Root arguments, as the programs read them (offsets in the table at s[0:1]; measured on the gfx1013 disassembly):

LeafRehydration (1024 threads; group g rehydrates leaf chunk g, threads t and t + 1024 of it, while below the count)
    +0x00 3 x f32  leaf-frame translation t            +0x0c u32  offsetToElements (chunk g at + g << 17)
    +0x10 3 x f32  leaf-frame scale s                  +0x1c u32  leaf frame on (0: vertices are stored as given)
    +0x20 3 x f32  rotation row 0                      +0x2c u32  numElements
    +0x30 3 x f32  rotation row 1                      +0x40 3 x f32  rotation row 2
    +0x4c u64  dehydrated structure (its header)       +0x54 u64  output structure (its header)
    +0x5c u64  geometry records, 48 bytes each, indexed by the leaf's geometry index: the D3D12 triangle geometry
               description (+0 transform VA or 0, +8 index format 42 / 57 / other, +12 vertex format, +24 index buffer
               VA, +32 vertex buffer VA, +40 vertex stride)
Box32Rehydration / Box16Rehydration (64 threads; group g rehydrates nodes 64g.. of one level's BOX32 or BOX16 run)
    +0x00 u32  offsetToNodes + start of the run (group g at + g * 8 KiB / 4 KiB)
    +0x04 u32  offsetToParents    +0x08 u32  offsetToElements    +0x0c u32  updatable (parents of TRI0 children too)
    +0x4c u64  dehydrated structure    +0x54 u64  output structure

A box group has no node count in its arguments: a program takes it from the number of active lanes (s_bcnt1 of EXEC),
so the console runs a run's last group with fewer lanes. vk_compute_test always runs whole groups. With --partial pad
(the default) the last group of a run is dispatched alone from a copy of its bytes padded to 64 node words (the extra
lanes see a node without children), which gives the real lanes the addresses a partial group computes; its extra
lanes still write empty nodes after the run, where a later dispatch writes the real ones, so those bytes are counted
apart. --partial raw dispatches whole groups over the stored bytes, the way a device without partial groups would.

Order: LeafRehydration over every chunk, then level by level from the bottom, the BOX32 run before the BOX16 run (a
box reads its children's rehydrated bounds from the output). Each dispatch is one vk_compute_test run whose heap
starts as the previous run's dump with the next root table written over VA 0.

The output starts as a fill pattern except the bytes the three programs do not write and the CPU reference does: the
header and the root's parent entry (another step of the builder writes those). The comparison counts, against the
reference, leaf records, BOX32 and BOX16 nodes and parent entries that match or differ, and output bytes outside them
that changed. One difference is expected and counted on its own, not as a mismatch: Box32Rehydration fills the
bounds of a slot without a child with +inf / -inf, the PC library (and tools/bvh_rehydrate.py) with +FLT_MAX /
-FLT_MAX ("empty-slot fill"). The report holds counts, statuses and heap addresses only; with --work the heap images stay in that
folder (they hold the BLAS and its geometry: keep them local). --plan-only writes the plan and the first heap image
and runs nothing. Exit status 0 when everything matches, 1 when something differs or a run fails, 2 on an input
error. Standard library only.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bvh_rehydrate as R  # noqa: E402

TABLE_BYTES = 4096
FILL = 0xEE
ALIGN = 256
GEOM_RECORD = 48
INDEX_R32_UINT, INDEX_R16_UINT, VERTEX_R32G32B32_FLOAT = 42, 57, 6
LEAF_THREADS_PER_GROUP = 2 * 1024
MAX_HEAP_BYTES = 16777216 * 4          # vk_compute_test's VKC_OUT_WORDS limit
DST_SLACK = 64 * 128                   # room for a padded last group's extra lanes past the structure
PROGRAMS = ("LeafRehydration", "Box32Rehydration", "Box16Rehydration")


def _align(x, a=ALIGN):
    return (x + a - 1) // a * a


class Heap:
    """A bump allocator over the heap's address space, past the root table."""

    def __init__(self):
        self.end = TABLE_BYTES
        self.regions = []              # (name, va, size)

    def take(self, name, size, align=ALIGN):
        va = _align(self.end, align)
        self.end = va + size
        self.regions.append((name, va, size))
        return va


def _u24(buf, at):
    return buf[at] | buf[at + 1] << 8 | buf[at + 2] << 16


def box_groups(src, hdr, start, count, wide):
    """[(group offset from the header, node count, word list, child list, value bytes)] of one BOX32/BOX16 run."""
    group_bytes = (1 << 13) if wide else (1 << 12)
    value_size = 4 if wide else 2
    out = []
    for g in range((count + 63) // 64):
        n = min(64, count - 64 * g)
        at = hdr.offset_to_nodes + start + g * group_bytes
        words = struct.unpack_from("<%dI" % n, src, at)
        kids = sum(w >> 24 for w in words)
        values = sum(bin(w & 0xFFFFFF).count("1") for w in words)
        children = struct.unpack_from("<%dI" % kids, src, at + 4 * n)
        vbytes = bytes(src[at + 4 * (n + kids): at + 4 * (n + kids) + value_size * values])
        out.append((at, n, words, children, vbytes))
    return out


def padded_group(words, children, vbytes):
    """A group's bytes with 64 node words (the missing ones 0: no children, no overrides), then its children and
    values, as a 64-lane group sees them when its real lanes number len(words)."""
    w = list(words) + [0] * (64 - len(words))
    return struct.pack("<64I", *w) + struct.pack("<%dI" % len(children), *children) + vbytes


def written_records(src, hdr):
    """What the three programs write, from the CPU model: (leaves, box32, box16, parents) as lists of offsets from the
    header (parents: the stored entries the CPU reference writes)."""
    leaves = [out for out, *_ in R.leaf_records(src, hdr)]
    b32, b16, parents = [], [], set()
    if hdr.root & 7 in (R.BOX16, R.BOX32):
        for s32, c32, s16, c16 in R.levels(src, hdr):
            for start, count, wide, sink in ((s32, c32, True, b32), (s16, c16, False, b16)):
                for at, n, words, children, _ in box_groups(src, hdr, start, count, wide):
                    sink += [at + i * (128 if wide else 64) for i in range(n)]
                    for c in children:
                        if (c & 6) == 4 or ((c & 7) == R.TRI0 and hdr.flags & R.FLAG_UPDATABLE):
                            e = hdr.stored_parent_entry(((c >> 3) << 6) & R.M32, hdr.total_bytes)
                            if e is not None:
                                parents.add(e)
    return leaves, b32, b16, sorted(parents)


def repack_leaves(src, hdr, n):
    """The dehydrated bytes with chunk 0 holding only the first n leaves. LeafRehydration finds a chunk's five arrays
    from the chunk's own leaf count (min(2048, the count at +0x2c)), so a smaller count needs arrays packed for it."""
    recs = list(R.leaf_records(src, hdr))[:n]
    out = bytearray(src)
    at, c0 = hdr.offset_to_elements, min(R.LEAVES_PER_CHUNK, hdr.num_elements)
    out[at:at + 11 * c0] = bytes(11 * c0)
    arr = bytearray()
    arr += b"".join(gi.to_bytes(3, "little") for _, gi, *_ in recs)
    arr += bytes(fl for _, _, fl, *_ in recs)
    arr += b"".join(p0.to_bytes(3, "little") for _, _, _, p0, _, _ in recs)
    arr += b"".join((0xFFFFFF if p1 is None else p1).to_bytes(3, "little") for _, _, _, _, p1, _ in recs)
    arr += bytes(o for *_, o in recs)
    out[at:at + len(arr)] = arr
    return bytes(out)


def frame_words(hdr):
    """Leaf-frame root words (t, s, R rows, on) from compressionParams, as the CPU reference applies them."""
    f = R.LeafFrame(hdr)
    return f.on, f.t, f.s, f.r


def build_plan(dehydrated, geometries, header_offset=0, partial="pad", seed_leaves=False, leaf_count=None):
    """Lay out the heap and list the dispatches. Returns (plan dict, initial heap bytearray, reference bytes).

    seed_leaves: no LeafRehydration; the output starts with the reference's leaves, so the box programs run alone.
    leaf_count: LeafRehydration alone, with this count at +0x2c (0: one group with no active lane), to find where
    it fails; leaves past the count are expected untouched."""
    if seed_leaves and leaf_count is not None:
        raise ValueError("--seed-leaves and --leaf-count exclude each other")
    hdr = R.Header(dehydrated, header_offset)
    if not hdr.flags & R.FLAG_DEHYDRATED:
        raise ValueError("the dehydrated flag (0x01000000) is not set")
    size = hdr.total_bytes
    src = bytes(dehydrated[header_offset:header_offset + size])
    if len(src) < size:
        raise ValueError("the input is shorter than the %d bytes the header declares" % size)
    reference = bytes(R.rehydrate(dehydrated, geometries, header_offset))

    heap = Heap()
    geoms = list(geometries)
    desc_va = heap.take("geometry records", GEOM_RECORD * max(1, len(geoms)))
    transforms, buffers = {}, []
    for i, g in enumerate(geoms):
        t_va = 0
        if g.transform is not None:
            key = tuple(v for row in g.transform for v in row)
            if key not in transforms:
                transforms[key] = heap.take("transform %d" % len(transforms), 48)
            t_va = transforms[key]
        vb_va = heap.take("vertices %d" % i, len(g.vertices))
        ib_va = heap.take("indices %d" % i, len(g.indices)) if g.indices is not None else 0
        buffers.append((t_va, vb_va, ib_va))
    src_va = heap.take("dehydrated", size, 1 << 12)
    dst_va = heap.take("output", size + DST_SLACK, 1 << 12)

    dispatches = []
    on, t, s, r = frame_words(hdr)
    leaf = bytearray(TABLE_BYTES)
    leaf_n = hdr.num_elements if leaf_count is None else leaf_count
    if leaf_count is not None and not 0 <= leaf_n <= min(hdr.num_elements, R.LEAVES_PER_CHUNK):
        raise ValueError("--leaf-count must be 0 to %d" % min(hdr.num_elements, R.LEAVES_PER_CHUNK))
    if leaf_count is not None:
        src = repack_leaves(src, hdr, leaf_n)
    struct.pack_into("<3fI3fI3fI3fI3f", leaf, 0, *t, hdr.offset_to_elements, *s, int(on), *r[0], leaf_n,
                     *r[1], 0, *r[2])
    struct.pack_into("<3Q", leaf, 0x4C, src_va, dst_va, desc_va)
    if (hdr.num_elements and not seed_leaves) or leaf_count is not None:
        dispatches.append({"program": "LeafRehydration", "groups": max(1, -(-leaf_n // LEAF_THREADS_PER_GROUP)),
                           "what": "leaves", "nodes": leaf_n, "table": bytes(leaf)})
    pads = []
    if hdr.root & 7 in (R.BOX16, R.BOX32) and leaf_count is None:
        for lv, (s32, c32, s16, c16) in enumerate(R.levels(src, hdr)):
            for start, count, wide in ((s32, c32, True), (s16, c16, False)):
                if not count:
                    continue
                prog = "Box32Rehydration" if wide else "Box16Rehydration"
                kind = "box32" if wide else "box16"
                groups = box_groups(src, hdr, start, count, wide)
                full = count // 64 if partial == "pad" else len(groups)

                def table(first, src_base):
                    tb = bytearray(TABLE_BYTES)
                    struct.pack_into("<4I", tb, 0, first, hdr.offset_to_parents, hdr.offset_to_elements,
                                     1 if hdr.flags & R.FLAG_UPDATABLE else 0)
                    struct.pack_into("<2Q", tb, 0x4C, src_base, dst_va)
                    return bytes(tb)

                if full:
                    dispatches.append({"program": prog, "groups": full, "what": "%s level %d" % (kind, lv),
                                       "nodes": min(count, 64 * full), "table": table(hdr.offset_to_nodes + start, src_va)})
                if partial == "pad" and count % 64:
                    at, n, words, children, vbytes = groups[-1]
                    body = padded_group(words, children, vbytes)
                    pad_va = heap.take("padded %s level %d" % (kind, lv), len(body))
                    pads.append((pad_va, body))
                    # the program reads src + table[0] + x; make that the padded copy
                    dispatches.append({"program": prog, "groups": 1, "what": "%s level %d last group" % (kind, lv),
                                       "nodes": n, "padded": True, "shadow": [at + n * (128 if wide else 64),
                                                                               at + 64 * (128 if wide else 64)],
                                       "table": table(at, pad_va - at)})
    heap_bytes = _align(heap.end, 4096)
    if heap_bytes > MAX_HEAP_BYTES:
        raise ValueError("the heap needs %d bytes; vk_compute_test takes at most %d" % (heap_bytes, MAX_HEAP_BYTES))

    image = bytearray(heap_bytes)
    for i, g in enumerate(geoms):
        t_va, vb_va, ib_va = buffers[i]
        image[vb_va:vb_va + len(g.vertices)] = g.vertices
        if g.indices is not None:
            image[ib_va:ib_va + len(g.indices)] = g.indices
        fmt = 0 if g.indices is None else INDEX_R16_UINT if g.index16 else INDEX_R32_UINT
        n_idx = (len(g.indices) // (2 if g.index16 else 4)) if g.indices is not None else 0
        struct.pack_into("<QIIIIQQQ", image, desc_va + GEOM_RECORD * i, t_va, fmt, VERTEX_R32G32B32_FLOAT, n_idx,
                         len(g.vertices) // g.stride, ib_va, vb_va, g.stride)
    for key, va in transforms.items():
        struct.pack_into("<12f", image, va, *key)
    image[src_va:src_va + size] = src
    image[dst_va:dst_va + size + DST_SLACK] = bytes([FILL]) * (size + DST_SLACK)
    seeded = seeded_ranges(hdr, size)
    if seed_leaves:
        seeded += [(o, o + 64) for o in written_records(src, hdr)[0]]
    for a, b in seeded:
        image[dst_va + a:dst_va + b] = reference[a:b]
    for va, body in pads:
        image[va:va + len(body)] = body
    if dispatches:
        image[:TABLE_BYTES] = dispatches[0]["table"]

    plan = {"heap_bytes": heap_bytes, "src_va": src_va, "dst_va": dst_va, "geometry_records_va": desc_va,
            "structure_bytes": size, "leaves": hdr.num_elements, "geometries": len(geoms), "partial": partial,
            "leaf_frame_on": bool(on), "seeded": seeded, "leaves_seeded": bool(seed_leaves),
            "leaf_count": leaf_n, "boxes_run": leaf_count is None,
            "regions": [{"name": n, "va": v, "bytes": b} for n, v, b in heap.regions],
            "dispatches": dispatches}
    return plan, image, reference


def seeded_ranges(hdr, size):
    """Output bytes the programs do not write but the reference holds: the kept header and the root's parent entry."""
    out = [(0, min(hdr.kept_header_bytes(), size))]
    e = hdr.stored_parent_entry(((hdr.root >> 3) << 6) & R.M32, size)
    if e is not None:
        out.append((e, e + 4))
    return out


INF32, NEG_INF32 = 0x7F800000, 0xFF800000
FLT_MAX32, NEG_FLT_MAX32 = 0x7F7FFFFF, 0xFF7FFFFF


def empty_slot_fill(got, reference, b32):
    """BOX32 slots without a child: Box32Rehydration fills their bounds with +inf (min) and -inf (max), while the
    PC library and tools/bvh_rehydrate.py write +FLT_MAX and -FLT_MAX. Returns the reference with that fill taken
    from `got` wherever an empty slot holds exactly the console's, and {nodes, slots} counting where it did."""
    ref = bytearray(reference)
    nodes = slots = 0
    console = struct.pack("<6I", *[INF32] * 3 + [NEG_INF32] * 3)
    library = struct.pack("<6I", *[FLT_MAX32] * 3 + [NEG_FLT_MAX32] * 3)
    for o in b32:
        children = struct.unpack_from("<4I", reference, o)
        hit = 0
        for slot, child in enumerate(children):
            at = o + 16 + 24 * slot
            if child == R.INVALID and reference[at:at + 24] == library and got[at:at + 24] == console:
                ref[at:at + 24] = console
                hit += 1
        nodes += hit > 0
        slots += hit
    return bytes(ref), {"nodes": nodes, "slots": slots}


def compare(after, plan, reference, src, hdr):
    """Count matching and differing records between the output region of `after` and the reference."""
    dst = plan["dst_va"]
    size = plan["structure_bytes"]
    got = bytes(after[dst:dst + size])
    leaves, b32, b16, parents = written_records(src, hdr)
    if plan.get("leaves_seeded"):
        leaves = []                        # copied from the reference: nothing of the GPU's to count
    leaves = leaves[:plan.get("leaf_count", len(leaves))]
    if not plan.get("boxes_run", True):
        b32, b16, parents = [], [], []
    out = {}
    reference, out["box32_empty_slot_fill"] = empty_slot_fill(got, reference, b32)
    mask = bytearray(size)
    for name, offsets, n in (("leaves", leaves, 64), ("box32", b32, 128), ("box16", b16, 64), ("parents", parents, 4)):
        same = sum(1 for o in offsets if got[o:o + n] == reference[o:o + n])
        out[name] = {"total": len(offsets), "match": same, "differ": len(offsets) - same,
                     "bytes_differ": sum(1 for o in offsets for k in range(n) if got[o + k] != reference[o + k])}
        for o in offsets:
            mask[o:o + n] = b"\x01" * n
    for a, b in plan["seeded"]:
        mask[a:b] = b"\x01" * (b - a)
    # expected: the reference where a record or a seeded range is, the fill elsewhere (built with slices)
    expected = bytearray(bytes([FILL]) * size)
    for name, offsets, n in (("leaves", leaves, 64), ("box32", b32, 128), ("box16", b16, 64), ("parents", parents, 4)):
        for o in offsets:
            expected[o:o + n] = reference[o:o + n]
    for a, b in plan["seeded"]:
        expected[a:b] = reference[a:b]
    if plan.get("boxes_run", True):        # a --leaf-count run models only what it writes
        unmodelled = sum(1 for i, r in enumerate(reference[:size]) if r not in (0, FILL) and not mask[i])
    else:
        unmodelled = 0
    shadow = bytearray(size + DST_SLACK)
    for d in plan["dispatches"]:
        if d.get("padded"):
            a, b = d["shadow"]
            shadow[a:min(b, size + DST_SLACK)] = b"\x01" * (min(b, size + DST_SLACK) - a)
    diffs = [i for i, (g, e) in enumerate(zip(got, expected)) if g != e]          # one pass over the structure
    stray = [i for i in diffs if not mask[i]]                                     # expected is FILL there
    tail = bytes(after[dst + size:dst + size + DST_SLACK])
    stray_tail = [size + i for i, b in enumerate(tail) if b != FILL]
    out["stray_bytes"] = len(stray) + len(stray_tail)
    out["stray_bytes_in_padded_shadow"] = sum(1 for i in stray + stray_tail if shadow[i])
    out["structure_bytes"] = size
    # bytes a padded group's extra lanes wrote outside every record are counted above, not as differences
    real = [i for i in diffs if mask[i] or not shadow[i]]
    out["structure_bytes_differ"] = len(real)
    out["structure_words_differ"] = len({i // 4 for i in real if i < size // 4 * 4})
    out["reference_bytes_not_modelled"] = unmodelled
    return out


def outside_changes(before, after, plan):
    """Heap bytes outside the root table and the output region (+ slack) that a run changed."""
    dst, end = plan["dst_va"], plan["dst_va"] + plan["structure_bytes"] + DST_SLACK
    n = 0
    for lo, hi in ((TABLE_BYTES, dst), (end, len(before))):
        a, b = before[lo:hi], after[lo:hi]
        if a != b:
            n += sum(1 for x, y in zip(a, b) if x != y)
    return n


def heap_loaded(before, after, plan):
    """False when the dump does not hold the image the run was given: the root table and the dehydrated input are
    read-only for all three programs, so a dump whose copies of them are vk_compute_test's fill (0xEE) means
    VKC_HEAP_INIT was not applied (a vk_compute_test built before it existed ignores the variable)."""
    src, size = plan["src_va"], plan["structure_bytes"]
    for lo, hi in ((0, TABLE_BYTES), (src, src + size)):
        if before[lo:hi] != after[lo:hi] and after[lo:hi] == bytes([FILL]) * (hi - lo):
            return False
    return True


def supports_heap_init(vkc):
    """Whether a vk_compute_test executable knows VKC_HEAP_INIT (the variable's name is in its strings)."""
    try:
        with open(vkc, "rb") as f:
            return b"VKC_HEAP_INIT" in f.read()
    except OSError:
        return False


def run_dispatch(vkc, spv, init, dump, heap_words, groups, validation, timeout):
    """One vk_compute_test run. Returns (status, last output line): ok, 'exit N', validation, timeout."""
    env = dict(os.environ, VKC_OUT_WORDS=str(heap_words), VKC_HEAP_INIT=str(init), VKC_DUMP=str(dump))
    if validation:
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    try:
        p = subprocess.run([str(vkc), str(spv), str(groups), "dump"], capture_output=True, text=True, env=env,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    out = (p.stdout or "") + (p.stderr or "")
    last = (out.strip().splitlines() or [""])[-1][:160]
    if p.returncode != 0:
        return "exit %d" % p.returncode, last
    if "Validation Error" in out:
        return "validation", next((line for line in out.splitlines() if "VUID" in line), last)[:160]
    return "ok", last


def run(plan, image, reference, src, hdr, spirv, vkc, work, *, validation=True, timeout=120.0, runner=run_dispatch):
    """Every dispatch in order, each heap the previous dump with the next table. Returns the report dict."""
    work = Path(work)
    heap = bytes(image)
    initial = bytes(image)
    steps = []
    failed = False
    for i, d in enumerate(plan["dispatches"]):
        spv = Path(spirv) / (d["program"] + ".spv")
        step = {"program": d["program"], "what": d["what"], "groups": d["groups"], "nodes": d["nodes"]}
        if not spv.is_file():
            step.update(status="missing", detail="%s not translated" % spv.name)
            steps.append(step)
            failed = True
            break
        cur = bytearray(heap)
        cur[:TABLE_BYTES] = d["table"]
        init, dump = work / ("heap%02d.in" % i), work / ("heap%02d.out" % i)
        init.write_bytes(bytes(cur))
        status, detail = runner(vkc, spv, init, dump, len(cur) // 4, d["groups"], validation, timeout)
        step.update(status=status, detail=detail if status != "ok" else "")
        if status == "ok":
            after = dump.read_bytes() if dump.is_file() else b""
            if len(after) < len(cur):
                step.update(status="short dump", detail="%d of %d bytes" % (len(after), len(cur)))
            elif not heap_loaded(cur, after[:len(cur)], plan):
                step.update(status="heap not loaded",
                            detail="the dump holds vk_compute_test's fill, not the image: VKC_HEAP_INIT was ignored")
            else:
                after = after[:len(cur)]
                step["outside_bytes_changed"] = outside_changes(cur, after, plan)
                if step["outside_bytes_changed"]:
                    step.update(status="input changed",
                                detail="%d heap bytes outside the output changed" % step["outside_bytes_changed"])
                heap = after
        steps.append(step)
        if step["status"] != "ok":
            failed = True
            break
    result = compare(heap, plan, reference, src, hdr)
    result["outside_bytes_changed"] = outside_changes(initial, heap, plan)
    # structure_bytes_differ also covers seeded leaves and the header, which no record count sees (Envyre, #8082)
    ok = (not failed and all(result[k]["differ"] == 0 for k in ("leaves", "box32", "box16", "parents"))
          and result["structure_bytes_differ"] == 0
          and result["stray_bytes"] == result["stray_bytes_in_padded_shadow"] and result["outside_bytes_changed"] == 0)
    return {"ok": ok, "steps": steps, "result": result}


def plan_summary(plan):
    keep = {k: v for k, v in plan.items() if k != "dispatches"}
    keep["dispatches"] = [{k: v for k, v in d.items() if k != "table"} for d in plan["dispatches"]]
    return keep


def find_vk_compute_test(build):
    exe = "vk_compute_test.exe" if os.name == "nt" else "vk_compute_test"
    for c in (Path(build) / exe, Path(build) / "Release" / exe, Path(build) / "RelWithDebInfo" / exe):
        if c.is_file():
            return c
    raise SystemExit("error: %s not found in %s (build the runtime first, or pass --vkc)" % (exe, build))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the builder's rehydration programs on the GPU (TEST-376).")
    ap.add_argument("dehydrated")
    ap.add_argument("--geom", action="append", default=[], metavar="VB[,IB]")
    ap.add_argument("--header-offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--index16", action="store_true")
    ap.add_argument("--transform")
    ap.add_argument("--spirv", help="folder with LeafRehydration.spv, Box32Rehydration.spv, Box16Rehydration.spv")
    ap.add_argument("--build", default=str(HERE.parent / "runtime" / "build"), help="folder with vk_compute_test")
    ap.add_argument("--vkc", help="vk_compute_test executable (overrides --build)")
    ap.add_argument("--partial", choices=("pad", "raw"), default="pad",
                    help="how a box run's last, partial group is dispatched (default pad; see the docstring)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per dispatch (default 120)")
    ap.add_argument("--no-validation", action="store_true", help="don't enable VK_LAYER_KHRONOS_validation")
    ap.add_argument("--work", help="keep the heap images here (they hold the BLAS and its geometry: local only)")
    ap.add_argument("--plan-only", action="store_true", help="write the plan and the first heap image; run nothing")
    ap.add_argument("--seed-leaves", action="store_true",
                    help="skip LeafRehydration: the output starts with the CPU reference's leaves (box programs alone)")
    ap.add_argument("--leaf-count", type=int, help="run LeafRehydration alone with this leaf count (0: no lane active)")
    ap.add_argument("--json", help="write the plan and the report here")
    a = ap.parse_args(argv)
    try:
        transform = R.parse_transform(a.transform) if a.transform else None
        geometries = []
        for spec in a.geom:
            parts = spec.split(",")
            with open(parts[0], "rb") as f:
                vb = f.read()
            ib = None
            if len(parts) > 1 and parts[1]:
                with open(parts[1], "rb") as f:
                    ib = f.read()
            geometries.append(R.Geometry(vb, a.stride, ib, a.index16, transform))
        with open(a.dehydrated, "rb") as f:
            data = f.read()
        plan, image, reference = build_plan(data, geometries, a.header_offset, a.partial, a.seed_leaves,
                                            a.leaf_count)
    except (OSError, ValueError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    hdr = R.Header(data, a.header_offset)
    src = bytes(data[a.header_offset:a.header_offset + hdr.total_bytes])
    counts = {}
    for d in plan["dispatches"]:
        counts[d["program"]] = counts.get(d["program"], 0) + 1
    print("heap %d bytes, structure %d bytes, %d leaves, %d dispatches (%s)" % (
        plan["heap_bytes"], plan["structure_bytes"], plan["leaves"], len(plan["dispatches"]),
        ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    report = {"plan": plan_summary(plan)}
    if a.plan_only:
        if a.work:
            os.makedirs(a.work, exist_ok=True)
            Path(a.work, "heap00.in").write_bytes(bytes(image))
            for i, d in enumerate(plan["dispatches"]):
                Path(a.work, "table%02d.bin" % i).write_bytes(d["table"])
        if a.json:
            Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        return 0
    if not a.spirv:
        ap.error("--spirv is required unless --plan-only")
    vkc = Path(a.vkc) if a.vkc else find_vk_compute_test(a.build)
    if not supports_heap_init(vkc):
        print("error: %s does not know VKC_HEAP_INIT (built before it was added): rebuild vk_compute_test" % vkc,
              file=sys.stderr)
        return 2
    if a.work:
        os.makedirs(a.work, exist_ok=True)
        rep = run(plan, image, reference, src, hdr, a.spirv, vkc, a.work, validation=not a.no_validation,
                  timeout=a.timeout)
    else:
        with tempfile.TemporaryDirectory(prefix="bvh_rehydrate_gpu_") as tmp:
            rep = run(plan, image, reference, src, hdr, a.spirv, vkc, tmp, validation=not a.no_validation,
                      timeout=a.timeout)
    for s in rep["steps"]:
        print("%-17s %-26s groups %-5d %-10s %s" % (s["program"], s["what"], s["groups"], s["status"], s["detail"]))
    r = rep["result"]
    if not any(s["status"] == "ok" for s in rep["steps"]):
        print("nothing compared: the chain stopped before any dispatch finished")
        report.update(rep)
        if a.json:
            Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        return 1
    if plan["leaves_seeded"]:
        print("leaves   seeded from the CPU reference (LeafRehydration not run)")
    for k in ("leaves", "box32", "box16", "parents"):
        if k == "leaves" and plan["leaves_seeded"]:
            continue
        print("%-8s %7d match, %7d differ (%d bytes) of %d" % (k, r[k]["match"], r[k]["differ"], r[k]["bytes_differ"],
                                                              r[k]["total"]))
    fill = r["box32_empty_slot_fill"]
    if fill["slots"]:
        print("empty-slot fill: console +-inf, PC library +-FLT_MAX in %d slots of %d BOX32 nodes (not a difference)"
              % (fill["slots"], fill["nodes"]))
    print("structure: %d of %d bytes and %d words differ; %d stray bytes (%d after a padded group); "
          "%d heap bytes outside the output changed" % (r["structure_bytes_differ"], r["structure_bytes"],
                                                         r["structure_words_differ"], r["stray_bytes"],
                                                         r["stray_bytes_in_padded_shadow"], r["outside_bytes_changed"]))
    report.update(rep)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
