"""tools/bvh_rehydrate_gpu.py without a GPU or game files: the heap layout, the root tables, the dispatch order, the
padded last groups and the comparison, on synthetic dehydrated BLASes, with a fake vk_compute_test that does what the
root-table layout in the tool's docstring says each program does.

    python -I tests/test_bvh_rehydrate_gpu.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bvh_rehydrate as R  # noqa: E402
import bvh_rehydrate_gpu as G  # noqa: E402

FLAGS = 0x80000000 | R.FLAG_DEHYDRATED | 0x00020006


def ptr(offset, kind):
    return (offset >> 6) << 3 | kind


def grid(n_quads, z=0.0):
    """A strip of unit quads: 2 triangles each, uint32 indices."""
    verts, idx = [], []
    for q in range(n_quads + 1):
        verts += [(q, 0.0, z), (q, 1.0, z + 0.25 * (q % 3))]
    for q in range(n_quads):
        a = 2 * q
        idx += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
    return b"".join(struct.pack("<3f", *v) for v in verts), struct.pack("<%dI" % len(idx), *idx)


def make_blas(n_leaves, levels, compression=R.NO_TRANSFORM, bounds=(0.5, 0, 0, 40, 1, 1), n_geoms=2):
    """A dehydrated BLAS per docs/bvh.md §4a. levels: [(BOX32 nodes, BOX16 nodes)] bottom-up; each level's nodes take
    the pointers of the level below in order, as evenly as they can (1 to 4 each). Returns the bytes."""
    elements = 0x2000
    chunks = -(-n_leaves // R.LEAVES_PER_CHUNK)
    nodes = elements + ((n_leaves * 64 + 127) // 128) * 128
    # node space: rehydrated sizes, run by run
    runs, cur = [], nodes
    for n32, n16 in levels:
        runs.append((cur, n32, cur + 128 * n32, n16))
        cur += 128 * n32 + 64 * n16
    table_at = (cur + 63) // 64 * 64
    total = (table_at + 8 * len(levels) + 4 + 63) // 64 * 64
    src = bytearray(total)
    # leaves: geometry i % n_geoms, paired every third leaf
    for c in range(chunks):
        lo, hi = c * R.LEAVES_PER_CHUNK, min(n_leaves, (c + 1) * R.LEAVES_PER_CHUNK)
        recs = []
        for i in range(lo, hi):
            g = i % n_geoms
            p = (i // n_geoms) % 40
            prim1 = 2 * p + 1 if i % 3 == 0 else None
            order = (0x09, 0x06, 0x24)[i % 3] | (0x80 if prim1 is not None and i % 2 else 0)
            recs.append((g, 0x01 | (0x80 if prim1 is not None else 0), 2 * p, prim1, order))
        arr = bytearray()
        for g, *_ in recs:
            arr += g.to_bytes(3, "little")
        for _, fl, *_ in recs:
            arr.append(fl)
        for _, _, p0, *_ in recs:
            arr += p0.to_bytes(3, "little")
        for _, _, _, p1, _ in recs:
            arr += (0xFFFFFF if p1 is None else p1).to_bytes(3, "little")
        for *_, o in recs:
            arr.append(o)
        at = elements + c * R.CHUNK_STRIDE
        src[at:at + len(arr)] = arr
    below = [ptr(elements + 64 * i, R.TRI0) for i in range(n_leaves)]
    pairs, k = [], 0
    for (s32, n32, s16, n16) in runs:
        count = n32 + n16
        share = [len(below) // count + (1 if j < len(below) % count else 0) for j in range(count)]
        assert all(1 <= s <= 4 for s in share), share
        made, pos = [], 0
        for wide, start, n in ((True, s32, n32), (False, s16, n16)):
            size, gbytes = (128, 8192) if wide else (64, 4096)
            for g0 in range(0, n, 64):
                words, kids, values = [], [], []
                for j in range(g0, min(n, g0 + 64)):
                    ch = below[pos:pos + share[k]]
                    pos += share[k]
                    k += 1
                    mask = 0
                    if j % 5 == 1:          # one stored override per fifth node: slot 0's max x
                        mask = 1 << 3
                        values.append(1000.0 if wide else 0x6400)
                    words.append(len(ch) << 24 | mask)
                    kids += ch
                    made.append(ptr(start + size * j, R.BOX32 if wide else R.BOX16))
                body = struct.pack("<%dI" % len(words), *words) + struct.pack("<%dI" % len(kids), *kids)
                body += struct.pack("<%df" % len(values), *values) if wide else struct.pack("<%dH" % len(values), *values)
                at = start + size * g0
                assert len(body) <= size * len(words)
                src[at:at + len(body)] = body
        k = 0
        pairs.append((s16 - nodes, s16 + 64 * n16 - nodes))
        below = made
    assert len(below) == 1
    root = below[0]
    units0 = (nodes - elements) // 64
    parents = (64 - 4 * units0) & R.M32
    assert 64 + 4 * ((cur - nodes) // 64) <= elements
    struct.pack_into("<4I6f6I", src, 0, FLAGS, total // 64 - 1, 0x1234, n_leaves, *bounds, nodes, 0, root, elements,
                     parents, compression)
    lv = b"".join(struct.pack("<2I", a, b) for a, b in pairs) + struct.pack("<I", len(pairs))
    src[total - len(lv):] = lv
    return bytes(src)


def geometries(n=2, transform=None, index16=False):
    out = []
    for g in range(n):
        vb, ib = grid(40, z=float(g))
        if index16:
            ib = struct.pack("<%dH" % (len(ib) // 4), *struct.unpack("<%dI" % (len(ib) // 4), ib))
        out.append(R.Geometry(vb, 12, ib, index16, transform))
    return out


class FakeHeader:
    """What a box program knows: the table's words. Parent entries are written wherever they land in the output."""

    def __init__(self, table, size):
        self.offset_to_nodes = 0
        self.offset_to_parents, self.offset_to_elements, upd = struct.unpack_from("<3I", table, 4)
        self.flags = R.FLAG_UPDATABLE if upd else 0
        self.size = size

    def stored_parent_entry(self, node_offset, size):
        unit = ((node_offset - self.offset_to_elements) & R.M32) >> 6
        e = (self.offset_to_parents + 4 * unit) & R.M32
        return e if e + 4 <= size else None


def ideal_gpu(heap, program, groups):
    """Run one dispatch the way the tool's docstring says the program works, reading only the table and the heap."""
    table = bytes(heap[:G.TABLE_BYTES])
    if program == "LeafRehydration":
        t = struct.unpack_from("<3f", table, 0)
        elements, = struct.unpack_from("<I", table, 0x0C)
        s = struct.unpack_from("<3f", table, 0x10)
        on, = struct.unpack_from("<I", table, 0x1C)
        rows = [struct.unpack_from("<3f", table, at) for at in (0x20, 0x30, 0x40)]
        n, = struct.unpack_from("<I", table, 0x2C)
        src_va, dst_va, desc_va = struct.unpack_from("<3Q", table, 0x4C)
        frame = object.__new__(R.LeafFrame)
        frame.on, frame.t, frame.s, frame.r = bool(on), list(t), list(s), tuple(rows)
        src = bytes(heap[src_va:])
        geoms = []
        for gi in range(64):
            at = desc_va + 48 * gi
            t_va, fmt, vfmt, n_idx, n_vtx, ib, vb, stride = struct.unpack_from("<QIIIIQQQ", heap, at)
            if vb == 0:
                break
            assert vfmt == G.VERTEX_R32G32B32_FLOAT
            idx = None
            if fmt in (G.INDEX_R32_UINT, G.INDEX_R16_UINT):
                idx = bytes(heap[ib:ib + n_idx * (4 if fmt == G.INDEX_R32_UINT else 2)])
            m = None
            if t_va:
                v = struct.unpack_from("<12f", heap, t_va)
                m = tuple(tuple(v[4 * r:4 * r + 4]) for r in range(3))
            geoms.append(R.Geometry(bytes(heap[vb:vb + n_vtx * stride]), stride, idx, fmt == G.INDEX_R16_UINT, m))
        hdr = object.__new__(R.Header)
        hdr.num_elements, hdr.offset_to_elements = min(n, groups * 2048), elements
        dst = bytearray(heap[dst_va:])
        R.write_leaves(src, dst, hdr, geoms, frame)
        heap[dst_va:] = dst
        return
    wide = program == "Box32Rehydration"
    first, = struct.unpack_from("<I", table, 0)
    src_base, dst_base = struct.unpack_from("<2Q", table, 0x4C)
    dst = bytearray(heap[dst_base:])
    hdr = FakeHeader(table, len(dst))
    # every group has 64 lanes: R's model with 64 nodes per group is what the program computes from EXEC
    R.write_box_groups(bytes(heap[src_base:]), dst, hdr, first, 64 * groups, wide)
    heap[dst_base:] = dst


def fake_runner(corrupt=None, fail_at=None, heap_init=True):
    """A vk_compute_test stand-in that sets up memory as runtime/test/vk_compute_test.cpp does: a buffer of
    VKC_OUT_WORDS words filled with 0xEE, then the VKC_HEAP_INIT file copied to its start (heap_init=False: a build
    from before that variable, which ignores it), ideal_gpu, then VKC_DUMP. corrupt(i, program, heap) may change
    the result of dispatch i; fail_at = (i, status) makes dispatch i fail."""
    calls = []

    def run(vkc, spv, init, dump, heap_words, groups, validation, timeout):
        i = len(calls)
        calls.append((Path(spv).stem, groups))
        if fail_at and fail_at[0] == i:
            return fail_at[1], "boom"
        heap = bytearray([0xEE]) * (heap_words * 4)
        if heap_init:
            image = Path(init).read_bytes()
            assert len(image) <= len(heap)
            heap[:len(image)] = image
        try:
            ideal_gpu(heap, Path(spv).stem, groups)
        except (struct.error, ValueError, IndexError, OverflowError, MemoryError):
            if heap_init:
                raise
            # a GPU does not raise: on garbage roots the program writes something or nothing and the run exits 0
        if corrupt:
            corrupt(i, Path(spv).stem, heap)
        Path(dump).write_bytes(bytes(heap))
        return "ok", ""
    return run, calls


LEVELS = [(70, 460), (0, 133), (34, 0), (5, 4), (0, 3), (1, 0)]   # partial last groups in most runs


class Base(unittest.TestCase):
    blas = None

    @classmethod
    def setUpClass(cls):
        cls.blas = make_blas(2100, LEVELS)
        cls.geoms = geometries()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spirv = Path(self.tmp.name) / "spirv"
        self.spirv.mkdir()
        for name in G.PROGRAMS:
            (self.spirv / (name + ".spv")).write_bytes(b"\x03\x02\x23\x07")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, blas, geoms, runner, partial="pad"):
        plan, image, ref = G.build_plan(blas, geoms, 0, partial)
        hdr = R.Header(blas)
        src = blas[:hdr.total_bytes]
        work = Path(self.tmp.name) / "work"
        work.mkdir(exist_ok=True)
        return plan, G.run(plan, image, ref, src, hdr, self.spirv, "vkc", work, validation=False, timeout=5,
                           runner=runner)


class Plan(Base):
    def test_dispatch_order_and_groups(self):
        plan, image, ref = G.build_plan(self.blas, self.geoms)
        got = [(d["program"], d["groups"], d["nodes"], bool(d.get("padded"))) for d in plan["dispatches"]]
        self.assertEqual(got[0], ("LeafRehydration", 2, 2100, False), "one group per 2,048-leaf chunk")
        # level 0: 70 BOX32 = one full group + a padded group of 6; 460 BOX16 = 7 full + 12
        self.assertEqual(got[1:5], [("Box32Rehydration", 1, 64, False), ("Box32Rehydration", 1, 6, True),
                                    ("Box16Rehydration", 7, 448, False), ("Box16Rehydration", 1, 12, True)])
        self.assertEqual(got[-1], ("Box32Rehydration", 1, 1, True), "the root's level last")
        nodes = sum(d["nodes"] for d in plan["dispatches"][1:])
        self.assertEqual(nodes, sum(a + b for a, b in LEVELS))
        raw, _, _ = G.build_plan(self.blas, self.geoms, partial="raw")
        self.assertEqual([d["groups"] for d in raw["dispatches"]][:3], [2, 2, 8], "raw: whole groups over the run")

    def test_tables_and_heap(self):
        plan, image, ref = G.build_plan(self.blas, self.geoms)
        hdr = R.Header(self.blas)
        leaf = plan["dispatches"][0]["table"]
        self.assertEqual(image[:G.TABLE_BYTES], leaf, "the first table is in the image")
        self.assertEqual(struct.unpack_from("<I", leaf, 0x0C)[0], hdr.offset_to_elements)
        self.assertEqual(struct.unpack_from("<I", leaf, 0x1C)[0], 0, "no leaf frame for 0x007f7f7f")
        self.assertEqual(struct.unpack_from("<I", leaf, 0x2C)[0], 2100)
        self.assertEqual(struct.unpack_from("<3Q", leaf, 0x4C), (plan["src_va"], plan["dst_va"],
                                                                  plan["geometry_records_va"]))
        src_va, dst_va, size = plan["src_va"], plan["dst_va"], plan["structure_bytes"]
        self.assertEqual(bytes(image[src_va:src_va + size]), self.blas[:size])
        self.assertEqual(bytes(image[dst_va:dst_va + 64]), ref[:64], "header seeded")
        self.assertEqual(image[dst_va + 64], G.FILL)
        rec = struct.unpack_from("<QIIIIQQQ", image, plan["geometry_records_va"] + 48)
        self.assertEqual(rec[1:5], (42, 6, 240, 82))
        self.assertEqual(bytes(image[rec[6]:rec[6] + len(self.geoms[1].vertices)]), self.geoms[1].vertices)
        self.assertEqual(bytes(image[rec[5]:rec[5] + len(self.geoms[1].indices)]), self.geoms[1].indices)
        box = plan["dispatches"][1]["table"]
        self.assertEqual(struct.unpack_from("<4I", box, 0), (hdr.offset_to_nodes, hdr.offset_to_parents,
                                                             hdr.offset_to_elements, 0))
        self.assertEqual(struct.unpack_from("<2Q", box, 0x4C), (src_va, dst_va))
        pad = plan["dispatches"][2]
        first, = struct.unpack_from("<I", pad["table"], 0)
        base, = struct.unpack_from("<Q", pad["table"], 0x4C)
        self.assertEqual(first, hdr.offset_to_nodes + 8192, "the second BOX32 group")
        words = struct.unpack_from("<64I", image, base + first)
        self.assertEqual(words[:6], struct.unpack_from("<6I", self.blas, first))
        self.assertEqual(words[6:], (0,) * 58, "padded to 64 node words")
        spans = sorted((r["va"], r["va"] + r["bytes"]) for r in plan["regions"])
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "regions do not overlap")
        self.assertGreaterEqual(spans[0][0], G.TABLE_BYTES)

    def test_written_records_cover_the_reference(self):
        """Every non-zero byte of the CPU reference is a record, a parent entry or a seeded byte."""
        plan, image, ref = G.build_plan(self.blas, self.geoms)
        hdr = R.Header(self.blas)
        empty = bytearray(image)
        r = G.compare(empty, plan, ref, self.blas[:hdr.total_bytes], hdr)
        self.assertEqual(r["reference_bytes_not_modelled"], 0)
        self.assertEqual(r["leaves"]["match"], 0)
        self.assertGreater(r["parents"]["total"], 700)

    def test_padded_group(self):
        body = G.padded_group([2 << 24 | 8, 1 << 24], [0x11, 0x22, 0x33], b"\x01\x02\x03\x04")
        self.assertEqual(len(body), 256 + 12 + 4)
        self.assertEqual(struct.unpack_from("<3I", body, 0), (2 << 24 | 8, 1 << 24, 0))
        self.assertEqual(struct.unpack_from("<3I", body, 256), (0x11, 0x22, 0x33))

    def test_leaf_frame_and_transform_words(self):
        comp = 0xC5000000 | 0x81 << 16 | 0x7F << 8 | 0x80          # scale 4, 1, 2; translation; rotation 5
        blas = make_blas(300, [(10, 65), (0, 19), (5, 0), (0, 2), (1, 0)], compression=comp)
        m = ((0.0, 0.0, 1.0, 2.0), (0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.5))
        plan, image, ref = G.build_plan(blas, geometries(transform=m, index16=True))
        leaf = plan["dispatches"][0]["table"]
        f = R.LeafFrame(R.Header(blas))
        self.assertEqual(struct.unpack_from("<I", leaf, 0x1C)[0], 1)
        self.assertEqual(struct.unpack_from("<3f", leaf, 0x10), tuple(f.s))
        self.assertEqual(struct.unpack_from("<3f", leaf, 0x00), tuple(struct.unpack("<3f", struct.pack("<3f", *f.t))))
        self.assertEqual(struct.unpack_from("<3f", leaf, 0x30), f.r[1])
        rec = struct.unpack_from("<QIIIIQQQ", image, plan["geometry_records_va"])
        self.assertEqual(rec[1], 57, "R16_UINT")
        self.assertEqual(struct.unpack_from("<12f", image, rec[0]), tuple(v for row in m for v in row))

    def test_heap_limit(self):
        old = G.MAX_HEAP_BYTES
        G.MAX_HEAP_BYTES = 1 << 16
        try:
            with self.assertRaises(ValueError):
                G.build_plan(self.blas, self.geoms)
        finally:
            G.MAX_HEAP_BYTES = old


class Runs(Base):
    def test_ideal_gpu_matches_the_reference(self):
        run, calls = fake_runner()
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        r = rep["result"]
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(len(calls), len(plan["dispatches"]))
        self.assertEqual(r["leaves"], {"total": 2100, "match": 2100, "differ": 0, "bytes_differ": 0})
        self.assertEqual(r["box32"]["match"], 110)
        self.assertEqual(r["box16"]["match"], 600)
        self.assertEqual(r["parents"]["differ"], 0)
        self.assertEqual(r["structure_bytes_differ"], 0)
        self.assertGreater(r["stray_bytes_in_padded_shadow"], 0, "the root group's extra lanes write past it")
        self.assertEqual(r["stray_bytes"], r["stray_bytes_in_padded_shadow"])
        self.assertEqual(r["outside_bytes_changed"], 0)

    def test_frame_transform_and_index16_run(self):
        comp = 0xC5000000 | 0x81 << 16 | 0x7F << 8 | 0x80
        blas = make_blas(300, [(10, 65), (0, 19), (5, 0), (0, 2), (1, 0)], compression=comp)
        m = ((0.0, 0.0, 1.0, 2.0), (0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.5))
        run, _ = fake_runner()
        plan, rep = self.run_tool(blas, geometries(transform=m, index16=True), run)
        self.assertTrue(rep["ok"], rep["result"])

    def test_one_wrong_byte_per_record_kind(self):
        def corrupt(i, program, heap):
            plan = self.plan
            dst = plan["dst_va"]
            hdr = R.Header(self.blas)
            if program == "LeafRehydration":
                heap[dst + hdr.offset_to_elements + 64 * 2050 + 4] ^= 1        # a leaf of the second chunk
            elif i == len(plan["dispatches"]) - 1:
                root = (hdr.root >> 3) << 6
                heap[dst + root + 20] ^= 0x40                                   # the root's slot 0 bounds
        self.plan, _, _ = G.build_plan(self.blas, self.geoms)
        run, _ = fake_runner(corrupt)
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        r = rep["result"]
        self.assertFalse(rep["ok"])
        self.assertEqual((r["leaves"]["differ"], r["leaves"]["bytes_differ"]), (1, 1))
        self.assertGreaterEqual(r["box32"]["differ"], 1)
        self.assertEqual(r["parents"]["differ"], 0)
        self.assertGreaterEqual(r["structure_words_differ"], 2)

    def test_a_failed_dispatch_stops_the_chain(self):
        run, calls = fake_runner(fail_at=(1, "timeout"))
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        self.assertFalse(rep["ok"])
        self.assertEqual([s["status"] for s in rep["steps"]], ["ok", "timeout"])
        self.assertEqual(rep["result"]["leaves"]["match"], 2100, "the leaves are still compared")
        self.assertEqual(rep["result"]["box32"]["match"], 0)

    def test_stray_write_outside_the_output(self):
        def corrupt(i, program, heap):
            if i == 0:
                heap[G.TABLE_BYTES + 3] ^= 0xFF
        run, _ = fake_runner(corrupt)
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["result"]["outside_bytes_changed"], 1)
        self.assertEqual(rep["steps"][0]["outside_bytes_changed"], 1)
        self.assertEqual([s["status"] for s in rep["steps"]], ["input changed"])

    def test_a_vk_compute_test_that_ignores_the_heap_image(self):
        """An old build runs every program on its 0xEE fill: the first dispatch must fail, not the comparison."""
        run, calls = fake_runner(heap_init=False)
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        self.assertFalse(rep["ok"])
        self.assertEqual(len(calls), 1, "the chain stops at the first dispatch")
        self.assertEqual(rep["steps"][0]["status"], "heap not loaded")

    def test_a_program_that_writes_its_input_stops_the_chain(self):
        def corrupt(i, program, heap):
            if i == 0:
                heap[self.src_va + 100] ^= 1
        self.src_va = G.build_plan(self.blas, self.geoms)[0]["src_va"]
        run, calls = fake_runner(corrupt)
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        self.assertEqual(rep["steps"][0]["status"], "input changed")
        self.assertEqual(len(calls), 1)

    def test_seed_leaves_runs_the_box_programs_alone(self):
        plan, image, ref = G.build_plan(self.blas, self.geoms, seed_leaves=True)
        self.assertNotIn("LeafRehydration", [d["program"] for d in plan["dispatches"]])
        hdr = R.Header(self.blas)
        at = hdr.offset_to_elements + 64 * 2050
        self.assertEqual(bytes(image[plan["dst_va"] + at:plan["dst_va"] + at + 64]), ref[at:at + 64])
        run, calls = fake_runner()
        work = Path(self.tmp.name) / "seed"
        work.mkdir()
        rep = G.run(plan, image, ref, self.blas[:hdr.total_bytes], hdr, self.spirv, "vkc", work, validation=False,
                    timeout=5, runner=run)
        self.assertTrue(rep["ok"], rep["result"])
        self.assertEqual(rep["result"]["leaves"]["total"], 0, "seeded leaves are not counted as GPU work")
        self.assertEqual(rep["result"]["box16"]["match"], 600)
        self.assertNotIn("LeafRehydration", [c[0] for c in calls])

    def test_seed_leaves_box_dispatch_that_corrupts_a_seeded_leaf_fails(self):
        # Envyre on #8082: no record count sees a seeded leaf, so `ok` must also require structure_bytes_differ == 0
        plan, image, ref = G.build_plan(self.blas, self.geoms, seed_leaves=True)
        hdr = R.Header(self.blas)
        at = plan["dst_va"] + hdr.offset_to_elements + 64 * 5

        def corrupt(i, program, heap):
            if i == 0:
                heap[at] ^= 0xFF
        run, _ = fake_runner(corrupt)
        work = Path(self.tmp.name) / "seedbad"
        work.mkdir()
        rep = G.run(plan, image, ref, self.blas[:hdr.total_bytes], hdr, self.spirv, "vkc", work, validation=False,
                    timeout=5, runner=run)
        self.assertFalse(rep["ok"], rep["result"])
        self.assertEqual(rep["result"]["structure_bytes_differ"], 1)
        self.assertEqual(rep["result"]["box32"]["differ"] + rep["result"]["box16"]["differ"], 0)

    def test_leaf_count_ladder(self):
        hdr = R.Header(self.blas)
        for count, groups in ((0, 1), (1, 1), (181, 1), (2048, 1)):
            plan, image, ref = G.build_plan(self.blas, self.geoms, leaf_count=count)
            self.assertEqual([(d["program"], d["groups"]) for d in plan["dispatches"]], [("LeafRehydration", groups)])
            self.assertEqual(struct.unpack_from("<I", plan["dispatches"][0]["table"], 0x2C)[0], count)
            run, _ = fake_runner()
            work = Path(self.tmp.name) / ("lc%d" % count)
            work.mkdir()
            rep = G.run(plan, image, ref, self.blas[:hdr.total_bytes], hdr, self.spirv, "vkc", work,
                        validation=False, timeout=5, runner=run)
            self.assertTrue(rep["ok"], (count, rep["result"]))
            self.assertEqual(rep["result"]["leaves"]["total"], count)
            self.assertEqual(rep["result"]["box32"]["total"], 0)
        with self.assertRaises(ValueError):
            G.build_plan(self.blas, self.geoms, leaf_count=2049)
        # the arrays are repacked for the smaller count: leaf 0's base primitive sits right after 4 bytes now
        plan, image, ref = G.build_plan(self.blas, self.geoms, leaf_count=1)
        at = plan["src_va"] + hdr.offset_to_elements
        self.assertEqual(image[at + 4:at + 7], self.blas[hdr.offset_to_elements + 4 * 2048:][:3])
        with self.assertRaises(ValueError):
            G.build_plan(self.blas, self.geoms, seed_leaves=True, leaf_count=1)

    def test_leaf_count_run_that_writes_past_the_count_fails(self):
        def corrupt(i, program, heap):
            dst = self.plan["dst_va"]
            heap[dst + R.Header(self.blas).offset_to_elements + 64 * 5] ^= 1     # leaf 5 of a 1-leaf run
        self.plan, image, ref = G.build_plan(self.blas, self.geoms, leaf_count=1)
        hdr = R.Header(self.blas)
        run, _ = fake_runner(corrupt)
        work = Path(self.tmp.name) / "past"
        work.mkdir()
        rep = G.run(self.plan, image, ref, self.blas[:hdr.total_bytes], hdr, self.spirv, "vkc", work,
                    validation=False, timeout=5, runner=run)
        self.assertFalse(rep["ok"])
        self.assertGreater(rep["result"]["stray_bytes"], 0)

    def console_fill(self, replacement):
        """A corrupt hook that rewrites every empty BOX32 slot after a Box32 dispatch, as the console program fills it."""
        plan, _, _ = G.build_plan(self.blas, self.geoms)
        hdr = R.Header(self.blas)
        b32 = G.written_records(self.blas[:hdr.total_bytes], hdr)[1]
        library = struct.pack("<6I", *[G.FLT_MAX32] * 3 + [G.NEG_FLT_MAX32] * 3)

        def corrupt(i, program, heap):
            if program != "Box32Rehydration":
                return
            dst = plan["dst_va"]
            for o in b32:
                for slot in range(4):
                    at = dst + o + 16 + 24 * slot
                    if struct.unpack_from("<I", heap, dst + o + 4 * slot)[0] == R.INVALID and heap[at:at + 24] == library:
                        heap[at:at + 24] = replacement
        return corrupt

    def test_console_empty_slot_fill_is_its_own_category(self):
        console = struct.pack("<6I", *[G.INF32] * 3 + [G.NEG_INF32] * 3)
        run, _ = fake_runner(self.console_fill(console))
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        r = rep["result"]
        self.assertTrue(rep["ok"], r)
        self.assertEqual(r["box32"]["differ"], 0)
        self.assertGreater(r["box32_empty_slot_fill"]["slots"], 0)
        self.assertGreaterEqual(r["box32_empty_slot_fill"]["slots"], r["box32_empty_slot_fill"]["nodes"])
        self.assertEqual(r["structure_bytes_differ"], 0)

    def test_other_empty_slot_values_still_differ(self):
        zeros = bytes(24)
        run, _ = fake_runner(self.console_fill(zeros))
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        r = rep["result"]
        self.assertFalse(rep["ok"])
        self.assertGreater(r["box32"]["differ"], 0)
        self.assertEqual(r["box32_empty_slot_fill"]["slots"], 0)

    def test_library_fill_needs_no_category(self):
        run, _ = fake_runner()
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        self.assertEqual(rep["result"]["box32_empty_slot_fill"], {"nodes": 0, "slots": 0})

    def test_missing_program(self):
        (self.spirv / "Box16Rehydration.spv").unlink()
        run, _ = fake_runner()
        plan, rep = self.run_tool(self.blas, self.geoms, run)
        self.assertEqual(rep["steps"][-1]["status"], "missing")
        self.assertFalse(rep["ok"])


class Cli(Base):
    def files(self):
        d = Path(self.tmp.name)
        (d / "blas.bin").write_bytes(b"\0" * 256 + self.blas)
        args = ["--header-offset", "256"]
        for i, g in enumerate(self.geoms):
            (d / ("v%d" % i)).write_bytes(g.vertices)
            (d / ("i%d" % i)).write_bytes(g.indices)
            args += ["--geom", "%s,%s" % (d / ("v%d" % i), d / ("i%d" % i))]
        return [str(d / "blas.bin")] + args

    def test_plan_only(self):
        d = Path(self.tmp.name)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = G.main(self.files() + ["--plan-only", "--work", str(d / "w"), "--json", str(d / "plan.json")])
        self.assertEqual(code, 0)
        rep = json.loads((d / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["plan"]["leaves"], 2100)
        self.assertNotIn("table", rep["plan"]["dispatches"][0])
        self.assertTrue((d / "w" / "heap00.in").is_file())
        self.assertTrue((d / "w" / ("table%02d.bin" % (len(rep["plan"]["dispatches"]) - 1))).is_file())
        self.assertIn("dispatches", out.getvalue())

    def test_spirv_required_and_bad_input(self):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                G.main(self.files())
            self.assertEqual(G.main([os.path.join(self.tmp.name, "missing.bin")]), 2)

    def test_vk_compute_test_without_heap_init(self):
        d = Path(self.tmp.name)
        old = d / "old_vkc.exe"
        old.write_bytes(b"MZ... VKC_DUMP VKC_OUT_WORDS ...")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = G.main(self.files() + ["--spirv", str(self.spirv), "--vkc", str(old)])
        self.assertEqual(code, 2)
        self.assertIn("VKC_HEAP_INIT", err.getvalue())
        new = d / "new_vkc.exe"
        new.write_bytes(b"MZ... VKC_HEAP_INIT ...")
        self.assertTrue(G.supports_heap_init(new))

    def test_missing_vk_compute_test(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                G.main(self.files() + ["--spirv", str(self.spirv), "--build", self.tmp.name])
        self.assertIn("vk_compute_test", str(e.exception))


if __name__ == "__main__":
    unittest.main()
