"""Synthetic unit tests for meshlet_page.py upgrades (17-bit bases, <HHII + 0x4000 tail, multi-LOD, CPU pages).

Run with: python -I tests/test_meshlet_page_lod.py
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import meshlet_page as mp


def _align16(buf):
    rem = len(buf) & 15
    if rem:
        buf.extend(bytes(16 - rem))
    return buf


def _make_group_bytes(meshlet_specs, flags=0x8000, tail_u16s=None, vref_base=0):
    """Build a valid synthetic meshlet group.

    meshlet_specs: sequence of (vcount, tcount) pairs, each 1..128.
    """
    meshlets = len(meshlet_specs)
    total_tris = sum(tc for _, tc in meshlet_specs)
    total_refs = sum(vc for vc, _ in meshlet_specs)
    records = bytearray()
    tris = bytearray()
    vbase = tbase = 0
    for m, (vc, tc) in enumerate(meshlet_specs):
        assert 1 <= vc <= 128 and 1 <= tc <= 128
        w0 = ((vc - 1) & 0x7F) | (((tc - 1) & 0x7F) << 7)
        if vbase & 0x10000:
            w0 |= 1 << 14
        if tbase & 0x10000:
            w0 |= 1 << 15
        w1 = (vbase & 0xFFFF) | ((tbase & 0xFFFF) << 16)
        records.extend(struct.pack("<4I", w0, w1, 0, 0))
        for t in range(tc):
            a = t % vc
            b = (t + 1) % vc
            c = (t + 2) % vc
            tris.extend(struct.pack("<I", a | (b << 7) | (c << 14) | (m << 21)))
        vbase += vc
        tbase += tc
    refs = bytearray()
    for i in range(total_refs):
        refs.extend(struct.pack("<H", (vref_base + (i % 64)) & 0xFFFF))
    out = bytearray(struct.pack("<HHII", meshlets, flags, total_tris, total_refs))
    out.extend(records)
    out.extend(tris)
    out.extend(refs)
    ref_end = len(out)
    if flags & 0x4000:
        entries = list(tail_u16s) if tail_u16s is not None else [0xFFFF] * meshlets
        assert len(entries) == meshlets
        for val in entries:
            out.extend(struct.pack("<H", val))
        alloc_size = (((ref_end + 3) & ~3) + 2 * meshlets + 15) & ~15
        out.extend(bytes(alloc_size - len(out)))
    else:
        _align16(out)
    return bytes(out), meshlets, total_tris, total_refs


def _make_4240_block(vertices, *, position_only=False):
    """Build a synthetic 0x4042 vertex block (either standard with slot 2 or position-only)."""
    header = bytearray(mp.OTHER_HEADER_SIZE)
    struct.pack_into("<I", header, 0, mp.OTHER_BLOCK_MAGIC)  # slot 0: tag=0x42, off=0x80
    struct.pack_into("<3f", header, 0x60, 0.0, 0.0, 0.0)
    struct.pack_into("<I", header, 0x6C, 2)
    struct.pack_into("<3f", header, 0x70, 1.0 / 0xFFFFFF, 1.0 / 0xFFFFFF, 1.0 / 0xFFFF)
    if position_only:
        block = bytearray(header)
        block.extend(bytes(8 * vertices))
        return bytes(block)
    nrm_off = mp.OTHER_HEADER_SIZE + 8 * vertices
    struct.pack_into("<I", header, 2 * 4, 0x21 | ((nrm_off // 2) << 8))
    block = bytearray(header)
    block.extend(bytes(8 * vertices))
    block.extend(bytes(4 * vertices))
    return bytes(block)


def _make_3063_block(vertices, *, position_only=False):
    """Build a synthetic 0x3063 float3 vertex block (standard or position-only)."""
    header = bytearray(mp.HEADER_SIZE)
    struct.pack_into("<I", header, 0, mp.BLOCK_MAGIC)  # slot 0: tag=0x63, off=0x60
    if position_only:
        block = bytearray(header)
        for i in range(vertices):
            block.extend(struct.pack("<3f", float(i), 0.0, 0.0))
        return bytes(block)
    uv_off = mp.HEADER_SIZE + 12 * vertices
    struct.pack_into("<I", header, 3 * 4, 0x43 | ((uv_off // 2) << 8))
    block = bytearray(header)
    for i in range(vertices):
        block.extend(struct.pack("<3f", float(i), 0.0, 0.0))
    for _ in range(vertices):
        block.extend(struct.pack("<2f", 0.0, 0.0))
    return bytes(block)


def _make_paired_drawable(lod_specs, *, drawable_offset=0):
    """Build paired (system, graphics) buffers with arbitrary LOD 0..3 specs.

    lod_specs: dict[int, list[dict]] mapping lod -> list of geometry spec dicts:
      - vertices (int)
      - meshlets (list[(vc, tc)])
      - flags (int, default 0x8000)
      - magic (int, default OTHER_BLOCK_MAGIC)
      - position_only (bool, default False)
      - tail_u16s (optional list[int])
    Returns (system_bytes, graphics_bytes, lod_geo_offsets).
    """
    sys_buf = bytearray(drawable_offset + 176)
    gpu_buf = bytearray()

    def alloc_sys(size):
        _align16(sys_buf)
        off = len(sys_buf)
        sys_buf.extend(bytes(size))
        return off

    def alloc_gpu(blob):
        _align16(gpu_buf)
        off = len(gpu_buf)
        gpu_buf.extend(blob)
        _align16(gpu_buf)
        return off

    lod_geo_offsets = {}
    for lod, specs in sorted(lod_specs.items()):
        geo_offs = []
        for spec in specs:
            verts = spec["vertices"]
            magic = spec.get("magic", mp.OTHER_BLOCK_MAGIC)
            pos_only = spec.get("position_only", False)
            gflags = spec.get("flags", 0x8000)
            if magic == mp.OTHER_BLOCK_MAGIC:
                vblob = _make_4240_block(verts, position_only=pos_only)
                hdr_sz, pos_sz, pos_fmt = mp.OTHER_HEADER_SIZE, 8, 16
            else:
                vblob = _make_3063_block(verts, position_only=pos_only)
                hdr_sz, pos_sz, pos_fmt = mp.HEADER_SIZE, 12, 6
            voff = alloc_gpu(vblob)
            gblob, m_cnt, t_cnt, r_cnt = _make_group_bytes(
                spec["meshlets"], flags=gflags, tail_u16s=spec.get("tail_u16s"))
            goff = alloc_gpu(gblob)

            layout = alloc_sys(320)
            struct.pack_into("<I", sys_buf, layout, hdr_sz)
            sys_buf[layout + 0xD0] = pos_sz
            sys_buf[layout + 0x104] = pos_fmt
            struct.pack_into("<I", sys_buf, layout + 0x138, (verts << 10) | 1)

            vb = alloc_sys(80)
            struct.pack_into("<IH", sys_buf, vb + 8, len(vblob) // 4, 4)
            struct.pack_into("<Q", sys_buf, vb + 24, 0x60000000 + voff)
            struct.pack_into("<Q", sys_buf, vb + 64, 0x50000000 + layout)

            ib = alloc_sys(64)
            struct.pack_into("<IH", sys_buf, ib + 8, len(gblob) // 4, 4)
            struct.pack_into("<Q", sys_buf, ib + 24, 0x60000000 + goff)

            desc = alloc_sys(16)
            binding = ((t_cnt >> 16) & 0x7F) | ((r_cnt & 0x7FFFFF) << 8)
            if gflags & 0x4000:
                binding |= 1 << 31
            struct.pack_into("<Q", sys_buf, desc, 0x50000000 + ib)
            struct.pack_into("<HHI", sys_buf, desc + 8, m_cnt, t_cnt & 0xFFFF, binding)

            geo = alloc_sys(160)
            struct.pack_into("<Q", sys_buf, geo + 8, 0x50000000 + vb)
            struct.pack_into("<Q", sys_buf, geo + 64, 0x50000000 + desc)
            geo_offs.append(geo)

        entries = alloc_sys(8 * len(geo_offs))
        for idx, go in enumerate(geo_offs):
            struct.pack_into("<Q", sys_buf, entries + 8 * idx, 0x50000000 + go)

        model = alloc_sys(56)
        struct.pack_into("<Q", sys_buf, model + 8, 0x50000000 + entries)
        struct.pack_into("<HH", sys_buf, model + 16, len(geo_offs), len(geo_offs))
        struct.pack_into("<H", sys_buf, model + 54, len(geo_offs))

        table = alloc_sys(8)
        struct.pack_into("<Q", sys_buf, table, 0x50000000 + model)

        lod_rec = alloc_sys(16)
        struct.pack_into("<Q", sys_buf, lod_rec, 0x50000000 + table)
        struct.pack_into("<HH", sys_buf, lod_rec + 8, 1, 1)

        struct.pack_into("<Q", sys_buf, drawable_offset + 80 + 8 * lod, 0x50000000 + lod_rec)
        lod_geo_offsets[lod] = geo_offs

    return bytes(sys_buf), bytes(gpu_buf), lod_geo_offsets


def _pack_zlib_stream(payload):
    comp = bytearray(zlib.compress(payload))
    _align16(comp)
    return bytes(comp)


class MeshletGroupUpgradeTests(unittest.TestCase):
    def test_meshlet_bases_17bit_vbase_and_tbase(self):
        self.assertEqual(mp.meshlet_bases(0, 0x00020001), (1, 2))
        self.assertEqual(mp.meshlet_bases(1 << 14, 0x00020001), (0x10001, 2))
        self.assertEqual(mp.meshlet_bases(1 << 15, 0x00020001), (1, 0x10002))
        self.assertEqual(mp.meshlet_bases((1 << 14) | (1 << 15), 0xFFFF0005), (0x10005, 0x1FFFF))

    def test_parse_group_and_triangles_exceeding_65535_refs_and_tris(self):
        # 520 meshlets x (128 verts, 128 tris) -> 66,560 refs and 66,560 tris (> 65,535)
        specs = [(128, 128)] * 520
        blob, m_cnt, t_cnt, r_cnt = _make_group_bytes(specs, flags=0x8000)
        self.assertGreater(t_cnt, 65535)
        self.assertGreater(r_cnt, 65535)
        rec, end = mp.parse_group(blob, 0)
        self.assertEqual(rec["meshlets"], 520)
        self.assertEqual(rec["triangles"], 66560)
        self.assertEqual(rec["vertex_refs"], 66560)
        self.assertEqual(rec["flags"], 0x8000)
        self.assertEqual((end + 15) & ~15, len(blob))

        tris = list(mp.group_triangles(blob, rec))
        self.assertEqual(len(tris), 66560)
        # Verify triangle from meshlet 513 (past 65,536 boundary) resolves without wrapping.
        self.assertEqual(tris[513 * 128], (0, 1, 2))

        # Corrupting bit 14 on meshlet 512 (vbase = 65536) must be rejected.
        corrupt_v = bytearray(blob)
        off_512 = 12 + 16 * 512
        w0 = struct.unpack_from("<I", corrupt_v, off_512)[0]
        struct.pack_into("<I", corrupt_v, off_512, w0 & ~(1 << 14))
        with self.assertRaises(mp.PageError):
            mp.parse_group(corrupt_v, 0)

        # Corrupting bit 15 on meshlet 512 (tbase = 65536) must be rejected.
        corrupt_t = bytearray(blob)
        struct.pack_into("<I", corrupt_t, off_512, w0 & ~(1 << 15))
        with self.assertRaises(mp.PageError):
            mp.parse_group(corrupt_t, 0)

    def test_group_flags_0xc000_0xe000_per_meshlet_u16_tail(self):
        # Pick (vc=3, tc=1) with 1 meshlet so ref_end = 12 + 16 + 4 + 6 = 38 (mod 4 == 2),
        # ref_end + 2*1 = 40, ((38+3)&~3 + 2 + 15)&~15 = 48.
        # Also pick (vc=7, tc=1) with 1 meshlet so ref_end = 12 + 16 + 4 + 14 = 46 (mod 4 == 2),
        # ref_end + 2*1 = 48 (multiple of 16!), allocated size = (((46+3)&~3) + 2 + 15)&~15 = 64 (+16 extra!).
        for flags, specs, tail in [
            (0xC000, [(3, 1)], [0x0004]),
            (0xE070, [(7, 1)], [0x0038]),
            (0xC002, [(4, 2), (5, 2), (3, 1)], [0x0001, 0xFFFF, 0x0007]),
        ]:
            with self.subTest(flags=hex(flags)):
                blob, m_cnt, t_cnt, r_cnt = _make_group_bytes(specs, flags=flags, tail_u16s=tail)
                rec, end = mp.parse_group(blob, 0)
                self.assertEqual(rec["flags"], flags)
                self.assertEqual(rec["meshlets"], m_cnt)
                self.assertEqual(end, 12 + 16 * m_cnt + 4 * t_cnt + 2 * r_cnt + 2 * m_cnt)
                self.assertFalse(any(blob[end:]))

                # Chained in a raw page followed immediately by a second group must parse cleanly.
                second, _, _, _ = _make_group_bytes([(4, 2)], flags=0x8000)
                page = blob + second
                chunks = mp.parse_page(page)
                self.assertEqual(len(chunks), 2)
                self.assertEqual(chunks[0]["flags"], flags)
                self.assertEqual(chunks[1]["flags"], 0x8000)

        # Truncated per-meshlet tail must be rejected.
        blob, _, _, _ = _make_group_bytes([(4, 1), (4, 1)], flags=0xC000, tail_u16s=[1, 2])
        rec_end = 12 + 32 + 8 + 16 + 4
        with self.assertRaises(mp.PageError):
            mp.parse_group(blob[:rec_end - 2], 0)

    def test_draw_binding_high_poly_0x4000_tail_and_3063_layout(self):
        # High-poly (>65,535 tris & refs) + 0xC000 tail + 0xE070 edge-alignment + 0x3063 float3 layout
        system, graphics, geos = _make_paired_drawable({
            0: [
                {"vertices": 256, "meshlets": [(128, 128)] * 520, "flags": 0x8000, "magic": mp.OTHER_BLOCK_MAGIC},
                {"vertices": 64, "meshlets": [(7, 1)], "flags": 0xE070, "tail_u16s": [0x0038],
                 "magic": mp.OTHER_BLOCK_MAGIC},
                {"vertices": 64, "meshlets": [(6, 2), (6, 2)], "flags": 0xC000, "tail_u16s": [0x0004, 0xFFFF],
                 "magic": mp.BLOCK_MAGIC},
            ],
        })
        for geo in geos[0]:
            bound = mp.draw_binding(system, graphics, geo)
            self.assertIn("vertex_block", bound)
            self.assertIn("meshlet_group", bound)

        # Mismatched descriptor bit 31 vs group flags bit 14 (0x4000) must fail.
        bad_sys = bytearray(system)
        desc = struct.unpack_from("<Q", bad_sys, geos[0][1] + 64)[0] - 0x50000000
        m, t_lo, bind = struct.unpack_from("<HHI", bad_sys, desc + 8)
        struct.pack_into("<HHI", bad_sys, desc + 8, m, t_lo, bind & 0x7FFFFFFF)
        with self.assertRaises(mp.PageError):
            mp.draw_binding(bad_sys, graphics, geos[0][1])


class PositionOnlyBlockTests(unittest.TestCase):
    def test_position_only_4240_and_3063_require_block_size(self):
        b42 = _make_4240_block(16, position_only=True)
        with self.assertRaises(mp.PageError):
            mp.parse_other_block(b42, 0)
        rec, end = mp.parse_other_block(b42, 0, block_size=len(b42))
        self.assertEqual(rec["vertices"], 16)
        self.assertEqual(rec["slots"], [[0, 0x42, mp.OTHER_HEADER_SIZE]])
        self.assertEqual(end, len(b42))

        b30 = _make_3063_block(12, position_only=True)
        with self.assertRaises(mp.PageError):
            mp.parse_block(b30, 0)
        rec30, end30 = mp.parse_block(b30, 0, block_size=len(b30))
        self.assertEqual(rec30["vertices"], 12)
        self.assertEqual(rec30["slots"], [[0, 0x63, mp.HEADER_SIZE]])
        self.assertEqual(end30, len(b30))


class MultiLodAndCpuPageTests(unittest.TestCase):
    def test_multi_lod_0_to_3_and_cli(self):
        system, graphics, geos = _make_paired_drawable({
            0: [{"vertices": 64, "meshlets": [(8, 4)], "flags": 0x8000},
                {"vertices": 64, "meshlets": [(6, 2)], "flags": 0xC000, "tail_u16s": [3]}],
            1: [{"vertices": 32, "meshlets": [(6, 2)], "flags": 0x8000}],
            2: [{"vertices": 16, "meshlets": [(4, 1)], "flags": 0x8000}],
            3: [{"vertices": 12, "meshlets": [(3, 1)], "flags": 0x8000}],
        })
        self.assertEqual(mp.drawable_lods(system, 0), [0, 1, 2, 3])
        for level in range(4):
            self.assertEqual(list(mp.drawable_geometries(system, 0, lod=level)), geos[level])
        with self.assertRaises(mp.PageError):
            list(mp.drawable_geometries(system, 0, lod=4))

        with tempfile.TemporaryDirectory() as tmp:
            sys_path = Path(tmp) / "sys.bin"
            gpu_path = Path(tmp) / "gpu.bin"
            sys_path.write_bytes(system)
            gpu_path.write_bytes(graphics)

            # Default without --lod keeps backward-compatible LOD 0 rows without 'lod' key.
            out = io.StringIO()
            with redirect_stdout(out):
                rc = mp.main([str(gpu_path), "--draw-system", str(sys_path), "--drawable-offset", "0"])
            self.assertEqual(rc, 0)
            rows = json.loads(out.getvalue())["bindings"]
            self.assertEqual([r["geometry"] for r in rows], geos[0])
            self.assertTrue(all("lod" not in r for r in rows))

            # --lod 2 selects only LOD 2.
            out = io.StringIO()
            with redirect_stdout(out):
                rc = mp.main([str(gpu_path), "--draw-system", str(sys_path), "--drawable-offset", "0", "--lod", "2"])
            self.assertEqual(rc, 0)
            rows = json.loads(out.getvalue())["bindings"]
            self.assertEqual([(r["lod"], r["geometry"]) for r in rows], [(2, geos[2][0])])

            # --lod all walks all 4 LODs.
            out = io.StringIO()
            with redirect_stdout(out):
                rc = mp.main([str(gpu_path), "--draw-system", str(sys_path), "--drawable-offset", "0", "--lod", "all"])
            self.assertEqual(rc, 0)
            rows = json.loads(out.getvalue())["bindings"]
            self.assertEqual([(r["lod"], r["geometry"]) for r in rows],
                             [(l, g) for l in range(4) for g in geos[l]])

            # --lod without --drawable-offset is rejected.
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mp.main([str(gpu_path), "--lod", "0"])

    def test_assemble_drawable_system_continuation_and_uncompressed_tail(self):
        # Place drawable header in CPU page 0 (64 KiB), LOD 0 + 1 records across page 1 (64 KiB)
        # and an uncompressed small CPU tail (128 bytes) before the first GPU zlib page.
        base_sys, graphics, geos = _make_paired_drawable({
            0: [{"vertices": 32, "meshlets": [(8, 4)], "flags": 0x8000}],
            1: [{"vertices": 16, "meshlets": [(4, 2)], "flags": 0x8000}],
        })
        # Shift all system pointers past 0x50000000 + 65536 by building drawable at offset 0 and padding
        # an artificial 2-page + tail system layout:
        # Easier: put drawable at 0, place its LOD tables in page 1 + tail by allocating a 64 KiB spacer
        # right after the 176-byte drawable header!
        full_sys = bytearray(mp.CPU_PAGE_SIZE * 2 + 256)
        # Copy drawable pointers from base_sys, relocating every >= 0x50000000 pointer by +shift
        shift = mp.CPU_PAGE_SIZE * 2 - 256
        # Build fresh with a spacer so the geometry/vb/layout tables cross the 128 KiB boundary into the tail.
        reloc_sys = bytearray(shift) + bytearray(base_sys)
        # Relocate pointers inside the copied region (every 8-byte aligned value in [0x50000000, 0x50100000))
        for off in range(shift, len(reloc_sys) - 7, 8):
            val = struct.unpack_from("<Q", reloc_sys, off)[0]
            if 0x50000000 <= val < 0x50100000:
                struct.pack_into("<Q", reloc_sys, off, val + shift)
        # Copy the relocated 176-byte drawable header back to offset 0 (page 0).
        reloc_sys[0:176] = reloc_sys[shift:shift + 176]
        # Ensure total system length is 2 * 65536 + small aligned tail (< 4096).
        rem = len(reloc_sys) % mp.CPU_PAGE_SIZE
        if rem == 0 or rem >= 4096:
            self.fail(f"unexpected test layout remainder {rem}")
        _align16(reloc_sys)
        page0 = bytes(reloc_sys[:mp.CPU_PAGE_SIZE])
        page1 = bytes(reloc_sys[mp.CPU_PAGE_SIZE:2 * mp.CPU_PAGE_SIZE])
        tail = bytes(reloc_sys[2 * mp.CPU_PAGE_SIZE:])
        self.assertTrue(16 <= len(tail) < 4096 and len(tail) % 16 == 0)

        # Stream: [zlib page0][zlib page1][uncompressed tail][zlib gpu_page]
        gpu_page = graphics + bytes(max(0, mp.CPU_PAGE_SIZE - len(graphics)))
        stream = _pack_zlib_stream(page0) + _pack_zlib_stream(page1) + tail + _pack_zlib_stream(gpu_page)

        sys0, next_pos = mp.read_zlib_stream(stream, 0)
        self.assertEqual(len(sys0), mp.CPU_PAGE_SIZE)
        assembled, gpu_start = mp.assemble_drawable_system(stream, sys0, next_pos, drawable=0)
        self.assertEqual(bytes(assembled), bytes(reloc_sys))
        gpu_inflated, _ = mp.read_zlib_stream(stream, gpu_start)
        self.assertEqual(gpu_inflated[:len(graphics)], graphics)
        self.assertEqual(mp.drawable_lods(assembled, 0), [0, 1])
        for level in (0, 1):
            for geo in mp.drawable_geometries(assembled, 0, lod=level):
                bound = mp.draw_binding(assembled, gpu_inflated, geo)
                self.assertGreater(bound["vertices"], 0)

    def test_consume_trailing_cpu_page_rules(self):
        sys64 = bytes(mp.CPU_PAGE_SIZE)
        # 1. Partial final CPU page (len % 65536 != 0): never consumed.
        self.assertIsNone(mp.consume_trailing_cpu_page(b"", bytes(1024), 0))

        # 2. Trailing full 64 KiB zlib page starting with 0x4042 or 0x3063 is a GPU page: do NOT consume.
        for magic in (mp.OTHER_BLOCK_MAGIC, mp.BLOCK_MAGIC):
            gpu64 = struct.pack("<I", magic) + bytes(mp.CPU_PAGE_SIZE - 4)
            stream = _pack_zlib_stream(gpu64)
            self.assertIsNone(mp.consume_trailing_cpu_page(stream, sys64, 0))

        # 3. Trailing zlib page < 64 KiB (even if starting with arbitrary bytes) IS consumed as CPU page.
        small_cpu = struct.pack("<I", mp.OTHER_BLOCK_MAGIC) + bytes(4092)
        gpu64 = struct.pack("<I", mp.OTHER_BLOCK_MAGIC) + bytes(mp.CPU_PAGE_SIZE - 4)
        stream = _pack_zlib_stream(small_cpu) + _pack_zlib_stream(gpu64)
        res = mp.consume_trailing_cpu_page(stream, sys64, 0)
        self.assertIsNotNone(res)
        ext_sys, nxt = res
        self.assertEqual(len(ext_sys), mp.CPU_PAGE_SIZE + len(small_cpu))
        gpu_chunk, _ = mp.read_zlib_stream(stream, nxt)
        self.assertEqual(gpu_chunk, gpu64)

        # 4. Trailing full 64 KiB zlib page NOT starting with 0x4042/0x3063 IS consumed as CPU page.
        non_mesh_cpu64 = b"CPU!" + bytes(mp.CPU_PAGE_SIZE - 4)
        stream2 = _pack_zlib_stream(non_mesh_cpu64) + _pack_zlib_stream(gpu64)
        ext_sys2, nxt2 = mp.consume_trailing_cpu_page(stream2, sys64, 0)
        self.assertEqual(len(ext_sys2), 2 * mp.CPU_PAGE_SIZE)
        # Next page IS a full GPU mesh page, so second call returns None.
        self.assertIsNone(mp.consume_trailing_cpu_page(stream2, ext_sys2, nxt2))

        # 5. Uncompressed trailing CPU tail (< 4096, 16-byte aligned) before zlib GPU page IS consumed.
        uncomp_tail = b"\xAA\xBB" + bytes(46)  # 48 bytes, not a zlib header
        stream3 = uncomp_tail + _pack_zlib_stream(gpu64)
        self.assertEqual(mp.uncompressed_cpu_tail(stream3, 0), 48)
        ext_sys3, nxt3 = mp.consume_trailing_cpu_page(stream3, sys64, 0)
        self.assertEqual(bytes(ext_sys3[-48:]), uncomp_tail)
        self.assertEqual(nxt3, 48)
        gpu_chunk3, _ = mp.read_zlib_stream(stream3, nxt3)
        self.assertEqual(gpu_chunk3, gpu64)


if __name__ == "__main__":
    unittest.main()
