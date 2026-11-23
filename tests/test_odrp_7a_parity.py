"""Parity tests for the table-walking fast paths in tools/odrp_7a.py.

    python -I tests/test_odrp_7a_parity.py

`tools/odrp_7a.py` reads the standalone struct/member/block tables, the `PSIN` section counts and the
`PSIN` block table with `struct.Struct.iter_unpack` over buffers instead of one `unpack_from` call per
field. This file keeps the one-field-at-a-time code that the fast paths replaced (REFERENCE below, copied
as it was) and checks that both give the same answer, and fail with the same message, on hand-built and
seeded-random buffers. Nothing here is pinned to a byte value, a hash or a platform: every expectation
comes from running the reference on the same input, and every input is built here from `random.Random`
with a fixed seed, which gives the same stream on every platform and Python 3 release this repo supports.
No game data is used.
"""
import gc
import os
import random
import struct
import sys
import tempfile
import tracemalloc
import unittest
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import odrp_7a  # noqa: E402
import test_odrp_7a as builders  # noqa: E402


# --- REFERENCE: the per-field implementations, as they were before the iter_unpack rewrite -------------


def ref_u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if 0 <= at and at + 4 <= len(data) else None


def ref_u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if 0 <= at and at + 8 <= len(data) else None


def ref_virtual_offset(pointer):
    if pointer is None:
        return None
    if (pointer >> 32) == 0 and odrp_7a.VIRTUAL_BASE <= pointer < odrp_7a.VIRTUAL_BASE + odrp_7a.SEGMENT_SPAN:
        return pointer - odrp_7a.VIRTUAL_BASE
    return None


def ref_physical_offset(pointer):
    if pointer is None:
        return None
    if (pointer >> 32) == 0 and odrp_7a.PHYSICAL_BASE <= pointer < odrp_7a.PHYSICAL_BASE + odrp_7a.SEGMENT_SPAN:
        return pointer - odrp_7a.PHYSICAL_BASE
    return None


def ref_optional_virtual_offset(pointer):
    if pointer == 0:
        return 0
    return ref_virtual_offset(pointer)


def ref_parse_standalone_header(data):
    if len(data) < odrp_7a.STANDALONE_HEADER_SIZE:
        return None
    if data[0x00:0x08] != b"\x00" * 8:
        return None
    page_map = ref_virtual_offset(ref_u64(data, 0x08))
    if page_map is None:
        return None
    if data[0x10:0x14] != odrp_7a.SIGNATURE or ref_u32(data, 0x14) != odrp_7a.VERSION:
        return None
    if data[0x18:0x20] != b"\x00" * 8:
        return None
    root_objects = ref_u32(data, 0x20)
    if ref_u32(data, 0x24) != 0:
        return None
    q28 = ref_virtual_offset(ref_u64(data, 0x28))
    q30 = ref_optional_virtual_offset(ref_u64(data, 0x30))
    q38 = ref_virtual_offset(ref_u64(data, 0x38))
    q40 = ref_optional_virtual_offset(ref_u64(data, 0x40))
    q50 = ref_optional_virtual_offset(ref_u64(data, 0x50))
    if None in (q28, q30, q38, q40, q50):
        return None
    if data[0x58:0x60] != b"\x00" * 8:
        return None
    structs, enums, blocks, unknown_4e = struct.unpack_from("<4H", data, 0x48)
    if not (1 <= structs <= 4096 and enums <= 4096 and 1 <= blocks <= 4096):
        return None
    if enums > 0 and ref_u64(data, 0x30) == 0:
        return None
    return {
        "page_map": page_map,
        "version": odrp_7a.VERSION,
        "root_objects": root_objects,
        "structs": structs,
        "enums": enums,
        "blocks": blocks,
        "unknown_4e": unknown_4e,
        "tables": {
            "q28": q28,
            "q30": None if ref_u64(data, 0x30) == 0 else q30,
            "q38": q38,
            "q40": None if ref_u64(data, 0x40) == 0 else q40,
            "q50": None if ref_u64(data, 0x50) == 0 else q50,
        },
    }


def ref_parse_standalone(data):
    hdr = ref_parse_standalone_header(data)
    if hdr is None:
        return None

    size = len(data)
    nst = hdr["structs"]
    nen = hdr["enums"]
    nbl = hdr["blocks"]
    q28 = hdr["tables"]["q28"]
    q30 = hdr["tables"]["q30"]
    q38 = hdr["tables"]["q38"]

    base_row = {
        "placement": "standalone",
        "framing": "resource",
        "version": odrp_7a.VERSION,
        "complete": False,
        "needs_continuation": False,
        "resource_kind": "meta",
        "structs": nst,
        "enums": nen,
        "blocks": nbl,
        "root_objects": hdr["root_objects"],
        "root_block": None,
        "unknown_4e": hdr["unknown_4e"],
        "virtual_blocks": 0,
        "physical_blocks": 0,
        "entities": 0,
        "schema_checked": False,
        "error": "",
    }

    if q28 + 32 * nst > size:
        base_row["needs_continuation"] = True
        base_row["error"] = "struct table extends past stream end"
        return base_row
    if nen > 0 and q30 is not None and q30 + 24 * nen > size:
        base_row["needs_continuation"] = True
        base_row["error"] = "enum table extends past stream end"
        return base_row
    if q38 + 16 * nbl > size:
        base_row["needs_continuation"] = True
        base_row["error"] = "instance table extends past stream end"
        return base_row

    struct_sizes = {}
    for i in range(nst):
        off = q28 + 32 * i
        shash, _sig, _flags, _zero0c, mptr_raw, ssize, _zero1c, mcount = struct.unpack_from("<IIIIQIHH", data, off)
        if ssize > (1 << 20) or mcount > 256:
            base_row["error"] = f"struct {i} exceeds size or member cap"
            return base_row
        if mcount > 0:
            moff = ref_virtual_offset(mptr_raw)
            if moff is None:
                base_row["error"] = f"struct {i} has non-virtual member pointer"
                return base_row
            if moff + 16 * mcount > size:
                base_row["needs_continuation"] = True
                base_row["error"] = f"struct {i} member table extends past stream end"
                return base_row
            for j in range(mcount):
                mhash, field_off, mtype, _sub, elem_idx, _ref = struct.unpack_from(
                    "<IIBBHI", data, moff + 16 * j
                )
                if mhash != odrp_7a.ARRAY_ELEMENT_HASH and field_off >= ssize:
                    base_row["error"] = f"struct {i} member {j} offset {field_off} >= size {ssize}"
                    return base_row
                if mtype == odrp_7a.ARRAY_MEMBER_TYPE:
                    if elem_idx >= mcount:
                        base_row["error"] = f"struct {i} array member {j} index {elem_idx} out of range"
                        return base_row
                    desc_hash = ref_u32(data, moff + 16 * elem_idx)
                    if desc_hash != odrp_7a.ARRAY_ELEMENT_HASH:
                        base_row["error"] = f"struct {i} array member {j} target is not 0x100"
                        return base_row
        struct_sizes.setdefault(shash, ssize)

    virtual_blocks = 0
    physical_blocks = 0
    resolved_struct_block = False
    needs_more_blocks = False
    block_list = []

    for i in range(nbl):
        off = q38 + 16 * i
        bhash, bsize, bptr = struct.unpack_from("<IIQ", data, off)
        if bsize > (1 << 24):
            base_row["error"] = f"block {i} exceeds 16 MiB cap"
            return base_row
        voff = ref_virtual_offset(bptr)
        if voff is not None:
            virtual_blocks += 1
            if voff + bsize > size:
                needs_more_blocks = True
            else:
                if bhash in struct_sizes:
                    resolved_struct_block = True
            block_list.append((bhash, bsize, voff))
            continue
        poff = ref_physical_offset(bptr)
        if poff is not None:
            physical_blocks += 1
            block_list.append((bhash, bsize, None))
            continue
        base_row["error"] = f"block {i} has invalid segment pointer"
        return base_row

    base_row["virtual_blocks"] = virtual_blocks
    base_row["physical_blocks"] = physical_blocks

    if needs_more_blocks:
        base_row["needs_continuation"] = True
        base_row["error"] = "virtual block payload extends past stream end"
        return base_row

    if not resolved_struct_block:
        base_row["error"] = "no virtual block matches a declared struct"
        return base_row

    entity_size = struct_sizes.get(odrp_7a.ENTITY_STRUCT, 0)
    map_size = struct_sizes.get(odrp_7a.MAPDATA_STRUCT, 0)
    if entity_size > 0 and map_size > 0:
        base_row["resource_kind"] = "map"
        base_row["entities"] = sum(
            bsize // entity_size
            for bhash, bsize, voff in block_list
            if bhash == odrp_7a.ENTITY_STRUCT
            and voff is not None
            and voff + bsize <= size
            and bsize % entity_size == 0
        )

    base_row["complete"] = True
    base_row["schema_checked"] = True
    return base_row


def ref_header_count(payload, header_size, elem_size, count_at, count_fmt):
    if not payload:
        return 0
    elements = 0
    for k in range(1, len(payload) // header_size + 1):
        elements += struct.unpack_from(count_fmt, payload, header_size * (k - 1) + count_at)[0]
        if header_size * k + elem_size * elements == len(payload):
            return k
    return None


def ref_psin_block_count(blob, first_section):
    if first_section < 16 or first_section > len(blob):
        return None
    at = 8
    count = 0
    sizes = []
    while at + 8 <= first_section:
        kind = ref_u32(blob, at)
        if kind == odrp_7a.BLOCK_TABLE_END:
            break
        if at + 16 > first_section or blob[at + 8:at + 16] != odrp_7a.BLOCK_ROW_FILL_8:
            return None
        sizes.append(ref_u32(blob, at + 4))
        count += 1
        at += 16
    else:
        return None
    end = (at + 15) & ~15
    if end > first_section or blob[at:end] != b"\x70" * (end - at):
        return None
    pos = end
    for bsize in sizes:
        if pos + bsize > first_section:
            return None
        pos += (bsize + 15) & ~15
    if pos != first_section:
        return None
    return count


# --- inputs ---------------------------------------------------------------------------------------------


def synthetic_standalone(structs, members, blocks, *, entity_rows=0, member_hash=0xA000):
    """A complete standalone resource with `structs` structs of `members` members and `blocks` blocks.

    When `entity_rows` is set, struct 0 is the map root and struct 1 the entity struct, and block 1 holds
    `entity_rows` entity rows, so the resource is classed as a map.
    """
    align = builders._align16
    hashes = [0x1000 + i for i in range(structs)]
    sizes = [256] * structs
    if entity_rows:
        hashes[0], hashes[1] = odrp_7a.MAPDATA_STRUCT, odrp_7a.ENTITY_STRUCT
        sizes[1] = 48
    pos = odrp_7a.STANDALONE_HEADER_SIZE
    block_rows = []
    for i in range(blocks):
        pos = align(pos)
        if entity_rows and i == 1:
            block_rows.append((odrp_7a.ENTITY_STRUCT, 48 * entity_rows, pos))
            pos += 48 * entity_rows
        else:
            block_rows.append((hashes[i % structs], 32, pos))
            pos += 32
    pos = align(pos)
    q38 = pos
    pos += 16 * blocks
    member_at = []
    for _ in range(structs):
        pos = align(pos)
        member_at.append(pos)
        pos += 16 * members
    pos = align(pos)
    q28 = pos
    pos += 32 * structs
    buf = bytearray(align(pos))
    struct.pack_into("<Q", buf, 0x08, odrp_7a.VIRTUAL_BASE + 0x10)
    buf[0x10:0x14] = odrp_7a.SIGNATURE
    struct.pack_into("<I", buf, 0x14, odrp_7a.VERSION)
    struct.pack_into("<II", buf, 0x20, 4, 0)
    struct.pack_into("<Q", buf, 0x28, odrp_7a.VIRTUAL_BASE + q28)
    struct.pack_into("<Q", buf, 0x38, odrp_7a.VIRTUAL_BASE + q38)
    struct.pack_into("<4H", buf, 0x48, structs, 0, blocks, 0)
    for i in range(structs):
        ptr = odrp_7a.VIRTUAL_BASE + member_at[i] if members else 0
        struct.pack_into("<IIIIQIHH", buf, q28 + 32 * i, hashes[i], 1, 0, 0, ptr, sizes[i], 0, members)
        for j in range(members):
            if j == 1 and members > 2:
                row = (odrp_7a.ARRAY_ELEMENT_HASH, 300, 4, 0, 0, 0)
            elif j == 2 and members > 2:
                row = (member_hash + j, 16, odrp_7a.ARRAY_MEMBER_TYPE, 0, 1, 0)
            else:
                row = (member_hash + j, (4 * j) % 40, 4, 0, 0, 0)
            struct.pack_into("<IIBBHI", buf, member_at[i] + 16 * j, *row)
    for i, (bhash, bsize, off) in enumerate(block_rows):
        struct.pack_into("<IIQ", buf, q38 + 16 * i, bhash, bsize, odrp_7a.VIRTUAL_BASE + off)
    return bytes(buf)


def mutate(rng, blob):
    """A few seeded byte edits: flips, interesting dwords, cuts, insertions, truncation."""
    out = bytearray(blob)
    for _ in range(rng.choice((1, 1, 2, 3, 8))):
        if not out:
            break
        kind = rng.random()
        at = rng.randrange(len(out))
        if kind < 0.45:
            out[at] = rng.randrange(256)
        elif kind < 0.6:
            out[at] ^= 1 << rng.randrange(8)
        elif kind < 0.8:
            at = max(0, min(at, len(out) - 4))
            word = rng.choice((0, 1, 0x100, 0x50000000, 0x50000010, 0x5FFFFFFF, 0x60000000, 0xFFFFFFFF))
            out[at:at + 4] = struct.pack("<I", word)[:len(out) - at]
        elif kind < 0.9:
            del out[at:at + rng.randrange(1, 40)]
        else:
            out[at:at] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 20)))
    if rng.random() < 0.1:
        out = out[:rng.randrange(len(out) + 1)]
    return bytes(out)


def seed_blobs():
    return [
        synthetic_standalone(4, 5, 4),
        synthetic_standalone(30, 3, 30),
        synthetic_standalone(1, 1, 1),
        synthetic_standalone(6, 40, 6),
        synthetic_standalone(5, 0, 3),
        synthetic_standalone(3, 6, 3, entity_rows=7),
        builders.build_standalone_resource(),
        builders.build_standalone_resource(as_map=True, entity_count=5),
        builders.build_standalone_resource(as_map=True, enums=0, include_physical_block=True),
        builders.build_psin_container(),
        builders.build_psin_container(structs=9, enums=5, block_sizes=(16, 32, 48, 64)),
        builders.build_psin_container(include_lrsc=False),
        builders.build_lrsc_payload(),
        b"LRSC" + (88).to_bytes(4, "big") + builders.build_lrsc_payload(),
    ]


def mutants(count, seed):
    rng = random.Random(seed)
    seeds = seed_blobs()
    for _ in range(count):
        yield mutate(rng, rng.choice(seeds))


class ReferenceParity(unittest.TestCase):
    def test_seed_blobs_agree_unmutated(self):
        for n, blob in enumerate(seed_blobs()):
            with self.subTest(blob=n):
                self.assertEqual(odrp_7a.parse_standalone_header(blob), ref_parse_standalone_header(blob))
                self.assertEqual(odrp_7a.parse_standalone(blob), ref_parse_standalone(blob))

    def test_standalone_header_and_tables_on_mutated_buffers(self):
        errors = set()
        for n, blob in enumerate(mutants(6000, seed=20260101)):
            want_header = ref_parse_standalone_header(blob)
            self.assertEqual(odrp_7a.parse_standalone_header(blob), want_header, f"header, mutant {n}")
            want = ref_parse_standalone(blob)
            self.assertEqual(odrp_7a.parse_standalone(blob), want, f"tables, mutant {n}")
            if want is not None:
                errors.add(want["error"][:12])
        # The walk must have reached more than one verdict, or this would only be checking one branch.
        self.assertGreater(len(errors), 4)

    def test_standalone_accepts_other_buffer_types(self):
        blob = synthetic_standalone(6, 8, 6, entity_rows=3)
        want = ref_parse_standalone(blob)
        self.assertEqual(want["resource_kind"], "map")
        self.assertEqual(want["entities"], 3)
        for wrap in (bytearray, memoryview):
            with self.subTest(wrap=wrap.__name__):
                self.assertEqual(odrp_7a.parse_standalone(wrap(blob)), want)
                self.assertEqual(odrp_7a.parse_standalone_header(wrap(blob)), ref_parse_standalone_header(blob))

    def test_header_count_matches_per_field_loop(self):
        rng = random.Random(7)
        for _ in range(4000):
            payload = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 160)))
            for header_size, elem_size, count_at, count_fmt in ((32, 16, 30, "<H"), (24, 16, 16, "<I")):
                self.assertEqual(
                    odrp_7a._header_count(payload, header_size, elem_size, count_at, count_fmt),
                    ref_header_count(payload, header_size, elem_size, count_at, count_fmt),
                )

    def test_header_count_finds_exact_fit_and_rejects_overshoot(self):
        # Three LSCH-shaped headers declaring 1, 2 and 0 members, then 3 members of 16 bytes.
        headers = b"".join(bytes(30) + struct.pack("<H", n) for n in (1, 2, 0))
        payload = headers + bytes(16 * 3)
        self.assertEqual(odrp_7a._header_count(payload, 32, 16, 30, "<H"), ref_header_count(payload, 32, 16, 30, "<H"))
        self.assertEqual(odrp_7a._header_count(payload, 32, 16, 30, "<H"), 3)
        short = payload[:-1]
        self.assertIsNone(odrp_7a._header_count(short, 32, 16, 30, "<H"))
        self.assertEqual(odrp_7a._header_count(b"", 32, 16, 30, "<H"), 0)
        # A layout the module does not pre-build still works.
        wide = struct.pack("<2xH4x", 1) + bytes(16)
        self.assertEqual(odrp_7a._header_count(wide, 8, 16, 2, "<H"), ref_header_count(wide, 8, 16, 2, "<H"))

    def test_block_table_count_matches_per_row_loop(self):
        rng = random.Random(11)
        checked = 0
        for _ in range(3000):
            rows = b"".join(
                struct.pack("<II", rng.choice((261, 0x70707070, rng.randrange(1 << 32))), rng.randrange(64))
                + rng.choice((odrp_7a.BLOCK_ROW_FILL_8, bytes(8)))
                for _ in range(rng.randrange(0, 6))
            )
            tail = bytes(rng.choice((0, 4, 8, 16, 0x70)) for _ in range(rng.randrange(0, 24)))
            blob = b"PSIN\x00\x00\x00\x00" + rows + tail
            for first in range(0, len(blob) + 3):
                checked += 1
                self.assertEqual(
                    odrp_7a._psin_block_count(blob, first), ref_psin_block_count(blob, first), (blob, first)
                )
        self.assertGreater(checked, 10000)

    def test_block_table_count_on_real_containers(self):
        for sizes in ((32, 48, 16), (16,), (16,) * 40, (1, 15, 16, 17, 31, 33)):
            blob = builders.build_psin_container(block_sizes=sizes, root_block=1)
            first = int.from_bytes(blob[4:8], "big")
            self.assertEqual(odrp_7a._psin_block_count(blob, first), ref_psin_block_count(blob, first))
            self.assertEqual(odrp_7a._psin_block_count(blob, first), len(sizes))

    def test_psin_and_classify_results_match_with_reference_helpers(self):
        # Run the whole container/classifier with the reference helpers swapped in, and compare.
        for n, blob in enumerate(mutants(3000, seed=424242)):
            got = (odrp_7a._parse_psin_container(blob), odrp_7a.parse_psin_lrsc(blob), odrp_7a.classify_blob(blob))
            with unittest.mock.patch.multiple(
                odrp_7a,
                _header_count=ref_header_count,
                _psin_block_count=ref_psin_block_count,
                parse_standalone_header=ref_parse_standalone_header,
                parse_standalone=ref_parse_standalone,
            ):
                want = (odrp_7a._parse_psin_container(blob), odrp_7a.parse_psin_lrsc(blob), odrp_7a.classify_blob(blob))
            self.assertEqual(got, want, f"mutant {n}")

    def test_directory_scan_rows_match_with_reference_parsers(self):
        rng = random.Random(99)
        with tempfile.TemporaryDirectory() as folder:
            seeds = seed_blobs()
            for i in range(120):
                blob = rng.choice(seeds) if i % 2 else mutate(rng, rng.choice(seeds))
                with open(os.path.join(folder, f"{i * 0x1000:08x}.bin"), "wb") as fh:
                    fh.write(blob)
            rows, summary = odrp_7a.scan_dir(folder)
            with unittest.mock.patch.multiple(
                odrp_7a,
                _header_count=ref_header_count,
                _psin_block_count=ref_psin_block_count,
                parse_standalone_header=ref_parse_standalone_header,
                parse_standalone=ref_parse_standalone,
            ):
                want_rows, want_summary = odrp_7a.scan_dir(folder)
        self.assertGreater(len(rows), 20)
        self.assertEqual(rows, want_rows)
        for key in ("records", "complete", "incomplete", "standalone_map", "standalone_entities", "framings"):
            self.assertEqual(summary[key], want_summary[key], key)


class PeakMemory(unittest.TestCase):
    """The fast paths must not raise the peak allocation, for a dense table and for near-empty ones."""

    SLACK = 512  # a few memoryview and iterator objects; the same for any table size

    @staticmethod
    def peak(func, *args):
        gc.collect()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            func(*args)
            return tracemalloc.get_traced_memory()[1] - before
        finally:
            tracemalloc.stop()

    def test_dense_standalone_does_not_raise_peak(self):
        dense = synthetic_standalone(2000, 24, 2000, entity_rows=0)
        self.assertEqual(odrp_7a.parse_standalone(dense), ref_parse_standalone(dense))
        got = self.peak(odrp_7a.parse_standalone, dense)
        want = self.peak(ref_parse_standalone, dense)
        self.assertLessEqual(got, want, (got, want))

    def test_dense_psin_tables_do_not_raise_peak(self):
        psin = builders.build_psin_container(structs=1500, enums=1500, block_sizes=(16,) * 1500)
        first = int.from_bytes(psin[4:8], "big")
        self.assertEqual(odrp_7a._psin_block_count(psin, first), ref_psin_block_count(psin, first))
        got = self.peak(odrp_7a._psin_block_count, psin, first)
        want = self.peak(ref_psin_block_count, psin, first)
        self.assertLessEqual(got, want + self.SLACK, (got, want))
        lsch = b"".join(struct.pack("<IIIIQIHH", 0x2000 + i, 1, 0, 0, 0, 16, 0, 1) for i in range(1500))
        lsch += bytes(16 * 1500)
        self.assertEqual(odrp_7a._header_count(lsch, 32, 16, 30, "<H"), 1500)
        got = self.peak(odrp_7a._header_count, lsch, 32, 16, 30, "<H")
        want = self.peak(ref_header_count, lsch, 32, 16, 30, "<H")
        self.assertLessEqual(got, want + self.SLACK, (got, want))

    def test_near_empty_tables_stay_within_a_few_hundred_bytes(self):
        for blob in (
            synthetic_standalone(1, 0, 1),
            synthetic_standalone(1, 1, 1),
            synthetic_standalone(4096, 0, 1),
            builders.build_standalone_resource(enums=0),
        ):
            got = self.peak(odrp_7a.parse_standalone, blob)
            want = self.peak(ref_parse_standalone, blob)
            self.assertLessEqual(got, want + self.SLACK, (got, want))
        psin = builders.build_psin_container(structs=1, enums=0, block_sizes=(16,))
        first = int.from_bytes(psin[4:8], "big")
        got = self.peak(odrp_7a._psin_block_count, psin, first)
        want = self.peak(ref_psin_block_count, psin, first)
        self.assertLessEqual(got, want + self.SLACK, (got, want))

    def test_no_table_cache_grows_with_input(self):
        before = dict(odrp_7a._COUNT_ROWS)
        for blob in mutants(500, seed=5):
            odrp_7a.classify_blob(blob)
        odrp_7a._header_count(bytes(48), 8, 16, 2, "<H")  # a layout that is not pre-built is not stored
        self.assertEqual(odrp_7a._COUNT_ROWS, before)
        self.assertLessEqual(len(odrp_7a._COUNT_ROWS), 4096)


if __name__ == "__main__":
    unittest.main()
