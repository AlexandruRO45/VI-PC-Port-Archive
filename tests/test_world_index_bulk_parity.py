"""Parity tests for the bulk entity reader in tools/world_index.py (synthetic 0x7a resources, no game data).

``parse_resource`` reads each block's entities a chunk of records at a time. These tests compare it with an
inline copy of the per-entity reader it replaced, on seeded random resources, so no platform-specific bytes or
hashes are pinned:

    python -I tests/test_world_index_bulk_parity.py
"""
import importlib.util
import io
import json
import math
import os
import random
import struct
import tempfile
import tracemalloc
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "world_index.py")
    spec = importlib.util.spec_from_file_location("world_index_bulk_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WI = load_tool()
PTR = 0x50000000
FIELD = WI.FIELD


# ---- inline reference: the per-entity reader that parse_resource used before the bulk reader ----------------

def ref_members(b, table, count, size):
    if count > 256 or (count and table is None) or (table is not None and table + 16 * count > size):
        return None
    out = {}
    if not count:
        return out
    for k in range(count):
        o = table + 16 * k
        out[WI._u32(b, o)] = (WI._u32(b, o + 4), b[o + 8], b[o + 9], WI._u16(b, o + 10), WI._u32(b, o + 12))
    return out


def ref_parse_resource(b, collect=None):
    if len(b) < 0x50 or b[0x10:0x14] != WI.SIGNATURE or b[0x14] != WI.VERSION:
        return None
    nst, nen, nbl = WI._u16(b, 0x48), WI._u16(b, 0x4A), WI._u16(b, 0x4C)
    if not (1 <= nst <= 4096 and nbl <= 4096 and nen <= 4096) or nbl == 0:
        return None
    st, it = WI._ptr(WI._u64(b, 0x28), len(b)), WI._ptr(WI._u64(b, 0x38), len(b))
    if st is None or it is None or st + 32 * nst > len(b) or it + 16 * nbl > len(b):
        return None
    structs = {}
    for i in range(nst):
        o = st + 32 * i
        name, size, count = WI._u32(b, o), WI._u32(b, o + 0x18), WI._u16(b, o + 0x1E)
        table = WI._ptr(WI._u64(b, o + 0x10), len(b))
        if size > (1 << 20) or count > 256:
            return None
        members = ref_members(b, table, count, len(b))
        if members is None:
            return None
        structs.setdefault(name, (size, members))
    for struct_hash, types in ((WI.ENTITY, WI.ENTITY_MEMBER_TYPES), (WI.MAPDATA, WI.MAPDATA_MEMBER_TYPES)):
        if struct_hash in structs:
            size, members = structs[struct_hash]
            for k, allowed in types.items():
                if k in members:
                    m = members[k]
                    w = WI.MEMBER_WIDTH.get(m[1], 0)
                    if m[1] not in allowed or (w and m[0] + w > size):
                        return None
    blocks = []
    for i in range(nbl):
        o = it + 16 * i
        name, size, p = WI._u32(b, o), WI._u32(b, o + 4), WI._u64(b, o + 8)
        if size > (1 << 24):
            return None
        blocks.append((name, size, WI._ptr(p, len(b))))
    if not any(name in structs and off is not None for name, _, off in blocks):
        return None

    entity_size = structs.get(WI.ENTITY, (0, {}))[0]
    map_size = structs.get(WI.MAPDATA, (0, {}))[0]
    row = {"signature": "0DRPz", "version": WI.VERSION, "kind": "meta", "structs": nst, "enums": nen,
           "blocks": nbl, "root_objects": WI._u32(b, 0x20)}
    if not (entity_size and map_size):
        return row
    em = structs[WI.ENTITY][1]
    mm = structs[WI.MAPDATA][1]
    map_name, ordinals, parents = None, {}, ()
    if getattr(collect, "placement_ids", False):
        map_name, ordinals, parents = WI._placement_tables(b, blocks, structs[WI.MAPDATA], structs[WI.ENTITY])
    lod = {}
    archetypes = set()
    positions = []
    lod_dists = []
    finite = math.isfinite
    value = WI._value
    for block_number, (name, size, off) in enumerate(blocks, 1):
        if name != WI.ENTITY or entity_size == 0 or size % entity_size:
            continue
        if off is None:
            if collect is not None:
                for _ in range(size // entity_size):
                    collect.append({"complete": False, "archetype": None})
            continue
        for k in range(size // entity_size):
            o = off + k * entity_size
            a = value(b, o, em.get(FIELD["archetypeName"]))
            if a is not None and isinstance(a, int):
                archetypes.add(a)
            pos = value(b, o, em.get(FIELD["position"]))
            if isinstance(pos, list) and len(pos) == 3:
                positions.append(pos)
            level = value(b, o, em.get(FIELD["lodLevel"]))
            if isinstance(level, int) and level in WI.LOD_NAMES:
                lod[WI.LOD_NAMES[level]] = lod.get(WI.LOD_NAMES[level], 0) + 1
            dist = value(b, o, em.get(FIELD["lodDist"]))
            if isinstance(dist, (int, float)) and finite(dist) and dist >= 0:
                lod_dists.append(dist)
            if collect is not None:
                rot = value(b, o, em.get(FIELD["rotation"]))
                sxy = value(b, o, em.get(FIELD["scaleXY"]))
                sz = value(b, o, em.get(FIELD["scaleZ"]))
                cld = value(b, o, em.get(FIELD["childLodDist"]))
                entity = {
                    "complete": o + entity_size <= len(b),
                    "archetype": a,
                    "archetype_key": value(b, o, em.get(WI.ARCHETYPE_KEY)),
                    "position": [v if (isinstance(v, (int, float)) and finite(v)) else None for v in pos] if pos else None,
                    "rotation": [v if (isinstance(v, (int, float)) and finite(v)) else None for v in rot] if rot else None,
                    "scale_xy": sxy if (isinstance(sxy, (int, float)) and finite(sxy)) else None,
                    "scale_z": sz if (isinstance(sz, (int, float)) and finite(sz)) else None,
                    "flags": value(b, o, em.get(FIELD["flags"])),
                    "guid": value(b, o, em.get(FIELD["guid"])),
                    "lod_level": level,
                    "parent_index": value(b, o, em.get(FIELD["parentIndex"])),
                    "parent_instance": value(b, o, em.get(FIELD["parentInstance"])),
                    "num_children": value(b, o, em.get(FIELD["numChildren"])),
                    "lod_dist": dist if (isinstance(dist, (int, float)) and finite(dist)) else None,
                    "child_lod_dist": cld if (isinstance(cld, (int, float)) and finite(cld)) else None,
                }
                ordinal = ordinals.get(((k * entity_size) << 20) | block_number) if ordinals else None
                if ordinal is not None and entity["complete"]:
                    entity.update(map_name=map_name, entity_index=ordinal)
                    if parents and entity["parent_instance"] == -1 and entity["parent_index"] >= 0:
                        slot = value(b, o, em[WI.PARENT_MAP_SLOT])
                        if slot is not None and 0 <= slot < len(parents):
                            entity["parent_map_name"] = f"{parents[slot]:08x}"
                collect.append(entity)
    row.update({
        "kind": "map",
        "entity_size": entity_size,
        "entities": sum(size // entity_size for n, size, off in blocks
                        if n == WI.ENTITY and off is not None and size % entity_size == 0),
        "archetypes": len(archetypes),
        "lod": lod,
        "lod_dist_min": min(lod_dists) if lod_dists else None,
        "lod_dist_max": max(lod_dists) if lod_dists else None,
    })
    if positions:
        pos_min = []
        pos_max = []
        for i in range(3):
            axis_vals = [p[i] for p in positions if isinstance(p[i], (int, float)) and finite(p[i])]
            pos_min.append(round(min(axis_vals), 4) if axis_vals else None)
            pos_max.append(round(max(axis_vals), 4) if axis_vals else None)
        if any(v is not None for v in pos_min):
            row["position_min"] = pos_min
            row["position_max"] = pos_max
    for name, size, off in blocks:
        if name != WI.MAPDATA or off is None or map_size == 0:
            continue
        row["map_name"] = WI._u32(b, off + mm.get(FIELD["name"], (8,))[0])
        row["parent"] = WI._u32(b, off + mm.get(FIELD["parent"], (0x68,))[0])
        if FIELD["name64"] in mm:
            row["map_name64"] = WI._u64(b, off + mm[FIELD["name64"]][0])
        for key, field in (("extent_min", "streamingExtentsMin"), ("extent_max", "streamingExtentsMax"),
                           ("entities_min", "entitiesExtentsMin"), ("entities_max", "entitiesExtentsMax")):
            v = value(b, off, mm.get(FIELD[field]))
            if v:
                r = WI._r4(v)
                if r is not None:
                    row[key] = r
        row["map_flags"] = value(b, off, mm.get(FIELD["flags_map"]))
        row["content_flags"] = value(b, off, mm.get(FIELD["contentFlags"]))
        break
    return row


# ---- synthetic resources -------------------------------------------------------------------------------------

INTS = (0x10, 0x11, 0x12, 0x13, 0x14, 0x15)
# (field hash, struct member types it may have; the byte width of each type is WI.MEMBER_WIDTH)
ENTITY_FIELDS = (
    (FIELD["archetypeName"], INTS + (0x4A,)), (WI.ARCHETYPE_KEY, (0x08, 0x4F)), (FIELD["position"], (0x32, 0x33)),
    (FIELD["rotation"], (0x34,)), (FIELD["scaleXY"], (0x21,)), (FIELD["scaleZ"], (0x21,)),
    (FIELD["flags"], INTS + (0x4A,)), (FIELD["guid"], (0x08, 0x4F)), (FIELD["lodLevel"], INTS + (0x60,)),
    (FIELD["parentIndex"], INTS), (FIELD["parentInstance"], INTS), (FIELD["numChildren"], INTS),
    (FIELD["lodDist"], (0x21,)), (FIELD["childLodDist"], (0x21,)), (WI.PARENT_MAP_SLOT, INTS),
)
SPECIAL_FLOATS = (0.0, -0.0, 1.5, -2.25, 250.0, 1e30, -1e30, math.inf, -math.inf, math.nan)
FILL = 0x800


def member(hash_, off, ty):
    return struct.pack("<IIBBHI", hash_, off, ty, 0, 0, 0)


def struct_def(hash_, size, table, count):
    return struct.pack("<IIIIQIHH", hash_, 0, 0, 0, PTR | table, size, 0, count)


def build_resource(rng, *, placement=False, truncate=0):
    """A random valid resource with entity blocks of different lengths, cut `truncate` bytes short."""
    esize = rng.choice((0x20, 0x34, 0x58, 0x90, 0xB4))
    members = []
    for hash_, types in ENTITY_FIELDS:
        if placement and hash_ in (FIELD["parentIndex"], FIELD["parentInstance"], WI.PARENT_MAP_SLOT):
            members.append((hash_, types[:1] if hash_ == WI.PARENT_MAP_SLOT else (0x12,)))
        elif rng.random() < 0.85:
            members.append((hash_, types))
    entity_members = []
    for hash_, types in members:
        ty = rng.choice(types)
        entity_members.append((hash_, rng.randrange(0, esize - WI.MEMBER_WIDTH[ty] + 1), ty))
    counts = [rng.choice((0, 1, 3, 17, 40, 130)) for _ in range(rng.randint(1, 3))]
    map_members = [(FIELD["name"], 0x08, 0x4A), (FIELD["name64"], 0x10, 0x4F), (FIELD["parent"], 0x68, 0x4A),
                   (FIELD["streamingExtentsMin"], 0x70, 0x32), (FIELD["streamingExtentsMax"], 0x7C, 0x32),
                   (FIELD["flags_map"], 0x88, 0x15), (FIELD["contentFlags"], 0x8C, 0x15)]
    if placement:
        map_members += [(WI.MAP_ENTITIES, 0x20, 0x52), (WI.PARENT_MAPS, 0x30, 0x52)]

    refs = []
    if placement:
        # Entity references by record start, as the typed array stores them; a repeated one has no ordinal.
        for bi, n in enumerate(counts):
            for k in range(n):
                if rng.random() < 0.7:
                    refs.append(((k * esize) << 20) | (bi + 1))
        rng.shuffle(refs)
        if refs and rng.random() < 0.3:
            refs.append(refs[0])
    parents = [rng.getrandbits(32) for _ in range(rng.randint(1, 4))] if placement else []

    # block order: entity blocks, the map root, the entity reference block (7), the parent map block (0x4A)
    payloads = [(WI.ENTITY, bytearray(esize * n)) for n in counts]
    root = bytearray(0x100)
    payloads.append((WI.MAPDATA, root))
    ref_block = par_block = None
    if placement:
        ref_block = bytearray(8 * max(len(refs), 1))
        payloads.append((7, ref_block))
        par_block = bytearray(4 * len(parents))
        payloads.append((0x4A, par_block))

    out = bytearray(FILL)
    out[0x10:0x14] = b"0DRP"
    out[0x14] = WI.VERSION
    struct.pack_into("<I", out, 0x20, len(counts) + 1)
    struct.pack_into("<Q", out, 0x28, PTR | 0x100)
    struct.pack_into("<Q", out, 0x38, PTR | 0x400)
    struct.pack_into("<HHHH", out, 0x48, 2, 0, len(payloads), 0)
    out[0x100:0x120] = struct_def(WI.ENTITY, esize, 0x200, len(entity_members))
    out[0x120:0x140] = struct_def(WI.MAPDATA, 0x100, 0x300, len(map_members))
    for i, m in enumerate(entity_members):
        out[0x200 + 16 * i:0x210 + 16 * i] = member(*m)
    for i, m in enumerate(map_members):
        out[0x300 + 16 * i:0x310 + 16 * i] = member(*m)

    # fill the entity blocks with random records, floats drawn from a pool that includes NaN, inf and -0.0
    for name, payload in payloads[:len(counts)]:
        for k in range(len(payload) // esize):
            rec = bytearray(rng.randbytes(esize))
            for h, off, ty in entity_members:
                width = WI.MEMBER_WIDTH[ty]
                if ty in (0x21, 0x32, 0x33, 0x34):
                    for j in range(width // 4):
                        struct.pack_into("<f", rec, off + 4 * j, rng.choice(SPECIAL_FLOATS) if rng.random() < 0.3
                                         else rng.uniform(-5000, 5000))
                elif h == FIELD["lodLevel"]:
                    rec[off] = rng.randrange(0, 9)
                    for j in range(1, width):
                        rec[off + j] = 0
                elif h == FIELD["archetypeName"]:
                    rec[off:off + width] = rng.randrange(0, 20).to_bytes(width, "little")
                elif h == FIELD["parentInstance"] and placement:
                    struct.pack_into("<h", rec, off, rng.choice((-1, -1, 0, 3)))
                elif h == FIELD["parentIndex"] and placement:
                    struct.pack_into("<h", rec, off, rng.choice((-2, 0, 1, 5)))
                elif h == WI.PARENT_MAP_SLOT and placement:
                    rec[off] = rng.randrange(0, len(parents) + 2)
            payload[k * esize:(k + 1) * esize] = rec
    struct.pack_into("<I", root, 0x08, rng.getrandbits(32))
    struct.pack_into("<Q", root, 0x10, rng.getrandbits(64))
    struct.pack_into("<I", root, 0x68, rng.getrandbits(32))
    struct.pack_into("<3f", root, 0x70, -10.0, -20.0, -30.0)
    struct.pack_into("<3f", root, 0x7C, 10.0, 20.0, 30.0)
    struct.pack_into("<I", root, 0x88, 2)
    struct.pack_into("<I", root, 0x8C, 65)
    if placement:
        struct.pack_into("<QHH", root, 0x20, len(counts) + 2, len(refs), len(refs) + 1)
        struct.pack_into("<QHH", root, 0x30, len(counts) + 3, len(parents), len(parents))
        for i, r in enumerate(refs):
            struct.pack_into("<Q", ref_block, 8 * i, r)
        for i, p in enumerate(parents):
            struct.pack_into("<I", par_block, 4 * i, p)

    at = FILL
    for i, (name, payload) in enumerate(payloads):
        struct.pack_into("<IIQ", out, 0x400 + 16 * i, name, len(payload), PTR | at)
        out.extend(payload)
        at += len(payload)
    if truncate:
        del out[len(out) - truncate:]
    return bytes(out)


def dense_resource(n):
    """One entity block of `n` records with the four members the plain index reads."""
    rng = random.Random(11)
    esize = 0x30
    members = [(FIELD["archetypeName"], 0x00, 0x15), (FIELD["position"], 0x04, 0x32),
               (FIELD["lodLevel"], 0x10, 0x11), (FIELD["lodDist"], 0x14, 0x21)]
    out = bytearray(FILL)
    out[0x10:0x14] = b"0DRP"
    out[0x14] = WI.VERSION
    struct.pack_into("<I", out, 0x20, 2)
    struct.pack_into("<Q", out, 0x28, PTR | 0x100)
    struct.pack_into("<Q", out, 0x38, PTR | 0x400)
    struct.pack_into("<HHHH", out, 0x48, 2, 0, 2, 0)
    out[0x100:0x120] = struct_def(WI.ENTITY, esize, 0x200, len(members))
    out[0x120:0x140] = struct_def(WI.MAPDATA, 0x100, 0x300, 1)
    for i, m in enumerate(members):
        out[0x200 + 16 * i:0x210 + 16 * i] = member(*m)
    out[0x300:0x310] = member(FIELD["name"], 8, 0x4A)
    records = bytearray()
    for _ in range(n):
        rec = bytearray(esize)
        struct.pack_into("<I", rec, 0, rng.getrandbits(32))
        struct.pack_into("<3f", rec, 4, rng.uniform(-4000, 4000), rng.uniform(-4000, 4000), rng.uniform(0, 300))
        rec[0x10] = rng.randrange(0, 7)
        struct.pack_into("<f", rec, 0x14, rng.choice((100.0, 250.0, 500.0)))
        records += rec
    struct.pack_into("<IIQ", out, 0x400, WI.ENTITY, len(records), PTR | FILL)
    struct.pack_into("<IIQ", out, 0x410, WI.MAPDATA, 0x100, PTR | (FILL + len(records)))
    return bytes(out) + bytes(records) + bytes(0x100)


def entity_text(entity):
    return repr(sorted(entity.items()))


def tiny_bulk(rng):
    """Patches for the module's bulk thresholds so small resources use chunks of a few records."""
    return (mock.patch.object(WI, "_BULK_MIN", rng.choice((1, 2, 16, 40))),
            mock.patch.object(WI, "_BULK_CHUNK", rng.choice((1, 3, 16, 1024))))


class Parity(unittest.TestCase):
    def test_plain_rows_match_the_per_entity_reader(self):
        rng = random.Random(0x1D)
        for case in range(150):
            data = build_resource(rng, truncate=rng.choice((0, 0, 0, 5, 40, 300)))
            minimum, chunk = tiny_bulk(rng)
            with minimum, chunk:
                row = WI.parse_resource(data)
            self.assertEqual(json.dumps(row), json.dumps(ref_parse_resource(data)), f"case {case}")

    def test_collected_entities_match_the_per_entity_reader(self):
        rng = random.Random(0xC011)
        for case in range(150):
            data = build_resource(rng, truncate=rng.choice((0, 0, 0, 7, 100, 500)))
            ref_rows, new_rows = [], []
            ref = ref_parse_resource(data, ref_rows)
            minimum, chunk = tiny_bulk(rng)
            with minimum, chunk:
                new = WI.parse_resource(data, new_rows)
            self.assertEqual(json.dumps(ref), json.dumps(new), f"case {case}")
            self.assertEqual([entity_text(e) for e in ref_rows], [entity_text(e) for e in new_rows], f"case {case}")
            self.assertEqual([list(e) for e in ref_rows], [list(e) for e in new_rows])

    def test_aggregates_and_entity_lines_match(self):
        rng = random.Random(0xA66)
        for case in range(120):
            data = build_resource(rng, placement=rng.random() < 0.7, truncate=rng.choice((0, 0, 9, 200)))
            ref_agg, new_agg = WI.ArchetypeCounts(), WI.ArchetypeCounts()
            ref_text, new_text = io.StringIO(), io.StringIO()
            ref_rows, new_rows = WI.EntityRows(ref_text), WI.EntityRows(new_text)
            ref = ref_parse_resource(data, WI._Both(ref_agg, ref_rows))
            minimum, chunk = tiny_bulk(rng)
            with minimum, chunk:
                new = WI.parse_resource(data, WI._Both(new_agg, new_rows))
            self.assertEqual(json.dumps(ref), json.dumps(new), f"case {case}")
            self.assertEqual(ref_text.getvalue(), new_text.getvalue(), f"case {case}")
            self.assertEqual(json.dumps(ref_agg.to_json()), json.dumps(new_agg.to_json()))
            self.assertEqual((ref_rows.written, ref_rows.incomplete), (new_rows.written, new_rows.incomplete))

    def test_placement_ids_and_parent_maps_are_kept(self):
        rng = random.Random(0xB10C)
        seen_ordinal = seen_parent = False
        for _ in range(60):
            data = build_resource(rng, placement=True)
            ref_text, new_text = io.StringIO(), io.StringIO()
            ref_parse_resource(data, WI.EntityRows(ref_text))
            with mock.patch.object(WI, "_BULK_MIN", 1), mock.patch.object(WI, "_BULK_CHUNK", 4):
                WI.parse_resource(data, WI.EntityRows(new_text))
            self.assertEqual(ref_text.getvalue(), new_text.getvalue())
            seen_ordinal = seen_ordinal or "entity_index" in new_text.getvalue()
            seen_parent = seen_parent or "parent_map_name" in new_text.getvalue()
        self.assertTrue(seen_ordinal and seen_parent, "the generator must reach the placement branches")

    def test_signed_zero_and_non_finite_extremes(self):
        # -0.0 and 0.0 compare equal: the first one in entity order stays the reported extreme
        for first, second, want in (((0.0, -0.0), (-0.0, 0.0), "0.0"), ((-0.0, 0.0), (0.0, -0.0), "-0.0")):
            stats = WI._EntityStats()
            stats.add_dists(first)
            stats.add_dists(second)
            self.assertEqual((repr(stats.dist_lo), repr(stats.dist_hi)), (want, want))
        for first, second in (((0.0, 0.0, 0.0), (-0.0, -0.0, -0.0)), ((-0.0, -0.0, -0.0), (0.0, 0.0, 0.0))):
            stats = WI._EntityStats()
            stats.add_positions([first])
            stats.add_positions([second])
            self.assertEqual([repr(v) for v in stats.pos_lo], [repr(v) for v in first])
            self.assertEqual([repr(v) for v in stats.pos_hi], [repr(v) for v in first])
        stats = WI._EntityStats()
        stats.add_positions([(math.nan, 1.0, math.inf), (math.nan, -2.0, 5.0)])
        self.assertEqual((stats.pos_lo, stats.pos_hi), ([None, -2.0, 5.0], [None, 1.0, 5.0]))
        stats.add_dists([math.nan, math.inf, -1.0, 3.0])
        self.assertEqual((stats.dist_lo, stats.dist_hi), (3.0, 3.0))

    def test_layout_agrees_with_parse_resource_on_damaged_headers(self):
        rng = random.Random(99)
        for _ in range(80):
            data = bytearray(build_resource(rng, truncate=rng.choice((0, 3, 60))))
            if rng.random() < 0.5:
                data[rng.randrange(0x10, 0x60)] ^= 1 << rng.randrange(8)
            data = bytes(data)
            self.assertEqual(WI._layout(data) is None, ref_parse_resource(data) is None)
            self.assertEqual(WI.parse_resource(data) is None, ref_parse_resource(data) is None)

    def test_rpf_scan_matches_the_reference_reader(self):
        rng = random.Random(5)
        parts = [build_resource(rng) for _ in range(6)]
        out = bytearray(0x40)
        out[:4] = b"9FPR"
        struct.pack_into("<I", out, 4, 0x40)
        for part in parts:
            out.extend(b"\0" * (-len(out) % 16))
            out.extend(zlib.compress(part))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.rpf")
            with open(path, "wb") as f:
                f.write(out)
            agg = WI.ArchetypeCounts()
            rows, _ = WI.scan_path(path, collect=agg)
        ref_agg = WI.ArchetypeCounts()
        ref_rows = [r for r in (ref_parse_resource(p, ref_agg) for p in parts) if r]
        self.assertEqual(len(rows), len(ref_rows))
        for row, ref in zip(rows, ref_rows):
            row = {k: v for k, v in row.items() if k not in ("offset", "size")}
            self.assertEqual(json.dumps(row), json.dumps(ref))
        self.assertEqual(json.dumps(agg.to_json()), json.dumps(ref_agg.to_json()))


class Memory(unittest.TestCase):
    """The bulk reader must not peak above the per-entity reader, for dense and for sparse input."""

    @staticmethod
    def peak(fn):
        tracemalloc.start()
        try:
            fn()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def assertNotLarger(self, ref_fn, new_fn):
        old, new = self.peak(ref_fn), self.peak(new_fn)
        self.assertLessEqual(new, old * 1.05 + 16 * 1024, (old, new))

    def test_dense_block(self):
        data = dense_resource(20000)
        self.assertNotLarger(lambda: ref_parse_resource(data), lambda: WI.parse_resource(data))
        self.assertNotLarger(lambda: ref_parse_resource(data, WI.ArchetypeCounts()),
                             lambda: WI.parse_resource(data, WI.ArchetypeCounts()))

    def test_sparse_random_resources(self):
        rng = random.Random(4)
        datas = [build_resource(rng) for _ in range(60)]
        self.assertNotLarger(lambda: [ref_parse_resource(d) for d in datas],
                             lambda: [WI.parse_resource(d) for d in datas])

    def test_many_tiny_resources(self):
        datas = [dense_resource(3) for _ in range(500)]
        self.assertNotLarger(lambda: [ref_parse_resource(d) for d in datas],
                             lambda: [WI.parse_resource(d) for d in datas])


if __name__ == "__main__":
    unittest.main()
