"""The decoder and extent checks of tools/world_archetypes.py give exactly what their straightforward forms give.

The reference functions below are the per-field and per-entity forms the tool used before they were sped up; the tests
compare the tool with them directly (values, key order, and the bit pattern of every float), on synthetic 0x7a
resources and random entities. No game data, and nothing pinned to a platform.

    python -I tests/test_world_archetypes_parity.py
"""
import math
import os
import random
import struct
import sys
import tempfile
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_world_archetypes as T  # noqa: E402  (its builders for synthetic resources)

WA = T.WA
WI = WA.WI
J = WA.joaat_cs


# ---- reference: the per-field decoder and per-entity transform, written the plain way

def ref_value(b, base, member):
    if member is not None and member[1] == WA.ENUM:
        p = base + member[0]
        return struct.unpack_from("<I", b, p)[0] if p + 4 <= len(b) else None
    return WI._value(b, base, member)


def ref_parse_archetypes(b):
    if WI.parse_resource(b) is None:
        return []
    structs, blocks = WA.schema(b)
    out = []
    for name, size, off in blocks:
        kind = WA.KINDS.get(name)
        if kind is None or off is None or name not in structs:
            continue
        ssize, members = structs[name]
        if not ssize or size % ssize or off + size > len(b):
            continue
        for k in range(size // ssize):
            base = off + k * ssize
            row = {key: ref_value(b, base, members.get(h)) for key, h in WA.FIELDS.items()}
            row = {key: v for key, v in row.items() if v is not None}
            row["kind"] = kind
            out.append(row)
    return out


def ref_rotation(q, conjugate):
    x, y, z, w = q or (0.0, 0.0, 0.0, 1.0)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    if conjugate:
        x, y, z = -x, -y, -z
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def ref_scale(entity, key):
    value = entity.get(key)
    return 1.0 if value is None else value


def ref_world_box(entity, lo, hi, conjugate, use_scale):
    r = ref_rotation(entity.get("rotation"), conjugate)
    sxy = ref_scale(entity, "scale_xy") if use_scale else 1.0
    sz = ref_scale(entity, "scale_z") if use_scale else 1.0
    s = (sxy, sxy, sz)
    c = [(lo[i] + hi[i]) * 0.5 * s[i] for i in range(3)]
    e = [(hi[i] - lo[i]) * 0.5 * abs(s[i]) for i in range(3)]
    p = entity.get("position") or (0.0, 0.0, 0.0)
    out_lo, out_hi = [], []
    for i in range(3):
        wc = p[i] + sum(r[i][j] * c[j] for j in range(3))
        we = sum(abs(r[i][j]) * e[j] for j in range(3))
        out_lo.append(wc - we)
        out_hi.append(wc + we)
    return out_lo, out_hi


def ref_union_bounds(entities, defs, conjugate, use_scale):
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for e in entities:
        d = defs[e["archetype"]]
        a, b = ref_world_box(e, d["bb_min"], d["bb_max"], conjugate, use_scale)
        for i in range(3):
            lo[i], hi[i] = min(lo[i], a[i]), max(hi[i], b[i])
    return lo, hi


def ref_union_error(entities, defs, ref_lo, ref_hi, conjugate, use_scale):
    lo, hi = ref_union_bounds(entities, defs, conjugate, use_scale)
    err = max(max(abs(lo[i] - ref_lo[i]), abs(hi[i] - ref_hi[i])) for i in range(3))
    inside = all(ref_lo[i] - WA.TOLERANCE <= lo[i] and hi[i] <= ref_hi[i] + WA.TOLERANCE for i in range(3))
    return err, inside


def bits(value):
    """A comparable form that tells 0.0 from -0.0 and one NaN from another, through lists and tuples."""
    if isinstance(value, float):
        return struct.pack("<d", value)
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    return value


# ---- synthetic resources with every member type the decoder reads, plus ones it does not

TYPES_ALL = (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x21, 0x32, 0x33, 0x34, 0x08, 0x4F, 0x4A, 0x60, 0x62, 0x77)
FIELD_NAMES = ("name", "assetName", "assetType", "lodDist", "hdTextureDist", "specialAttribute", "flags", "bbMin",
               "bbMax", "bsCentre", "bsRadius", "physicsDictionary", "clipDictionary", "timeFlags")


def random_resource(rng, count, ssize=0x90, shrink=0):
    """A resource with ``count`` instances of random bytes under a random member table. Members get random types and
    offsets, some close enough to the end of the struct that the last instance reads past its block; ``shrink`` bytes
    cut from the end of the buffer put the fields of the last instance outside it."""
    members = []
    for name in rng.sample(FIELD_NAMES, rng.randint(1, len(FIELD_NAMES))):
        members.append((J(name), rng.randrange(0, ssize + 16, 1 if rng.random() < 0.3 else 4), rng.choice(TYPES_ALL)))
    payload = bytes(rng.randrange(256) for _ in range(ssize * count))
    kind = rng.choice((T.BASE, T.MLO, J("CTimeArchetypeDef")))
    b = T.build([(WA.MAPTYPES, 0x50, []), (kind, ssize, members)],
                [(WA.MAPTYPES, bytes(0x50)), (kind, payload)])
    return b[:len(b) - shrink] if shrink else b


class ParseParityTest(unittest.TestCase):
    def check(self, data):
        got, want = WA.parse_archetypes(data), ref_parse_archetypes(data)
        # key order, then every value by bit pattern (random bytes decode to NaN, which is never equal to itself)
        self.assertEqual([list(r) for r in got], [list(r) for r in want])
        self.assertEqual([bits(list(r.values())) for r in got], [bits(list(r.values())) for r in want])
        return len(got)

    def test_random_layouts_match_the_reference(self):
        rng = random.Random(20261123)
        total = 0
        for _ in range(60):
            total += self.check(random_resource(rng, rng.randint(1, 40)))
        self.assertGreater(total, 100)  # the comparison is not between two empty lists

    def test_members_running_past_the_struct_or_the_buffer(self):
        rng = random.Random(7)
        for shrink in (1, 3, 8, 17):
            for _ in range(20):
                self.check(random_resource(rng, rng.randint(1, 6), shrink=shrink))

    def test_the_layouts_of_the_existing_tests_match_the_reference(self):
        for layout in (T.LAYOUT_A, T.LAYOUT_B):
            for kind in (T.BASE, T.MLO):
                data = T.maptypes([(T.ROCK, 2, *T.ROCK_BOX), (T.CRATE, 1, *T.CRATE_BOX)], layout, kind)
                self.assertEqual(self.check(data), 2)


def random_entity(rng, defs):
    e = {"archetype": rng.choice(list(defs))}
    roll = rng.random()
    if roll < 0.2:
        pass
    elif roll < 0.3:
        e["position"] = None
    else:
        e["position"] = [rng.choice((0, -0.0, 0.0, rng.uniform(-500, 500), rng.uniform(-500, 500),
                                     rng.uniform(-500, 500), rng.uniform(-500, 500), math.inf, math.nan))
                         for _ in range(3)]
    roll = rng.random()
    if roll < 0.25:
        e["rotation"] = None if rng.random() < 0.5 else [0.0, 0.0, 0.0, 1.0]
    elif roll < 0.3:
        e["rotation"] = [0.0, 0.0, 0.0, 0.0]  # zero length: normalised by one
    elif roll < 0.4:
        e["rotation"] = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    else:
        e["rotation"] = [rng.uniform(-1, 1) for _ in range(4)]
    for key in ("scale_xy", "scale_z"):
        if rng.random() < 0.3:
            e[key] = rng.choice((None, 0.0, -0.0, 1.0, -1.0, 2, rng.uniform(-3, 3)))
    return e


def random_defs(rng, count):
    defs = {}
    for i in range(count):
        lo = [rng.choice((0.0, rng.uniform(-9, 0))) for _ in range(3)]
        hi = [v + rng.choice((0.0, rng.uniform(0, 9))) for v in lo]
        defs[0x100 + i] = {"bb_min": lo, "bb_max": hi}
    return defs


class ExtentsParityTest(unittest.TestCase):
    def test_union_bounds_are_bit_identical_to_world_box(self):
        rng = random.Random(99)
        defs = random_defs(rng, 12)
        for _ in range(150):
            ents = [random_entity(rng, defs) for _ in range(rng.randint(0, 9))]
            for conjugate in (False, True):
                for use_scale in (False, True):
                    got = WA._union_bounds(ents, defs, conjugate, use_scale)
                    want = ref_union_bounds(ents, defs, conjugate, use_scale)
                    self.assertEqual(bits(list(got[0])), bits(want[0]))
                    self.assertEqual(bits(list(got[1])), bits(want[1]))

    def test_public_world_box_matches_the_reference(self):
        rng = random.Random(5)
        defs = random_defs(rng, 6)
        for _ in range(100):
            e = random_entity(rng, defs)
            d = defs[e["archetype"]]
            for conjugate in (False, True):
                for use_scale in (False, True):
                    self.assertEqual(bits(WA.world_box(e, d["bb_min"], d["bb_max"], conjugate, use_scale)),
                                     bits(ref_world_box(e, d["bb_min"], d["bb_max"], conjugate, use_scale)))

    def test_union_error_matches_the_reference(self):
        rng = random.Random(11)
        defs = random_defs(rng, 8)
        for _ in range(100):
            ents = [random_entity(rng, defs) for _ in range(rng.randint(1, 7))]
            ref_lo = [rng.uniform(-60, 0) for _ in range(3)]
            ref_hi = [rng.uniform(0, 60) for _ in range(3)]
            for conjugate in (False, True):
                for use_scale in (False, True):
                    self.assertEqual(bits(WA._union_error(ents, defs, ref_lo, ref_hi, conjugate, use_scale)),
                                     bits(ref_union_error(ents, defs, ref_lo, ref_hi, conjugate, use_scale)))

    def test_check_extents_matches_the_reference_on_random_maps(self):
        rng = random.Random(31)
        defs = random_defs(rng, 10)
        maps = []
        for _ in range(40):
            ents = [random_entity(rng, defs) for _ in range(rng.randint(1, 8))]
            lo, hi = ref_union_bounds(ents, defs, True, False)
            jitter = rng.choice((0.0, 0.01, 1.0))
            maps.append(({"entities_min": [v + rng.uniform(-jitter, jitter) for v in lo],
                          "entities_max": [v + rng.uniform(-jitter, jitter) for v in hi]}, ents))
        original = WA._union_error
        try:  # the same check_extents, once with the kernel replaced by the reference
            WA._union_error = ref_union_error
            want = WA.check_extents(maps, defs)
        finally:
            WA._union_error = original
        got = WA.check_extents(maps, defs)
        self.assertEqual(got, want)
        self.assertGreater(got["maps_fully_defined"], 10)
        self.assertGreater(got["rotation"]["maps"] + got["scale"]["maps"], 10)


def ref_run(path, start=None, limit=None, extents=False):
    """run() with the per-field reference decoder: the whole document must come out the same."""
    defs, maps, resources = {}, [], []
    stats = {"resources": 0, "maptypes": 0, "archetype_resources": 0, "archetypes": 0, "kinds": {},
             "asset_types": {}, "names": 0, "names_with_differing_boxes": 0}
    schemas = {}
    for row, data in WA.iter_resources(path, start, limit):
        stats["resources"] += 1
        if row.get("kind") == "map":
            if extents:
                ents = []
                WI.parse_resource(data, collect=ents)
                maps.append((row, ents))
            continue
        structs, _ = WA.schema(data)
        if WA.MAPTYPES in structs:
            stats["maptypes"] += 1
        found = ref_parse_archetypes(data)
        if not found:
            continue
        stats["archetype_resources"] += 1
        for h, kind in WA.KINDS.items():
            if h in structs and kind not in schemas:
                schemas[kind] = {"size": structs[h][0], "members": [
                    {"hash": "%08x" % mh, "offset": off, "type": "0x%02x" % ty}
                    for mh, off, ty in WA.member_table(data, h)]}
        for d in found:
            d["source"] = row.get("file") or row["offset"]
            resources.append(d)
            stats["archetypes"] += 1
            stats["kinds"][d["kind"]] = stats["kinds"].get(d["kind"], 0) + 1
            t = str(d.get("asset_type"))
            stats["asset_types"][t] = stats["asset_types"].get(t, 0) + 1
            prev = defs.get(d.get("name"))
            if prev is None:
                defs[d.get("name")] = d
            elif (prev.get("bb_min"), prev.get("bb_max")) != (d.get("bb_min"), d.get("bb_max")):
                prev["_differs"] = True
    stats["names"] = len(defs)
    stats["names_with_differing_boxes"] = sum(1 for d in defs.values() if d.pop("_differs", False))
    doc = {"summary": stats, "schemas": schemas, "archetypes": resources}
    if extents:
        doc["extents"] = WA.check_extents(maps, defs)
    return doc


class RunParityTest(unittest.TestCase):
    def test_a_folder_of_carved_files_gives_the_same_document(self):
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as folder:
            # three definition files (one name twice with different boxes), a map and a resource of random bytes
            for i, defs in enumerate(([(T.ROCK, 2, *T.ROCK_BOX), (T.CRATE, 1, *T.CRATE_BOX)],
                                      [(T.ROCK, 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
                                      [(0x33333333 + k, k % 3, *T.CRATE_BOX) for k in range(25)])):
                with open(os.path.join(folder, "t%d.bin" % i), "wb") as f:
                    f.write(T.maptypes(defs, T.LAYOUT_A if i != 1 else T.LAYOUT_B, T.BASE if i != 2 else T.MLO))
            with open(os.path.join(folder, "m.bin"), "wb") as f:
                f.write(T.scene()[1])
            with open(os.path.join(folder, "r.bin"), "wb") as f:
                f.write(random_resource(rng, 12))
            for extents in (False, True):
                self.assertEqual(WA.run(folder, extents=extents), ref_run(folder, extents=extents))


class MemoryTest(unittest.TestCase):
    """The row decoder keeps no cache and holds only the rows it returns: peak memory does not grow."""

    def peak(self, fn, *args):
        tracemalloc.start()
        try:
            fn(*args)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_dense_and_sparse_resources_do_not_raise_the_peak(self):
        dense = T.maptypes([(0x1000 + i, i % 5, *T.ROCK_BOX) for i in range(4000)])
        sparse = random_resource(random.Random(2), 1)  # one instance under a random member table
        empty = T.build([(WA.MAPTYPES, 0x50, []), (T.BASE, 0x90, [(J("notAField"), 0, 0x15)])],
                        [(WA.MAPTYPES, bytes(0x50)), (T.BASE, bytes(0x90 * 3000))])  # 3000 rows of only "kind"
        for name, data in (("dense", dense), ("sparse", sparse), ("empty rows", empty)):
            with self.subTest(name):
                new = self.peak(WA.parse_archetypes, data)
                old = self.peak(ref_parse_archetypes, data)
                self.assertLessEqual(new, old * 1.02 + 2048, (new, old))

    def test_extents_kernel_allocates_nothing_per_entity(self):
        rng = random.Random(4)
        defs = random_defs(rng, 5)
        ents = [random_entity(rng, defs) for _ in range(3000)]
        new = self.peak(WA._union_bounds, ents, defs, True, True)
        old = self.peak(ref_union_bounds, ents, defs, True, True)
        self.assertLessEqual(new, old + 2048, (new, old))


if __name__ == "__main__":
    unittest.main()
