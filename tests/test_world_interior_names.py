"""Tests for tools/world_interior_names.py, built from synthetic 0x7a resources (no game data).

    python -I tests/test_world_interior_names.py
"""
import io
import json
import math
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Imported by its real name: --jobs sends scan_window to worker processes, which import it by module name.
sys.path.insert(0, os.path.join(ROOT, "tools"))
import world_interior_names as T  # noqa: E402
WA, WI = T.WA, T.WI
J = WA.joaat_cs
SYS = 0x50000000
S = math.sqrt(0.5)


def build(structs, blocks):
    """A valid 0x7a resource. structs: [(hash, size, [(member hash, offset, type)])]; blocks: [(hash, bytes)]."""
    st = 0x60
    p = st + 32 * len(structs)
    member_at = []
    for _, _, members in structs:
        member_at.append(p)
        p += 16 * len(members)
    bt = (p + 15) & ~15
    p = bt + 16 * len(blocks)
    data_at = []
    for _, payload in blocks:
        p = (p + 15) & ~15
        data_at.append(p)
        p += len(payload)
    b = bytearray(p + 16)
    b[0x10:0x15] = b"0DRPz"
    struct.pack_into("<Q", b, 0x08, SYS)
    struct.pack_into("<I", b, 0x20, 1)
    struct.pack_into("<Q", b, 0x28, SYS | st)
    struct.pack_into("<Q", b, 0x38, SYS | bt)
    struct.pack_into("<HHH", b, 0x48, len(structs), 0, len(blocks))
    for i, (h, size, members) in enumerate(structs):
        o = st + 32 * i
        struct.pack_into("<I", b, o, h)
        struct.pack_into("<Q", b, o + 0x10, SYS | member_at[i])
        struct.pack_into("<I", b, o + 0x18, size)
        struct.pack_into("<H", b, o + 0x1E, len(members))
        for k, (mh, off, ty) in enumerate(members):
            struct.pack_into("<IIBBHI", b, member_at[i] + 16 * k, mh, off, ty, 0, 0, 0)
    for i, (h, payload) in enumerate(blocks):
        struct.pack_into("<IIQ", b, bt + 16 * i, h, len(payload), SYS | data_at[i])
        b[data_at[i]:data_at[i] + len(payload)] = payload
    return bytes(b)


ROOM_SIZE, MLO_SIZE, INST_SIZE = 0x40, 0x60, 0x40
ROOM_MEMBERS = [(T.NAME, 0x00, 0x4A), (T.BBMIN, 0x10, 0x33), (T.BBMAX, 0x20, 0x33), (T.ROOM_KEY, 0x34, 0x4A)]
KEY_MEMBER = J("someKey")
MLO_MEMBERS = [(T.NAME, 0x00, 0x4A), (KEY_MEMBER, 0x08, T.U64), (J("otherKey"), 0x10, T.U64),
               (T.BBMIN, 0x18, 0x32), (T.BBMAX, 0x24, 0x32), (J("rooms"), 0x30, T.ARRAY), (J("portals"), 0x40, T.ARRAY)]
INST_MEMBERS = [(WI.FIELD["archetypeName"], 0x00, 0x4A), (WI.FIELD["position"], 0x04, 0x32),
                (WI.FIELD["rotation"], 0x10, 0x34), (WI.FIELD["scaleXY"], 0x20, 0x21), (WI.FIELD["scaleZ"], 0x24, 0x21)]
ROOM_BOX = ((-2.0, -1.0, 0.0), (2.0, 1.0, 3.0))
BOX = ((-4.0, -1.0, 0.0), (4.0, 1.0, 3.0))  # long in X so the rotation convention changes the world box


def room(name, lo, hi, key=0):
    b = bytearray(ROOM_SIZE)
    struct.pack_into("<I", b, 0, name)
    struct.pack_into("<I", b, 0x34, key)
    struct.pack_into("<3f", b, 0x10, *lo)
    struct.pack_into("<3f", b, 0x20, *hi)
    return bytes(b)


def interior(name, key64, first_room, rooms):
    b = bytearray(MLO_SIZE)
    struct.pack_into("<IxxxxQQ", b, 0, name, key64, 0x1234)
    struct.pack_into("<3f", b, 0x18, *BOX[0])
    struct.pack_into("<3f", b, 0x24, *BOX[1])
    struct.pack_into("<QH", b, 0x30, 1 | ((first_room * ROOM_SIZE) << 20), rooms)  # block 1 = the room block
    return bytes(b)


def placement(name, pos, rot=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0)):
    b = bytearray(INST_SIZE)
    struct.pack_into("<I3f4f2f", b, 0, name, *pos, *rot, *scale)
    return bytes(b)


KEY_A, KEY_B, KEY_C = "TEST_INT01_ROOM_1", "TEST_INT02_HALL_1", "NOT_AN_INTERIOR_1"
HA, HB, HC = T.joaat(KEY_A), T.joaat(KEY_B), T.joaat(KEY_C)


def interiors():
    """Two interiors in one resource, two rooms each: limbo first, as in the game data."""
    rooms = (room(T.LIMBO, (0, 0, 0), (0, 0, 0)) + room(J("a"), *ROOM_BOX, key=HA)
             + room(T.LIMBO, (0, 0, 0), (0, 0, 0)) + room(J("b"), *ROOM_BOX))
    payload = interior(HA, T.hash64(KEY_A.lower()), 0, 2) + interior(HB, 0xDEAD, 2, 2)
    return build([(T.ROOM, ROOM_SIZE, ROOM_MEMBERS), (T.MLO, MLO_SIZE, MLO_MEMBERS)],
                 [(T.ROOM, rooms), (T.MLO, payload)])


def placements():
    rot = (0.0, 0.0, S, S)  # +90 degrees about Z as stored
    payload = (placement(HA, (100.0, 0.0, 0.0), rot) + placement(HB, (0.0, 200.0, 0.0))
               + placement(HB, (0.0, 300.0, 0.0)))
    return build([(T.INSTANCE, INST_SIZE, INST_MEMBERS)], [(T.INSTANCE, payload)])


NAMES = {"InteriorNames": {
    KEY_A: [{"Name": "LOCATION_A", "Pos": [100.0, 3.0, 1.0]}],
    KEY_B: [{"Name": "", "Pos": [0.0, 200.0, 1.0]}, {"Name": "", "Pos": [0.0, 310.0, 1.0]}],
    KEY_C: [{"Name": "", "Pos": [0.0, 0.0, 0.0]}],
}}


def rpf(parts, start=0x40):
    out = bytearray(start)
    out[:4] = b"9FPR"
    struct.pack_into("<I", out, 4, start)
    for part in parts:
        out.extend(b"\0" * (((len(out) + 15) & ~15) - len(out)))
        out.extend(zlib.compress(part))
    return bytes(out)


class ParseTest(unittest.TestCase):
    def test_each_interior_gets_its_own_rooms(self):
        found, placed = T.parse(interiors())
        self.assertEqual(placed, [])
        self.assertEqual([d["name"] for d in found], [HA, HB])
        self.assertEqual([[r[0] for r in d["rooms"]] for d in found], [[T.LIMBO, J("a")], [T.LIMBO, J("b")]])
        self.assertEqual(found[0]["keys"]["%08x" % KEY_MEMBER], T.hash64(KEY_A.lower()))
        self.assertEqual((found[0]["bb_min"], found[0]["bb_max"]), ([-4, -1, 0], [4, 1, 3]))

    def test_placements_read_by_member_hash(self):
        _, placed = T.parse(placements())
        self.assertEqual([p["archetype"] for p in placed], [HA, HB, HB])
        self.assertEqual(placed[0]["position"], [100.0, 0.0, 0.0])
        self.assertEqual((placed[0]["scale_xy"], placed[0]["scale_z"]), (1.0, 1.0))

    def test_room_key_member_is_read(self):
        found, _ = T.parse(interiors())
        self.assertEqual([r[3] for r in found[0]["rooms"]], [0, HA])

    def test_array_reference_out_of_range_is_ignored(self):
        b = bytearray(interiors())
        structs, blocks = WA.schema(bytes(b))
        _, _, off = blocks[1]
        struct.pack_into("<Q", b, off + 0x30, 9)  # block 9 does not exist
        self.assertEqual(T.parse(bytes(b))[0][0]["rooms"], [])

    def test_invalid_input_gives_nothing(self):
        self.assertEqual(T.parse(b"\0" * 0x100), ([], []))


class ResolveTest(unittest.TestCase):
    def resolve(self):
        found = {0: T.parse(interiors()), 1: T.parse(placements())}
        names = {k: [it["Pos"] for it in v] for k, v in NAMES["InteriorNames"].items()}
        return T.resolve(names, found)

    def test_keys_name_interiors_and_64bit_confirmation(self):
        s, rows = self.resolve()
        self.assertEqual((s["keys"], s["positions"], s["keys_naming_an_interior"]), (3, 4, 2))
        self.assertEqual(s["confirmed_by_64bit_key"], {"%08x" % KEY_MEMBER: 1})
        self.assertEqual(s["keys_not_confirmed"], 1)  # KEY_B's interior carries a different 64-bit key
        self.assertNotIn("placements", rows[KEY_C])

    def test_positions_against_placements(self):
        s, rows = self.resolve()
        self.assertEqual(s["keys_with_placements"], 2)
        self.assertEqual(s["positions_vs_placements"], {"equal": 2, "more_positions": 0, "fewer_positions": 0})
        self.assertEqual(rows[KEY_B]["outside_box_m_stored"], [0.0, 9.0])  # 10 m along Y from the second copy, box +-1 m

    def test_rotation_is_applied(self):
        s, rows = self.resolve()
        # +90 degrees about Z turns the 8 m long box along world Y, so a point 3 m along Y is inside it (unrotated
        # it would be 2 m outside). The box is symmetric, so both conventions agree here.
        self.assertEqual(rows[KEY_A]["outside_box_m_stored"], [0.0])
        self.assertEqual(rows[KEY_A]["outside_box_m_conjugate"], [0.0])
        # Rooms are 4 m long: KEY_A's point (3 m along the box) is outside room "a"; KEY_B's first point is at the
        # centre of room "b", its second 10 m away from either copy.
        self.assertEqual(s["positions_in_a_room_box"], {"stored": 1, "conjugate": 1})

    def test_room_counts(self):
        s, _ = self.resolve()
        self.assertEqual((s["interiors_with_rooms"], s["interiors_first_room_limbo"]), (2, 2))
        self.assertEqual(s["keys_in_a_room_key_member"], 1)  # room "a" holds KEY_A's hash

    def test_placement_scale_is_applied(self):
        # The reviewer's case: archetype and room both [-1, 1], placed at the origin with scaleXY 2, scaleZ 1, and
        # a position at x = 1.5: inside the scaled box (X spans [-2, 2]) and inside the scaled room.
        unit = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        d = {"name": HA, "keys": {}, "bb_min": unit[0], "bb_max": unit[1],
             "rooms": [[T.LIMBO, [0, 0, 0], [0, 0, 0], 0], [J("a"), unit[0], unit[1], 0]]}
        p = {"archetype": HA, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0],
             "scale_xy": 2.0, "scale_z": 1.0}
        s, rows = T.resolve({KEY_A: [[1.5, 0.0, 0.0], [0.0, 0.0, 1.5]]}, {0: ([d], [p])})
        # The second position is 1.5 m up: scaleZ 1 keeps the box at z <= 1, so it is 0.5 m outside (in metres).
        self.assertEqual(rows[KEY_A]["outside_box_m_stored"], [0.0, 0.5])
        self.assertEqual(rows[KEY_A]["outside_box_m_conjugate"], [0.0, 0.5])
        self.assertEqual(s["positions_in_a_room_box"], {"stored": 1, "conjugate": 1})

    def test_room_face_after_rotation_counts(self):
        # Review case on fcfed39: +90 degrees about X stored as float32, scale XY 1.5 and Z -2. The room's upper
        # world Z face is 30.75; the inverse rotation puts a point on it at local Y 0.7500000000000003.
        q = struct.unpack("<4f", struct.pack("<4f", S, 0.0, 0.0, S))
        d = {"name": HA, "keys": {}, "bb_min": [1.0, -2.0, 0.0], "bb_max": [4.0, 1.0, 3.0],
             "rooms": [[J("a"), [2.0, -1.0, 0.5], [3.0, 0.5, 2.0], 0]]}
        p = {"archetype": HA, "position": [10.0, 20.0, 30.0], "rotation": list(q), "scale_xy": 1.5, "scale_z": -2.0}
        for z, hits in ((30.749, 1), (30.75, 1), (30.751, 0)):
            s, _ = T.resolve({KEY_A: [[13.75, 22.5, z]]}, {0: ([d], [p])})
            self.assertEqual(s["positions_in_a_room_box"]["stored"], hits, z)

    def test_interior_face_after_rotation_counts_as_inside(self):
        # The room-face case above, on the interior box instead. Scale Z -2 puts the box's local Z face at 0, which
        # +90 degrees about X turns into world Y 20. The inverse rotation (float32 quaternion) leaves the point on
        # that face about 8e-18 m outside, so without the tolerance the "inside" bucket (within 0 m) missed it while
        # the room test counted the same kind of point. Found by mapping face points forward and back.
        q = struct.unpack("<4f", struct.pack("<4f", S, 0.0, 0.0, S))
        d = {"name": HA, "keys": {}, "bb_min": [1.0, -2.0, 0.0], "bb_max": [4.0, 1.0, 3.0], "rooms": []}
        p = {"archetype": HA, "position": [10.0, 20.0, 30.0], "rotation": list(q), "scale_xy": 1.5, "scale_z": -2.0}
        for y, inside in ((20.001, 1), (20.0, 1), (19.999, 0)):
            s, _ = T.resolve({KEY_A: [[11.613, y, 30.038]]}, {0: ([d], [p])})
            self.assertEqual(s["distance_outside_own_interior_box"]["stored"]["within_m"]["0"], inside, y)

    def test_placements_without_position_are_counted_not_fatal(self):
        unit = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        d = {"name": HA, "keys": {}, "bb_min": unit[0], "bb_max": unit[1], "rooms": []}
        p = {"archetype": HA, "position": None, "rotation": None, "scale_xy": None, "scale_z": None}
        s, rows = T.resolve({KEY_A: [[0.0, 0.0, 0.0]]}, {0: ([d], [p])})
        self.assertEqual((s["keys_with_placements"], s["keys_without_a_placement_position"]), (1, 1))
        self.assertNotIn("outside_box_m_stored", rows[KEY_A])
        self.assertEqual(s["distance_outside_own_interior_box"]["stored"]["n"], 0)
        # Counted even when the interior has no box either.
        d2 = dict(d, bb_min=None, bb_max=None)
        s, _ = T.resolve({KEY_A: [[0.0, 0.0, 0.0]]}, {0: ([d2], [p])})
        self.assertEqual(s["keys_without_a_placement_position"], 1)

    def test_scaled_box_negative_and_missing_scale(self):
        lo, hi = [-1.0, -2.0, 0.0], [3.0, 2.0, 4.0]
        self.assertEqual(T._scaled(lo, hi, {"scale_xy": -2.0, "scale_z": 0.5}), ([-6.0, -4.0, 0.0], [2.0, 4.0, 2.0]))
        # A missing scale is 1; an explicit zero collapses the axis, as world_archetypes.world_box does.
        self.assertEqual(T._scaled(lo, hi, {"scale_xy": None, "scale_z": 0.0}), ([-1.0, -2.0, 0.0], [3.0, 2.0, 0.0]))

    def test_zero_scale_collapses_the_axis(self):
        # Box [-1, 1] placed at the origin with scaleZ 0: the box is the z = 0 plane, so a point 0.5 m up is 0.5 m
        # outside. Read as scale 1 (the old rule), it was inside.
        unit = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        d = {"name": HA, "keys": {}, "bb_min": unit[0], "bb_max": unit[1], "rooms": []}
        p = {"archetype": HA, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0],
             "scale_xy": 1.0, "scale_z": 0.0}
        _, rows = T.resolve({KEY_A: [[0.5, 0.0, 0.5], [0.5, 0.0, 0.0]]}, {0: ([d], [p])})
        self.assertEqual(rows[KEY_A]["outside_box_m_stored"], [0.5, 0.0])

    def test_unknown_placements_are_not_counted_as_placed_interiors(self):
        found = {0: T.parse(interiors()), 1: T.parse(placements()),
                 2: T.parse(build([(T.INSTANCE, INST_SIZE, INST_MEMBERS)],
                                  [(T.INSTANCE, placement(HC, (0.0, 0.0, 0.0)))]))}
        s, _ = T.resolve({}, found)
        self.assertEqual((s["placements"], s["placed_interior_names"]), (4, 2))
        self.assertEqual((s["placed_unknown_names"], s["placements_of_unknown_interior"]), (1, 1))

    def test_duplicate_interior_definition_keeps_the_first(self):
        found = {0: T.parse(interiors()), 1: T.parse(interiors()), 2: T.parse(placements())}
        s, _ = T.resolve({}, found)
        self.assertEqual((s["interior_definitions"], s["interior_names"]), (4, 2))
        self.assertEqual(s["duplicate_interior_definitions"], 2)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _turn(q, v):
    """v rotated by unit quaternion q as q v q* (Hamilton product; independent of the tool's matrix)."""
    x, y, z, w = q
    return _qmul(_qmul(q, (v[0], v[1], v[2], 0.0)), (-x, -y, -z, w))[:3]


def _f32(values):
    return list(struct.unpack("<%df" % len(values), struct.pack("<%df" % len(values), *values)))


class ForwardConstructionTest(unittest.TestCase):
    """Points built in a placed box's own scaled frame from a known answer, mapped to world space, then measured.

    Seeded, 600 cases: 30% axis-aligned rotations, scales including negative, 0.998 and 0, float32 quaternions and
    positions up to 13 km. Expected bounds and rotations are computed here, not with the tool's helpers."""

    SCALES = (1.0, -1.0, 2.0, -2.0, 0.998, 1.5, 0.5, 0.0)

    def case(self, rng):
        lo = [rng.uniform(-10, 0) for _ in range(3)]
        hi = [lo[i] + rng.uniform(1, 20) for i in range(3)]
        rlo = [rng.uniform(lo[i], hi[i] - 0.5) for i in range(3)]
        rhi = [rng.uniform(rlo[i] + 0.25, hi[i]) for i in range(3)]
        if rng.random() < 0.3:
            axis, turns = rng.randrange(3), rng.randrange(4)
            q = [0.0, 0.0, 0.0, math.cos(turns * math.pi / 4)]
            q[axis] = math.sin(turns * math.pi / 4)
        else:
            q = [rng.gauss(0, 1) for _ in range(4)]
        q = _f32(q)
        n = math.sqrt(sum(c * c for c in q))
        unit = [c / n for c in q]
        sxy, sz = rng.choice(self.SCALES), rng.choice(self.SCALES)
        s = (sxy, sxy, sz)
        blo = [min(lo[i] * s[i], hi[i] * s[i]) for i in range(3)]
        bhi = [max(lo[i] * s[i], hi[i] * s[i]) for i in range(3)]
        mlo = [min(rlo[i] * s[i], rhi[i] * s[i]) for i in range(3)]
        mhi = [max(rlo[i] * s[i], rhi[i] * s[i]) for i in range(3)]
        kind = rng.choice(("inside", "face", "edge", "corner", "room face", "out"))
        u, gap, room = [rng.uniform(blo[i], bhi[i]) for i in range(3)], 0.0, None
        if kind in ("face", "edge", "corner", "out"):
            for i in rng.sample(range(3), {"face": 1, "edge": 2, "corner": 3, "out": 1}[kind]):
                u[i] = rng.choice((blo[i], bhi[i]))
                if kind == "out":
                    gap = rng.choice((5e-6, 1e-4, 0.01, 0.5, 3.0))  # the small ones pin EPS from above
                    u[i] += gap if u[i] == bhi[i] else -gap
        elif kind == "room face":
            u = [rng.uniform(mlo[i], mhi[i]) for i in range(3)]
            i = rng.randrange(3)
            u[i] = rng.choice((mlo[i], mhi[i]))
            room = 1
        if room is None:  # decided on exact values: random points are not within 1e-6 m of a room face
            room = int(all(mlo[i] < u[i] < mhi[i] for i in range(3)))
            if any(abs(u[i] - mlo[i]) < 1e-5 or abs(u[i] - mhi[i]) < 1e-5 for i in range(3)):
                return None
        conj = rng.random() < 0.5
        r = [-unit[0], -unit[1], -unit[2], unit[3]] if conj else unit
        pos = _f32([rng.uniform(-13000, 13000) for _ in range(3)])
        world = [pos[i] + c for i, c in enumerate(_turn(r, u))]
        d = {"name": HA, "keys": {}, "bb_min": lo, "bb_max": hi, "rooms": [[J("a"), rlo, rhi, 0]]}
        p = {"archetype": HA, "position": pos, "rotation": q, "scale_xy": sxy, "scale_z": sz}
        # The other convention, measured here from the world point: inverse-rotate by its quaternion and take
        # the distance to the scaled box, so a swap of the two conventions shows.
        other = unit if conj else [-unit[0], -unit[1], -unit[2], unit[3]]
        v = _turn([-other[0], -other[1], -other[2], other[3]], [world[i] - pos[i] for i in range(3)])
        other_gap = math.sqrt(sum(max(blo[i] - v[i], 0.0, v[i] - bhi[i]) ** 2 for i in range(3)))
        return ("conjugate" if conj else "stored"), world, gap, room, d, p, kind, other_gap

    def test_tool_recovers_constructed_distances_and_room_hits(self):
        rng, done = random.Random(3813), 0
        while done < 600:
            c = self.case(rng)
            if c is None:
                continue
            tag, world, gap, room, d, p, kind, other_gap = c
            s, rows = T.resolve({KEY_A: [world]}, {0: ([d], [p])})
            got = rows[KEY_A]["outside_box_m_" + tag][0]
            self.assertAlmostEqual(got, gap, delta=1.5e-3, msg=(done, kind, p))
            # The "inside" bucket, not only the rounded distance: a face point left 1e-17 m out reads 0.0 above.
            self.assertEqual(s["distance_outside_own_interior_box"][tag]["within_m"]["0"], int(gap == 0), (done, kind, p))
            self.assertEqual(s["positions_in_a_room_box"][tag], room, (done, kind, p))
            other = "stored" if tag == "conjugate" else "conjugate"
            self.assertAlmostEqual(rows[KEY_A]["outside_box_m_" + other][0], other_gap, delta=1.5e-3, msg=(done, kind))
            done += 1


class HelperTest(unittest.TestCase):
    def test_median_of_sorted_list(self):
        self.assertIsNone(T._median([]))
        self.assertEqual(T._median([1.0, 2.0, 7.0]), 2.0)
        self.assertEqual(T._median([1.0, 2.0, 4.0, 7.0]), 3.0)
        b = T._buckets([3.0, 0.0, 1.0, 2.0])
        self.assertEqual((b["n"], b["median_m"], b["within_m"]["1"]), (4, 1.5, 2))

    def test_duplicate_keys_keep_every_item_and_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "names.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"InteriorNames": {"K": [{"Pos": [1, 2, 3]}], "J": {"Pos": [0, 0, 0]},'
                        ' "K": {"Pos": [4, 5, 6]}}}')
            names, dup, malformed = T.load_names(path)
        self.assertEqual((dup, malformed), (1, 0))
        self.assertEqual(names, {"K": [[1, 2, 3], [4, 5, 6]], "J": [[0, 0, 0]]})

    def test_keys_repeated_inside_an_item_are_duplicates_not_malformed(self):
        # Review note on 7651e788: a repeated Pos inside one item was joined into a six-number list and counted as
        # malformed. It names two positions, so it is counted as a repeated key and gives none; a repeated Name
        # is counted and the item's Pos kept. No bookkeeping key leaks into the result.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "names.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"InteriorNames": {"K": [{"Pos": [1, 2, 3], "Pos": [4, 5, 6]},'
                        ' {"Name": "a", "Name": "b", "Pos": [7, 8, 9]}]}}')
            names, dup, malformed = T.load_names(path)
        self.assertEqual((names, dup, malformed), ({"K": [[7.0, 8.0, 9.0]]}, 2, 0))

    def test_repeated_section_keeps_both_and_is_counted(self):
        # Review note: a repeated InteriorNames section was dropped without a count. Both copies are kept now.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "names.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"InteriorNames": {"A": [{"Pos": [1, 2, 3]}]},'
                        ' "InteriorNames": {"B": [{"Pos": [4, 5, 6]}], "A": {"Pos": [7, 8, 9]}}}')
            names, dup, malformed = T.load_names(path)
            self.assertEqual((names, dup, malformed), ({"A": [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
                                                        "B": [[4.0, 5.0, 6.0]]}, 2, 0))
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"InteriorNames": {"A": []}, "InteriorNames": 5}')
            with self.assertRaises(ValueError):
                T.load_names(path)

    def test_malformed_positions_are_counted_not_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "names.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"InteriorNames": {"K": [{"Pos": [1, 2]}, {"Pos": "x"}, {"Pos": [1, 2, true]},'
                        ' {"Pos": [1, 2, 3]}, {"Pos": 5}, {"Pos": []}, {"Pos": 0}, {"Pos": false},'
                        ' {"Pos": [1, 2, 1%s]}, {"Name": "no position"}]}}' % ("0" * 400))
            names, dup, malformed = T.load_names(path)
        # Every present Pos that is not three finite numbers is counted, the empty and false ones and an integer
        # past float range included; an item without Pos is not a position at all.
        self.assertEqual((names, dup, malformed), ({"K": [[1.0, 2.0, 3.0]]}, 0, 8))

    def test_json_type_names(self):
        self.assertEqual([T._json_type(v) for v in (None, True, 1, 1.5, "s", [], {})],
                         ["null", "a boolean", "a number", "a number", "a string", "a list", "an object"])


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.names = os.path.join(self.tmp.name, "names.json")
        with open(self.names, "w", encoding="utf-8") as f:
            json.dump(NAMES, f)

    def run_cli(self, path, *extra):
        out = os.path.join(self.tmp.name, "out", "interiors.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(T.main([path, "--names", self.names, "--out", out, *extra]), 0)
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        return json.loads(raw)["summary"]

    def test_rpf9_archive_one_and_several_jobs(self):
        path = os.path.join(self.tmp.name, "levels.rpf")
        with open(path, "wb") as f:
            f.write(rpf([interiors(), placements()]))
        for jobs in ("1", "3"):
            s = self.run_cli(path, "--jobs", jobs)
            self.assertEqual((s["interior_names"], s["placements"], s["keys_naming_an_interior"]), (2, 3, 2), jobs)

    def test_folder_of_carved_files(self):
        folder = os.path.join(self.tmp.name, "carved")
        os.makedirs(folder)
        for name, data in (("a.bin", interiors()), ("b.bin", placements())):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(data)
        s = self.run_cli(folder)
        self.assertEqual((s["interior_names"], s["placed_interior_names"]), (2, 2))

    def test_out_never_overwrites_an_input(self):
        path = os.path.join(self.tmp.name, "levels.rpf")
        archive = rpf([interiors(), placements()])
        with open(path, "wb") as f:
            f.write(archive)
        folder = os.path.join(self.tmp.name, "carved")
        os.makedirs(folder)
        carved = os.path.join(folder, "a.bin")
        with open(carved, "wb") as f:
            f.write(interiors())
        with open(self.names, "rb") as f:
            names = f.read()
        for src, out in ((path, path), (path, self.names), (folder, carved), (folder, os.path.join(folder, "r.json"))):
            err = io.StringIO()
            sys.stderr, saved = err, sys.stderr
            try:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(T.main([src, "--names", self.names, "--out", out]), 2, out)
            finally:
                sys.stderr = saved
            self.assertEqual(len(err.getvalue().strip().splitlines()), 1)
        for p, data in ((path, archive), (self.names, names), (carved, interiors())):
            with open(p, "rb") as f:
                self.assertEqual(f.read(), data, p)
        self.assertFalse(os.path.exists(os.path.join(folder, "r.json")))

    def fails_with_one_line(self, argv):
        """main(argv) returns 2 with exactly one stderr line, prints nothing and writes no report."""
        out = os.path.join(self.tmp.name, "fail", "r.json")
        err = io.StringIO()
        sys.stderr, saved = err, sys.stderr
        try:
            with redirect_stdout(io.StringIO()) as so:
                self.assertEqual(T.main([*argv, "--out", out]), 2, argv)
        finally:
            sys.stderr = saved
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1, err.getvalue())
        self.assertEqual(so.getvalue(), "")
        self.assertFalse(os.path.exists(out))

    def test_out_under_a_file_fails_before_the_scan(self):
        folder = os.path.join(self.tmp.name, "carved")
        os.makedirs(folder)
        with open(os.path.join(folder, "a.bin"), "wb") as f:
            f.write(interiors())
        saved = T.scan
        T.scan = lambda *args: self.fail("scan started")
        try:
            for out in (os.path.join(self.names, "r.json"), os.path.join(self.names, "sub", "r.json")):
                err = io.StringIO()
                sys.stderr, saved_err = err, sys.stderr
                try:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(T.main([folder, "--names", self.names, "--out", out]), 2, out)
                finally:
                    sys.stderr = saved_err
                self.assertEqual(len(err.getvalue().strip().splitlines()), 1, err.getvalue())
        finally:
            T.scan = saved
        self.assertTrue(os.path.isfile(self.names))

    def test_pool_size_goes_through_process_workers(self):
        # Windows refuses a ProcessPoolExecutor above 61 workers; tools/pool_workers.py caps it there.
        path = os.path.join(self.tmp.name, "levels.rpf")
        with open(path, "wb") as f:
            f.write(rpf([interiors(), placements()]))
        asked = []
        saved = getattr(T, "process_workers", None)
        T.process_workers = lambda n: asked.append(n) or 2
        try:
            s = self.run_cli(path, "--jobs", "70")
        finally:
            T.process_workers = saved
        self.assertEqual(asked, [70])
        self.assertEqual((s["interior_names"], s["placements"]), (2, 3))  # 70 windows on 2 workers, same result

    def test_malformed_names_file_is_one_line(self):
        folder = os.path.join(self.tmp.name, "carved")
        os.makedirs(folder)
        for i, text in enumerate(("", "[]", '{"InteriorNames": null}', '{"InteriorNames": []}', "{")):
            bad = os.path.join(self.tmp.name, "bad%d.json" % i)
            with open(bad, "w", encoding="utf-8") as f:
                f.write(text)
            self.fails_with_one_line([folder, "--names", bad])
        self.fails_with_one_line([folder, "--names", folder])  # a folder, not a file

    def test_start_outside_the_data_area_is_one_line(self):
        path = os.path.join(self.tmp.name, "levels.rpf")
        archive = rpf([interiors(), placements()])
        with open(path, "wb") as f:
            f.write(archive)
        for jobs in ("1", "3"):  # with 3 the error used to come back from a worker as a pickled traceback
            for start in ("1", str(len(archive))):
                self.fails_with_one_line([path, "--names", self.names, "--start", start, "--jobs", jobs])

    def test_bad_arguments(self):
        for argv in (["nope", "--names", self.names], [self.names, "--names", self.names, "--jobs", "0"],
                     [self.names, "--names", self.names, "--start", "-1"],
                     [self.names, "--names", self.names, "--limit", "0"]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                sys.stderr, saved = io.StringIO(), sys.stderr
                try:
                    T.main(argv)
                finally:
                    sys.stderr = saved


if __name__ == "__main__":
    unittest.main()
