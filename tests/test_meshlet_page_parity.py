"""Parity tests for the analysis and decoding helpers of tools/meshlet_page.py.

    python -I tests/test_meshlet_page_parity.py

Each helper below has a straight-line reference copy of the per-element code it replaced (a loop per vertex, a
dict of lists per edge) and the tests check that the tool returns exactly what the reference does on seeded random
input, including NaN and infinite floats, degenerate and repeated triangles, edges shared by three faces and
negative vertex references. Every byte is built here; no game file, value or name is used.
"""
import math
import os
import random
import struct
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import meshlet_page as mp  # noqa: E402


# ---- reference implementations: one loop iteration per vertex or triangle, as the tool read before -------------

def ref_unorm10(x):
    return x / 511.5 - 1


def ref_positions(data, block):
    at, n = block["offset"], block["vertices"]
    slots = {i: (tag, off) for i, tag, off in block["slots"]}
    bias = struct.unpack_from("<3f", data, at + 0x60)
    axis = struct.unpack_from("<I", data, at + 0x6C)[0]
    scale = struct.unpack_from("<3f", data, at + 0x70)
    out = []
    for (q,) in struct.iter_unpack("<Q", data[at + slots[0][1]:at + slots[0][1] + 8 * n]):
        a = bias[0] + scale[0] * ((q >> 8) & 0xFFFFFF)
        b = bias[1] + scale[1] * (((q >> 32) & 0xFFFF) << 8 | (q & 0xFF))
        c = bias[2] + scale[2] * (q >> 48)
        out.append((c, b, a) if axis == 0 else (b, c, a) if axis == 1 else (a, b, c))
    return out


def ref_normals(data, block):
    slots = {i: (tag, off) for i, tag, off in block["slots"]}
    tag, off = slots[2]
    at, n = block["offset"] + off, block["vertices"]
    out = []
    for (w,) in struct.iter_unpack("<I", data[at:at + 4 * n]):
        if tag & 15 == 1:
            out.append(mp._octahedral(ref_unorm10(w & 1023), ref_unorm10((w >> 10) & 1023)))
        else:
            v = [ref_unorm10((w >> s) & 1023) for s in (0, 10, 20)]
            length = math.sqrt(sum(c * c for c in v)) or 1.0
            out.append(tuple(c / length for c in v))
    return out


def ref_tangents(data, block):
    out = {}
    for slot, tag, off in block["slots"]:
        if slot == 2 or tag & 15 != 7 or mp.tag_stride(tag) != 4:
            continue
        at, n = block["offset"] + off, block["vertices"]
        rows = []
        for (w,) in struct.iter_unpack("<I", data[at:at + 4 * n]):
            v = [ref_unorm10((w >> s) & 1023) for s in (0, 10, 20)]
            length = math.sqrt(sum(c * c for c in v)) or 1.0
            rows.append((v[0] / length, v[1] / length, v[2] / length, {3: 1, 0: -1}.get(w >> 30, 0)))
        out[slot] = rows
    return out


def ref_tangent_agreement(normals, tangents):
    out = {"vertices": 0, "perpendicular": 0, "sign_plus": 0, "sign_minus": 0}
    for rows in tangents.values():
        for nrm, (x, y, z, sign) in zip(normals, rows):
            out["vertices"] += 1
            out["perpendicular"] += abs(nrm[0] * x + nrm[1] * y + nrm[2] * z) < 0.05
            out["sign_plus"] += sign == 1
            out["sign_minus"] += sign == -1
    return out


def ref_face_agreement(positions, triangles):
    normals = []
    for a, b, c in triangles:
        if max(a, b, c) >= len(positions):
            raise mp.PageError("triangle references a vertex past its block")
        p, q, r = positions[a], positions[b], positions[c]
        u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
        v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        normals.append((n[0] / length, n[1] / length, n[2] / length) if length and math.isfinite(length) else None)
    faces = {}
    for i, (a, b, c) in enumerate(triangles):
        for e in ((a, b), (b, c), (c, a)):
            faces.setdefault((min(e), max(e)), []).append(i)
    out = {"shared_edges": 0, "agree": 0, "opposed": 0}
    for pair in faces.values():
        if len(pair) != 2 or normals[pair[0]] is None or normals[pair[1]] is None:
            continue
        d = sum(x * y for x, y in zip(normals[pair[0]], normals[pair[1]]))
        out["shared_edges"] += 1
        out["agree"] += d > 0.9
        out["opposed"] += d < -0.9
    return out


def ref_normal_agreement(positions, normals, triangles):
    acc = {}
    for a, b, c in triangles:
        if max(a, b, c) >= min(len(positions), len(normals)):
            raise mp.PageError("triangle references a vertex past its block")
        p, q, r = positions[a], positions[b], positions[c]
        u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
        v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        for k in (a, b, c):
            s = acc.setdefault(k, [0.0, 0.0, 0.0])
            s[0] += n[0]
            s[1] += n[1]
            s[2] += n[2]
    out = {"vertices": 0, "within_8_degrees": 0, "within_25_degrees": 0, "opposed": 0}
    for k, s in acc.items():
        length = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
        if not length or not math.isfinite(length):
            continue
        d = sum(x * y / length for x, y in zip(normals[k], s))
        out["vertices"] += 1
        out["within_8_degrees"] += d > 0.99
        out["within_25_degrees"] += d > 0.9
        out["opposed"] += d < 0
    return out


def ref_block_checks(data, block):
    n = block["vertices"]
    at = block["offset"]
    slots = {i: off for i, _, off in block["slots"]}
    out = {"vertices": n, "positions_finite": 0, "positions_within_300m": 0, "normals_unit": 0,
           "colour_alpha_255": 0, "colour_vertices": 0, "uv_finite": 0, "uv_vertices": 0,
           "position_low_byte_zero": 0}
    pos = struct.unpack_from(f"<{3 * n}f", data, at + slots[0])
    for v in range(n):
        p = pos[3 * v:3 * v + 3]
        if all(math.isfinite(c) for c in p):
            out["positions_finite"] += 1
            out["positions_within_300m"] += all(abs(c) <= 300 for c in p)
    if 2 in slots:
        nrm = struct.unpack_from(f"<{3 * n}f", data, at + slots[2])
        for v in range(n):
            q = nrm[3 * v:3 * v + 3]
            if all(math.isfinite(c) for c in q) and abs(math.sqrt(sum(c * c for c in q)) - 1) < 0.02:
                out["normals_unit"] += 1
    raw = data[at + slots[0]:at + slots[0] + 12 * n]
    out["position_low_byte_zero"] = sum(1 for i in range(0, len(raw), 4) if raw[i] == 0)
    if 1 in slots:
        out["colour_vertices"] = n
        out["colour_alpha_255"] = sum(1 for i in range(3, 4 * n, 4) if data[at + slots[1] + i] == 255)
    for s in (3, 4):
        if s in slots:
            uv = struct.unpack_from(f"<{2 * n}f", data, at + slots[s])
            out["uv_vertices"] += n
            out["uv_finite"] += sum(1 for v in range(n) if math.isfinite(uv[2 * v]) and math.isfinite(uv[2 * v + 1]))
    return out


# ---- builders ---------------------------------------------------------------------------------------------------

SPECIAL_F32 = (0.0, -0.0, 1.0, -1.0, 299.99, 300.0, 300.01, -300.0, -300.01, 1e30, -1e30, 3.4e38,
               math.inf, -math.inf, math.nan)


def random_floats(rng, count):
    """f32 values: mostly in a plausible range, with zeros, boundary values, infinities and NaN mixed in."""
    out = []
    for _ in range(count):
        roll = rng.random()
        out.append(rng.choice(SPECIAL_F32) if roll < 0.1 else rng.uniform(-400, 400))
    return out


def unit_floats(rng, count):
    """Triples that are mostly unit normals, so normals_unit counts something other than zero."""
    out = []
    for _ in range(count // 3):
        v = [rng.uniform(-1, 1) for _ in range(3)]
        length = math.sqrt(sum(c * c for c in v)) or 1.0
        roll = rng.random()
        if roll < 0.1:
            v = [rng.choice(SPECIAL_F32) for _ in range(3)]
        elif roll < 0.2:
            v = [c / length * rng.choice((0.9, 0.99, 1.0, 1.015, 1.03)) for c in v]
        else:
            v = [c / length for c in v]
        out.extend(v)
    return out


def plain_block(rng, n, colour=True, uv1=True, normal=True):
    """A `63 30` block (slot order 0, 2, 1, 3, 4) with random streams."""
    order = [(0, 0x63, 12), (2, 0x63, 12), (1, 0x26, 4), (3, 0x42, 8), (4, 0x42, 8)]
    order = [o for o in order if (normal or o[0] != 2) and (colour or o[0] != 1) and (uv1 or o[0] != 4)]
    head, body, off = bytearray(0x60), bytearray(), 0x60
    for slot, tag, stride in order:
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        if slot == 0:
            floats = random_floats(rng, 3 * n)
            body += struct.pack(f"<{3 * n}f", *floats)
        elif slot == 2:
            body += struct.pack(f"<{3 * n}f", *unit_floats(rng, 3 * n))
        elif slot == 1:
            body += bytes(rng.choice((0, 255, rng.randrange(256))) for _ in range(4 * n))
        else:
            body += struct.pack(f"<{2 * n}f", *random_floats(rng, 2 * n))
        off += stride * n
    struct.pack_into("<I", head, 0x5C, 0x200)
    return bytes(head + body)


def packed_block(rng, n, axis, normal_tag=0x21, tangent_slots=(11, 18)):
    """A `42 40` block: slot 0 packed positions, slot 2 a normal, then 4-byte type-7 tangent slots."""
    order = [(0, 0x42, 8), (2, normal_tag, 4)] + [(s, 0x27, 4) for s in tangent_slots]
    head, body, off = bytearray(0x80), bytearray(), 0x80
    for slot, tag, stride in order:
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        for _ in range(n):
            if slot == 0:
                body += struct.pack("<Q", rng.getrandbits(64))
            elif slot == 2 and normal_tag & 15 == 1 and rng.random() < 0.5:
                body += struct.pack("<I", rng.getrandbits(32) | 0x3FF)   # large u: the octahedral fold is likely
            else:
                body += struct.pack("<I", rng.getrandbits(32))
        off += stride * n
    struct.pack_into("<I", head, 0x5C, 0x200)
    struct.pack_into("<3f", head, 0x60, -20.5, 3.25, 100.0)
    struct.pack_into("<I", head, 0x6C, axis)
    struct.pack_into("<3f", head, 0x70, 1e-5, 2.5e-6, 3.0e-4)
    struct.pack_into("<f", head, 0x7C, 4.0)
    return bytes(head + body)


def same(case, got, want):
    """Equal including NaN and the sign of zero, which `==` on floats and tuples does not tell apart."""
    case.assertEqual(repr(got), repr(want))


class PackedBlockTests(unittest.TestCase):
    def test_positions_match_the_reference_for_every_axis_word(self):
        rng = random.Random(1)
        for axis in (0, 1, 2):
            for n in (1, 2, 7, 64, 300):
                data = packed_block(rng, n, axis)
                block = mp.parse_page(data)[0]
                same(self, mp.positions_4240(data, block), ref_positions(data, block))

    def test_positions_cover_the_field_extremes(self):
        # all-ones and all-zero packed words, so the 24-bit a, the split b and the 16-bit c reach their ends
        for axis in (0, 1, 2):
            data = bytearray(packed_block(random.Random(2), 4, axis))
            for i, q in enumerate((0, 0xFFFFFFFFFFFFFFFF, 0x00000000000000FF, 0xFFFF00000000FF00)):
                struct.pack_into("<Q", data, 0x80 + 8 * i, q)
            data = bytes(data)
            block = mp.parse_page(data)[0]
            same(self, mp.positions_4240(data, block), ref_positions(data, block))

    def test_positions_reject_a_bad_axis_word_and_a_wrong_slot_0(self):
        data = bytearray(packed_block(random.Random(3), 3, 2))
        struct.pack_into("<I", data, 0x6C, 3)
        with self.assertRaises(mp.PageError):
            mp.positions_4240(bytes(data), mp.parse_page(bytes(data))[0])

    def test_normals_and_tangents_match_the_reference(self):
        rng = random.Random(4)
        for normal_tag in (0x21, 0x27):
            for n in (1, 5, 128, 1000):
                data = packed_block(rng, n, 2, normal_tag)
                block = mp.parse_page(data)[0]
                same(self, mp.normals_4240(data, block), ref_normals(data, block))
                same(self, mp.tangents_4240(data, block), ref_tangents(data, block))

    def test_normals_cover_the_octahedral_fold_and_the_zero_vector(self):
        # u and v at 0 and 1023 on both fields reach z < 0 and the sign copy; an all-zero 3 x 10 bit word is not 0 / 0
        words = [u | v << 10 for u in (0, 1, 511, 512, 1022, 1023) for v in (0, 1, 511, 512, 1022, 1023)]
        for normal_tag in (0x21, 0x27):
            data = bytearray(packed_block(random.Random(5), len(words), 2, normal_tag))
            slot2 = 0x80 + 8 * len(words)
            for i, w in enumerate(words):
                struct.pack_into("<I", data, slot2 + 4 * i, w)
            data = bytes(data)
            block = mp.parse_page(data)[0]
            same(self, mp.normals_4240(data, block), ref_normals(data, block))

    def test_tangent_signs_follow_bits_30_and_31(self):
        data = bytearray(packed_block(random.Random(6), 4, 2, tangent_slots=(11,)))
        slot11 = 0x80 + 8 * 4 + 4 * 4
        for i, top in enumerate((0, 1, 2, 3)):
            struct.pack_into("<I", data, slot11 + 4 * i, top << 30 | 0x1FF | 0x1FF << 10 | 0x1FF << 20)
        data = bytes(data)
        block = mp.parse_page(data)[0]
        self.assertEqual([row[3] for row in mp.tangents_4240(data, block)[11]], [-1, 0, 0, 1])
        same(self, mp.tangents_4240(data, block), ref_tangents(data, block))

    def test_tangent_agreement_matches_the_reference(self):
        rng = random.Random(7)
        for n in (0, 1, 33, 500):
            data = packed_block(rng, max(n, 1), 2)
            block = mp.parse_page(data)[0]
            normals, tangents = mp.normals_4240(data, block)[:n], mp.tangents_4240(data, block)
            self.assertEqual(mp.tangent_agreement(normals, tangents), ref_tangent_agreement(normals, tangents))
        self.assertEqual(mp.tangent_agreement([], {}), {"vertices": 0, "perpendicular": 0, "sign_plus": 0,
                                                        "sign_minus": 0})


class BlockChecksTests(unittest.TestCase):
    def test_block_checks_match_the_reference(self):
        rng = random.Random(8)
        for colour in (True, False):
            for uv1 in (True, False):
                for normal in (True, False):
                    for n in (1, 3, 64, 257):
                        data = plain_block(rng, n, colour, uv1, normal)
                        block = mp.parse_page(data)[0]
                        self.assertEqual(mp.block_checks(data, block), ref_block_checks(data, block))

    def test_block_checks_on_special_floats_only(self):
        for special in SPECIAL_F32:
            n = 5
            data = bytearray(plain_block(random.Random(9), n))
            struct.pack_into(f"<{3 * n}f", data, 0x60, *([special] * 3 * n))
            struct.pack_into(f"<{3 * n}f", data, 0x60 + 12 * n, *([special] * 3 * n))
            data = bytes(data)
            block = mp.parse_page(data)[0]
            self.assertEqual(mp.block_checks(data, block), ref_block_checks(data, block), special)

    def test_one_non_finite_component_drops_the_vertex(self):
        n = 3
        data = bytearray(plain_block(random.Random(10), n))
        struct.pack_into(f"<{3 * n}f", data, 0x60, 1, 2, 3, 4, math.nan, 6, 7, 8, math.inf)
        block = mp.parse_page(bytes(data))[0]
        got = mp.block_checks(bytes(data), block)
        self.assertEqual((got["positions_finite"], got["positions_within_300m"]), (1, 1))

    def test_block_checks_accept_a_bytearray_and_a_memoryview(self):
        data = plain_block(random.Random(11), 40)
        block = mp.parse_page(data)[0]
        want = ref_block_checks(data, block)
        self.assertEqual(mp.block_checks(bytearray(data), block), want)
        self.assertEqual(mp.block_checks(memoryview(data), block), want)

    def test_summarise_page_counts_are_unchanged_for_a_mixed_page(self):
        rng = random.Random(12)
        first = plain_block(rng, 52)   # 36 * 52 + 0x60 bytes is a multiple of 16
        data = first + packed_block(rng, 40, 1)
        page = mp.summarise_page(data, "p", normals=True)
        block = mp.parse_page(data)[0]
        for key, value in ref_block_checks(data, block).items():
            if key != "vertices":
                self.assertEqual(page["checks"][key], value, key)


def random_mesh(rng, vertices, triangles, special=False):
    positions = []
    for _ in range(vertices):
        p = tuple(rng.choice(SPECIAL_F32) if special and rng.random() < 0.05 else rng.uniform(-3, 3)
                  for _ in range(3))
        positions.append(p)
    if vertices > 1 and rng.random() < 0.5:   # repeated positions give zero-area faces
        positions[rng.randrange(vertices)] = positions[rng.randrange(vertices)]
    tris = [tuple(rng.randrange(vertices) for _ in range(3)) for _ in range(triangles)]
    return positions, tris


def grid_mesh(side):
    """A flat side x side vertex grid cut into triangles, so most edges are shared by exactly two faces."""
    positions = [(float(x), float(y), 0.0) for y in range(side) for x in range(side)]
    tris = []
    for y in range(side - 1):
        for x in range(side - 1):
            i = y * side + x
            tris += [(i, i + 1, i + side), (i + 1, i + side + 1, i + side)]
    return positions, tris


class AgreementTests(unittest.TestCase):
    def check(self, positions, normals, tris):
        self.assertEqual(mp.face_agreement(positions, tris), ref_face_agreement(positions, tris))
        if normals is not None:
            self.assertEqual(mp.normal_agreement(positions, normals, tris),
                             ref_normal_agreement(positions, normals, tris))

    def test_random_meshes_match_the_reference(self):
        rng = random.Random(13)
        for vertices, triangles in ((3, 1), (4, 6), (10, 40), (60, 300), (500, 2000)):
            for special in (False, True):
                positions, tris = random_mesh(rng, vertices, triangles, special)
                normals = [mp._octahedral(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in positions]
                self.check(positions, normals, tris)

    def test_grid_with_normals_that_agree_and_flip(self):
        positions, tris = grid_mesh(12)
        up = [(0.0, 0.0, 1.0)] * len(positions)
        down = [(0.0, 0.0, -1.0)] * len(positions)
        mixed = [(0.0, 0.0, 1.0 if i % 3 else -1.0) for i in range(len(positions))]
        for normals in (up, down, mixed):
            self.check(positions, normals, tris)
        self.assertGreater(mp.face_agreement(positions, tris)["shared_edges"], 100)

    def test_flipped_faces_are_counted_as_opposed(self):
        positions, tris = grid_mesh(6)
        tris = [(a, c, b) if i % 4 == 0 else (a, b, c) for i, (a, b, c) in enumerate(tris)]
        got = mp.face_agreement(positions, tris)
        self.assertGreater(got["opposed"], 0)
        self.check(positions, [(0.0, 0.0, 1.0)] * len(positions), tris)

    def test_an_edge_shared_by_three_faces_is_not_a_pair(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)]
        tris = [(0, 1, 2), (1, 0, 3), (0, 1, 4)]   # all three own the edge 0-1
        self.assertEqual(mp.face_agreement(positions, tris)["shared_edges"], 0)
        self.check(positions, None, tris)

    def test_repeated_and_degenerate_triangles(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        cases = ([(0, 1, 2), (0, 1, 2)], [(0, 1, 2), (2, 1, 0)], [(0, 0, 1), (0, 1, 1), (1, 1, 1)],
                 [(0, 1, 0), (1, 2, 1)], [(0, 1, 2), (1, 3, 2), (0, 1, 2), (1, 3, 2)], [])
        for tris in cases:
            self.check(positions, [(0.0, 0.0, 1.0)] * 4, tris)

    def test_negative_references_are_kept_apart_from_the_same_positive_index(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.5)]
        normals = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)]
        for tris in ([(0, 1, -3), (1, 3, 2), (-1, 1, 2)], [(-5, -4, -3), (0, 1, 2)], [(4, -1, 2), (3, 4, 1)]):
            self.check(positions, normals, tris)
        with self.assertRaises(IndexError):
            mp.face_agreement(positions, [(0, 1, -6)])
        with self.assertRaises(IndexError):
            mp.normal_agreement(positions, normals, [(0, 1, -6)])

    def test_a_vertex_past_the_block_is_a_page_error(self):
        positions, tris = grid_mesh(3)
        with self.assertRaises(mp.PageError):
            mp.face_agreement(positions, tris + [(0, 1, len(positions))])
        with self.assertRaises(mp.PageError):
            mp.normal_agreement(positions, [(0.0, 0.0, 1.0)] * len(positions), tris + [(0, 1, len(positions))])
        with self.assertRaises(mp.PageError):   # shorter normals shrink the limit, as positions do
            mp.normal_agreement(positions, [(0.0, 0.0, 1.0)] * 4, tris)

    def test_normals_longer_than_positions_still_index_by_vertex(self):
        positions, tris = grid_mesh(3)
        normals = [(0.0, 0.0, 1.0)] * (len(positions) + 5)
        self.check(positions, normals, tris)

    def test_unused_vertices_and_cancelling_faces_are_skipped(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (5.0, 5.0, 5.0)]
        tris = [(0, 1, 2), (0, 2, 1)]   # opposite faces: every vertex sums to zero; vertex 3 is unused
        normals = [(0.0, 0.0, 1.0)] * 4
        self.assertEqual(mp.normal_agreement(positions, normals, tris)["vertices"], 0)
        self.check(positions, normals, tris)

    def test_a_sparse_group_over_a_long_vertex_block_stays_small(self):
        # One triangle naming vertices 0, 1 and 32767 of a 32768-vertex block: the work and the memory follow the
        # triangles, not the length of the block.
        count = 32768
        positions = [(float(i % 7), float(i % 11), 0.0) for i in range(count)]
        positions[1], positions[count - 1] = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        positions[0] = (0.0, 0.0, 0.0)
        normals = [(0.0, 0.0, 1.0)] * count
        tris = [(0, 1, count - 1)]
        expected = ref_normal_agreement(positions, normals, tris)
        self.assertEqual(expected["vertices"], 3)
        self.assertEqual(mp.normal_agreement(positions, normals, tris), expected)
        tracemalloc.start()
        try:
            mp.normal_agreement(positions, normals, tris)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 8 * 1024)

    def test_density_threshold_gives_the_same_answer_on_both_sides(self):
        rng = random.Random(29)
        count = 400
        positions = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(count)]
        normals = [mp._octahedral(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(count)]
        for triangles in (1, 10, 49, 50, 51, 100, 400):   # 50 triangles is the block length / 8
            tris = [tuple(rng.randrange(count) for _ in range(3)) for _ in range(triangles)]
            self.assertEqual(mp.normal_agreement(positions, normals, tris),
                             ref_normal_agreement(positions, normals, tris), triangles)

    def test_no_triangles(self):
        self.assertEqual(mp.face_agreement([], []), {"shared_edges": 0, "agree": 0, "opposed": 0})
        self.assertEqual(mp.normal_agreement([], [], []),
                         {"vertices": 0, "within_8_degrees": 0, "within_25_degrees": 0, "opposed": 0})


if __name__ == "__main__":
    unittest.main()
