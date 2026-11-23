"""The fast paths in restofworld_model_join.py must accept and reject exactly what the plain code did.

    python -I tests/test_restofworld_model_join_fast.py

Every byte here is built in memory. Each test compares the tool with a reference written out in this
file (the field readers and the per-word triangle loop as they were before the fast paths).
"""
import os
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import restofworld_model_join as joiner  # noqa: E402

JoinError = joiner.JoinError


def ref_read(fmt, size, name, data, at):
    if at < 0 or at + size > len(data):
        raise JoinError(f"truncated {name} at +0x{at:x}")
    return struct.unpack_from(fmt, data, at)[0]


def ref_triangles(gfx_bytes, at, c_tris, c_idx, c_verts):
    for t_word in struct.unpack_from(f"<{c_tris}I", gfx_bytes, at):
        if (t_word >> 21) != c_idx:
            raise JoinError("packed triangle cluster id mismatch")
        if (
            (t_word & 0x7F) >= c_verts
            or ((t_word >> 7) & 0x7F) >= c_verts
            or ((t_word >> 14) & 0x7F) >= c_verts
        ):
            raise JoinError("packed triangle local index out of range")


def ref_validate_meshlet_buffer(gfx_bytes, ib_off, ib_bytes, desc_clusters, desc_tris, desc_refs, desc_binding):
    """The body as it was before the fast paths: per-record u32 reads and the per-word triangle loop."""

    def u32(data, at):
        return ref_read("<I", 4, "u32", data, at)

    if ib_off < 0 or ib_off + ib_bytes > len(gfx_bytes) or ib_bytes < 32:
        raise JoinError("index buffer outside graphics segment")
    count_flags = u32(gfx_bytes, ib_off)
    bone_palette = (count_flags & 0x40000000) != 0
    clusters = count_flags & (0xFFFF if bone_palette else 0x7FFFFFFF)
    tris = u32(gfx_bytes, ib_off + 4)
    refs = u32(gfx_bytes, ib_off + 8)
    palette_no_marker = (desc_binding & 0xFF) == 0 and (count_flags & 0xFFFF0000) == 0xC0000000
    if not (
        0 < clusters <= 2048
        and 0 < tris <= 65535
        and 0 < refs <= 65535
        and clusters == desc_clusters
        and tris == desc_tris
        and refs == desc_refs
        and (((desc_binding & 0x80000000) != 0) == bone_palette)
        and (not bone_palette or (desc_binding & 0xFF) == 0x80 or palette_no_marker)
    ):
        raise JoinError("meshlet group header disagrees with system descriptor")
    ref_size = 2 if (count_flags & 0x80000000) else 4
    tri_at = ib_off + 12 + clusters * 16
    refs_at = tri_at + tris * 4
    pal_at = refs_at + refs * ref_size
    end_at = pal_at + (clusters * 2 if bone_palette else 0)
    if end_at > ib_off + ib_bytes:
        raise JoinError("meshlet payload exceeds index buffer allocation")
    acc_refs = 0
    acc_tris = 0
    for c_idx in range(clusters):
        rec_at = ib_off + 12 + c_idx * 16
        packed = u32(gfx_bytes, rec_at)
        offsets = u32(gfx_bytes, rec_at + 4)
        c_verts = (packed & 0x7F) + 1
        c_tris = ((packed >> 7) & 0xFF) + 1
        v_start = offsets & 0xFFFF
        t_start = offsets >> 16
        if v_start != acc_refs or t_start != acc_tris:
            raise JoinError("meshlet running offsets do not match accumulated counts")
        if c_verts > refs - v_start or c_tris > tris - t_start:
            raise JoinError("meshlet counts exceed group totals")
        ref_triangles(gfx_bytes, tri_at + t_start * 4, c_tris, c_idx, c_verts)
        acc_refs += c_verts
        acc_tris += c_tris
    if acc_refs != refs or acc_tris != tris:
        raise JoinError("meshlet sums do not match group header")
    if ref_size == 2:
        max_ref = max(struct.unpack_from(f"<{refs}H", gfx_bytes, refs_at))
    else:
        max_ref = max(struct.unpack_from(f"<{refs}I", gfx_bytes, refs_at))
    return max_ref, bone_palette


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except JoinError as exc:
        return ("error", str(exc))


def build_group(spec, bone_palette=False):
    """spec is [(c_verts, [(a, b, c), ...]), ...]; returns (bytes, clusters, tris, refs, binding)."""
    clusters = len(spec)
    total_tris = sum(len(t) for _, t in spec)
    total_refs = sum(v for v, _ in spec)
    flags = 0x80000000 | clusters | (0x40000000 if bone_palette else 0)
    tri_at = 12 + clusters * 16
    refs_at = tri_at + total_tris * 4
    pal_at = refs_at + total_refs * 2
    size = max(32, (pal_at + (clusters * 2 if bone_palette else 0) + 3) & ~3)
    buf = bytearray(size)
    struct.pack_into("<III", buf, 0, flags, total_tris, total_refs)
    v_acc = t_acc = 0
    for c_idx, (c_verts, tris) in enumerate(spec):
        packed = (c_verts - 1) | ((len(tris) - 1) << 7)
        struct.pack_into("<II", buf, 12 + c_idx * 16, packed, v_acc | (t_acc << 16))
        for ti, (i0, i1, i2) in enumerate(tris):
            word = (i0 & 0x7F) | ((i1 & 0x7F) << 7) | ((i2 & 0x7F) << 14) | (c_idx << 21)
            struct.pack_into("<I", buf, tri_at + (t_acc + ti) * 4, word)
        for vi in range(c_verts):
            struct.pack_into("<H", buf, refs_at + (v_acc + vi) * 2, v_acc + vi)
        v_acc += c_verts
        t_acc += len(tris)
    binding = (0x80000000 | (total_refs << 8) | 0x80) if bone_palette else (total_refs << 8)
    return bytes(buf), clusters, total_tris, total_refs, binding


class FieldReaderTests(unittest.TestCase):
    def test_readers_match_reference_for_every_offset_near_both_ends(self):
        data = bytes(range(1, 17))
        readers = (
            (joiner._u16, "<H", 2, "u16"),
            (joiner._u32, "<I", 4, "u32"),
            (joiner._u64, "<Q", 8, "u64"),
        )
        for fn, fmt, size, name in readers:
            for buf in (data, bytearray(data), memoryview(data), b""):
                for at in range(-3, len(buf) + 4):
                    with self.subTest(reader=name, at=at, kind=type(buf).__name__):
                        self.assertEqual(
                            outcome(fn, buf, at), outcome(ref_read, fmt, size, name, buf, at)
                        )


class TriangleCheckTests(unittest.TestCase):
    @staticmethod
    def _words(rng, count, c_idx, c_verts, bad):
        words = []
        for _ in range(count):
            idx = [rng.randrange(c_verts) for _ in range(3)]
            words.append(idx[0] | (idx[1] << 7) | (idx[2] << 14) | (c_idx << 21))
        if bad:
            at = rng.randrange(count)
            kind = rng.randrange(4)
            if kind == 3:
                words[at] = (words[at] & 0x1FFFFF) | (((c_idx + 1) % 2048) << 21)
            elif c_verts < 128:
                shift = kind * 7
                words[at] = (words[at] & ~(0x7F << shift)) | (rng.randrange(c_verts, 128) << shift)
        return words

    def test_fast_check_agrees_with_per_word_loop_on_random_clusters(self):
        rng = random.Random(1234)
        for _ in range(1500):
            c_verts = rng.choice((1, 2, 3, 31, 64, 126, 127, 128, rng.randrange(1, 129)))
            c_tris = rng.choice((1, 2, 3, 100, 255, 256, rng.randrange(1, 257)))
            c_idx = rng.choice((0, 1, 2, 1023, 2046, 2047, rng.randrange(2048)))
            pad = rng.randrange(0, 8)
            words = self._words(rng, c_tris, c_idx, c_verts, bad=rng.random() < 0.5)
            blob = bytes(rng.randrange(256) for _ in range(pad)) + struct.pack(f"<{c_tris}I", *words) + b"\xff" * 8
            expected = outcome(ref_triangles, blob, pad, c_tris, c_idx, c_verts)
            fast = joiner._triangles_in_range(blob, pad, c_tris, c_idx, c_verts)
            if expected[0] == "ok":
                self.assertTrue(fast, (c_verts, c_tris, c_idx))
            else:
                # False only routes to the slow check, which has to name the same error.
                self.assertFalse(fast, (c_verts, c_tris, c_idx))
                self.assertEqual(
                    outcome(joiner._check_triangles_slow, blob, pad, c_tris, c_idx, c_verts), expected
                )

    def test_every_index_value_against_every_cluster_size(self):
        for c_verts in range(1, 129):
            for index in range(128):
                for slot in range(3):
                    # The slots not under test hold 0, which is in range for every cluster size.
                    blob = struct.pack("<I", (index << (7 * slot)) | (5 << 21))
                    expected = outcome(ref_triangles, blob, 0, 1, 5, c_verts)
                    self.assertEqual(
                        joiner._triangles_in_range(blob, 0, 1, 5, c_verts),
                        expected[0] == "ok",
                        (c_verts, index, slot),
                    )


class MeshletBufferTests(unittest.TestCase):
    def _compare(self, blob, clusters, tris, refs, binding, ib_off=0, ib_bytes=None):
        ib_bytes = len(blob) - ib_off if ib_bytes is None else ib_bytes
        args = (blob, ib_off, ib_bytes, clusters, tris, refs, binding)
        self.assertEqual(
            outcome(joiner._validate_meshlet_buffer, *args), outcome(ref_validate_meshlet_buffer, *args)
        )

    def test_valid_groups_match_reference(self):
        rng = random.Random(99)
        for bone in (False, True):
            for _ in range(60):
                spec = []
                for _c in range(rng.randrange(1, 9)):
                    c_verts = rng.randrange(1, 129)
                    c_tris = rng.randrange(1, 40)
                    spec.append(
                        (c_verts, [tuple(rng.randrange(c_verts) for _ in range(3)) for _ in range(c_tris)])
                    )
                blob, clusters, tris, refs, binding = build_group(spec, bone)
                got = outcome(joiner._validate_meshlet_buffer, blob, 0, len(blob), clusters, tris, refs, binding)
                self.assertEqual(got[0], "ok")
                self._compare(blob, clusters, tris, refs, binding)
                # The same group placed after a prefix inside a larger segment.
                self._compare(
                    b"\xaa" * 20 + blob + b"\xbb" * 9, clusters, tris, refs, binding, ib_off=20, ib_bytes=len(blob)
                )

    def test_corrupted_groups_match_reference(self):
        rng = random.Random(4242)
        spec = [(5, [(0, 1, 2), (2, 3, 4)]), (9, [(8, 7, 6), (0, 4, 8), (1, 1, 1)]), (3, [(0, 1, 2)])]
        blob, clusters, tris, refs, binding = build_group(spec)
        for _ in range(600):
            mutated = bytearray(blob)
            for _m in range(rng.randrange(1, 4)):
                mutated[rng.randrange(len(mutated))] = rng.randrange(256)
            self._compare(bytes(mutated), clusters, tris, refs, binding)
        for cut in range(len(blob) + 1):
            self._compare(blob[:cut], clusters, tris, refs, binding)


if __name__ == "__main__":
    unittest.main()
