"""Parity tests for the precompiled-struct fast path in `tools/ped_catalogues.py`.

`parse_catalogue_blocks`, `read_tag_map` and `read_cull_drawable` decode records with shared `struct.Struct`
objects and skip calls whose count or pointer is zero.
These tests pin the exact output (records, warnings, stats, resolved bindings) of well-formed and damaged
synthetic catalogues, so a later speed-up cannot change what the parser reports. The golden digests were
produced by the parser as it was before the fast path. Everything is synthesized in memory: no game files, no
retail names or hashes.

    python -I tests/test_ped_catalogues_fastpath.py
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ped_catalogues as pc  # noqa: E402

MASK64 = (1 << 64) - 1


class Rng:
    """splitmix64: a fixed, dependency-free stream, so the golden digests do not depend on the Python version."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n


def ptr(block: int, offset: int) -> int:
    return (offset << 20) | (block & 0xFFFFF)


def pack_root(count: int, bucket_ptr: int) -> bytes:
    buf = bytearray(pc.CATALOG_ROOT_SIZE)
    struct.pack_into("<IIQ", buf, 0x08, count, count, bucket_ptr)
    return bytes(buf)


def pack_preset(pid, parent, state_count, state_ptr, category, tag_ptr, tag_count) -> bytes:
    buf = bytearray(pc.PRESET_STRIDE)
    struct.pack_into("<IIIIQI", buf, 0x00, pid, parent, state_count, state_count, state_ptr, category)
    struct.pack_into("<QQHH", buf, 0x28, 1 if tag_count else 0, tag_ptr, tag_count, tag_count)
    return bytes(buf)


def pack_state(state_id, flags, h0, alb, nrm, extra, ext_ptr, ext_count, tag_ptr, tag_count) -> bytes:
    buf = bytearray(pc.STATE_STRIDE)
    struct.pack_into("<II", buf, 0x00, state_id, flags)
    struct.pack_into("<QQQQ", buf, 0x08, h0, alb, nrm, extra)
    struct.pack_into("<QQHH", buf, 0x28, 1 if ext_count else 0, ext_ptr, ext_count, ext_count)
    struct.pack_into("<QQHH", buf, 0xA0, 1 if tag_count else 0, tag_ptr, tag_count, tag_count)
    return bytes(buf)


# Block numbers of `build_blocks`: 1 root, 2 presets, 3 states, 4 tag slots (0x101), 5 `0x4a` u32 values,
# 6 extension slots (0x103), 7 `0x3d780b74` cull records.
def build_blocks(n_presets: int, seed: int, states_per: int = 3) -> list[tuple[int, bytes]]:
    rng = Rng(seed)
    tag_vals = bytearray()
    tag_slots = bytearray()
    ext = bytearray()
    cull = bytearray()
    states = bytearray()
    presets = bytearray()

    def tag_map(entries: list[tuple[int, list[int]]]) -> tuple[int, int]:
        slot_off = len(tag_slots)
        for key, vals in entries:
            val_off = len(tag_vals)
            for v in vals:
                tag_vals.extend(struct.pack("<I", v))
            tag_slots.extend(struct.pack("<IxxxxIIQ", key, len(vals), len(vals), ptr(5, val_off) if vals else 0))
        return ptr(4, slot_off), len(entries)

    def ext_map() -> tuple[int, int]:
        ext_off, cull_off = len(ext), len(cull)
        cull.extend(struct.pack("<QQ", rng.next() & 0xFF, rng.next()))
        ext.extend(struct.pack("<IxxxxQ", 0x1A271DE4, 0))
        ext.extend(struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, ptr(7, cull_off)))
        return ptr(6, ext_off), 2

    for p in range(n_presets):
        parent = 0x1000 + rng.below(p) if p and rng.below(2) else 0
        count = states_per if rng.below(10) else 0
        state_ptr = ptr(3, len(states)) if count else 0
        for _ in range(count):
            tp = tc = ep = ec = 0
            if rng.below(2):
                tp, tc = tag_map(
                    [(0x4BFD27DD, [rng.next() & 0xFFFFFFFF]), (rng.next() & 0xFFFFFFFF, [1, 2, 3][: 1 + rng.below(3)])]
                )
            if rng.below(5) < 3:
                ep, ec = ext_map()
            states.extend(
                pack_state(
                    rng.next() & 0xFFFFFFFF,
                    rng.below(256),
                    rng.next() if rng.below(5) else 0,
                    rng.next() if rng.below(3) else 0,
                    rng.next() if rng.below(3) else 0,
                    rng.next() if rng.below(5) == 0 else 0,
                    ep,
                    ec,
                    tp,
                    tc,
                )
            )
        tp = tc = 0
        if rng.below(10) < 7:
            tp, tc = tag_map([(0x4BFD27DD if rng.below(10) < 6 else 0x1C114E61, [rng.next() & 0xFFFFFFFF])])
        presets.extend(pack_preset(0x1000 + p, parent, count, state_ptr, (pc.HEAD_CATEGORY, 0x3817131D, 5)[rng.below(3)], tp, tc))

    return [
        (pc.CATALOG_ROOT_HASH, pack_root(n_presets, ptr(2, 0))),
        (pc.PRESET_STRUCT_HASH, bytes(presets)),
        (pc.STATE_STRUCT_HASH, bytes(states)),
        (0x101, bytes(tag_slots)),
        (pc.TAG_VALUE_BLOCK_TYPE, bytes(tag_vals)),
        (0x103, bytes(ext)),
        (pc.CULL_DRAWABLE_STRUCT_HASH, bytes(cull)),
    ]


def as_blocks(blocks: list[tuple[int, bytes]]) -> dict[int, tuple[int, memoryview]]:
    return {i + 1: (key, memoryview(payload)) for i, (key, payload) in enumerate(blocks)}


def damage(blocks: list[tuple[int, bytes]], seed: int, flips: int) -> list[tuple[int, bytes]]:
    """Overwrite `flips` 8-byte words of non-root blocks with pointers and counts that point nowhere useful."""
    rng = Rng(seed)
    out = [[key, bytearray(payload)] for key, payload in blocks]
    for _ in range(flips):
        entry = out[1 + rng.below(len(out) - 1)]
        if len(entry[1]) < 16:
            continue
        off = rng.below(len(entry[1]) - 8)
        entry[1][off : off + 8] = struct.pack("<Q", ptr(rng.below(9), rng.below(1 << 12) * 4))
    return [(key, bytes(payload)) for key, payload in out]


def digest(cat: pc.CatalogueContainer) -> str:
    bindings = []
    for kwargs in (
        {},
        {"drawable_names": ["alpha", "beta", "gamma"]},
        {"cull_only": True},
        {"require_textures": True},
        {"filter_hashes_32": {0x4BFD27DD}, "filter_hashes_64": {1, 2, 3}},
    ):
        bindings.append([b.to_dict() for b in pc.resolve_preset_bindings(cat, **kwargs)])
    doc = {
        "summary": cat.summary_dict(),
        "presets": [p.to_dict() for p in cat.presets],
        "by_id": sorted(cat.presets_by_id),
        "cull_map": [[h, v] for h, v in pc.resolve_cull_map(cat).items()],
        "bindings": bindings,
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


class FastPathParityTests(unittest.TestCase):
    def parse(self, blocks):
        return pc.parse_catalogue_blocks(as_blocks(blocks), 1, framing="0drp", source="synthetic")

    def test_well_formed_catalogue_digest(self):
        cat = self.parse(build_blocks(60, seed=11))
        self.assertEqual(cat.stats["presets"], 60)
        self.assertEqual(cat.warnings, ())
        self.assertEqual(digest(cat), GOLDEN_CLEAN)

    def test_damaged_catalogue_digest(self):
        cat = self.parse(damage(build_blocks(60, seed=11), seed=5, flips=40))
        self.assertGreater(len(cat.warnings), 5)
        self.assertEqual(digest(cat), GOLDEN_DAMAGED)

    def test_record_equality_with_plain_constructor(self):
        cat = self.parse(build_blocks(8, seed=3))
        state = next(s for p in cat.presets for s in p.states if s.cull_ref is not None and s.tags)
        rebuilt = pc.StateRecord(**{k: getattr(state, k) for k in state.__dataclass_fields__})
        self.assertEqual(state, rebuilt)
        self.assertEqual(hash(state.cull_ref), hash(rebuilt.cull_ref))
        with self.assertRaises(Exception):
            state.flags = 1  # still frozen

    def test_struct_layouts_match_the_documented_offsets(self):
        self.assertEqual(pc._PRESET_FIELDS.size, 0x3A)  # up to and including the u16 tag count at +0x38
        self.assertEqual(pc._STATE_FIELDS.size, 0x3C)  # up to and including the u16 ext capacity at +0x3a
        self.assertEqual(pc._TAG_SLOT.size, pc.TAG_MAP_SLOT_SIZE)
        self.assertEqual(pc._EXT_SLOT.size, pc.EXT_MAP_SLOT_SIZE)
        self.assertEqual(pc.EXT_MAP_OFFSET + 0x14, pc._STATE_FIELDS.size)


class TagMapEdgeCases(unittest.TestCase):
    @staticmethod
    def blocks(slots: bytes, values: bytes = b"", slot_type: int = 0x101, value_type: int = 0x4A):
        return {
            1: (slot_type, memoryview(slots)),
            2: (value_type, memoryview(values)),
        }

    def test_zero_count_or_pointer_reads_nothing_and_warns_nothing(self):
        warnings: list[str] = []
        self.assertEqual(pc.read_tag_map({}, 0, 5, warnings=warnings), {})
        self.assertEqual(pc.read_tag_map({}, ptr(1, 0), 0, warnings=warnings), {})
        self.assertEqual(warnings, [])

    def test_truncated_slot_array_reads_the_whole_slots_that_fit(self):
        slots = struct.pack("<IxxxxIIQ", 7, 2, 2, ptr(2, 0)) + struct.pack("<IxxxxIIQ", 8, 0, 0, 0) + b"\xee" * 10
        warnings: list[str] = []
        tags = pc.read_tag_map(self.blocks(slots, struct.pack("<II", 5, 6)), ptr(1, 0), 4, warnings=warnings)
        self.assertEqual(tags, {7: (5, 6), 8: ()})
        self.assertEqual(warnings, ["tag map truncated at block 1+0x0 (count=4)"])

    def test_slot_offset_past_the_block_end_reads_nothing(self):
        warnings: list[str] = []
        tags = pc.read_tag_map(self.blocks(b"\0" * 24), ptr(1, 4096), 2, warnings=warnings)
        self.assertEqual(tags, {})
        self.assertEqual(warnings, ["tag map truncated at block 1+0x1000 (count=2)"])

    def test_truncated_value_array_keeps_the_values_that_fit(self):
        slots = struct.pack("<IxxxxIIQ", 9, 4, 4, ptr(2, 4))
        warnings: list[str] = []
        tags = pc.read_tag_map(
            self.blocks(slots, struct.pack("<III", 1, 2, 3)), ptr(1, 0), 1, warnings=warnings
        )
        self.assertEqual(tags, {9: (2, 3)})
        self.assertEqual(warnings, ["tag key 0x00000009 value array truncated in block 2+0x4"])

    def test_value_offset_past_the_block_end_gives_an_empty_tuple(self):
        slots = struct.pack("<IxxxxIIQ", 9, 2, 2, ptr(2, 64))
        warnings: list[str] = []
        tags = pc.read_tag_map(self.blocks(slots, b"\0" * 8), ptr(1, 0), 1, warnings=warnings)
        self.assertEqual(tags, {9: ()})
        self.assertEqual(warnings, ["tag key 0x00000009 value array truncated in block 2+0x40"])

    def test_value_count_cap_is_inclusive(self):
        values = struct.pack("<257I", *range(257))
        at_cap = struct.pack("<IxxxxIIQ", 1, 256, 256, ptr(2, 0))
        over_cap = struct.pack("<IxxxxIIQ", 2, 257, 257, ptr(2, 0))
        warnings: list[str] = []
        tags = pc.read_tag_map(self.blocks(at_cap + over_cap, values), ptr(1, 0), 2, warnings=warnings)
        self.assertEqual(tags, {1: tuple(range(256))})
        self.assertEqual(warnings, ["tag key 0x00000002 value count 257 exceeds cap 256"])
        self.assertEqual(len(pc._U32_ARRAYS), pc.MAX_TAG_VALUES + 1)  # the Struct cache cannot grow past the cap

    def test_missing_wrong_type_and_non_synthetic_blocks_warn(self):
        slots = struct.pack("<IxxxxIIQ", 3, 1, 1, ptr(9, 0)) + struct.pack("<IxxxxIIQ", 4, 1, 1, ptr(2, 0))
        warnings: list[str] = []
        tags = pc.read_tag_map(self.blocks(slots, b"\0" * 4, value_type=0x5B), ptr(1, 0), 2, warnings=warnings)
        self.assertEqual(tags, {})
        self.assertEqual(
            warnings,
            ["tag key 0x00000003 value block 9 missing or not 0x4a", "tag key 0x00000004 value block 2 missing or not 0x4a"],
        )
        warnings = []
        self.assertEqual(pc.read_tag_map(self.blocks(slots, slot_type=0x4A), ptr(1, 0), 2, warnings=warnings), {})
        self.assertEqual(warnings, ["tag map points to non-synthetic block type 0x4a"])
        warnings = []
        self.assertEqual(pc.read_tag_map(self.blocks(slots), ptr(8, 0), 2, warnings=warnings), {})
        self.assertEqual(warnings, ["tag map points to missing block 8"])

    def test_last_duplicate_key_wins_and_keeps_first_position(self):
        slots = struct.pack("<IxxxxIIQ", 5, 1, 1, ptr(2, 0)) + struct.pack("<IxxxxIIQ", 6, 0, 0, 0)
        slots += struct.pack("<IxxxxIIQ", 5, 1, 1, ptr(2, 4))
        tags = pc.read_tag_map(self.blocks(slots, struct.pack("<II", 10, 20)), ptr(1, 0), 3)
        self.assertEqual(list(tags.items()), [(5, (20,)), (6, ())])


class CullDrawableEdgeCases(unittest.TestCase):
    @staticmethod
    def blocks(ext: bytes, cull: bytes = b"", cull_type: int = pc.CULL_DRAWABLE_STRUCT_HASH, ext_type: int = 0x103):
        return {1: (ext_type, memoryview(ext)), 2: (cull_type, memoryview(cull))}

    def test_first_cull_key_wins_but_every_key_is_listed(self):
        ext = struct.pack("<IxxxxQ", 1, 0) + struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, ptr(2, 0))
        ext += struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, ptr(2, 16))
        cull = struct.pack("<QQQQ", 3, 0xAAAA, 4, 0xBBBB)
        warnings: list[str] = []
        ref, keys = pc.read_cull_drawable(self.blocks(ext, cull), ptr(1, 0), 3, warnings=warnings)
        self.assertEqual(keys, (1, pc.CULL_DRAWABLE_KEY, pc.CULL_DRAWABLE_KEY))
        self.assertEqual((ref.block, ref.offset, ref.drawable_hash, ref.header_qword), (2, 0, 0xAAAA, 3))
        self.assertEqual(warnings, [])

    def test_truncated_map_lists_the_keys_that_fit(self):
        ext = struct.pack("<IxxxxQ", 1, 0) + struct.pack("<IxxxxQ", 2, 0) + b"\0" * 7
        warnings: list[str] = []
        ref, keys = pc.read_cull_drawable(self.blocks(ext), ptr(1, 0), 5, warnings=warnings)
        self.assertIsNone(ref)
        self.assertEqual(keys, (1, 2))
        self.assertEqual(warnings, ["0xb70a22c1 map truncated at block 1+0x0 (count=5)"])

    def test_map_offset_past_the_block_end_returns_nothing(self):
        warnings: list[str] = []
        ref, keys = pc.read_cull_drawable(self.blocks(b"\0" * 16), ptr(1, 1 << 12), 1, warnings=warnings)
        self.assertEqual((ref, keys), (None, ()))
        self.assertEqual(warnings, ["0xb70a22c1 map truncated at block 1+0x1000 (count=1)"])

    def test_target_problems_warn_and_do_not_end_the_scan(self):
        ext = struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, ptr(7, 0)) + struct.pack("<IxxxxQ", 9, 0)
        warnings: list[str] = []
        ref, keys = pc.read_cull_drawable(self.blocks(ext), ptr(1, 0), 2, warnings=warnings)
        self.assertEqual((ref, keys), (None, (pc.CULL_DRAWABLE_KEY, 9)))
        self.assertEqual(warnings, ["cull_drawable points to missing block 7"])
        warnings = []
        ext = struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, ptr(2, 8))
        ref, _keys = pc.read_cull_drawable(self.blocks(ext, b"\0" * 16), ptr(1, 0), 1, warnings=warnings)
        self.assertIsNone(ref)
        self.assertEqual(warnings, ["cull_drawable record truncated at block 2+0x8"])
        warnings = []
        ref, _keys = pc.read_cull_drawable(self.blocks(ext, b"\0" * 16, cull_type=0x77), ptr(1, 0), 1, warnings=warnings)
        self.assertIsNone(ref)
        self.assertEqual(warnings, ["cull_drawable target block 2 has type 0x00000077, expected 0x3d780b74"])


class ParseCatalogueBlocksEdgeCases(unittest.TestCase):
    def test_truncated_state_array_parses_the_states_that_fit(self):
        states = pack_state(1, 2, 3, 4, 5, 6, 0, 0, 0, 0) * 2
        blocks = [
            (pc.CATALOG_ROOT_HASH, pack_root(1, ptr(2, 0))),
            (pc.PRESET_STRUCT_HASH, pack_preset(0x10, 0, 5, ptr(3, 0), 0, 0, 0)),
            (pc.STATE_STRUCT_HASH, states),
        ]
        cat = pc.parse_catalogue_blocks(as_blocks(blocks), 1, framing="0drp")
        self.assertEqual([s.index for s in cat.presets[0].states], [0, 1])
        self.assertEqual(cat.warnings, ("preset 0x00000010 state array truncated at block 3+0x0 (count=5)",))
        self.assertEqual(cat.stats["states"], 2)

    def test_state_array_problems_warn_per_preset(self):
        blocks = [
            (pc.CATALOG_ROOT_HASH, pack_root(2, ptr(2, 0))),
            (pc.PRESET_STRUCT_HASH, pack_preset(0x10, 0, 1, ptr(8, 0), 0, 0, 0) + pack_preset(0x11, 0, 1, ptr(1, 0), 0, 0, 0)),
        ]
        cat = pc.parse_catalogue_blocks(as_blocks(blocks), 1, framing="0drp")
        self.assertEqual(
            cat.warnings,
            (
                "preset 0x00000010 state array points to missing block 8",
                "preset 0x00000011 state array points to block type 0x8dcb4cf5, expected 0x37e52277",
            ),
        )

    def test_zero_pointer_with_a_count_reads_no_tags_and_no_extension_map(self):
        state = pack_state(1, 0, 0, 0, 0, 0, 0, 3, 0, 4)
        blocks = [
            (pc.CATALOG_ROOT_HASH, pack_root(1, ptr(2, 0))),
            (pc.PRESET_STRUCT_HASH, pack_preset(0x10, 0, 1, ptr(3, 0), 0, 0, 2)),
            (pc.STATE_STRUCT_HASH, state),
        ]
        cat = pc.parse_catalogue_blocks(as_blocks(blocks), 1, framing="0drp")
        self.assertEqual(cat.warnings, ())
        st = cat.presets[0].states[0]
        self.assertEqual((st.tags, st.ext_map_keys, st.cull_ref), ({}, (), None))
        self.assertEqual(cat.presets[0].tags, {})

    def test_parse_leaves_the_collector_setting_alone(self):
        blocks = as_blocks(build_blocks(4, seed=1))
        was = gc.isenabled()
        try:
            for enabled in (True, False):
                gc.enable() if enabled else gc.disable()
                pc.parse_catalogue_blocks(blocks, 1, framing="0drp")
                self.assertEqual(gc.isenabled(), enabled)
                with self.assertRaises(ValueError):
                    pc.parse_catalogue_blocks(blocks, 99, framing="0drp")  # missing root
                self.assertEqual(gc.isenabled(), enabled)
        finally:
            gc.enable() if was else gc.disable()


GOLDEN_CLEAN = "246a948b6eafc2a15f217c3dcb366edd7f484ea4db7b9bd92a3e0ec0395e4c4b"
GOLDEN_DAMAGED = "f119de7b2bae6b1e9647aa34e1d64f2b1b55638f2629715147bf24798368e525"


if __name__ == "__main__":
    unittest.main()
