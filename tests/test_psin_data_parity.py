"""The fast walker in tools/psin_data.py gives exactly what the plain one did (synthetic data, no game files).

`ReferenceWalker` below is the straightforward walker, one `value` call per member and one `struct` call per struct,
kept here as the yardstick. Nothing in this file pins a byte, a count or a hash from a platform: every check compares
the tool with the reference on the same input, so it holds on any OS and Python.

    python -I tests/test_psin_data_parity.py
"""
import bisect
import collections
import os
import random
import struct
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psin_data as D  # noqa: E402
from test_psin_data import ENUMS, ITEM_T, KIND, ROOT_T, SCHEMA, STR_T, make_container, ptr, sample  # noqa: E402
from test_psin_index import joaat  # noqa: E402

ELEMENT, SCALARS, HASHES, ENUM_FORMATS = D.ELEMENT, D.SCALARS, D.HASHES, D.ENUMS


# ------------------------------------------------------------------------------------------------ the reference
def ref_block_table(blob, first):
    blocks, at = [], 8
    while at + 8 <= first:
        kind = struct.unpack_from("<I", blob, at)[0]
        if kind == D.END_OF_TABLE:
            break
        if at + D.ROW > first or blob[at + 8:at + D.ROW] != D.FILL:
            raise ValueError(f"block table row at {at} has no 0xaa fill")
        blocks.append((kind, struct.unpack_from("<I", blob, at + 4)[0]))
        at += D.ROW
    else:
        raise ValueError("block table has no 0x70 end padding before the first section")
    end = (at + 15) & ~15
    if blob[at:end] != b"\x70" * (end - at):
        raise ValueError(f"block table end padding at {at} is not 0x70")
    at = end
    out = []
    for kind, size in blocks:
        if at + size > first:
            raise ValueError(f"block table declares blocks past the first section at {first}")
        out.append((kind, at, size))
        at += (size + 15) & ~15
    return out, at == first


def ref_size_of(c, m):
    t = m["type"]
    if t in SCALARS:
        return struct.calcsize(SCALARS[t])
    if t in HASHES:
        return struct.calcsize(HASHES[t])
    if t in ENUM_FORMATS:
        return struct.calcsize(ENUM_FORMATS[t])
    if t in D.POINTERS:
        return 8
    if t in D.CONTAINERS:
        return D.CONTAINERS[t]
    if t == 0x05:
        s = c.structs.get(m["ref"])
        return s["size"] if s else None
    if t == 0x40:
        return m["ref"]
    return None


def ref_align_of(c, m):
    t = m["type"]
    if t == 0x05:
        s = c.structs.get(m["ref"])
        return 1 << s["align_log2"] if s else 8
    if t in D.VECTOR16:
        return 16
    if t == 0x32:
        return 4
    if t == 0x40:
        return 1
    n = ref_size_of(c, m) or 8
    return n if n in (1, 2, 4) else 8


def ref_slot(c, key, value):
    ks, vs = ref_size_of(c, key), ref_size_of(c, value)
    if not ks or not vs:
        return None
    ka, va = ref_align_of(c, key), ref_align_of(c, value)
    voff = -(-ks // va) * va
    a = max(ka, va)
    return voff, -(-(voff + vs) // a) * a


def ref_pointer(c, at):
    v = struct.unpack_from("<Q", c.blob, at)[0]
    return v & 0xFFFFF, v >> 20


def ref_resolve(c, at, need, stats, kind):
    number, off = ref_pointer(c, at)
    if number == 0 and off == 0:
        stats[f"{kind} null"] += 1
        return None
    if not 1 <= number <= len(c.blocks):
        stats[f"{kind} block out of range"] += 1
        return None
    btype, base, size = c.blocks[number - 1]
    if off + need > size:
        stats[f"{kind} past block end"] += 1
        return None
    stats[f"{kind} resolved"] += 1
    return btype, base + off, number


class ReferenceWalker:
    """The walker as it was before it was batched: every member goes through `value`, every struct through `struct`."""

    def __init__(self, container, stats, keep=False):
        self.c = container
        self.stats = stats
        self.keep = keep
        self.cover = [bytearray(size) for _, _, size in container.blocks]
        self.seen = set()
        self._block_bases = [base for _, base, _ in container.blocks]
        self._block_ends = [base + size for _, base, size in container.blocks]
        self._limit = None

    def _block_end(self, at):
        i = bisect.bisect_right(self._block_bases, at) - 1
        if i >= 0 and at <= self._block_ends[i]:
            return self._block_ends[i]
        return None

    def _active_end(self, at):
        return self._limit if self._limit is not None else self._block_end(at)

    def _read_size(self, m):
        t = m["type"]
        if t in SCALARS:
            return struct.calcsize(SCALARS[t])
        if t in HASHES:
            return struct.calcsize(HASHES[t])
        if t in ENUM_FORMATS:
            return struct.calcsize(ENUM_FORMATS[t])
        if t == 0x40:
            return m["ref"]
        return D._HEADER_BYTES.get(t)

    def mark(self, number, absolute, n):
        btype, base, size = self.c.blocks[number - 1]
        lo = absolute - base
        self.cover[number - 1][lo:min(size, lo + n)] = b"\x01" * max(0, min(size, lo + n) - lo)

    def run(self):
        c, st = self.c, self.stats
        if not 1 <= c.root <= len(c.blocks):
            st["root out of range"] += 1
            return None
        btype, base, size = c.blocks[c.root - 1]
        if btype not in c.structs:
            st["root not a struct block"] += 1
            return None
        if size < c.structs[btype]["size"]:
            st["root past block end"] += 1
            return None
        value = self.struct(btype, base, 0, c.root)
        reached = sum(sum(b) for b in self.cover)
        st["bytes"] += sum(size for _, _, size in c.blocks)
        st["bytes reached"] += reached
        st["blocks"] += len(c.blocks)
        st["blocks reached"] += sum(1 for b in self.cover if any(b))
        return value

    def struct(self, name, at, depth, number=None):
        c, st = self.c, self.stats
        s = c.structs.get(name)
        if s is None:
            st["struct not in LSCH"] += 1
            return None
        if number:
            _, base, size = c.blocks[number - 1]
            limit = base + size
        else:
            limit = self._active_end(at)
        if limit is None or at < 0 or at + s["size"] > limit:
            st["struct past block end"] += 1
            return None
        if number:
            self.mark(number, at, s["size"])
        if (name, at) in self.seen or depth > D.MAX_DEPTH:
            return None
        self.seen.add((name, at))
        st["structs"] += 1
        out = {} if self.keep else None
        prev_limit = self._limit
        self._limit = limit
        try:
            for m in s["members"]:
                if m["hash"] == ELEMENT:
                    continue
                v = self.value(m, s["members"], at + m["offset"], depth)
                if self.keep:
                    out[f"{m['hash']:08x}"] = v
        finally:
            self._limit = prev_limit
        return out

    def elements(self, el, members, base, count, depth):
        size = ref_size_of(self.c, el)
        return [self.value(el, members, base + i * size, depth + 1) for i in range(count)]

    def value(self, m, members, at, depth):
        c, st, blob = self.c, self.stats, self.c.blob
        t = m["type"]
        st[f"type {t:02x}"] += 1
        need = self._read_size(m)
        if need is not None:
            end = self._active_end(at)
            if end is None or at < 0 or at + need > end:
                st["member past block end"] += 1
                return None
        if t in SCALARS:
            v = struct.unpack_from(SCALARS[t], blob, at)
            return (v[0] if len(v) == 1 else list(v)) if self.keep else None
        if t in HASHES:
            if not self.keep:
                return None
            return f"{struct.unpack_from(HASHES[t], blob, at)[0]:0{struct.calcsize(HASHES[t]) * 2}x}"
        if t in ENUM_FORMATS:
            return self.enum(m, t, at)
        if t == 0x05:
            return self.struct(m["ref"], at, depth + 1)
        if t == 0x07:
            number, _ = ref_pointer(c, at)
            need = 1
            if 1 <= number <= len(c.blocks):
                s = c.structs.get(c.blocks[number - 1][0])
                if s is not None:
                    need = max(1, s["size"])
            r = ref_resolve(c, at, need, st, "struct pointer")
            if r is None:
                return None
            btype, absolute, number = r
            if btype not in c.structs:
                st["struct pointer to a non-struct block"] += 1
                return None
            return self.struct(btype, absolute, depth + 1, number)
        if t == 0x44 and blob[at + 15] & 0x80:
            n = blob[at + 15] & 0x7F
            if n <= 15 and blob[at + n] == 0:
                st["short strings in place"] += 1
                return blob[at:at + n].decode("utf-8", "replace") if self.keep else None
            st["short string length mismatch"] += 1
            return None
        if t in (0x42, 0x44):
            r = ref_resolve(c, at, 1, st, "string pointer")
            if r is None:
                return None
            btype, absolute, number = r
            _, base, size = c.blocks[number - 1]
            end = blob.find(b"\0", absolute, base + size)
            if end < 0:
                st["string without NUL"] += 1
                return None
            st["strings"] += 1
            self.mark(number, absolute, end - absolute + 1)
            return blob[absolute:end].decode("utf-8", "replace") if self.keep else None
        if t == 0x40:
            st["char arrays"] += 1
            return blob[at:at + m["ref"]].split(b"\0")[0].decode("utf-8", "replace") if self.keep else None
        if t in (0x50, 0x53, 0x54, 0x5C, 0x52):
            return self.array(m, members, at, depth)
        if t in (0x71, 0x73, 0x75):
            return self.map(m, members, at, depth)
        st[f"type {t:02x} not decoded"] += 1
        return None

    def enum(self, m, t, at):
        c, st = self.c, self.stats
        v = struct.unpack_from(ENUM_FORMATS[t], c.blob, at)[0]
        e = c.enums.get(m["ref"])
        if e is None:
            st["enum not in LENM"] += 1
        elif t == 0x65:
            allowed = 0
            for x in e["values"]:
                if 0 <= x["value"] < 32:
                    allowed |= 1 << x["value"]
            st["bit set within its enum" if v & ~allowed == 0 else "bit set outside its enum"] += 1
        else:
            values = {x["value"] for x in e["values"]}
            if t == 0x62 and v not in values and (v & 0xFFFFFFFF) in values:
                v &= 0xFFFFFFFF
            st["enum value listed" if v in values else "enum value not listed"] += 1
        return v if self.keep else None

    def array(self, m, members, at, depth):
        c, st, blob = self.c, self.stats, self.c.blob
        t = m["type"]
        idx = m["index"]
        if idx >= len(members) or members[idx]["hash"] != ELEMENT:
            st["array without element descriptor"] += 1
            return None
        el = members[idx]
        size = ref_size_of(c, el)
        if not size:
            st[f"array of undecoded type {el['type']:02x}"] += 1
            return None
        if t in (0x50, 0x53):
            st["fixed arrays"] += 1
            n = m["ref"] & 0xFFFF
            end = self._active_end(at)
            if end is None or at < 0 or at + size * n > end:
                st["fixed array out of bounds"] += 1
                return None
            return self.elements(el, members, at, n, depth)
        if t == 0x54:
            end = self._active_end(at)
            if end is None or at < 0 or at + 8 > end:
                st["inline array out of bounds"] += 1
                return None
            capacity, count = struct.unpack_from("<II", blob, at)
            st["inline arrays"] += 1
            st["count <= capacity" if count <= capacity else "count > capacity"] += 1
            n = min(count, capacity)
            if at + 8 + size * n > end:
                st["inline array out of bounds"] += 1
                return None
            return self.elements(el, members, at + 8, n, depth)
        if t == 0x5C:
            count, capacity = struct.unpack_from("<II", blob, at)
            where = at + 8
        else:
            count, capacity = struct.unpack_from("<HH", blob, at + 8)
            where = at
        kind = f"array {t:02x}"
        st["count <= capacity" if count <= capacity else "count > capacity"] += 1
        if count == 0:
            st[f"{kind} empty"] += 1
            return []
        r = ref_resolve(c, where, size * count, st, kind)
        if r is None:
            return None
        btype, absolute, number = r
        want = el["ref"] if el["type"] == 0x05 else el["type"]
        st[f"{kind} block type matches element" if btype == want else f"{kind} block type differs"] += 1
        self.mark(number, absolute, size * count)
        _, blk_base, blk_size = c.blocks[number - 1]
        prev_limit = self._limit
        self._limit = blk_base + blk_size
        try:
            return self.elements(el, members, absolute, count, depth)
        finally:
            self._limit = prev_limit

    def map(self, m, members, at, depth):
        c, st, blob = self.c, self.stats, self.c.blob
        t = m["type"]
        ki, vi = m["ref"] & 0xFFFF, m["ref"] >> 16
        if ki >= len(members) or vi >= len(members) or members[ki]["hash"] != ELEMENT or members[vi]["hash"] != ELEMENT:
            st["map without element descriptors"] += 1
            return None
        key, val = members[ki], members[vi]
        layout = ref_slot(c, key, val)
        if layout is None:
            st[f"map of undecoded types {key['type']:02x}/{val['type']:02x}"] += 1
            return None
        voff, slot = layout
        kind = f"map {t:02x}"
        pairs = []
        if t == 0x71:
            flag = struct.unpack_from("<Q", blob, at)[0]
            count, capacity = struct.unpack_from("<HH", blob, at + 16)
            st["map 71 first word 1" if flag == 1 else "map 71 first word not 1"] += 1
            st["count <= capacity" if count <= capacity else "count > capacity"] += 1
            if count == 0:
                st[f"{kind} empty"] += 1
                return {} if self.keep else None
            r = ref_resolve(c, at + 8, (count - 1) * slot + voff + ref_size_of(c, val), st, kind)
            if r is None:
                return None
            _, absolute, number = r
            self.mark(number, absolute, count * slot)
            keys = [blob[absolute + i * slot:absolute + i * slot + ref_size_of(c, key)][::-1] for i in range(count)]
            st[f"{kind} keys ascending" if keys == sorted(keys) else f"{kind} keys not ascending"] += 1
            slots = [absolute + i * slot for i in range(count)]
        elif t == 0x75:
            groups, count = struct.unpack_from("<II", blob, at)
            if count == 0:
                st[f"{kind} empty"] += 1
                return {} if self.keep else None
            group = 16 + 16 * slot
            r = ref_resolve(c, at + 8, groups * group, st, kind)
            if r is None:
                return None
            _, absolute, number = r
            self.mark(number, absolute, groups * group)
            power = groups > 0 and (groups & (groups - 1) == 0)
            st[f"{kind} groups a power of two" if power else f"{kind} groups not a power of two"] += 1
            slots = []
            for g in range(groups):
                base = absolute + g * group
                for i in range(16):
                    if not blob[base + i] & 0x80:
                        slots.append(base + 16 + i * slot)
            st[f"{kind} used slots == count" if len(slots) == count else f"{kind} used slots != count"] += 1
        else:
            capacity, count = struct.unpack_from("<II", blob, at)
            if count == 0:
                st[f"{kind} empty"] += 1
                return {} if self.keep else None
            r = ref_resolve(c, at + 8, capacity * slot, st, kind)
            if r is None:
                return None
            _, absolute, number = r
            self.mark(number, absolute, capacity * slot)
            ks = ref_size_of(c, key)
            slots = [absolute + i * slot for i in range(capacity)
                     if blob[absolute + i * slot:absolute + i * slot + ks] != bytes(ks)]
            st[f"{kind} used slots == count" if len(slots) == count else f"{kind} used slots != count"] += 1
        st[f"{kind} pairs"] += len(slots)
        _, blk_base, blk_size = c.blocks[number - 1]
        prev_limit = self._limit
        self._limit = blk_base + blk_size
        try:
            for s in slots:
                k = self.value(key, members, s, depth + 1)
                v = self.value(val, members, s + voff, depth + 1)
                if self.keep:
                    pairs.append([k, v])
        finally:
            self._limit = prev_limit
        return pairs if self.keep else None


# ------------------------------------------------------------------------------------------------ the containers
WIDE_T, SMALL_T, FLAGS = joaat("WideThing"), joaat("SmallThing"), joaat("FlagsEnum")
EL_ITEM, EL_U32, EL_U8, EL_V3, EL_U16, EL_H64, EL_H32, EL_ENUM, EL_I64, EL_F32, EL_PTR, EL_STR, EL_SHORT = range(13)
WIDE_ELEMENTS = [(0x100, 0, t, 0, 0, ref) for t, ref in (
    (0x05, ITEM_T), (0x15, 0), (0x11, 0), (0x32, 0), (0x13, 0), (0x4F, 0), (0x4A, 0), (0x62, KIND), (0x16, 0),
    (0x21, 0), (0x07, 0), (0x42, 0), (0x44, 0))]

# (name, type, index, ref, size, alignment): the members of WideThing after the element descriptors
WIDE_NAMED = [(f"s{t:02x}", t, 0, 0, struct.calcsize(f), min(8, struct.calcsize(f)))
              for t, f in sorted(SCALARS.items())]
WIDE_NAMED += [(f"h{t:02x}", t, 0, 0, struct.calcsize(f), struct.calcsize(f)) for t, f in sorted(HASHES.items())]
WIDE_NAMED += [("e60", 0x60, 0, KIND, 1, 1), ("e62", 0x62, 0, KIND, 4, 4), ("e62x", 0x62, 0, 0xDEAD, 4, 4),
               ("e64", 0x64, 0, KIND, 2, 2), ("e65", 0x65, 0, FLAGS, 4, 4), ("e68", 0x68, 0, KIND, 8, 8),
               ("e63", 0x63, 0, KIND, 1, 1),
               ("emb", 0x05, 0, ITEM_T, 8, 4), ("emb2", 0x05, 0, 0x1234, 8, 4),
               ("chars", 0x40, 0, 12, 12, 1), ("shortA", 0x44, 0, 0, 16, 8), ("shortB", 0x44, 0, 0, 16, 8),
               ("str", 0x42, 0, 0, 8, 8), ("ptrA", 0x07, 0, 0, 8, 8), ("ptrB", 0x07, 0, 0, 8, 8),
               ("odd", 0x77, 0, 0, 8, 8), ("odd2", 0x5A, 0, 0, 8, 8)]
# containers: (name, type, element index, ref, size)
WIDE_CONTAINERS = [
    ("arr_items", 0x5C, EL_ITEM, 0, 16), ("arr_u8", 0x5C, EL_U8, 0, 16), ("arr_u32", 0x52, EL_U32, 0, 12),
    ("arr_v3", 0x52, EL_V3, 0, 12), ("arr_h64", 0x5C, EL_H64, 0, 16), ("arr_enum", 0x5C, EL_ENUM, 0, 16),
    ("arr_ptr", 0x5C, EL_PTR, 0, 16), ("arr_str", 0x5C, EL_STR, 0, 16), ("arr_f32", 0x5C, EL_F32, 0, 16),
    ("arr_i64", 0x52, EL_I64, 0, 12), ("arr_short", 0x5C, EL_SHORT, 0, 16), ("arr_bad", 0x5C, 0x40, 0, 16),
    ("inline_u16", 0x54, EL_U16, 0, 16), ("inline_item", 0x54, EL_ITEM, 0, 24),
    ("fixed_u32", 0x50, EL_U32, 5, 20), ("fixed_item", 0x53, EL_ITEM, 2, 16), ("fixed_f32", 0x50, EL_F32, 3, 12),
    ("map_u8_u32", 0x71, 0, EL_U8 | EL_U32 << 16, 24), ("map_u16_item", 0x71, 0, EL_U16 | EL_ITEM << 16, 24),
    ("map_h64_f32", 0x71, 0, EL_H64 | EL_F32 << 16, 24), ("map_h32_ptr", 0x71, 0, EL_H32 | EL_PTR << 16, 24),
    ("map_v3_u32", 0x71, 0, EL_V3 | EL_U32 << 16, 24), ("map_i64_short", 0x71, 0, EL_I64 | EL_SHORT << 16, 24),
    ("open_h32_enum", 0x73, 0, EL_H32 | EL_ENUM << 16, 16), ("open_h64_item", 0x73, 0, EL_H64 | EL_ITEM << 16, 16),
    ("open_v3_u32", 0x73, 0, EL_V3 | EL_U32 << 16, 16), ("open_u8_u16", 0x73, 0, EL_U8 | EL_U16 << 16, 16),
    ("swiss_h32_u32", 0x75, 0, EL_H32 | EL_U32 << 16, 16), ("swiss_h64_str", 0x75, 0, EL_H64 | EL_STR << 16, 16),
    ("swiss_item_u32", 0x75, 0, EL_ITEM | EL_U32 << 16, 16), ("swiss_u16_h32", 0x75, 0, EL_U16 | EL_H32 << 16, 16),
    ("map_nokey", 0x71, 0, 99 | EL_U32 << 16, 24), ("map_undecoded", 0x71, 0, 0x40 | EL_U32 << 16, 24),
]


def wide_schema():
    """(WideThing member list, its size, the member offsets by name). WideThing has one member of every shape."""
    members, at, where = list(WIDE_ELEMENTS), 0, {}
    for name, t, index, ref, size, align in WIDE_NAMED:
        at = -(-at // align) * align
        members.append((joaat(name), at, t, 0, index, ref))
        where[name] = at
        at += size
    for name, t, index, ref, size in WIDE_CONTAINERS:
        at = -(-at // 8) * 8
        members.append((joaat(name), at, t, 0, index, ref))
        where[name] = at
        at += size
    # a member that sticks out of its struct, to leave the fast path on the others alone
    members.append((joaat("outside"), at, 0x21, 0, 0, 0))
    where["outside"] = at
    size = -(-(at + 2) // 16) * 16
    return members, size, where


def wide_container(seed, keep_sorted=None):
    """A container whose root has one member of every shape, over random bytes, with headers that resolve."""
    rnd = random.Random(seed)
    members, size, where = wide_schema()
    # the SmallThing struct: a u64 that does not fit its 4 bytes, and a member far outside it
    small = [(joaat("big"), 0, 0x17, 0, 0, 0), (joaat("far"), 100, 0x15, 0, 0, 0), (joaat("ok"), 0, 0x15, 0, 0, 0)]
    structs = [(WIDE_T, 0x1, size, 3, members), (SMALL_T, 0x2, 4, 2, small), SCHEMA[1]]
    enums = [(KIND, 3, [(0, joaat("A")), (1, joaat("B")), (-1, joaat("C")), (0x80000000, joaat("D"))]),
             (FLAGS, 4, [(0, joaat("F0")), (1, joaat("F1")), (5, joaat("F5")), (40, joaat("F40"))])]
    root = bytearray(rnd.getrandbits(8) for _ in range(size))
    blocks = [(WIDE_T, root)]

    def new_block(btype, data):
        blocks.append((btype, bytearray(data)))
        return len(blocks)

    strings = b"".join(b"text%d\0" % i for i in range(7))
    n_str = new_block(STR_T, strings)
    n_item = new_block(ITEM_T, bytes(rnd.getrandbits(8) for _ in range(8 * 6)))
    n_small = new_block(SMALL_T, bytes(rnd.getrandbits(8) for _ in range(16)))
    n_wide = new_block(WIDE_T, bytes(rnd.getrandbits(8) for _ in range(size)))   # a pointer can reach another one

    def rand_ptr(n):
        return ptr(rnd.choice((n_str, n_item, n_small, n_wide, 1, n, 0, 999)), rnd.choice((0, 0, 8, 16, 3)))

    def random_bytes(n):
        return bytes(rnd.getrandbits(8) for _ in range(n))

    for name, t, index, ref, csize in WIDE_CONTAINERS:
        at = where[name]
        if t in (0x5C, 0x52):
            count = rnd.choice((0, 1, 3, 5, 6))
            esize = ref_size_of(SIZER, wide_member(members, index)) or 8
            data = bytearray(random_bytes(max(1, esize) * 8))
            if index == EL_PTR or index == EL_STR:
                for i in range(0, len(data) - 7, 8):
                    struct.pack_into("<Q", data, i, rand_ptr(0))
            block = new_block(rnd.choice((0x15, ITEM_T, 0x10, 0x4A)), data)
            if t == 0x5C:
                struct.pack_into("<IIQ", root, at, count, count + rnd.choice((0, 0, 2)), ptr(block, rnd.choice((0, 0, 8))))
            else:
                struct.pack_into("<QHH", root, at, ptr(block, rnd.choice((0, 0, 8))), count, count + rnd.choice((0, 3)))
        elif t == 0x54:
            cap = 3
            struct.pack_into("<II", root, at, cap, rnd.choice((0, 1, 2, 3, 4)))
            root[at + 8:at + csize] = random_bytes(csize - 8)
        elif t in (0x50, 0x53):
            root[at:at + csize] = random_bytes(csize)
        else:
            layout = ref_slot(SIZER, wide_member(members, ref & 0xFFFF), wide_member(members, ref >> 16))                 if (ref & 0xFFFF) < len(members) and (ref >> 16) < len(members) else None
            if layout is None:   # a map the walker cannot lay out: the bytes are never read
                layout, ks = (8, 16), 4
            else:
                ks = ref_size_of(SIZER, wide_member(members, ref & 0xFFFF))
            voff, slot = layout
            n = rnd.choice((0, 1, 4, 9, 33))
            if t == 0x71:
                data = bytearray(random_bytes(slot * max(n, 1) + 8))
                keys = sorted(rnd.getrandbits(8 * min(ks, 8)) for _ in range(max(n, 1)))
                if keep_sorted is False or (keep_sorted is None and rnd.random() < 0.3):
                    rnd.shuffle(keys)
                for i, k in enumerate(keys):
                    if i * slot + ks <= len(data):
                        data[i * slot:i * slot + min(ks, 8)] = k.to_bytes(min(ks, 8), "little")
                block = new_block(0x100, data)
                struct.pack_into("<QQHH", root, at, rnd.choice((1, 1, 1, 0)), ptr(block), n, n + rnd.choice((0, 2)))
            elif t == 0x73:
                cap = rnd.choice((1, 4, 16, 40))
                data = bytearray(random_bytes(slot * cap))
                used = 0
                for i in range(cap):
                    if rnd.random() < 0.5:
                        data[i * slot:i * slot + ks] = bytes(ks)
                    else:
                        data[i * slot:i * slot + ks] = bytes([1 + rnd.randrange(255)]) * ks
                        used += 1
                block = new_block(0x102, data)
                struct.pack_into("<IIQ", root, at, cap, n if n == 0 else used + rnd.choice((0, 0, 1)), ptr(block))
            else:
                groups = rnd.choice((1, 2, 3, 4))
                data = bytearray(random_bytes((16 + 16 * slot) * groups))
                used = 0
                for g in range(groups):
                    for i in range(16):
                        if rnd.random() < 0.4:
                            data[g * (16 + 16 * slot) + i] |= 0x80
                        else:
                            data[g * (16 + 16 * slot) + i] &= 0x7F
                            used += 1
                block = new_block(0x101, data)
                struct.pack_into("<IIQ", root, at, groups, n if n == 0 else used + rnd.choice((0, 0, 1)), ptr(block))
    for name in ("ptrA", "ptrB", "str", "emb"):
        struct.pack_into("<Q", root, where[name], rand_ptr(0))
    root[where["shortA"] + 15] = 0x80 | 5
    root[where["shortA"]:where["shortA"] + 5] = b"hello"
    root[where["shortA"] + 5] = 0
    struct.pack_into("<Q", root, where["shortB"], rand_ptr(0))
    root[where["shortB"] + 15] = 0
    struct.pack_into("<i", root, where["e62"], rnd.choice((0, 1, -1, 0x80000000 - (1 << 32), 7)))
    struct.pack_into("<I", root, where["e65"], rnd.choice((0, 1, 5, 7, 1 << 20)))
    root[where["chars"]:where["chars"] + 12] = b"chars\0abc\0\0\0"
    blocks[0] = (WIDE_T, root)
    return make_container([(t, bytes(d)) for t, d in blocks], structs, enums)


def wide_member(members, index):
    t, i, tp, _s, idx, ref = members[index]
    return {"hash": t, "offset": i, "type": tp, "sub": 0, "index": idx, "ref": ref}


class _Sizer:
    """Just enough of a Container for the reference's size and slot arithmetic: the one struct the maps hold."""

    structs = {ITEM_T: {"size": 8, "align_log2": 2}}


SIZER = _Sizer()


def dense_container(items=3000, values=6000, pairs=1200, groups=64, capacity=1024, strings=300, sparse=False):
    """The `sample()` schema over a large dense (or, with `sparse`, a nearly empty but large) set of blocks."""
    rnd = random.Random(5)
    root = bytearray(0x90)
    item_bytes = b"".join(struct.pack("<fI", rnd.random(), rnd.getrandbits(32)) for _ in range(items))
    value_bytes = struct.pack(f"<{values}I", *[rnd.getrandbits(32) for _ in range(values)])
    keys = sorted(rnd.getrandbits(32) | 1 for _ in range(pairs))
    pair_bytes = b"".join(struct.pack("<II", k, rnd.getrandbits(32)) for k in keys)
    swiss, used = bytearray(), 0
    for g in range(groups):
        ctrl, slots = bytearray(b"\x80" * 16), bytearray(128)
        for i in range(12 if not sparse else (1 if g == 0 else 0)):
            ctrl[i] = rnd.getrandbits(7)
            struct.pack_into("<II", slots, i * 8, rnd.getrandbits(32) | 1, i)
            used += 1
        swiss += ctrl + slots
    open_slots, n_open = bytearray(capacity * 8), 0
    for i in range(capacity):
        if (not sparse and rnd.random() < 0.7) or (sparse and i == 3):
            struct.pack_into("<II", open_slots, i * 8, rnd.getrandbits(32) | 1, i)
            n_open += 1
    text = b"".join(b"string%d\0" % i for i in range(strings))
    struct.pack_into("<IIQ", root, 0x00, items, items, ptr(2))
    struct.pack_into("<QHH", root, 0x10, ptr(3), values, values)
    struct.pack_into("<QQHH", root, 0x20, 1, ptr(4), pairs, pairs)
    struct.pack_into("<IIQ", root, 0x38, groups, used, ptr(5))
    struct.pack_into("<IIQ", root, 0x48, capacity, n_open, ptr(6))
    struct.pack_into("<Q", root, 0x58, ptr(7))
    struct.pack_into("<Q", root, 0x60, ptr(8))
    root[0x68:0x6B] = b"abc"
    root[0x68 + 15] = 0x83
    struct.pack_into("<i", root, 0x78, 1)
    struct.pack_into("<3I", root, 0x7C, 7, 8, 9)
    blocks = [(ROOT_T, bytes(root)), (ITEM_T, item_bytes), (0x15, value_bytes), (0x100, pair_bytes),
              (0x101, bytes(swiss)), (0x102, bytes(open_slots)), (ITEM_T, struct.pack("<fI", 3.5, 30)), (STR_T, text)]
    return make_container(blocks, SCHEMA, ENUMS)


# ------------------------------------------------------------------------------------------------ the checks
def outcome(walker_class, blob, keep, container=None):
    """Everything a walk produces: the counts, the tree, the coverage, or the exception that ended it."""
    stats = collections.Counter()
    walker = walker_class(container or D.Container(blob), stats, keep=keep)
    try:
        tree = walker.run()
    except (struct.error, IndexError) as e:
        return "raised", type(e).__name__, dict(stats)
    return dict(stats), repr(tree), [bytes(b) for b in walker.cover]   # repr: a NaN float is not equal to itself


def logging_class(base):
    """`base` with `struct` and `value` overridden to log each call, the way psin_names.HashWalker hooks them."""
    class Logged(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.log = []

        def struct(self, name, at, depth, number=None):
            self.log.append(("struct", name, at, depth, number))
            return super().struct(name, at, depth, number)

        def value(self, m, members, at, depth):
            self.log.append(("value", m["hash"], m["type"], at, depth))
            return super().value(m, members, at, depth)
    return Logged


def mutated(blob, rnd, flips):
    b = bytearray(blob)
    for _ in range(flips):
        i = rnd.randrange(len(b)) if rnd.random() < 0.6 else rnd.randrange(min(len(b), 700))
        b[i] = rnd.choice((0, 0xFF, rnd.getrandbits(8), (b[i] + 1) & 0xFF))
    return bytes(b)


def damaged(blob, rnd, flips):
    """A mutated copy of `blob` that still opens as a container (the first few tries may not)."""
    while True:
        out = mutated(blob, rnd, flips)
        try:
            D.Container(out)
            return out
        except ValueError:
            pass


class TestWalkMatchesReference(unittest.TestCase):
    def assertSame(self, blob, label=""):
        container = D.Container(blob)
        for keep in (False, True):
            self.assertEqual(outcome(D.Walker, blob, keep, container),
                             outcome(ReferenceWalker, blob, keep, container), f"{label} keep={keep}")

    def test_the_sample_container(self):
        for kind in (0, 1, 5, -1):
            self.assertSame(sample(kind), f"kind {kind}")

    def test_wide_containers_cover_every_member_shape(self):
        seen = collections.Counter()
        for seed in range(40):
            blob = wide_container(seed)
            self.assertSame(blob, f"seed {seed}")
            seen.update(outcome(D.Walker, blob, False)[0])
        for key in ("map 71 keys ascending", "map 71 keys not ascending", "map 73 pairs", "map 75 pairs",
                    "map 71 pairs", "array 5c resolved", "array 52 resolved", "inline arrays", "fixed arrays",
                    "struct pointer resolved", "string pointer resolved", "short strings in place",
                    "enum value listed", "enum value not listed", "bit set outside its enum", "char arrays",
                    "type 77 not decoded", "map 73 used slots == count", "map 75 used slots == count",
                    "map 75 used slots != count", "member past block end"):
            self.assertGreater(seen[key], 0, key)

    def test_dense_and_sparse_containers(self):
        self.assertSame(dense_container(), "dense")
        self.assertSame(dense_container(items=30, values=40, pairs=9, groups=512, capacity=9000, sparse=True),
                        "sparse")
        self.assertSame(dense_container(items=9000, values=10, pairs=2000, groups=4, capacity=9000), "big blocks")

    def test_damaged_containers(self):
        rnd = random.Random(2024)
        bases = [sample(), wide_container(3), wide_container(11), dense_container(200, 300, 40, 8, 64, 20)]
        for n in range(240):
            blob = mutated(rnd.choice(bases), rnd, rnd.choice((1, 3, 8, 24)))
            try:
                container = D.Container(blob)
            except ValueError:
                with self.assertRaises(ValueError):
                    D.Container(blob)
                continue
            for keep in (False, True):
                self.assertEqual(outcome(D.Walker, blob, keep, container),
                                 outcome(ReferenceWalker, blob, keep, container), f"mutation {n} keep={keep}")

    def test_a_walker_with_hooks_sees_the_same_calls_in_the_same_order(self):
        New, Ref = logging_class(D.Walker), logging_class(ReferenceWalker)
        rnd = random.Random(8)
        for blob in (sample(), wide_container(1), wide_container(2), damaged(wide_container(4), rnd, 6)):
            container = D.Container(blob)
            for keep in (False, True):
                a, b = New(container, collections.Counter(), keep=keep), Ref(container, collections.Counter(), keep=keep)
                self.assertEqual((repr(a.run()), dict(a.stats), a.log), (repr(b.run()), dict(b.stats), b.log))

    def test_psin_names_hash_walker_records_what_it_always_did(self):
        import psin_names

        class RefHashWalker(ReferenceWalker):     # psin_names.HashWalker over the reference walker
            def __init__(self, container, stats):
                super().__init__(container, stats)
                self.found, self._structs, self._fields = [], [], []

            def struct(self, name, at, depth, number=None):
                self._structs.append(name)
                try:
                    return super().struct(name, at, depth, number)
                finally:
                    self._structs.pop()

            def value(self, m, members, at, depth):
                named = m["hash"] != ELEMENT
                if named:
                    self._fields.append(m["hash"])
                try:
                    t = m["type"]
                    if t in HASHES and self._structs and self._fields:
                        v = struct.unpack_from(HASHES[t], self.c.blob, at)[0]
                        self.found.append((self._structs[-1], self._fields[-1], t, v))
                    return super().value(m, members, at, depth)
                finally:
                    if named:
                        self._fields.pop()

        rnd = random.Random(31)
        for blob in (sample(), wide_container(5), wide_container(6), damaged(wide_container(7), rnd, 5)):
            container = D.Container(blob)
            new = psin_names.HashWalker(container, collections.Counter())
            old = RefHashWalker(container, collections.Counter())
            self.assertEqual(repr(new.run()), repr(old.run()))
            self.assertEqual((new.found, dict(new.stats)), (old.found, dict(old.stats)))
            self.assertTrue(new.found)


class TestParts(unittest.TestCase):
    def test_block_table_matches_reference(self):
        rnd = random.Random(4)
        blob = make_container([(0x15, b"\1\0\0\0"), (0x16, b"\0" * 17), (ITEM_T, b"\2" * 40)], [])
        first = struct.unpack_from(">I", blob, 4)[0]
        cases = [(blob, first)]
        for _ in range(150):
            cases.append((mutated(blob, rnd, rnd.choice((1, 2, 3))), first + rnd.choice((0, 0, 16, -16))))
        for data, end in cases:
            try:
                want = ref_block_table(data, end)
            except ValueError as e:
                with self.assertRaises(ValueError) as got:
                    D.block_table(data, end)
                self.assertEqual(str(got.exception), str(e))
                continue
            self.assertEqual(D.block_table(data, end), want)

    def test_container_helpers_match_reference(self):
        c = D.Container(wide_container(9))
        members = [wide_member(wide_schema()[0], i) for i in range(len(wide_schema()[0]))]
        for m in members + [{"type": t, "ref": 7, "hash": 1, "offset": 0, "index": 0} for t in range(256)]:
            self.assertEqual(c.size_of(m), ref_size_of(c, m), m["type"])
            self.assertEqual(c.align_of(m), ref_align_of(c, m), m["type"])
        for k in members[:20]:
            for v in members[:20]:
                self.assertEqual(c.slot(k, v), ref_slot(c, k, v))
        stats_a, stats_b = collections.Counter(), collections.Counter()
        for at in range(0, len(c.blob) - 8, 61):
            for kind in ("struct pointer", "string pointer", "array 5c", "map 73"):
                self.assertEqual(c.resolve(at, 12, stats_a, kind), ref_resolve(c, at, 12, stats_b, kind))
            self.assertEqual(c.pointer(at), ref_pointer(c, at))
        self.assertEqual(stats_a, stats_b)

    def test_marking_matches_reference_for_every_range(self):
        container = D.Container(dense_container(items=9000, values=10, pairs=2, groups=1, capacity=4, strings=2))
        new, old = D.Walker(container, collections.Counter()), ReferenceWalker(container, collections.Counter())
        rnd = random.Random(6)
        for number in (1, 2, 3):
            size = container.blocks[number - 1][2]
            base = container.blocks[number - 1][1]
            spans = [(0, size), (0, size + 50), (5, 70000), (size - 3, 9), (size, 4), (size + 9, 1), (-7, 4), (-3, 0),
                     (3, 0)] + [(rnd.randrange(size), rnd.randrange(150000)) for _ in range(30)]
            for lo, n in spans:
                new.mark(number, base + lo, n)
                old.mark(number, base + lo, n)
        self.assertEqual([bytes(b) for b in new.cover], [bytes(b) for b in old.cover])

    def test_plans_and_stat_keys_stay_bounded(self):
        container = D.Container(sample())
        walker = D.Walker(container, collections.Counter())
        before = len(D._RESOLVE_KEYS)
        for n in range(D._CACHE_MAX + 500):
            walker._plan(n, {"size": 8, "members": [{"hash": 1, "offset": 0, "type": 0x15, "index": 0, "ref": 0}]})
            container.resolve(0, 1, collections.Counter(), f"kind {n}")
        self.assertEqual(len(walker._plans), D._CACHE_MAX)
        self.assertEqual(len(D._RESOLVE_KEYS), before)
        self.assertLessEqual(len(D._RESOLVE_KEYS), D._CACHE_MAX)


class TestPeakMemory(unittest.TestCase):
    def peak(self, walker_class, blob, keep):
        container = D.Container(blob)
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            walker_class(container, collections.Counter(), keep=keep).run()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def assertNoWorse(self, blob, label):
        for keep in (False, True):
            new, old = self.peak(D.Walker, blob, keep), self.peak(ReferenceWalker, blob, keep)
            self.assertLessEqual(new, old, f"{label} keep={keep}: {new} bytes against {old}")

    def test_dense(self):
        self.assertNoWorse(dense_container(items=1200, values=2400, pairs=500, groups=24, capacity=400), "dense")
        self.assertNoWorse(dense_container(items=40, values=40, pairs=1500, groups=4, capacity=16), "many pairs")
        self.assertNoWorse(dense_container(items=40, values=40, pairs=10, groups=600, capacity=16), "many groups")
        self.assertNoWorse(dense_container(items=40, values=40, pairs=10, groups=2, capacity=9000), "big open table")

    def test_sparse(self):
        self.assertNoWorse(dense_container(items=30, values=40, pairs=9, groups=700, capacity=9000, sparse=True),
                           "sparse")

    def test_small(self):
        self.assertNoWorse(sample(), "sample")
        self.assertNoWorse(wide_container(1), "wide")


if __name__ == "__main__":
    unittest.main()
