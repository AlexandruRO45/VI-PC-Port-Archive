"""Decode the instance data of PSIN containers: the data blocks between the header and LSCH, read through the schema.

`psin_index.py` walks the sections and decodes the schema (`LSCH` structs, `LENM` enums). This tool reads the data
those structs describe. Everything below is measured over the 17,507 complete PSIN containers with an `LSCH` and an
`LRSC` section in package 1.0.2.1 (see docs/psin.md, "Instance data").

Block table. The 16-byte rows start right after "PSIN" and its big-endian first-section offset:

    +0  u32 type     a struct name hash from LSCH, a member type code (< 0x100, e.g. 0x16 for an i64 array) or
                     0x100 + n for an anonymous block (the pair array of a map, for example)
    +4  u32 size     bytes of data in the block
    +8  8 fill bytes (0xaa)

The rows start at +8, so the table always ends 8 bytes short of a 16-byte boundary; those 8 bytes are 0x70 ("pppppppp")
and the blocks start right after them, in table order, each one padded to 16 bytes. Together they end exactly at the
first section. Blocks are numbered from 1. (The first row is what psin_index.py reports as "version" and
"table_bytes".)

Root. `LRSC` starts with a 0DRP header; the u32 at its +0x10 is the number of the root block, a struct block.

Pointers are u64: the low 20 bits are the block number (0 = null), the bits above are the byte offset in the block.

Member types (LSCH member +8), with the layouts this tool reads:

    0x01 bool, 0x10/0x11 s8/u8, 0x12/0x13 s16/u16, 0x14/0x15 s32/u32, 0x16/0x17/0x08 64-bit int, 0x20 f16,
    0x21 f32, 0x30 2xf32, 0x32 3xf32, 0x31/0x33/0x34/0x35/0x0c/0x0d 16-byte vectors, 0x36 4x4 f32
    0x4a, 0x4d, 0x4b  u32 hash          0x4f  u64 hash
    0x05  embedded struct (ref = struct name hash)
    0x07  pointer to a struct block (the block's type is the struct)
    0x42  pointer to a NUL-terminated string in a type-0x10 block
    0x44  16-byte string: when bit 7 of byte 15 is set, the low 7 bits are the length of a string stored in place
          (bytes 0..14, NUL-terminated); otherwise the first 8 bytes are a string pointer as for 0x42
    0x40  char[ref]                      0x50, 0x53  fixed array of (ref & 0xffff) elements, in place
    0x5c  {u32 count, u32 capacity, u64 pointer}       0x52  {u64 pointer, u16 count, u16 capacity}
    0x54  {u32 capacity, u32 count, elements[capacity]} in place
    0x60/0x63 u8, 0x64 u16, 0x62 32-bit enum (LENM values may be negative or >= 2**31, so both readings are
    checked), 0x65 u32 enum bit set, 0x68 64-bit enum (ref = LENM enum hash)
    0x71  {u64 1, u64 pointer, u16 count, u16 capacity, u32 0}: a sorted array of (key, value) pairs
    0x73  {u32 capacity, u32 count, u64 pointer}: open addressing over `capacity` (key, value) slots; key 0 = empty
    0x75  {u32 groups, u32 count, u64 pointer}: groups of 16 control bytes + 16 (key, value) slots; a control
          byte with bit 7 set is an empty slot

An array's element is the member at its element index (+10); a map's key and value elements are the members at
`ref & 0xffff` and `ref >> 16`. In a (key, value) slot the value starts at the key size rounded up to the value's
alignment, and the slot is rounded up to the larger alignment. Alignment is the struct's 2^(+9) for type 5, 16 for
the 16-byte vectors and 0x36, 1 for 0x40, 4 for 0x32, else the element size capped at 8.

Usage:
    python -I psin_data.py <PSIN file or carve directory> [--json] [--limit N]
    python -I psin_data.py <PSIN file> --dump

The default output is counts only: containers, block tables, bytes reached from the root, pointers, arrays and maps,
with every check. `--dump` prints the decoded tree of one container on stdout, hashes as hex, for local inspection;
it writes no file. Unknown member types (0x5a, 0x76, 0x77, ...) are counted and skipped, never guessed.

In a carve directory, a stream that does not start with the `PSIN` magic is skipped (the folder holds other files
too). As the only argument it is a wrong path, so it is one `ERROR:` line and exit 2 rather than an empty count.
"""
import argparse
import array
import bisect
import collections
import itertools
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psin_index  # noqa: E402

END_OF_TABLE = 0x70707070
FILL = b"\xaa" * 8
ROW = 16
ELEMENT = psin_index.ARRAY_ELEMENT
MAX_DEPTH = 64

SCALARS = {
    0x01: "<?", 0x10: "<b", 0x11: "<B", 0x12: "<h", 0x13: "<H", 0x14: "<i", 0x15: "<I", 0x16: "<q", 0x17: "<Q",
    0x08: "<q", 0x20: "<e", 0x21: "<f", 0x30: "<2f", 0x32: "<3f", 0x31: "<4f", 0x33: "<4f", 0x34: "<4f",
    0x35: "<4f", 0x0C: "<4f", 0x0D: "<4f", 0x36: "<16f",
}
HASHES = {0x4A: "<I", 0x4D: "<I", 0x4B: "<I", 0x4F: "<Q"}
ENUMS = {0x60: "<B", 0x63: "<B", 0x64: "<H", 0x62: "<i", 0x65: "<I", 0x68: "<q"}
POINTERS = (0x07, 0x42)
CONTAINERS = {0x5C: 16, 0x52: 16, 0x73: 16, 0x75: 16, 0x71: 24, 0x44: 16}
_HEADER_BYTES = {0x07: 8, 0x42: 8, 0x44: 16, 0x52: 12, 0x5C: 16, 0x71: 20, 0x73: 16, 0x75: 16}
VECTOR16 = (0x31, 0x33, 0x34, 0x35, 0x36, 0x0C, 0x0D)

# Per-type constants, built once so the walker does no format parsing or f-string work per value.
_FIXED_SIZE = {}                      # byte size of a scalar, hash or enum member
for _table in (ENUMS, HASHES, SCALARS):  # SCALARS wins if a type were ever listed twice
    for _t, _fmt in _table.items():
        _FIXED_SIZE[_t] = struct.calcsize(_fmt)
_READ_SIZE = {**_HEADER_BYTES, **_FIXED_SIZE}   # bytes a member needs inside its block (0x40 is its `ref`)
_SIZE_OF = {**{t: 8 for t in POINTERS}, **CONTAINERS, **_FIXED_SIZE}   # Container.size_of for the fixed types
# type -> (bytes, unpack_from, components, hex format or None) for the members whose value is a plain unpack
_UNPACK = {}
for _table, _hex in ((HASHES, True), (SCALARS, False)):
    for _t, _fmt in _table.items():
        _st = struct.Struct(_fmt)
        _UNPACK[_t] = (_st.size, _st.unpack_from, len(_st.unpack_from(bytes(_st.size))),
                       f"0{_st.size * 2}x" if _hex else None)
_ENUM_UNPACK = {t: struct.Struct(fmt).unpack_from for t, fmt in ENUMS.items()}
_TYPE_KEY = {t: f"type {t:02x}" for t in range(256)}   # the stats key of a member type (a u8 in LSCH)
_U32 = struct.Struct("<I").unpack_from
_U64 = struct.Struct("<Q").unpack_from


def _resolve_keys(kind):
    """The four stats keys Container.resolve counts under for a pointer of this kind."""
    return f"{kind} null", f"{kind} block out of range", f"{kind} past block end", f"{kind} resolved"


# the kinds this tool resolves; any other kind formats its keys on each call, so nothing here grows with the file
_RESOLVE_KEYS = {k: _resolve_keys(k) for k in (
    "struct pointer", "string pointer", *(f"array {t:02x}" for t in (0x5C, 0x52)),
    *(f"map {t:02x}" for t in (0x71, 0x73, 0x75)))}
_UU = struct.Struct("<II").unpack_from
_HH = struct.Struct("<HH").unpack_from
# the stats keys of the two array layouts that point to a block: kind, empty, block type matches, block type differs
_ARRAY_KEYS = {t: (f"array {t:02x}", f"array {t:02x} empty", f"array {t:02x} block type matches element",
                   f"array {t:02x} block type differs") for t in (0x5C, 0x52)}
_ONES = memoryview(b"\x01" * 65536)      # source of the coverage marks, so a large mark allocates nothing
_CTRL_USED = bytes(1 if b < 0x80 else 0 for b in range(256))   # 0x75 control byte -> 1 when the slot is in use
_ZERO16 = bytes(16)
_KEY_INT = {1: "B", 2: "H", 4: "I", 8: "Q"}    # key sizes whose little-endian value orders like the reversed bytes
_SCAN_SLOTS = 4096                             # map slots handled at a time, to bound the temporaries
_CACHE_MAX = 4096                              # bound for every cache keyed by something the file chooses


def block_table(blob, first):
    """The block table: a list of (type, offset, size) and whether the blocks end exactly at `first`.

    Raises ValueError when a row has the wrong fill or no 0x70 padding ends the table before `first`.
    """
    blocks, at = [], 8
    while at + 8 <= first:
        kind = _U32(blob, at)[0]
        if kind == END_OF_TABLE:
            break
        if at + ROW > first or blob[at + 8:at + ROW] != FILL:
            raise ValueError(f"block table row at {at} has no 0xaa fill")
        blocks.append((kind, _U32(blob, at + 4)[0]))
        at += ROW
    else:
        raise ValueError("block table has no 0x70 end padding before the first section")
    end = (at + 15) & ~15
    if blob[at:end] != b"\x70" * (end - at):
        raise ValueError(f"block table end padding at {at} is not 0x70")
    at = end
    out = []
    for kind, size in blocks:
        if at + size > first:  # a damaged table must not make the walker allocate past the region
            raise ValueError(f"block table declares blocks past the first section at {first}")
        out.append((kind, at, size))
        at += (size + 15) & ~15
    return out, at == first


class Container:
    """One PSIN container's schema, blocks and root, ready to walk."""

    def __init__(self, blob):
        self.blob = blob
        self.error = ""
        record = psin_index.parse(blob)
        if record is None:
            raise ValueError("not a PSIN container")
        if not record["complete"]:
            raise ValueError(f"incomplete container: {record['error']}")
        sections = {s["tag"]: s for s in record["sections"]}
        if "LSCH" not in sections or "LRSC" not in sections:
            raise ValueError("no LSCH or LRSC section")
        lsch, lenm, lrsc = sections["LSCH"], sections.get("LENM"), sections["LRSC"]
        self.structs = {s["hash"]: s for s in psin_index.decode_lsch(
            blob[lsch["offset"] + 8:lsch["offset"] + lsch["size"]])}
        self.enums = {}
        if lenm:
            self.enums = {e["hash"]: e for e in psin_index.decode_lenm(
                blob[lenm["offset"] + 8:lenm["offset"] + lenm["size"]])}
        self.blocks, self.fills = block_table(blob, record["first_section"])
        if lrsc["payload"] < 0x14 or blob[lrsc["offset"] + 8:lrsc["offset"] + 12] != b"0DRP":
            raise ValueError("LRSC does not start with a 0DRP header")
        self.root = struct.unpack_from("<I", blob, lrsc["offset"] + 8 + 0x10)[0]

    # ---------------------------------------------------------------- layout
    def size_of(self, m):
        t = m["type"]
        n = _SIZE_OF.get(t)
        if n is not None:
            return n
        if t == 0x05:
            s = self.structs.get(m["ref"])
            return s["size"] if s else None
        if t == 0x40:
            return m["ref"]
        return None

    def align_of(self, m):
        t = m["type"]
        if t == 0x05:
            s = self.structs.get(m["ref"])
            return 1 << s["align_log2"] if s else 8
        if t in VECTOR16:
            return 16
        if t == 0x32:
            return 4
        if t == 0x40:
            return 1
        n = self.size_of(m) or 8
        return n if n in (1, 2, 4) else 8

    def slot(self, key, value):
        """(value offset, slot size) of a (key, value) pair, or None when either size is unknown."""
        ks, vs = self.size_of(key), self.size_of(value)
        if not ks or not vs:
            return None
        ka, va = self.align_of(key), self.align_of(value)
        voff = -(-ks // va) * va
        a = max(ka, va)
        return voff, -(-(voff + vs) // a) * a

    # ---------------------------------------------------------------- pointers
    def pointer(self, at):
        """(block number, offset) of the u64 pointer at `at`; (0, 0) for null."""
        v = _U64(self.blob, at)[0]
        return v & 0xFFFFF, v >> 20

    def resolve(self, at, need, stats, kind):
        """The (block type, absolute offset, block number) a pointer reaches, or None (null or out of range)."""
        v = _U64(self.blob, at)[0]
        number, off = v & 0xFFFFF, v >> 20
        keys = _RESOLVE_KEYS.get(kind) or _resolve_keys(kind)
        if number == 0 and off == 0:
            stats[keys[0]] += 1
            return None
        blocks = self.blocks
        if not 1 <= number <= len(blocks):
            stats[keys[1]] += 1
            return None
        btype, base, size = blocks[number - 1]
        if off + need > size:
            stats[keys[2]] += 1
            return None
        stats[keys[3]] += 1
        return btype, base + off, number


class Walker:
    """Visits every value reachable from the root; collects counts, byte coverage and (optionally) the values.

    A subclass may override `value` and `struct`: they are called once per member and per struct, in file order, as
    ever. Only a walker that overrides none of `value`, `struct`, `enum` and `mark` takes the batched paths (`_run`,
    the per-struct plan, the map scans), which add up the same counts and visit the same bytes in the same order.
    """

    def __init__(self, container, stats, keep=False):
        self.c = container
        self.stats = stats
        self.keep = keep
        self.cover = [bytearray(size) for _, _, size in container.blocks]
        self.seen = set()
        self._block_bases = [base for _, base, _ in container.blocks]
        self._block_ends = [base + size for _, base, size in container.blocks]
        self._limit = None
        self._plans = {}
        self._plain = all(getattr(getattr(self, n), "__func__", None) is getattr(Walker, n)
                          for n in ("value", "struct", "enum", "mark"))

    def _block_end(self, at):
        i = bisect.bisect_right(self._block_bases, at) - 1
        if i >= 0 and at <= self._block_ends[i]:
            return self._block_ends[i]
        return None

    def _active_end(self, at):
        return self._limit if self._limit is not None else self._block_end(at)

    def _read_size(self, m):
        t = m["type"]
        return m["ref"] if t == 0x40 else _READ_SIZE.get(t)

    def mark(self, number, absolute, n):
        btype, base, size = self.c.blocks[number - 1]
        lo = absolute - base
        hi = min(size, lo + n)
        if 0 <= lo < hi:
            cover = self.cover[number - 1]
            if hi - lo <= len(_ONES):
                cover[lo:hi] = _ONES[:hi - lo]
            else:  # in pieces, so a large mark does not allocate a large temporary
                for p in range(lo, hi, len(_ONES)):
                    q = min(hi, p + len(_ONES))
                    cover[p:q] = _ONES[:q - p]
        else:
            self.cover[number - 1][lo:hi] = b"\x01" * max(0, hi - lo)

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
        # the coverage bytes are 0 or 1, so the bytes reached are the length minus the zeros
        reached = sum(len(b) - b.count(0) for b in self.cover)
        st["bytes"] += sum(size for _, _, size in c.blocks)
        st["bytes reached"] += reached
        st["blocks"] += len(c.blocks)
        st["blocks reached"] += sum(1 for b in self.cover if b.count(0) != len(b))
        return value

    def _plan(self, name, s):
        """What struct() does for each member of `s`, worked out once: the (stats key, count) of the members it can
        count without looking at them, the others to hand to `value`, and one row per member for a keeping walk."""
        keep = self.keep
        plan = self._plans.get(name)
        if plan is not None and (plan[3] is not None or not keep):
            return plan
        size, members = s["size"], s["members"]
        counts, generic, rows = collections.Counter(), [], []
        for m in members:
            if m["hash"] == ELEMENT:
                continue
            t, off = m["type"], m["offset"]
            key = _TYPE_KEY.get(t) or f"type {t:02x}"
            spec = _UNPACK.get(t)
            if spec is not None and 0 <= off and off + spec[0] <= size:
                # inside the struct, which struct() has already checked against its block: no bounds check needed
                counts[key] += 1
            else:
                generic.append((m, off))
                spec = None
            if keep:
                rows.append((f"{m['hash']:08x}", key, m, off, spec))
        plan = (members, tuple(counts.items()), tuple(generic), tuple(rows) if keep else None)
        if name in self._plans or len(self._plans) < _CACHE_MAX:
            self._plans[name] = plan
        return plan

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
            limit = self._limit if self._limit is not None else self._block_end(at)
        if limit is None or at < 0 or at + s["size"] > limit:
            st["struct past block end"] += 1
            return None
        if number:
            self.mark(number, at, s["size"])
        seen = self.seen
        key = at << 32 | name   # a struct name is a u32 here, so this is the pair (name, at) as one int
        if key in seen or depth > MAX_DEPTH:
            return None
        seen.add(key)
        st["structs"] += 1
        out = {} if self.keep else None
        prev_limit = self._limit
        self._limit = limit
        try:
            if self._plain:
                members, counts, generic, rows = self._plan(name, s)
                value = self.value
                if not self.keep:
                    for k, n in counts:
                        st[k] += n
                    for m, off in generic:
                        value(m, members, at + off, depth)
                else:
                    blob = c.blob
                    for hexname, k, m, off, spec in rows:
                        if spec is None:
                            out[hexname] = value(m, members, at + off, depth)
                        else:
                            st[k] += 1
                            v = spec[1](blob, at + off)
                            if spec[3] is not None:
                                out[hexname] = format(v[0], spec[3])
                            else:
                                out[hexname] = v[0] if spec[2] == 1 else list(v)
            else:
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
        size = self.c.size_of(el)
        if self._plain and count > 0 and size:
            return self._run(el, members, range(base, base + count * size, size), depth + 1, True)
        return [self.value(el, members, base + i * size, depth + 1) for i in range(count)]

    def _run(self, m, members, positions, depth, collect, shift=0):
        """`value(m, members, p + shift, depth)` for each p of `positions`, in order, with the counts added in one go.

        `positions` is a non-empty ascending range or list. Returns the list of values when `collect`, else None.
        Only a plain walker calls this (see the class doc).
        """
        t = m["type"]
        n = len(positions)
        st = self.stats
        spec = _UNPACK.get(t)
        limit = self._limit
        if spec is not None and limit is not None and positions[0] + shift >= 0 \
                and positions[-1] + shift + spec[0] <= limit:
            st[_TYPE_KEY.get(t) or f"type {t:02x}"] += n    # all of them are inside the block
            if not self.keep:
                return [None] * n if collect else None
            blob, unpack, width, fmt = self.c.blob, spec[1], spec[2], spec[3]
            if fmt is not None:
                return [format(unpack(blob, p + shift)[0], fmt) for p in positions]
            if width == 1:
                return [unpack(blob, p + shift)[0] for p in positions]
            return [list(unpack(blob, p + shift)) for p in positions]
        if t == 0x05:
            st["type 05"] += n
            struct_, ref, d = self.struct, m["ref"], depth + 1
            if collect:
                return [struct_(ref, p + shift, d) for p in positions]
            for p in positions:
                struct_(ref, p + shift, d)
            return None
        value = self.value
        if collect:
            return [value(m, members, p + shift, depth) for p in positions]
        for p in positions:
            value(m, members, p + shift, depth)
        return None

    def value(self, m, members, at, depth):
        c, st = self.c, self.stats
        t = m["type"]
        st[_TYPE_KEY.get(t) or f"type {t:02x}"] += 1
        need = m["ref"] if t == 0x40 else _READ_SIZE.get(t)
        if need is not None:
            end = self._limit if self._limit is not None else self._block_end(at)
            if end is None or at < 0 or at + need > end:
                st["member past block end"] += 1
                return None
        spec = _UNPACK.get(t)
        if spec is not None:
            v = spec[1](c.blob, at)
            if not self.keep:
                return None
            if spec[3] is not None:
                return format(v[0], spec[3])
            return v[0] if spec[2] == 1 else list(v)
        if t in ENUMS:
            return self.enum(m, t, at)
        if t == 0x05:
            return self.struct(m["ref"], at, depth + 1)
        blob = c.blob
        if t == 0x07:
            number, _ = c.pointer(at)
            need = 1
            if 1 <= number <= len(c.blocks):
                s = c.structs.get(c.blocks[number - 1][0])
                if s is not None:
                    need = max(1, s["size"])
            r = c.resolve(at, need, st, "struct pointer")
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
            r = c.resolve(at, 1, st, "string pointer")
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
        v = _ENUM_UNPACK[t](c.blob, at)[0]
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
                v &= 0xFFFFFFFF  # 32-bit enum: LENM lists both negative values and values >= 2**31
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
        size = c.size_of(el)
        if not size:
            st[f"array of undecoded type {el['type']:02x}"] += 1
            return None
        if t in (0x50, 0x53):
            st["fixed arrays"] += 1
            n = m["ref"] & 0xFFFF
            end = self._limit if self._limit is not None else self._block_end(at)
            if end is None or at < 0 or at + size * n > end:
                st["fixed array out of bounds"] += 1
                return None
            return self.elements(el, members, at, n, depth)
        if t == 0x54:
            end = self._limit if self._limit is not None else self._block_end(at)
            if end is None or at < 0 or at + 8 > end:
                st["inline array out of bounds"] += 1
                return None
            capacity, count = _UU(blob, at)
            st["inline arrays"] += 1
            st["count <= capacity" if count <= capacity else "count > capacity"] += 1
            n = min(count, capacity)
            if at + 8 + size * n > end:
                st["inline array out of bounds"] += 1
                return None
            return self.elements(el, members, at + 8, n, depth)
        if t == 0x5C:
            count, capacity = _UU(blob, at)
            where = at + 8
        else:
            count, capacity = _HH(blob, at + 8)
            where = at
        kind, empty, matches, differs = _ARRAY_KEYS[t]
        st["count <= capacity" if count <= capacity else "count > capacity"] += 1
        if count == 0:
            st[empty] += 1
            return []
        r = c.resolve(where, size * count, st, kind)
        if r is None:
            return None
        btype, absolute, number = r
        want = el["ref"] if el["type"] == 0x05 else el["type"]
        st[matches if btype == want else differs] += 1
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
        layout = c.slot(key, val)
        if layout is None:
            st[f"map of undecoded types {key['type']:02x}/{val['type']:02x}"] += 1
            return None
        voff, slot = layout
        kind = f"map {t:02x}"
        pairs = []
        if t == 0x71:
            flag = _U64(blob, at)[0]
            count, capacity = _HH(blob, at + 16)
            st["map 71 first word 1" if flag == 1 else "map 71 first word not 1"] += 1
            st["count <= capacity" if count <= capacity else "count > capacity"] += 1
            if count == 0:
                st[f"{kind} empty"] += 1
                return {} if self.keep else None
            r = c.resolve(at + 8, (count - 1) * slot + voff + c.size_of(val), st, kind)
            if r is None:
                return None
            _, absolute, number = r
            self.mark(number, absolute, count * slot)
            ks = c.size_of(key)
            code = _KEY_INT.get(ks)
            if code is not None and count > 8:
                # a key's reversed bytes order like its little-endian value: read them all in one pass
                last = absolute + (count - 1) * slot
                ints = [k for (k,) in struct.iter_unpack(f"<{code}{slot - ks}x", memoryview(blob)[absolute:last])]
                ints.append(int.from_bytes(blob[last:last + ks], "little"))
                ascending = ints == sorted(ints)
            else:
                keys = [blob[absolute + i * slot:absolute + i * slot + ks][::-1] for i in range(count)]
                ascending = keys == sorted(keys)
            st[f"{kind} keys ascending" if ascending else f"{kind} keys not ascending"] += 1
            slots = range(absolute, absolute + count * slot, slot)
        elif t == 0x75:
            groups, count = _UU(blob, at)
            if count == 0:
                st[f"{kind} empty"] += 1
                return {} if self.keep else None
            group = 16 + 16 * slot
            r = c.resolve(at + 8, groups * group, st, kind)
            if r is None:
                return None
            _, absolute, number = r
            self.mark(number, absolute, groups * group)
            power = groups > 0 and (groups & (groups - 1) == 0)
            st[f"{kind} groups a power of two" if power else f"{kind} groups not a power of two"] += 1
            slots = array.array("q")    # 8 bytes a slot, where a list of ints takes 40
            for base in range(absolute, absolute + groups * group, group):
                used = blob[base:base + 16].translate(_CTRL_USED)   # 1 for a control byte with bit 7 clear
                if used != _ZERO16:
                    slots.extend(itertools.compress(range(base + 16, base + 16 + 16 * slot, slot), used))
            st[f"{kind} used slots == count" if len(slots) == count else f"{kind} used slots != count"] += 1
        else:
            capacity, count = _UU(blob, at)
            if count == 0:
                st[f"{kind} empty"] += 1
                return {} if self.keep else None
            r = c.resolve(at + 8, capacity * slot, st, kind)
            if r is None:
                return None
            _, absolute, number = r
            self.mark(number, absolute, capacity * slot)
            ks = c.size_of(key)
            slots = array.array("q")
            if capacity <= 16:
                zero = bytes(ks)
                for at_ in range(absolute, absolute + capacity * slot, slot):
                    if blob[at_:at_ + ks] != zero:
                        slots.append(at_)
            else:
                # a slot is in use when any byte of its key is not zero: OR the key's byte columns, then pick the slots
                region = memoryview(blob)[absolute:absolute + capacity * slot]
                for lo in range(0, capacity, _SCAN_SLOTS):   # in pieces, so the masks stay small
                    n = min(_SCAN_SLOTS, capacity - lo)
                    piece = region[lo * slot:(lo + n) * slot]
                    used = 0
                    for j in range(ks):
                        used |= int.from_bytes(piece[j::slot], "little")
                    slots.extend(itertools.compress(range(absolute + lo * slot, absolute + (lo + n) * slot, slot),
                                                    used.to_bytes(n, "little")))
            st[f"{kind} used slots == count" if len(slots) == count else f"{kind} used slots != count"] += 1
        st[f"{kind} pairs"] += len(slots)
        _, blk_base, blk_size = c.blocks[number - 1]
        prev_limit = self._limit
        self._limit = blk_base + blk_size
        try:
            if self._plain and slots and key["type"] in _UNPACK:
                # a plain key has no effect but its count, so a piece of keys can be read, then a piece of values
                for lo in range(0, len(slots), _SCAN_SLOTS):
                    piece = slots[lo:lo + _SCAN_SLOTS]
                    ks_ = self._run(key, members, piece, depth + 1, self.keep)
                    vs_ = self._run(val, members, piece, depth + 1, self.keep, voff)
                    if self.keep:
                        pairs.extend([k, v] for k, v in zip(ks_, vs_))
            else:
                for s in slots:
                    k = self.value(key, members, s, depth + 1)
                    v = self.value(val, members, s + voff, depth + 1)
                    if self.keep:
                        pairs.append([k, v])
        finally:
            self._limit = prev_limit
        return pairs if self.keep else None


def measure(blob, stats):
    """Walk one PSIN container from its root and add its counts to `stats`. Returns False when it was skipped."""
    try:
        container = Container(blob)
    except ValueError as e:
        stats["skipped: " + str(e).split(":")[0]] += 1
        return False
    stats["containers"] += 1
    stats["block tables fill the region" if container.fills else "block tables do not fill the region"] += 1
    try:
        Walker(container, stats).run()
    except (struct.error, IndexError) as e:
        stats["walk error"] += 1
        return False
    return True


def collect(src, limit=None):
    stats = collections.Counter()
    seen = 0
    for _name, blob in psin_index.files_in(src):
        if blob[:4] != psin_index.MAGIC:
            continue
        stats["PSIN files"] += 1
        before = stats["bytes reached"], stats["bytes"]
        if measure(blob, stats):
            r, b = stats["bytes reached"] - before[0], stats["bytes"] - before[1]
            stats["containers fully reached" if r == b else "containers partly reached"] += 1
        seen += 1
        if limit and seen >= limit:
            break
    return dict(sorted(stats.items()))


def main():
    ap = argparse.ArgumentParser(description="Decode PSIN instance data through its LSCH schema.")
    ap.add_argument("src", help="PSIN file or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="print the counts as JSON")
    ap.add_argument("--limit", type=psin_index.count, help="stop after this many PSIN files")
    ap.add_argument("--dump", action="store_true", help="print one container's decoded tree (local inspection only)")
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print(f"ERROR: {a.src}: no such file or folder", file=sys.stderr)
        return 2
    if os.path.isfile(a.src):
        # a directory skips a stream without the magic, but a lone file named on the command line is a wrong path
        try:
            with open(a.src, "rb") as f:
                magic = f.read(4)
        except OSError as e:
            print(f"ERROR: {a.src}: {e}", file=sys.stderr)
            return 2
        if magic != psin_index.MAGIC:
            print(f"ERROR: {a.src}: not a PSIN container", file=sys.stderr)
            return 2
    if a.dump:
        if not os.path.isfile(a.src):
            print("ERROR: --dump takes one PSIN file", file=sys.stderr)
            return 2
        try:
            with open(a.src, "rb") as f:
                blob = f.read()
            container = Container(blob)
            stats = collections.Counter()
            tree = Walker(container, stats, keep=True).run()
        except (OSError, ValueError, struct.error, IndexError) as e:
            print(f"ERROR: {a.src}: {e}", file=sys.stderr)
            return 2
        root_type = container.blocks[container.root - 1][0] if 1 <= container.root <= len(container.blocks) else 0
        json.dump({f"{root_type:08x}": tree}, sys.stdout, indent=1, default=str)
        sys.stdout.write("\n")
        return 0
    try:
        stats = collect(a.src, a.limit)
    except OSError as e:
        print(f"ERROR: {a.src}: {e}", file=sys.stderr)
        return 2
    if a.json:
        json.dump(stats, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for k, v in stats.items():
        print(f"{k:48} {v:,}")
    if stats.get("bytes"):
        print(f"{'bytes reached from the root':48} {100 * stats['bytes reached'] / stats['bytes']:.2f}%")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
