"""psin_psch.py must refuse malformed PMAP/PSCH/PSIG/STNS sections, each for its own reason, without game data.

    python -I tests/test_psin_psch_rejects.py

The inputs are synthetic. Most cases damage an input the same test class shows to be accepted; the rest are the
smallest section that reaches the check. The container cases use the builder in tests/test_psin_psch.py.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import psin_psch  # noqa: E402
from test_psin_psch import build_synthetic_psch_psin  # noqa: E402

FIRST = 64                                  # the data area is 0x10..0x40
ROWS = ((0x11, 16, 0, 32), (0x22, 48, 0, 16))
BLOB = b"PSIN" + struct.pack(">I", FIRST) + b"pppppppp"
ELEMENT_HASH, ARRAY, ENUM, MAP = 0x100, 0x0D, 0x0E, 0x10    # as docs/psin.md gives them, not read from the tool


def pmap(rows=ROWS, root=1, z0=0, count=None, fill=b"pppppp", flags=0):
    head = struct.pack(">IHHH6s", root, flags, z0, len(rows) if count is None else count, fill)
    return head + b"".join(struct.pack(">4I", *row) for row in rows)


def struct_entry(members=(), size=16, count=None):
    head = struct.pack(">BBHII", 0, 0, len(members) if count is None else count, size, 0)
    return head + b"".join(struct.pack(">IBBHI", *m) for m in members)


def enum_entry(values=(), count=None):
    head = struct.pack(">BBH", 1, 0, len(values) if count is None else count)
    return head + b"".join(struct.pack(">IIq", *v) for v in values)


def psch(entries, offsets=None, count=None):
    """A PSCH section; `offsets` replaces the directory's section offsets, `count` the declared entry count."""
    at, real = 12 + 8 * len(entries), []
    for _, body in entries:
        real.append(at)
        at += len(body)
    directory = b"".join(struct.pack(">II", name, off)
                         for (name, _), off in zip(entries, real if offsets is None else offsets))
    body = struct.pack(">I", len(entries) if count is None else count) + directory + b"".join(b for _, b in entries)
    return b"PSCH" + struct.pack(">I", 8 + len(body)) + body


def without(blob, tag):
    """`blob` with the one section called `tag` renamed, so the container lacks it."""
    if blob.count(tag) != 1:
        raise AssertionError(f"{tag!r} occurs {blob.count(tag)} times in the fixture")
    return blob.replace(tag, b"XXXX")


class RejectCase(unittest.TestCase):
    def rejects(self, cases, call):
        for label, (reason, *args) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, reason):
                call(*args)


class PmapTests(RejectCase):
    def test_accepts_an_exact_and_a_16_byte_padded_end(self):
        exact = psin_psch.decode_pmap(pmap(), BLOB, FIRST)
        self.assertTrue(exact["exact_end"])
        self.assertTrue(exact["reserved_zero"])
        self.assertEqual([(b["id"], b["hash"], b["offset"], b["length"]) for b in exact["blocks"]],
                         [(1, 0x11, 16, 32), (2, 0x22, 48, 16)])
        for length in (1, 2, 15):
            with self.subTest(length=length):
                padded = psin_psch.decode_pmap(pmap((ROWS[0], (0x22, 48, 0, length))), BLOB, FIRST)
                self.assertFalse(padded["exact_end"])
        self.assertEqual(psin_psch.decode_pmap(pmap(root=2), BLOB, FIRST)["root_block_id"], 2)

    def test_flags_and_reserved_fields_are_reported_not_refused(self):
        self.assertEqual(psin_psch.decode_pmap(pmap(flags=0xFFFE), BLOB, FIRST)["flags"], 0xFFFE)
        for payload in (pmap(z0=1), pmap(((0x11, 16, 1, 32), ROWS[1]))):
            self.assertFalse(psin_psch.decode_pmap(payload, BLOB, FIRST)["reserved_zero"])

    def test_rejects(self):
        self.rejects({
            "short header": ("shorter than 16-byte", pmap()[:15], BLOB),
            "short blob": ("header fill", pmap(), BLOB[:15]),
            "container fill": ("header fill", pmap(), BLOB[:15] + b"\0"),
            "PMAP fill": ("lacks 6-byte", pmap(fill=b"ppppp\0"), BLOB),
            "truncated row": ("PMAP size", pmap()[:-1], BLOB),
            "trailing byte": ("PMAP size", pmap() + b"\0", BLOB),
            "count past the rows": ("PMAP size", pmap(count=3), BLOB),
            "no blocks": ("root_block_id", pmap(()), BLOB),
            "root 0": ("root_block_id 0 outside", pmap(root=0), BLOB),
            "root past the last block": (r"root_block_id 3 outside 1\.\.2", pmap(root=3), BLOB),
            "block inside the header": ("block 1 .* outside data area", pmap(((0x11, 15, 0, 32), ROWS[1])), BLOB),
            "block one byte past the data": ("block 2 .* outside data area", pmap((ROWS[0], (0x22, 48, 0, 17))), BLOB),
            "offset + length over 32 bits": ("block 2 .* outside data area",
                                             pmap((ROWS[0], (0x22, 48, 0, 0xFFFFFFFF))), BLOB),
            "first block not at 0x10": ("first PMAP block", pmap(((0x11, 32, 0, 16), ROWS[1])), BLOB),
            "gap of a whole row at the end": ("does not end at first_section", pmap((ROWS[0], (0x22, 48, 0, 0))), BLOB),
        }, lambda payload, blob: psin_psch.decode_pmap(payload, blob, FIRST))


class PschTests(RejectCase):
    GOOD = ((0x11, struct_entry([(0xA1, 7, 0, 0, 0)])), (0x22, enum_entry([(0xB1, 0, -2), (0xB2, 0, 3)])))

    def test_accepts_structs_enums_and_an_empty_table(self):
        out = psin_psch.decode_psch(psch(self.GOOD))
        self.assertEqual([(s["hash"], s["size"], len(s["members"])) for s in out["structs"]], [(0x11, 16, 1)])
        self.assertEqual([(e["hash"], [(v["hash"], v["value"]) for v in e["values"]]) for e in out["enums"]],
                         [(0x22, [(0xB1, -2), (0xB2, 3)])])
        self.assertEqual(psin_psch.decode_psch(psch(()))["entry_count"], 0)

    def test_rejects(self):
        good = psch(self.GOOD)
        self.rejects({
            "short section": ("not a PSCH section", good[:11]),
            "other tag": ("not a PSCH section", b"LSCH" + good[4:]),
            "directory past the end": ("runs past section end", psch((), count=1)),
            "huge entry count": ("runs past section end", psch(self.GOOD, count=0xFFFFFFFF)),
            "first entry not at the directory end": ("!= directory end 28", psch(self.GOOD, offsets=(32, 52))),
            "equal offsets": ("strictly ascending", psch(self.GOOD, offsets=(28, 28))),
            "descending offsets": ("strictly ascending", psch(self.GOOD, offsets=(28, 20))),
            "offset past the section": ("strictly ascending", psch(self.GOOD, offsets=(28, len(good) + 1))),
            "entry of 3 bytes": ("entry 0 shorter than 4", psch(((0x11, b"\0\0\0"),))),
            "struct count past its members": ("struct 00000011 size", psch(((0x11, struct_entry(count=1)),))),
            "struct with a trailing byte": ("struct 00000011 size", psch(((0x11, struct_entry() + b"\0"),))),
            "struct header only half there": ("struct 00000011 size", psch(((0x11, struct_entry()[:8]),))),
            "enum count past its values": ("enum 00000022 size", psch(((0x22, enum_entry(count=1)),))),
            "enum with a trailing byte": ("enum 00000022 size", psch(((0x22, enum_entry() + b"\0"),))),
            "unknown entry kind": ("unknown PSCH entry kind 2 for 00000033", psch(((0x33, b"\2\0\0\0"),))),
        }, psin_psch.decode_psch)


class PsigStnsTests(RejectCase):
    def test_psig(self):
        self.assertEqual(psin_psch.decode_psig(struct.pack(">2I", 1, 2), 2), [1, 2])
        self.assertEqual(psin_psch.decode_psig(b"", 0), [])
        self.rejects({
            "one signature short": ("PSIG payload 4 ", bytes(4), 2),
            "one byte over": ("PSIG payload 9 ", bytes(9), 2),
            "signatures for no entries": ("PSIG payload 4 ", bytes(4), 0),
        }, psin_psch.decode_psig)

    def test_stns(self):
        record = bytes(range(16))
        good = struct.pack("<3I", 1, 0x101, 16) + record + bytes(4)
        out = psin_psch.decode_stns(good)
        self.assertEqual(out["blocks"], [{"kind": 0x101, "bytes": 16, "records": 1}])
        self.assertEqual((out["raw_blocks"], out["tail_pad"]), (record, 4))
        two = psin_psch.decode_stns(struct.pack("<5I", 2, 0x101, 16, 0x102, 240) + record * 16 + bytes(12))
        self.assertEqual([(b["kind"], b["records"]) for b in two["blocks"]], [(0x101, 1), (0x102, 15)])
        self.assertEqual((two["records"], two["raw_blocks"], two["tail_pad"]), (16, record * 16, 12))
        self.rejects({
            "short count": ("shorter than 4-byte", b"\0\0\0"),
            "table past the end": ("runs past section end", struct.pack("<3I", 2, 0x101, 16)),
            "huge block count": ("runs past section end", struct.pack("<I", 0xFFFFFFFF)),
            "partial record": ("size 15 is not a multiple of 16", struct.pack("<3I", 1, 0x101, 15) + bytes(20)),
            "block past the end": ("length/padding", struct.pack("<3I", 1, 0x101, 32) + record + bytes(4)),
            "padding missing": ("length/padding", good[:-4]),
            "empty table without padding": ("length/padding", struct.pack("<I", 0)),
            "a row too much padding": ("length/padding", good + bytes(16)),
            "nonzero padding": ("length/padding", good[:-1] + b"\1"),
        }, psin_psch.decode_stns)


class SchemaTests(unittest.TestCase):
    ELEMENT = (ELEMENT_HASH, 7, 0, 0, 0)

    def errors(self, members, size=16, rows=((0x11, 16, 0, 48),)):
        blocks = psin_psch.decode_pmap(pmap(rows), BLOB, FIRST)
        table = psin_psch.decode_psch(psch(((0x11, struct_entry(members, size)), (0x22, enum_entry()))))
        return psin_psch.check_psch_schema(blocks, table)

    def test_accepts_resolved_descriptors(self):
        self.assertEqual(self.errors([
            self.ELEMENT,
            (0xA1, ARRAY, 0, 0, 0),
            (0xA2, ENUM, 0, 8, 0x22),
            (0xA3, MAP, 0, 15, 0),
        ]), [])

    def test_an_element_descriptor_needs_no_offset_inside_the_struct(self):
        self.assertEqual(self.errors([(ELEMENT_HASH, 7, 0, 99, 0)]), [])

    def test_root_block_without_a_struct(self):
        self.assertEqual(self.errors([], rows=((0x99, 16, 0, 48),)),
                         ["PMAP root block hash 00000099 missing from PSCH"])

    def test_block_length_not_a_multiple_of_the_struct_size(self):
        self.assertEqual(self.errors([], size=32), ["PMAP block 1 length 48 not divisible by struct size 32"])

    def test_a_zero_size_struct_is_not_a_divisor(self):
        self.assertEqual(self.errors([], size=0), [])

    def test_each_bad_member_is_named(self):
        for member, want in (
            ((0xA1, 7, 0, 16, 0), ["member 1 offset 16 >= size 16"]),
            ((0xA1, ARRAY, 0, 0, 2), ["array member 1 bad descriptor 2"]),
            ((0xA1, ARRAY, 0, 0, 1), ["array member 1 bad descriptor 1"]),      # itself: not an element
            ((0xA1, ENUM, 0, 0, 0x11), ["enum member 1 missing enum 00000011"]),  # a struct's hash
            ((0xA1, MAP, 0, 0, 2), ["map member 1 bad key descriptor 2"]),
            ((0xA1, MAP, 0, 0, 2 << 16), ["map member 1 bad value descriptor 2"]),
            ((0xA1, MAP, 0, 0, 1 << 16 | 1),
             ["map member 1 bad key descriptor 1", "map member 1 bad value descriptor 1"]),
        ):
            with self.subTest(member=member):
                self.assertEqual(self.errors([self.ELEMENT, member]), [f"struct 00000011 {w}" for w in want])


class ContainerTests(RejectCase):
    def test_rejects(self):
        blob = build_synthetic_psch_psin()
        report = psin_psch.parse_psch_container(blob)
        self.assertTrue(report["valid"])
        self.assertEqual(report["member_type_counts"],
                         {"0x07": 2, "0x0b": 1, "0x0c": 1, "0x0d": 1, "0x0e": 1, "0x10": 1})
        cases = {
            "not PSIN": ("not a PSIN container", b"XXXX" + blob[4:]),
            "truncated": ("incomplete PSIN container", blob[:-1]),
        }
        for tag in ("PMAP", "PSCH", "PSIG", "STNS"):
            cases[f"no {tag}"] = (f"missing {tag} section", without(blob, tag.encode("ascii")))
        self.rejects(cases, psin_psch.parse_psch_container)

    def test_a_complete_container_with_pmap_or_psch_is_recognised(self):
        blob = build_synthetic_psch_psin()
        self.assertTrue(psin_psch.is_psch_container(blob))
        self.assertTrue(psin_psch.is_psch_container(without(blob, b"PSCH")))     # either tag is enough
        self.assertTrue(psin_psch.is_psch_container(without(blob, b"PMAP")))
        self.assertFalse(psin_psch.is_psch_container(without(without(blob, b"PSCH"), b"PMAP")))
        self.assertFalse(psin_psch.is_psch_container(blob[:-1]))
        self.assertFalse(psin_psch.is_psch_container(b""))

    def test_stdn_must_be_the_stns_block_data_prefix(self):
        blob = bytearray(build_synthetic_psch_psin())
        blob[blob.index(b"STDN") + 8] ^= 1
        report = psin_psch.parse_psch_container(bytes(blob))
        self.assertFalse(report["stdn_match"])
        self.assertEqual(report["errors"], ["STDN payload does not match STNS block data prefix"])
        absent = psin_psch.parse_psch_container(build_synthetic_psch_psin(with_stdn=False))
        self.assertTrue(absent["valid"])
        self.assertIsNone(absent["stdn_match"])


if __name__ == "__main__":
    unittest.main()
