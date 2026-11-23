"""The fast tools/psin_index.py parse and decoders equal an inline reference, byte for byte (no game data).

The reference below is the straightforward implementation the fast path replaced: one `struct.unpack_from` per
header and member, a dict per struct, member, enum and value, and the hex copy of the schema made in a second
pass. Every input is built by hand or by a seeded generator, so the tests compare against the reference and pin
no byte of output: they hold the same on every platform.
"""
import json
import os
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import psin_index as P  # noqa: E402


# --- the reference: the implementation before the fast path --------------------------------------------------

def ref_header_count(payload, header, element, count_at, count_fmt):
    if not payload:
        return 0
    elements = 0
    for k in range(1, len(payload) // header + 1):
        elements += struct.unpack_from(count_fmt, payload, header * (k - 1) + count_at)[0]
        if header * k + element * elements == len(payload):
            return k
    return None


def ref_decode_lsch(payload):
    k = ref_header_count(payload, 32, 16, 30, "<H")
    if k is None:
        raise ValueError(f"LSCH: no struct count fills {len(payload)} bytes")
    structs = []
    at = 32 * k
    for i in range(k):
        name, sig, flags, align, _z10, _z12, size, _z28, count = struct.unpack_from(
            "<IIBBHI8xIHH", payload, 32 * i)
        members = []
        for _ in range(count):
            mname, off, mtype, sub, index, ref = struct.unpack_from("<IIBBHI", payload, at)
            at += 16
            members.append({"hash": mname, "offset": off, "type": mtype, "sub": sub,
                            "index": index, "ref": ref})
        structs.append({"hash": name, "signature": sig, "flags": flags, "align_log2": align,
                        "size": size, "members": members})
    return structs


def ref_decode_lenm(payload):
    k = ref_header_count(payload, 24, 16, 16, "<I")
    if k is None:
        raise ValueError(f"LENM: no enum count fills {len(payload)} bytes")
    enums = []
    at = 24 * k
    for i in range(k):
        name, sig = struct.unpack_from("<II", payload, 24 * i)
        count = struct.unpack_from("<I", payload, 24 * i + 16)[0]
        values = []
        for _ in range(count):
            value, vname = struct.unpack_from("<qQ", payload, at)
            at += 16
            values.append({"value": value, "hash": vname})
        enums.append({"hash": name, "signature": sig, "values": values})
    return enums


def ref_check_schema(structs):
    for s in structs:
        for i, m in enumerate(s["members"]):
            if m["hash"] != 0x100 and m["offset"] >= s["size"]:
                return f"struct {s['hash']:08x} member {i} at offset {m['offset']} is past size {s['size']}"
            if m["type"] == 0x5C:
                j = m["index"]
                if j >= len(s["members"]) or s["members"][j]["hash"] != 0x100:
                    return f"struct {s['hash']:08x} array member {i} points at {j}, not an element descriptor"
    return ""


def ref_hexed(structs, enums):
    out_s = [dict(s, hash=f"{s['hash']:08x}", signature=f"{s['signature']:08x}",
                  members=[dict(m, hash=f"{m['hash']:08x}", ref=f"{m['ref']:08x}") for m in s["members"]])
             for s in structs]
    out_e = [dict(e, hash=f"{e['hash']:08x}", signature=f"{e['signature']:08x}",
                  values=[{"value": v["value"], "hash": f"{v['hash']:08x}"} for v in e["values"]])
             for e in enums]
    return {"structs": out_s, "enums": out_e}


def ref_with_schema(record, schema, structs, enums, lsch_error="", lenm_error=""):
    if not schema:
        return record
    entry = ref_hexed(structs or [], enums or [])
    entry["lsch"] = structs is not None
    entry["lenm"] = enums is not None
    reasons = []
    for name, decoded, section_error in (
        ("LSCH", entry["lsch"], lsch_error),
        ("LENM", entry["lenm"], lenm_error),
    ):
        if decoded:
            continue
        if any(s["tag"] == name for s in record["sections"]):
            reasons.append(section_error or f"{name} not decoded")
        elif record["error"]:
            reasons.append(f"no whole {name} section: {record['error']}")
        else:
            reasons.append(f"no {name} section")
    entry["reason"] = "; ".join(dict.fromkeys(reasons))
    record["schema"] = entry
    return record


def ref_parse(blob, schema=False):
    if len(blob) < 16 or blob[:4] != b"PSIN":
        return None
    first = struct.unpack_from(">I", blob, 4)[0]
    record = {
        "magic": "PSIN",
        "size": len(blob),
        "first_section": first,
        "version": struct.unpack_from("<I", blob, 8)[0],
        "table_bytes": struct.unpack_from("<I", blob, 12)[0],
        "sections": [],
        "rsc_version": None,
        "checksum_size": None,
        "checksum": "",
        "structs": None,
        "members": None,
        "enums": None,
        "enum_values": None,
        "schema_error": "",
        "lsns_blocks": None,
        "lsns_records": None,
        "lsns_error": "",
        "complete": False,
        "error": "",
    }
    structs = enums = None
    lsch_seen = lenm_seen = False
    lsch_error = lenm_error = ""
    if first < 16 or first > len(blob):
        record["error"] = f"first section offset {first} is outside the file"
        return ref_with_schema(record, schema, structs, enums)
    if len(blob) - first < 8:
        record["error"] = f"missing or truncated first section header at {first}"
        return ref_with_schema(record, schema, structs, enums)
    offset = first
    while offset + 8 <= len(blob):
        tag = blob[offset:offset + 4]
        size = struct.unpack_from(">I", blob, offset + 4)[0]
        if any(c < 0x20 or c > 0x7E for c in tag):
            record["error"] = f"non-text section tag at {offset}"
            break
        if size < 8 or offset + size > len(blob):
            record["error"] = f"section {tag.decode('latin-1')} at {offset} has size {size}"
            break
        payload_off = offset + 8
        section = {
            "tag": tag.decode("latin-1"),
            "offset": offset,
            "size": size,
            "payload": size - 8,
        }
        if tag == b"LRSC" and size - 8 >= 8 and blob[payload_off:payload_off + 4] == b"0DRP":
            section["rsc_magic"] = "0DRP"
            section["rsc_version"] = struct.unpack_from("<I", blob, payload_off + 4)[0]
            if record["rsc_version"] is None:
                record["rsc_version"] = section["rsc_version"]
        elif tag == b"CHKS" and size - 8 >= 8:
            section["checksum_size"] = struct.unpack_from(">I", blob, payload_off)[0]
            section["checksum"] = blob[payload_off + 4:payload_off + 8].hex()
            if record["checksum_size"] is None:
                record["checksum_size"] = section["checksum_size"]
                record["checksum"] = section["checksum"]
        elif tag == b"LSNS":
            try:
                blocks = P.decode_lsns(blob[payload_off:offset + size])
            except ValueError as e:
                section["lsns_error"] = record["lsns_error"] = str(e)
            else:
                section["blocks"] = [{"kind": f"0x{k:x}", "bytes": b} for k, b in blocks]
                record["lsns_blocks"] = len(blocks)
                record["lsns_records"] = sum(b for _k, b in blocks) // 16
        elif tag == b"LSCH" and not lsch_seen:
            lsch_seen = True
            try:
                structs = ref_decode_lsch(blob[payload_off:offset + size])
            except ValueError as e:
                lsch_error = str(e)
            else:
                record["structs"] = len(structs)
                record["members"] = sum(len(s["members"]) for s in structs)
                lsch_error = ref_check_schema(structs)
        elif tag == b"LENM" and not lenm_seen:
            lenm_seen = True
            try:
                enums = ref_decode_lenm(blob[payload_off:offset + size])
            except ValueError as e:
                lenm_error = str(e)
            else:
                record["enums"] = len(enums)
                record["enum_values"] = sum(len(e["values"]) for e in enums)
        record["sections"].append(section)
        offset += size
    record["schema_error"] = "; ".join(e for e in (lsch_error, lenm_error) if e)
    if not record["error"] and offset != len(blob):
        record["error"] = f"section walk ended at {offset} of {len(blob)}"
    record["complete"] = not record["error"] and offset == len(blob)
    return ref_with_schema(record, schema, structs, enums, lsch_error, lenm_error)


# --- inputs --------------------------------------------------------------------------------------------------

def section(tag, payload=b""):
    return tag + struct.pack(">I", 8 + len(payload)) + payload


def container(*sections, version=22, table_bytes=0):
    head = bytearray(16)
    head[0:4] = b"PSIN"
    struct.pack_into(">I", head, 4, 16)
    struct.pack_into("<I", head, 8, version)
    struct.pack_into("<I", head, 12, table_bytes)
    return bytes(head) + b"".join(sections)


def lsch_payload(rng, structs=None):
    """A random LSCH payload whose header count fits; members mix valid and invalid offsets and array links."""
    out_heads, out_body = [], []
    for _ in range(rng.randint(0, 6) if structs is None else structs):
        n = rng.randint(0, 7)
        size = rng.choice([0, 4, 16, 64])
        out_heads.append(struct.pack("<IIBBHI8sIHH", rng.getrandbits(32), rng.getrandbits(32),
                                     rng.getrandbits(8), rng.getrandbits(4), 0, 0, b"\xaa" * 8, size, 0, n))
        for _i in range(n):
            out_body.append(struct.pack("<IIBBHI", rng.choice([0x100, rng.getrandbits(32)]),
                                        rng.randrange(0, 80), rng.choice([5, 0x5C, 0x21, 0x62]),
                                        rng.getrandbits(8), rng.randrange(0, 9), rng.getrandbits(32)))
    return b"".join(out_heads) + b"".join(out_body)


def clean_lsch_payload(rng):
    """A schema check_schema accepts: offsets inside the size and every array member points at a descriptor."""
    heads, body = [], []
    for _ in range(rng.randint(1, 5)):
        members = [(0x100, 0, 5, 0, 0, rng.getrandbits(32))]
        for _i in range(rng.randint(0, 5)):
            kind = rng.choice([0x21, 0x5C, 0x62])
            members.append((rng.getrandbits(32) | 1, rng.randrange(0, 32), kind, 0, 0, rng.getrandbits(32)))
        heads.append(struct.pack("<IIBBHI8sIHH", rng.getrandbits(32), 1, 0, 3, 0, 0, b"\xaa" * 8, 64, 0, len(members)))
        body.extend(struct.pack("<IIBBHI", *m) for m in members)
    return b"".join(heads) + b"".join(body)


def lenm_payload(rng):
    heads, body = [], []
    for _ in range(rng.randint(0, 5)):
        n = rng.randint(0, 6)
        heads.append(struct.pack("<II8sII", rng.getrandbits(32), rng.getrandbits(32), b"\xee" * 8, n, 0))
        body.extend(struct.pack("<qQ", rng.randrange(-(1 << 40), 1 << 40), rng.getrandbits(32)) for _ in range(n))
    return b"".join(heads) + b"".join(body)


def random_container(rng):
    sections = []
    for _ in range(rng.randint(0, 8)):
        tag = rng.choice([b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS", b"ABCD", b"LSCH", b"LENM"])
        if tag == b"LSCH":
            payload = rng.choice([lsch_payload, clean_lsch_payload])(rng)
        elif tag == b"LENM":
            payload = lenm_payload(rng)
        elif tag == b"LSNS":
            payload = rng.choice([
                struct.pack("<I", 0),
                struct.pack("<I", 2) + struct.pack("<II", 0x10100, 16) + struct.pack("<II", 0x20003, 32) + bytes(48),
                struct.pack("<I", 1) + struct.pack("<II", 0x10100, 15) + bytes(15),
                bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 20))),
            ])
        elif tag == b"LRSC":
            payload = rng.choice([b"0DRP\x7a\0\0\0" + bytes(8), b"0DRP\x01\0\0\0", b"0DRP", b"", b"XXXX" + bytes(8)])
        elif tag == b"CHKS":
            payload = bytes(rng.getrandbits(8) for _ in range(rng.choice([0, 4, 8, 12])))
        else:
            payload = b""
        sections.append(section(tag, payload))
    return container(*sections, version=rng.choice([22, 21, 0]), table_bytes=rng.choice([0, 16]))


def mutate(rng, blob):
    """A damaged copy: truncated, bytes flipped, bytes appended, or a section size rewritten."""
    data = bytearray(blob)
    kind = rng.randrange(5)
    if kind == 0 and len(data) > 17:
        del data[rng.randrange(16, len(data)):]
    elif kind == 1 and data:
        for _ in range(rng.randint(1, 4)):
            data[rng.randrange(len(data))] = rng.getrandbits(8)
    elif kind == 2:
        data += bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 20)))
    elif kind == 3 and len(data) > 30:
        at = rng.randrange(16, len(data) - 4)
        data[at:at + 4] = struct.pack(">I", rng.randrange(0, 600))
    return bytes(data)


def dump(value):
    """A comparison that also sees the key order, which the JSON output carries."""
    return json.dumps(value, sort_keys=False)


class TestParseParity(unittest.TestCase):
    def check(self, blob):
        for schema in (False, True):
            with self.subTest(schema=schema, head=blob[:24].hex()):
                self.assertEqual(dump(P.parse(blob, schema)), dump(ref_parse(blob, schema)))

    def test_seeded_corpus(self):
        rng = random.Random(20260301)
        for _ in range(1500):
            self.check(random_container(rng))

    def test_seeded_damaged_corpus(self):
        rng = random.Random(20260302)
        for _ in range(1500):
            self.check(mutate(rng, random_container(rng)))

    def test_not_psin_and_short_inputs(self):
        for blob in (b"", b"PSI", b"PSIN", b"PSIN" + bytes(11), b"NOPE" + bytes(20)):
            self.check(blob)

    def test_header_variants(self):
        base = container(section(b"LGID"), section(b"CHKS", bytes(8)))
        for first in (0, 15, 16, 17, 24, len(base), len(base) + 1, 0xFFFFFFFF):
            blob = base[:4] + struct.pack(">I", first) + base[8:]
            self.check(blob)
            self.check(blob + bytes(3))

    def test_section_edge_cases(self):
        sections = [
            section(b"LGID"),
            b"AB\x00D" + struct.pack(">I", 8),            # a non-printable tag byte
            b"AB\x7fD" + struct.pack(">I", 8),
            b"AB\xe9D" + struct.pack(">I", 8),
            b"A B~" + struct.pack(">I", 8),               # the printable limits 0x20 and 0x7e
            b"LGID" + struct.pack(">I", 7),               # size below the section header
            b"LGID" + struct.pack(">I", 1 << 31),         # size past the end
            b"LRSC" + struct.pack(">I", 15) + bytes(7),   # LRSC and CHKS payloads of 7 and 8 bytes
            b"LRSC" + struct.pack(">I", 16) + b"0DRP\x01\0\0\0",
            b"CHKS" + struct.pack(">I", 15) + bytes(7),
            b"CHKS" + struct.pack(">I", 16) + bytes(8),
        ]
        for tag_blob in sections:
            self.check(container(section(b"LGID"), tag_blob))
            self.check(container(tag_blob))
            self.check(container(section(b"LENM"), tag_blob, section(b"LSCH")))

    def test_duplicate_sections_keep_the_first(self):
        rng = random.Random(5)
        for _ in range(100):
            a, b = lsch_payload(rng), lsch_payload(rng)
            c, d = lenm_payload(rng), lenm_payload(rng)
            self.check(container(section(b"LENM", c), section(b"LSCH", a), section(b"LSCH", b),
                                 section(b"LENM", d), section(b"CHKS", bytes(8)), section(b"CHKS", bytes(12))))

    def test_first_error_of_each_schema_section_survives(self):
        bad_lsch = bytes(33)                          # 33 bytes fill no header count
        bad_lenm = bytes(25)
        self.check(container(section(b"LENM", bad_lenm), section(b"LSCH", bad_lsch)))
        self.check(container(section(b"LSCH", bad_lsch), section(b"LENM", bad_lenm), section(b"LSCH", bytes(8))))

    def test_clean_schema_has_no_error(self):
        rng = random.Random(11)
        for _ in range(200):
            blob = container(section(b"LSCH", clean_lsch_payload(rng)), section(b"LENM", lenm_payload(rng)))
            self.check(blob)
            self.assertEqual(P.parse(blob)["schema_error"], "")

    def test_records_do_not_share_state(self):
        blob = container(section(b"LSCH", clean_lsch_payload(random.Random(3))), section(b"LENM", lenm_payload(random.Random(4))))
        first, second = P.parse(blob, True), P.parse(blob, True)
        first["schema"]["structs"][0]["members"].clear()
        first["sections"].clear()
        self.assertEqual(dump(second), dump(ref_parse(blob, True)))
        self.assertEqual(dump(P.parse(blob, True)), dump(ref_parse(blob, True)))


class TestDecoderParity(unittest.TestCase):
    def test_decode_lsch_and_lenm(self):
        rng = random.Random(20260303)
        for _ in range(1500):
            for payload_fn, fast, ref in ((lsch_payload, P.decode_lsch, ref_decode_lsch),
                                          (clean_lsch_payload, P.decode_lsch, ref_decode_lsch),
                                          (lenm_payload, P.decode_lenm, ref_decode_lenm)):
                payload = payload_fn(rng)
                if rng.random() < 0.3:
                    payload = mutate(rng, payload)
                for view in (payload, memoryview(payload), bytearray(payload)):
                    try:
                        want = ref(bytes(view))
                    except ValueError as e:
                        with self.assertRaises(ValueError) as caught:
                            fast(view)
                        self.assertEqual(str(caught.exception), str(e))
                    else:
                        self.assertEqual(dump(fast(view)), dump(want))

    def test_empty_payloads(self):
        self.assertEqual(P.decode_lsch(b""), [])
        self.assertEqual(P.decode_lenm(b""), [])
        self.assertEqual(P.lsch_summary(b""), (0, 0, ""))
        self.assertEqual(P.lenm_summary(b""), (0, 0))
        self.assertEqual(P.lsch_hexed(b""), ([], 0, ""))
        self.assertEqual(P.lenm_hexed(b""), ([], 0))

    def test_summaries_equal_the_decoded_counts(self):
        rng = random.Random(20260304)
        for _ in range(1500):
            payload = rng.choice([lsch_payload, clean_lsch_payload])(rng)
            if rng.random() < 0.3:
                payload = mutate(rng, payload)
            try:
                structs = ref_decode_lsch(payload)
            except ValueError as e:
                for fn in (P.lsch_summary, P.lsch_hexed):
                    with self.assertRaises(ValueError) as caught:
                        fn(payload)
                    self.assertEqual(str(caught.exception), str(e))
                continue
            members = sum(len(s["members"]) for s in structs)
            problem = ref_check_schema(structs)
            self.assertEqual(P.lsch_summary(payload), (len(structs), members, problem))
            hexed, n_members, hexed_problem = P.lsch_hexed(payload)
            self.assertEqual((n_members, hexed_problem), (members, problem))
            self.assertEqual(dump(hexed), dump(ref_hexed(structs, [])["structs"]))

            payload = lenm_payload(rng)
            if rng.random() < 0.3:
                payload = mutate(rng, payload)
            try:
                enums = ref_decode_lenm(payload)
            except ValueError as e:
                for fn in (P.lenm_summary, P.lenm_hexed):
                    with self.assertRaises(ValueError) as caught:
                        fn(payload)
                    self.assertEqual(str(caught.exception), str(e))
                continue
            values = sum(len(x["values"]) for x in enums)
            self.assertEqual(P.lenm_summary(payload), (len(enums), values))
            hexed, n_values = P.lenm_hexed(payload)
            self.assertEqual(n_values, values)
            self.assertEqual(dump(hexed), dump(ref_hexed([], enums)["enums"]))

    def test_check_schema_problem_order(self):
        """The first problem in struct order, member order, wins; an offset problem precedes an array problem."""
        def struct_bytes(size, members):
            head = struct.pack("<IIBBHI8sIHH", 0xABCDEF01, 0, 0, 3, 0, 0, b"\xaa" * 8, size, 0, len(members))
            return head, b"".join(struct.pack("<IIBBHI", *m) for m in members)

        cases = [
            [(0x100, 0, 5, 0, 0, 0), (7, 4, 0x5C, 0, 0, 0)],             # array points at a non-descriptor
            [(7, 4, 0x5C, 0, 3, 0)],                                     # array index past the members
            [(7, 99, 0x21, 0, 0, 0), (7, 0, 0x5C, 0, 5, 0)],             # offset problem first
            [(0x100, 99, 5, 0, 0, 0), (7, 8, 0x5C, 0, 0, 0)],            # a descriptor may sit past the size
            [(7, 0, 0x5C, 0, 1, 0), (0x100, 0, 5, 0, 0, 0)],             # the descriptor sits after the member
        ]
        for members in cases:
            head, body = struct_bytes(16, members)
            payload = head + body
            structs = ref_decode_lsch(payload)
            want = ref_check_schema(structs)
            self.assertEqual(P.check_schema(P.decode_lsch(payload)), want)
            self.assertEqual(P.lsch_summary(payload)[2], want)
            self.assertEqual(P.lsch_hexed(payload)[2], want)
            self.assertEqual(dump(P.parse(container(section(b"LSCH", payload)), True)),
                             dump(ref_parse(container(section(b"LSCH", payload)), True)))


class TestLargeInputs(unittest.TestCase):
    """Dense and sparse payloads, where a count loop or an eager decode would show."""

    def check(self, blob):
        for schema in (False, True):
            self.assertEqual(dump(P.parse(blob, schema)), dump(ref_parse(blob, schema)))

    def test_one_struct_with_many_members(self):
        members = b"".join(struct.pack("<IIBBHI", 0x100 if j == 0 else j + 1, 0, 0x21, 0, 0, j) for j in range(3000))
        head = struct.pack("<IIBBHI8sIHH", 1, 2, 0, 3, 0, 0, b"\xaa" * 8, 64, 0, 3000)
        self.check(container(section(b"LSCH", head + members)))

    def test_many_structs_without_members(self):
        heads = b"".join(struct.pack("<IIBBHI8sIHH", j, 0, 0, 0, 0, 0, b"\xaa" * 8, 0, 0, 0) for j in range(2000))
        self.check(container(section(b"LSCH", heads)))

    def test_zero_bytes_that_fill_no_count(self):
        for size in (31, 33, 1000, 40000):
            self.check(container(section(b"LSCH", bytes(size)), section(b"LENM", bytes(size))))

    def test_one_enum_with_many_values(self):
        head = struct.pack("<II8sII", 1, 2, b"\xee" * 8, 2500, 0)
        body = b"".join(struct.pack("<qQ", j - 7, j) for j in range(2500))
        self.check(container(section(b"LENM", head + body)))

    def test_many_small_sections(self):
        self.check(container(*[section(b"LGID") for _ in range(5000)]))
        self.check(container(*[section(b"LGID") for _ in range(5000)]) + b"\0\0\0")


if __name__ == "__main__":
    unittest.main()
