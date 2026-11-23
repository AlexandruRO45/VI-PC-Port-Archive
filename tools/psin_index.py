"""Index the PSIN metadata container used by the game's data and script archives.

PSIN is a small metadata container. It appears at the start of carved zlib streams in `data.rpf` and
`script.rpf`; it is not a script program and it is not a model or texture resource. Every measured
file uses the same header:

    +0   50 53 49 4e          "PSIN"
    +4   u32 big-endian       absolute offset of the first section (LSCH)
    +8   u32 little-endian    version (22 on every measured file)
    +12  u32 little-endian    byte count of the hash table that follows the header

The section table starts at the offset in `+4` and runs to the end of the file. Every section is:

    +0   4-byte ASCII tag
    +4   u32 big-endian       total size, including this 8-byte header

Sections are contiguous: the next section starts at `offset + size`, and the last one ends at EOF.
The measured tags are `LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC` and `CHKS`. `LGID` is an empty section
(8 bytes). `LRSC` starts with a `0DRP` page directory header. `CHKS` carries the file size and a
checksum.

`LSNS` is a block table followed by the blocks (measured over every `LSNS` section in the package; docs/psin.md):

    +0   u32 little-endian    block count n (0 for the empty 12-byte section)
    +4   n x (u32 kind, u32 byte count)
         then the n blocks back to back, each a whole number of 16-byte records

The byte counts fill the payload exactly. The kind's high byte is 1 or 2 (records read as 4- or 8-byte words);
its low byte varies. The records themselves are uniformly random bytes that match no tested name hash; this tool
reads the layout, not the contents.

`LSCH` is a type schema and `LENM` its enum table (measured over all 18,640 PSIN streams in the archives of package
1.0.2.1; see docs/psin.md). Both store all headers first and then all elements, and neither stores the
header count: it is the one count k for which the headers' element counts fill the payload exactly.

    LSCH struct header, 32 bytes        LSCH member, 16 bytes        LENM enum header, 24 bytes
    +0  u32 name hash                    +0  u32 name hash            +0  u32 name hash
    +4  u32 layout signature             +4  u32 byte offset          +4  u32 layout signature
    +8  u8  flags                        +8  u8  type code            +8  8 fill bytes (0xee)
    +9  u8  log2 alignment               +9  u8  sub-code             +16 u32 value count
    +10 u16 0, +12 u32 0                 +10 u16 element index        +20 u32 0
    +16 8 fill bytes (0xaa)              +12 u32 reference hash      LENM value, 16 bytes
    +24 u32 byte size                                                  +0  i64 value
    +28 u16 0, +30 u16 member count                                    +8  u64 name hash (high half 0)

Names are case-sensitive joaat. A member whose name hash is 0x100 is the element descriptor of an array member
(type 0x5c), whose element index points at it; a struct member (type 5) references a struct name hash, and the
enum members (0x60, 0x62, 0x65, ...) reference an enum name hash in `LENM`.

On the package measured here the `script.rpf` carve contains 671 streams that start with `PSIN`:
670 are complete and one is the first 64 KiB of a larger record (its first-section offset is beyond
the carved block). A complete record in `data.rpf` is validated by the project's existing
`ArchiveReader` path, not by this tool. The tool deliberately indexes containers; it does not assign
meaning to the CHKS payload or to the bytes between the header and the first section.

Usage:
    python -I psin_index.py <file-or-dir> [--json] [--schema] [--limit N]

Text output prints one line per PSIN file and a summary; `--json` writes `{files, summary}`. `--schema` adds the
decoded LSCH structs and LENM enums (hashes as hex) to each JSON record, or prints them in text mode.
"""
import argparse
import itertools
import json
import os
import struct
import sys

MAGIC = b"PSIN"
HEADER = 16
SECTION_HEADER = 8
STRUCT_HEADER = 32
MEMBER = 16
ENUM_HEADER = 24
ENUM_VALUE = 16
ARRAY_ELEMENT = 0x100   # member name hash of an array's element descriptor
ARRAY_TYPE = 0x5C

_SECTION = struct.Struct(">4sI").unpack_from
_FIRST = struct.Struct(">4xI").unpack_from
_VERSION_TABLE = struct.Struct("<8xII").unpack_from
_U32_BE = struct.Struct(">I").unpack_from
_U32_LE = struct.Struct("<I").unpack_from
# The six measured tags, so the walk skips the printable-ASCII check and the decode for them.
_KNOWN_TAGS = {b"LSCH": ("LSCH", 1), b"LENM": ("LENM", 2), b"LSNS": ("LSNS", 3),
               b"LGID": ("LGID", 0), b"LRSC": ("LRSC", 4), b"CHKS": ("CHKS", 5)}
_PRINTABLE = bytes(range(0x20, 0x7F))   # bytes.translate(None, _PRINTABLE) is empty for a printable-ASCII tag
_LSCH_COUNTS = struct.Struct("<30xH")   # the member count of each 32-byte struct header
_LENM_COUNTS = struct.Struct("<16xI4x")   # the value count of each 24-byte enum header
_LSCH_SUMMARY = struct.Struct("<I20xI2xH")   # name hash, byte size, member count
_LSCH_HEAD = struct.Struct("<IIBBHI8xIHH")
_LSCH_MEMBER = struct.Struct("<IIBBHI")
_LENM_HEAD = struct.Struct("<II8xII")
_LENM_VALUE = struct.Struct("<qQ")


def _u32_be(blob, off):
    return struct.unpack_from(">I", blob, off)[0]


def _u32_le(blob, off):
    return struct.unpack_from("<I", blob, off)[0]


def _fit(payload, header, element, counts):
    """(k, elements): the header count k for which k headers and their `elements` elements fill the payload exactly.

    Neither LSCH nor LENM stores this count. `counts` unpacks the element count of one header. Returns None
    when no k fits.
    """
    length = len(payload)
    if not length:
        return 0, 0
    elements = 0
    k = 0
    for (count,) in counts.iter_unpack(memoryview(payload)[:length - length % header]):
        k += 1
        elements += count
        if header * k + element * elements == length:
            return k, elements
    return None


def decode_lsch(payload):
    """Decode an LSCH payload into a list of structs; raises ValueError when no struct count fits."""
    fit = _fit(payload, STRUCT_HEADER, MEMBER, _LSCH_COUNTS)
    if fit is None:
        raise ValueError(f"LSCH: no struct count fills {len(payload)} bytes")
    k = fit[0]
    view = memoryview(payload)
    at = STRUCT_HEADER * k
    rows = _LSCH_MEMBER.iter_unpack(view[at:])
    structs = []
    for name, sig, flags, align, _z10, _z12, size, _z28, count in _LSCH_HEAD.iter_unpack(view[:at]):
        members = [{"hash": mname, "offset": off, "type": mtype, "sub": sub, "index": index, "ref": ref}
                   for mname, off, mtype, sub, index, ref in itertools.islice(rows, count)]
        structs.append({"hash": name, "signature": sig, "flags": flags, "align_log2": align,
                        "size": size, "members": members})
    return structs


def decode_lenm(payload):
    """Decode an LENM payload into a list of enums; raises ValueError when no enum count fits."""
    fit = _fit(payload, ENUM_HEADER, ENUM_VALUE, _LENM_COUNTS)
    if fit is None:
        raise ValueError(f"LENM: no enum count fills {len(payload)} bytes")
    k = fit[0]
    view = memoryview(payload)
    at = ENUM_HEADER * k
    rows = _LENM_VALUE.iter_unpack(view[at:])
    enums = []
    for name, sig, count, _z20 in _LENM_HEAD.iter_unpack(view[:at]):
        values = [{"value": value, "hash": vname} for value, vname in itertools.islice(rows, count)]
        enums.append({"hash": name, "signature": sig, "values": values})
    return enums


def _struct_problem(name, size, rows):
    """check_schema's verdict on one struct, from its member rows (hash, offset, type, sub, index, ref)."""
    for i, (mname, off, mtype, _sub, index, _ref) in enumerate(rows):
        if mname != ARRAY_ELEMENT and off >= size:
            return f"struct {name:08x} member {i} at offset {off} is past size {size}"
        if mtype == ARRAY_TYPE and (index >= len(rows) or rows[index][0] != ARRAY_ELEMENT):
            return f"struct {name:08x} array member {i} points at {index}, not an element descriptor"
    return ""


def lsch_summary(payload):
    """(structs, members, problem) of an LSCH payload without building the struct dicts.

    Equal to `len(decode_lsch(payload))`, the member total and `check_schema(decode_lsch(payload))`, and it
    raises the same ValueError. The member total is the element count the header fit found.
    """
    fit = _fit(payload, STRUCT_HEADER, MEMBER, _LSCH_COUNTS)
    if fit is None:
        raise ValueError(f"LSCH: no struct count fills {len(payload)} bytes")
    k, members = fit
    view = memoryview(payload)
    at = STRUCT_HEADER * k
    rows = _LSCH_MEMBER.iter_unpack(view[at:])
    for name, size, count in _LSCH_SUMMARY.iter_unpack(view[:at]):
        problem = _struct_problem(name, size, list(itertools.islice(rows, count)))
        if problem:
            return k, members, problem
    return k, members, ""


def lsch_hexed(payload):
    """(structs with hashes as 8-digit hex, members, problem) of an LSCH payload, for the "schema" entry.

    The structs are the dicts of `decode_lsch` with the name, signature and reference hashes formatted as hex;
    the other two are as in `lsch_summary`.
    """
    fit = _fit(payload, STRUCT_HEADER, MEMBER, _LSCH_COUNTS)
    if fit is None:
        raise ValueError(f"LSCH: no struct count fills {len(payload)} bytes")
    k, members = fit
    view = memoryview(payload)
    at = STRUCT_HEADER * k
    rows = _LSCH_MEMBER.iter_unpack(view[at:])
    out = []
    problem = ""
    for name, sig, flags, align, _z10, _z12, size, _z28, count in _LSCH_HEAD.iter_unpack(view[:at]):
        chunk = list(itertools.islice(rows, count))
        if not problem:
            problem = _struct_problem(name, size, chunk)
        out.append({"hash": "%08x" % name, "signature": "%08x" % sig, "flags": flags, "align_log2": align,
                    "size": size,
                    "members": [{"hash": "%08x" % mname, "offset": off, "type": mtype, "sub": sub,
                                 "index": index, "ref": "%08x" % ref}
                                for mname, off, mtype, sub, index, ref in chunk]})
    return out, members, problem


def lenm_summary(payload):
    """(enums, values) of an LENM payload: the counts `decode_lenm` would give, without the enum dicts."""
    fit = _fit(payload, ENUM_HEADER, ENUM_VALUE, _LENM_COUNTS)
    if fit is None:
        raise ValueError(f"LENM: no enum count fills {len(payload)} bytes")
    return fit


def lenm_hexed(payload):
    """(enums with hashes as 8-digit hex, values) of an LENM payload: the dicts of `decode_lenm` with hex hashes."""
    fit = _fit(payload, ENUM_HEADER, ENUM_VALUE, _LENM_COUNTS)
    if fit is None:
        raise ValueError(f"LENM: no enum count fills {len(payload)} bytes")
    k, values = fit
    view = memoryview(payload)
    at = ENUM_HEADER * k
    rows = _LENM_VALUE.iter_unpack(view[at:])
    out = [{"hash": "%08x" % name, "signature": "%08x" % sig,
            "values": [{"value": value, "hash": "%08x" % vname} for value, vname in itertools.islice(rows, count)]}
           for name, sig, count, _z20 in _LENM_HEAD.iter_unpack(view[:at])]
    return out, values


LSNS_RECORD = 16


def decode_lsns(payload):
    """The LSNS block table: [(kind, byte count)]. Raises ValueError unless the blocks fill the payload exactly."""
    if len(payload) < 4:
        raise ValueError("LSNS shorter than its block count")
    n = _u32_le(payload, 0)
    table_end = 4 + 8 * n
    if table_end > len(payload):
        raise ValueError(f"LSNS block table ({n} blocks) runs past the section")
    blocks = [struct.unpack_from("<II", payload, 4 + 8 * i) for i in range(n)]
    for kind, size in blocks:
        if size % LSNS_RECORD:
            raise ValueError(f"LSNS block of kind 0x{kind:x} has {size} bytes, not a multiple of {LSNS_RECORD}")
    if table_end + sum(size for _kind, size in blocks) != len(payload):
        raise ValueError("LSNS blocks do not fill the section")
    return blocks


def check_schema(structs):
    """Checks that hold on every measured LSCH. Returns the first problem, or ''."""
    for s in structs:
        for i, m in enumerate(s["members"]):
            if m["hash"] != ARRAY_ELEMENT and m["offset"] >= s["size"]:
                return f"struct {s['hash']:08x} member {i} at offset {m['offset']} is past size {s['size']}"
            if m["type"] == ARRAY_TYPE:
                j = m["index"]
                if j >= len(s["members"]) or s["members"][j]["hash"] != ARRAY_ELEMENT:
                    return f"struct {s['hash']:08x} array member {i} points at {j}, not an element descriptor"
    return ""


def parse(blob, schema=False):
    """Parse one PSIN container. Returns None when the blob is not PSIN.

    With `schema`, the record also carries the decoded LSCH structs and LENM enums under "schema".
    """
    if len(blob) < HEADER or blob[:4] != MAGIC:
        return None
    first = _FIRST(blob)[0]
    version, table_bytes = _VERSION_TABLE(blob)
    record = {
        "magic": "PSIN",
        "size": len(blob),
        "first_section": first,
        "version": version,
        "table_bytes": table_bytes,
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
    if first < HEADER or first > len(blob):
        record["error"] = f"first section offset {first} is outside the file"
        return _with_schema(record, schema, structs, enums)
    if len(blob) - first < SECTION_HEADER:
        record["error"] = f"missing or truncated first section header at {first}"
        return _with_schema(record, schema, structs, enums)
    offset = first
    length = len(blob)
    sections = record["sections"]
    known = _KNOWN_TAGS
    while offset + SECTION_HEADER <= length:
        tag, size = _SECTION(blob, offset)
        entry = known.get(tag)
        if entry is None:
            if tag.translate(None, _PRINTABLE):
                record["error"] = f"non-text section tag at {offset}"
                break
            entry = (tag.decode("latin-1"), 0)
        name, kind = entry
        end = offset + size
        if size < SECTION_HEADER or end > length:
            record["error"] = f"section {name} at {offset} has size {size}"
            break
        payload_off = offset + SECTION_HEADER
        payload = size - SECTION_HEADER
        section = {
            "tag": name,
            "offset": offset,
            "size": size,
            "payload": payload,
        }
        if kind == 0:
            pass
        elif kind == 3:   # LSNS
            try:
                blocks = decode_lsns(blob[payload_off:end])
            except ValueError as e:
                section["lsns_error"] = record["lsns_error"] = str(e)
            else:
                section["blocks"] = [{"kind": f"0x{k:x}", "bytes": b} for k, b in blocks]
                record["lsns_blocks"] = len(blocks)
                record["lsns_records"] = sum([b for _k, b in blocks]) // LSNS_RECORD
        elif kind == 1:   # LSCH
            if not lsch_seen:
                lsch_seen = True
                try:
                    if schema:
                        structs, n_members, problem = lsch_hexed(blob[payload_off:end])
                        n_structs = len(structs)
                    else:
                        n_structs, n_members, problem = lsch_summary(blob[payload_off:end])
                except ValueError as e:
                    lsch_error = str(e)
                else:
                    record["structs"] = n_structs
                    record["members"] = n_members
                    lsch_error = problem
        elif kind == 2:   # LENM
            if not lenm_seen:
                lenm_seen = True
                try:
                    if schema:
                        enums, n_values = lenm_hexed(blob[payload_off:end])
                        n_enums = len(enums)
                    else:
                        n_enums, n_values = lenm_summary(blob[payload_off:end])
                except ValueError as e:
                    lenm_error = str(e)
                else:
                    record["enums"] = n_enums
                    record["enum_values"] = n_values
        elif kind == 4:   # LRSC
            if payload >= 8 and blob[payload_off:payload_off + 4] == b"0DRP":
                section["rsc_magic"] = "0DRP"
                section["rsc_version"] = _U32_LE(blob, payload_off + 4)[0]
                if record["rsc_version"] is None:
                    record["rsc_version"] = section["rsc_version"]
        elif payload >= 8:   # CHKS
            section["checksum_size"] = _U32_BE(blob, payload_off)[0]
            section["checksum"] = blob[payload_off + 4:payload_off + 8].hex()
            if record["checksum_size"] is None:
                record["checksum_size"] = section["checksum_size"]
                record["checksum"] = section["checksum"]
        sections.append(section)
        offset = end
    record["schema_error"] = "; ".join(e for e in (lsch_error, lenm_error) if e)
    if not record["error"] and offset != len(blob):
        record["error"] = f"section walk ended at {offset} of {len(blob)}"
    record["complete"] = not record["error"] and offset == len(blob)
    return _with_schema(record, schema, structs, enums, lsch_error, lenm_error)


def _with_schema(record, schema, structs, enums, lsch_error="", lenm_error=""):
    """With `schema`, give every record the same "schema" shape, whether or not the sections were reached.

    `structs` and `enums` are what `lsch_hexed` and `lenm_hexed` returned, or None for a section not decoded.

    "lsch" and "lenm" say whether each section was decoded; "reason" says why one was not (a structural error
    that stopped the walk first, a missing section, or the decode error), and is "" when both were.
    """
    if not schema:
        return record
    entry = {"structs": structs or [], "enums": enums or []}
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


def input_paths(src):
    """Paths read from a single source file or a directory of carved PSIN streams."""
    if os.path.isfile(src):
        yield src
        return
    for name in sorted(os.listdir(src)):
        if not name.endswith(".bin"):
            continue
        path = os.path.join(src, name)
        if os.path.isfile(path):
            yield path


def files_in(src):
    for path in input_paths(src):
        with open(path, "rb") as f:
            yield os.path.basename(path), f.read()


def collect(src, limit=None, schema=False):
    records = []
    for name, blob in files_in(src):
        record = parse(blob, schema)
        if record is None:
            continue
        record["file"] = name
        records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def summarize(records):
    tags = {}
    versions = {}
    for r in records:
        versions[r["version"]] = versions.get(r["version"], 0) + 1
        for s in r["sections"]:
            tags[s["tag"]] = tags.get(s["tag"], 0) + 1
    return {
        "files": len(records),
        "complete": sum(1 for r in records if r["complete"]),
        "incomplete": sum(1 for r in records if not r["complete"]),
        "versions": {str(k): v for k, v in sorted(versions.items())},
        "section_tags": dict(sorted(tags.items())),
        "section_bytes": sum(s["size"] for r in records for s in r["sections"]),
        "rsc_122": sum(1 for r in records if r["rsc_version"] == 122),
        "checksummed": sum(1 for r in records if r["checksum_size"] is not None),
        "lsch_decoded": sum(1 for r in records if r["structs"] is not None),
        "lenm_decoded": sum(1 for r in records if r["enums"] is not None),
        "schema_errors": sum(1 for r in records if r["schema_error"]),
        "structs": sum(r["structs"] or 0 for r in records),
        "members": sum(r["members"] or 0 for r in records),
        "enums": sum(r["enums"] or 0 for r in records),
        "enum_values": sum(r["enum_values"] or 0 for r in records),
        "lsns_decoded": sum(1 for r in records if r["lsns_blocks"] is not None),
        "lsns_empty": sum(1 for r in records if r["lsns_blocks"] == 0),
        "lsns_errors": sum(1 for r in records if r["lsns_error"]),
        "lsns_records": sum(r["lsns_records"] or 0 for r in records),
        "lsns_kinds": dict(sorted(_lsns_kinds(records).items())),
    }


def _lsns_kinds(records):
    kinds = {}
    for r in records:
        for s in r["sections"]:
            for b in s.get("blocks", ()):
                kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    return kinds


def count(text):
    """--limit: 0 or a negative value used to mean no limit (0 is falsy, a negative count is never reached)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    ap = argparse.ArgumentParser(description="Index PSIN metadata containers.")
    ap.add_argument("src", help="PSIN file or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="write records and summary as JSON")
    ap.add_argument("--schema", action="store_true", help="include the decoded LSCH structs and LENM enums")
    ap.add_argument("--limit", type=count, help="stop after this many PSIN files")
    a = ap.parse_args()

    if not os.path.exists(a.src):   # files_in took any non-file for a folder: a typo ended in a traceback
        print(f"ERROR: {a.src}: no such file or folder", file=sys.stderr)
        return 2
    try:
        records = collect(a.src, a.limit, a.schema)
    except OSError as e:
        print(f"ERROR: {a.src}: {e}", file=sys.stderr)
        return 2
    summary = summarize(records)
    if a.json:
        json.dump({"files": records, "summary": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for r in records:
        tags = " ".join(f"{s['tag']}:{s['size']}" for s in r["sections"]) or "-"
        state = "ok" if r["complete"] else f"INCOMPLETE ({r['error']})"
        extra = ""
        if r["rsc_version"] is not None:
            extra += f"  rsc={r['rsc_version']}"
        if r["checksum_size"] is not None:
            extra += f"  chks={r['checksum_size']}"
        if r["structs"] is not None:
            extra += f"  structs={r['structs']} members={r['members']}"
        if r["enums"] is not None:
            extra += f"  enums={r['enums']} values={r['enum_values']}"
        if r["schema_error"]:
            extra += f"  SCHEMA ({r['schema_error']})"
        print(f"{r['file']}  size={r['size']}  v={r['version']}  {state}  {tags}{extra}")
        if a.schema:
            if r["schema"]["reason"]:
                print(f"  schema: {r['schema']['reason']}")
            for s in r["schema"]["structs"]:
                print(f"  struct {s['hash']} sig={s['signature']} size={s['size']} "
                      f"align=2^{s['align_log2']} flags={s['flags']:#x} members={len(s['members'])}")
                for m in s["members"]:
                    print(f"    {m['hash']} +{m['offset']:#x} type={m['type']:#04x} sub={m['sub']} "
                          f"index={m['index']} ref={m['ref']}")
            for e in r["schema"]["enums"]:
                print(f"  enum {e['hash']} sig={e['signature']} values={len(e['values'])}")
                for v in e["values"]:
                    print(f"    {v['hash']} = {v['value']}")
    print(f"{summary['files']} PSIN files, {summary['complete']} complete, "
          f"{summary['incomplete']} incomplete; tags {summary['section_tags']}; "
          f"rsc122 {summary['rsc_122']}, checksummed {summary['checksummed']}; "
          f"LSCH decoded {summary['lsch_decoded']} ({summary['structs']} structs, {summary['members']} members), "
          f"LENM decoded {summary['lenm_decoded']} ({summary['enums']} enums, {summary['enum_values']} values), "
          f"schema errors {summary['schema_errors']}; "
          f"LSNS decoded {summary['lsns_decoded']} ({summary['lsns_empty']} empty, {summary['lsns_records']} records), "
          f"LSNS errors {summary['lsns_errors']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
