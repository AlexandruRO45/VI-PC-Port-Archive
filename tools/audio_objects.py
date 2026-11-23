"""Find the 24-byte audio-config entries that hold each music record's hash A.

`tools/audio_config.py` reads the music/radio record table; its `hash A` (record + 8) has no known meaning.
Every record's hash A is also stored in one 24-byte entry of a table elsewhere in the config, six
little-endian u32:

    +0   flags      a multiple of 0x800 (0x800 on almost every music entry)
    +4   u32        not decoded; three distinct values over the music entries
    +8   kind       small integer: 2 on almost every music entry, 0 on the rest
    +12  hash A     the record's hash A
    +16  value V    not decoded; it is not a record's hash A or hash B
    +20  value V    the same word again

The entries follow one another with no gap, and a table can continue from one carved stream into the next:
an entry ends exactly at a stream's end and the following stream starts with the next entry's flags. The
entries are listed in ascending record id, a V value can be shared by several entries, and no other part of
the config lists the music records by these values. See docs/radio_stations.md ("Hash A entries") for the
counts on the measured package.

Entries are found by their shape and by the join (the word at +12 must be a record's hash A), not by a stream
offset, so another carve order or an incomplete carve gives fewer matches, not wrong ones.

Usage:
    python -I audio_objects.py <carve dir or blob> [--json]

Text output is a summary; --json writes one row per record (id, hash A, kind, flags, V, the stream and offset
of its entry) and the summary. Hash values stay in your local output: never paste them into an issue or pull
request (CONTRIBUTING.md).
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_config  # noqa: E402

ENTRY_BYTES = 24
FLAG_UNIT = 0x800
MAX_FLAGS = 16 * FLAG_UNIT     # the largest measured is 12 * 0x800
MAX_KIND = 15


def entry_at(words, i, hash_a):
    """The entry whose hash-A word is words[i] (entry + 12), or None if the words around it lack its shape."""
    if i < 3 or i + 2 >= len(words) or words[i] not in hash_a:
        return None
    flags, _, kind = words[i - 3:i]
    if not (flags and flags % FLAG_UNIT == 0 and flags <= MAX_FLAGS and kind <= MAX_KIND):
        return None
    return {"flags": flags, "kind": kind, "hash_a": words[i], "value": words[i + 1],
            "value_repeated": words[i + 2] == words[i + 1]}


def scan(blob, hash_a):
    """Yield (byte offset of the entry, entry) for every entry in `blob` whose hash-A word is in `hash_a`.

    Each of the four byte alignments is read, because a carved stream does not always start on a word boundary
    of the config.
    """
    for align in range(4):
        n = (len(blob) - align) // 4
        if n < ENTRY_BYTES // 4:
            continue
        words = struct.unpack_from(f"<{n}I", blob, align)
        for i, v in enumerate(words):
            if v in hash_a:
                e = entry_at(words, i, hash_a)
                if e:
                    yield align + (i - 3) * 4, e


def join(src, records):
    """Return (rows, summary): one row per record, with its entry when exactly one was found."""
    by_hash = {}
    for r in records:
        by_hash.setdefault(r["hash_a"], []).append(r)
    found = {}                  # hash A -> list of (file, offset, entry)
    for name, blob in audio_config.files_in(src):
        for off, e in scan(blob, by_hash):
            found.setdefault(e["hash_a"], []).append((name, off, e))
    rows = []
    for r in records:
        hits = found.get(r["hash_a"], [])
        row = {"id": r["id"], "hash_a": r["hash_a"], "stream_id": r["stream_id"], "entries": len(hits)}
        if len(hits) == 1:
            name, off, e = hits[0]
            row.update(kind=e["kind"], flags=e["flags"], value=e["value"], value_repeated=e["value_repeated"],
                       file=name, file_offset=off)
        rows.append(row)
    one = [x for x in rows if x["entries"] == 1]
    values = Counter(x["value"] for x in one)
    record_keys = set(by_hash) | {r["hash_b"] for r in records}
    # entries in file order: are the record ids ascending along each stream?
    ordered = sorted(one, key=lambda x: (x["file"], x["file_offset"]))
    ascending = sum(1 for a, b in zip(ordered, ordered[1:]) if a["file"] == b["file"] and b["id"] > a["id"])
    neighbours = sum(1 for a, b in zip(ordered, ordered[1:]) if a["file"] == b["file"])
    summary = {
        "records": len(rows),
        "shared_hash_a": sum(1 for v in by_hash.values() if len(v) > 1),
        "with_one_entry": len(one),
        "with_no_entry": sum(1 for x in rows if x["entries"] == 0),
        "with_several_entries": sum(1 for x in rows if x["entries"] > 1),
        "value_repeated": sum(1 for x in one if x["value_repeated"]),
        "distinct_values": len(values),
        "values_shared": sum(1 for c in values.values() if c > 1),
        "most_entries_per_value": max(values.values(), default=0),
        "values_that_are_record_hashes": sum(1 for v in values if v in record_keys),
        "kinds": {str(k): c for k, c in sorted(Counter(x["kind"] for x in one).items())},
        "flags": {hex(k): c for k, c in sorted(Counter(x["flags"] for x in one).items())},
        "streams": dict(sorted(Counter(x["file"] for x in one).items())),
        "ascending_neighbours": ascending,
        "neighbours": neighbours,
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Find the audio-config entries that hold each music record's hash A.")
    ap.add_argument("src", help="inflated config blob or directory of carved .bin streams (rpf9_carve.py --until-adat)")
    ap.add_argument("--json", action="store_true", help="write one row per record and the summary as JSON")
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print(f"ERROR: {a.src}: does not exist", file=sys.stderr)
        return 2
    try:
        records = audio_config.collect(a.src, join=os.path.isdir(a.src))
        if not records:
            print(f"ERROR: {a.src}: no audio-config records found", file=sys.stderr)
            return 1
        rows, summary = join(a.src, records)
    except OSError as e:
        print(f"ERROR: {e.filename or a.src}: cannot read: {e.strerror or e}", file=sys.stderr)
        return 1
    if a.json:
        json.dump({"records": rows, "summary": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    s = summary
    print(f"{s['records']} records; {s['with_one_entry']} with one entry, {s['with_no_entry']} with none, "
          f"{s['with_several_entries']} with several")
    print(f"V repeated at +20: {s['value_repeated']}; {s['distinct_values']} distinct V, {s['values_shared']} "
          f"shared (at most {s['most_entries_per_value']} entries); {s['values_that_are_record_hashes']} "
          f"equal a record hash")
    print(f"kinds: {s['kinds']}  flags: {s['flags']}")
    print(f"entries per stream: {s['streams']}")
    print(f"record id ascending between neighbouring entries: {s['ascending_neighbours']} of {s['neighbours']}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
