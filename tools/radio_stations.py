"""Resolve the audio config's station descriptor hashes against the local debug list.

`docs/radio_stations.md` measures the station descriptors as eight-byte
``[station hash][byte end]`` pairs and notes the missing piece: recognizing a station
needs a ``joaat`` dictionary of the station identifiers, so the page records offsets and
counts instead of asking a scanner to guess names.

The package carries that dictionary: a plaintext debug stream lists
``Core.RadioStation.<slug>`` names. The descriptor hash is ``joaat("radio_<slug>")``, so
the debug list turns a matching descriptor into a bare identifier without guessing.

Usage:
    python -I radio_stations.py <carve dir> [--json] [--out FILE]

The input is the inflated output of ``rpf9_carve.py --until-adat`` for ``audio.rpf`` (one
``.bin`` per zlib stream, named by its archive offset). Nothing here reads an archive, a
key or a decrypted file, and the names it prints come from the game the user already has;
keep the output in the local work folder.

Descriptors are found by their shape, not by a stream offset: a 4-byte-aligned
``[hash][end]`` pair whose end is a multiple of 1000 in ``0x03E8``-``0x55F0`` and whose hash
is one of the debug list's ``radio_<slug>`` hashes. A stream with no debug-list hash is
skipped. The tool reports what it resolved; it does not guess a name for a descriptor whose
hash is absent from the debug list.
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_config import joaat  # noqa: E402  (same hash rule as the record table)

DEBUG_MARKER = b"Core.RadioStation."
END_FIRST = 0x03E8  # first station end offset seen in the measured stream
END_LAST = 0x55F0   # last station end offset seen in the measured stream
END_UNIT = 1000


def find_debug_stream(parts):
    """Return (name, slugs) for the stream holding the ``Core.RadioStation.<slug>`` list."""
    for name, path in parts:
        try:
            with open(path, "rb") as handle:
                blob = handle.read()
        except OSError:
            continue
        if DEBUG_MARKER not in blob:
            continue
        slugs = []
        seen = set()
        start = 0
        while True:
            at = blob.find(DEBUG_MARKER, start)
            if at < 0:
                break
            end = at + len(DEBUG_MARKER)
            while end < len(blob) and (blob[end:end + 1].isalnum() or blob[end:end + 1] == b"_"):
                end += 1
            slug = blob[at + len(DEBUG_MARKER):end].decode("latin-1")
            if slug and slug not in seen:
                seen.add(slug)
                slugs.append(slug)
            start = end + 1
        return name, slugs
    return None


def station_hashes(slugs):
    """Map each ``joaat("radio_<slug>")`` to its slug."""
    table = {}
    for slug in slugs:
        table.setdefault(joaat("radio_" + slug), []).append(slug)
    return table


def is_end(value):
    return END_FIRST <= value <= END_LAST and value % END_UNIT == 0


def scan_stream(name, blob, hashes):
    """Return the resolved descriptors in one inflated stream."""
    descriptors = []
    limit = len(blob) - 8
    for at in range(0, limit + 1, 4):
        end = struct.unpack_from("<I", blob, at + 4)[0]
        if not is_end(end):
            continue
        station = struct.unpack_from("<I", blob, at)[0]
        slugs = hashes.get(station)
        if slugs:
            descriptors.append({
                "stream": name,
                "offset": at,
                "end": end,
                "slug": slugs[0],
                "hash": station,
            })
    return descriptors


def run(carve_dir):
    names = sorted(n for n in os.listdir(carve_dir) if n.endswith(".bin"))
    parts = [(n, os.path.join(carve_dir, n)) for n in names]
    found = find_debug_stream(parts)
    if not found:
        raise SystemExit("no stream with Core.RadioStation. names in %s" % carve_dir)
    debug_name, slugs = found
    hashes = station_hashes(slugs)

    descriptors = []
    for name, path in parts:
        if name == debug_name:
            continue
        try:
            with open(path, "rb") as handle:
                blob = handle.read()
        except OSError:
            continue
        descriptors.extend(scan_stream(name, blob, hashes))

    descriptors.sort(key=lambda d: (d["stream"], d["offset"]))
    return {
        "debug_stream": debug_name,
        "slugs": len(slugs),
        "hashes": len(hashes),
        "descriptors": descriptors,
        "matched": len(descriptors),
        "ends": sorted({d["end"] for d in descriptors}),
        "streams": sorted({d["stream"] for d in descriptors}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("carve", help="directory of rpf9_carve.py --until-adat .bin parts")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--out", help="write the report to this file instead of stdout")
    args = parser.parse_args(argv)

    report = run(args.carve)
    text = json.dumps(report, indent=2, sort_keys=True) if args.json else format_text(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0 if report["descriptors"] else 1


def format_text(report):
    lines = [
        "debug stream: %s" % report["debug_stream"],
        "station slugs: %d (%d distinct hashes)" % (report["slugs"], report["hashes"]),
        "descriptors matched: %d" % report["matched"],
        "end offsets: %s" % ", ".join("0x%X" % e for e in report["ends"]),
    ]
    for item in report["descriptors"]:
        lines.append("%s @0x%04X end=0x%X hash=0x%08X %s" % (
            item["stream"], item["offset"], item["end"], item["hash"], item["slug"]))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
