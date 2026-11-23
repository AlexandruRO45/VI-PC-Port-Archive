"""Count documented q38 probes within each record and within its joined page streams."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import page_index


PROBES = [(magic.decode("ascii"), 0, magic) for magic in
          (b"PSIN", b"RBF0", b"0DRP", b"RSC7", b"RSC8")]
PROBES += [
    ("zero_prefix", 0, bytes(8)),
    ("directory_marker_at_0x10", 0x10, b"0DRP"),
    ("typed_word_at_0x14", 0x14, bytes.fromhex("15 00 01 00")),
]


def census(streams):
    probes = {
        name: {"offset": offset, "width": len(pattern),
               "record": {"eligible": 0, "matches": 0},
               "joined": {"eligible": 0, "matches": 0}}
        for name, offset, pattern in PROBES
    }
    report = {"pages": 0, "unread_q38_pages": 0, "targets": 0, "probes": probes}

    def visit(page, base, continuations):
        report["pages"] += 1
        if not page["complete"] or page["error"]:
            report["unread_q38_pages"] += 1
            return
        data = base + b"".join(s[2] for s in continuations)
        for record in page["q38_records"]:
            report["targets"] += 1
            target = record["target"]
            for name, offset, pattern in PROBES:
                end = offset + len(pattern)
                if target + end > len(data):
                    continue
                matched = data.startswith(pattern, target + offset)
                counts = probes[name]
                counts["joined"]["eligible"] += 1
                counts["joined"]["matches"] += matched
                if end <= record["size"]:
                    counts["record"]["eligible"] += 1
                    counts["record"]["matches"] += matched

    page_index.visit_pages(streams, visit)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="local RPF archive")
    args = parser.parse_args()
    try:
        report = census(page_index.scan_rpf(args.archive))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
