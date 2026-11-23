"""Keep docs/textures.md's BCPack history table in step with the shipped tables.

    python -I tests/test_docs_bcpack_status.py

The confirmed/provisional counts in the "Codec tables and history loading" section
are a release-facing status summary. Commit 660b7fab updated the tables but left the
docs at the previous counts, which made the BC1 predictor window look unrecovered
when all 255 reachable entries were already present. This test pins those four rows
to the actual tables, including the provisional keys and unfitted-slot notes, so a
future table update cannot silently leave stale status.

Standard library only; no game files or build tools.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import bcpack_tables as tables  # noqa: E402

ROW = re.compile(
    r"^\|\s*(BC[1345])\s*\|.*?\|\s*(\d+)\s+confirmed\s*/\s*(\d+)\s+provisional\s*\|([^|]*)\|",
    re.MULTILINE,
)


def parse_status_rows(text):
    """codec -> (confirmed count, provisional count) for the simple-codec status rows."""
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in ROW.finditer(text)}


def parse_status_notes(text):
    """codec -> notes cell for the same rows, so prose claims can be pinned too."""
    return {m.group(1): m.group(4).strip() for m in ROW.finditer(text)}


def provisional_keys_from_note(note):
    """The keys in 'entries 6, 71 and 122 lack exact confirmation'."""
    match = re.search(r"entries ([\d, and]+) lack exact confirmation", note)
    return {int(key) for key in re.findall(r"\d+", match.group(1))} if match else set()


class DocsBcpackStatus(unittest.TestCase):
    def test_parser_reads_counts_and_notes_and_ignores_other_rows(self):
        text = "\n".join([
            "| Format | Source | Status | Notes |",
            "|---|---|---|---|",
            "| BC1 | `tools/bcpack_tables.py` | 252 confirmed / 3 provisional | note one |",
            "| BC6H | external `--history` | Empty by default | note two |",
            "| BC7 | external `--history` | Empty by default | note three |",
        ])
        self.assertEqual(parse_status_rows(text), {"BC1": (252, 3)})
        self.assertEqual(parse_status_notes(text), {"BC1": "note one"})

    def test_provisional_note_parser(self):
        self.assertEqual(
            provisional_keys_from_note("entries 6, 71 and 122 lack exact confirmation"),
            {6, 71, 122},
        )
        self.assertEqual(provisional_keys_from_note("no provisional entries"), set())

    def test_docs_rows_match_the_shipped_tables(self):
        with open(os.path.join(ROOT, "docs", "textures.md"), encoding="utf-8") as f:
            text = f.read()
        rows = parse_status_rows(text)
        notes = parse_status_notes(text)
        self.assertEqual(set(rows), {"BC1", "BC3", "BC4", "BC5"})
        for codec in sorted(rows):
            number = int(codec[2:])
            confirmed = getattr(tables, f"BC{number}_HISTORY_CONFIRMED")
            provisional = getattr(tables, f"BC{number}_HISTORY_PROVISIONAL")
            expected = (len(confirmed), len(provisional))
            self.assertEqual(rows[codec], expected, f"stale status row for {codec} in docs/textures.md")
            reachable = len(set(confirmed) | set(provisional))
            if 0 not in confirmed and 0 not in provisional and reachable == 255:
                self.assertIn(f"all {reachable} reachable slots populated", notes[codec])
            unfitted = 256 - reachable
            if number in (4, 5) and unfitted:
                self.assertIn(f"{unfitted} history slots still unfitted", notes[codec])
        self.assertEqual(provisional_keys_from_note(notes["BC1"]), set(tables.BC1_HISTORY_PROVISIONAL))


if __name__ == "__main__":
    unittest.main()
