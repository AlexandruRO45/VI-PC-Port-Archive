"""Synthetic end-to-end checks for the opt-in q38 census."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_page_index import header, make_rpf, put_q38


class WindowCensusTest(unittest.TestCase):
    def test_record_and_joined_windows(self):
        # First "PSIN" crosses both a record and a stream boundary. The last
        # target has exactly four available bytes even though another page follows.
        page = header(0x180, q38=0x60, counts=(0, 0, 4, 0))
        put_q38(page, 0x60, [(1, 2, 0x100), (2, 24, 0x102),
                             (3, 24, 0x120), (4, 4, 0x17c)])
        page[0x100:0x104] = b"PSIN"
        page[0x114:0x118] = bytes.fromhex("15 00 01 00")
        page[0x134:0x138] = bytes.fromhex("15 00 01 00")
        page[0x17c:] = b"PSIN"
        unfinished = header(0x80, q38=0x200, counts=(0, 0, 1, 0))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "control.rpf"
            make_rpf(archive, [page[:0x101], page[0x101:], unfinished])
            tool = Path(__file__).resolve().parents[1] / "tools" / "q38_window_census.py"
            result = json.loads(subprocess.check_output([sys.executable, "-I", tool, archive]))
        self.assertEqual((result["pages"], result["targets"], result["unread_q38_pages"]), (2, 4, 1))
        probes = result["probes"]
        for name, joined, record in [
            ("PSIN", (4, 2), (3, 1)),
            ("zero_prefix", (3, 1), (2, 1)),
            ("typed_word_at_0x14", (3, 2), (2, 1)),
        ]:
            for scope, (eligible, matches) in [("joined", joined), ("record", record)]:
                with self.subTest(name=name, scope=scope):
                    self.assertEqual(probes[name][scope], {"eligible": eligible, "matches": matches})


if __name__ == "__main__":
    unittest.main()
