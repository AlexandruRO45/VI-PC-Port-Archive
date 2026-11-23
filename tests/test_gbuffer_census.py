"""Hand-built census inputs exercise both denominator and export coverage."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gbuffer_census import census


class CensusTests(unittest.TestCase):
    def test_cohorts_and_independent_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"file": "PS_DeferredTextured.dxbc", "name": "ordinary", "stage": "ps"},
                    {"file": "unavailable.dxbc", "name": "deFerRed", "stage": "ps"},
                    {"file": "ignored.dxbc", "name": "Deferred", "stage": "vs"}]
            (root / "index.json").write_text(json.dumps(rows), encoding="utf-8", newline="\n")
            sig = struct.pack("<II", 2, 8)
            for system, reg in [(64, 2), (65, 0xffffffff)]:
                sig += struct.pack("<6IBB2xI", 0, 0, 0, system, 3, reg, 1, 0, 0)
            header = b"DXBC" + bytes(24) + struct.pack("<II", 1, 36)
            (root / rows[0]["file"]).write_bytes(header + b"OSG1" + struct.pack("<I", len(sig)) + sig)
            listing = "; synthetic code 0x0..0x4\n\texp mrt0 v1, off, off, off compr ; encoding: [0,0,0,0]\n"
            (root / "PS_DeferredTextured.s").write_text(listing, encoding="utf-8", newline="\n")
            report = census(root, root)
            group = report["stem contains PS_DeferredTextured (case-sensitive)"]
            self.assertEqual(group["declared_counts"], {"2 rows, 1 colour, 1 depth": 1})
            self.assertEqual(group["observed_exports"], {"mrt0 compr": 1})
            self.assertEqual(group["coverage"]["listing_complete"], 1)
            self.assertEqual(report["entry contains deferred (casefold)"]["coverage"],
                             {"indexed": 1, "container_missing": 1, "listing_missing": 1})
            (root / "unavailable.s").write_text(listing.replace("0x4", "0x8"), encoding="utf-8", newline="\n")
            (root / "unavailable.dxbc").write_bytes(b"bad")
            group = census(root, root)["all indexed ps"]
            self.assertEqual(group["coverage"]["signature_unreadable"], 1)
            self.assertEqual(group["coverage"]["listing_incomplete"], 1)
            self.assertEqual(group["observed_exports"], {"mrt0 compr": 2})


if __name__ == "__main__":
    unittest.main()
