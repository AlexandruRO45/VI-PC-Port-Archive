"""Hand-built census inputs exercise both denominator and export coverage."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gbuffer_census.py"
sys.path.insert(0, str(TOOL.parent))
from gbuffer_census import census

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fssm_unpack import make_dxbc, make_xbch, make_xdat_list


class CensusTests(unittest.TestCase):
    def test_whole_payload_multiplicity_with_independent_coverage(self):
        a = make_xbch(b"FSSM01a")
        b = make_xbch(b"FSSM01b")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i, slots in enumerate(([a], [a, a], [b], [a, b], [], [b"bad"], None)):
                name = f"shader{i}.dxbc"
                rows.append({"file": name, "stage": "ps", "name": "deferred" if i else "ordinary"})
                if slots is not None:
                    dxil = make_xdat_list([(13, slot) for slot in slots])
                    (root / name).write_bytes(make_dxbc([(b"DXIL", dxil)], digest=bytes([i]) * 16))
            (root / "index.json").write_text(json.dumps(rows), encoding="utf-8", newline="\n")
            report = census(root, root, payload_coverage=True)
            default = census(root, root)
            self.assertEqual(report["all indexed ps"]["payload_coverage"],
                             {"identified": 3, "distinct_payloads": 2, "ambiguous": 1,
                              "unavailable": 3, "payloads_by_indexed_entry_count": {1: 1, 2: 1}})
            deferred = report["entry contains deferred (casefold)"]["payload_coverage"]
            self.assertEqual(deferred["payloads_by_indexed_entry_count"], {1: 2})
            for key, group in report.items():
                group.pop("payload_coverage")
                self.assertEqual(group, default[key])

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


class BadInputTests(unittest.TestCase):
    """A wrong carve folder is one error line and exit 2, never a traceback."""

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8")

    def check_error(self, result, path, reason):
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        lines = result.stderr.splitlines()
        self.assertEqual(len(lines), 1, result.stderr)
        self.assertTrue(lines[0].startswith("ERROR: "), lines[0])
        self.assertIn(str(path), lines[0])
        self.assertIn(reason, lines[0])

    def test_missing_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "absent"
            self.check_error(self.run_tool(path), path, "does not exist")

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            self.check_error(self.run_tool(temp), temp, "no index.json")

    def test_regular_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.json"
            path.write_text("[]", encoding="utf-8", newline="\n")
            self.check_error(self.run_tool(path), path, "not a directory")

    def test_malformed_index(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "index.json").write_text("[{", encoding="utf-8", newline="\n")
            self.check_error(self.run_tool(temp), Path(temp) / "index.json", "not valid JSON")

    def test_empty_index_is_still_a_report(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "index.json").write_text("[]", encoding="utf-8", newline="\n")
            result = self.run_tool(temp)
            self.assertEqual((result.returncode, result.stderr), (0, ""))
            self.assertEqual(json.loads(result.stdout)["all indexed ps"]["coverage"], {})


if __name__ == "__main__":
    unittest.main()
