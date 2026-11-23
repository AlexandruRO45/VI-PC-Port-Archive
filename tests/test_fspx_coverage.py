import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fspx_coverage import compare, coverage


class CoverageTests(unittest.TestCase):
    def test_partial_and_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "unpacked", root / "translated"
            listings = source / "disasm"
            listings.mkdir(parents=True)
            names = ["ok", "refused", "absent", "unknown", "orphan", "lost", "conflict"]
            rows = [{"file": name + ".dxbc"} for name in names]
            (source / "index.json").write_text(json.dumps(rows), encoding="utf-8")
            for name in names:
                if name != "absent":
                    (listings / (name + ".s")).touch()
            self.assertEqual(coverage(source, output)["unaccounted"], 7)
            output.mkdir()
            for name in ("ok", "orphan", "conflict", "extra"):
                (output / (name + ".spv")).touch()
            index = dict.fromkeys(("ok", "lost", "extra"), {})
            (output / "index.json").write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(coverage(source, output)["refused"], 0)
            (output / "refusals.json").write_text('["refused","conflict"]', encoding="utf-8")
            self.assertEqual(coverage(source, output), {
                "programs": 7, "eligible": 6, "translated": 1, "refused": 1, "unaccounted": 5,
                "unaccounted_by_reason": {"missing_listing": 1, "no_result": 1, "unindexed_module": 1,
                                          "missing_module": 1, "conflicting_results": 1},
            })


    def test_renamed_shuffled_partial_runs(self):
        def run(root, source, rows):
            output = root / "out"
            output.mkdir(parents=True)
            (root / "fspx").mkdir()
            (root / "disasm").mkdir()
            index, translated, refused = [], {}, []
            for name, code, outcome in rows:
                index.append({"file": name + ".dxbc"})
                if code:
                    (root / "fspx" / (name + ".code")).write_bytes(code)
                if source == "listing":
                    (root / "disasm" / (name + ".s")).touch()
                if outcome == "refused":
                    refused.append(name)
                elif outcome in ("valid", "invalid"):
                    translated[name] = {}
                    (output / (name + ".spv")).write_bytes(outcome.encode())
            for path, data in ((root / "index.json", index), (output / "index.json", translated),
                               (output / "refusals.json", refused)):
                path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
            return root, output, source

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = run(root / "a", "listing", [
                ("a", b"a", "refused"), ("b", b"b", "valid"), ("c", b"c", "refused"),
                ("d", b"d", "valid"), ("e", b"e", "valid"), ("f", b"f", "none"),
                ("g", b"g", "refused"), ("h", b"g", "valid"), ("i", None, "none")])
            new = run(root / "b", "code", [
                ("a", b"g", "valid"), ("b", b"f", "valid"), ("c", b"e", "invalid"),
                ("d", b"c", "refused"), ("e", b"b", "refused"), ("f", b"a", "valid")])
            with patch("fspx_coverage.validated", side_effect=lambda p: p.read_bytes() == b"valid"):
                self.assertEqual(compare(old, new), {
                    "missing_code_before": 1, "missing_code_after": 0,
                    "transitions": {"refused": {"validated": 1, "refused": 1},
                                    "validated": {"absent": 1, "refused": 1, "validation_failed": 1},
                                    "no_result": {"validated": 1}, "duplicate_identity": {"validated": 1}}})


if __name__ == "__main__":
    unittest.main()
