import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fspx_coverage import coverage


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


if __name__ == "__main__":
    unittest.main()
