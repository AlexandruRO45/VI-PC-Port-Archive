"""tools/world_drawable_lookup.py: a blank summary.archive is unusable, not a report label. No game files."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import world_drawable_lookup as lookup  # noqa: E402
import world_names  # noqa: E402

NAME = "one"


def write_reports(root, archive):
    h = world_names.joaat(NAME)
    key = world_names.hash64(NAME)
    definitions = [{"name": h, "asset_name": h, "asset_type": 2}]
    placements = [{"archetype": f"{h:x}", "key": f"{key:x}", "position": [0, 0, 0],
                   "rotation": [0, 0, 0, 1], "lod_level": 6}]
    names = {"summary": {"archive": archive}, "files": [{"offset": 16, "names": [NAME + ".#dr"]}]}
    (root / "defs").write_text(json.dumps({"archetypes": definitions}), encoding="utf-8", newline="\n")
    (root / "names").write_text(json.dumps(names), encoding="utf-8", newline="\n")
    (root / "entities").write_text("\n".join(map(json.dumps, placements)), encoding="utf-8", newline="\n")
    return root / "entities", root / "defs", root / "names"


class BlankArchive(unittest.TestCase):
    def test_blank_summary_archive_is_refused(self):
        for blank in ("", "   ", "\t\n"):
            with self.subTest(archive=repr(blank)), tempfile.TemporaryDirectory() as tmp:
                paths = write_reports(Path(tmp), blank)
                with self.assertRaises(ValueError) as caught:
                    lookup.lookup(*paths, [])
                self.assertIn("missing archive in summary", str(caught.exception))

    def test_non_blank_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_reports(Path(tmp), "test.rpf")
            result = lookup.lookup(*paths, [])
            self.assertEqual(result["archive"], "test.rpf")
            self.assertEqual(result["outcomes"], {"candidate": 1})


if __name__ == "__main__":
    unittest.main()
