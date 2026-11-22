"""CPU-only first-light selection checks on a synthetic carve index."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import first_light as fl


class ShaderSelectionTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.work = Path(scratch.name)
        self.index = self.work / "shaders" / "dxbc" / "index.json"
        self.index.parent.mkdir(parents=True)
        self.entries = [
            {"stage": "vs", "name": "vertex:main", "file": "vertex_main.dxbc", "offset": 8},
            {"stage": "ps", "name": "shared", "file": "shared.dxbc", "offset": 16},
            {"stage": "ps", "name": "shared", "file": "shared~1.dxbc", "offset": 32},
        ]
        self.index.write_text(json.dumps(self.entries), encoding="utf-8", newline="\n")

    def test_ambiguous_name_lists_offsets_and_each_selects_its_own_entry(self):
        with self.assertRaisesRegex(SystemExit, r"ambiguous ps.*--ps-offset.*0x10.*0x20"):
            fl.choose(self.index, "vertex:main", "shared")
        for entry in self.entries[1:]:
            self.assertEqual(fl.choose(self.index, "vertex:main", "shared", ps_offset=entry["offset"]),
                             [self.entries[0], entry])
        for wrong_offset in (8, 999):
            with self.subTest(offset=wrong_offset), self.assertRaisesRegex(SystemExit, "no ps shader"):
                fl.choose(self.index, "vertex:main", "shared", ps_offset=wrong_offset)

    def test_list_shows_selectors(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(fl.main(["--list", "--game", str(self.work), "--work", str(self.work)]), 0)
        self.assertIn("vs vertex:main\t--vs-offset 0x8", stdout.getvalue())
        self.assertIn("ps shared\t--ps-offset 0x10", stdout.getvalue())
        self.assertIn("ps shared\t--ps-offset 0x20", stdout.getvalue())

    def test_selected_files_and_metadata_survive_assembly(self):
        # Stand in for the external compiler stages with distinct outputs for each unique carved file.
        spirv = self.work / "first_light" / "spirv"
        spirv.mkdir(parents=True)
        metadata = {}
        for entry in self.entries:
            stem = Path(entry["file"]).stem
            (self.index.parent / entry["file"]).write_bytes(("container " + stem).encode())
            (spirv / (stem + ".spv")).write_bytes(("program " + stem).encode())
            metadata[stem] = {"stage": entry["stage"], "fssm_sha256": str(entry["offset"])}
        (spirv / "index.json").write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
        args = ["--game", str(self.work), "--work", str(self.work), "--no-fit",
                "--vs-offset", "8", "--ps-offset", "0x20"]
        with patch.object(fl, "run"), redirect_stdout(io.StringIO()):
            self.assertEqual(fl.main(args), 0)
        out = self.work / "first_light" / "out"
        for stage, stem in (("vs", "vertex_main"), ("ps", "shared~1")):
            self.assertEqual((out / (stage + ".spv")).read_bytes(), ("program " + stem).encode())
            self.assertEqual((out / (stage + ".dxbc")).read_bytes(), ("container " + stem).encode())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["ps"]["offset"], 32)
        self.assertEqual(manifest["ps"]["fssm_sha256"], "32")
        del metadata["shared~1"]
        (spirv / "index.json").write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
        with patch.object(fl, "run"), self.assertRaisesRegex(SystemExit, "no ps result for shared~1"):
            fl.main(args)


if __name__ == "__main__":
    unittest.main()
