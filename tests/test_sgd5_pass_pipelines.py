"""Hand-built SGD5 passes become a link-check manifest: python -I tests/test_sgd5_pass_pipelines.py"""
import collections
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sgd5_effects
import sgd5_pass_pipelines
import shader_carve
from test_sgd5_effects import database, effect

# Group 0 = VS or MS, 1 = PS, 2 = GS, 5 = CS. Programs are carved in group order.
NAMES = {0: ("Vert", "Other"), 1: ("Pix", "Pix2"), 2: ("Geo",), 5: ("Comp",)}
STAGES = ["vs", "vs", "ps", "ps", "gs", "cs"]
PASSES = [
    (1, 1, 0, 0, 0, 0),  # VS+PS; its state bytes are patched to 3, 4, 5 below
    (1, 2, 1, 0, 0, 0),  # VS+GS+PS
    (1, 1, 0, 0, 0, 0),  # the first set again
    (0, 0, 0, 0, 0, 1),  # compute
    (0, 0, 0, 0, 0, 0),  # no program
    (1, 0, 0, 0, 0, 0),  # VS only
]


class PassPipelinesTest(unittest.TestCase):
    def setUp(self):
        item = effect("Mock", {0: (0, 1), 1: (0, 1), 2: (0,), 5: (0,)}, [("T", PASSES)])
        data, self.layout = database(NAMES, [item])
        data = bytearray(data)
        at = data.index(bytes(PASSES[0]) + bytes(18))
        data[at + 6:at + 9] = bytes((3, 4, 5))
        self.data = bytes(data)
        carve = [m.start() for m in shader_carve.ENTRY.finditer(self.data, 0, self.data.index(b"Mock") - 4)]
        self.assertEqual(len(carve), 6)
        files = ["Vert", "Other", "Pix", "Pix2", "Geo", "Comp"]
        self.index = [{"offset": o, "file": f"{n}.dxbc", "stage": s, "name": "dup"}
                      for o, n, s in zip(carve, files, STAGES)]

    def run_tool(self, spirv_names=None):
        with tempfile.TemporaryDirectory() as folder, patch.object(sgd5_effects, "DATABASES", [self.layout]):
            folder = Path(folder)
            (folder / "a.rpf").write_bytes(self.data)
            (folder / "index.json").write_text(json.dumps(self.index), encoding="utf-8")
            args = [str(folder / "a.rpf"), str(folder / "index.json"), "--out", str(folder / "m.json"),
                    "--passes", str(folder / "p.json")]
            if spirv_names is not None:
                (folder / "spv").mkdir()
                for name in spirv_names:
                    (folder / "spv" / (name + ".spv")).write_bytes(b"")
                args += ["--spirv-dir", str(folder / "spv")]
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(sgd5_pass_pipelines.main(args), 0)
            return (json.loads((folder / "m.json").read_text(encoding="utf-8")),
                    json.loads((folder / "p.json").read_text(encoding="utf-8")), out.getvalue())

    @staticmethod
    def sets(manifest):
        return [{k: v for k, v in p.items() if k != "name"} for p in manifest["pipelines"]]

    def test_manifest_and_rows(self):
        manifest, rows, summary = self.run_tool()
        self.assertEqual(manifest["version"], 1)
        # VS+PS and VS+GS+PS; the repeated VS+PS pass is written once.
        self.assertEqual(self.sets(manifest), [{"vs": "Vert", "gs": "Geo", "ps": "Pix2"},
                                               {"vs": "Vert", "ps": "Pix"}])
        names = [p["name"] for p in manifest["pipelines"]]
        self.assertEqual(len(set(names)), 2)
        # Rows: every pass with a program (the empty pass is counted, not listed).
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["programs"], {"0": "Vert", "1": "Pix"})
        self.assertEqual(rows[0]["stages"], {"0": "vs", "1": "ps"})
        self.assertEqual((rows[0]["raster"], rows[0]["depth"], rows[0]["blend"]), (3, 4, 5))
        self.assertEqual((rows[1]["raster"], rows[1]["depth"], rows[1]["blend"]), (0, 0, 0))
        self.assertEqual(rows[3]["programs"], {"5": "Comp"})
        for line in ("6  passes", "1  skipped: compute", "1  skipped: no program", "1  skipped: no PS",
                     "2  distinct graphics sets with a PS"):
            self.assertIn(line, summary)

    def test_spirv_dir_leaves_out_missing(self):
        manifest, _, summary = self.run_tool(["Vert", "Pix", "Geo"])
        self.assertEqual(self.sets(manifest), [{"vs": "Vert", "ps": "Pix"}])
        self.assertIn("1  skipped: no .spv", summary)

    def test_only_stage_sets_the_link_check_takes(self):
        # An ms+gs+ps set would make `shader_link_check.py batch` refuse the whole manifest, so it is skipped by rule.
        rows = [{"keys": {"ms": "m", "gs": "g", "ps": "p"}}, {"keys": {"vs": "v", "gs": "g", "ps": "p"}}]
        counts = collections.Counter()
        pipes = sgd5_pass_pipelines.pipelines_from(rows, counts)
        self.assertEqual([sorted(k for k in p if k != "name") for p in pipes], [["gs", "ps", "vs"]])
        self.assertEqual(counts["skipped: a stage set shader_link_check does not take"], 1)


if __name__ == "__main__":
    unittest.main()
