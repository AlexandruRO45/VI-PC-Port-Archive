"""Hand-built SGD5 passes become a link-check manifest: python -I tests/test_sgd5_pass_pipelines.py"""
import collections
from contextlib import redirect_stderr, redirect_stdout
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


RASTER = [bytes([3, 1]).ljust(16, b"\0"), bytes([3, 2]).ljust(16, b"\0"), bytes([3, 3]).ljust(16, b"\0"),
          bytes([3, 2, 1, 9]).ljust(16, b"\0")]
DEPTH = [bytes([0, 0, 1]).ljust(24, b"\0")] * 4 + [bytes([1, 1, 7, 1, 255, 127, 1, 1, 3, 6]).ljust(24, b"\0")]
BLEND = [bytes(68)] * 5 + [bytes([0, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 15]).ljust(68, b"\0")]


def build(passes, states=None):
    """A one-effect database and its carve index; pass 0 of ``passes`` is the one the tests look at."""
    item = effect("Mock", {0: (0, 1), 1: (0, 1), 2: (0,), 5: (0,)}, [("T", passes)], states=states)
    data, layout = database(NAMES, [item])
    carve = [m.start() for m in shader_carve.ENTRY.finditer(data, 0, data.index(b"Mock") - 4)]
    files = ["Vert", "Other", "Pix", "Pix2", "Geo", "Comp"]
    index = [{"offset": o, "file": f"{n}.dxbc", "stage": s, "name": "dup"} for o, n, s in zip(carve, files, STAGES)]
    return bytes(data), layout, index


class PassPipelinesTest(unittest.TestCase):
    def setUp(self):
        passes = [PASSES[0][:6] + (3, 4, 5)] + PASSES[1:]
        self.data, self.layout, self.index = build(passes, {16: RASTER, 24: DEPTH, 68: BLEND})
        self.assertEqual(len(self.index), 6)

    def run_tool(self, spirv_names=None, status=0):
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
                self.assertEqual(sgd5_pass_pipelines.main(args), status)
            if status:
                self.assertFalse((folder / "m.json").exists())
                return None, None, out.getvalue()
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

    def census(self, data, layout, index, **kwargs):
        with tempfile.TemporaryDirectory() as folder, patch.object(sgd5_effects, "DATABASES", [layout]):
            path = Path(folder) / "a.rpf"
            path.write_bytes(data)
            return sgd5_pass_pipelines.census(str(path), index, **kwargs)

    def test_states_are_decoded_next_to_the_selector_bytes(self):
        states = collections.Counter()
        rows, counts = self.census(self.data, self.layout, self.index, states=states)
        self.assertEqual(rows[0]["raster_state"], {"fill_mode": 3, "cull_mode": 2, "cull": "front",
                                                   "raw_from_2": [1, 9] + [0] * 12})
        self.assertEqual(rows[0]["depth_state"], {"depth_enable": 1, "depth_write": 1, "depth_func": 7,
                                                  "depth_compare": "greater_equal", "stencil_enable": 1,
                                                  "raw_from_4": [255, 127, 1, 1, 3, 6] + [0] * 14})
        blend = rows[0]["blend_state"]
        self.assertEqual(blend["prefix_raw"], [0, 1, 0, 0])
        self.assertEqual(blend["targets"][0], {"enable": 1, "write_mask": 15, "raw": [2, 1, 1, 1, 1, 1]})
        self.assertEqual([t["enable"] for t in blend["targets"][1:]], [0] * 7)
        self.assertEqual(rows[1]["raster_state"]["cull_mode"], 1)
        self.assertEqual(rows[1]["depth_state"]["depth_compare"], "never")
        self.assertEqual(rows[1]["blend_state"]["targets"][0]["write_mask"], 0)
        # Every stored pass is counted once per kind, including the one without a program.
        for kind in ("raster", "depth", "blend"):
            self.assertEqual(sum(n for (k, _), n in states.items() if k == kind), 6)
        self.assertNotIn("empty", " ".join(counts))

    def test_empty_table_gives_null(self):
        passes = [(1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0, 0)]
        data, layout, index = build(passes, {16: [], 24: [], 68: []})
        states = collections.Counter()
        rows, counts = self.census(data, layout, index, states=states)
        for row in rows:
            self.assertEqual((row["raster_state"], row["depth_state"], row["blend_state"]), (None, None, None))
        self.assertEqual(counts["passes with an empty blend table"], 2)
        self.assertEqual(states[("raster", "null")], 2)

    def test_selector_outside_the_table_is_an_error(self):
        for at in (6, 7, 8):
            selector = [0] * 9
            selector[at] = 1  # each default table has one row, so index 1 is out of range
            data, layout, index = build([(1, 1, 0, 0, 0, 0) + tuple(selector[6:])])
            with self.subTest(byte=at), self.assertRaisesRegex(ValueError, "outside the effect's 1 rows"):
                self.census(data, layout, index)
        # A nonzero selector into an empty table is out of range too.
        data, layout, index = build([(1, 1, 0, 0, 0, 0, 1, 0, 0)], {16: []})
        with self.assertRaisesRegex(ValueError, "outside the effect's 0 rows"):
            self.census(data, layout, index)

    def test_row_outside_the_measured_ranges_is_an_error(self):
        good_depth = bytes([1, 1, 7, 1]).ljust(24, b"\0")
        cases = (
            ("rasterizer row", {16: [bytes([2, 1]).ljust(16, b"\0")]}),        # fill mode other than 3
            ("depth/stencil row", {24: [bytes([1, 1, 7, 2]).ljust(24, b"\0")]}),  # stencil enable 2
            ("depth/stencil row", {24: [bytes([1, 1, 9, 1]).ljust(24, b"\0")]}),  # compare function 9
            ("blend row", {24: [good_depth], 68: [bytes([0, 0, 0, 0, 2]).ljust(68, b"\0")]}),  # target enable 2
            ("blend row", {68: [bytes(4 + 8 * 3 + 7) + bytes([16]).ljust(68 - 35, b"\0")]}),  # target 3 mask 16
        )
        for message, tables in cases:
            with self.subTest(message=message):
                data, layout, index = build([(1, 1, 0, 0, 0, 0)], tables)
                with self.assertRaisesRegex(ValueError, message):
                    self.census(data, layout, index)

    def test_states_file_lists_distinct_rows_with_counts(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sgd5_effects, "DATABASES", [self.layout]):
            folder = Path(folder)
            (folder / "a.rpf").write_bytes(self.data)
            (folder / "index.json").write_text(json.dumps(self.index), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                sgd5_pass_pipelines.main([str(folder / "a.rpf"), str(folder / "index.json"), "--out",
                                          str(folder / "m.json"), "--states", str(folder / "s.json")])
            doc = json.loads((folder / "s.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(doc), ["blend", "depth", "raster"])
        self.assertEqual(sorted(x["passes"] for x in doc["raster"]), [1, 5])
        self.assertEqual({x["state"]["cull"]: x["passes"] for x in doc["raster"]}, {"front": 1, "none": 5})

    def test_conflicting_index_entries_are_refused_in_either_order(self):
        clash = dict(self.index[2], file="Pix2.dxbc")
        for index in (self.index + [clash], [clash] + self.index):
            with self.subTest(first=index[0]["file"]):
                with patch.object(sgd5_effects, "DATABASES", [self.layout]), self.assertRaisesRegex(
                        ValueError, "two different entries"):
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / "a.rpf"
                        path.write_bytes(self.data)
                        sgd5_pass_pipelines.census(str(path), index)
        # The same entry listed twice is not a conflict.
        with tempfile.TemporaryDirectory() as folder, patch.object(sgd5_effects, "DATABASES", [self.layout]):
            path = Path(folder) / "a.rpf"
            path.write_bytes(self.data)
            rows, _ = sgd5_pass_pipelines.census(str(path), self.index + [dict(self.index[2])])
        self.assertEqual(len(rows), 5)

    def test_spirv_dir_leaves_out_missing(self):
        manifest, _, summary = self.run_tool(["Vert", "Pix", "Geo"])
        self.assertEqual(self.sets(manifest), [{"vs": "Vert", "ps": "Pix"}])
        self.assertIn("1  skipped: no .spv", summary)

    def test_nothing_left_is_an_error_not_an_empty_manifest(self):
        # `batch` refuses an empty pipeline list, so the tool fails here instead of handing one over.
        with redirect_stderr(io.StringIO()) as err:
            _, _, summary = self.run_tool([], status=1)
        self.assertIn("2  skipped: no .spv", summary)
        self.assertIn("no pipeline left", err.getvalue())

    def test_only_stage_sets_the_link_check_takes(self):
        # An ms+gs+ps set would make `shader_link_check.py batch` refuse the whole manifest, so it is skipped by rule.
        rows = [{"keys": {"ms": "m", "gs": "g", "ps": "p"}}, {"keys": {"vs": "v", "gs": "g", "ps": "p"}}]
        counts = collections.Counter()
        pipes = sgd5_pass_pipelines.pipelines_from(rows, counts)
        self.assertEqual([sorted(k for k in p if k != "name") for p in pipes], [["gs", "ps", "vs"]])
        self.assertEqual(counts["skipped: a stage set shader_link_check does not take"], 1)


if __name__ == "__main__":
    unittest.main()
