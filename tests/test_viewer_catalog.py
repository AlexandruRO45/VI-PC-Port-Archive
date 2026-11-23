"""Synthetic integration coverage for the local viewer catalog."""
from pathlib import Path
from io import StringIO
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dds_png
import viewer_catalog

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
PNG = dds_png.encode_png(1, 1, bytes((0, 128, 255, 255)))


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.root / "catalog"
        self.exports = self.root / "exports"
        self.exports.mkdir()
        self.work = self.root / "work"

    def write_export(self, archive, rows, rejected=0, root=None):
        folder = (root or self.exports) / archive / "models"
        folder.mkdir(parents=True, exist_ok=True)
        for row in rows:
            filename = row["File"]
            if Path(filename).name == filename and not row.get("Missing"):
                (folder / filename).write_text(OBJ, encoding="utf-8", newline="\n")
        (folder / "manifest.json").write_text(
            json.dumps({"Count": len(rows), "Scan": {"NotRecovered": rejected}, "Items": rows}),
            encoding="utf-8", newline="\n",
        )
        return folder

    def write_glb(self, path):
        tool = viewer_catalog.glb_writer
        tool.write_glb(str(path), tool.parse_obj(OBJ, is_path=False))

    def catalog(self):
        return json.loads((self.out / "catalog.json").read_text(encoding="utf-8"))

    def build(self, *args):
        return viewer_catalog.main([
            "--exports", str(self.exports), "--out", str(self.out), "--work", str(self.work), *args,
        ])

    def test_reuses_legacy_glb_and_indexes_ped_vehicle_and_preview(self):
        self.write_export("data", [
            {"File": "Crate +#.obj", "Name": "Crate"},
            {"File": "skip.obj", "Name": "Empty Bounds", "IsFragment": True, "Verts": 25, "Tris": 32},
            {"File": "roadster_hi.obj", "Name": "roadster_hi.#ft", "IsFragment": True, "Verts": 800, "Tris": 400},
        ])
        ped = self.write_export("packs", [
            {"File": "head_fr1.obj", "Name": "head_fr1"},
            {"File": "neutral.obj", "Name": "Other Drawable"},
            {"File": "head_fr1.obj", "Name": "Duplicate LOD"},
        ])
        (ped / "head_fr1.shader.png").write_bytes(PNG)
        (self.exports / "vehicle-names.txt").write_text("roadster.yft\n", encoding="utf-8", newline="\n")
        crate = self.out / "props" / "data" / "Crate +#.glb"
        self.write_glb(crate)
        self.write_glb(crate.with_name("skip.glb"))
        crate.with_name("Crate +#.shader.png").write_bytes(PNG)
        (self.exports / "data" / "models" / "Crate +#.obj").unlink()
        before = crate.stat().st_mtime_ns
        with patch.object(viewer_catalog, "export_models", side_effect=AssertionError("reuse must not export")):
            self.assertEqual(self.build(), 0)
        self.assertEqual(crate.stat().st_mtime_ns, before)
        result = self.catalog()
        self.assertEqual(result["version"], 1)
        self.assertTrue(result["status"]["complete"])
        self.assertEqual(result["status"]["processed"], 4)
        self.assertIn("1 fragment or empty rows skipped", result["status"]["message"])
        assets = {entry["name"]: entry for entry in result["assets"]}
        self.assertEqual(
            assets["Crate"],
            {"name": "Crate", "category": "props", "model": "props/data/Crate%20%2B%23.glb",
             "preview": "props/data/Crate%20%2B%23.shader.png"},
        )
        self.assertEqual(assets["head_fr1"]["category"], "ped-parts")
        self.assertEqual(assets["Other Drawable"]["category"], "props")
        self.assertEqual(assets["roadster_hi.#ft"]["category"], "vehicles")
        self.assertTrue(all((self.out / unquote(entry["model"])).is_file() for entry in assets.values()))
        self.assertEqual((self.out / unquote(assets["head_fr1"]["preview"])).read_bytes(), PNG)

    def test_checkpoint_resume_preserves_glb_and_picks_up_new_sidecar(self):
        folder = self.write_export("data", [{"File": "a.obj"}, {"File": "b.obj"}])
        self.assertEqual(self.build("--batch-size", "1", "--max-new", "1"), 0)
        first = self.catalog()
        self.assertFalse(first["status"]["complete"])
        self.assertEqual((first["status"]["processed"], first["status"]["total"]), (1, 2))
        model = self.out / unquote(first["assets"][0]["model"])
        before = model.stat().st_mtime_ns
        (folder / "a.shader.png").write_bytes(PNG)
        self.assertEqual(self.build(), 0)
        result = self.catalog()
        self.assertTrue(result["status"]["complete"])
        self.assertEqual(result["status"]["processed"], 2)
        self.assertEqual(model.stat().st_mtime_ns, before)
        self.assertIn("preview", next(entry for entry in result["assets"] if entry["name"] == "a"))

    def test_reports_missing_and_unrecovered_exports_but_publishes_ready_models(self):
        self.write_export("data", [{"File": "ready.obj"}, {"File": "missing.obj", "Missing": True}], rejected=2)
        self.assertEqual(self.build(), 1)
        result = self.catalog()
        self.assertFalse(result["status"]["complete"])
        self.assertIn("2 unrecovered candidates", result["status"]["message"])
        self.assertIn("1 models failed", result["status"]["message"])
        self.assertEqual([asset["name"] for asset in result["assets"]], ["ready"])

    def test_exports_missing_selected_archives_in_order_then_reuses_them(self):
        self.write_export("data", [{"File": "prepared.obj"}])
        game = self.root / "synthetic-package"
        story = "pack/story000/story000.rpf"
        for relative in (story, "levels.rpf"):
            source = game / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.touch()
        called = []

        def export(command, source, exports, reserve):
            ready = [entry["name"] for entry in self.catalog()["assets"]]
            self.assertIn("prepared", ready)
            self.assertEqual(command, ["fake-rpf9dump"])
            called.append(source.relative_to(game).as_posix())
            self.write_export(source.stem, [{"File": f"{source.stem}.obj"}], root=exports)

        flags = ["--game", str(game), "--archive", "data.rpf", "--archive", story, "--archive", "levels.rpf"]
        with patch.multiple(viewer_catalog, free_space=lambda *args: None,
                            local_exporter=lambda *args: ["fake-rpf9dump"], export_models=export):
            self.assertEqual(self.build(*flags), 0)
            self.assertEqual(self.build(*flags), 0)
        self.assertEqual(called, [story, "levels.rpf"])
        self.assertEqual(len(self.catalog()["assets"]), 3)

    def test_space_preflight_keeps_ready_assets_and_skips_exporter(self):
        self.write_glb(self.out / "ready.glb")
        game = self.root / "synthetic-package"
        game.mkdir()
        (game / "levels.rpf").touch()
        with patch.multiple(
            viewer_catalog, local_exporter=lambda *args: ["fake-rpf9dump"],
            export_models=lambda *args: self.fail("exporter launched without space"),
        ), patch.object(viewer_catalog.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
            self.assertEqual(self.build("--game", str(game), "--archive", "levels.rpf"), 1)
        result = self.catalog()
        self.assertFalse(result["status"]["complete"])
        self.assertIn("levels", result["status"]["message"])
        self.assertEqual([asset["name"] for asset in result["assets"]], ["ready"])

    def test_streamed_manifest_reads_rows_and_trailing_metadata_across_chunks(self):
        rows = [{"File": f"{i}.obj", "metadata": "m" * 20000} for i in range(10)]
        document = {"Count": len(rows), "Items": rows, "ModelDecodeFailures": {"unreadable": 4}}
        reader = viewer_catalog.ManifestReader(StringIO(json.dumps(document)))
        self.assertEqual(list(reader.rows()), rows)
        self.assertEqual(reader.metadata, {"Count": 10, "ModelDecodeFailures": {"unreadable": 4}})

    def test_exports_command_runs_in_isolated_python(self):
        self.write_export("data", [{"File": "box.obj"}])
        command = [
            sys.executable, "-I", viewer_catalog.__file__, "--exports", str(self.exports),
            "--out", str(self.out), "--work", str(self.work),
        ]
        run = subprocess.run(command, text=True, capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(self.catalog()["status"]["complete"])


if __name__ == "__main__":
    unittest.main()
