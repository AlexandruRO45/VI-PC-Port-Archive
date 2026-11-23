"""Synthetic integration coverage for the local viewer catalog."""
from pathlib import Path
from io import StringIO
import json
import os
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

    def write_export(self, archive, rows, rejected=0, root=None, scan=None, src=None):
        folder = (root or self.exports) / archive / "models"
        folder.mkdir(parents=True, exist_ok=True)
        for row in rows:
            filename = row["File"]
            if Path(filename).name == filename and not row.get("Missing"):
                (folder / filename).write_text(OBJ, encoding="utf-8", newline="\n")
        (folder / "manifest.json").write_text(
            json.dumps({"src": str(src), "Count": len(rows), "Scan": {"NotRecovered": rejected, **(scan or {})},
                        "Items": rows} if src else
                       {"Count": len(rows), "Scan": {"NotRecovered": rejected}, "Items": rows}),
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

    def window_setup(self, windows=3, ignore=False):
        game = self.root / "synthetic-package"
        source = game / "levels.rpf"
        source.parent.mkdir(parents=True)
        with source.open("wb") as f:
            f.truncate(windows * viewer_catalog.MIB - 5)
        calls = []

        def export(command, src, exports, reserve, window=None):
            calls.append(window)
            start = window[0] // viewer_catalog.MIB
            rows = [{"File": f"{name}.obj", "Name": name} for name in (f"w{start}_a", f"w{start}_same")]
            asked = {"Start": window[0], "Length": window[1]}
            folder = self.write_export("levels", rows, root=exports, src=src,
                                       scan=asked if not ignore else {})
            # Same filename in every window: the window must keep the converted paths apart.
            (folder / "w_same.obj").write_text(OBJ, encoding="utf-8")
            self.assertEqual(window[1], min(viewer_catalog.MIB, src.stat().st_size - window[0]))

        patcher = patch.multiple(viewer_catalog, free_space=lambda *args: None,
                                 local_exporter=lambda *args: ["fake"], export_models=export)
        return game, calls, patcher

    def windowed(self, game, *extra):
        return self.build("--game", str(game), "--archive", "levels.rpf", "--window-mib", "1", *extra)

    def test_windowed_import_releases_each_window_and_resumes_from_checkpoint(self):
        game, calls, patcher = self.window_setup()
        with patcher:
            self.assertEqual(self.windowed(game), 0)
            result = self.catalog()
            self.assertTrue(result["status"]["complete"])
            self.assertEqual(len(result["assets"]), 6)
            self.assertEqual(len({asset["model"] for asset in result["assets"]}), 6)
            self.assertEqual(len(calls), 3)
            self.assertEqual([c[0] for c in calls], [0, 1048576, 2097152])
            state = json.loads((self.out / ".window-state.json").read_text())
            self.assertEqual(state["archives"]["levels.rpf"]["completed"], [0, 1048576, 2097152])
            staging = self.work / "rpf9dump-models"
            self.assertEqual([p for p in staging.rglob("*.obj")], [])
            self.assertEqual(self.windowed(game), 0)
        self.assertEqual(len(calls), 3)
        again = self.catalog()
        self.assertEqual(sorted(a["name"] for a in again["assets"]), sorted(a["name"] for a in result["assets"]))

    def test_windowed_import_keeps_finished_windows_when_storage_runs_out(self):
        game, calls, patcher = self.window_setup()
        budget = iter([True, False])

        def space(folder, reserve, extra=0):
            if extra >= viewer_catalog.MIB * viewer_catalog.WINDOW_OBJ_FACTOR - 20 and not next(budget, False):
                raise viewer_catalog.SpaceError("synthetic full disk")

        with patcher, patch.object(viewer_catalog, "free_space", space):
            self.assertEqual(self.windowed(game), 1)
            partial = self.catalog()
            self.assertFalse(partial["status"]["complete"])
            self.assertEqual(len(partial["assets"]), 2)
            self.assertIn("levels windows complete: 1/3", partial["status"]["message"])
            self.assertIn("archives deferred for free space", partial["status"]["message"])
        self.assertEqual(len(calls), 1)
        with patcher:
            self.assertEqual(self.windowed(game), 0)
        self.assertEqual([c[0] for c in calls], [0, 1048576, 2097152])
        self.assertTrue(self.catalog()["status"]["complete"])
        self.assertEqual(len(self.catalog()["assets"]), 6)

    def test_windowed_pause_keeps_window_partial_until_converted(self):
        game, calls, patcher = self.window_setup(2)
        with patcher:
            self.assertEqual(self.windowed(game, "--max-new", "1"), 0)
            paused = self.catalog()
            self.assertFalse(paused["status"]["complete"])
            self.assertEqual(len(paused["assets"]), 1)
            self.assertFalse((self.out / ".window-state.json").exists())
            self.assertEqual(self.windowed(game), 0)
        self.assertTrue(self.catalog()["status"]["complete"])
        self.assertEqual(len(self.catalog()["assets"]), 4)

    def test_windowed_import_retains_manifest_and_glb_map_without_duplicates(self):
        game, calls, patcher = self.window_setup()
        index = self.work / "window-manifests" / "levels"
        with patcher:
            self.assertEqual(self.windowed(game), 0)
            names = sorted(p.name for p in index.iterdir())
            self.assertEqual(names, [f"{n:020d}-{min(2**20, 3 * 2**20 - 5 - n):020d}" for n in (0, 2**20, 2 * 2**20)])
            document = json.loads((index / names[1] / "catalog-map.json").read_text())
            self.assertEqual(document["window"], {"start": 2**20, "length": 2**20})
            self.assertEqual(document["source"]["archive"], "levels.rpf")
            self.assertEqual(document["source"]["size"], 3 * 2**20 - 5)
            import hashlib
            raw = (index / names[1] / "manifest.json").read_bytes()
            self.assertEqual(document["manifest"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads(raw)["Scan"]["Start"], 2**20)
            self.assertEqual({m["status"] for m in document["models"]}, {"ready"})
            self.assertTrue(all((self.out / unquote(m["glb"])).is_file() for m in document["models"]))
            self.assertFalse(list(self.out.rglob("catalog-map.json")))
            before = {p: p.read_bytes() for p in index.rglob("*") if p.is_file()}
            (self.out / ".window-state.json").unlink()  # force every window to be processed again
            self.assertEqual(self.windowed(game), 0)
        self.assertEqual({p: p.read_bytes() for p in index.rglob("*") if p.is_file()}, before)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(self.catalog()["assets"]), 6)

    def test_failed_retention_blocks_checkpoint_and_release(self):
        game, calls, patcher = self.window_setup(2)
        with patcher, patch.object(viewer_catalog, "retain_window", side_effect=OSError("synthetic")):
            self.assertEqual(self.windowed(game), 1)
        self.assertFalse((self.out / ".window-state.json").exists())
        self.assertTrue(list((self.work / "rpf9dump-models").rglob("*.obj")))
        self.assertFalse((self.work / "window-manifests").exists())
        self.assertIn("window index not retained", self.catalog()["status"]["message"])

    def test_exporter_ignoring_window_is_rejected_before_models_or_checkpoint(self):
        game, calls, patcher = self.window_setup(2, ignore=True)
        with patcher:
            self.assertEqual(self.windowed(game), 1)
        result = self.catalog()
        self.assertEqual(result["assets"], [])
        self.assertFalse(result["status"]["complete"])
        self.assertFalse((self.out / ".window-state.json").exists())
        self.assertTrue(list((self.work / "rpf9dump-models").rglob("manifest.json")))
        self.assertEqual(list(self.out.rglob("*.glb")), [])

    def test_same_size_mutated_source_refuses_checkpoint_and_keeps_assets(self):
        game, calls, patcher = self.window_setup()
        with patcher:
            self.assertEqual(self.windowed(game), 0)
            glbs = sorted(self.out.rglob("*.glb"))
            os.utime(game / "levels.rpf", ns=(1, 1))
            self.assertEqual(self.windowed(game), 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sorted(self.out.rglob("*.glb")), glbs)
        self.assertEqual(len(self.catalog()["assets"]), 6)
        self.assertIn("stale window checkpoint refused", self.catalog()["status"]["message"])

    def test_release_only_removes_exact_owned_window_folder(self):
        root = self.root / "x" / "windows" / "levels"
        (root / "0-5").mkdir(parents=True)
        other = self.root / "x" / "windows" / "other"
        other.mkdir()
        for bad in (other, root, root / "0-5" / "..", root / "not-a-window", self.root / "x"):
            with self.assertRaises(OSError):
                viewer_catalog.release(bad, root)
        self.assertTrue(other.is_dir() and root.is_dir())
        viewer_catalog.release(root / "5-5", root)
        viewer_catalog.release(root / "0-5", root)
        self.assertFalse((root / "0-5").exists())

    def test_window_options_need_game_and_default_stays_whole_file(self):
        with self.assertRaises(SystemExit), patch("sys.stderr", new=StringIO()):
            self.build("--window-mib", "1")
        command = []
        with patch.object(viewer_catalog.subprocess, "Popen") as popen:
            popen.return_value.__enter__.return_value.wait.return_value = 0
            viewer_catalog.export_models(["x"], Path("s.rpf"), Path("o"), 0)
            viewer_catalog.export_models(["x"], Path("s.rpf"), Path("o"), 0, window=(5, 7))
        self.assertEqual(popen.call_args_list[0].args[0], ["x", "models", "s.rpf", "o"])
        self.assertEqual(popen.call_args_list[1].args[0][-4:], ["--start", "5", "--length", "7"])

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
