"""Exercise the local parts index with synthetic exporter documents."""
from contextlib import closing, redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import viewer_parts_index as index


ARCHIVE = r"Q:\Local\pack\sample.rpf"
CRATE_JOAAT = "80644999"
CRATE_KEY = "DEED14E77056AFE2"
LOWER_KEY = "0B13518CD3D39B79"
ASSET_KEY = "FEDCBA9876543210"


def model(offset, name="Crate.#dr", drawable=None, **fields):
    return {"Offset": offset, "Drawable": drawable, "File": f"part-{offset}-{drawable}.obj",
            "Name": name, "Verts": 3, "Tris": 1, "MeshShaders": [7], **fields}


def entity(archetype=CRATE_JOAAT, key=None, x=1):
    row = {"archetype": archetype, "position": [x, 2, 3], "rotation": [0, 0, 0, 1],
           "scale_xy": 1, "scale_z": 2, "lod_level": 4, "map_name": "abcdef00", "entity_index": 5}
    if key is not None:
        row["key"] = key
    return row


class PartsIndexTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.root / "index" / "parts.sqlite"

    def write(self, name, value, raw=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value, ensure_ascii=False), encoding="utf-8", newline="\n")
        return path

    def manifest(self, directory, rows, start=128, length=2048, src=ARCHIVE):
        scan = {"Entries": len(rows) + 1, "Candidates": len(rows) + 2, "NotRecovered": 1}
        if start is not None:
            scan.update(Start=start, Length=length)
        return self.write(f"{directory}/manifest.json", {
            "src": src, "Count": len(rows) + 1, "Scan": scan, "Items": rows,
            "ModelDecodeFailures": {"synthetic": 1},
        })

    def entities(self, directory, rows):
        return self.write(f"{directory}/entities.jsonl", "".join(
            (json.dumps(row) if isinstance(row, dict) else row) + "\n" for row in rows
        ), raw=True)

    def report(self, directory, start=128):
        return self.write(f"{directory}/report.json", {"resources": [{"offset": start}], "summary": {
            "path": ARCHIVE, "bytes": 8192, "scan_start": start, "scan_end": start + 1024,
            "streams": 2, "scanned_bytes": 420,
        }})

    def run_index(self, *args, result=0, output=None):
        with redirect_stdout(StringIO()) as out, redirect_stderr(StringIO()) as err:
            code = index.main(["--out", str(output or self.out), *map(str, args)])
        self.assertEqual(code, result, err.getvalue())
        return json.loads(out.getvalue()) if result == 0 else err.getvalue()

    def read(self, sql, args=()):
        with closing(sqlite3.connect(self.out)) as db:
            return db.execute(sql, args).fetchall()

    def test_model_identity_and_exact_hash_semantics(self):
        ordinary = self.manifest("models/a", [model(140), model(140, "crate.#dd", 512),
                                              model(165, "  Caf\u00e9.#dr  ")])
        copy = self.manifest("models/b", [model(140), model(140, "crate.#dd", 512),
                                          model(165, "  Caf\u00e9.#dr  ")])
        distinct = self.manifest("models/c", [model(140)], src=r"Q:\Local\other\sample.rpf")
        entities = self.entities("world/a", [entity(key=LOWER_KEY), entity(key=CRATE_KEY), entity(),
            entity(key="7777777777777777"), entity("DEADBEEF"), entity("EEFBF723", "EB272F1D9018080E"),
            entity("EEFBF723", "bad key"), entity("EEFBF723", "10000000000000000"),
            entity(x=100), "[invalid", {"archetype": CRATE_JOAAT, "position": "123"}])
        report = self.report("world/a")
        entities_copy = self.entities("world/copy", [])
        entities_copy.write_bytes(entities.read_bytes())
        another_region = self.report("world/b", start=2048)
        args = ("--models", copy, "--models", distinct, "--models", ordinary, "--models", ordinary,
                "--world-window", entities, report, "--world-window", entities_copy, report,
                "--world-window", self.entities("world/b", [entity(x=2)]), another_region,
                "--region", 0, 0, 10, 10)
        totals = self.run_index(*args)
        self.assertEqual(totals["models"], 4)
        self.assertEqual(totals["placements"], 9)
        self.assertEqual(totals["input_duplicate_models"], 3)
        self.assertEqual(totals["duplicate_placement_inputs"], 1)
        self.assertEqual(totals["input_invalid_placements"], 2)
        self.assertEqual(totals["input_outside_region"], 1)
        self.assertEqual(self.read("""SELECT p.line, p.key_status, m.status, m.eligible_candidates
            FROM placements p JOIN placement_matches m USING (key_id)
            WHERE p.input_id = (SELECT id FROM inputs WHERE path=?) ORDER BY p.line""", (str(entities.resolve()),)), [
            (1, "present", "verified", 1), (2, "present", "ambiguous", 2), (3, "absent", "ambiguous", 3),
            (4, "present", "key_mismatch", 0), (5, "absent", "unresolved_archetype", 0),
            (6, "present", "verified", 1), (7, "invalid", "unverified", 1), (8, "present", "key_mismatch", 0),
        ])
        self.assertEqual(self.read("SELECT size FROM archives WHERE archive='q:/local/pack/sample.rpf'"), [(8192,)])
        self.assertEqual(self.read("PRAGMA foreign_key_check"), [])
        before = self.out.read_bytes()
        self.assertEqual(self.run_index(*args), totals)
        self.assertEqual(self.out.read_bytes(), before)

    def test_material_slots_embedded_precedence_and_ttpf_candidates(self):
        model_row = model(200, Materials=[{"ShaderIndex": 7, "Shader": "33333333", "Textures": [
            {"Param": "11223344", "TextureIndex": 2, "Id64": ASSET_KEY}, {"Id64": ASSET_KEY},
            {"TextureIndex": 90, "Name": "99887766"}, {"Id64": "9999999999999999"},
            {"Id64": "9999999999999999", "TextureIndex": -1},
        ]}], Embedded=[{"Index": 2, "Hash": "99887766", "Key": ASSET_KEY}])
        manifest = self.manifest("models/pool", [model_row])
        self.write("models/pool/part-200-None.obj", "synthetic export", raw=True)
        self.write("models/pool/part-200-None_tex/002_embedded.dds", "fixture", raw=True)
        ttpf = self.write("textures/manifest.json", {"Textures": [
            {"AssetKey": ASSET_KEY.lower(), "File": "extra/a.dds"},
            {"AssetKey": int(ASSET_KEY, 16), "File": r"extra\b.dds"},
            {"AssetKey": ASSET_KEY, "File": "extra/a.dds"},
            {"AssetKey": "9999999999999999", "File": "gone.dds"},
            {"AssetKey": "9999999999999999", "File": None},
            {"AssetKey": "9999999999999999", "File": "../escape.dds"},
        ], "Count": 6})
        self.write("textures/extra/a.dds", "fixture", raw=True)
        self.write("textures/extra/b.dds", "fixture", raw=True)
        self.write("escape.dds", "fixture", raw=True)
        totals = self.run_index("--models", manifest, "--textures", ttpf)
        self.assertEqual(self.read("SELECT shader_index, shader FROM materials"), [(7, "33333333")])
        self.assertEqual(self.read("SELECT param FROM texture_refs WHERE ordinal=0"), [("11223344",)])
        self.assertEqual(self.read("SELECT priority,candidates,status FROM texture_matches ORDER BY ref_id"), [
            (1, 1, "available"), (2, 1, "available"), (3, 1, "available"),
            (None, 0, "missing"), (None, 0, "missing"),
        ])
        self.assertEqual(self.read("SELECT count(*) FROM texture_candidates WHERE priority=4"), [(4,)])
        self.assertEqual(totals["input_texture_files_invalid"], 1)
        self.assertEqual(totals["input_texture_files_missing"], 1)
        self.assertEqual(totals["input_texture_files_unexported"], 1)
        # With no embedded export on disk, the same references expose both distinct DDS candidates.
        (manifest.parent / "part-200-None_tex" / "002_embedded.dds").unlink()
        totals = self.run_index("--models", manifest, "--textures", ttpf)
        self.assertEqual(totals["texture_refs_ambiguous"], 2)
        self.assertEqual(self.read("SELECT priority,candidates FROM texture_matches WHERE status='ambiguous'"),
                         [(4, 2), (4, 2)])

    def test_window_commit_marker_joins_ready_glbs_and_deduplicates_models(self):
        rows = [model(180, File="box #.obj"), model(200, File="shared.obj"),
                model(220, File="shared.obj"), model(250, File="old.obj"), model(5000)]
        ordinary = self.manifest("models/whole", rows[:1], start=None)
        directory = "work/window-manifests/sample/00000000000000000128-00000000000000002048"
        manifest = self.manifest(directory, rows)
        catalog = self.root / "served"
        glb = self.write("served/models/box #.glb", "fixture", raw=True)
        marker_data = {
            "version": 1, "kind": "leonida-window-parts",
            "source": {"archive": "pack/sample.rpf", "size": 8192, "mtime_ns": 12345},
            "window": {"start": 128, "length": 2048},
            "manifest": {"file": "manifest.json", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
            "models": [{"file": file, "name": "Crate", "category": "props", "status": status, "glb": path}
                       for file, status, path in [
                           ("box #.obj", "ready", "models/box%20%23.glb"), ("shared.obj", "ready", "shared.glb"),
                           ("shared.obj", "duplicate", None), ("old.obj", "ready", "models/gone.glb"),
                       ]],
        }
        marker = self.write(f"{directory}/catalog-map.json", marker_data)
        args = ("--models", ordinary, "--window", marker, "--catalog-root", catalog)
        totals = self.run_index(*args)
        self.assertEqual(totals["models"], 4)
        self.assertEqual(totals["input_out_of_window"], 1)
        self.assertEqual(totals["input_catalog_ambiguous"], 2)
        self.assertEqual(totals["input_catalog_missing"], 1)
        self.assertEqual(self.read("SELECT m.offset,a.path FROM model_assets a JOIN models m ON m.id=a.model_id"),
                         [(180, index.existing(glb.resolve()))])
        self.assertEqual(self.read("SELECT size,mtime_ns FROM archives"), [(8192, 12345)])
        before = self.out.read_bytes()
        self.assertIn("commit marker", self.run_index("--models", manifest, result=2))
        self.manifest(directory, rows + [model(170)])
        self.assertIn("SHA-256 differs", self.run_index(*args, result=2))
        self.assertEqual(self.out.read_bytes(), before)

    def test_dictionary_export_inventories_dds_without_inferred_matches(self):
        manifest = self.manifest("models/dictionary-user", [model(200, Materials=[{
            "ShaderIndex": 7, "Textures": [{"Name": "11223344", "Id64": ASSET_KEY}],
        }])])
        dictionary = self.manifest("dictionaries", [{"Offset": 200, "Dir": "sample", "Textures": [
            {"Index": 0, "Name": "test", "Hash": "11223344", "Key": None, "Width": 4, "Height": 8},
            {"Index": 1, "Name": "other", "Hash": "11223344", "Key": "0000000000000000"},
            {"Index": 2, "Name": "keyed", "Key": ASSET_KEY},
        ]}], src=r"Q:\Local\textures.rpf")
        for name in ("000_a.dds", "000_b.dds", "002_a.dds"):
            self.write(f"dictionaries/sample/{name}", "fixture", raw=True)
        totals = self.run_index("--models", manifest, "--texture-dictionaries", dictionary)
        self.assertEqual(totals["input_dictionary_textures_ambiguous"], 1)
        self.assertEqual(totals["input_dictionary_textures_missing"], 1)
        self.assertEqual(totals["texture_refs_missing"], 1)
        self.assertEqual(self.read("SELECT texture_index,asset_key,status FROM dictionary_textures ORDER BY texture_index"),
                         [(0, None, "ambiguous"), (0, None, "ambiguous"), (1, None, "missing"),
                          (2, ASSET_KEY, "ready")])
        self.assertEqual(self.read("SELECT * FROM texture_candidates"), [])

    def test_failure_preserves_previous_database_and_source_tree(self):
        good = self.manifest("source/good", [model(200)])
        self.run_index("--models", good)
        before = self.out.read_bytes()
        bad = self.manifest("source/bad", [model(200, "Conflicting")])
        self.assertIn("conflicting model metadata", self.run_index("--models", good, "--models", bad, result=2))
        broken = self.write("source/broken/manifest.json", '{"src":"/u/test.rpf","Items":[', raw=True)
        self.run_index("--models", broken, result=2)
        self.assertEqual(self.out.read_bytes(), before)
        self.assertIn("outside", self.run_index("--models", good, output=good.with_suffix(".sqlite"), result=2))
        self.assertIn("outside", self.run_index("--models", good, "--source-root", self.root / "source",
                                                output=self.root / "source" / "index.sqlite", result=2))

    def test_isolated_cli_accepts_separate_model_and_placement_exports(self):
        models = self.manifest("model_export", [model(320)])
        placements = self.entities("placement_export", [entity(key=CRATE_KEY)])
        report = self.report("placement_export")
        proc = subprocess.run([sys.executable, "-I", index.__file__, "--out", str(self.out), "--models", str(models),
                               "--world-window", str(placements), str(report)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["placements_verified"], 1)


if __name__ == "__main__":
    unittest.main()
