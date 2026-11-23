"""End-to-end textured kit export using two hand-built outfits and a local parts index."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
import anim_glb
import dds_png
import glb_skinning
import glb_writer
import ped_catalogues
import skinned_glb
import test_dds_png as dds_fixture
import test_ped_catalogues as catalogue_fixture
import test_skinned_glb as skin_fixture


# Same low 32 bits. Looking up these IDs with truncated integers selects the wrong jacket.
WARM = 0xFA00000000000011
COOL = 0xCF00000000000011
TROUSERS = 0xBD00000000000012
NORMAL = 0xAB00000000000013
PART_NAMES = ("synthetic_trousers", "synthetic_jacket")


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")


def make_part(folder, name, outline, indices):
    count = len(outline)
    # Give both weight sets nonzero contributions using the fixture's two-entry palette.
    stride = (4 * count + 15) & ~15
    offsets = {slot: 256 + (slot - 6) * stride for slot in range(6, 10)}
    buffer = skin_fixture._skin_buffer(
        bytes((0, 1, 0, 0)) * count, bytes((1, 0, 0, 0)) * count,
        bytes((160, 63, 0, 0)) * count, bytes((32, 0, 0, 0)) * count,
        offsets=offsets, size=256 + 4 * stride,
    )
    part = Path(skin_fixture._component(folder, buffer=buffer))
    model = skin_fixture._read_model(part)
    model["drawable"]["name"] = name + ".#dr"
    model["vertex_count"] = model["meshes"][0]["vertex_count"] = count
    write_json(part / "model.json", model)
    (part / "model.glb").write_bytes(glb_writer.build_glb([{
        "positions": [(x, y, 0) for x, y in outline], "normals": [(0, 0, 1)] * count,
        "uvs": [((x + 1) / 2, (y + 1) / 2) for x, y in outline], "indices": indices,
    }]))
    return part


def make_kit_fixture(root: Path) -> dict:
    """Build decoded parts, manifests consumable by viewer_parts_index.py, and the equivalent test index."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    skeleton = Path(skin_fixture._skeleton_dir(root))
    parts = [
        make_part(root / "lower", PART_NAMES[0],
                  [(-.42, 0), (.42, 0), (0, -.38), (-.4, -.98), (-.08, -.98), (.08, -.98), (.4, -.98)],
                  [0, 2, 1, 0, 3, 4, 0, 4, 2, 1, 2, 5, 1, 5, 6]),
        make_part(root / "upper", PART_NAMES[1],
                  [(-.43, 0), (.43, 0), (.43, .6), (.28, .86), (.16, .90), (0, .77), (-.16, .90),
                   (-.28, .86), (-.43, .6), (-.77, .17), (-.92, .27), (.77, .17), (.92, .27)],
                  [0, 1, 2, 0, 2, 8, 8, 2, 5, 8, 5, 7, 7, 5, 6, 2, 3, 5, 3, 4, 5,
                   8, 7, 10, 8, 10, 9, 2, 11, 12, 2, 12, 3]),
    ]
    trousers, jacket = map(ped_catalogues.murmur64, PART_NAMES)
    pack = catalogue_fixture._pack_state
    blocks = catalogue_fixture.build_kit_blocks(
        kits=[(101, [(10, 0), (11, 1)]), (102, [(10, 0), (11, 2)])],
        presets=[(10, 0, [pack(state_id=0, drawable_hash=trousers, albedo_hash=TROUSERS, normal_hash=NORMAL)]),
                 (11, 12, [pack(state_id=1, albedo_hash=WARM), pack(state_id=2, albedo_hash=COOL)]),
                 (12, 0, [pack(state_id=i, drawable_hash=jacket, normal_hash=NORMAL) for i in (1, 2)])],
    )
    catalogue = root / "outfits.0drp"
    catalogue.write_bytes(catalogue_fixture.build_synthetic_0drp(blocks))

    rows = []
    for i, (part, name) in enumerate(zip(parts, PART_NAMES), 1):
        rows.append({"Offset": 128 * i, "Drawable": None, "Name": name + ".#dr",
                     "Glb": (part / "model.glb").relative_to(root).as_posix(),
                     "File": (part / "model.obj").relative_to(root).as_posix(),
                     "MeshShaders": [0], "Materials": [{"ShaderIndex": 0, "Textures": []}]})
    models_manifest = root / "models.json"
    archive = "/Synthetic archives/character packs.rpf"
    write_json(models_manifest, {"src": archive, "Count": len(rows), "Items": rows})

    texture_dir = root / "textures"
    texture_dir.mkdir()
    colors = {
        TROUSERS: ((29, 43, 69), (52, 71, 101)), NORMAL: ((128, 128, 255), (141, 128, 254)),
        WARM: ((212, 63, 35), (255, 164, 63)), COOL: ((11, 106, 144), (78, 204, 197)),
    }
    pixels, png = {}, {}
    for key, palette in colors.items():
        pixels[key] = b"".join(bytes((*palette[(x // 4 + y // 4) % 2], 255))
                               for y in range(16) for x in range(16))
        png[key] = dds_png.encode_png(16, 16, pixels[key])
        (texture_dir / f"{key:016x}.dds").write_bytes(dds_fixture.dds_buffer(
            16, 16, pixels[key], flags=4, fourcc=b"DX10", dxgi=28))
    textures_manifest = texture_dir / "manifest.json"
    texture_rows = [{"AssetKey": f"{key:016X}", "File": f"{key:016x}.dds"} for key in colors]
    write_json(textures_manifest, {"Count": len(texture_rows), "Textures": texture_rows})

    index = root / "parts.sqlite"
    # The subset of the viewer parts index read by this CLI. Its producer is a separate tool.
    with closing(sqlite3.connect(index)) as db:
        db.executescript("""
            CREATE TABLE models(id INTEGER PRIMARY KEY, archive TEXT NOT NULL, offset INTEGER NOT NULL,
                                drawable INTEGER NOT NULL, name TEXT, murmur64 TEXT, UNIQUE(archive, offset, drawable));
            CREATE TABLE model_exports(input_id INTEGER, model_id INTEGER, file TEXT, obj_path TEXT,
                                       glb_path TEXT, glb_status TEXT);
            CREATE VIEW model_assets AS
                WITH files AS (SELECT DISTINCT model_id, glb_path AS path FROM model_exports)
                SELECT model_id, path, COUNT(*) OVER (PARTITION BY path) AS models_sharing_path FROM files;
            CREATE TABLE materials(model_id INTEGER, shader_index INTEGER);
            CREATE TABLE ttpf_textures(input_id INTEGER, asset_key TEXT, file TEXT, dds_path TEXT, status TEXT);
        """)
        for i, (row, key) in enumerate(zip(rows, (trousers, jacket)), 1):
            db.execute("INSERT INTO models VALUES (?,?,?,?,?,?)",
                       (i, archive, row["Offset"], -1, row["Name"], f"{key:016X}"))
            db.execute("INSERT INTO model_exports VALUES (1,?,?,NULL,?,'ready')",
                       (i, row["File"], str(root / row["Glb"])))
            db.execute("INSERT INTO materials VALUES (?,0)", (i,))
        for row in texture_rows:
            db.execute("INSERT INTO ttpf_textures VALUES (2,?,?,?,'ready')",
                       (row["AssetKey"], row["File"], str(texture_dir / row["File"])))
        db.commit()
    return dict(catalogue=catalogue, skeleton=skeleton, index=index, archive=archive, models_manifest=models_manifest,
                textures_manifest=textures_manifest, parts=parts, pixels=pixels, png=png)


class PedKitGlbTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.fixture = make_kit_fixture(self.root)

    def export(self, kit_id, out, index=None, *, sources=()):
        return subprocess.run([
            sys.executable, "-I", str(ROOT / "tools" / "ped_kit_glb.py"), str(self.fixture["catalogue"]),
            "--id", str(kit_id), "--index", str(index or self.fixture["index"]),
            "--skeleton", str(self.fixture["skeleton"]), "--out", str(out),
            *(arg for source in sources for arg in ("--model-source", *map(str, source))),
        ], capture_output=True, text=True, timeout=20)

    def image(self, gltf, binary, texture_info):
        self.assertEqual(texture_info.get("texCoord", 0), 0)
        texture = gltf["textures"][texture_info["index"]]
        image = gltf["images"][texture["source"]]
        self.assertNotIn("uri", image)
        self.assertEqual(image["mimeType"], "image/png")
        view = gltf["bufferViews"][image["bufferView"]]
        offset = view.get("byteOffset", 0)
        self.assertEqual(offset % 4, 0)
        return binary[offset:offset + view["byteLength"]]

    def test_two_kits_keep_skin_geometry_uv_and_select_distinct_jacket_images(self):
        neutral = self.root / "neutral.glb"
        skinned_glb.write_skinned_glb(str(neutral), str(self.fixture["skeleton"]), list(map(str, self.fixture["parts"])))
        base, base_binary = skinned_glb.read_glb(str(neutral))
        for kit, color in [(101, WARM), (102, COOL)]:
            with self.subTest(kit=kit):
                out = self.root / f"kit-{kit}.glb"
                result = self.export(kit, out)
                self.assertEqual(result.returncode, 0, result.stderr)
                gltf, binary = skinned_glb.read_glb(str(out))
                validation = glb_skinning.analyze_glb(out.read_bytes())
                self.assertTrue(validation["ok"], validation["errors"])
                self.assertEqual(validation["summary"]["skinned_primitive_count"], 2)
                self.assertEqual(gltf["nodes"], base["nodes"])
                self.assertEqual(gltf["skins"][0]["joints"], base["skins"][0]["joints"])
                self.assertEqual(
                    skinned_glb.accessor_bytes(gltf, binary, gltf["skins"][0]["inverseBindMatrices"])[0],
                    skinned_glb.accessor_bytes(base, base_binary, base["skins"][0]["inverseBindMatrices"])[0],
                )
                for i, selected in enumerate((TROUSERS, color)):
                    primitive = gltf["meshes"][0]["primitives"][i]
                    original = base["meshes"][0]["primitives"][i]
                    self.assertEqual(skinned_glb.accessor_bytes(gltf, binary, primitive["indices"])[0],
                                     skinned_glb.accessor_bytes(base, base_binary, original["indices"])[0])
                    for attr in ("POSITION", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0", "JOINTS_1", "WEIGHTS_1"):
                        actual = skinned_glb.accessor_bytes(gltf, binary, primitive["attributes"][attr])[0]
                        expected = skinned_glb.accessor_bytes(base, base_binary, original["attributes"][attr])[0]
                        self.assertEqual(actual, expected, attr)
                    material = gltf["materials"][primitive["material"]]
                    base_color = material["pbrMetallicRoughness"]["baseColorTexture"]
                    self.assertEqual(self.image(gltf, binary, base_color), self.fixture["png"][selected])
                    self.assertEqual(self.image(gltf, binary, material["normalTexture"]), self.fixture["png"][NORMAL])

    def test_bad_index_evidence_preserves_existing_output(self):
        key = f"{ped_catalogues.murmur64(PART_NAMES[1]):016X}"
        warm_key = f"{WARM:016X}"
        cases = [
            ("duplicate drawable identity", "INSERT INTO models SELECT 9,archive,offset+1,drawable,name,murmur64 "
             "FROM models WHERE murmur64=?",
             (key,), key, "ambiguous"),
            ("DDS not ready", "UPDATE ttpf_textures SET status='missing' WHERE asset_key=?",
             (warm_key,), warm_key, "no match"),
            ("two ready DDS paths", "INSERT INTO ttpf_textures SELECT 3,?,file,dds_path,status "
             "FROM ttpf_textures WHERE asset_key=?", (warm_key, f"{COOL:016X}"), warm_key, "ambiguous"),
            ("ambiguous shader slot", "INSERT INTO materials VALUES (2,?)", (7,), key, "ambiguous"),
        ]
        out = self.root / "previous.glb"
        out.write_bytes(b"previous complete export")
        for i, (label, sql, params, resource_id, reason) in enumerate(cases):
            with self.subTest(case=label):
                index = self.root / f"case-{i}.sqlite"
                shutil.copyfile(self.fixture["index"], index)
                with closing(sqlite3.connect(index)) as db:
                    db.execute(sql, params)
                    db.commit()
                result = self.export(101, out, index)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(resource_id.lower(), result.stderr.lower())
                self.assertIn(reason, result.stderr.lower())
                self.assertEqual(out.read_bytes(), b"previous complete export")

    def test_physical_sources_select_geometry_after_index_rebuild(self):
        original = self.fixture["parts"][1]
        alternate = make_part(self.root / "alternate", PART_NAMES[1], [(-.2, 0), (.2, 0), (0, .3)], [0, 1, 2])
        jackets = [(-1, original), (7, alternate)]
        trousers_key, jacket_key = (f"{ped_catalogues.murmur64(name):016x}" for name in PART_NAMES)
        archive = self.fixture["archive"]
        out = self.root / "selected.glb"
        first_exports = {}
        for order in [jackets, jackets[::-1]]:
            # Rebuild the jacket rows in a different order, as two manifest imports would do.
            index = self.root / f"rebuilt-{order[0][0]}.sqlite"
            shutil.copyfile(self.fixture["index"], index)
            with closing(sqlite3.connect(index)) as db:
                db.execute("DELETE FROM model_exports WHERE model_id=2")
                db.execute("DELETE FROM materials WHERE model_id=2")
                db.execute("DELETE FROM models WHERE id=2")
                for child, folder in order:
                    model_id = db.execute(
                        "INSERT INTO models(archive,offset,drawable,name,murmur64) VALUES (?,?,?,?,?)",
                        (archive, 256, child, PART_NAMES[1] + ".#dr", jacket_key.upper()),
                    ).lastrowid
                    db.execute("INSERT INTO model_exports VALUES (1,?,?,NULL,?,'ready')",
                               (model_id, str(folder / "model.obj"), str(folder / "model.glb")))
                    db.execute("INSERT INTO materials VALUES (?,0)", (model_id,))
                db.commit()
            for child, folder in jackets:
                with self.subTest(first_child=order[0][0], selected=child):
                    sources = [(trousers_key, archive, 128, -1), ("0x" + jacket_key, archive, "0x100", child)]
                    result = self.export(101, out, index, sources=sources)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    gltf, binary = skinned_glb.read_glb(str(out))
                    prim = gltf["meshes"][0]["primitives"][1]
                    source, raw = skinned_glb.read_glb(str(folder / "model.glb"))
                    expected = source["meshes"][0]["primitives"][0]
                    self.assertEqual(skinned_glb.accessor_bytes(gltf, binary, prim["attributes"]["POSITION"])[0],
                                     skinned_glb.accessor_bytes(source, raw, expected["attributes"]["POSITION"])[0])
                    self.assertEqual(skinned_glb.accessor_bytes(gltf, binary, prim["indices"])[0],
                                     skinned_glb.accessor_bytes(source, raw, expected["indices"])[0])
                    payload = out.read_bytes()
                    self.assertEqual(payload, first_exports.setdefault(child, payload))
        self.assertNotEqual(first_exports[-1], first_exports[7])

    def test_invalid_physical_sources_preserve_previous_export(self):
        archive = self.fixture["archive"]
        trousers, jacket = (f"{ped_catalogues.murmur64(name):016X}" for name in PART_NAMES)
        selected = (jacket, archive, 256, -1)
        cases = [
            ("no match", [(trousers, archive, 256, -1)]),  # Right physical tuple, wrong hash.
            ("no match", [(jacket, archive.lower(), 256, -1)]),
            ("no match", [(jacket, archive, 128, -1)]),
            ("no match", [(jacket, archive, 256, 0)]),
            ("duplicate", [selected, selected]),
            ("not selected", [selected, ("0000000000000042", archive, 128, -1)]),
            ("OFFSET", [(jacket, archive, -1, -1)]),
            ("OFFSET", [(jacket, archive, 2**63, -1)]),
            ("DRAWABLE", [(jacket, archive, 256, -2)]),
            ("DRAWABLE", [(jacket, archive, 256, 2**63)]),
            ("16 hexadecimal", [("00000042", archive, 256, -1)]),
            ("16 hexadecimal", [("10000000000000000", archive, 256, -1)]),
        ]
        out = self.root / "existing.glb"
        out.write_bytes(b"previous export")
        for reason, sources in cases:
            with self.subTest(sources=sources):
                result = self.export(101, out, sources=sources)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(out.read_bytes(), b"previous export")

    def test_selected_model_still_requires_one_ready_owned_glb(self):
        jacket = f"{ped_catalogues.murmur64(PART_NAMES[1]):016X}"
        selected = [(jacket, self.fixture["archive"], 256, -1)]
        out = self.root / "existing.glb"
        out.write_bytes(b"previous export")
        # The other namesake has a ready export. It cannot substitute for the selected source.
        for reason, status in [("ready GLB", "missing"), ("shared", "ready")]:
            with self.subTest(reason=reason):
                index = self.root / f"{status}.sqlite"
                shutil.copyfile(self.fixture["index"], index)
                with closing(sqlite3.connect(index)) as db:
                    db.execute("INSERT INTO models SELECT 3,archive,offset,7,name,murmur64 FROM models WHERE id=2")
                    db.execute("INSERT INTO model_exports SELECT 9,3,file,obj_path,glb_path,glb_status "
                               "FROM model_exports WHERE model_id=2")
                    db.execute("UPDATE model_exports SET glb_status=? WHERE model_id=2", (status,))
                    db.commit()
                result = self.export(101, out, index, sources=selected)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(jacket.lower(), result.stderr.lower())
                self.assertIn(reason.lower(), result.stderr.lower())
                self.assertEqual(out.read_bytes(), b"previous export")

    def test_output_cannot_replace_an_unselected_namesake_glb(self):
        alternate = make_part(self.root / "alternate", PART_NAMES[1], [(-.2, 0), (.2, 0), (0, .3)], [0, 1, 2])
        alternate_glb = alternate / "model.glb"
        originals = {p: p.read_bytes() for p in (self.fixture["parts"][1] / "model.glb", alternate_glb)}
        with closing(sqlite3.connect(self.fixture["index"])) as db:
            db.execute("INSERT INTO models SELECT 3,archive,offset,7,name,murmur64 FROM models WHERE id=2")
            db.execute("INSERT INTO model_exports VALUES (1,3,?,NULL,?,'ready')",
                       (str(alternate / "model.obj"), str(alternate_glb)))
            db.commit()
        key = f"{ped_catalogues.murmur64(PART_NAMES[1]):016X}"
        for path in (alternate_glb, self.root / "alias.glb"):
            with self.subTest(output=path.name):
                if path != alternate_glb:
                    try:
                        path.symlink_to(alternate_glb)
                    except (NotImplementedError, OSError) as exc:
                        self.skipTest(f"cannot create symbolic links: {exc}")
                result = self.export(101, path, sources=[(key, self.fixture["archive"], 256, -1)])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("replace", result.stderr.lower())
                for source, before in originals.items():
                    self.assertEqual(source.read_bytes(), before)
                if path != alternate_glb:
                    self.assertTrue(path.is_symlink())

    def test_export_preserves_source_glb_when_it_is_the_output(self):
        source = self.fixture["parts"][1] / "model.glb"
        before = source.read_bytes()
        result = self.export(101, source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("replace", result.stderr.lower())
        self.assertEqual(source.read_bytes(), before)

    def test_output_cannot_replace_an_indexed_model_outside_the_kit(self):
        trousers = catalogue_fixture._pack_state(state_id=0, drawable_hash=ped_catalogues.murmur64(PART_NAMES[0]))
        blocks = catalogue_fixture.build_kit_blocks(kits=[(101, [(10, 0)])], presets=[(10, 0, [trousers])])
        self.fixture["catalogue"].write_bytes(catalogue_fixture.build_synthetic_0drp(blocks))
        source = self.fixture["parts"][1] / "model.glb"  # Indexed jacket outside this trousers-only kit.
        original = source.read_bytes()
        for kind in ("direct", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = source
                if kind != "direct":
                    output = self.root / f"{kind}.glb"
                    try:
                        if kind == "symlink":
                            output.symlink_to(source)
                        else:
                            output.hardlink_to(source)
                    except (NotImplementedError, OSError) as exc:
                        self.skipTest(f"cannot create {kind}: {exc}")
                result = self.export(101, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("indexed GLB", result.stderr)
                self.assertEqual(source.read_bytes(), original)
                self.assertTrue(output.samefile(source))

    def test_textured_primitive_requires_source_uv(self):
        source = self.fixture["parts"][1] / "model.glb"
        gltf, binary = skinned_glb.read_glb(str(source))
        del gltf["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
        source.write_bytes(anim_glb.pack_glb(gltf, binary))
        out = self.root / "previous.glb"
        out.write_bytes(b"previous complete export")
        result = self.export(101, out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TEXCOORD_0", result.stderr)
        self.assertEqual(out.read_bytes(), b"previous complete export")

    def test_texture_override_preserves_the_source_alpha_cutoff(self):
        jacket = self.fixture["parts"][1] / "model.glb"
        gltf, binary = skinned_glb.read_glb(str(jacket))
        source = gltf["meshes"][0]["primitives"][0]["material"]
        gltf["materials"][source].update(alphaMode="MASK", alphaCutoff=0.37)
        jacket.write_bytes(anim_glb.pack_glb(gltf, binary))
        out = self.root / "masked.glb"
        result = self.export(101, out)
        self.assertEqual(result.returncode, 0, result.stderr)
        gltf, _ = skinned_glb.read_glb(str(out))
        trousers, jacket = [gltf["materials"][p["material"]] for p in gltf["meshes"][0]["primitives"]]
        self.assertEqual(trousers.get("alphaMode", "OPAQUE"), "OPAQUE")
        self.assertNotIn("alphaCutoff", trousers)
        self.assertEqual(jacket.get("alphaMode"), "MASK")
        self.assertEqual(jacket.get("alphaCutoff"), 0.37)


if __name__ == "__main__":
    unittest.main()
