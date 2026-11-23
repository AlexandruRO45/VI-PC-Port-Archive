"""Exercise repeated embedded DDS exports through the real world_build CLI with synthetic assets."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_world_build as twb
from test_world_place import entity, write_entities


CLI = Path(__file__).resolve().parents[1] / "tools" / "world_build.py"
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)
MODES = ("gpu", "nodes", "bake")


def write_manifest(path, data):
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")


def write_export(root, stem, width, *, color=None, external=None, offset=512, **overrides):
    models = root / "models"
    models.mkdir(parents=True)
    (models / f"{stem}.obj").write_text(
        f"v 0 0 0\nv {width} 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
        encoding="utf-8", newline="\n",
    )
    embedded = [{"Index": 0, "Name": "synthetic", "Width": 4, "Height": 4, "Hash": "12345678"}]
    ref = twb._param_ref(twb.ALBEDO_PARAM, 0, twb.TTPF_ALBEDO_KEY)
    item = dict(File=f"{stem}.obj", Name="synthetic.#dr", Offset=offset, Drawable=None, MeshShaders=[0],
                Materials=[{"ShaderIndex": 0, "Shader": "23456789", "Textures": [ref]}], Embedded=embedded)
    item.update(overrides)
    write_manifest(models / "manifest.json", dict(src="/synthetic/archive.rpf", Lod=0, Items=[item]))
    if color is not None:
        directory = models / f"{stem}_tex"
        directory.mkdir()
        (directory / "000_synthetic.dds").write_bytes(twb._uniform_bgra_dds(color))
    if external is not None:
        directory = root / "ttpf"
        directory.mkdir()
        (directory / "external.dds").write_bytes(twb._uniform_bgra_dds(external))
        write_manifest(directory / "manifest.json", {
            "Textures": [{"AssetKey": twb.TTPF_ALBEDO_KEY, "File": "external.dds"}],
        })


class RepeatEmbeddedDDSTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.placements = self.base / "placements.jsonl"
        write_entities(self.placements, [entity("synthetic", (x, 0, 0)) for x in (10, 500)])

    def run_cli(self, mode, root, *args):
        out = self.base / "world"
        result = subprocess.run(
            [sys.executable, "-I", str(CLI), str(root), "--out", str(out), "--placements", str(self.placements),
             "--instancing", mode, *map(str, args)],
            capture_output=True, text=True, timeout=30,
        )
        return result, out

    def read_world(self, mode, root, *args, skipped=0):
        result, out = self.run_cli(mode, root, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads((out / "world.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["stats"]["textures_skipped"], skipped)
        stats = doc["stats"]["placements"]
        self.assertEqual((stats["placed"], stats["distinct_models"], stats["ambiguous_model"]), (2, 1, 0))
        chunks = sorted(doc["chunks"], key=lambda chunk: chunk["bounds"][0])
        self.assertEqual([chunk["bounds"][3] - chunk["bounds"][0] for chunk in chunks], [1, 1])
        paths = [twb._chunk_disk_path(out, chunk["url"]) for chunk in chunks]
        pixels = [twb._glb_first_base_color_pixel(path)[1] for path in paths]
        return pixels, [Path(path).read_bytes() for path in paths]

    def test_repeat_restores_pixels_without_replacing_geometry_or_resolved_bindings(self):
        owner, repeat = self.base / "a", self.base / "b"
        write_export(owner, "owner", 1)
        write_export(repeat, "repeat", 3, color=RED)
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(self.read_world(mode, owner)[0], [None, None])
                self.assertEqual(self.read_world(mode, owner, "--extra-assets", repeat)[0], [RED, RED])
                self.assertEqual(self.read_world(mode, repeat, "--extra-assets", owner,
                                                 "--base-color-param", twb.ALBEDO_PARAM)[0], [RED, RED])
                self.assertEqual(self.read_world(mode, owner, "--extra-assets", repeat,
                                                 "--base-color-param", twb.NORMAL_PARAM)[0], [None, None])
                self.assertEqual(self.read_world(mode, owner, "--extra-assets", repeat,
                                                 "--no-textures")[0], [None, None])

        directory = owner / "models" / "owner_tex"
        directory.mkdir()
        dds = directory / "000_synthetic.dds"
        dds.write_bytes(twb._uniform_bgra_dds(BLUE))
        for mode in MODES:
            with self.subTest(mode=mode, owner="complete"):
                complete = self.read_world(mode, owner)
                self.assertEqual(complete[0], [BLUE, BLUE])
                self.assertEqual(self.read_world(mode, owner, "--extra-assets", repeat), complete)

        dds.write_bytes(b"unreadable DDS")
        for mode in MODES:
            with self.subTest(mode=mode, owner="unreadable"):
                self.assertEqual(self.read_world(mode, owner, "--extra-assets", repeat, skipped=2)[0], [None, None])

    def test_external_bindings_keep_precedence(self):
        for local in (True, False):
            owner, repeat = self.base / f"a{local}", self.base / f"b{local}"
            write_export(owner, "owner", 1, external=GREEN if local else None)
            write_export(repeat, "repeat", 3, color=RED, external=BLUE)
            for mode in MODES:
                with self.subTest(mode=mode, owner_has_external=local):
                    expected = GREEN if local else BLUE
                    self.assertEqual(self.read_world(mode, repeat, "--extra-assets", owner)[0], [expected, expected])

    def test_repeat_requires_matching_source_mapping_material_and_embedded_record(self):
        cases = {
            "mapping": ({}, {"MeshShaders": [1, 0]}),
            "material": ({}, {"Materials": [{"ShaderIndex": 0, "Shader": "34567890",
                                            "Textures": [twb._param_ref(twb.ALBEDO_PARAM, 0, twb.TTPF_ALBEDO_KEY)]}]}),
            "embedded": ({}, {"Embedded": [{"Index": 0, "Name": "different"}]}),
            "no mapping": ({"MeshShaders": None}, {"MeshShaders": None}),
            "no material": ({"Materials": None}, {"Materials": None}),
            "no embedded metadata": ({"Embedded": None}, {"Embedded": None}),
            "distinct source": ({}, {"offset": 640}),
        }
        for index, (case, (owner_fields, repeat_fields)) in enumerate(cases.items()):
            owner, repeat = self.base / f"a{index}", self.base / f"b{index}"
            write_export(owner, "owner", 1, **owner_fields)
            write_export(repeat, "repeat", 3, color=RED, **repeat_fields)
            for mode in MODES:
                with self.subTest(mode=mode, case=case):
                    if case == "distinct source":
                        result, _ = self.run_cli(mode, owner, "--extra-assets", repeat)
                        self.assertEqual(result.returncode, 1)
                        self.assertIn("'ambiguous_model': 2", result.stderr)
                    else:
                        self.assertEqual(self.read_world(mode, owner, "--extra-assets", repeat)[0], [None, None])


if __name__ == "__main__":
    unittest.main()
