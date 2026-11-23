"""Embedded PNGs survive the real ped writer, GLB importer, and assembly CLI."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dds_png
import glb_writer
import ped_assemble as pa


SLOTS = (
    ("base_color_png", ("pbrMetallicRoughness", "baseColorTexture")),
    ("normal_png", ("normalTexture",)),
    ("metallic_roughness_png", ("pbrMetallicRoughness", "metallicRoughnessTexture")),
)


def texture_info(material, keys):
    value = material
    for key in keys:
        value = value[key]
    return value


def split_glb(data):
    doc = glb_writer.read_glb_summary(data)["json"]
    json_size = struct.unpack_from("<I", data, 12)[0]
    bin_header = 20 + json_size
    bin_size = struct.unpack_from("<I", data, bin_header)[0]
    return doc, data[bin_header + 8:bin_header + 8 + bin_size]


def container(doc, payload):
    text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    payload += b"\0" * (-len(payload) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(text) + len(payload))
            + struct.pack("<I4s", len(text), b"JSON") + text
            + struct.pack("<I4s", len(payload), b"BIN\0") + payload)


class PedGlbTextureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pngs = [dds_png.encode_png(1, 1, bytes(color)) for color in (
            (211, 37, 19, 255), (128, 128, 255, 255), (0, 191, 63, 255))]
        self.parts = []
        for index in range(2):
            self.parts.append(pa.MeshPart(
                name=f"part_{index}", coord_space="y-up",
                positions=[(index * 2.0, 0.0, 0.0), (index * 2.0 + 1, 0.0, 0.0),
                           (index * 2.0, 1.0, 0.0)],
                triangles=[(0, 1, 2)], normals=[(0.0, 0.0, 1.0)] * 3,
                uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                material=pa.MaterialSpec(
                    name=f"material_{index}", base_color_factor=(0.25, 0.5, 0.75, 0.875),
                    metallic_factor=0.625, roughness_factor=0.375, alpha_mode="MASK",
                    alpha_cutoff=0.375, double_sided=True,
                    base_color_png=self.pngs[0], normal_png=self.pngs[1],
                    metallic_roughness_png=self.pngs[2]),
            ))
        self.source = self.root / "source.glb"
        pa.write_assembled_glb(str(self.source), self.parts, apply_standing=False)

    def assert_materials(self, parts):
        self.assertEqual(len(parts), len(self.parts))
        for actual, expected in zip(parts, self.parts):
            self.assertEqual(actual.material, expected.material)
            self.assertEqual(actual.positions, expected.positions)
            self.assertEqual(actual.uvs, expected.uvs)

    def assert_embedded_output(self, path):
        doc, payload = split_glb(path.read_bytes())
        self.assertEqual(len(doc.get("images", [])), 3)
        self.assertEqual(len(doc.get("textures", [])), 3)  # Both materials share the same images.
        self.assertEqual(len(doc["materials"]), 2)
        for material in doc["materials"]:
            for (_, keys), expected in zip(SLOTS, self.pngs):
                texture = doc["textures"][texture_info(material, keys)["index"]]
                image = doc["images"][texture["source"]]
                self.assertEqual(image["mimeType"], "image/png")
                view = doc["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                self.assertEqual(payload[start:start + view["byteLength"]], expected)

    def test_writer_importer_roundtrip_preserves_all_slots_and_factors(self):
        self.assert_materials(pa.read_glb_parts(str(self.source)))

    def test_texture_and_image_indices_are_independent(self):
        doc, payload = split_glb(self.source.read_bytes())
        doc["textures"] = [doc["textures"][i] for i in (2, 0, 1)]
        for material in doc["materials"]:
            for _, keys in SLOTS:
                info = texture_info(material, keys)
                info["index"] = (info["index"] + 1) % 3
        doc["images"] = [doc["images"][i] for i in (1, 2, 0)]
        for texture in doc["textures"]:
            texture["source"] = (texture["source"] + 2) % 3
        # The final image ends exactly at the declared buffer boundary, before BIN padding.
        doc["buffers"][0]["byteLength"] = max(
            view.get("byteOffset", 0) + view["byteLength"] for view in doc["bufferViews"])
        self.source.write_bytes(container(doc, payload))
        self.assert_materials(pa.read_glb_parts(str(self.source)))

    def test_real_cli_preserves_png_bytes_and_shared_images(self):
        original = self.source.read_bytes()
        target = self.root / "assembled.glb"
        result = subprocess.run(
            [sys.executable, "-I", pa.__file__, "assemble", str(self.source),
             "-o", str(target), "--no-cull", "--no-standing-offset"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.source.read_bytes(), original)
        self.assert_embedded_output(target)
        self.assert_materials(pa.read_glb_parts(str(target)))

    def test_embedded_image_views_stay_inside_declared_and_actual_buffers(self):
        original, payload = split_glb(self.source.read_bytes())
        image = original["images"][-1]
        view_index = image["bufferView"]
        end = (original["bufferViews"][view_index]["byteOffset"]
               + original["bufferViews"][view_index]["byteLength"])
        for case in ("declared-end", "actual-end", "negative-offset", "negative-length", "view-index"):
            with self.subTest(case=case):
                doc = copy.deepcopy(original)
                view = doc["bufferViews"][view_index]
                if case == "declared-end":
                    doc["buffers"][0]["byteLength"] = end - 1
                elif case == "actual-end":
                    view["byteLength"] = len(payload) - view["byteOffset"] + 1
                    doc["buffers"][0]["byteLength"] = len(payload) + 1
                elif case == "negative-offset":
                    view["byteOffset"] = -1
                elif case == "negative-length":
                    view["byteLength"] = -1
                else:
                    doc["images"][-1]["bufferView"] = -1
                self.source.write_bytes(container(doc, payload))
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.read_glb_parts(str(self.source))
                self.assertIn(str(self.source), str(caught.exception))
                self.assertIn("normalTexture", str(caught.exception))

    def test_untextured_and_external_uri_inputs_keep_existing_behavior(self):
        doc, payload = split_glb(self.source.read_bytes())
        # The first material is untextured; only the second one's base color is external.
        plain = doc["materials"][0]
        plain["pbrMetallicRoughness"].pop("baseColorTexture")
        plain["pbrMetallicRoughness"].pop("metallicRoughnessTexture")
        plain.pop("normalTexture")
        image_index = doc["textures"][texture_info(doc["materials"][1], SLOTS[0][1])["index"]]["source"]
        doc["images"][image_index] = {"mimeType": "image/png", "uri": "not-opened.png"}
        self.source.write_bytes(container(doc, payload))
        parts = pa.read_glb_parts(str(self.source))
        self.assertTrue(all(getattr(parts[0].material, slot) is None for slot, _ in SLOTS))
        self.assertIsNone(parts[1].material.base_color_png)
        self.assertEqual(parts[1].material.normal_png, self.pngs[1])
        self.assertEqual(parts[1].material.metallic_roughness_png, self.pngs[2])


if __name__ == "__main__":
    unittest.main()
