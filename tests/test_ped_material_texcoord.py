"""ped_assemble: material textures keep the UV set they were bound to. No game data."""
import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ped_assemble as pa  # noqa: E402

PNG_BASE = b"\x89PNG\r\n\x1a\nbase"
PNG_MR = b"\x89PNG\r\n\x1a\nmetal"
PNG_NORMAL = b"\x89PNG\r\n\x1a\nnormal"


def glb_doc(glb_bytes):
    length = struct.unpack_from("<I", glb_bytes, 12)[0]
    return json.loads(glb_bytes[20:20 + length])


def material_part(**overrides):
    fields = dict(name="skin", base_color_png=PNG_BASE, base_color_texcoord=1,
                  metallic_roughness_png=PNG_MR, metallic_roughness_texcoord=2,
                  normal_png=PNG_NORMAL, normal_texcoord=3)
    fields.update(overrides)
    return pa.MeshPart(
        name="skin", positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], triangles=[(0, 1, 2)],
        coord_space="y-up", uvs=[(0, 0), (1, 0), (0, 1)],
        uvs_extra={1: [(0.25, 0), (0.75, 0), (0.25, 1)]},
        material=pa.MaterialSpec(**fields),
    )


class MaterialTexCoord(unittest.TestCase):
    def test_glb_material_textures_keep_their_uv_set(self):
        glb_bytes, _ = pa.build_assembled_glb([material_part()], apply_standing=False)
        doc = glb_doc(glb_bytes)
        material = doc["materials"][0]
        pbr = material["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorTexture"], {"index": 0, "texCoord": 1})
        self.assertEqual(pbr["metallicRoughnessTexture"], {"index": 1, "texCoord": 2})
        self.assertEqual(material["normalTexture"], {"index": 2, "texCoord": 3})
        attrs = doc["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("TEXCOORD_0", attrs)
        self.assertIn("TEXCOORD_1", attrs)

    def test_default_uv_set_stays_implicit(self):
        part = material_part(base_color_texcoord=0, metallic_roughness_texcoord=0, normal_texcoord=0)
        glb_bytes, _ = pa.build_assembled_glb([part], apply_standing=False)
        doc = glb_doc(glb_bytes)
        material = doc["materials"][0]
        self.assertEqual(material["pbrMetallicRoughness"]["baseColorTexture"], {"index": 0})
        self.assertEqual(material["pbrMetallicRoughness"]["metallicRoughnessTexture"], {"index": 1})
        self.assertEqual(material["normalTexture"], {"index": 2})

    def test_json_report_round_trip_preserves_bindings(self):
        serialized = pa.part_to_dict(material_part())["material"]
        self.assertEqual(serialized["base_color_texcoord"], 1)
        self.assertEqual(serialized["normal_texcoord"], 3)
        restored = pa._parse_material_dict(serialized, "")
        self.assertEqual(restored.base_color_texcoord, 1)
        self.assertEqual(restored.metallic_roughness_texcoord, 2)
        self.assertEqual(restored.normal_texcoord, 3)
        self.assertTrue(restored.base_color_png.startswith(b"\x89PNG"))

    def test_camel_case_alias_and_invalid_values(self):
        spec = pa._parse_material_dict({"baseColorTexCoord": 4, "normalTexCoord": 5}, "")
        self.assertEqual((spec.base_color_texcoord, spec.normal_texcoord), (4, 5))
        for value in (-1, "1", True):
            with self.subTest(value=value), self.assertRaises(pa.PedAssembleError):
                pa._parse_material_dict({"base_color_texcoord": value}, "")


if __name__ == "__main__":
    unittest.main()
