"""Static GLB transforms: literal coordinates/normals, mirrors, parents, and skin isolation.

For Rz(90) * scale(2,3,4), the oblique normal (1,1,1) becomes (-4,6,3)/sqrt(61).
For scale(2,3,4) * Rz(45), it becomes (0,4*sqrt(2),3)/sqrt(41).
These independent vectors also check parent-induced shear and inverse-transpose ordering.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glb_writer
import ped_assemble as pa


POSITIONS = [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 1.0, -1.0)]
NORMAL = (1 / math.sqrt(3),) * 3
UVS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
ROTATION = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
MATRIX = [0, 2, 0, 0, -3, 0, 0, 0, 0, 0, 4, 0, 5, 7, 11, 1]


def split_glb(data):
    doc = glb_writer.read_glb_summary(data)["json"]
    header = 20 + struct.unpack_from("<I", data, 12)[0]
    size = struct.unpack_from("<I", data, header)[0]
    return doc, data[header + 8:header + 8 + size]


def container(doc, payload):
    text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    payload += b"\0" * (-len(payload) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(text) + len(payload))
            + struct.pack("<I4s", len(text), b"JSON") + text
            + struct.pack("<I4s", len(payload), b"BIN\0") + payload)


class PedGlbNodeTransformTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.glb"
        self.output = self.root / "assembled.glb"
        self.primitive = glb_writer.pack_primitive({
            "positions": POSITIONS, "normals": [NORMAL] * 3, "indices": [0, 1, 2], "uvs": UVS,
        })
        self.write_instances([((5, 7, 11), ROTATION, (2, 3, 4))])

    def write_instances(self, instances):
        glb_writer.write_instanced_glb(str(self.source), [
            {"primitives": [self.primitive], "instances": instances}], mode="nodes")

    def assemble(self):
        original = self.source.read_bytes()
        result = subprocess.run([
            sys.executable, "-I", pa.__file__, "assemble", str(self.source),
            "-o", str(self.output), "--no-cull", "--no-standing-offset",
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.source.read_bytes(), original)
        return pa.read_glb_parts(str(self.output))

    def assert_vectors(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=6)

    def assert_geometry(self, part, positions, normal, triangles=((0, 1, 2),)):
        self.assert_vectors(part.positions, positions)
        self.assert_vectors(part.normals, [normal] * 3)
        for n in part.normals:
            self.assertAlmostEqual(math.hypot(*n), 1.0, places=6)
        self.assertEqual(part.triangles, list(triangles))
        self.assertEqual(part.uvs, UVS)

    def test_actual_writer_cli_applies_trs_and_inverse_transpose_normals(self):
        part, = self.assemble()
        self.assert_geometry(part, [(5, 7, 11), (5, 9, 7), (2, 7, 7)],
                             tuple(v / math.sqrt(61) for v in (-4, 6, 3)))

    def test_column_major_matrix_and_trs_have_equivalent_output(self):
        trs, = self.assemble()
        doc, payload = split_glb(self.source.read_bytes())
        node = doc["nodes"][0]
        for key in ("translation", "rotation", "scale"):
            node.pop(key, None)
        node["matrix"] = MATRIX
        self.source.write_bytes(container(doc, payload))
        matrix, = self.assemble()
        self.assert_geometry(matrix, [(5, 7, 11), (5, 9, 7), (2, 7, 7)],
                             tuple(v / math.sqrt(61) for v in (-4, 6, 3)))
        self.assert_vectors(matrix.positions, trs.positions)
        self.assert_vectors(matrix.normals, trs.normals)

    def test_parent_scale_and_child_rotation_preserve_shear(self):
        doc, payload = split_glb(self.source.read_bytes())
        child = doc["nodes"][0]
        child.update(translation=[1, 2, 3], scale=[1, 1, 1],
                     rotation=[0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)])
        doc["nodes"].append({"children": [0], "translation": [10, 20, 30], "scale": [2, 3, 4]})
        doc["scenes"] = [{"nodes": [1]}]
        self.source.write_bytes(container(doc, payload))
        part, = self.assemble()
        root_two = math.sqrt(2)
        self.assert_geometry(part, [(12, 26, 42), (12 + root_two, 26 + 3 / root_two, 38),
                                    (12 - root_two, 26 + 3 / root_two, 38)],
                             (0, 4 * root_two / math.sqrt(41), 3 / math.sqrt(41)))

    def test_negative_determinant_preserves_normals_and_front_faces(self):
        original, payload = split_glb(self.source.read_bytes())
        for has_normals in (True, False):
            with self.subTest(has_normals=has_normals):
                doc = copy.deepcopy(original)
                doc["nodes"][0]["scale"] = [-2, 3, 4]
                if not has_normals:
                    doc["meshes"][0]["primitives"][0]["attributes"].pop("NORMAL")
                self.source.write_bytes(container(doc, payload))
                part, = self.assemble()
                self.assert_geometry(part, [(5, 7, 11), (5, 5, 7), (2, 7, 7)],
                                     tuple(v / math.sqrt(61) for v in (-4, -6, 3)), triangles=((0, 2, 1),))
                a, b, c = (part.positions[i] for i in part.triangles[0])
                u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
                cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0])
                self.assertGreater(sum(x * n for x, n in zip(cross, part.normals[0])), 0)

    def test_nonzero_uniform_scales_keep_unit_normals(self):
        for magnitude in (1e-12, 1e12):
            with self.subTest(magnitude=magnitude):
                self.write_instances([((0, 0, 0), ROTATION, (magnitude,) * 3)])
                part, = self.assemble()
                self.assert_vectors([tuple(v / magnitude for v in p) for p in part.positions],
                                    [(0, 0, 0), (0, 1, -1), (-1, 0, -1)])
                self.assert_vectors(part.normals, [(-NORMAL[0], NORMAL[1], NORMAL[2])] * 3)
                self.assertEqual(part.triangles, [(0, 1, 2)])

    def test_flat_order_translation_and_bare_mesh_contracts(self):
        self.write_instances([((1, 2, 3), (0, 0, 0, 1), (1, 1, 1)),
                              ((-4, 5, 6), (0, 0, 0, 1), (1, 1, 1))])
        doc, payload = split_glb(self.source.read_bytes())
        doc["nodes"][0]["name"] = "first"
        doc["nodes"][1]["name"] = "second"
        doc["nodes"].append({"children": [1], "translation": [10, 0, 0]})
        doc["scenes"] = [{"nodes": [2, 0]}]
        self.source.write_bytes(container(doc, payload))
        parts = pa.read_glb_parts(str(self.source))
        self.assertEqual([part.name for part in parts], ["first", "second"])
        self.assertEqual([part.translation for part in parts], [(1, 2, 3), (6, 5, 6)])
        for part in parts:
            self.assertEqual(part.positions, POSITIONS)
            self.assertEqual(part.triangles, [(0, 1, 2)])
        self.assertEqual(parts[0].normals, parts[1].normals)
        # Preserve the old reader's float-convertible translation tolerance.
        doc["nodes"][0]["translation"] = ["1", "2", "3"]
        self.source.write_bytes(container(doc, payload))
        self.assertEqual(pa.read_glb_parts(str(self.source))[0].translation, (1, 2, 3))
        for key in ("nodes", "scene", "scenes"):
            doc.pop(key, None)
        self.source.write_bytes(container(doc, payload))
        bare, = pa.read_glb_parts(str(self.source))
        self.assertEqual(bare.translation, (0, 0, 0))
        self.assertEqual(bare.positions, POSITIONS)
        self.assertEqual(bare.normals, parts[0].normals)

    def test_static_transform_path_never_bakes_skin_payloads(self):
        part = pa.MeshPart(
            name="skin_part", positions=POSITIONS, triangles=[(0, 1, 2)], coord_space="y-up",
            normals=[NORMAL] * 3, uvs=UVS, joints_0=[(0, 0, 0, 0)] * 3,
            weights_0=[(1.0, 0.0, 0.0, 0.0)] * 3, joint_names=["root"],
        )
        skeleton = pa.SkeletonSpec(name="rig", coord_space="y-up",
                                   joints=[pa.SkeletonJoint(name="root", parent=-1)])
        pa.write_assembled_glb(str(self.source), [part], skeleton=skeleton, apply_standing=False)
        baseline, = pa.read_glb_parts(str(self.source))
        original, payload = split_glb(self.source.read_bytes())
        for has_skin_property in (True, False):  # Keep the existing single-skin fallback too.
            with self.subTest(has_skin_property=has_skin_property):
                doc = copy.deepcopy(original)
                mesh = doc["nodes"][0]
                mesh.update(translation=[7, 8, 9], rotation=list(ROTATION), scale=[-2, 3, 4])
                if not has_skin_property:
                    mesh.pop("skin")
                parent = len(doc["nodes"])
                doc["nodes"].append({"children": [0], "translation": [4, 5, 6], "scale": [2, 3, 4]})
                doc["scenes"][0]["nodes"] = [parent] + [n for n in doc["scenes"][0]["nodes"] if n != 0]
                self.source.write_bytes(container(doc, payload))
                actual, = pa.read_glb_parts(str(self.source))
                for field in ("positions", "normals", "triangles", "joints_0", "weights_0", "joint_names"):
                    self.assertEqual(getattr(actual, field), getattr(baseline, field))
                # This fix preserves the existing importer-local translation convention for skin parts.
                self.assertEqual(actual.translation, (7, 8, 9))

    def test_unsupported_transform_keeps_existing_output(self):
        original, payload = split_glb(self.source.read_bytes())
        for case in ("singular", "perspective", "cycle"):
            with self.subTest(case=case):
                doc = copy.deepcopy(original)
                if case == "singular":
                    doc["nodes"][0]["scale"] = [0, 1, 1]
                elif case == "perspective":
                    doc["nodes"][0] = {"mesh": 0, "matrix": list(MATRIX)}
                    doc["nodes"][0]["matrix"][3] = 0.5
                else:
                    doc["nodes"][0]["children"] = [1]
                    doc["nodes"].append({"children": [0]})
                self.source.write_bytes(container(doc, payload))
                self.output.write_bytes(b"existing output")
                result = subprocess.run([
                    sys.executable, "-I", pa.__file__, "assemble", str(self.source),
                    "-o", str(self.output), "--no-cull", "--no-standing-offset",
                ], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("ERROR:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(self.output.read_bytes(), b"existing output")


if __name__ == "__main__":
    unittest.main()
