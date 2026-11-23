"""GLB topology import checked against explicit triangle lists and the real CLI.

Synthetic writer fixtures only. Primitive assembly follows:
https://github.com/KhronosGroup/glTF/blob/5b00b913b74c8f0260c64578b98a09ef2dde0b41/specification/2.0/Specification.adoc#meshes-topology
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ped_assemble as pa

# Literal lists keep the expected winding independent of the implementation.
CASES = (
    (5, [0, 1, 2], [(0, 1, 2)]),
    (5, [0, 1, 2, 3], [(0, 1, 2), (1, 3, 2)]),
    (5, [0, 1, 2, 3, 4, 5],
     [(0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 5, 4)]),
    (6, [0, 1, 2], [(1, 2, 0)]),
    (6, [0, 1, 2, 3], [(1, 2, 0), (2, 3, 0)]),
    (6, [0, 1, 2, 3, 4, 5],
     [(1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0)]),
)
INDEXED_CASES = (
    (5, [4, 1, 3, 0, 5, 2],
     [(4, 1, 3), (1, 0, 3), (3, 0, 5), (0, 2, 5)]),
    (6, [4, 1, 3, 0, 5, 2],
     [(1, 3, 4), (3, 0, 4), (0, 5, 4), (5, 2, 4)]),
    (5, [0, 1, 2, 2, 3, 4],
     [(0, 1, 2), (1, 2, 2), (2, 2, 3), (2, 4, 3)]),
    (6, [0, 1, 2, 2, 3, 4],
     [(1, 2, 0), (2, 2, 0), (2, 3, 0), (3, 4, 0)]),
)


def split_glb(data):
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    offset = 20 + length
    length, kind = struct.unpack_from("<II", data, offset)
    assert kind == pa.CHUNK_TYPE_BIN
    return doc, bytearray(data[offset + 8:offset + 8 + length])


def container(doc, blob):
    data = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    data += b" " * (-len(data) % 4)
    return b"".join((
        struct.pack("<4sII", pa.GLB_MAGIC, 2, 28 + len(data) + len(blob)),
        struct.pack("<II", len(data), pa.CHUNK_TYPE_JSON), data,
        struct.pack("<II", len(blob), pa.CHUNK_TYPE_BIN), bytes(blob),
    ))


def part(count, triangles, *, name="surface", skinned=False):
    p = pa.MeshPart(
        name, [(float(i // 2), float(i % 2), 0.0) for i in range(count)],
        triangles, coord_space="y-up",
        normals=[(0.0, 0.0, -1.0)] * count,
        uvs=[(i / 8.0, 0.5) for i in range(count)],
        colors=[(i / 8.0, 0.25, 0.5, 1.0) for i in range(count)],
        material=pa.MaterialSpec(name="surface_material", metallic_factor=0.25),
    )
    if skinned:
        p.joints_0 = [(0, 0, 0, 0)] * count
        p.weights_0 = [(1.0, 0.0, 0.0, 0.0)] * count
        p.joint_names = ["root"]
    return p


def set_topology(doc, blob, primitive, mode, stream, indexed):
    if mode == "default":
        primitive.pop("mode", None)
    else:
        primitive["mode"] = mode
    if not indexed:
        primitive.pop("indices")
        return
    accessor = doc["accessors"][primitive["indices"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    char, size = pa.COMPONENT_STRUCT[accessor["componentType"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    assert len(stream) * size <= view["byteLength"]
    struct.pack_into(f"<{len(stream)}{char}", blob, offset, *stream)
    accessor["count"] = len(stream)
    view["byteLength"] = len(stream) * size


def fixture(mode, stream, triangles, *, indexed=True, skinned=False):
    p = part(max(stream) + 1, triangles, skinned=skinned)
    skeleton = pa.SkeletonSpec("rig", [pa.SkeletonJoint("root")], coord_space="y-up") if skinned else None
    control, _ = pa.build_assembled_glb([p], skeleton=skeleton, apply_standing=False)
    doc, blob = split_glb(control)
    set_topology(doc, blob, doc["meshes"][0]["primitives"][0], mode, stream, indexed)
    return container(doc, blob), control


class GlbTopologyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def read(self, data, name="source"):
        path = self.root / (name + ".glb")
        path.write_bytes(data)
        result = pa.read_glb_parts(str(path))
        self.assertEqual(path.read_bytes(), data)
        return result

    def run_cli(self, data, name):
        source, output = self.root / (name + ".glb"), self.root / (name + "-out.glb")
        source.write_bytes(data)
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ped_assemble.py"),
             "assemble", str(source), "-o", str(output), "--no-cull", "--no-standing-offset"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(source.read_bytes(), data)
        return output.read_bytes()

    def test_indexed_modes_preserve_winding_and_degenerate_parity(self):
        for mode, stream, triangles in CASES + INDEXED_CASES:
            with self.subTest(mode=mode, stream=stream):
                data, control = fixture(mode, stream, triangles)
                actual = self.read(data)
                self.assertEqual(actual[0].triangles, triangles)
                self.assertEqual(actual, self.read(control, "control"))

    def test_nonindexed_minimum_and_longer_primitives(self):
        for mode, stream, triangles in CASES:
            with self.subTest(mode=mode, vertices=len(stream)):
                data, control = fixture(mode, stream, triangles, indexed=False)
                actual = self.read(data)
                self.assertEqual(actual[0].triangles, triangles)
                self.assertEqual(actual, self.read(control, "control"))

    def test_cli_matches_independent_triangle_lists(self):
        for mode, stream, triangles in CASES:
            if len(stream) == 3:
                continue
            for indexed in (True, False):
                with self.subTest(mode=mode, vertices=len(stream), indexed=indexed):
                    data, control = fixture(mode, stream, triangles, indexed=indexed)
                    self.assertEqual(self.run_cli(data, "source"), self.run_cli(control, "control"))

    def test_mixed_primitives_keep_order(self):
        selected = (CASES[2], (4, [0, 1, 2, 3, 4, 5], [(0, 1, 2), (3, 4, 5)]), CASES[5])
        parts = [part(6, triangles, name=f"part{i}") for i, (_, _, triangles) in enumerate(selected)]
        raw, _ = pa.build_assembled_glb(parts, apply_standing=False)
        doc, blob = split_glb(raw)
        primitives = [mesh["primitives"][0] for mesh in doc["meshes"]]
        doc["meshes"] = [{"name": "ordered", "primitives": primitives}]
        doc["nodes"] = [{"name": "ordered", "mesh": 0}]
        doc["scenes"] = [{"nodes": [0]}]
        control = container(doc, blob)
        for primitive, (mode, stream, _) in zip(primitives, selected):
            set_topology(doc, blob, primitive, mode, stream, True)
        actual = self.read(container(doc, blob))
        self.assertEqual([p.name for p in actual], ["ordered_p0", "ordered_p1", "ordered_p2"])
        self.assertEqual(actual, self.read(control, "control"))

    def test_skin_binding_and_vertex_attributes_are_preserved(self):
        for mode, stream, triangles in (CASES[2], CASES[5]):
            with self.subTest(mode=mode):
                data, control = fixture(mode, stream, triangles, skinned=True)
                actual = self.read(data)
                self.assertEqual(actual, self.read(control, "control"))
                self.assertEqual(actual[0].joint_names, ["root"])
                self.assertEqual(actual[0].joints_0, [(0, 0, 0, 0)] * 6)
                self.assertEqual(actual[0].weights_0, [(1.0, 0.0, 0.0, 0.0)] * 6)

    def test_default_and_whole_number_modes(self):
        triangles = [(0, 1, 2), (3, 4, 5)]
        for mode in ("default", 4, 4.0):
            with self.subTest(mode=mode):
                data, control = fixture(mode, list(range(6)), triangles)
                self.assertEqual(self.read(data), self.read(control, "control"))
        for mode, stream, triangles in (CASES[2], CASES[5]):
            with self.subTest(mode=float(mode)):
                data, control = fixture(float(mode), stream, triangles)
                self.assertEqual(self.read(data), self.read(control, "control"))

    def test_non_surface_modes_do_not_publish_false_triangles(self):
        for mode in (0, 1, 2, 3, -1, 7, True, False, None, "5", 5.5):
            with self.subTest(mode=mode):
                data, _ = fixture(mode, list(range(6)), [(0, 1, 2), (3, 4, 5)])
                source, output = self.root / "source.glb", self.root / "out.glb"
                source.write_bytes(data)
                output.write_bytes(b"existing output")
                with self.assertRaisesRegex(pa.PedAssembleError, "primitive mode"):
                    pa.read_glb_parts(str(source))
                result = subprocess.run(
                    [sys.executable, "-I", str(ROOT / "tools" / "ped_assemble.py"),
                     "assemble", str(source), "-o", str(output), "--no-cull"],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("primitive mode", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(output.read_bytes(), b"existing output")
                self.assertEqual(source.read_bytes(), data)

    def test_incomplete_strip_and_fan_do_not_silently_disappear(self):
        for mode in (5, 6):
            with self.subTest(mode=mode):
                data, _ = fixture(mode, [0, 1, 2], [(0, 1, 2)])
                doc, blob = split_glb(data)
                primitive = doc["meshes"][0]["primitives"][0]
                doc["accessors"][primitive["indices"]]["count"] = 2
                with self.assertRaisesRegex(pa.PedAssembleError, "at least three vertices"):
                    self.read(container(doc, blob))


if __name__ == "__main__":
    unittest.main()
