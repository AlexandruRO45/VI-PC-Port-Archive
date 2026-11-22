"""Unit tests for tools/glb_writer.py: OBJ->GLB conversion, container structure, axis swap.

    python -I tests/test_glb_writer.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "glb_writer.py")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

OBJ_CUBE = """# Simple triangle pair
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""


def parse_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "bad magic"  # glTF
    assert version == 2, "bad version"
    assert length == len(data), "length mismatch"
    off = 12
    chunks = []
    while off < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, off)
        off += 8
        chunks.append((chunk_type, data[off:off + chunk_len]))
        off += chunk_len
    return magic, version, chunks


class GlbWriterTest(unittest.TestCase):
    def _convert(self, obj_text, up="y", tmp=None):
        if tmp is None:
            tmp = tempfile.mkdtemp()
        obj_path = os.path.join(tmp, "in.obj")
        glb_path = os.path.join(tmp, "out.glb")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(obj_text)
        p = subprocess.run([sys.executable, "-I", TOOL, obj_path, glb_path, "--up", up],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return glb_path

    def test_container_magic_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = self._convert(OBJ_CUBE, tmp=tmp)
            magic, version, chunks = parse_glb(glb)
            self.assertEqual(magic, 0x46546C67)
            self.assertEqual(version, 2)
            # JSON chunk then BIN chunk
            self.assertEqual(chunks[0][0], 0x4E4F534A)  # JSON
            self.assertEqual(chunks[1][0], 0x004E4942)  # BIN
            # JSON is valid
            json_text = chunks[0][1].rstrip(b"\x00 ").decode("utf-8")
            doc = json.loads(json_text)
            self.assertIn("meshes", doc)
            self.assertIn("accessors", doc)
            self.assertIn("bufferViews", doc)
            self.assertIn("buffers", doc)

    def test_mesh_primitive_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = self._convert(OBJ_CUBE, tmp=tmp)
            _, _, chunks = parse_glb(glb)
            doc = json.loads(chunks[0][1].rstrip(b"\x00 ").decode("utf-8"))
            self.assertEqual(len(doc["meshes"]), 1)
            self.assertEqual(len(doc["meshes"][0]["primitives"]), 1)
            # 4 unique verts (dedup), 6 indices (2 tris)
            accs = doc["accessors"]
            # find position accessor
            prim = doc["meshes"][0]["primitives"][0]
            pos_acc = accs[prim["attributes"]["POSITION"]]
            self.assertEqual(pos_acc["count"], 4)

    def test_index_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = self._convert(OBJ_CUBE, tmp=tmp)
            _, _, chunks = parse_glb(glb)
            doc = json.loads(chunks[0][1].rstrip(b"\x00 ").decode("utf-8"))
            prim = doc["meshes"][0]["primitives"][0]
            idx_acc = doc["accessors"][prim["indices"]]
            self.assertEqual(idx_acc["count"], 6)

    def test_empty_obj_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "in.obj")
            glb_path = os.path.join(tmp, "out.glb")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write("")
            p = subprocess.run([sys.executable, "-I", TOOL, obj_path, glb_path],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0, "empty input should fail")
            self.assertIn("no non-empty primitives", p.stderr)

    def test_z_up_axis_swap(self):
        # With --up z, coordinates should be converted; just check it doesn't crash
        # and produces a valid container.
        with tempfile.TemporaryDirectory() as tmp:
            glb = self._convert(OBJ_CUBE, up="z", tmp=tmp)
            magic, version, chunks = parse_glb(glb)
            self.assertEqual(magic, 0x46546C67)


if __name__ == "__main__":
    unittest.main()
