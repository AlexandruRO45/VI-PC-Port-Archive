"""Real GLB imports retain owned PNGs without whole-file copies or open mappings."""
from __future__ import annotations

import copy
import hashlib
import json
import mmap
from pathlib import Path
import random
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dds_png
import ped_assemble as pa
from test_ped_glb_textures import container, split_glb

SLOTS = ("base_color_png", "normal_png", "metallic_roughness_png")
POSITIONS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

# An isolated child ignores any parent tracing state and imports before measuring.
MEASURE = r'''
import hashlib, json, sys, tracemalloc
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ped_assemble as pa
source = Path(sys.argv[2])
before = hashlib.sha256(source.read_bytes()).hexdigest()
tracemalloc.start()
parts = pa.read_glb_parts(str(source))
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
images = [getattr(part.material, slot) for part in parts for slot in
          ("base_color_png", "normal_png", "metallic_roughness_png")]
print(json.dumps({"current": current, "peak": peak, "parts": len(parts),
                  "positions": [part.positions for part in parts],
                  "png_hashes": [hashlib.sha256(image).hexdigest() for image in images],
                  "png_types": [type(image).__name__ for image in images],
                  "distinct_pngs": len({id(image) for image in images}),
                  "source_unchanged": before == hashlib.sha256(source.read_bytes()).hexdigest()}))
'''


class PedGlbImportMemoryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / "source.glb"
        self.png = dds_png.encode_png(1, 1, bytes((29, 43, 71, 255)))
        self.write_source(self.png)

    def write_source(self, png):
        self.original = pa.MeshPart(
            name="triangle", coord_space="y-up", positions=POSITIONS, triangles=[(0, 1, 2)],
            normals=[(0.0, 0.0, 1.0)] * 3, uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            material=pa.MaterialSpec(name="shared", base_color_factor=(0.25, 0.5, 0.75, 1.0),
                                     base_color_png=png, normal_png=png, metallic_roughness_png=png),
        )
        pa.write_assembled_glb(str(self.source), [self.original], apply_standing=False)

    def track_mappings(self):
        mappings = []
        real_mmap = mmap.mmap

        def create(*args, **kwargs):
            mapped = real_mmap(*args, **kwargs)
            mappings.append(mapped)
            return mapped

        return mappings, mock.patch.object(mmap, "mmap", side_effect=create)

    def test_import_allocation_is_owned_payload_plus_small_metadata(self):
        png = dds_png.encode_png(1024, 1024, random.Random(505).randbytes(1024 * 1024 * 4))
        self.write_source(png)
        result = subprocess.run(
            [sys.executable, "-I", "-c", MEASURE, str(ROOT / "tools"), str(self.source)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["parts"], 1)
        self.assertEqual(report["positions"], [[list(v) for v in POSITIONS]])
        self.assertEqual(report["png_hashes"], [hashlib.sha256(png).hexdigest()] * 3)
        self.assertEqual(report["png_types"], ["bytes"] * 3)
        self.assertEqual(report["distinct_pngs"], 1)
        self.assertTrue(report["source_unchanged"])
        # This allows the retained PNG and ample JSON/part overhead, but not a second payload copy.
        self.assertLess(report["peak"], len(png) + 2 * 1024 * 1024, report)

    def test_returned_parts_outlive_the_readonly_source_and_keep_shared_bytes(self):
        before = self.source.read_bytes()
        self.source.chmod(stat.S_IREAD)
        self.addCleanup(lambda: self.source.chmod(stat.S_IWRITE | stat.S_IREAD) if self.source.exists() else None)
        mappings, track = self.track_mappings()
        with track:
            parts = pa.read_glb_parts(str(self.source))
        self.assertTrue(all(mapped.closed for mapped in mappings))
        self.assertEqual(self.source.read_bytes(), before)
        self.source.chmod(stat.S_IWRITE | stat.S_IREAD)
        self.source.unlink()  # Windows refuses this if a file mapping is still open.
        part, = parts
        self.assertEqual(part.material, self.original.material)
        images = [getattr(part.material, slot) for slot in SLOTS]
        self.assertTrue(all(type(image) is bytes for image in images))
        self.assertEqual(len({id(image) for image in images}), 1)
        output = self.root / "owned.glb"
        pa.write_assembled_glb(str(output), parts, apply_standing=False)
        restored, = pa.read_glb_parts(str(output))
        self.assertEqual(restored.positions, POSITIONS)
        self.assertEqual(restored.material, self.original.material)

    def test_errors_release_mappings_with_live_exception_tracebacks(self):
        original = self.source.read_bytes()
        doc, payload = split_glb(original)
        bad_png = copy.deepcopy(doc)
        # Base color has populated the cache before this second slot fails.
        bad_png["materials"][0]["normalTexture"]["index"] = len(doc["textures"])
        no_json = struct.pack("<4sIIII", b"glTF", 2, 20 + len(payload), len(payload), pa.CHUNK_TYPE_BIN) + payload
        truncated = bytearray(original)
        struct.pack_into("<I", truncated, 8, len(original) + 4)
        malformed_json = bytearray(original)
        malformed_json[20] = ord("!")
        invalid_utf8 = bytearray(original)
        invalid_utf8[20] = 255
        cases = (
            ("empty", b"", pa.PedAssembleError, "not a valid GLB container"),
            ("short", b"glTF", pa.PedAssembleError, "not a valid GLB container"),
            ("truncated", bytes(truncated), pa.PedAssembleError, "unsupported GLB version or truncated length"),
            ("no-json", no_json, pa.PedAssembleError, "GLB has no JSON chunk"),
            ("bad-json", bytes(malformed_json), ValueError, "Expecting value"),
            ("bad-utf8", bytes(invalid_utf8), ValueError, "decode"),
            ("short-accessor", container(doc, b""), (struct.error, pa.PedAssembleError), "buffer|BIN chunk"),
            ("bad-png", container(bad_png, payload), pa.PedAssembleError, "normalTexture"),
        )
        held_errors = []
        for label, data, kind, message in cases:
            with self.subTest(case=label):
                self.source.write_bytes(data)
                mappings, track = self.track_mappings()
                error = None
                with track:
                    try:
                        pa.read_glb_parts(str(self.source))
                    except Exception as caught:
                        error = caught
                self.assertIsInstance(error, kind)
                self.assertRegex(str(error), message)
                self.assertIsNotNone(error.__traceback__)
                held_errors.append(error)  # Do not clear tracebacks or rely on garbage collection.
                self.assertTrue(all(mapped.closed for mapped in mappings))
                self.assertEqual(self.source.read_bytes(), data)
                self.source.unlink()
        self.assertEqual(len(held_errors), len(cases))


if __name__ == "__main__":
    unittest.main()
