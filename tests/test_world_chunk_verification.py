"""Bounded staged-GLB validation and buffer lifetime checks, using synthetic writer output."""
from __future__ import annotations

import array
import gc
import hashlib
import mmap
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glb_writer
import world_build


def mesh(triangles=1):
    primitive = glb_writer.pack_primitive({
        "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        "normals": [(0, 0, 1)] * 3,
        "indices": [0, 1, 2],
    })
    primitive["positions"] *= triangles
    primitive["normals"] *= triangles
    vertices = triangles * 3
    indices = array.array("I", range(vertices))
    if indices.itemsize != 4:
        raise unittest.SkipTest("fixture requires 32-bit array integers")
    if sys.byteorder != "little":
        indices.byteswap()
    primitive.update(indices=indices.tobytes(), index_type=5125,
                     vertices=vertices, index_count=vertices)
    return {"primitives": [primitive],
            "instances": [((0, 0, 0), (0, 0, 0, 1), (1, 1, 1))]}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            result.update(block)
    return result.digest()


class StagedChunkVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_real_writer_chunk_verification_does_not_copy_the_binary_payload(self):
        path = self.root / "large.glb"
        stats = glb_writer.write_instanced_glb(str(path), [mesh(100_000)], mode="nodes")
        size = path.stat().st_size
        self.assertEqual(size, stats["bytes"])
        self.assertGreater(size, 8_000_000)
        before = digest(path)

        # Isolate the verifier from fixture/import peaks and leave caller-owned tracing alone.
        was_tracing = tracemalloc.is_tracing()
        measured = subprocess.run([
            sys.executable, "-I", "-c", """
import gc
import sys
import tracemalloc
sys.path.insert(0, sys.argv[1])
import glb_writer
import world_build
path = sys.argv[2]
size = int(sys.argv[3])
gc.collect()
tracemalloc.start()
try:
    world_build._verify_staged_chunk(path, size, glb_writer)
    peak = tracemalloc.get_traced_memory()[1]
finally:
    tracemalloc.stop()
print(peak)
""", str(Path(__file__).resolve().parents[1] / "tools"), str(path), str(size),
        ], capture_output=True, text=True, check=True, timeout=30)
        peak = int(measured.stdout)
        self.assertEqual(tracemalloc.is_tracing(), was_tracing)
        # JSON metadata still allocates; the 8.4 MB geometry payload need not.
        self.assertLess(peak, 2 * 1024 * 1024, f"verification allocated {peak} bytes")
        self.assertEqual(digest(path), before)

    def test_size_and_format_errors_preserve_the_file_and_release_handles(self):
        valid, _ = glb_writer.build_instanced_glb([mesh()], mode="nodes")
        bad_json = bytearray(valid)
        bad_json[20] = ord("!")
        bad_bin = bytearray(valid)
        bin_header = 20 + struct.unpack_from("<I", valid, 12)[0]
        struct.pack_into("<I", bad_bin, bin_header, len(valid) * 4)
        cases = (
            ("size", valid, len(valid) + 1, f"is {len(valid)} bytes, expected {len(valid) + 1}"),
            ("empty", b"", 0, "GLB data too short: 0 bytes"),
            ("short", valid[:12], 12, "GLB data too short: 12 bytes"),
            ("magic", b"nope" + valid[4:], len(valid), "invalid GLB magic"),
            ("json", bytes(bad_json), len(valid), "invalid JSON chunk"),
            ("bin", bytes(bad_bin), len(valid), "exceeds GLB total length"),
        )
        for name, data, expected_size, message in cases:
            with self.subTest(name=name):
                path = self.root / (name + ".glb")
                path.write_bytes(data)
                error = None
                try:
                    world_build._verify_staged_chunk(str(path), expected_size, glb_writer)
                except glb_writer.GlbError as caught:
                    error = caught
                self.assertIsNotNone(error, "invalid staged GLB was accepted")
                self.assertIn(message, str(error))
                self.assertEqual(path.read_bytes(), data)
                # Keep the exception and its traceback alive during Windows rename/removal.
                moved = path.with_suffix(".moved")
                path.rename(moved)
                moved.unlink()
                self.assertIsNotNone(error.__traceback__)

    def test_summary_releases_its_view_without_releasing_the_callers_view(self):
        valid, _ = glb_writer.build_instanced_glb([mesh()], mode="nodes")
        bad_json = bytearray(valid)
        bad_json[20] = ord("!")
        expected = glb_writer.read_glb_summary(valid)
        for name, data in (("valid", valid), ("magic", b"nope" + valid[4:]),
                           ("json", bytes(bad_json))):
            with self.subTest(name=name):
                path = self.root / (name + ".glb")
                path.write_bytes(data)
                with path.open("rb") as source:
                    mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                view = memoryview(mapping)
                error = None
                try:
                    try:
                        result = glb_writer.read_glb_summary(view)
                    except glb_writer.GlbError as caught:
                        error = caught
                    if name == "valid":
                        self.assertIsNone(error)
                        self.assertEqual(result, expected)
                    else:
                        self.assertIsNotNone(error)
                    self.assertEqual(bytes(view[:4]), data[:4])
                    view.release()
                    # A live parser traceback must not keep another exported view.
                    mapping.close()
                    self.assertTrue(mapping.closed)
                finally:
                    view.release()
                    if error is not None:
                        error.__traceback__ = None
                        error.__cause__ = None
                        error.__context__ = None
                    gc.collect()
                    if not mapping.closed:
                        mapping.close()


if __name__ == "__main__":
    unittest.main()
