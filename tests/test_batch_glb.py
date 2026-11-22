"""Unit tests for tools/batch_glb.py batch OBJ -> glTF 2.0 (.glb) converter.

Usage:
    python -I tests/test_batch_glb.py
"""
from __future__ import annotations

import contextlib
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import batch_glb
import dds_png


def _create_minimal_bc1_dds() -> bytes:
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)
    struct.pack_into("<I", hdr, 8, 0x1007)
    struct.pack_into("<I", hdr, 12, 2)
    struct.pack_into("<I", hdr, 16, 2)
    struct.pack_into("<I", hdr, 20, 8)
    struct.pack_into("<I", hdr, 76, 32)
    struct.pack_into("<I", hdr, 80, 0x4)
    hdr[84:88] = b"DXT1"
    struct.pack_into("<I", hdr, 108, 0x1000)
    bc1_block = struct.pack("<HHBBBB", 0xF800, 0x07E0, 0, 0, 0, 0)
    return bytes(hdr) + bc1_block


class TestBatchGlb(unittest.TestCase):
    def test_cli_reports_directory_setup_errors(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            missing = os.path.join(root, "missing")
            blocked = os.path.join(root, "blocked")
            with open(blocked, "wb"):
                pass

            for source, output, error in [
                (missing, os.path.join(root, "out"), "Input directory does not exist"),
                (root, blocked, "blocked"),
            ]:
                with self.subTest(source=source, output=output):
                    result = subprocess.run(
                        [sys.executable, "-I", os.path.join(TOOLS, "batch_glb.py"), source, output],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertTrue(result.stderr.startswith("ERROR: "), result.stderr)
                    self.assertIn(error, result.stderr)
                    self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_batch_convert_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            for i in range(3):
                obj_path = os.path.join(in_dir, f"test_model_{i}.obj")
                with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(
                        "o test\n"
                        "v 0.0 0.0 0.0\n"
                        "v 1.0 0.0 0.0\n"
                        "v 0.0 1.0 0.0\n"
                        "f 1 2 3\n"
                    )

            stats = batch_glb.batch_convert(in_dir, out_dir, up_axis="z", jobs=2)
            self.assertEqual(stats["total"], 3)
            self.assertEqual(stats["succeeded"], 3)
            self.assertEqual(stats["failed"], 0)

            for i in range(3):
                glb_file = os.path.join(out_dir, f"test_model_{i}.glb")
                self.assertTrue(os.path.isfile(glb_file))
                self.assertGreater(os.path.getsize(glb_file), 100)

    def test_batch_convert_with_png_texture(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            # Create a 2x2 PNG texture
            dds_data = _create_minimal_bc1_dds()
            png_bytes = dds_png.dds_to_png_bytes(dds_data)
            tex_file = os.path.join(tex_dir, "mat_custom.png")
            with open(tex_file, "wb") as f:
                f.write(png_bytes)

            # Create an OBJ with UVs and material reference
            obj_path = os.path.join(in_dir, "textured_cube.obj")
            with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "o textured_cube\n"
                    "usemtl mat_custom\n"
                    "v 0.0 0.0 0.0\n"
                    "v 1.0 0.0 0.0\n"
                    "v 0.0 1.0 0.0\n"
                    "vt 0.0 0.0\n"
                    "vt 1.0 0.0\n"
                    "vt 0.0 1.0\n"
                    "f 1/1 2/2 3/3\n"
                )

            stats = batch_glb.batch_convert(
                in_dir, out_dir, up_axis="y", texture_dir=tex_dir
            )
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["succeeded"], 1)
            self.assertEqual(stats["failed"], 0)

            glb_file = os.path.join(out_dir, "textured_cube.glb")
            self.assertTrue(os.path.isfile(glb_file))
            with open(glb_file, "rb") as f:
                content = f.read()
            self.assertIn(b"image/png", content)
            self.assertIn(b"mat_custom", content)

    def test_batch_convert_with_dds_texture(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            # Create a DDS texture directly
            dds_data = _create_minimal_bc1_dds()
            tex_file = os.path.join(tex_dir, "car_body.dds")
            with open(tex_file, "wb") as f:
                f.write(dds_data)

            obj_path = os.path.join(in_dir, "car_body.obj")
            with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "o car_body\n"
                    "v 0.0 0.0 0.0\n"
                    "v 1.0 0.0 0.0\n"
                    "v 0.0 1.0 0.0\n"
                    "vt 0.0 0.0\n"
                    "vt 1.0 0.0\n"
                    "vt 0.0 1.0\n"
                    "f 1/1 2/2 3/3\n"
                )

            stats = batch_glb.batch_convert(
                in_dir, out_dir, up_axis="z", texture_dir=tex_dir
            )
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["succeeded"], 1)

            glb_file = os.path.join(out_dir, "car_body.glb")
            self.assertTrue(os.path.isfile(glb_file))
            with open(glb_file, "rb") as f:
                content = f.read()
            self.assertIn(b"image/png", content)


class TestBatchGlbOutputCollisions(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = temporary.name
        self.in_dir = os.path.join(root, "input")
        self.out_dir = os.path.join(root, "output")
        os.mkdir(self.in_dir)
        os.mkdir(self.out_dir)
        self.inputs = [os.path.join(self.in_dir, name) for name in ("model.obj", "model.OBJ")]
        self.output = os.path.join(self.out_dir, "model.glb")
        self.previous = b"previous model output\x00\xff"
        with open(self.output, "wb") as output:
            output.write(self.previous)
        self.write_obj(self.inputs[0], 1)

    @staticmethod
    def write_obj(path: str, extent: int) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as obj:
            obj.write(f"v 0 0 0\nv {extent} 0 0\nv 0 {extent} 0\nf 1 2 3\n")

    @contextlib.contextmanager
    def colliding_listing(self):
        # A case-sensitive source can contain both names, even when the test host cannot.
        isfile = os.path.isfile
        with mock.patch.object(batch_glb.os, "listdir", return_value=["model.obj", "model.OBJ"]), \
             mock.patch.object(batch_glb.os.path, "isfile",
                               side_effect=lambda path: path in self.inputs or isfile(path)):
            yield

    def assert_previous_output(self) -> None:
        with open(self.output, "rb") as output:
            self.assertEqual(output.read(), self.previous)

    def test_rejects_collision_before_texture_loading_or_workers(self) -> None:
        with self.colliding_listing(), \
             mock.patch.object(batch_glb, "_build_texture_index") as textures, \
             mock.patch.object(batch_glb.concurrent.futures, "ThreadPoolExecutor") as executor, \
             mock.patch.object(batch_glb, "convert_single") as convert:
            with self.assertRaises(ValueError) as error:
                batch_glb.batch_convert(self.in_dir, self.out_dir, jobs=2, texture_dir="textures")
            for name in ("model.obj", "model.OBJ", "model.glb"):
                self.assertIn(name, str(error.exception))
            textures.assert_not_called()
            executor.assert_not_called()
            convert.assert_not_called()
        self.assert_previous_output()

    def test_cli_reports_collision_once_without_a_success_summary(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with self.colliding_listing(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
             mock.patch.object(batch_glb, "convert_single", side_effect=lambda task: (True, task[0], "")) as convert:
            result = batch_glb.main([self.in_dir, self.out_dir, "--jobs", "2"])
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1, stderr.getvalue())
        self.assertTrue(lines[0].startswith("ERROR: "), lines[0])
        for name in ("model.obj", "model.OBJ", "model.glb"):
            self.assertIn(name, lines[0])
        convert.assert_not_called()
        self.assert_previous_output()

    def test_cli_rejects_two_distinct_files_with_the_same_output(self) -> None:
        if os.path.exists(self.inputs[1]):
            self.skipTest("source volume folds case")
        self.write_obj(self.inputs[1], 2)
        result = subprocess.run(
            [sys.executable, "-I", os.path.join(TOOLS, "batch_glb.py"),
             self.in_dir, self.out_dir, "--jobs", "2"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
        self.assertTrue(result.stderr.startswith("ERROR: "), result.stderr)
        for name in ("model.obj", "model.OBJ", "model.glb"):
            self.assertIn(name, result.stderr)
        self.assert_previous_output()

    def test_distinct_stems_with_mixed_case_extensions_export_both_models(self) -> None:
        self.write_obj(os.path.join(self.in_dir, "other.OBJ"), 2)
        stats = batch_glb.batch_convert(self.in_dir, self.out_dir, jobs=2)
        self.assertEqual(stats, {"total": 2, "succeeded": 2, "failed": 0})
        for name in ("model.glb", "other.glb"):
            with open(os.path.join(self.out_dir, name), "rb") as output:
                contents = output.read()
            magic, version, length = struct.unpack_from("<4sII", contents)
            self.assertEqual((magic, version, length), (b"glTF", 2, len(contents)))


if __name__ == "__main__":
    unittest.main()
