"""Unit tests for tools/batch_glb.py batch OBJ -> glTF 2.0 (.glb) converter.

Usage:
    python -I tests/test_batch_glb.py
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import batch_glb
import dds_png


def _create_minimal_bc1_dds(color: int = 0xF800) -> bytes:
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
    bc1_block = struct.pack("<HHBBBB", color, 0x07E0, 0, 0, 0, 0)
    return bytes(hdr) + bc1_block


class TestBatchGlb(unittest.TestCase):
    def test_cache_evicts_least_recent_payload_and_keeps_errors(self) -> None:
        loader_results = {"a": b"aa", "b": b"bb", "c": b"ccc", "big": b"123456", "bad": None}
        loads = []

        def load(path, max_size=0):
            loads.append(path)
            data = loader_results[path]
            return data, "unreadable" if data is None else ""

        with mock.patch.object(batch_glb, "_PNG_CACHE_BUDGET", 5), \
                mock.patch.object(batch_glb, "_load_png_bytes", load):
            cache = batch_glb.TextureCache()
            for path in ("bad", "a", "b", "a", "c", "big", "big", "a", "b", "bad"):
                self.assertEqual(cache.get_or_load(path)[0], loader_results[path])
        self.assertEqual(loads, ["bad", "a", "b", "c", "big", "big", "b"])

    def test_cache_separates_dds_decode_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "red.dds"
            path.write_bytes(_create_minimal_bc1_dds())
            cache = batch_glb.TextureCache()
            for limit, size in ((1, 1), (0, 2), (1, 1)):
                data, reason = cache.get_or_load(str(path), max_size=limit)
                self.assertEqual(reason, "")
                self.assertEqual(struct.unpack_from(">II", data, 16), (size, size))

    def test_waiters_keep_their_result_after_eviction_or_failure(self) -> None:
        for case in ("evicted", "oversize", "exception"):
            with self.subTest(case=case):
                started = threading.Event()
                joined = threading.Event()
                release = threading.Event()
                calls = []

                class ObservedFuture(Future):
                    def result(self, timeout=None):
                        joined.set()
                        if not release.wait(timeout=5):
                            raise TimeoutError("waiter was not released")
                        return super().result(timeout=5)

                def load(path, max_size=0):
                    calls.append(path)
                    if len(calls) == 1:
                        started.set()
                        self.assertTrue(joined.wait(timeout=5), "no worker joined the pending load")
                        if case == "exception":
                            raise RuntimeError("decoder stopped")
                    return path.encode(), ""

                def check_result(future):
                    if case == "exception":
                        with self.assertRaisesRegex(RuntimeError, "decoder stopped"):
                            future.result(timeout=8)
                    else:
                        self.assertEqual(future.result(timeout=8), (b"shared", ""))

                budget = 2 if case == "oversize" else len(b"shared")
                with mock.patch.object(batch_glb, "_PNG_CACHE_BUDGET", budget), \
                        mock.patch.object(batch_glb.concurrent.futures, "Future", ObservedFuture), \
                        mock.patch.object(batch_glb, "_load_png_bytes", load):
                    cache = batch_glb.TextureCache()
                    with ThreadPoolExecutor(2) as pool:
                        owner = pool.submit(cache.get_or_load, "shared")
                        self.assertTrue(started.wait(timeout=5))
                        waiter = pool.submit(cache.get_or_load, "shared")
                        try:
                            check_result(owner)
                            if case == "evicted":
                                cache.get_or_load("other")
                        finally:
                            release.set()
                        check_result(waiter)
                    self.assertEqual(calls.count("shared"), 1)
                    self.assertEqual(cache.get_or_load("shared"), (b"shared", ""))
                    self.assertEqual(calls.count("shared"), 2)

    def test_parallel_texture_loads_match_serial_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, tex = root / "models", root / "textures"
            src.mkdir()
            tex.mkdir()
            red_dds = _create_minimal_bc1_dds()
            blue_png = dds_png.dds_to_png_bytes(_create_minimal_bc1_dds(0x001F))
            (tex / "red.dds").write_bytes(red_dds)
            (tex / "blue.png").write_bytes(blue_png)
            (tex / "bad.dds").write_bytes(b"invalid DDS")
            models = {"one": "red", "two": "red", "three": "blue", "four": "bad"}
            for name, material in models.items():
                (src / (name + ".obj")).write_text(
                    "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n"
                    f"usemtl {material}\nf 1/1 2/2 3/3\n", encoding="utf-8", newline="\n")
            serial = batch_glb.batch_convert(str(src), str(root / "serial"), jobs=1, texture_dir=str(tex))
            rendezvous = threading.Barrier(2, timeout=5)
            load = batch_glb._load_png_bytes
            calls = []

            def concurrent_load(path, max_size=0):
                name = Path(path).stem
                calls.append(name)
                if name in ("red", "blue"):
                    rendezvous.wait()
                return load(path, max_size)

            with mock.patch.object(batch_glb, "_load_png_bytes", concurrent_load):
                parallel = batch_glb.batch_convert(str(src), str(root / "parallel"), jobs=4, texture_dir=str(tex))
            self.assertEqual(parallel, serial)
            self.assertEqual(parallel["succeeded"], 4)
            self.assertEqual(len(parallel["dropped_textures"]), 1)
            self.assertCountEqual(calls, ["red", "blue", "bad"])
            textures = {"red": dds_png.dds_to_png_bytes(red_dds), "blue": blue_png}
            for name, material in models.items():
                glb = (root / "parallel" / (name + ".glb")).read_bytes()
                self.assertEqual(glb, (root / "serial" / (name + ".glb")).read_bytes())
                json_size, = struct.unpack_from("<I", glb, 12)
                gltf = json.loads(glb[20:20 + json_size])
                if material == "bad":
                    self.assertFalse(gltf.get("images"))
                else:
                    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
                    start = 28 + json_size + view.get("byteOffset", 0)
                    self.assertEqual(glb[start:start + view["byteLength"]], textures[material])

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
                        capture_output=True, text=True, errors="replace", check=False,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertTrue(result.stderr.startswith("ERROR: "), result.stderr)
                    self.assertIn(error, result.stderr)
                    self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_cli_rejects_invalid_jobs_and_max_tex_size(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for flag, value, error in [
                ("--jobs", "0", "argument --jobs"),
                ("--jobs", "-1", "argument --jobs"),
                ("--max-tex-size", "-1", "argument --max-tex-size"),
            ]:
                with self.subTest(flag=flag, value=value):
                    result = subprocess.run(
                        [
                            sys.executable, "-I",
                            os.path.join(TOOLS, "batch_glb.py"),
                            root, os.path.join(root, "out"),
                            flag, value,
                        ],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(error, result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_cli_accepts_valid_jobs_and_max_tex_size(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            obj_path = os.path.join(in_dir, "test_model.obj")
            with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "o test\n"
                    "v 0.0 0.0 0.0\n"
                    "v 1.0 0.0 0.0\n"
                    "v 0.0 1.0 0.0\n"
                    "f 1 2 3\n"
                )

            # zero texture size (unconstrained) and a positive jobs limit are valid
            result = subprocess.run(
                [
                    sys.executable, "-I",
                    os.path.join(TOOLS, "batch_glb.py"),
                    in_dir, out_dir,
                    "--jobs", "1",
                    "--max-tex-size", "0",
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("converted 1/1 models", result.stdout)

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

    def test_unnamed_primitive_keeps_its_own_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            in_dir, out_dir, tex_dir = root / "obj", root / "glb", root / "textures"
            in_dir.mkdir()
            tex_dir.mkdir()
            (in_dir / "panel.detail.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\n"
                "f 1/1 2/2 3/3\nusemtl panel\nf 1/1 2/2 3/3\n",
                encoding="utf-8",
            )
            red_png = dds_png.dds_to_png_bytes(_create_minimal_bc1_dds())
            blue_png = dds_png.dds_to_png_bytes(_create_minimal_bc1_dds(0x001F))
            (tex_dir / "panel.detail.png").write_bytes(red_png)
            (tex_dir / "panel.png").write_bytes(blue_png)

            stats = batch_glb.batch_convert(str(in_dir), str(out_dir), texture_dir=str(tex_dir), jobs=1)
            # batch_glb also names failures and dropped textures (d876da115); none here.
            self.assertEqual(stats, {"total": 1, "succeeded": 1, "failed": 0, "failures": [], "dropped_textures": []})
            glb = (out_dir / "panel.detail.glb").read_bytes()
            json_size, = struct.unpack_from("<I", glb, 12)
            gltf = json.loads(glb[20:20 + json_size])
            binary = glb[28 + json_size:]
            textures = []
            for primitive in gltf["meshes"][0]["primitives"]:
                material = gltf["materials"][primitive["material"]]
                texture = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                image = gltf["images"][gltf["textures"][texture]["source"]]
                view = gltf["bufferViews"][image["bufferView"]]
                offset = view.get("byteOffset", 0)
                textures.append(binary[offset:offset + view["byteLength"]])

            self.assertEqual(textures, [red_png, blue_png])

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


    def test_undecodable_dds_is_reported_and_decoded_once(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            with open(os.path.join(tex_dir, "car_body.dds"), "wb") as f:
                f.write(b"DDS " + b"\x00" * 20)      # magic, then a truncated header
            # Two groups, one material: two primitives that both want car_body.dds
            with open(os.path.join(in_dir, "car_body.obj"), "w", encoding="utf-8", newline="\n") as f:
                f.write("o car_body\n"
                        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                        "g door\nusemtl car_body\nf 1 2 3\n"
                        "g hood\nusemtl car_body\nf 1 2 3\n")

            calls = []

            def broken_decode(path, max_size=0):
                calls.append(path)
                raise ValueError("synthetic decode failure")

            real = dds_png.dds_file_to_png_bytes
            dds_png.dds_file_to_png_bytes = broken_decode
            try:
                stats = batch_glb.batch_convert(in_dir, out_dir, up_axis="z", texture_dir=tex_dir)
            finally:
                dds_png.dds_file_to_png_bytes = real

            self.assertEqual(stats["succeeded"], 1)          # the model still converts
            self.assertEqual(len(calls), 1)                  # ...and the broken DDS is decoded once, then cached
            drops = [(os.path.basename(m), os.path.basename(t), w) for m, t, w in stats["dropped_textures"]]
            self.assertEqual([d[1] for d in drops], ["car_body.dds"])   # one line per texture, not per primitive
            self.assertTrue(all("synthetic decode failure" in d[2] for d in drops))
            with open(os.path.join(out_dir, "car_body.glb"), "rb") as f:
                self.assertNotIn(b"image/png", f.read())     # no texture, and it says so

    def test_broken_dds_shared_by_two_models_is_decoded_once_across_workers(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            with open(os.path.join(tex_dir, "car_body.dds"), "wb") as f:
                f.write(b"DDS " + b"\x00" * 20)
            for name in ("model_a", "model_b"):              # two models: the pool runs them in parallel
                with open(os.path.join(in_dir, name + ".obj"), "w", encoding="utf-8", newline="\n") as f:
                    f.write("o %s\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl car_body\nf 1 2 3\n" % name)

            calls = []
            both_inside = threading.Barrier(2, timeout=2.0)  # releases only if the lock is missing

            def slow_broken_decode(path, max_size=0):
                calls.append(path)
                try:
                    both_inside.wait()                       # a second thread proves the cache is not exclusive
                except threading.BrokenBarrierError:
                    pass                                     # alone inside: the cache held the second one out
                raise ValueError("synthetic decode failure")

            real = dds_png.dds_file_to_png_bytes
            dds_png.dds_file_to_png_bytes = slow_broken_decode
            try:
                stats = batch_glb.batch_convert(in_dir, out_dir, up_axis="z", texture_dir=tex_dir, jobs=2)
            finally:
                dds_png.dds_file_to_png_bytes = real

            self.assertEqual(stats["succeeded"], 2)
            self.assertEqual(len(calls), 1)                  # one decode per batch, not one per worker
            self.assertEqual(len(stats["dropped_textures"]), 2)   # one line per model

    def test_empty_png_is_reported_not_silent(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            with open(os.path.join(tex_dir, "car_body.png"), "wb") as f:
                pass                                         # exists, but zero bytes
            with open(os.path.join(in_dir, "car_body.obj"), "w", encoding="utf-8", newline="\n") as f:
                f.write("o car_body\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

            stats = batch_glb.batch_convert(in_dir, out_dir, up_axis="z", texture_dir=tex_dir)
            self.assertEqual(stats["succeeded"], 1)
            self.assertEqual(len(stats["dropped_textures"]), 1)
            _, tex, why = stats["dropped_textures"][0]
            self.assertTrue(tex.endswith("car_body.png"))
            self.assertTrue(why)
            with open(os.path.join(out_dir, "car_body.glb"), "rb") as f:
                self.assertNotIn(b"image/png", f.read())

    def test_unreadable_png_is_reported_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            with open(os.path.join(in_dir, "car_body.obj"), "w", encoding="utf-8", newline="\n") as f:
                f.write("o car_body\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            missing = os.path.join(in_dir, "deleted_mid_run.png")

            ok, path, err, dropped = batch_glb.convert_single(
                (os.path.join(in_dir, "car_body.obj"), os.path.join(out_dir, "car_body.glb"), "z",
                 {"car_body": missing}, 0, batch_glb.TextureCache())
            )
            self.assertTrue(ok)
            self.assertEqual(err, "")
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0][0], missing)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "car_body.glb")))

    def test_failed_models_are_named_with_their_error(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            with open(os.path.join(in_dir, "broken.obj"), "w", encoding="utf-8", newline="\n") as f:
                f.write("o broken\nf 1 2 3\n")               # faces with no vertices: parse_obj refuses

            stats = batch_glb.batch_convert(in_dir, out_dir, up_axis="z")
            self.assertEqual((stats["succeeded"], stats["failed"]), (0, 1))
            self.assertEqual(len(stats["failures"]), 1)
            path, err = stats["failures"][0]
            self.assertTrue(path.endswith("broken.obj"))
            self.assertTrue(err)
            self.assertFalse(os.path.exists(os.path.join(out_dir, "broken.glb")))

    def test_main_names_failures_and_dropped_textures(self) -> None:
        import contextlib
        import io
        import subprocess

        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            with open(os.path.join(in_dir, "car_body.obj"), "w", encoding="utf-8", newline="\n") as f:
                f.write("o car_body\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            with open(os.path.join(in_dir, "broken.obj"), "w", encoding="utf-8", newline="\n") as f:
                f.write("o broken\nf 1 2 3\n")
            with open(os.path.join(tex_dir, "car_body.dds"), "wb") as f:
                f.write(b"DDS " + b"\x00" * 20)

            result = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "batch_glb.py"), in_dir, out_dir,
                 "--textures", tex_dir],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("broken.obj", result.stdout)
            self.assertIn("texture", result.stdout)
            self.assertIn("car_body.dds", result.stdout)
            self.assertIn("not embedded", result.stdout)
            self.assertIn("(1 failed, 1 unembedded texture)", result.stdout)


if __name__ == "__main__":
    unittest.main()
