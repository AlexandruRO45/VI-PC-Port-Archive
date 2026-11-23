"""First Light mesh preparation, replay and opt-in PNG output."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import first_light


class FirstLightMeshDrawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.work / "out"
        self.out.mkdir()
        self.mesh = self.out / "mesh.bin"
        self.mesh.write_bytes(b"previous mesh placeholder")
        dxbc = self.work / "shaders" / "dxbc"
        spirv = self.work / "first_light" / "spirv"
        dxbc.mkdir(parents=True)
        spirv.mkdir(parents=True)
        picks = []
        for stage in ("vs", "ps"):
            (dxbc / f"{stage}.dxbc").touch()
            (spirv / f"{stage}.spv").touch()
            picks.append({"stage": stage, "name": stage, "file": f"{stage}.dxbc"})
        self.enterContext(patch.multiple(
            first_light,
            ensure_carve=lambda *args: "stub-index",
            choose=lambda *args: picks,
            translate=lambda *args: {"vs": {}, "ps": {}},
            find_harness=lambda *args: "stub-harness",
        ))
        self.draw = self.enterContext(patch.object(first_light.subprocess, "run", side_effect=self.write_readback))

    @staticmethod
    def write_readback(arguments: list[str], *, env: dict[str, str], check: bool) -> None:
        if not Path(env["VKH_DRAW_MESH"]).is_file():
            raise AssertionError("the harness received no mesh")
        pixels = (0.25, 0.75, 0.0, 1.0) * 2 + (-1.0,) * 8
        Path(arguments[3]).write_bytes(struct.pack("<16f", *pixels))

    def prepare(self, *args: str) -> int:
        return first_light.main([
            "--game", str(self.work), "--work", str(self.work), "--out", str(self.out),
            "--no-fit", "--width", "2", "--height", "2", *args,
        ])

    def test_run_without_obj_never_draws_the_previous_mesh(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires an OBJ with faces"):
            self.prepare("--run")
        self.draw.assert_not_called()
        self.assertFalse((self.out / "image.raw").exists())

    def test_prepare_without_obj_is_allowed_but_cannot_be_replayed(self) -> None:
        self.assertEqual(self.prepare(), 0)
        self.draw.assert_not_called()
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["vertex_count"], manifest["index_count"]), (0, 0))
        self.assertFalse(self.mesh.is_file())   # a run without an OBJ removes the previous run's mesh.bin (95ebfb8e9)
        with self.assertRaisesRegex(SystemExit, "requires an OBJ with faces"):
            first_light.main(["--check", "--out", str(self.out)])
        self.draw.assert_not_called()

    def test_check_can_replay_a_manifest_for_prepared_geometry(self) -> None:
        self.prepare_replay()
        self.assertEqual(first_light.main(["--check", "--out", str(self.out)]), 0)
        self.draw.assert_called_once()
        self.assertTrue((self.out / "image.ppm").is_file())
        self.assertFalse((self.out / "image.png").exists())

    def test_run_draw_defaults_to_no_png_without_png_option(self) -> None:
        self.prepare_replay()
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        result = first_light.run_draw(SimpleNamespace(harness=None), manifest, str(self.out))
        self.assertEqual(result["covered"], 2)
        self.assertTrue((self.out / "image.ppm").is_file())
        self.assertFalse((self.out / "image.png").exists())

    def prepare_replay(self) -> None:
        vertices = [(0.0, 0.0, 0.5, 0.0, 0.0), (1.0, 0.0, 0.5, 1.0, 0.0), (0.0, 1.0, 0.5, 0.0, 1.0)]
        first_light.write_mesh(self.mesh, vertices, [0, 1, 2])
        (self.out / "manifest.json").write_text(
            '{"width": 2, "height": 2, "vertex_count": 3, "index_count": 3, "min_coverage": 0.5, '
            '"clear_pixel": [0, 1], "uv_probes": [{"pixel": [0, 0], "uv": [0.25, 0.75]}], '
            '"files": {"mesh": "mesh.bin"}}',
            encoding="utf-8", newline="\n",
        )

    def assert_preview_pixels(self) -> None:
        png = (self.out / "image.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack_from(">I4sIIBB", png, 8), (13, b"IHDR", 2, 2, 8, 6))
        size, chunk = struct.unpack_from(">I4s", png, 33)
        self.assertEqual(chunk, b"IDAT")
        rows = zlib.decompress(png[41 : 41 + size])
        self.assertEqual(rows, b"\x00" + bytes((64, 191, 0, 255)) * 2 + b"\x00" * 9)
        self.assertEqual(
            (self.out / "image.ppm").read_bytes(),
            b"P6\n2 2\n255\n" + bytes((64, 191, 0)) * 2 + b"\x00" * 6,
        )

    def test_check_png_exports_only_a_fresh_passing_readback(self) -> None:
        self.prepare_replay()
        args = ["--check", "--png", "--out", str(self.out)]
        with patch.object(first_light, "ensure_carve", side_effect=AssertionError("replay must not prepare")):
            self.assertEqual(first_light.main(args), 0)
        self.assert_preview_pixels()
        self.draw.side_effect = lambda *args, **kwargs: None
        with self.assertRaisesRegex(SystemExit, "did not write"):
            first_light.main(args)
        self.assertFalse((self.out / "image.png").exists())
        self.assertFalse((self.out / "image.raw").exists())

    def test_check_png_drops_the_previous_preview_when_the_oracle_fails(self) -> None:
        self.prepare_replay()
        png = self.out / "image.png"
        png.write_bytes(b"previous preview")

        def write_clear(arguments: list[str], **kwargs) -> None:
            Path(arguments[3]).write_bytes(struct.pack("<16f", *([-1.0] * 16)))

        self.draw.side_effect = write_clear
        with self.assertRaisesRegex(SystemExit, "coverage"):
            first_light.main(["--check", "--png", "--out", str(self.out)])
        self.assertFalse(png.exists())
        self.assertTrue((self.out / "image.raw").exists())

    def test_check_png_reports_a_failed_harness(self) -> None:
        self.prepare_replay()
        png = self.out / "image.png"
        png.write_bytes(b"previous preview")
        self.draw.side_effect = subprocess.CalledProcessError(2, "stub-harness")
        with self.assertRaisesRegex(SystemExit, "stub-harness.*exit status 2"):
            first_light.main(["--check", "--png", "--out", str(self.out)])
        self.assertFalse(png.exists())

    def test_run_png_uses_the_prepared_mesh(self) -> None:
        obj = self.work / "mesh.obj"
        obj.write_text(
            "v -0.8 -0.8 0.5\nv 0.8 -0.8 0.5\nv 0.8 0.0 0.5\nv -0.8 0.0 0.5\n"
            "vt 0.25 0.75\nf 1/1 2/1 3/1 4/1\n",
            encoding="utf-8", newline="\n",
        )
        self.assertEqual(self.prepare("--obj", str(obj), "--run", "--png"), 0)
        self.assert_preview_pixels()


if __name__ == "__main__":
    unittest.main()
