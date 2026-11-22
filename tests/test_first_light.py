"""Package-gated first-light test: draw the real model a first_light.py run assembled.

tools/first_light.py writes a directory with vs.spv, ps.spv, mesh.bin and manifest.json from the user's own
package. This test runs vk_harness on that directory and checks the pixels against the manifest: a coverage
window, a corner the mesh does not cover (the harness clear value) and the UV the model's last triangle writes at
the centre pixel. The oracle is also exercised on synthetic images, without a GPU or a package, so the
all-opaque and swapped-UV negatives stay red.

    LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py
    LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/camera python -I tests/test_first_light.py   # --camera

The image is written to the directory as image_first_light.raw/.ppm; no game data is in the repository. The camera
tests check --camera's matrices, its constants (read back the way CAMERA_VS loads them) and the perspective-correct
UV prediction on the CPU.
"""

import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import first_light as fl

CLEAR = fl.CLEAR
check_image = fl.check_image
write_ppm = fl.write_ppm


def harness() -> str | None:
    exe = ".exe" if os.name == "nt" else ""
    for path in (
        os.environ.get("LEONIDA_VK_HARNESS"),
        os.path.join(ROOT, "runtime", "build", "linux", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", "windows", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", f"vk_harness{exe}"),
    ):
        if path and os.path.isfile(path):
            return path
    return None


class FirstLightTests(unittest.TestCase):
    def setUp(self):
        self.out = os.environ.get("LEONIDA_FIRST_LIGHT_DIR")
        if not self.out:
            self.skipTest(
                "LEONIDA_FIRST_LIGHT_DIR is not set (run tools/first_light.py first)"
            )
        self.out = os.path.abspath(os.path.expanduser(self.out))
        self.harness = harness()
        if not self.harness:
            self.skipTest(
                "vk_harness is not built (runtime/build/linux or runtime/build/windows, target "
                "vk_harness); set LEONIDA_VK_HARNESS"
            )
        with open(os.path.join(self.out, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_draw_matches_the_manifest(self):
        raw = os.path.join(self.out, "image_first_light.raw")
        if os.path.exists(raw):
            os.unlink(
                raw
            )  # a fresh run: a harness that writes nothing cannot pass on a stale file
        env = fl.harness_env(self.manifest, self.out)  # a --camera folder also passes its constants
        subprocess.run(
            [
                self.harness,
                os.path.join(self.out, "vs.spv"),
                os.path.join(self.out, "ps.spv"),
                raw,
                str(self.manifest["width"]),
                str(self.manifest["height"]),
            ],
            check=True,
            env=env,
        )
        self.assertTrue(os.path.isfile(raw), "the harness did not write a fresh image")
        with open(raw, "rb") as f:
            data = f.read()
        stats = check_image(data, self.manifest)
        write_ppm(
            os.path.join(self.out, "image_first_light.ppm"),
            stats["pixels"],
            self.manifest["width"],
            self.manifest["height"],
        )
        self.assertGreater(stats["covered"], 0)

    def test_check_cli_matches_the_oracle(self):
        """The tool's --check must accept and reject the same images the test does."""
        subprocess.run(
            [
                sys.executable,
                "-I",
                os.path.join(ROOT, "tools", "first_light.py"),
                "--check",
                "--out",
                self.out,
                "--harness",
                self.harness,
            ],
            check=True,
        )

    def test_manifest_names_the_assets(self):
        self.assertGreater(self.manifest["vertex_count"], 0)
        self.assertGreater(self.manifest["triangle_count"], 0)
        for key in ("vs", "ps"):
            self.assertTrue(
                self.manifest[key].get("fssm_sha256"), f"{key} has no FSSM hash"
            )
            self.assertTrue(
                os.path.isfile(os.path.join(self.out, self.manifest["files"][key]))
            )


class OracleTests(unittest.TestCase):
    """The readback oracle on synthetic images: the controls run without a GPU or a game package."""

    def setUp(self):
        self.manifest = {
            "width": 4,
            "height": 4,
            "min_coverage": 0.2,
            "max_coverage": 0.5,
            "centre_pixel": [2, 2],
            "centre_uv": [0.25, 0.75],
            "uv_probes": [{"pixel": [2, 2], "uv": [0.25, 0.75]}],
            "clear_pixel": [0, 0],
            "clear": CLEAR,
        }

    def image(self, centre=(0.25, 0.75), fill=None) -> bytes:
        pixels = [[CLEAR] * 4 for _ in range(16)]
        if fill is not None:
            pixels = [[fill, fill, fill, 1.0] for _ in range(16)]
        else:
            for x, y in ((1, 1), (2, 1), (1, 2), (2, 2)):
                pixels[y * 4 + x] = [centre[0], centre[1], 0.0, 1.0]
        return struct.pack("<64f", *(v for pixel in pixels for v in pixel))

    def test_accepts_the_expected_image(self):
        check_image(self.image(), self.manifest)

    def test_rejects_an_all_opaque_image(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(fill=0.5), self.manifest)

    def test_rejects_a_swapped_uv(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(centre=(0.75, 0.25)), self.manifest)

    def test_rejects_a_short_image(self):
        with self.assertRaises(AssertionError):
            check_image(b"", self.manifest)


def vs_on_cpu(sgprs: str, heap: str, x: float, y: float, z: float):
    """CAMERA_VS's arithmetic, reading its constants the way its scalar loads do: the table address from s9, a V#
    at table + 0x20 (world rows at 0x00..0x30) and one at table + 0x60 (view-projection rows at 0x160..0x19f)."""
    s = [int(w, 0) for w in sgprs.split(",")]
    words = {}
    for item in heap.split(","):
        addr, word = item.split("=")
        words[int(addr, 0)] = int(word, 0)

    def dword(addr):
        return words.get(addr, 0)

    def f(addr):
        return struct.unpack("<f", struct.pack("<I", dword(addr)))[0]

    def load(vsharp_addr, offset, count):
        base, size = dword(vsharp_addr), dword(vsharp_addr + 8)
        assert offset + 4 * count <= size, "read past num_records"
        return [f(base + offset + 4 * i) for i in range(count)]

    table = s[fl.CAMERA_TABLE_SGPR]
    w = [load(table + fl.CAMERA_WORLD_VSHARP, 16 * r, 4) for r in range(4)]
    vp = load(table + fl.CAMERA_VIEW_VSHARP, fl.CAMERA_VIEWPROJ, 16)
    world = [w[0][c] * x + w[1][c] * y + w[2][c] * z + w[3][c] for c in range(3)]
    return [vp[c] * world[0] + vp[4 + c] * world[1] + vp[8 + c] * world[2] + vp[12 + c] for c in range(4)]


class CameraTests(unittest.TestCase):
    """--camera on the CPU: no GPU or package. The cube stands in for a model."""

    CUBE = [(x, y, z, 0.5 + x / 4, 0.5 - y / 4) for x in (-2.0, 2.0) for y in (3.0, 7.0) for z in (0.0, 4.0)]

    def test_the_model_is_in_view_and_upright(self):
        world, viewproj = fl.camera_matrices(self.CUBE, 160, 120, 30.0, 25.0, 50.0)
        projected = fl.project(self.CUBE, world, viewproj)
        for x, y, z, _, _, w in projected:
            self.assertGreater(w, 0.0)
            self.assertTrue(0.0 < z < 1.0 and -1.0 < x < 1.0 and -1.0 < y < 1.0)
        centre = fl.project([(0.0, 5.0, 2.0, 0.0, 0.0)], world, viewproj)[0]
        self.assertAlmostEqual(centre[0], 0.0, places=5)  # the camera looks at the bounding-box centre
        self.assertAlmostEqual(centre[1], 0.0, places=5)
        top = fl.project([(0.0, 5.0, 4.0, 0.0, 0.0)], world, viewproj)[0]
        self.assertLess(top[1], centre[1])  # +Z (the game's up) is up the image, and the image's y grows down

    def test_constants_are_where_the_vs_reads_them(self):
        world, viewproj = fl.camera_matrices(self.CUBE, 128, 128, -40.0, 10.0, 60.0)
        sgprs, heap = fl.camera_constants(world, viewproj)
        for x, y, z, *_ in self.CUBE:
            got = vs_on_cpu(sgprs, heap, x, y, z)
            wx, wy, wz, _ = fl.row_times((x, y, z, 1.0), world)
            want = fl.row_times((wx, wy, wz, 1.0), viewproj)
            for g, e in zip(got, want):
                self.assertAlmostEqual(g, e, places=5)

    def test_uv_is_perspective_correct(self):
        # One triangle with clip w 1, 1 and 3: at the screen-space centroid the UV leans towards the near vertices.
        tri = [(-1.0, -1.0, 0.5, 0.0, 0.0, 1.0), (1.0, -1.0, 0.5, 1.0, 0.0, 1.0), (0.0, 1.0, 0.5, 0.0, 1.0, 3.0)]
        affine = [v[:5] for v in tri]
        # pixel (32, 21) of 64 x 64 is at NDC (0.015625, -0.328125), about the screen centroid
        u_aff, v_aff = fl.uv_at(affine, [0, 1, 2], 32, 21, 64, 64)
        u, v = fl.uv_at(tri, [0, 1, 2], 32, 21, 64, 64)
        b = fl.barycentric(*fl.pixel_ndc(32, 21, 64, 64), -1.0, -1.0, 1.0, -1.0, 0.0, 1.0)
        q = (b[0] / 1.0, b[1] / 1.0, b[2] / 3.0)
        self.assertAlmostEqual(u, q[1] / sum(q), places=9)
        self.assertAlmostEqual(v, q[2] / sum(q), places=9)
        self.assertAlmostEqual(v_aff, b[2], places=9)
        self.assertLess(v, v_aff - 0.1)

    def test_a_clipped_vertex_is_an_error(self):
        world, viewproj = fl.camera_matrices(self.CUBE, 128, 128, 0.0, 0.0, 50.0)
        behind = [(0.0, -1000.0, 2.0, 0.0, 0.0)]  # far behind the camera, which looks along +Y from -Y
        with self.assertRaises(SystemExit):
            fl.project(behind, world, viewproj)

    def test_camera_flags_are_checked(self):
        for argv in (
            ["--camera", "--vs", fl.DEFAULT_VS],  # another VS reads other constants
            ["--camera", "--no-fit"],
            ["--camera", "--pitch", "90"],  # straight down: the up axis is the view axis
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    fl.main(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_harness_env_passes_only_the_manifests_constants(self):
        saved = {k: os.environ.get(k) for k in ("VKH_SGPRS", "VKH_HEAP")}
        os.environ.update(VKH_SGPRS="1,2,3", VKH_HEAP="0x0=0x1")
        try:
            plain = fl.harness_env({"camera": None}, "out")
            self.assertNotIn("VKH_SGPRS", plain)
            self.assertNotIn("VKH_HEAP", plain)
            camera = fl.harness_env({"camera": {"sgprs": "0,9", "heap": "0x100=0x1"}}, "out")
            self.assertEqual((camera["VKH_SGPRS"], camera["VKH_HEAP"]), ("0,9", "0x100=0x1"))
            self.assertEqual(camera["VKH_DRAW_MESH"], os.path.join("out", "mesh.bin"))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    sys.exit(unittest.main())
