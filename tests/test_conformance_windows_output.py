"""Exercise compare.py with stdout redirected to a Windows code page."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


COMPARE = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"
PNG = runpy.run_path(str(COMPARE))


def write_pixel(path, rgb):
    raster = PNG["make_solid_raster"](1, 1, 2, rgb, "synthetic")
    PNG["write_png"](str(path), raster)


@unittest.skipUnless(sys.platform == "win32", "Windows redirected stdout")
class RedirectedOutput(unittest.TestCase):
    def run_python(self, encoding, *args):
        env = dict(os.environ, PYTHONIOENCODING=encoding + ":strict", PYTHONPATH="")
        return subprocess.run([sys.executable, "-B", "-S", *args], cwd=self.root, env=env, capture_output=True)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_expect_diff_and_inspect(self):
        scene = "caf\u00e9_\u6e2c"
        manifest = {
            "renderer": "test", "adapter": "cpu",
            "scenes": [{"name": scene, "png": "pixel.png", "checks": [
                {"x": 0, "y": 0, "r": 1, "g": 2, "b": 3, "a": 255},
            ]}],
        }
        ref, candidate = (self.root / scene / side for side in ("reference", "candidate"))
        for folder in (ref, candidate):
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
            write_pixel(folder / "pixel.png", (1, 2, 3))
        diff = ["diff", "--reference", str(ref / "manifest.json"), "--candidate", str(candidate / "manifest.json"),
                "--max-differing-percent", "0", "--max-mean-delta", "0"]
        heatmap = candidate / (scene + ".diff.png")
        cases = [
            ("expect", ["expect", "--manifest", str(ref / "manifest.json")], 0),
            ("inspect", ["inspect", "--image", str(ref / "pixel.png")], 0),
            ("match", diff, 0),
            ("mismatch", diff, 1),
        ]
        for label, args, expected_rc in cases:
            with self.subTest(command=label):
                if label == "mismatch":
                    write_pixel(candidate / "pixel.png", (4, 2, 3))
                control = self.run_python("utf-8", str(COMPARE), *args)
                self.assertEqual(control.returncode, expected_rc, control.stderr)
                self.assertIn(scene.encode("utf-8"), control.stdout)
                if args[0] == "diff":
                    expected_heatmap = heatmap.read_bytes()
                    heatmap.unlink()
                run = self.run_python("cp1252", str(COMPARE), *args)
                self.assertEqual(run.returncode, expected_rc, run.stderr)
                self.assertEqual(run.stderr, b"")
                self.assertEqual(run.stdout, control.stdout.decode("utf-8").encode("cp1252", "backslashreplace"))
                if args[0] == "diff":
                    self.assertEqual(heatmap.read_bytes(), expected_heatmap)
                    heatmap.unlink()

    def test_missing_image_keeps_error_status(self):
        run = self.run_python("cp1252", str(COMPARE), "inspect", "--image", "missing_\u6e2c.png")
        self.assertEqual(run.returncode, 2)
        self.assertIn(b"missing_", run.stderr)
        self.assertNotIn(b"Traceback", run.stderr)

    def test_import_preserves_stdout(self):
        code = (
            "import runpy, sys\n"
            "before = (sys.stdout.encoding, sys.stdout.errors)\n"
            "runpy.run_path(sys.argv[1])\n"
            "assert (sys.stdout.encoding, sys.stdout.errors) == before\n"
        )
        run = self.run_python("cp1252", "-c", code, str(COMPARE))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, b"")


if __name__ == "__main__":
    unittest.main()
