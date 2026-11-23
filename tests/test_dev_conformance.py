"""Check developer CLI conformance discovery and parity with the PNG comparator."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "scripts" / "dev.py"
COMPARE_PATH = "tools/conformance/compare.py"
COMPARE = ROOT / COMPARE_PATH


class ConformanceEntry(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cwd = self.root / "external caller"
        self.cwd.mkdir()

    def invoke(self, *argv):
        return subprocess.run([sys.executable, "-I", "-B", *argv], cwd=self.cwd, capture_output=True, timeout=30)

    def test_discovery_and_missing_producer(self):
        checkout = self.root / "fixture checkout"
        cli = checkout / "scripts/dev.py"
        cli.parent.mkdir(parents=True)
        shutil.copyfile(DEV, cli)
        script = checkout / COMPARE_PATH
        script.parent.mkdir(parents=True)
        # A namesake in the caller's directory must not satisfy the inventory or dispatch.
        decoy = self.cwd / COMPARE_PATH
        decoy.parent.mkdir(parents=True)
        decoy.touch()
        for present in (False, True):
            with self.subTest(present=present):
                if present:
                    script.write_text("from pathlib import Path\nPath(__file__).with_suffix('.ran').touch()\n",
                                      encoding="utf-8", newline="\n")
                result = self.invoke(cli, "discover", "--json")
                self.assertEqual(result.returncode, 0, result.stderr)
                tools = {row["id"]: row for row in json.loads(result.stdout)["tools"]}
                self.assertIn("conformance", tools)
                entry = tools["conformance"]
                self.assertIs(entry["present"], present)
                self.assertEqual(entry["path"], COMPARE_PATH)
                self.assertTrue(entry["requirements"])
                expected_help = [sys.executable, "-I", COMPARE_PATH, "--help"] if present else None
                self.assertEqual(entry["help_argv"], expected_help)
                self.assertFalse(script.with_suffix(".ran").exists())
        script.unlink()
        missing = self.invoke(cli, "conformance", "--help")
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(missing.stdout, b"")
        self.assertIn(COMPARE_PATH.encode(), missing.stderr)
        self.assertIn(b"checkout", missing.stderr)

    def test_help_and_producer_error(self):
        help_result = self.invoke(DEV, "--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(b"conformance", help_result.stdout)
        for args, status in [(("--help",), 0), (("diff", "--help"), 0),
                             (("inspect", "--image", "missing file.png"), 2)]:
            with self.subTest(args=args):
                direct = self.invoke(COMPARE, *args)
                routed = self.invoke(DEV, "conformance", *args)
                self.assertEqual(direct.returncode, status, direct.stderr)
                self.assertEqual((routed.returncode, routed.stdout, routed.stderr),
                                 (status, direct.stdout, direct.stderr))
                if status == 2:
                    self.assertIn(b"missing file.png", routed.stderr)

    def test_diff_matches_status_streams_and_heatmap(self):
        png = runpy.run_path(str(COMPARE))
        manifest = {"renderer": "test", "adapter": "cpu", "scenes": [
            {"name": "sample", "png": "pixel.png", "checks": [{"x": 0, "y": 0, "r": 1, "g": 2, "b": 3, "a": 255}]}
        ]}
        reference, candidate = Path("reference images"), Path("candidate images")
        for folder in (reference, candidate):
            (self.cwd / folder).mkdir()
            (self.cwd / folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")

        def write_pixel(folder, rgb):
            raster = png["make_solid_raster"](1, 1, 2, rgb, "synthetic")
            png["write_png"](str(self.cwd / folder / "pixel.png"), raster)

        write_pixel(reference, (1, 2, 3))
        args = ("diff", "--reference", reference / "manifest.json", "--candidate", candidate / "manifest.json",
                "--max-differing-percent", "0", "--max-mean-delta", "0")
        heatmap = self.cwd / candidate / "sample.diff.png"
        for status, rgb in [(0, (1, 2, 3)), (1, (4, 2, 3))]:
            with self.subTest(status=status):
                write_pixel(candidate, rgb)
                direct = self.invoke(COMPARE, *args)
                self.assertEqual(direct.returncode, status, direct.stderr)
                expected = heatmap.read_bytes()
                self.assertTrue(expected.startswith(b"\x89PNG\r\n\x1a\n"))
                heatmap.unlink()
                routed = self.invoke(DEV, "conformance", *args)
                self.assertEqual((routed.returncode, routed.stdout, routed.stderr),
                                 (status, direct.stdout, direct.stderr))
                self.assertEqual(heatmap.read_bytes(), expected)
                heatmap.unlink()


if __name__ == "__main__":
    unittest.main()
