"""Exercise conformance PNG input identity and the effective image directories."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"
SPEC = importlib.util.spec_from_file_location("compare_input_names", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)

DISTINCT = ("plain.png", "other.png")
CASE_PAIR = ("pixel.png", "PIXEL.png")


class InputNames(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def manifest(self, folder, names):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "manifest.json"
        scenes = [
            {"name": scene, "png": name, "checks": [{"x": 0, "y": 0, "r": 1, "g": 2, "b": 3, "a": 255}]}
            for scene, name in zip(("first", "second"), names)
        ]
        path.write_text(json.dumps({"renderer": "test", "adapter": "cpu", "scenes": scenes}),
                        encoding="utf-8", newline="\n")
        return path

    def images(self, folder, names):
        folder.mkdir(parents=True, exist_ok=True)
        first, second = [folder / name for name in names]
        raster = compare.make_solid_raster(1, 1, 2, (1, 2, 3), "synthetic")
        compare.write_png(str(first), raster)
        if names == CASE_PAIR:
            if not second.exists():  # Case-sensitive volume: give both spellings the same file identity.
                os.link(first, second)
            self.assertTrue(first.samefile(second))
        else:
            compare.write_png(str(second), raster)
            self.assertFalse(first.samefile(second))

    def cli(self, *args):
        return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *map(str, args)],
                              cwd=self.root, capture_output=True, text=True)

    def assert_alias_error(self, run):
        self.assertEqual(run.returncode, 2, (run.stdout, run.stderr))
        self.assertEqual(run.stdout, "")
        for name in CASE_PAIR:
            self.assertIn(name, run.stderr)
        self.assertNotIn("Traceback", run.stderr)

    def test_expect_rejects_alias_in_effective_directory(self):
        for override in (False, True):
            with self.subTest(override=override):
                manifest = self.manifest(self.root / str(override), CASE_PAIR)
                images = manifest.parent / "images" if override else manifest.parent
                self.images(images, CASE_PAIR)
                args = ["--dir", images] if override else []
                self.assert_alias_error(self.cli("expect", "--manifest", manifest, *args))

    def test_diff_validates_both_sides_before_writing_heatmaps(self):
        for side in ("reference", "candidate"):
            for override in (False, True):
                with self.subTest(side=side, override=override):
                    case = self.root / side / str(override)
                    args = ["diff"]
                    for role in ("reference", "candidate"):
                        names = CASE_PAIR if role == side else DISTINCT
                        manifest = self.manifest(case / role, names)
                        images = manifest.parent / "images" if override else manifest.parent
                        self.images(images, names)
                        args.extend(["--" + role, manifest])
                        if role == "candidate":
                            destination = images
                        if override:
                            args.extend(["--" + role + "-dir", images])
                    sentinel = destination / "first.diff.png"
                    sentinel.write_bytes(b"existing output")
                    self.assert_alias_error(self.cli(*args))
                    self.assertEqual(sentinel.read_bytes(), b"existing output")
                    self.assertEqual(list(case.rglob("*.diff.png")), [sentinel])

    def test_distinct_missing_and_exact_duplicate(self):
        reference = self.manifest(self.root / "reference", DISTINCT)
        candidate = self.manifest(self.root / "candidate", DISTINCT)
        for directory in (reference.parent, candidate.parent):
            self.images(directory, DISTINCT)
        run = self.cli("diff", "--reference", reference, "--candidate", candidate)
        self.assertEqual(run.returncode, 0, run.stderr)
        for name in ("first", "second"):
            png = candidate.parent / (name + ".diff.png")
            self.assertTrue(png.read_bytes().startswith(compare.PNG_SIGNATURE))
        self.assertEqual(self.cli("expect", "--manifest", reference).returncode, 0)
        reference = self.manifest(reference.parent, ("plain.png", "missing.png"))
        self.assertEqual(self.cli("expect", "--manifest", reference).returncode, 1)
        reference = self.manifest(reference.parent, CASE_PAIR)  # Both spellings are absent.
        self.assertEqual(self.cli("expect", "--manifest", reference).returncode, 1)
        reference = self.manifest(reference.parent, ("plain.png", "plain.png"))
        self.assertEqual(self.cli("expect", "--manifest", reference).returncode, 2)

    def test_case_sensitive_lookup_accepts_two_distinct_files(self):
        manifest = self.manifest(self.root, CASE_PAIR)
        self.images(self.root, DISTINCT)
        paths = dict(zip(CASE_PAIR, [self.root / name for name in DISTINCT]))
        # Model case-sensitive name resolution only. samefile and PNG reads use real files.
        with patch.object(compare, "scene_image_path", side_effect=lambda directory, name: str(paths[name])):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(["expect", "--manifest", str(manifest)]), 0)


if __name__ == "__main__":
    unittest.main()
