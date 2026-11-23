"""Conformance diff must preserve its PNG and manifest inputs. Synthetic, CPU only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "conformance" / "compare.py"
SPEC = importlib.util.spec_from_file_location("conformance_heatmap_checker", CHECKER)
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)


class HeatmapPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()

    def image(self, directory, name, colour=(10, 20, 30)):
        path = directory / name
        C.write_png(str(path), C.make_solid_raster(
            1, 1, C.PNG_COLOUR_TYPE_RGB, colour, "synthetic"))
        return path

    def manifest(self, directory, scenes, name="manifest.json"):
        path = directory / name
        document = {
            "renderer": "synthetic", "adapter": "CPU",
            "scenes": [{"name": scene, "png": png, "checks": [
                {"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255}
            ]} for scene, png in scenes],
        }
        path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
        return path

    def run_diff(self, reference, candidate, *options):
        return subprocess.run(
            [sys.executable, "-I", str(CHECKER), "diff",
             "--reference", str(reference), "--candidate", str(candidate), *options],
            capture_output=True, text=True, timeout=30)

    def assert_refused_without_writes(self, reference, candidate, *options, message="would overwrite input"):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.run_diff(reference, candidate, *options)
        after = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(after, before, "diff changed files before refusing the collision")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_candidate_image_is_its_own_heatmap_path(self):
        self.image(self.reference, "ref.png")
        self.image(self.candidate, "clear.diff.png")
        reference = self.manifest(self.reference, [("clear", "ref.png")])
        candidate = self.manifest(self.candidate, [("clear", "clear.diff.png")])
        self.assert_refused_without_writes(reference, candidate)

    def test_later_scene_input_is_protected_before_any_heatmap_write(self):
        scenes = [("safe", "safe.png"), ("clear", "clear.png"),
                  ("later", "clear.diff.png")]
        for directory in (self.reference, self.candidate):
            for _, png in scenes:
                self.image(directory, png)
        reference = self.manifest(self.reference, scenes)
        candidate = self.manifest(self.candidate, scenes)
        previous_heatmap = self.image(self.candidate, "safe.diff.png", (70, 80, 90))
        self.assert_refused_without_writes(reference, candidate)
        self.assertEqual(C.read_png(str(previous_heatmap)).stored_pixel(0, 0), (70, 80, 90))

    def test_reference_input_in_shared_override_directory_is_protected(self):
        self.image(self.candidate, "clear.diff.png")
        self.image(self.candidate, "candidate.png")
        reference = self.manifest(self.reference, [("clear", "clear.diff.png")])
        candidate = self.manifest(self.candidate, [("clear", "candidate.png")])
        # The override contains a parent component and shares the candidate folder.
        override = self.candidate / ".." / "candidate"
        self.assert_refused_without_writes(
            reference, candidate, "--reference-dir", str(override))

    def test_candidate_override_directory_is_checked(self):
        images = self.root / "candidate-images"
        images.mkdir()
        self.image(self.reference, "ref.png")
        self.image(images, "clear.diff.png")
        reference = self.manifest(self.reference, [("clear", "ref.png")])
        candidate = self.manifest(self.candidate, [("clear", "clear.diff.png")])
        self.assert_refused_without_writes(
            reference, candidate, "--candidate-dir", str(images))

    def test_both_manifests_are_protected(self):
        for role in ("reference", "candidate"):
            with self.subTest(role=role):
                self.image(self.reference, "ref.png")
                self.image(self.candidate, "candidate.png")
                reference = self.manifest(self.reference, [("clear", "ref.png")])
                candidate = self.manifest(self.candidate, [("clear", "candidate.png")])
                output = self.candidate / "clear.diff.png"
                source = reference if role == "reference" else candidate
                output.write_bytes(source.read_bytes())
                if role == "reference":
                    reference = output
                else:
                    candidate = output
                self.assert_refused_without_writes(
                    reference, candidate, "--reference-dir", str(self.reference))

    def test_hard_link_to_reference_image_is_protected(self):
        source = self.image(self.reference, "ref.png")
        self.image(self.candidate, "candidate.png")
        reference = self.manifest(self.reference, [("clear", "ref.png")])
        candidate = self.manifest(self.candidate, [("clear", "candidate.png")])
        try:
            os.link(source, self.candidate / "clear.diff.png")
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        self.assert_refused_without_writes(reference, candidate)

    def test_symlink_to_candidate_image_is_protected(self):
        self.image(self.reference, "ref.png")
        source = self.image(self.candidate, "candidate.png")
        reference = self.manifest(self.reference, [("clear", "ref.png")])
        candidate = self.manifest(self.candidate, [("clear", "candidate.png")])
        try:
            (self.candidate / "clear.diff.png").symlink_to(source)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        self.assert_refused_without_writes(reference, candidate)

    @unittest.skipUnless(os.name == "nt", "Windows case-insensitive path spelling")
    def test_case_alias_of_candidate_image_is_protected(self):
        self.image(self.reference, "ref.png")
        self.image(self.candidate, "CLEAR.DIFF.PNG")
        reference = self.manifest(self.reference, [("clear", "ref.png")])
        candidate = self.manifest(self.candidate, [("clear", "CLEAR.DIFF.PNG")])
        self.assert_refused_without_writes(reference, candidate)

    @unittest.skipUnless(os.name == "nt", "Windows case-insensitive path spelling")
    def test_case_only_scene_names_cannot_share_a_heatmap_path(self):
        scenes = [("Clear", "first.png"), ("clear", "second.png")]
        for directory in (self.reference, self.candidate):
            for _, png in scenes:
                self.image(directory, png)
        reference = self.manifest(self.reference, scenes)
        candidate = self.manifest(self.candidate, scenes)
        # Neither planned output exists yet: the check must recognise the case alias by name.
        self.assert_refused_without_writes(
            reference, candidate, message="would alias another heatmap")

    def test_existing_heatmaps_cannot_alias_each_other_through_a_hard_link(self):
        scenes = [("first", "first.png"), ("second", "second.png")]
        for directory in (self.reference, self.candidate):
            for _, png in scenes:
                self.image(directory, png)
        reference = self.manifest(self.reference, scenes)
        candidate = self.manifest(self.candidate, scenes)
        first = self.image(self.candidate, "first.diff.png", (70, 80, 90))
        try:
            os.link(first, self.candidate / "second.diff.png")
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        self.assert_refused_without_writes(
            reference, candidate, message="would alias another heatmap")

    def test_distinct_scenes_keep_both_heatmaps(self):
        scenes = [("first", "first.png"), ("second", "second.png")]
        for directory in (self.reference, self.candidate):
            for _, png in scenes:
                self.image(directory, png)
        self.image(self.candidate, "second.png", (255, 20, 30))
        reference = self.manifest(self.reference, scenes)
        candidate = self.manifest(self.candidate, scenes)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.run_diff(reference, candidate)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        first = C.read_png(str(self.candidate / "first.diff.png")).stored_pixel(0, 0)
        second = C.read_png(str(self.candidate / "second.diff.png")).stored_pixel(0, 0)
        self.assertEqual(first, (48, 48, 48))
        self.assertNotEqual(second, first)

    def test_normal_diff_preserves_inputs_and_replaces_existing_heatmap(self):
        ref_image = self.image(self.reference, "ref.png")
        candidate_image = self.image(self.candidate, "candidate.png")
        reference = self.manifest(self.reference, [("clear", "ref.png")])
        candidate = self.manifest(self.candidate, [("clear", "candidate.png")])
        heatmap = self.candidate / "clear.diff.png"
        heatmap.write_bytes(b"old heatmap")
        inputs = (ref_image, candidate_image, reference, candidate)
        before = [path.read_bytes() for path in inputs]
        result = self.run_diff(reference, candidate)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([path.read_bytes() for path in inputs], before)
        self.assertEqual(C.read_png(str(heatmap)).stored_pixel(0, 0), (48, 48, 48))
        first_heatmap = heatmap.read_bytes()
        self.assertEqual(self.run_diff(reference, candidate).returncode, 0)
        self.assertEqual(heatmap.read_bytes(), first_heatmap)
        self.image(self.candidate, "candidate.png", (255, 20, 30))
        before = [path.read_bytes() for path in inputs]
        result = self.run_diff(reference, candidate)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual([path.read_bytes() for path in inputs], before)
        self.assertNotEqual(heatmap.read_bytes(), first_heatmap)


if __name__ == "__main__":
    unittest.main()
