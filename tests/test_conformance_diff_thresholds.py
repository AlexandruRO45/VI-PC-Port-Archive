"""Check the CLI's threshold explanations using a single changed alpha sample."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"
SPEC = importlib.util.spec_from_file_location("conformance_compare_thresholds", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class DiffThresholdTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for side, colour_type, pixel in (("reference", 2, (1, 2, 3)), ("candidate", 6, (1, 2, 3, 255))):
            folder = self.root / side
            folder.mkdir()
            image = compare.make_solid_raster(2, 1, colour_type, pixel, side)
            if side == "candidate":
                image.pixels[3] = 254
            compare.write_png(str(folder / "scene.png"), image)
            compare.write_selftest_manifest(str(folder / "manifest.json"), side, "synthetic", [{
                "name": "scene", "png": "scene.png",
                "checks": [{"x": 1, "y": 0, "r": 1, "g": 2, "b": 3, "a": 255}],
            }])

    def diff(self, percent, mean, expected_status):
        result = subprocess.run(
            [sys.executable, "-I", "-B", SCRIPT, "diff", "--reference", "reference/manifest.json",
             "--candidate", "candidate/manifest.json", "--max-differing-percent", percent,
             "--max-mean-delta", mean],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
        self.assertIn("differing 1/2 (50.000%)  mean 0.1250  max 1", result.stdout)
        return result.stdout

    def test_exact_limits_pass(self):
        self.assertIn("PASS  scene", self.diff("50", "0.125", 0))

    def test_close_limits_remain_distinguishable_in_the_failure_reason(self):
        for percent, mean, reason, expected in (
            ("49.99999", "0.125", r"differing pixels ([\d.]+)% exceed --max-differing-percent ([\d.]+)%",
             (50.0, 49.99999)),
            ("50", "0.124999", r"mean absolute channel delta ([\d.]+) exceeds --max-mean-delta ([\d.]+)",
             (0.125, 0.124999)),
        ):
            with self.subTest(percent=percent, mean=mean):
                output = self.diff(percent, mean, 1)
                match = re.search(reason, output)
                self.assertIsNotNone(match, output)
                self.assertEqual(tuple(map(float, match.groups())), expected, match.group())

    def test_unambiguous_limit_keeps_the_usual_precision(self):
        output = self.diff("49", "0.125", 1)
        self.assertIn("differing pixels 50.000% exceed --max-differing-percent 49.000%", output)


if __name__ == "__main__":
    unittest.main()
