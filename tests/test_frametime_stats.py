"""Tests for tools/frametime_stats.py."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import frametime_stats  # noqa: E402


class FrametimeStatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_csv(self, filename: str, header: str, values: list[float] | list[str]) -> str:
        path = os.path.join(self.tmp_dir.name, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{header}\n")
            for v in values:
                f.write(f"{v}\n")
        return path

    def test_percentile_calculation(self):
        data = [10.0, 10.0, 10.0, 10.0, 20.0]
        self.assertEqual(frametime_stats.percentile(data, 0), 10.0)
        self.assertEqual(frametime_stats.percentile(data, 100), 20.0)
        self.assertEqual(frametime_stats.percentile(data, 50), 10.0)

    def test_mangohud_ns_parsing(self):
        # 16.666 ms = 16,666,666 ns (~60 FPS)
        ns_values = [16_666_666.0] * 60 + [33_333_333.0]  # 1 dropped frame
        csv_path = self._create_csv("mangohud.csv", "frametime_ns", ns_values)

        parsed = frametime_stats.parse_frametimes_ms(csv_path)
        self.assertEqual(len(parsed), 61)
        self.assertAlmostEqual(parsed[0], 16.666, places=2)
        self.assertAlmostEqual(parsed[-1], 33.333, places=2)

        metrics = frametime_stats.compute_metrics(parsed, stutter_threshold_ms=25.0)
        self.assertEqual(metrics["count"], 61)
        self.assertAlmostEqual(metrics["avg_fps"], 59.03, places=1)
        self.assertEqual(metrics["stutter_count"], 1)

    def test_generic_ms_parsing(self):
        ms_values = [16.6] * 100
        csv_path = self._create_csv("generic.csv", "frametime_ms", ms_values)

        parsed = frametime_stats.parse_frametimes_ms(csv_path)
        self.assertEqual(len(parsed), 100)
        metrics = frametime_stats.compute_metrics(parsed)
        self.assertAlmostEqual(metrics["p50_ms"], 16.6, places=1)
        self.assertEqual(metrics["stutter_count"], 0)

    def test_presentmon_legacy_columns_and_priority(self):
        # PresentMon legacy format: Application,ProcessID,MsBetweenPresents
        lines = ["fixture.exe,4321,16", "fixture.exe,4321,24"]
        csv_path = self._create_csv("presentmon.csv", "Application,ProcessID,MsBetweenPresents", lines)

        parsed = frametime_stats.parse_frametimes_ms(csv_path)
        self.assertEqual(parsed, [16.0, 24.0])
        metrics = frametime_stats.compute_metrics(parsed)
        self.assertEqual(metrics["avg_fps"], 50.0)
        self.assertEqual(metrics["p50_ms"], 20.0)

    def test_frametime_priority_over_fps_column(self):
        # Header has both fps and frametime_ms; measured frametime must win regardless of column order
        lines = ["60,16", "60,24"]
        csv_path = self._create_csv("mixed.csv", "fps,frametime_ms", lines)

        parsed = frametime_stats.parse_frametimes_ms(csv_path)
        self.assertEqual(parsed, [16.0, 24.0])
        metrics = frametime_stats.compute_metrics(parsed)
        self.assertEqual(metrics["avg_fps"], 50.0)

    def test_explicit_ms_magnitude_not_scaled(self):
        # When column is explicitly frametime_ms, values >= 100,000 are not scaled down
        csv_path = self._create_csv("large_ms.csv", "frametime_ms", [150000.0])
        parsed = frametime_stats.parse_frametimes_ms(csv_path)
        self.assertEqual(parsed, [150000.0])

    def test_target_fps_hit_pct(self):
        # 16.6 ms (60 FPS) and 33.3 ms (30 FPS)
        values = [16.6] * 80 + [33.3] * 20
        csv_path = self._create_csv("target.csv", "frametime_ms", values)
        parsed = frametime_stats.parse_frametimes_ms(csv_path)
        metrics = frametime_stats.compute_metrics(parsed, target_fps=60.0)
        self.assertEqual(metrics["target_fps"], 60.0)
        self.assertAlmostEqual(metrics["target_hit_pct"], 80.0, places=1)

    def test_cli_execution_json(self):
        ms_values = [16.6] * 50 + [40.0]
        csv_path = self._create_csv("run.csv", "ms", ms_values)

        res = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "frametime_stats.py"), csv_path, "--json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        self.assertEqual(data["count"], 51)
        self.assertIn("avg_fps", data)
        self.assertIn("one_percent_low_fps", data)
        self.assertEqual(data["stutter_count"], 1)

    def test_cli_compare_mode(self):
        base_csv = self._create_csv("base.csv", "frametime_ms", [20.0] * 50)  # 50 FPS
        tuned_csv = self._create_csv("tuned.csv", "frametime_ms", [16.6] * 50)  # 60 FPS

        res = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "frametime_stats.py"),
             "--compare", base_csv, tuned_csv],
            capture_output=True, text=True, check=True
        )
        self.assertIn("Baseline [A]:", res.stdout)
        self.assertIn("Tuned    [B]:", res.stdout)
        self.assertIn("Average FPS", res.stdout)
        self.assertIn("+20.", res.stdout)  # approx +20% FPS improvement


if __name__ == "__main__":
    unittest.main()
