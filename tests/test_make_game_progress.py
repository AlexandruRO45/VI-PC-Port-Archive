"""Progress / ETA helpers used by make_game.py (no game data).

    python -I tests/test_make_game_progress.py
"""
import os
import sys
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


class FmtDurationTest(unittest.TestCase):
    def test_seconds_minutes_hours(self):
        self.assertEqual(make_game.fmt_duration(0), "0s")
        self.assertEqual(make_game.fmt_duration(9.4), "9s")
        self.assertEqual(make_game.fmt_duration(60), "1m")
        self.assertEqual(make_game.fmt_duration(75), "1m 15s")
        self.assertEqual(make_game.fmt_duration(3600), "1h 00m")
        self.assertEqual(make_game.fmt_duration(3725), "1h 02m")


class ProgressTest(unittest.TestCase):
    def test_logs_percent_and_eta(self):
        lines = []
        # __init__ reads monotonic for t0 and the first emit; each later emit reads once more.
        with mock.patch.object(make_game, "log", side_effect=lines.append), \
                mock.patch.object(make_game.time, "monotonic", side_effect=[100.0, 100.0, 120.0, 160.0]):
            p = make_game.Progress("step", 4, every_s=0.0)
            p.tick(force=True)
            p.set(4, force=True)
        self.assertIn("step: 0/4 (0%)", lines[0])
        self.assertTrue(any("1/4 (25%)" in line and "left" in line for line in lines), lines)
        self.assertTrue(any("4/4 (100%)" in line and "elapsed" in line for line in lines), lines)

    def test_empty_total(self):
        lines = []
        with mock.patch.object(make_game, "log", side_effect=lines.append):
            make_game.Progress("empty", 0)
        self.assertEqual(lines, ["empty: nothing to do"])


class StageBoardTest(unittest.TestCase):
    def test_weighted_percent_and_remaining(self):
        lines = []
        with mock.patch.object(make_game, "log", side_effect=lines.append), \
                mock.patch.object(make_game.time, "monotonic", side_effect=[0.0, 10.0, 20.0]):
            board = make_game.StageBoard(["shaders", "assets", "runtime"])
            board.complete("runtime")
            board.complete("shaders")
        self.assertIn("build: 0%", lines[0])
        # runtime is weight 2 of 40+45+2
        self.assertTrue(any("runtime done" in line and "still" in line for line in lines))
        self.assertTrue(any("shaders done" in line for line in lines))

    def test_complete_is_idempotent(self):
        lines = []
        with mock.patch.object(make_game, "log", side_effect=lines.append), \
                mock.patch.object(make_game.time, "monotonic", return_value=1.0):
            board = make_game.StageBoard(["runtime"])
            board.complete("runtime")
            before = len(lines)
            board.complete("runtime")
        self.assertEqual(len(lines), before)


class ParallelStageProgressTest(unittest.TestCase):
    def build(self, stages):
        make_game.run_build_stages(
            stages, game="game", work="work", out="out", v={}, wav=False, jobs=4,
            runtime_build=None, emulate=False, keep_emulated=False, fallbacks=(),
        )

    def test_assets_completion_is_reported_while_shaders_run(self):
        for stages in (["shaders", "assets"], ["shaders", "assets", "runtime", "pipelines"]):
            with self.subTest(stages=stages):
                assets_reported = threading.Event()
                shaders_saw_assets_report = []

                def shaders(*args, **kwargs):
                    shaders_saw_assets_report.append(assets_reported.wait(timeout=2))

                def log(message):
                    if "assets done" in message:
                        assets_reported.set()

                with (mock.patch.object(make_game, "shaders", side_effect=shaders),
                      mock.patch.object(make_game, "assets"),
                      mock.patch.object(make_game, "runtime"),
                      mock.patch.object(make_game, "pipelines"),
                      mock.patch.object(make_game, "log", side_effect=log)):
                    self.build(stages)

                self.assertEqual(shaders_saw_assets_report, [True])

    def test_pipelines_completion_is_reported_while_assets_run(self):
        pipelines_reported = threading.Event()
        assets_saw_pipeline_report = []

        def assets(*args, **kwargs):
            assets_saw_pipeline_report.append(pipelines_reported.wait(timeout=2))

        def log(message):
            if "pipelines done" in message:
                pipelines_reported.set()

        with (mock.patch.object(make_game, "shaders"),
              mock.patch.object(make_game, "assets", side_effect=assets),
              mock.patch.object(make_game, "runtime"),
              mock.patch.object(make_game, "pipelines"),
              mock.patch.object(make_game, "log", side_effect=log)):
            self.build(["shaders", "assets", "runtime", "pipelines"])

        self.assertEqual(assets_saw_pipeline_report, [True])

    def test_stage_failures_drain_siblings_and_only_gate_dependent_pipelines(self):
        for failed in ("shaders", "assets", "runtime"):
            with self.subTest(failed=failed):
                lines = []
                with (mock.patch.object(make_game, "shaders") as shaders,
                      mock.patch.object(make_game, "assets") as assets,
                      mock.patch.object(make_game, "runtime") as runtime,
                      mock.patch.object(make_game, "pipelines") as pipelines,
                      mock.patch.object(make_game, "log", side_effect=lines.append),
                      self.assertRaises(SystemExit) as stopped):
                    builders = {"shaders": shaders, "assets": assets, "runtime": runtime}
                    builders[failed].side_effect = SystemExit(1)
                    self.build(["shaders", "assets", "runtime", "pipelines"])

                self.assertEqual(stopped.exception.code, 1)
                self.assertEqual(pipelines.call_count, 1 if failed == "assets" else 0)
                for name in builders:
                    self.assertEqual(sum(f"{name} done" in line for line in lines),
                                     0 if name == failed else 1)

    def test_pipelines_only_starts_without_selected_prerequisites(self):
        with (mock.patch.object(make_game, "pipelines") as pipelines,
              mock.patch.object(make_game, "log")):
            self.build(["pipelines"])
        pipelines.assert_called_once()


if __name__ == "__main__":
    unittest.main()
