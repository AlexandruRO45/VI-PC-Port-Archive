"""Progress / ETA helpers used by make_game.py (no game data).

    python -I tests/test_make_game_progress.py
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
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


class InterruptedBuildTest(unittest.TestCase):
    def assert_stage_stops_its_child(self, stage, start_child):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            script = scratch / "rdna2spirv.py"
            script.write_text(
                "from pathlib import Path\n"
                "import time\n"
                "folder = Path(__file__).parent\n"
                "(folder / 'started').touch()\n"
                "deadline = time.monotonic() + 10\n"
                "while not (folder / 'release').exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "(folder / 'finished').touch()\n",
                encoding="utf-8", newline="\n",
            )
            stage_done = threading.Event()

            def build(*args, **kwargs):
                try:
                    start_child(script)
                finally:
                    stage_done.set()

            def interrupt(future, *args, **kwargs):
                deadline = time.monotonic() + 5
                while not (scratch / "started").exists():
                    self.assertLess(time.monotonic(), deadline, "the synthetic stage child did not start")
                    time.sleep(0.01)
                raise KeyboardInterrupt

            with mock.patch.object(make_game, stage, build), mock.patch.object(Future, "result", interrupt), \
                    mock.patch.object(make_game, "TOOLS", str(scratch)), \
                    mock.patch.object(make_game, "translator_digest", return_value="synthetic"), \
                    mock.patch.object(make_game, "log"):
                try:
                    with self.assertRaises(KeyboardInterrupt):
                        make_game.run_build_stages(
                            [stage], game=directory, work=directory, out=directory, v={}, wav=False, jobs=1,
                            runtime_build=None, emulate=False, keep_emulated=False, fallbacks=(),
                        )
                    self.assertTrue(stage_done.wait(4), "the interrupted stage is still waiting on its child")
                    self.assertFalse((scratch / "finished").exists(), "the child ran to completion")
                finally:
                    (scratch / "release").touch()
                    stage_done.wait(5)

    def test_headless_interrupt_stops_an_assets_command(self):
        self.assert_stage_stops_its_child("assets", lambda script: make_game.run([sys.executable, "-I", script]))

    def test_headless_interrupt_stops_a_shader_shard(self):
        def translate(script):
            make_game.translate(str(script.parent / "disasm"), str(script.parent / "spirv"), 1)

        self.assert_stage_stops_its_child("shaders", translate)


if __name__ == "__main__":
    unittest.main()
