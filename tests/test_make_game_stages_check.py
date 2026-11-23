"""A --stages list without check identifies the version from the probe files only, so a package whose large archives
are still downloading can build the runtime stage. The default stages, an explicit check, --check and --full-check
still check every file.

    python -I tests/test_make_game_stages_check.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game

MANIFEST = {
    "title": "test package",
    "version": "9.9",
    "expected_exe": "game.exe",
    "files": {
        "MicrosoftGame.Config": {"size": 4},
        "big.rpf": {"size": 64},
        "game.exe": {"size": 2},
    },
}


class Stop(Exception):
    pass


class CheckSizesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        versions = os.path.join(tmp.name, "versions")
        os.makedirs(versions)
        with open(os.path.join(versions, "xbox-9.9.json"), "w", encoding="utf-8") as f:
            json.dump(MANIFEST, f)
        self.game = os.path.join(tmp.name, "game")
        os.makedirs(self.game)
        with open(os.path.join(self.game, "MicrosoftGame.Config"), "wb") as f:
            f.write(b"<x/>")
        patcher = patch.object(make_game, "VERSIONS", versions)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_check(self, **kw):
        logs, out, err = [], io.StringIO(), io.StringIO()
        with patch.object(make_game, "log", logs.append), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            v = make_game.check(self.game, False, **kw)
        return v, logs, out.getvalue()

    def test_default_fails_on_a_partial_package(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_check()
        self.assertEqual(raised.exception.code, 1)

    def test_without_sizes_a_partial_package_passes(self):
        v, logs, out = self.run_check(sizes=False)
        self.assertEqual(v["version"], "9.9")
        self.assertNotIn("missing", out)
        self.assertIn("game.exe is not in the folder", logs[-1])

    def test_without_sizes_reports_the_executable(self):
        with open(os.path.join(self.game, "game.exe"), "wb") as f:
            f.write(b"MZ")
        _, logs, _ = self.run_check(sizes=False)
        self.assertIn("Executable found.", logs[-1])

    def test_without_sizes_still_needs_the_probes(self):
        os.remove(os.path.join(self.game, "MicrosoftGame.Config"))
        with self.assertRaises(SystemExit):
            self.run_check(sizes=False)


class MainStagesTests(unittest.TestCase):
    def sizes_for(self, *argv):
        """The sizes= main() hands to check() for the given arguments."""
        seen = []

        def fake_check(game, full, sizes=True):
            seen.append(sizes)
            raise Stop

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(sys, "argv", ["make_game.py", "--game", os.path.join(tmp, "game"),
                                        "--out", os.path.join(tmp, "out"), *argv]), \
             patch.object(make_game, "below_normal"), \
             patch.object(make_game, "check", fake_check):
            with self.assertRaises(Stop):
                make_game.main()
        return seen[0]

    def test_default_stages_check_sizes(self):
        self.assertTrue(self.sizes_for())

    def test_stages_with_check_check_sizes(self):
        self.assertTrue(self.sizes_for("--stages", "check,runtime"))

    def test_stages_without_check_skip_sizes(self):
        self.assertFalse(self.sizes_for("--stages", "runtime"))
        self.assertFalse(self.sizes_for("--stages", "shaders,runtime"))

    def test_check_flags_override_the_stage_list(self):
        self.assertTrue(self.sizes_for("--stages", "runtime", "--check"))
        self.assertTrue(self.sizes_for("--stages", "runtime", "--full-check"))


if __name__ == "__main__":
    unittest.main()
