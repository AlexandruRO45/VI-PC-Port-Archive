"""Build destinations must stay outside the game folder, including through directory links.

    python -I tests/test_make_game_paths.py

Only empty temporary directories are used; no game files or build tools are needed.
"""
from contextlib import ExitStack
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game


def version_with(*paths, exe=None):
    """A version manifest listing these package paths (sizes and hashes are not read by these checks)."""
    version = {"files": {p: {"size": 0, "sha256": "0" * 64} for p in paths}}
    if exe:
        version["expected_exe"] = exe
    return version


class BuildDestinationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.game = os.path.join(self.root, "game")
        os.mkdir(self.game)
        self.alias = os.path.join(self.root, "game-link")

    def link(self, target, path):
        if os.name == "nt":
            # Junctions work without Developer Mode or symlink privileges on Windows.
            result = subprocess.run(["cmd", "/c", "mklink", "/J", path, target],
                                    capture_output=True, text=True)
            if result.returncode:
                self.skipTest("cannot create a directory junction: " + result.stderr)
            self.addCleanup(os.rmdir, path)
        else:
            os.symlink(target, path, target_is_directory=True)
            self.addCleanup(os.unlink, path)

    def run_build(self, game, out, work, rejected, version=None):
        argv = ["make_game.py", "--game", game, "--out", out, "--work", work,
                "--stages", "runtime", "--no-tui"]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch("make_game.below_normal"))
            stack.enter_context(patch("make_game.check", return_value=version or {}))
            stack.enter_context(patch("make_game.log"))
            preflight = stack.enter_context(patch("make_game.preflight"))
            build_runtime = stack.enter_context(patch("make_game.runtime"))
            stderr = stack.enter_context(patch("sys.stderr", new_callable=io.StringIO))
            if rejected:
                with self.assertRaises(SystemExit) as error:
                    make_game.main()
                self.assertEqual(error.exception.code, 1)
                self.assertIn("--out and --work must be outside the game folder", stderr.getvalue())
                preflight.assert_not_called()
                build_runtime.assert_not_called()
            else:
                self.assertEqual(make_game.main(), 0)
                preflight.assert_called_once()
                build_runtime.assert_called_once()

    def test_destinations_through_game_link_are_rejected(self):
        self.link(self.game, self.alias)
        for option in ("out", "work"):
            for suffix in ("", "new", os.path.join("new", "nested")):
                with self.subTest(option=option, suffix=suffix):
                    destinations = {name: os.path.join(self.root, name) for name in ("out", "work")}
                    destinations[option] = os.path.join(self.alias, suffix)
                    self.run_build(self.game, **destinations, rejected=True)
                    self.assertFalse(os.path.exists(os.path.join(self.game, "new")))

    def test_game_link_cannot_hide_destinations_in_real_game_folder(self):
        self.link(self.game, self.alias)
        for option in ("out", "work"):
            with self.subTest(option=option):
                destinations = {name: os.path.join(self.root, name) for name in ("out", "work")}
                destinations[option] = os.path.join(self.game, "new")
                self.run_build(self.alias, **destinations, rejected=True)

    def test_link_to_unrelated_output_folder_is_allowed(self):
        target = os.path.join(self.root, "output")
        os.mkdir(target)
        alias = os.path.join(self.root, "output-link")
        self.link(target, alias)
        self.run_build(self.game, os.path.join(alias, "new"),
                       os.path.join(self.root, "work"), rejected=False)

    def test_drive_root_game_folder_rules(self):
        if os.name != "nt":
            self.skipTest("Windows drive root check")
        drive = os.path.splitdrive(self.root)[0] + "\\"
        content_dir = os.path.join(drive, "Content")
        independent_dir = os.path.join(drive, "Leonida_Build")
        version = version_with("Content/data.rpf")
        # Allowed: destination is an independent folder on the drive root
        self.run_build(drive, independent_dir, os.path.join(independent_dir, "work"), rejected=False,
                       version=version)
        # Rejected: destination is inside the game dump's own top-level entry
        self.run_build(drive, os.path.join(content_dir, "out"), os.path.join(self.root, "work"), rejected=True,
                       version=version)
        # Rejected: destination is the drive root itself
        self.run_build(drive, drive, os.path.join(self.root, "work"), rejected=True, version=version)

    def test_drive_root_resume_is_not_refused_by_the_build_folder_itself(self):
        # BUILD-2367: the first run creates --out on the game's drive, and a listing of the drive then holds it.
        # The game is what the version manifest lists, so the rerun goes ahead.
        if os.name != "nt":
            self.skipTest("Windows drive root check")
        drive = os.path.splitdrive(self.root)[0] + "\\"
        out = os.path.join(drive, "Leonida_Port")
        version = version_with("Content/data.rpf", "shaders.rpf", exe="game.exe")
        on_drive = ["Content", "shaders.rpf", "game.exe", "Leonida_Port", "Unrelated"]
        with patch("os.listdir", return_value=on_drive):
            self.run_build(drive, out, os.path.join(out, "work"), rejected=False, version=version)
            self.run_build(drive, os.path.join(drive, "Unrelated", "build"), os.path.join(self.root, "work"),
                           rejected=False, version=version)
            for listed in ("Content", "shaders.rpf", "game.exe"):
                with self.subTest(listed=listed):
                    self.run_build(drive, os.path.join(drive, listed, "out"), os.path.join(self.root, "work"),
                                   rejected=True, version=version)

    def test_game_top_level_entries(self):
        self.assertEqual(
            make_game.game_top_level_entries(version_with("a.rpf", "Content/x/y.bin", "Content\\z.bin", exe="g.exe")),
            ["Content", "a.rpf", "g.exe"])
        self.assertEqual(make_game.game_top_level_entries({}), [])

    def test_drive_root_checks_entries_on_each_volume(self):
        if os.name != "nt":
            self.skipTest("Windows drive root check")
        root = os.path.splitdrive(self.root)[0] + "\\"
        other_root = "Z:\\" if root.upper() != "Z:\\" else "Y:\\"
        junction = os.path.join(root, "MountedContent")
        mounted_content = os.path.join(other_root, "Package")

        def realpath(path):
            if os.path.normcase(path) == os.path.normcase(junction):
                return mounted_content
            return os.path.normpath(path)

        # A junction target on a different volume must not end the search before the matching entry.
        for entries, protected in (
            (["MountedContent", "Content"], os.path.join(root, "Content")),
            (["Content", "MountedContent"], mounted_content),
        ):
            version = version_with(*(e + "/file.bin" for e in entries))
            with patch("os.path.realpath", side_effect=realpath):
                for option in ("out", "work"):
                    with self.subTest(protected=protected, option=option):
                        destinations = {name: os.path.join(self.root, name) for name in ("out", "work")}
                        destinations[option] = os.path.join(protected, "build")
                        self.run_build(root, **destinations, rejected=True, version=version)
                self.run_build(root, os.path.join(other_root, "Build"),
                               os.path.join(other_root, "Scratch"), rejected=False, version=version)


if __name__ == "__main__":
    unittest.main()
