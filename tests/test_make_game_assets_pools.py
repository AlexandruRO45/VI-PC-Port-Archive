"""Texture pools in make_game.assets() must not collide on their basename (no game files).

    python -I tests/test_make_game_assets_pools.py
"""
import json
import os
import sys
import tempfile
import io
import contextlib
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_game  # noqa: E402
import ttpf_assets  # noqa: E402


def deny_listing(blocked):
    """Patch os.scandir so listing `blocked` raises PermissionError, as an unreadable folder does (chmod does not
    deny root, so the test does not rely on it)."""
    real = os.scandir

    def scandir(path="."):
        if os.path.abspath(os.fspath(path)) == os.path.abspath(blocked):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real(path)

    return mock.patch.object(os, "scandir", scandir)


def deny_stat(blocked):
    """Patch os.stat so looking at `blocked` raises PermissionError, as a path under a folder that cannot be traversed
    does; os.path.isdir turns that into False, "not there"."""
    real = os.stat

    def stat(path, *a, **k):
        if os.path.abspath(os.fspath(path)) == os.path.abspath(blocked):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real(path, *a, **k)

    return mock.patch.object(os, "stat", stat)


class AssetsPoolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game = os.path.join(self.tmp.name, "game")
        self.work = os.path.join(self.tmp.name, "work")
        self.out = os.path.join(self.work, "assets")
        # Mark the other assets stages done so only the texture-pool loop runs.
        for rel in make_game.MODEL_ARCHIVES:
            stem = os.path.splitext(os.path.basename(rel))[0]
            for mode in ("models", "textures"):
                self.stub(os.path.join(self.out, stem, mode, "manifest.json"))
                with open(os.path.join(self.out, stem, mode, make_game.DECODER_RECORD), "w") as f:
                    f.write(make_game.decoder_digest())
        self.stub(os.path.join(self.out, "audio", "audio_manifest.json"))
        self.stub(os.path.join(self.out, "audio", "ogg.done"))
        self.stub(os.path.join(self.out, "audio", "xma.done"))

    def stub(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w"):
            pass

    def pool(self, rel, size=100):
        path = os.path.join(self.game, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        return path

    def run_assets(self):
        """Run assets() with a fake exporter; return (source, pool directory) per ttpf call."""
        calls = []

        def fake_run(cmd, *a, **k):
            if cmd[1] == "ttpf":
                src, dest = cmd[2], cmd[3]
                pool_dir = os.path.join(dest, os.path.splitext(os.path.basename(src))[0])
                os.makedirs(pool_dir, exist_ok=True)
                with open(os.path.join(pool_dir, "manifest.json"), "w") as f:
                    json.dump({"src": src, "Count": 0, "Textures": []}, f)
                calls.append((src, pool_dir))

        with mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
             mock.patch.object(make_game, "run", fake_run), \
             mock.patch.object(make_game.shutil, "which", lambda n: None), \
             mock.patch.object(make_game, "log", lambda *a: None), \
             mock.patch.object(ttpf_assets, "complete_pool", lambda *a: (0, 0)):
            make_game.assets(self.game, self.work, False, 1)
        return calls

    def test_same_basename_in_two_folders_exports_both(self):
        a = self.pool("data/ttdb/A/pool.ttpf")
        b = self.pool("data/ttdb/B/pool.ttpf", 101)
        calls = self.run_assets()
        self.assertEqual([src for src, _ in calls], [a, b])
        dirs = [d for _, d in calls]
        self.assertEqual(len(set(dirs)), 2)
        for src, d in calls:
            self.assertEqual(os.path.basename(d), "pool")
            self.assertTrue(os.path.isfile(os.path.join(d, "manifest.json")), src)
        # A resumed build finds both manifests and must not export either pool again.
        self.assertEqual(self.run_assets(), [])

    def test_unique_pool_keeps_its_existing_output_folder(self):
        p = self.pool("data/ttdb/pool.ttpf")
        self.assertEqual(self.run_assets(), [(p, os.path.join(self.out, "ttpf", "pool"))])

    def test_same_basename_in_different_packs_stays_separate(self):
        a = self.pool("pack/a/data/ttdb/pool.ttpf")
        b = self.pool("pack/b/data/ttdb/pool.ttpf", 101)
        self.assertEqual([d for _, d in self.run_assets()], [
            os.path.join(self.out, "ttpf_pack", "a", "pool"),
            os.path.join(self.out, "ttpf_pack", "b", "pool"),
        ])

    def test_collision_is_scoped_to_one_destination_namespace(self):
        loose = self.pool("data/ttdb/pool.ttpf")
        packed = self.pool("pack/a/data/ttdb/pool.ttpf", 101)
        self.assertEqual([d for _, d in self.run_assets()], [
            os.path.join(self.out, "ttpf", "pool"),
            os.path.join(self.out, "ttpf_pack", "a", "pool"),
        ])
        self.assertEqual(len({loose, packed}), 2)

    def test_unlistable_folder_fails_the_pool_scan(self):
        blocked = os.path.join(self.game, "data", "ttdb", "blocked")
        os.makedirs(blocked)
        self.pool("data/ttdb/A/pool.ttpf")
        err = io.StringIO()
        with deny_listing(blocked), mock.patch.object(make_game, "log", lambda *a: None), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run") as run, \
                mock.patch.object(make_game.shutil, "which", lambda n: None), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            make_game.assets(self.game, self.work, False, 1)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("blocked", err.getvalue())
        self.assertIn("Permission denied", err.getvalue())
        run.assert_not_called()   # no export ran on a partial list of pools

    def test_game_folder_that_cannot_be_checked_fails_instead_of_having_no_pools(self):
        self.pool("data/ttdb/A/pool.ttpf")
        err = io.StringIO()
        with deny_stat(self.game), mock.patch.object(make_game, "log", lambda *a: None), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run") as run, \
                mock.patch.object(make_game.shutil, "which", lambda n: None), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            make_game.assets(self.game, self.work, False, 1)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("Permission denied", err.getvalue())
        run.assert_not_called()


class UnlistableFolderTest(unittest.TestCase):
    """A sub-folder that cannot be listed fails the stage; it must not pass for one without pools or streams."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_xma_decode_fails_on_unlistable_folder_instead_of_finishing(self):
        xma = os.path.join(self.tmp.name, "xma")
        blocked = os.path.join(xma, "bank0")
        os.makedirs(blocked)
        err = io.StringIO()
        with deny_listing(blocked), mock.patch.object(make_game, "run") as run, \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            make_game.decode_xma("ffmpeg", xma, 1)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("bank0", err.getvalue())
        run.assert_not_called()

    def test_xma_folder_that_cannot_be_checked_is_not_a_finished_decode(self):
        xma = os.path.join(self.tmp.name, "xma")
        os.makedirs(xma)
        with open(os.path.join(xma, "a.xma.wav"), "wb"):
            pass
        err = io.StringIO()
        with deny_stat(xma), mock.patch.object(make_game, "run") as run, \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            make_game.decode_xma("ffmpeg", xma, 1)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("Permission denied", err.getvalue())
        run.assert_not_called()

    def test_a_missing_xma_folder_has_nothing_pending(self):
        self.assertEqual(make_game.pending_xma(os.path.join(self.tmp.name, "nowhere")), [])


if __name__ == "__main__":
    unittest.main()
