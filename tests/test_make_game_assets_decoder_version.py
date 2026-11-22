"""Tests for decoder version tracking in make_game assets stage (BUILD-866).

    python -I tests/test_make_game_assets_decoder_version.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_game  # noqa: E402


class AssetsDecoderVersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game = os.path.join(self.tmp.name, "game")
        self.work = os.path.join(self.tmp.name, "work")
        self.out = os.path.join(self.work, "assets")
        os.makedirs(self.game, exist_ok=True)
        os.makedirs(self.out, exist_ok=True)

        # Mark audio done so only model/texture export is exercised
        self.stub(os.path.join(self.out, "audio", "audio_manifest.json"))
        self.stub(os.path.join(self.out, "audio", "ogg.done"))
        self.stub(os.path.join(self.out, "audio", "xma.done"))

    def stub(self, path, content=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def run_assets(self, test_archive="archive.rpf", current_digest=None):
        calls = []
        logs = []

        if current_digest is None:
            current_digest = make_game.rpf9dump_digest()

        def fake_run(cmd, *a, **k):
            # cmd is [rpf9dump_exe, mode, src, out]
            mode = cmd[1]
            src = cmd[2]
            archive_name = os.path.splitext(os.path.basename(src))[0]
            dest_dir = os.path.join(cmd[3], archive_name, mode)
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"src": src, "mode": mode}, f)
            calls.append((mode, src))

        def fake_log(msg, *a, **k):
            logs.append(str(msg))

        with mock.patch.object(make_game, "MODEL_ARCHIVES", [test_archive]), \
             mock.patch.object(make_game, "rpf9dump_digest", lambda: current_digest), \
             mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
             mock.patch.object(make_game, "run", fake_run), \
             mock.patch.object(make_game, "log", fake_log), \
             mock.patch.object(make_game.shutil, "which", lambda n: None):
            make_game.assets(self.game, self.work, False, 1)

        return calls, logs

    def test_digest_format_and_determinism(self):
        sig1 = make_game.rpf9dump_digest()
        sig2 = make_game.rpf9dump_digest()
        self.assertEqual(len(sig1), 64)
        self.assertEqual(sig1, sig2)
        # Verify valid hex
        int(sig1, 16)

    def test_initial_export_writes_decoder_version(self):
        sig = "a" * 64
        calls, logs = self.run_assets(test_archive="test.rpf", current_digest=sig)

        # Both models and textures should have been exported
        self.assertEqual([c[0] for c in calls], ["models", "textures"])

        # Check decoder.sha256 was written in both destination directories
        for mode in ("models", "textures"):
            ver_file = os.path.join(self.out, "test", mode, "decoder.sha256")
            self.assertTrue(os.path.isfile(ver_file))
            with open(ver_file, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), sig)

    def test_matching_decoder_skips_reexport(self):
        sig = "b" * 64
        for mode in ("models", "textures"):
            dest_dir = os.path.join(self.out, "test", mode)
            self.stub(os.path.join(dest_dir, "manifest.json"))
            self.stub(os.path.join(dest_dir, "decoder.sha256"), sig + "\n")

        calls, logs = self.run_assets(test_archive="test.rpf", current_digest=sig)
        # Nothing should be exported
        self.assertEqual(calls, [])

    def test_changed_decoder_reexports_and_updates_hash(self):
        old_sig = "c" * 64
        new_sig = "d" * 64
        for mode in ("models", "textures"):
            dest_dir = os.path.join(self.out, "test", mode)
            self.stub(os.path.join(dest_dir, "manifest.json"))
            self.stub(os.path.join(dest_dir, "decoder.sha256"), old_sig + "\n")

        calls, logs = self.run_assets(test_archive="test.rpf", current_digest=new_sig)

        # Should re-export both models and textures
        self.assertEqual([c[0] for c in calls], ["models", "textures"])

        # Should log that decoder updated
        self.assertTrue(any("assets: decoder updated for test models: exporting again" in m for m in logs))
        self.assertTrue(any("assets: decoder updated for test textures: exporting again" in m for m in logs))

        # Check decoder.sha256 was updated to new_sig
        for mode in ("models", "textures"):
            ver_file = os.path.join(self.out, "test", mode, "decoder.sha256")
            with open(ver_file, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), new_sig)

    def test_missing_decoder_sha256_reexports_legacy_work_folder(self):
        sig = "e" * 64
        for mode in ("models", "textures"):
            dest_dir = os.path.join(self.out, "test", mode)
            # Manifest exists, but decoder.sha256 does not
            self.stub(os.path.join(dest_dir, "manifest.json"))

        calls, logs = self.run_assets(test_archive="test.rpf", current_digest=sig)

        # Should re-export both models and textures
        self.assertEqual([c[0] for c in calls], ["models", "textures"])

        # Should log that no decoder fingerprint was present
        self.assertTrue(any("assets: no decoder fingerprint for test models: exporting again" in m for m in logs))
        self.assertTrue(any("assets: no decoder fingerprint for test textures: exporting again" in m for m in logs))

        # Check decoder.sha256 is now written
        for mode in ("models", "textures"):
            ver_file = os.path.join(self.out, "test", mode, "decoder.sha256")
            self.assertTrue(os.path.isfile(ver_file))
            with open(ver_file, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), sig)

    def test_changed_decoder_clears_stale_files(self):
        old_sig = "f" * 64
        new_sig = "g" * 64
        dest_dir = os.path.join(self.out, "test", "models")
        self.stub(os.path.join(dest_dir, "manifest.json"))
        self.stub(os.path.join(dest_dir, "decoder.sha256"), old_sig + "\n")
        stray = self.stub(os.path.join(dest_dir, "stale_model.obj"), "old content")
        self.assertTrue(os.path.isfile(stray))

        self.run_assets(test_archive="test.rpf", current_digest=new_sig)

        # Stale file should be gone after re-export
        self.assertFalse(os.path.isfile(stray))

    def test_prerequisite_failure_preserves_existing_export(self):
        old_sig = "h" * 64
        new_sig = "i" * 64
        dest_dir = os.path.join(self.out, "test", "models")
        self.stub(os.path.join(dest_dir, "manifest.json"))
        self.stub(os.path.join(dest_dir, "decoder.sha256"), old_sig + "\n")
        kept = self.stub(os.path.join(dest_dir, "good_model.obj"), "must survive")

        def failing_rpf9dump(work):
            raise RuntimeError("dotnet not found or build failed")

        with mock.patch.object(make_game, "MODEL_ARCHIVES", ["test.rpf"]), \
             mock.patch.object(make_game, "rpf9dump_digest", lambda: new_sig), \
             mock.patch.object(make_game, "rpf9dump", failing_rpf9dump):
            with self.assertRaises(RuntimeError):
                make_game.assets(self.game, self.work, False, 1)

        # Existing files must still be preserved since prerequisite failed
        self.assertTrue(os.path.isfile(kept))
        with open(kept, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "must survive")


if __name__ == "__main__":
    unittest.main()
