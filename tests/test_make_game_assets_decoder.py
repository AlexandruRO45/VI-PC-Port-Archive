"""make_game.py: the assets stage exports models and textures again when the decoder that made them changed.

Standard library only; no game data, no .NET: the exporter is a fake that writes a manifest.json the way rpf9dump
does when an export finishes.

    python -I tests/test_make_game_assets_decoder.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


class AssetsDecoderTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = tmp.name
        self.models = os.path.join(self.work, "assets", "scene", "models")
        # The later steps of the stage are marked done so only the model and texture loop runs.
        audio = os.path.join(self.work, "assets", "audio")
        os.makedirs(audio)
        for name in ("audio_manifest.json", "ogg.done", "xma.done"):
            with open(os.path.join(audio, name), "w"):
                pass
        xma_dir = os.path.join(audio, "xma")
        os.makedirs(xma_dir, exist_ok=True)
        with open(os.path.join(xma_dir, make_game.DECODER_RECORD), "w") as f:
            f.write("d1\n")

    def export(self, digest):
        """Run assets() with a fake exporter and the given decoder digest; returns the modes it exported."""
        ran = []

        def fake_run(cmd, *a, **k):
            mode = cmd[1]
            if mode in ("models", "textures"):
                folder = os.path.join(cmd[3], "scene", mode)
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, "manifest.json"), "w") as f:
                    f.write("{}")
            elif mode == "xma":
                xma = os.path.join(cmd[3], "audio", "xma")
                os.makedirs(xma, exist_ok=True)
            ran.append(mode)
        with mock.patch.object(make_game, "MODEL_ARCHIVES", ["scene.rpf"]), \
                mock.patch.object(make_game, "decoder_digest", lambda: digest), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run", fake_run), \
                mock.patch.object(make_game.shutil, "which", lambda n: None), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.assets(os.path.join(self.work, "game"), self.work, False, 1)
        return ran

    def test_the_same_decoder_skips_and_another_one_exports_again(self):
        self.assertEqual(self.export("d1"), ["models", "textures"])
        with open(os.path.join(self.models, make_game.DECODER_RECORD), encoding="utf-8") as f:
            self.assertEqual(f.read(), "d1\n")
        self.assertEqual(self.export("d1"), [])
        # A file of the old decoder must not survive: rpf9dump keeps the files it finds.
        with open(os.path.join(self.models, "old.bin"), "w"):
            pass
        self.assertEqual(self.export("d2"), ["models", "textures", "xma"])
        self.assertFalse(os.path.exists(os.path.join(self.models, "old.bin")))
        self.assertEqual(self.export("d2"), [])

    def test_a_folder_without_the_record_exports_again(self):
        self.export("d1")
        os.remove(os.path.join(self.models, make_game.DECODER_RECORD))
        self.assertEqual(self.export("d1"), ["models"])

    def test_an_exporter_that_cannot_be_built_keeps_the_old_export(self):
        self.export("d1")
        with mock.patch.object(make_game, "MODEL_ARCHIVES", ["scene.rpf"]), \
                mock.patch.object(make_game, "decoder_digest", lambda: "d2"), \
                mock.patch.object(make_game, "rpf9dump", side_effect=SystemExit(1)), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit):
                make_game.assets(os.path.join(self.work, "game"), self.work, False, 1)
        self.assertTrue(os.path.isfile(os.path.join(self.models, "manifest.json")))
        with open(os.path.join(self.models, make_game.DECODER_RECORD), encoding="utf-8") as f:
            self.assertEqual(f.read(), "d1\n")
        self.assertEqual(self.export("d2"), ["models", "textures", "xma"])

    def test_xma_exports_again_when_decoder_changed(self):
        # Initial export with d1 writes decoder record into xma folder
        ran_modes = []

        def fake_dump(cmd, *a, **k):
            mode = cmd[1]
            ran_modes.append(mode)
            if mode == "xma":
                os.makedirs(os.path.join(cmd[3], "audio", "xma", "bank"), exist_ok=True)
                with open(os.path.join(cmd[3], "audio", "xma", "bank", "sample.xma.wav"), "w") as f:
                    f.write("data")

        audio = os.path.join(self.work, "assets", "audio")
        xma_dir = os.path.join(audio, "xma")
        xma_done = os.path.join(audio, "xma.done")
        xma_record = os.path.join(xma_dir, make_game.DECODER_RECORD)
        # Clear existing xma done marker
        if os.path.isfile(xma_done):
            os.remove(xma_done)
        shutil.rmtree(xma_dir, ignore_errors=True)

        with mock.patch.object(make_game, "MODEL_ARCHIVES", ()), \
                mock.patch.object(make_game, "decoder_digest", lambda: "d1"), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run", fake_dump), \
                mock.patch.object(make_game.shutil, "which", lambda n: None), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.assets(os.path.join(self.work, "game"), self.work, False, 1)

        self.assertIn("xma", ran_modes)
        self.assertTrue(os.path.isfile(xma_done))
        self.assertTrue(os.path.isfile(xma_record))
        with open(xma_record, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "d1")

        # Rerun with same decoder (d1) skips xma export
        ran_modes.clear()
        with mock.patch.object(make_game, "MODEL_ARCHIVES", ()), \
                mock.patch.object(make_game, "decoder_digest", lambda: "d1"), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run", fake_dump), \
                mock.patch.object(make_game.shutil, "which", lambda n: None), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.assets(os.path.join(self.work, "game"), self.work, False, 1)

        self.assertNotIn("xma", ran_modes)

        # Rerun with changed decoder (d2) re-exports xma and updates decoder record
        ran_modes.clear()
        with mock.patch.object(make_game, "MODEL_ARCHIVES", ()), \
                mock.patch.object(make_game, "decoder_digest", lambda: "d2"), \
                mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
                mock.patch.object(make_game, "run", fake_dump), \
                mock.patch.object(make_game.shutil, "which", lambda n: None), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.assets(os.path.join(self.work, "game"), self.work, False, 1)

        self.assertIn("xma", ran_modes)
        self.assertTrue(os.path.isfile(xma_done))
        with open(xma_record, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "d2")

    def test_the_digest_follows_the_exporter_sources(self):
        with tempfile.TemporaryDirectory() as tools:
            os.makedirs(os.path.join(tools, "rpf9dump"))
            digests = []
            for text in ("a", "b"):
                with open(os.path.join(tools, "rpf9dump", "Program.cs"), "w", newline="\n") as f:
                    f.write(text)
                make_game.decoder_digest.cache_clear()
                with mock.patch.object(make_game, "TOOLS", tools):
                    digests.append(make_game.decoder_digest())
            make_game.decoder_digest.cache_clear()
        self.assertNotEqual(*digests)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_the_digest_follows_an_uncommitted_submodule_edit(self):
        with tempfile.TemporaryDirectory() as root:
            sub = os.path.join(root, "third_party", "rpf9-explorer")
            os.makedirs(sub)
            src = os.path.join(sub, "Decoder.cs")

            def git(*args):
                subprocess.run(["git", "-C", sub, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                                "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)

            with open(src, "w", newline="\n") as f:
                f.write("a\n")
            git("init", "-q")
            git("add", "Decoder.cs")
            git("commit", "-q", "-m", "x")
            digests = []
            for text in ("a\n", "b\n"):
                with open(src, "w", newline="\n") as f:
                    f.write(text)
                make_game.decoder_digest.cache_clear()
                with mock.patch.object(make_game, "ROOT", root):
                    digests.append(make_game.decoder_digest())
            make_game.decoder_digest.cache_clear()
        self.assertNotEqual(*digests)


if __name__ == "__main__":
    unittest.main()
