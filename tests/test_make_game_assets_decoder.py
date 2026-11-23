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

    def export(self, digest):
        """Run assets() with a fake exporter and the given decoder digest; returns the modes it exported."""
        ran = []

        def fake_run(cmd, *a, **k):
            folder = os.path.join(cmd[3], "scene", cmd[1])
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "manifest.json"), "w") as f:
                f.write("{}")
            ran.append(cmd[1])
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
        self.assertEqual(self.export("d2"), ["models", "textures"])
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
        self.assertEqual(self.export("d2"), ["models", "textures"])

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

            # A contributor's global Git config must not reach this fixture: a `core.hooksPath` there makes the
            # commit below run an unrelated hook (a Conventional Commits commit-msg hook rejects "x"), and this
            # test is about the digest. tests/test_make_game_submodule.py isolates its fixtures the same way.
            git_env = dict(os.environ)
            git_env.pop("GIT_CONFIG_COUNT", None)
            git_env.pop("GIT_CONFIG_PARAMETERS", None)
            git_env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})

            def git(*args):
                subprocess.run(["git", "-C", sub, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                                "-c", "commit.gpgsign=false", *args], check=True, capture_output=True,
                               env=git_env)

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

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_the_fixture_commit_ignores_a_global_git_hook(self):
        """A contributor's global `core.hooksPath` must not decide whether the fixture can commit.

        A Conventional Commits commit-msg hook rejects the fixture's "x" message, so this file failed on every
        machine that sets one. The control commit proves the poisoned config reaches Git; the fixture run in a
        subprocess with that same config must still commit.
        """
        with tempfile.TemporaryDirectory() as root:
            hooks = os.path.join(root, "hooks")
            os.makedirs(hooks)
            hook = os.path.join(hooks, "commit-msg")
            with open(hook, "w", newline="\n") as f:
                f.write("#!/bin/sh\nexit 1\n")
            os.chmod(hook, 0o755)
            # Let Git write the value: a hand-written Windows path ("C:\\Users\\...") is a bad config line because
            # the parser reads backslashes as escapes, while `git config` escapes them itself. Reading it back
            # proves the file parses before the control commit runs.
            config = os.path.join(root, "gitconfig")
            subprocess.run(["git", "config", "--file", config, "core.hooksPath", hooks],
                           check=True, capture_output=True)
            readback = subprocess.run(["git", "config", "--file", config, "--get", "core.hooksPath"],
                                      check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(readback, hooks, "the hook path did not survive Git's own config quoting")
            env = dict(os.environ)
            env.pop("GIT_CONFIG_COUNT", None)
            env.pop("GIT_CONFIG_PARAMETERS", None)
            env.update({"GIT_CONFIG_GLOBAL": config, "GIT_CONFIG_NOSYSTEM": "1"})

            # Control: a commit made with this environment really does run the hook and is rejected.
            control = os.path.join(root, "control")
            os.makedirs(control)
            with open(os.path.join(control, "file"), "w", newline="\n") as f:
                f.write("x\n")
            for args in (("init", "-q"), ("add", "file")):
                subprocess.run(["git", "-C", control, *args], check=True, capture_output=True, env=env)
            rejected = subprocess.run(
                ["git", "-C", control, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                 "commit", "-q", "-m", "x"], capture_output=True, env=env)
            if rejected.returncode == 0:
                # A Git build or filesystem that does not run hooks from an arbitrary core.hooksPath cannot be
                # poisoned this way; the fixture check below still runs, it just cannot fail for this reason.
                self.skipTest("git did not run the commit-msg hook here")

            result = subprocess.run(
                [sys.executable, "-I", os.path.abspath(__file__),
                 "AssetsDecoderTest.test_the_digest_follows_an_uncommitted_submodule_edit"],
                cwd=root, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
