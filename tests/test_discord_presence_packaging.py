"""Private Discord configuration and automatic Windows launch packaging; synthetic files only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_game


class PackagingTest(unittest.TestCase):
    def test_default_setup_is_created_and_preserves_user_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game, output = root / "game", root / "output"
            game.mkdir()
            output.mkdir()
            image = game / "synthetic.exe"
            image.write_bytes(b"synthetic image")
            logo = game / "xbs_logo_480.png"
            logo.write_bytes(b"synthetic artwork fixture")
            source = {p.name: p.read_bytes() for p in game.iterdir()}
            make_game.package_discord_presence(str(game), str(output), str(image))
            config = output / "discord_presence.json"
            setup = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(setup, {"richpresence_application_id": ""})
            self.assertFalse((output / "leonida" / "presence_profile.json").exists())
            self.assertFalse((output / logo.name).exists())
            setup["richpresence_application_id"] = "123456789012345678"
            with config.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(setup, stream)
            before = config.read_bytes()
            make_game.package_discord_presence(str(game), str(output), str(image))
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual({p.name: p.read_bytes() for p in game.iterdir()}, source)
            with config.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump({"_comment": ["old setup"], "application_id": "", "call_rva": "0x0", "strings": {}}, stream)
            make_game.package_discord_presence(str(game), str(output), str(image))
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"richpresence_application_id": ""})

    def test_stage_runner_forwards_private_configuration(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap), mock.patch.object(make_game, "runtime") as runtime, \
                    mock.patch.object(make_game, "log"):
                make_game.run_build_stages(
                    ["runtime"], game="game", work="work", out="output", v={}, wav=False, jobs=1,
                    runtime_build=None, emulate=False, keep_emulated=True, fallbacks=(), overlap=overlap,
                    discord_presence="private.json")
                runtime.assert_called_once_with("game", "output", {}, None, 1, work="work",
                                                discord_presence="private.json")

    def test_configuration_and_launcher_are_bundled_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game, build, output = (root / name for name in ("game", "build", "output"))
            game.mkdir()
            build.mkdir()
            (game / "synthetic.exe").write_bytes(b"synthetic image")
            for name in make_game.RUNTIME_FILES:
                (build / name).write_bytes(b"synthetic runtime")
            config = root / "private.json"
            profile = {"application_id": "123456789012345678",
                       "image_sha256": hashlib.sha256(b"synthetic image").hexdigest(),
                       "call_rva": "0x1010", "setter_rva": "0x1100",
                       "strings": {"synthetic-service": {"presences": {"menu": "In the menu"}, "tokens": {}}}}
            with config.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(profile, stream)
            before = config.read_bytes()
            with mock.patch.object(make_game, "log"):
                make_game.runtime(str(game), str(output), {"expected_exe": "synthetic.exe"},
                                  str(build), discord_presence=str(config))
            self.assertEqual(json.loads((output / "discord_presence.json").read_text(encoding="utf-8")),
                             {"richpresence_application_id": profile["application_id"]})
            self.assertEqual(json.loads((output / "leonida" / "presence_profile.json").read_text(encoding="utf-8")), profile)
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(list(game.iterdir()), [game / "synthetic.exe"])
            # Migrate a previously bundled complete profile into internal data while preserving its ID.
            with (output / "discord_presence.json").open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(profile, stream)
            make_game.package_discord_presence(str(game), str(output), str(game / "synthetic.exe"))
            self.assertEqual(json.loads((output / "discord_presence.json").read_text(encoding="utf-8")),
                             {"richpresence_application_id": profile["application_id"]})
            self.assertEqual(json.loads((output / "leonida" / "presence_profile.json").read_text(encoding="utf-8")), profile)
            if os.name == "nt":
                launcher = (output / "Play.ps1").read_text(encoding="utf-8")
                self.assertIn("--run $image --overlay", launcher)
                self.assertIn("@GameArguments", launcher)
                self.assertTrue((output / "Play.cmd").is_file())
                command = (output / "Play.cmd").read_text(encoding="utf-8")
                self.assertNotIn("ExecutionPolicy", command)
                self.assertNotIn("Bypass", command)
            profile["image_sha256"] = "0" * 64
            with config.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(profile, stream)
            with mock.patch.object(make_game, "log"), self.assertRaises(SystemExit):
                make_game.runtime(str(game), str(output), {"expected_exe": "synthetic.exe"},
                                  str(build), discord_presence=str(config))


if __name__ == "__main__":
    unittest.main()
