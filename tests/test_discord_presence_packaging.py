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
            self.assertEqual(json.loads((output / "discord_presence.json").read_text(encoding="utf-8")), profile)
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(list(game.iterdir()), [game / "synthetic.exe"])
            if os.name == "nt":
                launcher = (output / "Play.ps1").read_text(encoding="utf-8")
                self.assertIn("--run $image --overlay", launcher)
                self.assertIn("@GameArguments", launcher)
                self.assertTrue((output / "Play.cmd").is_file())
            profile["image_sha256"] = "0" * 64
            with config.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(profile, stream)
            with mock.patch.object(make_game, "log"), self.assertRaises(SystemExit):
                make_game.runtime(str(game), str(output), {"expected_exe": "synthetic.exe"},
                                  str(build), discord_presence=str(config))


if __name__ == "__main__":
    unittest.main()
