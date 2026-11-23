"""Checkout-local model and texture folders must not be committed.

    python -I tests/test_game_data_gitignore.py

These are output paths used by the local gallery. Test the real Git ignore rules with nonexistent names, so this
test does not create or inspect any asset files.
"""
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "requires git")
class GameDataGitignoreTest(unittest.TestCase):
    def is_ignored(self, path):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--", path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return result.returncode == 0

    def test_checkout_local_asset_outputs_are_ignored(self):
        for path in (
            "models/synthetic.gltf",
            "models/nested/synthetic.obj",
            "textures/synthetic.png",
            "textures/nested/synthetic.jpg",
            "textures/synthetic.webp",
            "models/synthetic.glb",
            "textures/synthetic.dds",
        ):
            with self.subTest(path=path):
                self.assertTrue(self.is_ignored(path))

    def test_whitelisted_repository_logo_is_not_ignored(self):
        self.assertFalse(self.is_ignored("assets/logo.png"))


if __name__ == "__main__":
    unittest.main()
