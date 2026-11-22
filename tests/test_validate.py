"""Unit tests for tools/validate.py (standard library only, no game data)."""
import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import validate  # noqa: E402


class ValidateManifestSchemaTest(unittest.TestCase):
    def test_real_repo_manifest_is_valid(self):
        manifest_path = os.path.join(ROOT, "game", "xbox-1.0.2.1.json")
        self.assertEqual(validate.validate_manifest_schema(manifest_path), [])

    def test_bad_manifest_reports_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as f:
                json.dump({"title": "X", "version": "1", "files": {"a.bin": {"size": -1, "sha256": "nope"}}}, f)
            errs = validate.validate_manifest_schema(bad)
            self.assertEqual(len(errs), 2)


class ValidateGameFolderTest(unittest.TestCase):
    def _make_synthetic_repo_and_game(self, tmp: str):
        repo = os.path.join(tmp, "repo")
        game = os.path.join(tmp, "game")
        os.makedirs(os.path.join(repo, "game"), exist_ok=True)
        os.makedirs(game, exist_ok=True)

        files_data = {
            "MicrosoftGame.Config": b"<Game/>",
            "Catalog000.bin": b"CAT0",
            "appdata0.rpf": b"RPF9TEST",
            "extra.bin": b"1234567890",
        }
        manifest_files = {}
        for name, data in files_data.items():
            with open(os.path.join(game, name), "wb") as f:
                f.write(data)
            manifest_files[name] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        manifest_doc = {
            "title": "Synthetic Title",
            "version": "1.0.0.0",
            "expected_exe": "game.exe",
            "files": manifest_files,
        }
        with open(os.path.join(repo, "game", "xbox-1.0.0.0.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_doc, f)
        return repo, game

    def test_quick_and_full_validation_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, game = self._make_synthetic_repo_and_game(tmp)
            rep_quick = validate.validate_game_folder(game, full_sha256=False, root_dir=repo)
            self.assertTrue(rep_quick.ok, rep_quick.summary)
            rep_full = validate.validate_game_folder(game, full_sha256=True, jobs=1, root_dir=repo)
            self.assertTrue(rep_full.ok, rep_full.summary)

    def test_corrupted_file_detected_by_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, game = self._make_synthetic_repo_and_game(tmp)
            with open(os.path.join(game, "extra.bin"), "wb") as f:
                f.write(b"0987654321")  # same size, different hash
            rep_quick = validate.validate_game_folder(game, full_sha256=False, root_dir=repo)
            self.assertTrue(rep_quick.ok)
            rep_full = validate.validate_game_folder(game, full_sha256=True, jobs=1, root_dir=repo)
            self.assertFalse(rep_full.ok)
            self.assertEqual(rep_full.details["issues"][0]["kind"], "CHANGED")


class ValidateCodebaseTest(unittest.TestCase):
    def test_codebase_self_check_passes(self):
        rep = validate.validate_codebase(root_dir=ROOT)
        self.assertTrue(rep.ok, rep.summary)


if __name__ == "__main__":
    unittest.main()
