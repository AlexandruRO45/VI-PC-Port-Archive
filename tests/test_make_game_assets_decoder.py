"""make_game.py: model and texture exports resume by decoder fingerprint and output inventory.

Standard library only; no game data, no .NET: the exporter is a fake that writes a manifest.json the way rpf9dump
does when an export finishes.

    python -I tests/test_make_game_assets_decoder.py
"""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402
import rpf_asset_resume as resume  # noqa: E402


class ManifestItemsTest(unittest.TestCase):
    def test_invalid_token_does_not_read_the_rest_of_a_large_manifest(self):
        stream = io.BytesIO(b'{"src":?,"Items":[' + b'{"File":"shape.obj"},' * 100_000 + b'{}]}')
        with self.assertRaisesRegex(ValueError, "invalid JSON token"):
            list(resume.ManifestItems(stream))
        self.assertLess(stream.tell(), 128 * 1024)

    def test_each_value_is_decoded_once_across_chunks(self):
        row = {"File": "shape.obj", "Extra": ['é"\\]{}' * 20_000, [True, False, None]]}
        payload = json.dumps({"Count": 1, "Items": [row], "Scan": {"Entries": 1}},
                             ensure_ascii=False).encode("utf-8")
        real_loads = json.loads
        with mock.patch.object(resume.json, "loads", side_effect=real_loads) as loads:
            items = resume.ManifestItems(io.BytesIO(payload))
            self.assertEqual(list(items), [row])
        self.assertEqual(loads.call_count, 6)  # three property names and one decoded value for each


class AssetsDecoderTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = tmp.name
        self.models = os.path.join(self.work, "assets", "scene", "models")
        self.textures = Path(self.work, "assets", "scene", "textures")
        self.made_dds = self.textures / "dictionary" / "000_test.dds"
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
            mode = cmd[1]
            folder = Path(cmd[3], "scene", mode)
            if mode == "models":
                rows = [{"File": "shape.obj", "Embedded": None}]
                output = folder / "shape.obj"
            else:
                rows = [{"Dir": "dictionary", "Textures": [{"Index": 0, "Name": "test"},
                                                           {"Index": 1, "Name": "unsupported"}]}]
                output = self.made_dds
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.is_file():
                output.write_bytes(b"fixture")
            # Count is the number of resources considered by rpf9dump, including failed decodes.
            (folder / resume.MANIFEST).write_text(json.dumps({"Count": 4, "Items": rows}),
                                                encoding="utf-8", newline="\n")
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
        self.assertEqual(self.export("d2"), ["models", "textures"])
        self.assertFalse(os.path.exists(os.path.join(self.models, "old.bin")))
        self.assertEqual(self.export("d2"), [])

    def test_a_folder_without_the_record_exports_again(self):
        self.export("d1")
        os.remove(os.path.join(self.models, make_game.DECODER_RECORD))
        self.assertEqual(self.export("d1"), ["models"])

    def test_missing_texture_is_repaired_without_treating_extras_as_outputs(self):
        self.export("d1")
        sibling = self.made_dds.with_name("002_user.dds")
        sibling.write_bytes(b"extra")
        model = Path(self.models, "shape.obj")
        model.write_bytes(b"already exported")
        self.made_dds.unlink()
        self.assertEqual(self.export("d1"), ["textures"])
        self.assertEqual(self.made_dds.read_bytes(), b"fixture")
        self.assertEqual(sibling.read_bytes(), b"extra")
        self.assertEqual(model.read_bytes(), b"already exported")
        sibling.unlink()
        self.assertEqual(self.export("d1"), [])

    def test_legacy_manifest_with_all_candidates_is_migrated_without_export(self):
        self.export("d1")
        inventory = Path(self.models, resume.INVENTORY)
        inventory.unlink()
        self.assertEqual(self.export("d1"), [])
        self.assertTrue(inventory.is_file())

    def test_legacy_texture_decode_skip_causes_at_most_one_refresh(self):
        self.export("d1")
        self.made_dds.write_bytes(b"existing texture")
        (self.textures / resume.INVENTORY).unlink()
        self.assertEqual(self.export("d1"), ["textures"])
        self.assertEqual(self.made_dds.read_bytes(), b"existing texture")
        self.assertEqual(self.export("d1"), [])

    def test_corrupt_manifest_or_inventory_retries_only_the_affected_mode(self):
        self.export("d1")
        for name in (resume.MANIFEST, resume.INVENTORY):
            with self.subTest(name=name):
                (self.textures / name).write_text("{", encoding="utf-8", newline="\n")
                self.assertEqual(self.export("d1"), ["textures"])
                self.assertEqual(self.export("d1"), [])

    def test_failed_inventory_publication_keeps_outputs_and_no_temporary(self):
        self.export("d1")
        inventory = Path(self.models, resume.INVENTORY)
        inventory.unlink()
        with mock.patch.object(resume, "replace", side_effect=PermissionError("injected rename failure")):
            with self.assertRaises(SystemExit):
                self.export("d1")
        self.assertFalse(inventory.exists())
        self.assertEqual(list(Path(self.models).glob("*.partial")), [])
        self.assertEqual(Path(self.models, "shape.obj").read_bytes(), b"fixture")
        self.assertEqual(self.export("d1"), [])

    def test_inventory_cannot_name_an_output_outside_the_export(self):
        self.export("d1")
        inventory = Path(self.models, resume.INVENTORY)
        doc = json.loads(inventory.read_text(encoding="utf-8"))
        doc["files"] = [["..", "outside.obj"]]
        outside = Path(self.models).parent / "outside.obj"
        outside.write_bytes(b"unrelated")
        inventory.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
        self.assertEqual(self.export("d1"), ["models"])
        self.assertEqual(outside.read_bytes(), b"unrelated")

    def test_linked_dictionary_cannot_be_used_for_a_retry(self):
        self.export("d1")
        outside = Path(self.work, "outside")
        self.made_dds.parent.rename(outside)
        link = self.made_dds.parent
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                                    capture_output=True, text=True)
            if result.returncode:
                self.skipTest(f"cannot create directory junction: {result.stderr}")
        else:
            link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(SystemExit):
            self.export("d1")
        self.assertEqual((outside / self.made_dds.name).read_bytes(), b"fixture")

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
