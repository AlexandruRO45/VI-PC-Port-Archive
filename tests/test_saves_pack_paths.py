"""Tests for path containment in tools/saves.py pack_from_dir().

All inputs are synthetic in-memory/temporary JSAV structures; no game files are used.

    python -I tests/test_saves_pack_paths.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "saves.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import saves  # noqa: E402


def _make_extracted_dir(root_dir):
    block_bytes = b'{"$0x11111111":{"$0x10000000":{"$0xaaaa0000":1}}}'
    jsav = saves.build_jsav(3, 1, [0x10000000], [1], [block_bytes])
    extracted = os.path.join(root_dir, "extracted")
    saves.extract(self_write(os.path.join(root_dir, "in.jsav"), jsav), extracted)
    return extracted


def self_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


class PackPathContainmentTest(unittest.TestCase):
    def test_valid_relative_paths_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _make_extracted_dir(tmp)
            packed = saves.pack_from_dir(extracted)
            h, es = saves.parse_jsav(packed)
            self.assertEqual((h["version"], h["code_blocks"], len(es)), (3, 1, 1))

    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _make_extracted_dir(tmp)
            secret = self_write(os.path.join(tmp, "secret.json"), b'{"secret":true}')
            self.assertTrue(os.path.isfile(secret))
            idx_path = os.path.join(extracted, "index.json")
            with open(idx_path, encoding="utf-8") as fh:
                idx = json.load(fh)
            idx["entries"][0]["file"] = "../secret.json"
            with open(idx_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(idx, fh)
            with self.assertRaisesRegex(ValueError, r"entries\[0\]\.file must be a relative path inside"):
                saves.pack_from_dir(extracted)

    def test_rejects_absolute_and_current_dir_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _make_extracted_dir(tmp)
            outside = self_write(os.path.join(tmp, "outside.json"), b'{"x":1}')
            idx_path = os.path.join(extracted, "index.json")
            with open(idx_path, encoding="utf-8") as fh:
                idx = json.load(fh)
            for bad_path in (outside, os.path.abspath(outside), ".", "blocks/.."):
                with self.subTest(bad_path=bad_path):
                    idx["entries"][0]["file"] = bad_path
                    with open(idx_path, "w", encoding="utf-8", newline="\n") as fh:
                        json.dump(idx, fh)
                    with self.assertRaisesRegex(ValueError, r"entries\[0\]\.file must be a relative path inside"):
                        saves.pack_from_dir(extracted)

    def test_rejects_symlink_escaping_source_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _make_extracted_dir(tmp)
            outside = self_write(os.path.join(tmp, "outside.json"), b'{"x":1}')
            link = os.path.join(extracted, "blocks", "escape.json")
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            idx_path = os.path.join(extracted, "index.json")
            with open(idx_path, encoding="utf-8") as fh:
                idx = json.load(fh)
            idx["entries"][0]["file"] = "blocks/escape.json"
            with open(idx_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(idx, fh)
            with self.assertRaisesRegex(ValueError, r"entries\[0\]\.file must be a relative path inside"):
                saves.pack_from_dir(extracted)

    def test_cli_pack_reports_one_line_error_for_escaping_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _make_extracted_dir(tmp)
            self_write(os.path.join(tmp, "outside.json"), b'{"x":1}')
            idx_path = os.path.join(extracted, "index.json")
            with open(idx_path, encoding="utf-8") as fh:
                idx = json.load(fh)
            idx["entries"][0]["file"] = "../outside.json"
            with open(idx_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(idx, fh)
            out_jsav = os.path.join(tmp, "out.jsav")
            res = subprocess.run(
                [sys.executable, "-I", TOOL, "pack", extracted, out_jsav],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 2)
            self.assertFalse(os.path.exists(out_jsav))
            self.assertIn("ERROR:", res.stderr)
            self.assertIn("entries[0].file", res.stderr)


if __name__ == "__main__":
    unittest.main()
