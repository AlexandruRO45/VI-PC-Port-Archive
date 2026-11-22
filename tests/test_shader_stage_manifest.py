"""tools/shader_stage_manifest.py: the shader lists the carve, unpack and disassembly steps read.

    python -I tests/test_shader_stage_manifest.py

Every exit is a ValueError naming the file and the field, so the stages fail with their own message instead of a
KeyError or a silently short list. Standard library only; no game files.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from shader_stage_manifest import shader_stems  # noqa: E402


class ShaderStemsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "index.json")

    def stems(self, doc, field="name", suffix=""):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(doc if isinstance(doc, str) else json.dumps(doc))
        return shader_stems(self.path, field, suffix)

    def refused(self, doc, field="name", suffix="", match="expected"):
        with self.assertRaises(ValueError) as cm:
            self.stems(doc, field, suffix)
        self.assertIn(match, str(cm.exception))
        self.assertIn(self.path, str(cm.exception))

    def test_carved_files_give_their_stems(self):
        self.assertEqual(self.stems([{"file": "s0.dxbc"}, {"file": "s1.dxbc"}], "file", ".dxbc"),
                         {"s0", "s1"})

    def test_unpacked_names_need_no_suffix(self):
        self.assertEqual(self.stems([{"name": "upscaler"}, {"name": "sharpen"}]), {"upscaler", "sharpen"})

    def test_duplicate_rows_collapse_to_one_stem(self):
        self.assertEqual(self.stems([{"name": "s0"}, {"name": "s0"}]), {"s0"})

    def test_a_non_array_is_refused(self):
        for doc in ("{}", "null", '"s0"'):
            with self.subTest(doc=doc):
                self.refused(doc, match="expected an array")

    def test_a_row_that_is_not_an_object_is_refused(self):
        for doc in ('["s0"]', "[3]", "[null]"):
            with self.subTest(doc=doc):
                self.refused(doc, match="expected a shader basename")

    def test_a_missing_or_unusable_field_is_refused(self):
        for doc in ("[{}]", '[{"name": 3}]', '[{"name": ""}]', '[{"name": "."}]', '[{"name": ".."}]'):
            with self.subTest(doc=doc):
                self.refused(doc, match="expected a shader basename")

    def test_a_field_that_carries_a_path_is_refused(self):
        for value in ("dir/s0.dxbc", "dir\\s0.dxbc", "/s0.dxbc"):
            with self.subTest(value=value):
                self.refused([{"file": value}], "file", ".dxbc", match="expected a shader basename")

    def test_a_suffix_mismatch_is_refused(self):
        self.refused([{"file": "s0.dxbc"}], "file", ".fssm", match="expected a .fssm filename")

    def test_a_name_that_is_only_the_suffix_is_refused(self):
        self.refused([{"file": ".dxbc"}], "file", ".dxbc", match="expected a .dxbc filename")

    def test_a_missing_file_is_an_oserror(self):
        with self.assertRaises(OSError):
            shader_stems(self.path, "name")


if __name__ == "__main__":
    unittest.main()
