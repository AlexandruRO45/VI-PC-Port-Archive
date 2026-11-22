"""make_game.py: an emptied shader disassembly is decoded again, and a translation that produces nothing fails.

Standard library only; no GPU, no Vulkan SDK, no game data: shader_disasm.py, rdna2spirv.py and the shard processes
are mocked, and the disassembly below is a hand-written .s file.

    python -I tests/test_make_game_shader_disassembly.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402

SIGNATURE = "test llvm-mc 21"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class DisassemblyMatchesTest(unittest.TestCase):
    """disassembly_matches(): a signed folder counts only while it still holds the unpacked shaders' assembly."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dxbc = os.path.join(tmp.name, "work", "shaders", "dxbc")
        self.disasm = os.path.join(self.dxbc, "disasm")

    def indexes(self, carved=("shader_0",), unpacked=("shader_0",)):
        write(os.path.join(self.dxbc, "index.json"), json.dumps([{"file": n + ".dxbc"} for n in carved]))
        write(os.path.join(self.dxbc, "fssm_index.json"), json.dumps([{"name": n} for n in unpacked]))

    def sign(self, signature=SIGNATURE):
        write(os.path.join(self.dxbc, "disasm.done"), signature + "\n")

    def assembly(self, *stems):
        for stem in stems:
            write(os.path.join(self.disasm, stem + ".s"), "s_endpgm\n")
        write(os.path.join(self.disasm, "stems.json"), json.dumps(list(stems)))

    def test_a_signed_disassembly_of_the_unpacked_shaders_matches(self):
        self.indexes()
        self.sign()
        self.assembly("shader_0")
        self.assertTrue(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_an_emptied_disassembly_does_not_match(self):
        # The folder, its signature and the listing survive when the .s files are deleted to save disk: decode again.
        self.indexes()
        self.sign()
        self.assembly("shader_0")
        os.remove(os.path.join(self.disasm, "shader_0.s"))
        self.assertFalse(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_a_partly_emptied_disassembly_does_not_match(self):
        # Two listings written, one .s left: translating this would publish a corpus that lacks the other silently.
        self.indexes(carved=("shader_0", "shader_1"), unpacked=("shader_0", "shader_1"))
        self.sign()
        self.assembly("shader_0", "shader_1")
        os.remove(os.path.join(self.disasm, "shader_1.s"))
        self.assertFalse(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_a_blob_that_wrote_no_listing_is_not_a_redo(self):
        # An unpacked blob with no s_endpgm produces no .s and the run still succeeds; the listing is the baseline.
        self.indexes(carved=("shader_0", "shader_1"), unpacked=("shader_0", "shader_1"))
        self.sign()
        self.assembly("shader_0")
        self.assertTrue(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_assembly_for_a_shader_that_is_no_longer_unpacked_does_not_match(self):
        self.indexes()
        self.sign()
        self.assembly("shader_0", "stale")
        self.assertFalse(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_a_file_that_is_not_assembly_does_not_match(self):
        self.indexes()
        self.sign()
        write(os.path.join(self.disasm, "notes.txt"), "deleted\n")
        self.assertFalse(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_another_llvm_mc_does_not_match(self):
        self.indexes()
        self.sign()
        self.assembly("shader_0")
        self.assertFalse(make_game.disassembly_matches(self.dxbc, "other llvm-mc"))

    def test_no_signature_does_not_match(self):
        self.indexes()
        self.assembly("shader_0")
        self.assertFalse(make_game.disassembly_matches(self.dxbc, SIGNATURE))

    def test_missing_unpack_indexes_do_not_match(self):
        self.sign()
        self.assembly("shader_0")
        self.assertFalse(make_game.disassembly_matches(self.dxbc, SIGNATURE))


class TranslateTest(unittest.TestCase):
    """translate(): a merge that publishes nothing is an error, not a finished stage."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.disasm = os.path.join(tmp.name, "dxbc", "disasm")
        self.spirv = os.path.join(tmp.name, "spirv")
        os.makedirs(self.disasm)

    def translate_with_index(self, text):
        class Done:
            def __init__(self, *a, **kw):
                pass

            def wait(self):
                return 0

        def fake_py(tool, *args):
            if "--merge" in args:
                write(os.path.join(self.spirv, "index.json"), text)

        with mock.patch.object(make_game, "log", lambda *a: None), \
                mock.patch.object(make_game, "py", side_effect=fake_py), \
                mock.patch.object(make_game.subprocess, "Popen", Done):
            make_game.translate(self.disasm, self.spirv, 2)

    def test_an_empty_disassembly_publishes_no_index(self):
        # The reported case: the .s files are gone, every shard writes an empty part, and --merge publishes {}.
        with self.assertRaises(SystemExit) as cm:
            self.translate_with_index("{}")
        self.assertEqual(cm.exception.code, 1)
        # The empty file must not survive to be read as a finished translation by the next run, and neither may the
        # shard plan: with it, the retry would merge the same empty parts again even after the input was repaired.
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, make_game.SHARD_PLAN)))
        with self.assertRaises(SystemExit) as cm:
            self.translate_with_index("{}")
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, make_game.SHARD_PLAN)))

    def test_an_unreadable_merge_fails(self):
        with self.assertRaises(SystemExit) as cm:
            self.translate_with_index("")
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))

    def test_a_non_empty_merge_passes(self):
        self.translate_with_index(json.dumps({"shader_0": {"stage": "ps"}}))
        with open(os.path.join(self.spirv, "index.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"shader_0": {"stage": "ps"}})


class RequireTranslatedShadersTest(unittest.TestCase):
    """require_translated_shaders(): a marker left by an older run is not a translated corpus."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = os.path.join(tmp.name, "work")
        self.index = os.path.join(self.work, "shaders", "spirv", "index.json")
        os.makedirs(os.path.dirname(self.index))

    def test_an_empty_index_is_refused(self):
        write(self.index, "{}")
        with self.assertRaises(SystemExit) as cm:
            make_game.require_translated_shaders(self.work)
        self.assertEqual(cm.exception.code, 1)

    def test_an_unreadable_index_is_refused(self):
        write(self.index, "")
        with self.assertRaises(SystemExit) as cm:
            make_game.require_translated_shaders(self.work)
        self.assertEqual(cm.exception.code, 1)

    def test_a_missing_index_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            make_game.require_translated_shaders(self.work)
        self.assertEqual(cm.exception.code, 1)

    def test_a_non_empty_index_returns_the_spirv_folder(self):
        write(self.index, json.dumps({"shader_0": {"stage": "ps"}}))
        self.assertEqual(make_game.require_translated_shaders(self.work),
                         os.path.join(self.work, "shaders", "spirv"))


if __name__ == "__main__":
    unittest.main()
