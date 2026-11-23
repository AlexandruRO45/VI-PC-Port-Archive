"""Resume the ordinary shader pipeline with hand-built archives and the Python shader tools."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import make_game
from test_fssm_unpack import make_dxbc, make_fssm, make_xbch, make_xdat
from test_shader_carve import make_rpf_entry


def shader_bytes(immediate):
    # A pixel shader containing s_mov_b32 s0, <inline integer> and s_endpgm.
    words = struct.pack("<II", 0xBE800380 + immediate, 0xBF810000)
    fssm = make_fssm(instructions=[(0xC0, struct.pack("<I", 0x100)), (0x128, words)])
    dxil = make_xdat({0x0D: make_xbch(fssm)}, pre_padding=struct.pack("<HH", 0x65, 0))
    return make_dxbc([(b"DXIL", dxil)]), fssm


class ShaderArchiveResumeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.game = Path(tmp.name, "source")
        self.game.mkdir()
        self.work = Path(tmp.name, "work")
        self.dxbc = self.work / "shaders" / "dxbc"
        self.spirv = self.work / "shaders" / "spirv"
        self.archive = self.game / "shaders.rpf"

    def write_archive(self, shaders):
        data = b"".join(make_rpf_entry(name, 1, shader_bytes(value)[0]) for name, value in shaders.items())
        # The compressed instructions can differ in length. Keep both file size and timestamp fixed.
        self.archive.write_bytes(data.ljust(1024, b"\0"))
        os.utime(self.archive, (1_700_000_000, 1_700_000_000))

    def build(self, game=None, stop_after=None):
        calls = []
        run_py = make_game.py

        def run_tool(tool, *args):
            calls.append(tool)
            if tool == "shader_disasm.py":
                args += ("--backend", "python")
            run_py(tool, *args)
            if tool == stop_after:
                raise KeyboardInterrupt

        # Use the project's CPU disassembler and SPIR-V assembler. This tests
        # the real shard and merge paths without requiring LLVM or a Vulkan SDK.
        with mock.patch.object(make_game, "py", side_effect=run_tool), \
                mock.patch.object(make_game, "llvm_mc_signature", return_value="test Python disassembler"), \
                mock.patch.object(make_game, "need"):
            make_game.shaders(str(game or self.game), str(self.work), 1,
                              fallbacks=("--assembler", "python", "--no-validate"))
        return calls

    def assert_outputs(self, shaders):
        carved = json.loads((self.dxbc / "index.json").read_text(encoding="utf-8"))
        unpacked = json.loads((self.dxbc / "fssm_index.json").read_text(encoding="utf-8"))
        merged = json.loads((self.spirv / "index.json").read_text(encoding="utf-8"))
        self.assertEqual({row["file"] for row in carved}, {name + ".dxbc" for name in shaders})
        self.assertEqual({row["name"] for row in unpacked}, set(shaders))
        self.assertEqual(set(merged), set(shaders))
        for folder, pattern in ((self.dxbc, "*.dxbc"), (self.dxbc / "fssm", "*.fssm"),
                                (self.dxbc / "disasm", "*.s"), (self.spirv, "*.spv")):
            self.assertEqual({p.stem for p in folder.glob(pattern)}, set(shaders), folder)
        for name, value in shaders.items():
            dxbc, fssm = shader_bytes(value)
            self.assertEqual((self.dxbc / (name + ".dxbc")).read_bytes(), dxbc)
            self.assertEqual((self.dxbc / "fssm" / (name + ".fssm")).read_bytes(), fssm)
            listing = (self.dxbc / "disasm" / (name + ".s")).read_bytes()
            self.assertIn(f"s_mov_b32 s0, {value} ".encode(), listing)
            self.assertEqual(merged[name]["src_sha256"], hashlib.sha256(listing).hexdigest())
            self.assertEqual(merged[name]["fssm_sha256"], hashlib.sha256(fssm).hexdigest())

    def test_same_filename_size_and_mtime_with_changed_code_rebuilds(self):
        self.write_archive({"shared": 1})
        self.build()
        self.assert_outputs({"shared": 1})
        old_module = (self.spirv / "shared.spv").read_bytes()
        old_stat = self.archive.stat()
        self.write_archive({"shared": 2})
        stat = self.archive.stat()
        self.assertEqual((stat.st_size, stat.st_mtime_ns), (old_stat.st_size, old_stat.st_mtime_ns))
        self.build()
        self.assertNotEqual((self.spirv / "shared.spv").read_bytes(), old_module)
        self.assert_outputs({"shared": 2})

    def test_identical_archive_in_another_game_folder_skips_all_stages(self):
        self.write_archive({"shared": 1})
        self.build()
        copy = self.game.with_name("moved")
        copy.mkdir()
        shutil.copyfile(self.archive, copy / "shaders.rpf")
        self.assertEqual(self.build(game=copy), [])
        self.assert_outputs({"shared": 1})

    def test_stop_after_replacement_carve_then_resume_with_changed_inventory(self):
        self.write_archive({"shared": 1, "removed": 1})
        self.build()
        self.assert_outputs({"shared": 1, "removed": 1})
        new = {"shared": 2, "added": 3}
        self.write_archive(new)
        with self.assertRaises(KeyboardInterrupt):
            self.build(stop_after="shader_carve.py")
        self.build()
        self.assert_outputs(new)
        self.assertEqual(self.build(), [])

    def test_switch_back_after_interrupted_disassembly_restores_original_module(self):
        self.write_archive({"shared": 1})
        self.build()
        original = (self.spirv / "shared.spv").read_bytes()
        self.write_archive({"shared": 2})
        with self.assertRaises(KeyboardInterrupt):
            self.build(stop_after="shader_disasm.py")
        self.write_archive({"shared": 1})
        self.build()
        self.assert_outputs({"shared": 1})
        self.assertEqual((self.spirv / "shared.spv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
