"""make_game.py's pipeline-library compute step (library_compute_shaders, D3D-703) and its FSPX index keys.

    python -I tests/test_library_compute_stage.py

Zero game data: synthetic pipeline libraries and RPF9-like archives are built in memory via tests/test_shader_library.py.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import make_game  # noqa: E402
import shader_meta  # noqa: E402
from test_shader_library import D_CS, archive, build, kind0, layout_blob, prog, sample  # noqa: E402

SHADER_LIB = os.path.join(ROOT, "tools", "shader_library.py")


class LibraryIndexKeys(unittest.TestCase):
    """Unpacked library compute shaders carry both fspx_code_sha256 and fspx_state_sha256 for slot-67 lookup."""

    def test_unpacked_compute_keys_match_slot67_code_and_state(self):
        state_a = kind0(threads=(32, 1, 1), rsrc2=0xB8A)
        state_b = kind0(threads=(64, 1, 1), rsrc2=0xB8C)
        code = prog(0x13572468, 2)
        blobs = [state_a, state_b, b"\x02" * 20, layout_blob(3, D_CS, 1)]
        lib_bytes = build([], [(0, 2, 0, 3), (1, 2, 0, 3)], blobs, [code])
        with tempfile.TemporaryDirectory() as t:
            rpf = os.path.join(t, "shaders.rpf")
            with open(rpf, "wb") as f:
                f.write(archive(lib_bytes))
            out = os.path.join(t, "library")
            r = subprocess.run(
                [sys.executable, "-I", SHADER_LIB, rpf, "--out", out, "--stages", "cs",
                 "--disasm", "--backend", "python", "--jobs", "1"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for name, expected_state in (("lib_cs_00000", state_a), ("lib_cs_00000_s1", state_b)):
                disasm = os.path.join(out, "disasm", name + ".s")
                self.assertTrue(os.path.isfile(disasm))
                meta = shader_meta.collect(disasm, name, [[]], [[]])
                self.assertEqual(meta.get("fspx_code_sha256"), hashlib.sha256(code).hexdigest())
                self.assertEqual(meta.get("fspx_state_sha256"), hashlib.sha256(expected_state).hexdigest())
                self.assertNotIn("fssm_sha256", meta)


class LibraryComputeStage(unittest.TestCase):
    """Caching, invalidation, clash detection and cleanup in make_game.library_compute_shaders()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)
        self.rpf = os.path.join(self.game, "shaders.rpf")
        with open(self.rpf, "wb") as f:
            f.write(archive(sample()))
        self.unpacked = os.path.join(self.work, "shaders", "library")
        self.spirv = os.path.join(self.work, "shaders", "spirv", "library")
        self.calls = []
        self.listings = ["lib_cs_00003.s"]
        self.translations = {"lib_cs_00003": ("c" * 64, "s" * 64, b"synthetic module")}

    def fake_py(self, tool, *args):
        self.calls.append((tool, args))
        if tool == "shader_library.py":
            disasm = os.path.join(self.unpacked, "disasm")
            os.makedirs(disasm, exist_ok=True)
            for name in self.listings:
                with open(os.path.join(disasm, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write("\ts_endpgm\n")

    def fake_translate(self, disasm, spirv, jobs, emulate=False, fallbacks=()):
        self.calls.append(("translate", (disasm, spirv, jobs, emulate, fallbacks)))
        os.makedirs(spirv, exist_ok=True)
        index = {}
        for name, (code, state, module) in self.translations.items():
            index[name] = {"stage": "cs", "fspx_code_sha256": code, "fspx_state_sha256": state}
            with open(os.path.join(spirv, name + ".spv"), "wb") as f:
                f.write(module)
        with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(index, f)

    def run_step(self, signature="llvm-mc 23", jobs=2, emulate=False, fallbacks=()):
        self.calls = []
        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "translate", self.fake_translate), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.library_compute_shaders(
                self.game, self.work, signature, jobs, emulate=emulate, fallbacks=fallbacks,
            )
        return [tool for tool, _ in self.calls]

    def test_runs_once_then_skips_when_cached(self):
        self.assertEqual(self.run_step(), ["shader_library.py", "translate"])
        (_, unpack_args), (_, tr_args) = self.calls
        self.assertEqual(
            unpack_args,
            (self.rpf, "--out", self.unpacked, "--stages", "cs", "--disasm", "--jobs", "2"),
        )
        self.assertEqual(tr_args, (os.path.join(self.unpacked, "disasm"), self.spirv, 2, False, ()))
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "library_build.json")))
        self.assertEqual(self.run_step(), [])

    def test_passes_carve_when_dxbc_index_exists(self):
        dxbc = os.path.join(self.work, "shaders", "dxbc")
        os.makedirs(dxbc, exist_ok=True)
        with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("[]\n")
        self.assertEqual(self.run_step(), ["shader_library.py", "translate"])
        self.assertIn("--carve", self.calls[0][1])
        self.assertIn(dxbc, self.calls[0][1])

    def test_no_archive_skips_step(self):
        os.remove(self.rpf)
        self.assertEqual(self.run_step(), [])
        self.assertFalse(os.path.exists(self.spirv))

    def test_no_compute_listings_skips_translate_and_marker(self):
        self.listings = []
        self.assertEqual(self.run_step(), ["shader_library.py"])
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "library_build.json")))

    def test_missing_cached_module_rebuilds(self):
        self.assertEqual(self.run_step(), ["shader_library.py", "translate"])
        os.remove(os.path.join(self.spirv, "lib_cs_00003.spv"))
        self.assertEqual(self.run_step(), ["shader_library.py", "translate"])
        self.assertEqual(self.run_step(), [])

    def test_each_input_change_invalidates_cache(self):
        self.run_step()
        self.assertEqual(self.run_step(signature="llvm-mc 24"), ["shader_library.py", "translate"])
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True), ["shader_library.py", "translate"])
        self.assertTrue(self.calls[1][1][3])
        flags = ("--no-float-controls",)
        self.assertEqual(
            self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags),
            ["shader_library.py", "translate"],
        )
        self.assertEqual(self.calls[1][1][4], flags)
        with open(self.rpf, "ab") as f:
            f.write(b"\0\0\0\0")
        self.assertEqual(
            self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags),
            ["shader_library.py", "translate"],
        )
        with mock.patch.object(make_game, "translation_sources_digest", return_value="other"):
            self.assertEqual(
                self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags),
                ["shader_library.py", "translate"],
            )

    def test_same_keys_with_different_modules_fail_and_drop_index(self):
        self.translations = {
            "lib_cs_00003": ("c" * 64, "s" * 64, b"module A"),
            "lib_cs_00004": ("c" * 64, "s" * 64, b"module B"),
        }
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.run_step()
        self.assertIn("lib_cs_00003", err.getvalue())
        self.assertIn("lib_cs_00004", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "library_build.json")))

    def test_same_code_key_with_different_state_keys_is_allowed(self):
        self.translations = {
            "lib_cs_00003": ("c" * 64, "s" * 64, b"module A"),
            "lib_cs_00003_s1": ("c" * 64, "t" * 64, b"module B"),
        }
        self.assertEqual(self.run_step(), ["shader_library.py", "translate"])
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "library_build.json")))

    def test_interrupted_translation_cleans_stale_modules_and_shards(self):
        self.run_step()
        with open(os.path.join(self.spirv, "index.part0.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("{}")
        with open(os.path.join(self.spirv, "shard0.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("{}")
        with open(os.path.join(self.spirv, "notes.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("keep\n")

        def stop(*args, **kwargs):
            raise KeyboardInterrupt

        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "translate", stop), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(KeyboardInterrupt):
                make_game.library_compute_shaders(self.game, self.work, "llvm-mc 24", 2)
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "library_build.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "lib_cs_00003.spv")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.part0.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "shard0.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "notes.txt")))


class ShadersStageWithLibrary(unittest.TestCase):
    """make_game.shaders() runs library_compute_shaders after fspx_shaders without re-running the corpus."""

    def test_library_runs_after_corpus_and_rebuilds_independently(self):
        with tempfile.TemporaryDirectory() as t:
            work, game = os.path.join(t, "work"), os.path.join(t, "game")
            os.makedirs(game)
            rpf = os.path.join(game, "shaders.rpf")
            with open(rpf, "wb") as f:
                f.write(archive(sample()))
            dxbc = os.path.join(work, "shaders", "dxbc")
            lib_unpacked = os.path.join(work, "shaders", "library")
            ran = []

            def fake_py(tool, *args):
                ran.append(tool)
                if tool == "shader_carve.py":
                    os.makedirs(dxbc, exist_ok=True)
                    with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                        f.write('[{"file": "shader.dxbc"}]')
                elif tool == "fssm_unpack.py":
                    with open(os.path.join(dxbc, "fssm_index.json"), "w", encoding="utf-8", newline="\n") as f:
                        f.write('[{"name": "shader"}]')
                elif tool == "shader_disasm.py":
                    os.makedirs(os.path.join(dxbc, "disasm"), exist_ok=True)
                    with open(os.path.join(dxbc, "disasm", "shader.s"), "w", encoding="utf-8", newline="\n") as f:
                        f.write("s_endpgm\n")
                elif tool == "shader_library.py":
                    os.makedirs(os.path.join(lib_unpacked, "disasm"), exist_ok=True)
                    with open(os.path.join(lib_unpacked, "disasm", "lib_cs_00003.s"), "w",
                              encoding="utf-8", newline="\n") as f:
                        f.write("s_endpgm\n")

            def fake_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
                ran.append(("translate", os.path.basename(spirv)))
                os.makedirs(spirv, exist_ok=True)
                name = "lib_cs_00003" if os.path.basename(spirv) == "library" else "shader"
                with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump({name: {}}, f)
                with open(os.path.join(spirv, name + ".spv"), "wb") as f:
                    f.write(b"synthetic module")

            def stage():
                ran.clear()
                with mock.patch.object(make_game, "py", fake_py), \
                        mock.patch.object(make_game, "need", lambda *a: None), \
                        mock.patch.object(make_game, "translate", fake_translate), \
                        mock.patch.object(make_game, "llvm_mc_signature", return_value="llvm-mc 23"), \
                        mock.patch.object(make_game, "log", lambda *a: None):
                    make_game.shaders(game, work, 2)
                return list(ran)

            self.assertEqual(
                stage(),
                ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", ("translate", "spirv"),
                 "shader_library.py", ("translate", "library")],
            )
            self.assertEqual(stage(), [])
            # Removing only the library marker rebuilds only the library compute shaders.
            os.remove(os.path.join(work, "shaders", "spirv", "library", "library_build.json"))
            self.assertEqual(stage(), ["shader_library.py", ("translate", "library")])


if __name__ == "__main__":
    unittest.main()
