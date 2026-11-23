"""make_game.py's step for the retail executable's own programs (exe_shaders): the steps of bvh_shaders() on the
executable, with each program's stage read from its container's PSV0.

    python -I tests/test_exe_shaders_stage.py

Zero game data: the "executable" is synthetic bytes around hand-built DXBC containers (no ILDN, a PSV0 part, an XBCH
record that is a zlib stream of a made-up FSSM blob), and llvm-mc and the translator are stood in for the way
tests/test_bvh_shaders_stage.py does.
"""
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import make_game  # noqa: E402
from test_bvh_shaders_stage import dll_like, fssm_blob  # noqa: E402
from test_pe_shader_containers import container, psv0, xdat  # noqa: E402

TOOLS = ["pe_shader_containers.py", "fssm_unpack.py", "shader_disasm.py", "rdna2spirv.py"]
STAGES = {"container_00": "vs", "container_01": "ps", "container_02": "ps"}   # pe_shader_containers.py's names
PSV0_CODE = {"ps": 0, "vs": 1}


def exe_container(stage: str, blob: bytes) -> bytes:
    """DXBC { PSV0 = the stage, DXIL = program header + XDAT { 0x0D = XBCH + zlib(blob), 0x07 } }: the parts of the
    executable's containers the tools read (docs/embedded_shaders.md, "Retail executable")."""
    packed = zlib.compress(blob)
    xbch = b"XBCH" + struct.pack("<IIIII", 1, len(packed), 0, len(blob), 0) + packed
    prefix = b"e\x00\x05\x00" + struct.pack("<I", 24) + b"DXIL" + b"\x05\x01\x00\x00" + struct.pack("<II", 0x70, 0x668)
    dxil = prefix + xdat([(0x0D, xbch), (0x07, b"NOBC" + bytes(8))], len(prefix))
    return container([(b"PSV0", psv0(PSV0_CODE[stage])), (b"DXIL", dxil)])


class ExeStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)
        self.exe = os.path.join(self.game, make_game.GAME_EXE)
        self.blobs = {name: fssm_blob(k) for k, name in enumerate(STAGES)}
        self.untranslated = set()        # the programs the stand-in translator has no rule for
        self.write_exe()
        self.unpacked = os.path.join(self.work, "shaders", "exe")
        self.spirv = os.path.join(self.work, "shaders", "spirv", "exe")
        self.index = os.path.join(self.spirv, "index.json")
        self.settings = os.path.join(self.spirv, "exe_build.json")
        self.calls, self.logs = [], []
        mc = os.path.join(self.tmp.name, "fake_mc.py")
        with open(mc, "w", encoding="utf-8", newline="\n") as f:
            f.write("print('\\ts_endpgm')\n")
        patch = mock.patch.dict(os.environ, {"LLVM_MC": f'"{sys.executable}" "{mc}"'})
        patch.start()
        self.addCleanup(patch.stop)

    def write_exe(self):
        with open(self.exe, "wb") as f:
            f.write(dll_like(*(exe_container(STAGES[name], blob) for name, blob in self.blobs.items())))

    def fake_translate(self, *args):
        """rdna2spirv.py: a module and an index entry, with the stage of the extraction index, per listing."""
        out = args[args.index("--out") + 1]
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(os.path.dirname(args[0]), "index.json"), encoding="utf-8") as f:
            stages = {e["name"]: e["stage"] for e in json.load(f)}
        index = {}
        for listing in sorted(os.listdir(args[0])):
            name = listing[:-2]
            if name in self.untranslated:
                continue
            with open(os.path.join(os.path.dirname(args[0]), "fssm", name + ".fssm"), "rb") as f:
                index[name] = {"stage": stages[name], "fssm_sha256": hashlib.sha256(f.read()).hexdigest()}
            with open(os.path.join(out, name + ".spv"), "wb") as f:
                f.write(b"module of " + name.encode())
        with open(os.path.join(out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(index, f)

    def run_tool(self, tool_name, *args):
        """The repository's own extracting, unpacking and disassembling tools; only the translator is stood in for."""
        self.calls.append((tool_name, args))
        if tool_name == "rdna2spirv.py":
            return self.fake_translate(*args)
        make_game.run([sys.executable, "-I", os.path.join(make_game.TOOLS, tool_name), *args],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run_step(self, signature="llvm-mc 23", emulate=False, fallbacks=()):
        self.calls, self.logs = [], []
        with mock.patch.object(make_game, "py", self.run_tool), \
                mock.patch.object(make_game, "log", self.logs.append):
            make_game.exe_shaders(self.game, self.work, signature, 3, emulate=emulate, fallbacks=fallbacks)
        return [name for name, _ in self.calls]

    def test_runs_once_with_the_stage_of_each_container_then_only_reports(self):
        self.assertEqual(self.run_step(), TOOLS)
        (_, extract), (_, unpack), (_, disasm), (_, translate) = self.calls
        self.assertEqual(extract, (self.exe, "--extract", self.unpacked))
        self.assertEqual(unpack, (self.unpacked, "--carve-index", os.path.join(self.unpacked, "index.json")))
        self.assertEqual(disasm, (os.path.join(self.unpacked, "fssm"), "--unpack-index",
                                  os.path.join(self.unpacked, "fssm_index.json"), "--out",
                                  os.path.join(self.unpacked, "disasm"), "--mcpu", "gfx1013", "--jobs", 3))
        self.assertEqual(translate, (os.path.join(self.unpacked, "disasm"), "--out", self.spirv, "--no-asm"))
        # What the runtime looks a graphics program up by: its stage (from PSV0) and the SHA-256 of its FSSM blob.
        with open(self.index, encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual({n: (m["stage"], m["fssm_sha256"]) for n, m in index.items()},
                         {n: (STAGES[n], hashlib.sha256(blob).hexdigest()) for n, blob in self.blobs.items()})
        self.assertTrue(os.path.isfile(self.settings))
        report = f"shaders: 3 of 3 {make_game.GAME_EXE} programs translated"
        self.assertIn(report, self.logs)
        self.assertEqual(self.run_step(), [])
        self.assertEqual(self.logs, [report])

    def test_each_input_change_runs_it_again_and_leaves_no_duplicates(self):
        self.run_step()
        self.assertEqual(self.run_step(signature="llvm-mc 24"), TOOLS)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True), TOOLS)
        self.assertIn("--wave64-emulate", self.calls[3][1])
        flags = ("--no-float-controls",)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        self.assertIn("--no-float-controls", self.calls[3][1])
        self.blobs["container_02"] = fssm_blob(9)
        self.write_exe()
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), [])
        with mock.patch.object(make_game, "translation_sources_digest", return_value="another translator"):
            self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        os.remove(os.path.join(self.spirv, "container_00.spv"))
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        # Five runs into the same folder: pe_shader_containers.py alone would have added container_NN_NN.dxbc.
        self.assertEqual(sorted(f for f in os.listdir(self.unpacked) if f.endswith(".dxbc")),
                         [n + ".dxbc" for n in sorted(STAGES)])

    def test_no_executable_no_step_and_an_old_index_goes_with_it(self):
        """The executable is often absent. A work folder that had it still holds that run's index, which the runtime
        loads whenever it exists."""
        self.assertEqual(self.run_step(), TOOLS)
        os.remove(self.exe)
        for _ in range(2):                                         # the second time with nothing left to remove
            self.assertEqual(self.run_step(), [])
            self.assertFalse(os.path.exists(self.index))
            self.assertFalse(os.path.exists(self.settings))
            self.assertEqual(self.logs, [f"shaders: no {make_game.GAME_EXE} in the game folder, so none of its own "
                                         "programs to translate"])
        self.write_exe()
        self.assertEqual(self.run_step(), TOOLS)
        self.assertTrue(os.path.isfile(self.index) and os.path.isfile(self.settings))

    def test_no_executable_in_a_new_work_folder_creates_nothing(self):
        os.remove(self.exe)
        self.assertEqual(self.run_step(), [])
        self.assertFalse(os.path.exists(self.spirv))
        self.assertFalse(os.path.exists(self.unpacked))

    def test_an_untranslated_program_is_named_on_every_run(self):
        self.untranslated = {"container_01"}
        for expected in (TOOLS, []):
            self.assertEqual(self.run_step(), expected)
            self.assertIn(f"shaders: 2 of 3 {make_game.GAME_EXE} programs translated", self.logs)
            self.assertEqual(len([line for line in self.logs if "NOT translated: container_01." in line]), 1)
        with open(self.index, encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)), ["container_00", "container_02"])

    def test_nothing_translated_fails_and_keeps_no_index(self):
        self.untranslated = set(STAGES)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.run_step()
        self.assertIn(f"none of the {make_game.GAME_EXE} programs translated", err.getvalue())
        self.assertFalse(os.path.exists(self.index))
        self.assertFalse(os.path.exists(self.settings))
        self.untranslated = set()
        self.assertEqual(self.run_step(), TOOLS)


class ShadersStageWithTheExecutable(unittest.TestCase):
    """The whole shaders() stage: the executable's step comes last and does not make the game's shaders run again."""

    def test_the_step_runs_last_and_alone_when_only_the_executable_changes(self):
        with tempfile.TemporaryDirectory() as t:
            work, game = os.path.join(t, "work"), os.path.join(t, "game")
            os.makedirs(game)
            exe = os.path.join(game, make_game.GAME_EXE)
            with open(exe, "wb") as f:
                f.write(dll_like(exe_container("vs", fssm_blob(1))))
            ran = []
            dxbc = os.path.join(work, "shaders", "dxbc")
            own = os.path.join(work, "shaders", "exe")

            def write(text, *parts):
                os.makedirs(os.path.dirname(os.path.join(*parts)), exist_ok=True)
                with open(os.path.join(*parts), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)

            def fake_py(tool_name, *args):
                ran.append(tool_name)
                folder, name = (own, "container_00") if args[0].startswith(own) else (dxbc, "shader")
                if tool_name == "shader_carve.py":
                    write('[{"file": "shader.dxbc"}]', dxbc, "index.json")
                elif tool_name == "pe_shader_containers.py":
                    write('[{"name": "container_00", "file": "container_00.dxbc", "stage": "vs"}]', own, "index.json")
                elif tool_name == "fssm_unpack.py":
                    write('[{"name": "%s"}]' % name, folder, "fssm_index.json")
                elif tool_name == "shader_disasm.py":
                    write("s_endpgm\n", folder, "disasm", name + ".s")
                elif tool_name == "rdna2spirv.py":
                    out = args[args.index("--out") + 1]
                    write('{"container_00": {"stage": "vs", "fssm_sha256": "%s"}}' % ("a" * 64), out, "index.json")
                    write("", out, "container_00.spv")

            def fake_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
                ran.append("translate")
                write("", spirv, "index.json")

            def stage():
                ran.clear()
                with mock.patch.object(make_game, "py", fake_py), \
                        mock.patch.object(make_game, "need", lambda *a: None), \
                        mock.patch.object(make_game, "translate", fake_translate), \
                        mock.patch.object(make_game, "llvm_mc_signature", return_value="llvm-mc 23"), \
                        mock.patch.object(make_game, "log", lambda *a: None):
                    make_game.shaders(game, work, 2)
                return list(ran)

            self.assertEqual(stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate", *TOOLS])
            self.assertEqual(stage(), [])
            with open(exe, "wb") as f:
                f.write(dll_like(exe_container("vs", fssm_blob(2))))
            self.assertEqual(stage(), TOOLS)
            for sub in ((), ("exe",)):
                self.assertTrue(os.path.isfile(os.path.join(work, "shaders", "spirv", *sub, "index.json")))


if __name__ == "__main__":
    unittest.main()
