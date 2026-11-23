"""make_game.py's exe step (exe_shaders): the DXBC containers embedded in the game executable are carved and translated
like the builder DLL's, into <work>/shaders/exe and <work>/shaders/spirv/exe.

    python -I tests/test_exe_shaders_stage.py

Zero game data: the "executable" is synthetic bytes around hand-built containers (no ILDN part, as the real ones
have none) and one corrupt DXBC hit. The extraction is the repository's own tool; the later tools are stood in for.
"""
import json
import os
import struct
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
from test_bvh_shaders_stage import fssm_blob, builder_container  # noqa: E402
from test_pe_shader_containers import container  # noqa: E402


def exe_container(seed: int) -> bytes:
    """A container with the parts the game executable's have (SFI0, DXIL; no ILDN) and a console program."""
    full = builder_container("x", fssm_blob(seed))
    # Rebuild without the ILDN part: take the SFI0 and DXIL payloads of the builder container.
    parts, count = {}, struct.unpack_from("<I", full, 28)[0]
    for k in range(count):
        off = struct.unpack_from("<I", full, 32 + 4 * k)[0]
        fourcc, size = struct.unpack_from("<4sI", full, off)
        parts[fourcc] = full[off + 8:off + 8 + size]
    return container([(b"SFI0", parts[b"SFI0"]), (b"DXIL", parts[b"DXIL"])])


def exe_like(*pieces: bytes) -> bytes:
    return b"MZ" + b"\0" * 126 + b"".join(p + b"\xcc" * 12 for p in pieces)


class ExeStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)
        self.exe = os.path.join(self.game, make_game.GAME_EXE)
        self.out = os.path.join(self.work, "shaders", "exe")
        self.spirv = os.path.join(self.work, "shaders", "spirv", "exe")
        self.calls, self.logs = [], []

    def run_step(self, py):
        self.calls, self.logs = [], []
        with mock.patch.object(make_game, "py", py), mock.patch.object(make_game, "log", self.logs.append):
            make_game.exe_shaders(self.game, self.work, "llvm-mc 23", 2)

    def test_missing_executable_skips_the_step(self):
        self.run_step(lambda *a: self.calls.append(a))
        self.assertEqual(self.calls, [])
        self.assertTrue(any(make_game.GAME_EXE in line for line in self.logs))

    def test_the_real_extraction_finds_valid_containers_and_skips_a_corrupt_one(self):
        corrupt = bytearray(exe_container(9))
        struct.pack_into("<I", corrupt, 24, 1 << 30)   # a total size that does not fit the image
        with open(self.exe, "wb") as f:
            f.write(exe_like(exe_container(1), bytes(corrupt), b"DXBC", exe_container(2)))
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "pe_shader_containers.py"),
                            self.exe, "--extract", self.out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual(len(index), 2)
        self.assertEqual([e["name"] for e in index], ["container_00", "container_01"])
        self.assertTrue(all(e["stage"] == "cs" and os.path.isfile(os.path.join(self.out, e["file"]))
                            for e in index))

    def test_paths_and_settings_file_are_the_exe_ones(self):
        with open(self.exe, "wb") as f:
            f.write(exe_like(exe_container(1)))

        def fake(tool, *args):
            self.calls.append((tool, args))
            if tool == "pe_shader_containers.py":
                os.makedirs(self.out, exist_ok=True)
                open(os.path.join(self.out, "a.dxbc"), "wb").close()
                with open(os.path.join(self.out, "index.json"), "w", encoding="utf-8") as f:
                    json.dump([{"name": "a", "file": "a.dxbc", "stage": "cs"}], f)
            elif tool == "rdna2spirv.py":
                os.makedirs(self.spirv, exist_ok=True)
                open(os.path.join(self.spirv, "a.spv"), "wb").close()
                with open(os.path.join(self.spirv, "index.json"), "w", encoding="utf-8") as f:
                    json.dump({"a": {"stage": "cs", "fssm_sha256": "0" * 64}}, f)

        self.run_step(fake)
        self.assertEqual(self.calls[0], ("pe_shader_containers.py", (self.exe, "--extract", self.out)))
        self.assertEqual(self.calls[-1][1][:3], (os.path.join(self.out, "disasm"), "--out", self.spirv))
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "exe_build.json")))
        self.assertFalse(os.path.exists(os.path.join(self.work, "shaders", "bvh")))
        self.assertIn(f"shaders: 1 of 1 {make_game.GAME_EXE} programs translated", self.logs)


if __name__ == "__main__":
    unittest.main()
