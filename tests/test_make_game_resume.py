"""Exercise interrupted build stages using synthetic filenames and mock tool outputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_game


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_shader_retry_and_completed_fast_path(self):
        dxbc = self.work / "shaders" / "dxbc"
        spirv = dxbc.parent / "spirv"
        for folder in (dxbc / "disasm", dxbc / "fssm", spirv):
            folder.mkdir(parents=True)
        for rel in ("example.dxbc", "fssm/example.fssm", "disasm/example.s",
                    "index.json", "fssm_index.json", "disasm.done"):
            (dxbc / rel).write_text("fixture")
        for i in range(1001):
            (spirv / f"leftover{i}.spv").touch()

        def finish(*args):
            (spirv / "example.spv").write_text("module")
            (spirv / "index.json").write_text(json.dumps({"example": {}}))

        self.enterContext(mock.patch.object(make_game, "need"))
        retry = self.enterContext(mock.patch.object(make_game, "translate", side_effect=finish))
        make_game.shaders("", str(self.work), 1)
        (spirv / "example.spv").unlink()
        make_game.shaders("", str(self.work), 1)
        make_game.shaders("", str(self.work), 1)
        self.assertEqual(retry.call_count, 2)
        (spirv / "index.json").write_text("{}")
        retry.side_effect = None
        with self.assertRaises(SystemExit):
            make_game.shaders("", str(self.work), 1)

    def test_xma_retry_after_missing_ffmpeg_and_failed_decode(self):
        src = self.work / "example.xma.wav"
        dst = self.work / "example.wav"
        src.touch()
        with mock.patch.object(make_game.shutil, "which", return_value=None):
            make_game.decode_xma(str(self.work))
        self.assertFalse(dst.exists())

        def decode(cmd):
            Path(cmd[-1]).write_bytes(b"synthetic output")
            return subprocess.CompletedProcess(cmd, next(codes))

        codes = iter((1, 0))
        self.enterContext(mock.patch.object(make_game.shutil, "which", return_value="ffmpeg"))
        run = self.enterContext(mock.patch.object(make_game.subprocess, "run", side_effect=decode))
        with self.assertRaises(SystemExit):
            make_game.decode_xma(str(self.work))
        self.assertEqual(list(self.work.iterdir()), [src])
        make_game.decode_xma(str(self.work))
        self.assertTrue(dst.is_file())
        make_game.decode_xma(str(self.work))
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
