"""A sandbox check owns its temporary frame, not an existing image beside the scene."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import sandbox


class SandboxOutput(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.entities = self.root / "entities.jsonl"
        self.entities.write_text("", encoding="ascii", newline="\n")
        self.scene = self.root / "saved scene.bin"
        self.sibling = Path(str(self.scene) + ".ppm")
        self.stats = {"placed": 1, "draws": 1, "textures": 0, "textured_draws": 0, "center": [0, 0, 0]}

    def run_check(self, *, explicit=True, returncode=0, launch_error=False):
        paths = {}

        def build(args):
            paths["scene"] = Path(args.out)
            paths["scene"].write_bytes(b"synthetic scene")
            return self.stats

        def render(command, **kwargs):
            paths["frame"] = Path(command[command.index("--out") + 1])
            if launch_error:
                raise OSError("synthetic launch failure")
            paths["frame"].write_bytes(b"P6\n1 1\n255\n\0\0\0")
            return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

        args = ["--work", str(self.root), "--entities", str(self.entities),
                "--models", str(self.root), "--check"]
        if explicit:
            args += ["--scene", str(self.scene)]
        self.paths = paths
        with patch.object(sandbox, "find_tool", return_value="renderer"), \
                patch.object(sandbox, "pick_pair", return_value=("vs.spv", "ps.spv")), \
                patch.object(sandbox.sandbox_scene, "build", side_effect=build), \
                patch.object(sandbox.subprocess, "run", side_effect=render), \
                redirect_stdout(io.StringIO()):
            return sandbox.main(args)

    def test_existing_sibling_image_survives_success_and_failure(self):
        for returncode in (0, 1, 77):
            with self.subTest(returncode=returncode):
                self.sibling.write_bytes(b"an existing image that belongs to the user")
                self.assertEqual(self.run_check(returncode=returncode), returncode)
                self.assertTrue(self.sibling.exists())
                self.assertEqual(self.sibling.read_bytes(), b"an existing image that belongs to the user")
                self.assertEqual(self.scene.read_bytes(), b"synthetic scene")
                self.assertFalse(self.paths["frame"].exists())

    def test_temporary_scene_and_frame_are_removed(self):
        for returncode in (0, 1, 77):
            with self.subTest(returncode=returncode):
                self.assertEqual(self.run_check(explicit=False, returncode=returncode), returncode)
                self.assertFalse(self.paths["scene"].exists())
                self.assertFalse(self.paths["frame"].exists())

    def test_launch_error_preserves_sibling_and_removes_temporary_frame(self):
        self.sibling.write_bytes(b"keep me")
        with self.assertRaisesRegex(OSError, "synthetic launch failure"):
            self.run_check(launch_error=True)
        self.assertEqual(self.sibling.read_bytes(), b"keep me")
        self.assertFalse(self.paths["frame"].exists())

    def test_check_does_not_leave_an_image_beside_saved_scene(self):
        self.assertEqual(self.run_check(), 0)
        self.assertTrue(self.scene.exists())
        self.assertFalse(self.sibling.exists())
        self.assertFalse(self.paths["frame"].exists())


if __name__ == "__main__":
    unittest.main()
