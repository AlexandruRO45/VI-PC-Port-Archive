"""tools/sandbox.py: a scene the filesystem refuses is one ERROR line and exit 2, not a traceback.

sandbox_scene.main() already answers a refused --out that way; sandbox.py calls build() directly, so the same
folder-or-missing-parent case used to leave a bare PermissionError/FileNotFoundError escaping main().
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import sandbox  # noqa: E402

ROW = '{"archetype": 0, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "lod_level": 0}\n'


class SceneWriteErrors(unittest.TestCase):
    """One placed row, no exported model: the scene is written, so the --scene path itself is what fails."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.assets = os.path.join(self.root, "assets")
        os.makedirs(self.assets)
        self.entities = self._file("entities.jsonl", ROW)
        self.tool = self._file("vk_sandbox.exe", "")
        self._find_tool, self._pick_pair = sandbox.find_tool, sandbox.pick_pair
        sandbox.find_tool = lambda build: self.tool
        sandbox.pick_pair = lambda shaders: ("vs.spv", "ps.spv")

    def tearDown(self):
        sandbox.find_tool, sandbox.pick_pair = self._find_tool, self._pick_pair
        self.tmp.cleanup()

    def _file(self, name, text):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def _run(self, scene):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = sandbox.main(["--work", self.root, "--entities", self.entities, "--models", self.assets,
                               "--center", "0", "0", "--check", "--scene", scene])
        return rc, err.getvalue()

    def test_scene_folder_is_one_error_line(self):
        folder = os.path.join(self.root, "scene")
        os.makedirs(folder)
        rc, err = self._run(folder)
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertTrue(err.startswith("ERROR:"), err)
        self.assertNotIn("Traceback", err)

    def test_scene_under_a_regular_file_is_one_error_line(self):
        regular = self._file("regular", "")
        rc, err = self._run(os.path.join(regular, "scene.bin"))
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertTrue(err.startswith("ERROR:"), err)


if __name__ == "__main__":
    unittest.main()
