"""w64_graphics.py finds vk_harness where the CMake generator put it (TEST-1764): the build folder itself (Ninja,
Makefiles) or the RelWithDebInfo/Release subfolder a Visual Studio build uses, and reports a harness that was not built
as one error line instead of a FileNotFoundError traceback. Fake files only, so this runs on any OS."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import w64_graphics

HARNESS = "vk_harness" + w64_graphics.EXE


def touch(folder: Path, names=(HARNESS,)):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"")


class HarnessDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.build = Path(tmp.name) / "build"
        self.build.mkdir()

    def test_the_build_folder_itself(self):
        touch(self.build)
        self.assertEqual(w64_graphics.harness_dir(str(self.build)), str(self.build))

    def test_a_visual_studio_configuration_subfolder(self):
        touch(self.build / "RelWithDebInfo")
        self.assertEqual(w64_graphics.harness_dir(str(self.build)), str(self.build / "RelWithDebInfo"))

    def test_a_release_subfolder(self):
        touch(self.build / "Release")
        self.assertEqual(w64_graphics.harness_dir(str(self.build)), str(self.build / "Release"))

    def test_not_built_anywhere(self):
        self.assertIsNone(w64_graphics.harness_dir(str(self.build)))
        touch(self.build / "Custom")          # only the generator's configuration folders are searched
        self.assertIsNone(w64_graphics.harness_dir(str(self.build)))


class MainUsesTheHarnessFolder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.out = self.tmp / "out"
        (self.out / "spirv").mkdir(parents=True)
        (self.out / "spirv" / "ps_w64_a_skip.spv").write_bytes(b"")
        self.build = self.tmp / "build"
        self.build.mkdir()

    def run_main(self, tools):
        seen = []
        with patch.object(w64_graphics, "check", side_effect=lambda t, s, n: seen.append(t) or ""), \
                redirect_stdout(io.StringIO()):
            code = w64_graphics.main([str(self.out), str(tools), "ps_w64_a_skip"])
        return code, seen

    def test_the_configuration_folder_is_handed_to_the_fixtures(self):
        touch(self.build / "RelWithDebInfo")
        code, seen = self.run_main(self.build)
        self.assertEqual(code, 0)
        self.assertEqual([Path(t).resolve() for t in seen], [(self.build / "RelWithDebInfo").resolve()])

    def test_a_configuration_folder_given_as_the_tools_dir(self):
        touch(self.build / "Release")
        code, seen = self.run_main(self.build / "Release")
        self.assertEqual(code, 0)
        self.assertEqual([Path(t).resolve() for t in seen], [(self.build / "Release").resolve()])

    def test_a_missing_harness_is_one_error_line(self):
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = w64_graphics.main([str(self.out), str(self.build), "ps_w64_a_skip"])
        self.assertEqual(code, 1)
        self.assertIn(HARNESS, err.getvalue())
        for config in w64_graphics.CONFIG_DIRS:
            self.assertIn(config, err.getvalue())


if __name__ == "__main__":
    unittest.main()
