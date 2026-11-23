"""Tests for the vk_harness directory resolution in w64_graphics.py (AnaNazzDev's finding on the
coordination board: a Visual Studio build folder puts vk_harness.exe in RelWithDebInfo, and the script
stopped with a FileNotFoundError traceback instead of finding it or saying what was searched)."""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT_PATH = os.path.join(ROOT, "runtime", "test", "shaders", "w64_graphics.py")

spec = importlib.util.spec_from_file_location("w64_graphics", SCRIPT_PATH)
w64_graphics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w64_graphics)

EXE = w64_graphics.EXE


class TestHarnessDir(unittest.TestCase):
    def _build(self, layout):
        """A temporary build dir with vk_harness<EXE> placed per `layout` (None, "", or a config name)."""
        tmp = tempfile.mkdtemp()
        if layout is not None:
            target = os.path.join(tmp, layout) if layout else tmp
            os.makedirs(target, exist_ok=True)
            with open(os.path.join(target, "vk_harness" + EXE), "w", encoding="utf-8") as f:
                f.write("")
        return tmp

    def test_harness_in_the_directory_itself(self) -> None:
        tmp = self._build("")
        self.assertEqual(w64_graphics._harness_dir(tmp), tmp)

    def test_harness_in_a_vs_configuration_subdirectory(self) -> None:
        tmp = self._build("RelWithDebInfo")
        self.assertEqual(w64_graphics._harness_dir(tmp), os.path.join(tmp, "RelWithDebInfo"))

    def test_relwithdebinfo_wins_over_debug(self) -> None:
        tmp = self._build("Debug")
        os.makedirs(os.path.join(tmp, "RelWithDebInfo"))
        with open(os.path.join(tmp, "RelWithDebInfo", "vk_harness" + EXE), "w", encoding="utf-8") as f:
            f.write("")
        self.assertEqual(w64_graphics._harness_dir(tmp), os.path.join(tmp, "RelWithDebInfo"))

    def test_missing_harness_is_none(self) -> None:
        tmp = self._build(None)
        os.makedirs(os.path.join(tmp, "RelWithDebInfo"))  # a config dir without the harness
        self.assertIsNone(w64_graphics._harness_dir(tmp))

    def test_main_names_the_searched_directories_instead_of_a_traceback(self) -> None:
        build, tools = self._build(None), self._build(None)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = w64_graphics.main([build, tools, "ps_w64_a_skip"])
        self.assertEqual(rc, 1)
        self.assertIn(f"vk_harness{EXE} not found in {os.path.abspath(tools)}", stderr.getvalue())
        self.assertIn("configuration subdirectories", stderr.getvalue())

    def test_main_uses_the_harness_from_the_configuration_subdirectory(self) -> None:
        # With the harness resolvable, the run must get past the harness lookup: the next failure comes from
        # the fixture check itself (no spirv directory here), not from the harness resolution.
        build, tools = self._build(None), self._build("RelWithDebInfo")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = w64_graphics.main([build, tools, "ps_w64_a_skip"])
        self.assertEqual(rc, 1)
        self.assertNotIn("vk_harness", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
