"""support.shaders_importable(): the GPU check's helper modules for one test, and no trace in the next one.

pytest runs every test file in one process, so a sys.path entry or a module name (`build`, `check`) left behind by
one test is seen by all the tests that run after it."""
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402


class ShadersImportable(unittest.TestCase):
    def setUp(self):
        # whatever "build" the process had before this test is what it has after it, even if an assertion fails
        had, previous = "build" in sys.modules, sys.modules.get("build")
        self.addCleanup(lambda: sys.modules.__setitem__("build", previous) if had else sys.modules.pop("build", None))

    def test_the_helpers_import_and_nothing_stays_behind(self):
        path_before = list(sys.path)
        sys.modules.pop("build", None)
        with support.shaders_importable():
            import build
            self.assertTrue(callable(build.gs_container))
        self.assertEqual(sys.path, path_before)
        self.assertNotIn("build", sys.modules)

    def test_a_build_module_loaded_elsewhere_is_set_aside_and_put_back(self):
        elsewhere = types.ModuleType("build")
        sys.modules["build"] = elsewhere
        with support.shaders_importable():
            import build
            self.assertTrue(callable(build.gs_container))
        self.assertIs(sys.modules.get("build"), elsewhere)


if __name__ == "__main__":
    unittest.main()
