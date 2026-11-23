"""support.shaders_importable(): the GPU check's helper modules for one test, and no trace in the next one.

pytest runs every test file in one process, so a sys.path entry or a module name (`build`, `check`) left behind by
one test is seen by all the tests that run after it."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402


class ShadersImportable(unittest.TestCase):
    def test_the_helpers_import_and_nothing_stays_behind(self):
        path_before = list(sys.path)
        sys.modules.pop("build", None)
        with support.shaders_importable():
            import build
            self.assertTrue(callable(build.gs_container))
        self.assertEqual(sys.path, path_before)
        self.assertNotIn("build", sys.modules)


if __name__ == "__main__":
    unittest.main()
