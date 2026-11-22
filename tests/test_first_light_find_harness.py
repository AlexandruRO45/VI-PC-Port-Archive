"""first_light.find_harness finds vk_harness in a CMake per-configuration subfolder (Visual Studio generators).

Empty placeholder files only; nothing is run. No game data, no GPU.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import first_light  # noqa: E402

EXE = ".exe" if os.name == "nt" else ""


class FindHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_root, self.old_env = first_light.ROOT, os.environ.pop("LEONIDA_VK_HARNESS", None)
        first_light.ROOT = self.tmp.name
        self.addCleanup(self.restore)

    def restore(self):
        first_light.ROOT = self.old_root
        if self.old_env is not None:
            os.environ["LEONIDA_VK_HARNESS"] = self.old_env

    def place(self, *parts):
        path = os.path.join(self.tmp.name, "runtime", "build", *parts, "vk_harness" + EXE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()
        return path

    def test_build_root(self):
        path = self.place()
        self.assertEqual(first_light.find_harness(None), path)

    def test_config_subfolders(self):
        for parts in (("RelWithDebInfo",), ("Release",), ("windows", "Release"), ("windows", "RelWithDebInfo"),
                      ("linux", "Release")):
            with self.subTest(parts=parts):
                path = self.place(*parts)
                self.assertEqual(first_light.find_harness(None), path)
                os.remove(path)

    def test_missing_is_a_clean_exit(self):
        with self.assertRaises(SystemExit):
            first_light.find_harness(None)


if __name__ == "__main__":
    unittest.main()
