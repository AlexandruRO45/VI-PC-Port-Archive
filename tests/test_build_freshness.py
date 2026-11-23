"""A runtime tool older than the runtime sources is rebuilt, not run (#7148).

    python -I tests/test_build_freshness.py

Standard library only. Files and modification times are synthetic and cmake is mocked: no GPU, no build.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import first_light  # noqa: E402
import make_game  # noqa: E402

OLD, NEW = 1_000_000_000, 1_100_000_000


class StaleBinary(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.build = os.path.join(self.root, "runtime", "build", "linux")
        os.makedirs(self.build)
        os.makedirs(os.path.join(self.root, "runtime", "test"))
        self.source = os.path.join(self.root, "runtime", "test", "vk_harness.cpp")
        for name in ("vk_harness", "vk_pipeline_check"):
            self.touch(os.path.join(self.build, name), OLD)

    @staticmethod
    def touch(path, when):
        with open(path, "wb") as f:
            f.write(b"x")
        os.utime(path, (when, when))

    def harness(self, source_time):
        self.touch(self.source, source_time)
        with mock.patch.object(first_light, "ROOT", self.root), \
                mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(first_light.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
                mock.patch("sys.stderr", io.StringIO()):
            os.environ.pop("LEONIDA_VK_HARNESS", None)
            found = first_light.find_harness(None)
        return found, run

    def test_first_light_rebuilds_a_harness_older_than_the_sources(self):
        found, run = self.harness(NEW)
        self.assertEqual(found, os.path.join(self.build, "vk_harness"))
        run.assert_called_once_with(["cmake", "--build", self.build, "--target", "vk_harness"])

    def test_first_light_keeps_a_harness_newer_than_the_sources(self):
        _, run = self.harness(OLD - 10)
        run.assert_not_called()

    def test_first_light_stops_when_the_rebuild_fails(self):
        self.touch(self.source, NEW)
        with mock.patch.object(first_light, "ROOT", self.root), \
                mock.patch.object(first_light.subprocess, "run", return_value=mock.Mock(returncode=1)), \
                mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                first_light.find_harness(None)

    def pipelines_preflight(self, source_time):
        self.touch(self.source, source_time)
        work = os.path.join(self.root, "work")
        spirv = os.path.join(work, "shaders", "spirv")
        os.makedirs(spirv, exist_ok=True)
        with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8") as f:
            f.write('{"shader_0": {}}')
        checker = os.path.join(self.build, "vk_pipeline_check")
        with mock.patch("sys.version_info", (3, 12, 0)), \
                mock.patch.dict(os.environ, {"LLVM_MC": "llvm-mc"}), \
                mock.patch("shutil.which", return_value="/bin/tool"), \
                mock.patch("shutil.disk_usage", return_value=mock.Mock(free=500 * 2**30)), \
                mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=checker), \
                mock.patch.object(make_game, "build_pipeline_check") as build, \
                mock.patch("sys.stderr", io.StringIO()):
            make_game.preflight(["pipelines"], work)
        return build

    @unittest.skipIf(os.name == "nt", "the Linux build path")
    def test_make_game_rebuilds_a_checker_older_than_the_sources(self):
        self.pipelines_preflight(NEW).assert_called_once()

    @unittest.skipIf(os.name == "nt", "the Linux build path")
    def test_make_game_keeps_a_checker_newer_than_the_sources(self):
        self.pipelines_preflight(OLD - 10).assert_not_called()


if __name__ == "__main__":
    unittest.main()
