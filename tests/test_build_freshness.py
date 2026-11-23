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
from build_freshness import executable_name  # noqa: E402
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
        self.harness_path = os.path.join(self.build, executable_name("vk_harness"))
        self.touch(self.harness_path, OLD)
        checker = os.path.join(self.build, executable_name("vk_pipeline_check"))
        self.touch(checker, OLD)
        os.chmod(checker, 0o755)

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
        self.assertEqual(found, self.harness_path)
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
        with mock.patch("sys.version_info", (3, 12, 0)), \
                mock.patch("shutil.disk_usage", return_value=mock.Mock(free=500 * 2**30)), \
                mock.patch.object(make_game, "ROOT", self.root):
            make_game.preflight(["pipelines"], work)

    @unittest.skipIf(os.name == "nt", "the Linux build path")
    def test_make_game_rebuilds_a_checker_older_than_the_sources(self):
        with mock.patch.object(make_game, "build_pipeline_check") as build:
            self.pipelines_preflight(NEW)
        build.assert_called_once()

    @unittest.skipIf(os.name == "nt", "the Linux build path")
    def test_make_game_keeps_a_checker_newer_than_the_sources(self):
        with mock.patch.object(make_game, "build_pipeline_check") as build:
            self.pipelines_preflight(OLD - 10)
        build.assert_not_called()

    @unittest.skipIf(os.name == "nt", "the Linux build path")
    def test_make_game_reports_missing_cmake_when_rebuilding_a_stale_checker(self):
        err = io.StringIO()
        with mock.patch("shutil.which", return_value=None), \
                mock.patch.object(make_game.subprocess, "run", side_effect=FileNotFoundError), \
                mock.patch("sys.stderr", err), self.assertRaises(SystemExit) as exit_ctx:
            self.pipelines_preflight(NEW)
        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("ERROR: cmake not found, cannot rebuild vk_pipeline_check", err.getvalue())


if __name__ == "__main__":
    unittest.main()
