"""tools/probe_gfx1036_vcmp16.py finds vk_compute_test in any build folder (TEST-953): the platform's file name, a
Visual Studio configuration subfolder, an explicit --build, and the missing-exe ERROR line naming the folder given.
Fake files only, no GPU, no spirv-as.

    python -I tests/test_probe_vcmp16_build_dir.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import probe_gfx1036_vcmp16 as p  # noqa: E402

PROBES = ("test_vcmp_inactive_lanes", "test_vcmpx_exec_update", "test_constants_and_literals",
          "test_sgpr_high_halves", "test_f16_nan_and_modifiers")


def touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()
    return path


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(p, "VK_COMPUTE_TEST", p.VK_COMPUTE_TEST), \
            mock.patch.object(p.shutil, "which", return_value="spirv-as"), \
            mock.patch.multiple(p, **{name: mock.DEFAULT for name in PROBES}) as probes, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = p.main(argv)
        exe = p.VK_COMPUTE_TEST
    return rc, exe, probes, out.getvalue(), err.getvalue()


class FindChecker(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.build = os.path.join(tmp.name, "build")
        os.mkdir(self.build)

    def test_windows_name(self):
        with mock.patch.object(p, "EXE", ".exe"):
            want = touch(self.build, "vk_compute_test.exe")
            touch(self.build, "vk_compute_test")
            self.assertEqual(p.find_vk_compute_test(self.build), want)

    def test_posix_name(self):
        with mock.patch.object(p, "EXE", ""):
            want = touch(self.build, "vk_compute_test")
            touch(self.build, "vk_compute_test.exe")
            self.assertEqual(p.find_vk_compute_test(self.build), want)

    def test_release_subfolder(self):
        want = touch(self.build, "Release", "vk_compute_test" + p.EXE)
        self.assertEqual(p.find_vk_compute_test(self.build), want)

    def test_build_folder_wins_over_a_subfolder(self):
        want = touch(self.build, "vk_compute_test" + p.EXE)
        touch(self.build, "Release", "vk_compute_test" + p.EXE)
        self.assertEqual(p.find_vk_compute_test(self.build), want)

    def test_explicit_build_is_the_one_used(self):
        other = os.path.join(self.build, "clang")
        want = touch(other, "Debug", "vk_compute_test" + p.EXE)
        rc, exe, probes, _, err = run_main(["--build", other])
        self.assertFalse(rc, err)
        self.assertEqual(exe, want)
        self.assertTrue(all(probe.called for probe in probes.values()))

    def test_default_is_runtime_build(self):
        _, exe, _, _, _ = run_main([])
        self.assertTrue(exe.startswith(os.path.join(ROOT, "runtime", "build") + os.sep), exe)

    def test_nothing_built_is_the_missing_exe_error(self):
        touch(self.build, "Release", "vk_store_test" + p.EXE)
        rc, _, probes, out, err = run_main(["--build", self.build])
        self.assertEqual(rc, 2)
        self.assertEqual(out, "")
        lines = err.splitlines()
        self.assertEqual(len(lines), 1, err)
        self.assertTrue(lines[0].startswith("ERROR: no " + os.path.join(self.build, "vk_compute_test" + p.EXE)),
                        lines[0])
        self.assertFalse(any(probe.called for probe in probes.values()))


if __name__ == "__main__":
    unittest.main()
