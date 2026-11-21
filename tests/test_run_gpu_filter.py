"""A filtered GPU run still builds the helper shaders its selected tests draw with.

`--filter ps_flat` selects ps_flat.asm but not vs_ids.asm, which vk_harness needs as the vertex shader. The
runner used to pass a missing vs_ids.spv and fail before drawing (issue #192).

    python -I tests/test_run_gpu_filter.py
"""
import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_gpu_tests.py")

spec = importlib.util.spec_from_file_location("run_gpu_tests", SCRIPT)
run_gpu_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_gpu_tests)


class SelectAsmFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        for name in ("ps_flat.asm", "ps_ops.asm", "vs_ids.asm"):
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
                f.write("")

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_filter_selects_everything(self):
        selected, build = run_gpu_tests.select_asm_files(self.dir, "")
        self.assertEqual(selected, ["ps_flat.asm", "ps_ops.asm", "vs_ids.asm"])
        self.assertEqual(build, selected)

    def test_ps_flat_builds_its_vertex_shader(self):
        selected, build = run_gpu_tests.select_asm_files(self.dir, "ps_flat")
        self.assertEqual(selected, ["ps_flat.asm"])
        self.assertEqual(build, ["ps_flat.asm", "vs_ids.asm"])

    def test_a_prefix_filter_also_builds_the_dependency(self):
        selected, build = run_gpu_tests.select_asm_files(self.dir, "ps_")
        self.assertEqual(selected, ["ps_flat.asm", "ps_ops.asm"])
        self.assertEqual(build, ["ps_flat.asm", "ps_ops.asm", "vs_ids.asm"])

    def test_unrelated_filter_does_not_add_the_dependency(self):
        selected, build = run_gpu_tests.select_asm_files(self.dir, "ps_ops")
        self.assertEqual(selected, ["ps_ops.asm"])
        self.assertEqual(build, ["ps_ops.asm"])

    def test_dependency_is_not_added_twice(self):
        selected, build = run_gpu_tests.select_asm_files(self.dir, "vs_ids")
        self.assertEqual(selected, ["vs_ids.asm"])
        self.assertEqual(build, ["vs_ids.asm"])

    def test_empty_selection(self):
        selected, build = run_gpu_tests.select_asm_files(self.dir, "not-a-test")
        self.assertEqual(selected, [])
        self.assertEqual(build, [])


if __name__ == "__main__":
    unittest.main()
