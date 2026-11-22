"""w64_graphics.py argument handling: a mistyped fixture name is refused, not silently dropped.

    python -I tests/test_w64_graphics_args.py

`w64_graphics.py <build> [tools] [name ...]` filtered unknown names out of the argument list and exited 0, so a
targeted GPU verification looked complete while omitting the intended case, and an argument list with no known
name at all silently expanded to every fixture. The runner is used with `check` stubbed here, so no fixtures,
SPIR-V build, GPU or game files are needed.
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "runtime", "test", "shaders", "w64_graphics.py")

spec = importlib.util.spec_from_file_location("w64_graphics", RUNNER)
w64_graphics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w64_graphics)


class W64GraphicsArgsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.build = os.path.join(self.tmp.name, "build")
        self.tools = os.path.join(self.tmp.name, "tools")
        os.makedirs(self.tools)
        self.ran = []

    def run_main(self, *args):
        """Stub check() so every selection decision is observable without a SPIR-V build."""
        w64_graphics.check = lambda tools, spirv, name: self.ran.append((tools, spirv, name)) or ""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = w64_graphics.main([self.build, *args])
        return rc, out.getvalue(), err.getvalue()

    def test_unknown_name_is_refused_and_no_fixture_runs(self):
        rc, out, err = self.run_main("ps_w64_a_skip", "ps_w64_c_lane_id_typo")
        self.assertEqual(rc, 2, (out, err))
        self.assertEqual(self.ran, [])
        self.assertEqual(out, "")
        self.assertEqual(err,
                         "Unknown test shaders: ps_w64_c_lane_id_typo"
                         " (run with no names for all fixtures)\n")

    def test_an_all_unknown_list_is_refused(self):
        rc, out, err = self.run_main("totally_unknown_name")
        self.assertEqual(rc, 2, (out, err))
        self.assertEqual(self.ran, [])
        self.assertTrue(err.startswith("Unknown test shaders: totally_unknown_name"), err)

    def test_every_valid_name_runs(self):
        rc, out, err = self.run_main("ps_w64_a_skip", "vs_w64_a_skip")
        self.assertEqual(rc, 0, (out, err))
        self.assertEqual([name for _, _, name in self.ran], ["ps_w64_a_skip", "vs_w64_a_skip"])

    def test_no_names_runs_every_fixture(self):
        rc, out, err = self.run_main()
        self.assertEqual(rc, 0, (out, err))
        self.assertEqual([name for _, _, name in self.ran], sorted(w64_graphics.FIXTURES))

    def test_tools_dir_argument_is_still_taken_and_forwarded(self):
        rc, out, err = self.run_main(self.tools, "ps_w64_a_skip")
        self.assertEqual(rc, 0, (out, err))
        self.assertEqual([name for _, _, name in self.ran], ["ps_w64_a_skip"])
        self.assertEqual(self.ran[0][0], os.path.abspath(self.tools))

    def test_tools_dir_defaults_to_the_build_tree(self):
        rc, out, err = self.run_main("ps_w64_a_skip")
        self.assertEqual(rc, 0, (out, err))
        self.assertEqual(self.ran[0][0], os.path.abspath(os.path.join(ROOT, "runtime", "build")))
        self.assertEqual(self.ran[0][1], os.path.join(self.build, "spirv"))


if __name__ == "__main__":
    unittest.main()
