"""w64_graphics.py: a requested fixture name that does not exist stops the run instead of being dropped.

    python -I tests/test_w64_graphics_names.py -v
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import w64_graphics


class NameSelection(unittest.TestCase):
    def run_main(self, *args):
        """(exit code, stdout, stderr, fixtures checked) of main(), with the GPU checks stubbed to `ok`."""
        checked = []

        def fake_check(tools, spirv, name):
            checked.append(name)
            return ""

        with tempfile.TemporaryDirectory() as build:
            out, err = io.StringIO(), io.StringIO()
            with patch.object(w64_graphics, "check", side_effect=fake_check):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = w64_graphics.main([build, *args])
        return code, out.getvalue(), err.getvalue(), checked

    def test_unknown_name_next_to_a_valid_one_errors(self):
        code, out, err, checked = self.run_main("ps_w64_a_skip", "ps_w64_c_lane_id_typo")
        self.assertEqual(code, 1)
        self.assertIn("ps_w64_c_lane_id_typo", err)
        self.assertEqual(checked, [])          # the valid name must not pass for the whole run

    def test_all_unknown_names_do_not_run_every_fixture(self):
        code, out, err, checked = self.run_main("vs_w64_typo")
        self.assertEqual(code, 1)
        self.assertIn("vs_w64_typo", err)
        self.assertEqual(checked, [])

    def test_valid_selection_runs_exactly_those_fixtures(self):
        code, out, err, checked = self.run_main("vs_w64_a_skip", "gs_w64_a_ring")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(checked, ["vs_w64_a_skip", "gs_w64_a_ring"])
        for name in checked:
            self.assertIn(name, out)

    def test_no_names_still_runs_every_fixture(self):
        code, out, err, checked = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(checked, sorted(w64_graphics.FIXTURES))

    def test_a_tools_dir_argument_is_not_an_unknown_name(self):
        with tempfile.TemporaryDirectory() as tools:
            code, out, err, checked = self.run_main(tools, "ps_w64_a_skip")
            self.assertEqual((code, checked), (0, ["ps_w64_a_skip"]))
            code, out, err, checked = self.run_main(tools, "ps_w64_c_lane_id_typo")
            self.assertEqual(code, 1)
            self.assertEqual(checked, [])
            self.assertNotIn(os.path.basename(tools), err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
