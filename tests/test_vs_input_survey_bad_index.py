"""A missing or malformed index.json gives vs_input_survey.py one ERROR line and exit 2, never a traceback.

Run with ``python -I tests/test_vs_input_survey_bad_index.py``. Every input is built here; no game data.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "scripts" / "vs_input_survey.py"


class BadVsSurveyIndexTest(unittest.TestCase):
    def run_tool(self, dxbc):
        return subprocess.run([sys.executable, "-I", str(TOOL), str(dxbc)], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)

    def make_dxbc(self, folder, doc=None, text=None):
        dxbc = pathlib.Path(folder) / "dxbc"
        dxbc.mkdir()
        (dxbc / "index.json").write_text(text if text is not None else json.dumps(doc), encoding="utf-8")
        return dxbc

    def assert_one_error(self, result):
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
        self.assertTrue(result.stderr.startswith("ERROR: cannot read "), result.stderr)

    def test_missing_index(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(pathlib.Path(td) / "nowhere"))

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_dxbc(td, text="{oops")))

    def test_index_is_an_object(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_dxbc(td, {"name": "a_vs"})))

    def test_vertex_entry_without_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_dxbc(td, [{"name": "a_vs", "stage": "vs"}])))

    def test_entry_is_not_an_object(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_dxbc(td, ["a_vs"])))

    def test_empty_index_still_reports(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.run_tool(self.make_dxbc(td, []))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("0 vertex shaders", result.stdout)


if __name__ == "__main__":
    unittest.main()
