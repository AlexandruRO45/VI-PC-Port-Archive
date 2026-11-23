"""Bad carve indexes must give one ERROR line in gamedata_xml.py, never a traceback.

Run with ``python -I tests/test_gamedata_xml_bad_index.py``. Set ``LEONIDA_GAMEDATA_XML``
to another copy of the tool to use as a negative control (main's version fails these
tests). All inputs are hand-built; no game data is used.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_XML", str(ROOT / "tools" / "gamedata_xml.py")))
XML = (b'<?xml version="1.0" encoding="utf-8"?>\n'
       b'<CDataFileMgr__ContentChangeSetDef version="1"><id>fixture</id></CDataFileMgr__ContentChangeSetDef>')


class BadGamedataXmlIndexTest(unittest.TestCase):
    def run_tool(self, carve, extra=None):
        argv = [sys.executable, "-I", str(TOOL), str(carve)]
        if extra:
            argv += extra
        return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60)

    def make_carve(self, folder, doc=None, text=None, payload=None):
        carve = pathlib.Path(folder) / "x.rpf"
        carve.mkdir()
        index = carve / "index.json"
        if text is not None:
            index.write_text(text, encoding="utf-8")
        else:
            index.write_text(json.dumps(doc), encoding="utf-8")
        if payload is not None:
            (carve / "000000000000.bin").write_bytes(payload)
        return carve

    def assert_one_error(self, result):
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
        self.assertTrue(result.stderr.startswith("ERROR: cannot read "), result.stderr)

    def test_missing_files_key(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, {})))

    def test_row_missing_stored(self):
        with tempfile.TemporaryDirectory() as td:
            # The stream is present, so the run reaches the row's missing 'stored' rather than the missing file.
            doc = {"files": [{"off": 0, "size": len(XML), "head": XML[:16].hex()}]}
            result = self.run_tool(self.make_carve(td, doc, payload=XML))
            self.assert_one_error(result)
            self.assertIn("stored", result.stderr)

    def test_missing_stream_is_one_error_line(self):
        # An --index-only carve has rows but no .bin; the 16 index head bytes used to stand in for the document.
        with tempfile.TemporaryDirectory() as td:
            doc = {"files": [{"off": 0, "stored": len(XML), "size": len(XML), "head": XML[:16].hex()}]}
            result = self.run_tool(self.make_carve(td, doc))
            self.assertNotIn("Traceback", result.stderr, result.stderr)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
            self.assertTrue(result.stderr.startswith("ERROR: "), result.stderr)
            self.assertIn("000000000000.bin is missing", result.stderr)
            self.assertIn("--index-only", result.stderr)

    def test_stream_of_the_wrong_length_is_one_error_line(self):
        # A .bin is the inflated stream, its row's "size" bytes; a cut-off one used to be scanned as the document.
        with tempfile.TemporaryDirectory() as td:
            doc = {"files": [{"off": 0, "stored": len(XML), "size": len(XML), "head": XML[:16].hex()}]}
            carve = self.make_carve(td, doc, payload=XML[:50])
            result = self.run_tool(carve)
            self.assertNotIn("Traceback", result.stderr, result.stderr)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
            self.assertTrue(result.stderr.startswith(f"ERROR: {carve / '000000000000.bin'} is 50 bytes"),
                            result.stderr)
            self.assertIn(str(len(XML)), result.stderr)
            self.assertNotIn("IndexMismatch(", result.stderr)

    def test_missing_target_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "nope")
            result = self.run_tool(missing)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(result.stderr.strip(), f"ERROR: {missing} does not exist")

    def test_files_is_a_string(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, {"files": "oops"})))

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, text="{oops")))

    def test_valid_carve_still_reports(self):
        with tempfile.TemporaryDirectory() as td:
            doc = {"files": [{"off": 0, "stored": len(XML), "size": len(XML), "head": XML[:16].hex()}]}
            carve = self.make_carve(td, doc, payload=XML)
            result = self.run_tool(carve)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("CDataFileMgr__ContentChangeSetDef", result.stdout)


if __name__ == "__main__":
    unittest.main()
