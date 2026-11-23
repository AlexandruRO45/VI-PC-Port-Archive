"""An RPF9 archive is not an XML document: the catalog, reader and graph say so and exit 2.

The fixture is eight bytes: the on-disk ``9FPR`` tag and padding. No game data is used.

    python -I tests/test_gamedata_xml_archive_target.py

Set ``LEONIDA_GAMEDATA_XML``, ``LEONIDA_GAMEDATA_READER`` or ``LEONIDA_GAMEDATA_GRAPH`` to another copy of a
tool for a negative control (main's copies catalogue the archive and exit 0).
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = {
    "gamedata_xml": pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_XML",
                                                str(ROOT / "tools" / "gamedata_xml.py"))),
    "gamedata_reader": pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_READER",
                                                   str(ROOT / "tools" / "gamedata_reader.py"))),
    "gamedata_graph": pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_GRAPH",
                                                  str(ROOT / "tools" / "gamedata_graph.py"))),
}
XML = (b'<?xml version="1.0" encoding="utf-8"?>\n'
       b'<CDataFileMgr__ContentChangeSetDef version="1"><id>fixture</id></CDataFileMgr__ContentChangeSetDef>')


class ArchiveTargetTest(unittest.TestCase):
    def run_tool(self, name, target):
        return subprocess.run([sys.executable, "-I", str(TOOLS[name]), str(target)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_each_tool_names_an_rpf9_archive(self):
        with tempfile.TemporaryDirectory() as td:
            archive = pathlib.Path(td) / "common.rpf"
            archive.write_bytes(b"9FPR" + bytes(12))
            for name in TOOLS:
                with self.subTest(tool=name):
                    result = self.run_tool(name, archive)
                    self.assertNotIn("Traceback", result.stderr, result.stderr)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    lines = [line for line in result.stderr.strip().splitlines() if line]
                    self.assertEqual(len(lines), 1, result.stderr)
                    self.assertTrue(lines[0].startswith("ERROR: cannot read "), lines[0])
                    self.assertIn(str(archive), lines[0])
                    self.assertIn("an RPF9 archive, not an XML document", lines[0])
                    self.assertIn("rpf9_carve.py", lines[0])

    def test_a_real_xml_file_is_still_read(self):
        with tempfile.TemporaryDirectory() as td:
            document = pathlib.Path(td) / "document.xml"
            document.write_bytes(XML)
            for name in TOOLS:
                with self.subTest(tool=name):
                    result = self.run_tool(name, document)
                    self.assertNotIn("Traceback", result.stderr, result.stderr)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("document", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
