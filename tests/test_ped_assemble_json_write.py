"""ped_assemble.py: an unwritable --json report path is one ERROR line and exit 2, not a traceback.

The report writer makes the parent folder and then opens the file, so a parent that is an existing file raises
FileExistsError past the `PedAssembleError` handler. The input is a hand-written OBJ triangle, so no game files.

    python -I tests/test_ped_assemble_json_write.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "ped_assemble.py")
TRIANGLE = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


class JsonWriteTest(unittest.TestCase):
    def test_unwritable_json_path(self):
        with tempfile.TemporaryDirectory() as d:
            part = os.path.join(d, "part.obj")
            with open(part, "w", encoding="utf-8", newline="\n") as f:
                f.write(TRIANGLE)
            blocker = os.path.join(d, "blocker")
            with open(blocker, "w", encoding="utf-8", newline="\n") as f:
                f.write("not a folder\n")
            report = os.path.join(blocker, "report.json")
            r = subprocess.run([sys.executable, "-I", TOOL, "assemble", part,
                                "-o", os.path.join(d, "out.glb"), "--json", report],
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("ERROR:", r.stderr)
        self.assertIn("blocker", r.stderr)


if __name__ == "__main__":
    unittest.main()
