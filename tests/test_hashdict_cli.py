"""CLI validation regression tests for tools/hashdict.py.

    python -I tests/test_hashdict_cli.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "hashdict.py")


class HashdictCliValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.carve = os.path.join(self.tmp.name, "carve")
        os.makedirs(self.carve)
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"files": []}, f)

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_lookup_rejects_out_of_range_u32_hashes(self):
        table = os.path.join(self.tmp.name, "hashes.json")
        with open(table, "w", encoding="utf-8") as f:
            json.dump({"deadbeef": "alpha"}, f)
        for bad in ("-1", "0x100000000", "100000000"):
            with self.subTest(bad=bad):
                p = self.run_tool("--lookup", table, bad)
                self.assertEqual(p.returncode, 2)
                self.assertIn("32-bit hash range", p.stderr)
                self.assertEqual(p.stdout, "")

    def test_out_directory_is_refused_before_scanning(self):
        out_dir = os.path.join(self.tmp.name, "out_dir")
        os.makedirs(out_dir)
        for bad in (out_dir, os.path.join(self.tmp.name, "newdir") + os.sep):
            with self.subTest(bad=bad):
                p = self.run_tool(self.carve, "--out", bad)
                self.assertEqual(p.returncode, 2)
                self.assertIn("is a directory", p.stderr)
                self.assertEqual(p.stdout, "")
                self.assertNotIn("Traceback", p.stderr)

    def test_out_with_file_parent_is_refused_before_scanning(self):
        parent_file = os.path.join(self.tmp.name, "not_a_dir")
        with open(parent_file, "w", encoding="utf-8") as f:
            f.write("x")
        p = self.run_tool(self.carve, "--out", os.path.join(parent_file, "hashes.json"))
        self.assertEqual(p.returncode, 2)
        self.assertIn("is not a directory", p.stderr)
        self.assertEqual(p.stdout, "")
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
