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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hashdict  # noqa: E402


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
        p = self.run_tool(self.carve, "--out", out_dir)
        self.assertEqual(p.returncode, 2)
        self.assertIn("is a directory", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_missing_extra_path_is_refused_cleanly(self):
        out = os.path.join(self.tmp.name, "out.json")
        missing = os.path.join(self.tmp.name, "missing_extra.txt")
        p = self.run_tool(self.carve, "--extra", missing, "--out", out)
        self.assertEqual(p.returncode, 2)
        self.assertIn("does not exist", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertFalse(os.path.exists(out))

    def test_extra_directory_respects_min_and_max_name_len(self):
        extra_dir = os.path.join(self.tmp.name, "extra")
        os.makedirs(extra_dir)
        long_stem = "a" * (hashdict.MAX_NAME_LEN + 5)
        for name in ("ab.txt", "abcde.txt", f"{long_stem}.txt"):
            with open(os.path.join(extra_dir, name), "wb") as f:
                f.write(b"")
        out = os.path.join(self.tmp.name, "out.json")
        p = self.run_tool(self.carve, "--extra", extra_dir, "--min", "4", "--out", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d.get(f"{hashdict.joaat('abcde'):08x}"), "abcde")
        self.assertNotIn(f"{hashdict.joaat('ab'):08x}", d)
        self.assertNotIn(f"{hashdict.joaat(long_stem):08x}", d)


if __name__ == "__main__":
    unittest.main()
