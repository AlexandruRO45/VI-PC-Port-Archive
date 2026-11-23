"""Regression tests for clean OSError handling when writing carved streams in tools/rpf9_carve.py.

    python -I tests/test_rpf9_carve_errors.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_carve.py")


class Rpf9CarveWriteErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write_archive(self, name, payload, base=16):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(b"9FPR" + struct.pack("<I", base) + b"\x00" * (base - 8) + payload)
        return path

    def test_unwritable_bin_stream_exits_without_traceback(self):
        zdata = zlib.compress(b"A" * 128)
        rpf = self._write_archive("bin.rpf", zdata, base=16)
        out_dir = os.path.join(self.tmp.name, "out_bin")
        os.makedirs(os.path.join(out_dir, f"{16:012x}.bin"))
        p = subprocess.run([sys.executable, "-I", TOOL, rpf, out_dir], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("cannot write", p.stderr)
        self.assertIn(f"{16:012x}.bin", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_unwritable_bk2_stream_exits_without_traceback(self):
        blen = 64
        bk2 = bytearray(blen)
        bk2[0:4] = b"KB2i"
        struct.pack_into("<II", bk2, 4, blen - 8, 10)
        struct.pack_into("<II", bk2, 0x14, 1280, 720)
        rpf = self._write_archive("bk2.rpf", bytes(bk2), base=16)
        out_dir = os.path.join(self.tmp.name, "out_bk2")
        os.makedirs(os.path.join(out_dir, f"{16:012x}.bk2"))
        p = subprocess.run([sys.executable, "-I", TOOL, rpf, out_dir], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("cannot write", p.stderr)
        self.assertIn(f"{16:012x}.bk2", p.stderr)
        self.assertNotIn("Traceback", p.stderr)


class Rpf9CarveExitCodes(unittest.TestCase):
    """A bad argument or an archive it cannot use is exit 2, the family's code for "could not run".

    tools/bvh_groups.py:22: "Exit status: 0 after a scan (whatever it found), 2 for a bad argument or an
    unreadable archive." rpf9_carve used to answer these with sys.exit("text"): one stderr line, but status 1.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _archive(self, name="tiny.rpf"):
        zdata = zlib.compress(b"A" * 128)
        data = bytearray(16 + len(zdata))
        data[0:4] = b"9FPR"
        struct.pack_into("<I", data, 4, 16)
        data[16:] = zdata
        return self._write(name, bytes(data))

    def _run(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def assert_cannot_run(self, *args):
        p = self._run(*args)
        self.assertEqual(p.returncode, 2, (p.stdout, p.stderr))
        self.assertEqual(p.stdout, "", p.stdout)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(len(p.stderr.splitlines()), 1, p.stderr)

    def test_an_unusable_archive_is_one_line_and_exit_2(self):
        out = os.path.join(self.tmp.name, "out")
        os.makedirs(os.path.join(self.tmp.name, "adirectory"))
        cases = (
            ("missing", os.path.join(self.tmp.name, "missing.rpf")),
            ("directory", os.path.join(self.tmp.name, "adirectory")),
            ("empty", self._write("empty.rpf", b"")),
            ("short", self._write("short.rpf", b"9FPR")),
            ("not an archive", self._write("wrong.rpf", b"XXXX" + b"\x00" * 12)),
            ("header past the end", self._write("base.rpf", b"9FPR" + struct.pack("<I", 999) + b"\x00" * 8)),
        )
        for name, archive in cases:
            with self.subTest(case=name):
                self.assert_cannot_run(archive, out)

    def test_an_output_it_cannot_create_is_one_line_and_exit_2(self):
        archive = self._archive()
        blocker = self._write("blocker", b"x")
        self.assert_cannot_run(archive, os.path.join(blocker, "out"))

    def test_a_bad_start_is_a_usage_error(self):
        archive = self._archive()
        size = os.path.getsize(archive)
        p = self._run(archive, os.path.join(self.tmp.name, "out"), "--start", "999")
        self.assertEqual(p.returncode, 2, (p.stdout, p.stderr))
        self.assertEqual(p.stdout, "")
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stderr.splitlines()[-1],
                         f"rpf9_carve.py: error: {archive}: --start 999 is outside the file (0..{size - 1})")

    def test_an_output_path_that_is_a_file_is_a_usage_error(self):
        archive = self._archive()
        out_file = self._write("out_file", b"")
        p = self._run(archive, out_file)
        self.assertEqual(p.returncode, 2, (p.stdout, p.stderr))
        self.assertEqual(p.stdout, "")
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stderr.splitlines()[-1],
                         f"rpf9_carve.py: error: {out_file}: output path is a file, not a directory")

    def test_a_valid_carve_still_exits_0(self):
        archive = self._archive()
        out = os.path.join(self.tmp.name, "out_ok")
        p = self._run(archive, out, "--index-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stderr, "")
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], 1, doc["summary"])


if __name__ == "__main__":
    unittest.main()
