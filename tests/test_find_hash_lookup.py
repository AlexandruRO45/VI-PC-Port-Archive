"""tools/find_hash_lookup.py ranks the function a model hash is passed to (standard library only, no game data).

    python -I tests/test_find_hash_lookup.py

The images are synthetic PE files: one executable section holding `mov ecx, <hash>` followed by calls, with a
"lookup" called from many places and a decoy called once.
"""
import contextlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "find_hash_lookup.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import find_hash_lookup as fhl  # noqa: E402

TEXT_RVA, TEXT_RAW = 0x1000, 0x400


def pe(code):
    """A minimal PE32+ file whose only section, .text (executable), holds `code`."""
    header = bytearray(TEXT_RAW)
    header[0:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 0x40)
    header[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", header, 0x44, 0x8664, 1, 0, 0, 0, 0xF0, 0x22)
    first = 0x44 + 20 + 0xF0
    struct.pack_into("<8sIIIIIIHHI", header, first, b".text", len(code), TEXT_RVA, len(code), TEXT_RAW,
                     0, 0, 0, 0, 0x60000020)
    return bytes(header) + code


def call(at, target):
    """E8 rel32 placed at offset `at` of .text, calling .text offset `target`."""
    return b"\xE8" + struct.pack("<i", target - (at + 5))


def image():
    lucia, mandy = fhl.joaat("player_story000_00"), fhl.joaat("cs_mandy")
    code = bytearray(b"\xCC" * 0x400)
    lookup, decoy = 0x300, 0x380
    code[lookup] = code[decoy] = 0xC3
    # mov ecx, lucia ; call lookup   (at 0x00)
    code[0x00:0x05] = b"\xB9" + struct.pack("<I", lucia)
    code[0x05:0x0A] = call(0x05, lookup)
    # mov ecx, mandy ; nop ; call lookup   (at 0x20)
    code[0x20:0x25] = b"\xB9" + struct.pack("<I", mandy)
    code[0x25] = 0x90
    code[0x26:0x2B] = call(0x26, lookup)
    # mov ecx, lucia ; call decoy   (at 0x40): led by one hash, called once
    code[0x40:0x45] = b"\xB9" + struct.pack("<I", lucia)
    code[0x45:0x4A] = call(0x45, decoy)
    for k in range(20):                          # other callers of the lookup
        at = 0x100 + 8 * k
        code[at:at + 5] = call(at, lookup)
    return pe(bytes(code)), TEXT_RVA + lookup, TEXT_RVA + decoy


class FindHashLookupTest(unittest.TestCase):
    def test_joaat_matches_the_measured_model_ids(self):
        self.assertEqual(fhl.joaat("player_story000_00"), 0xE472B687)
        self.assertEqual(fhl.joaat("CS_Mandy"), 0xD7C0CDFF)
        self.assertEqual(fhl.parse_hash("0xD7C0CDFF"), 0xD7C0CDFF)
        with self.assertRaises(ValueError):
            fhl.parse_hash("0x123456789")

    def test_the_lookup_ranks_first(self):
        data, lookup, decoy = image()
        report = fhl.rank(data, [("player_story000_00", fhl.joaat("player_story000_00")),
                                 ("cs_mandy", fhl.joaat("cs_mandy"))])
        top = report["candidates"][0]
        self.assertEqual(int(top["rva"], 16), lookup)
        self.assertEqual(top["led_by"], ["cs_mandy", "player_story000_00"])
        self.assertEqual(top["direct_callers"], 22)
        self.assertEqual(int(report["candidates"][1]["rva"], 16), decoy)
        self.assertEqual({h["name"]: h["occurrences"] for h in report["hashes"]},
                         {"player_story000_00": 2, "cs_mandy": 1})

    def test_an_encrypted_file_is_refused_clearly(self):
        with self.assertRaisesRegex(ValueError, "encrypted"):
            fhl.rank(b"\x6f\x1b\xbc\xae" * 64, [("x", 1)])

    def test_cli_writes_the_report_and_fails_cleanly(self):
        data, lookup, _ = image()
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "image.exe")
            with open(exe, "wb") as f:
                f.write(data)
            out = os.path.join(tmp, "report.json")
            r = subprocess.run([sys.executable, "-I", TOOL, exe, "--json", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"0x{lookup:X}", r.stdout)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(int(json.load(f)["candidates"][0]["rva"], 16), lookup)
            bad = os.path.join(tmp, "packed.exe")
            with open(bad, "wb") as f:
                f.write(b"\x00" * 64)
            r = subprocess.run([sys.executable, "-I", TOOL, bad], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("ERROR", r.stderr)


class ReportPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.exe = os.path.join(self.dir, "fake.exe")
        data, _, _ = image()
        with open(self.exe, "wb") as f:
            f.write(data)

    def run_main(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            try:
                code = fhl.main(list(args))
            except SystemExit as e:
                return e.code, err.getvalue()
        return code, err.getvalue()

    def test_json_must_not_overwrite_the_input(self):
        with open(self.exe, "rb") as f:
            before = f.read()
        code, err = self.run_main(self.exe, "--json", self.exe)
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the input", err)
        with open(self.exe, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_json_creates_a_missing_folder(self):
        out = os.path.join(self.dir, "reports", "report.json")
        code, err = self.run_main(self.exe, "--json", out)
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.isfile(out))

    def test_json_refuses_a_folder(self):
        code, err = self.run_main(self.exe, "--json", self.dir)
        self.assertEqual(code, 2)
        self.assertIn("is a folder, not a file", err)

    def test_a_failed_write_is_one_line(self):
        blocker = os.path.join(self.dir, "not_a_folder")
        with open(blocker, "wb") as f:
            f.write(b"keep me")
        code, err = self.run_main(self.exe, "--json", os.path.join(blocker, "r.json"))
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot write", err)
        self.assertNotIn("Traceback", err)


class ReadOnlyReportTests(unittest.TestCase):
    def test_an_existing_read_only_report_is_refused_before_the_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "fake.exe")
            data, _, _ = image()
            with open(exe, "wb") as f:
                f.write(data)
            target = os.path.join(tmp, "report.json")
            with open(target, "w", encoding="utf-8") as f:
                f.write("previous report")
            os.chmod(target, stat.S_IREAD)
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    rc = fhl.main([exe, "--json", target])
            finally:
                os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
            self.assertEqual(rc, 2)
            self.assertIn("not writable", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
            with open(target, encoding="utf-8") as f:
                self.assertEqual(f.read(), "previous report")


if __name__ == "__main__":
    unittest.main()
