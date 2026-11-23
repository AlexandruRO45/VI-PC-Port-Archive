"""tools/find_hash_lookup.py ranks the function a model hash is passed to (standard library only, no game data).

    python -I tests/test_find_hash_lookup.py

The images are synthetic PE files: one executable section holding `mov ecx, <hash>` followed by calls, with a
"lookup" called from many places and a decoy called once.
"""
import json
import os
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

    def test_non_contiguous_code_sections_ignore_non_executable_gaps(self):
        """E8 targets landing in .rdata between .text and .init must not count as candidates or callers."""
        lucia = fhl.joaat("player_story000_00")
        text_rva, rdata_rva, init_rva = 0x1000, 0x2000, 0x3000
        text = bytearray(b"\xCC" * 0x400)
        rdata = bytearray(b"\x00" * 0x400)
        init = bytearray(b"\xCC" * 0x200)
        rdata_target = rdata_rva + 0x100
        init_lookup = init_rva + 0x40
        init[0x40] = 0xC3
        # In .text: mov ecx, lucia; call rdata_target; call init_lookup
        text[0x00:0x05] = b"\xB9" + struct.pack("<I", lucia)
        text[0x05:0x0A] = b"\xE8" + struct.pack("<i", rdata_target - (text_rva + 0x05 + 5))
        text[0x0A:0x0F] = b"\xE8" + struct.pack("<i", init_lookup - (text_rva + 0x0A + 5))
        # Extra direct callers in .text to rdata_target (must be ignored) and in .init to init_lookup (must count)
        for k in range(5):
            at = 0x100 + 8 * k
            text[at:at + 5] = b"\xE8" + struct.pack("<i", rdata_target - (text_rva + at + 5))
        init[0x00:0x05] = b"\xE8" + struct.pack("<i", init_lookup - (init_rva + 0x00 + 5))

        raw_offset = TEXT_RAW
        specs = [
            (b".text", text_rva, bytes(text), 0x60000020),
            (b".rdata", rdata_rva, bytes(rdata), 0x40000040),
            (b".init", init_rva, bytes(init), 0x60000020),
        ]
        header = bytearray(TEXT_RAW)
        header[0:2] = b"MZ"
        struct.pack_into("<I", header, 0x3C, 0x40)
        header[0x40:0x44] = b"PE\0\0"
        struct.pack_into("<HHIIIHH", header, 0x44, 0x8664, len(specs), 0, 0, 0, 0xF0, 0x22)
        first = 0x44 + 20 + 0xF0
        body = bytearray()
        for idx, (name, rva, payload, flags) in enumerate(specs):
            raw = raw_offset + len(body)
            struct.pack_into("<8sIIIIIIHHI", header, first + 40 * idx, name, len(payload), rva, len(payload), raw,
                             0, 0, 0, 0, flags)
            body.extend(payload)
        data = bytes(header) + bytes(body)

        report = fhl.rank(data, [("player_story000_00", lucia)])
        rvas = [int(c["rva"], 16) for c in report["candidates"]]
        self.assertEqual(rvas, [init_lookup])
        self.assertNotIn(rdata_target, rvas)
        self.assertEqual(report["candidates"][0]["direct_callers"], 2)

    def test_duplicate_and_aliased_hashes_are_deduplicated(self):
        data, lookup, decoy = image()
        lucia = fhl.joaat("player_story000_00")
        mandy = fhl.joaat("cs_mandy")
        # Exact duplicate + hex/case aliases; first (name, value) order must be preserved without double-counting.
        # window=16 keeps decoy (at 0x45) outside cs_mandy's tail (0x25..0x35), so decoy is led only by lucia.
        report = fhl.rank(data, [
            ("cs_mandy", mandy),
            ("cs_mandy", mandy),
            ("0xD7C0CDFF", mandy),
            ("CS_Mandy", mandy),
            ("player_story000_00", lucia),
            ("0xE472B687", lucia),
        ], window=16)
        self.assertEqual(report["hashes"], [
            {"name": "cs_mandy", "value": "0xD7C0CDFF", "occurrences": 1},
            {"name": "player_story000_00", "value": "0xE472B687", "occurrences": 2},
        ])
        top = report["candidates"][0]
        self.assertEqual(int(top["rva"], 16), lookup)
        self.assertEqual(top["led_by"], ["cs_mandy", "player_story000_00"])
        self.assertEqual(int(report["candidates"][1]["rva"], 16), decoy)
        self.assertEqual(report["candidates"][1]["led_by"], ["player_story000_00"])

        # Order preservation when hex alias comes first.
        report2 = fhl.rank(data, [("0xD7C0CDFF", mandy), ("cs_mandy", mandy)], window=16)
        self.assertEqual(report2["hashes"], [{"name": "0xD7C0CDFF", "value": "0xD7C0CDFF", "occurrences": 1}])
        self.assertEqual(report2["candidates"][0]["led_by"], ["0xD7C0CDFF"])

    def test_parse_hash_rejects_empty_and_malformed_hex(self):
        for bad in ("", "0x", "0X", "0x_1", "0x1_2", "0x-1", "0x+1", "0x123g", "0x123456789"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    fhl.parse_hash(bad)

    def test_negative_window_and_top_rejected_and_top_zero_allowed(self):
        data, _, _ = image()
        hashes = [("cs_mandy", fhl.joaat("cs_mandy"))]
        with self.assertRaises(ValueError):
            fhl.rank(data, hashes, window=-1)
        with self.assertRaises(ValueError):
            fhl.rank(data, hashes, top=-1)
        zero = fhl.rank(data, hashes, top=0)
        self.assertEqual(zero["candidates"], [])
        self.assertEqual(zero["hashes"][0]["occurrences"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "image.exe")
            with open(exe, "wb") as f:
                f.write(data)
            for args in (["--window", "-1"], ["--top", "-1"], ["--hash", ""], ["--hash", "0x_1"]):
                with self.subTest(args=args):
                    r = subprocess.run([sys.executable, "-I", TOOL, exe, *args], capture_output=True, text=True)
                    self.assertEqual(r.returncode, 2)
                    self.assertIn("ERROR:", r.stderr)
                    self.assertEqual(r.stdout, "")
            r = subprocess.run([sys.executable, "-I", TOOL, exe, "--top", "0"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("no direct call follows", r.stdout)

    def test_cli_unwritable_json_fails_cleanly_before_stdout(self):
        data, _, _ = image()
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "image.exe")
            with open(exe, "wb") as f:
                f.write(data)
            for bad_json in (os.path.join(tmp, "missing_dir", "report.json"), tmp):
                with self.subTest(bad_json=bad_json):
                    r = subprocess.run([sys.executable, "-I", TOOL, exe, "--json", bad_json],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, 2)
                    self.assertIn("ERROR:", r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertEqual(r.stdout, "")


if __name__ == "__main__":
    unittest.main()
