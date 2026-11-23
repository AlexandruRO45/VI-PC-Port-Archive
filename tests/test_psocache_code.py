"""CPU test for tools/psocache_code.py on a synthetic cache built by this test, not the game's.

The cache has the measured region layout with empty PSO, meta and meta-index regions and three shader-index
entries over a hand-written code blob, each a 36-byte non-code header followed by:

- entry 0: s_nop, s_load_dword (an 8-byte SMEM instruction), s_endpgm, then s_code_end filler;
- entry 1: s_nop, s_endpgm, s_nop, s_endpgm (two programs in one entry), then one zero word;
- entry 2: s_nop, 0xFFFFFFFF (which the decoder rejects), s_endpgm.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psocache_code.py")

S_NOP = 0xBF800000
S_ENDPGM = 0xBF810000
S_CODE_END = 0xBF9F0000
S_LOAD_DWORD = (0xF4000000, 0x00000000)
BAD = 0xFFFFFFFF

ENTRIES = [
    [S_NOP, *S_LOAD_DWORD, S_ENDPGM, S_CODE_END, S_CODE_END],
    [S_NOP, S_ENDPGM, S_NOP, S_ENDPGM, 0],
    [S_NOP, BAD, S_ENDPGM],
]


HEADER = bytes([0xFF]) * 36  # stands in for the 36-byte entry header; 0xFFFFFFFF does not decode


def build(entries=ENTRIES, header_bytes=HEADER):
    code = b""
    index = b""
    for words in entries:
        blob = header_bytes + struct.pack("<%dI" % len(words), *words)
        index += struct.pack("<2I", len(code), len(blob))
        code += blob
    header = bytearray(0x80)
    header[0:4] = b"PS\x00\x09"
    struct.pack_into("<4I", header, 0x50, 0, 0, 0, len(entries))
    struct.pack_into("<2I", header, 0x60, 0, len(code))
    return bytes(header) + index + code


def run(path, *args):
    return subprocess.run([sys.executable, "-I", TOOL, path, *args], capture_output=True, text=True)


class PsocacheCodeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.cache")
        with open(self.path, "wb") as f:
            f.write(build())

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, *args):
        p = run(self.path, "--json", *args)
        self.assertIn(p.returncode, (0, 1), p.stderr)
        return p.returncode, json.loads(p.stdout)

    def test_counts(self):
        rc, r = self.report()
        self.assertEqual(rc, 1)  # entry 2 has an invalid word
        self.assertEqual(r["entries"], 3)
        self.assertEqual(r["entries_clean"], 2)
        self.assertEqual(r["entries_with_invalid"], 1)
        self.assertEqual(r["invalid_words"], 1)
        self.assertEqual(r["programs"], 4)
        # 3 + 4 + 2 decoded instructions; the 8-byte s_load_dword counts once
        self.assertEqual(r["instructions"], 9)
        self.assertEqual(r["tail_words"], {"s_code_end": 2, "zero": 1, "other": 0})
        self.assertEqual(r["families"], {"s": 9})

    def test_limit_clean(self):
        rc, r = self.report("--limit", "2")
        self.assertEqual(rc, 0)
        self.assertEqual(r["entries"], 2)
        self.assertEqual(r["invalid_words"], 0)

    def test_no_endpgm(self):
        with open(self.path, "wb") as f:
            f.write(build([[S_NOP, S_CODE_END]]))
        rc, r = self.report()
        self.assertEqual(rc, 0)
        self.assertEqual(r["entries_without_endpgm"], 1)
        self.assertEqual(r["instructions"], 0)
        self.assertEqual(r["tail_words"]["other"], 1)

    def test_text_report_has_no_bytes(self):
        p = run(self.path)
        self.assertEqual(p.returncode, 1)
        self.assertIn("invalid words: 1", p.stdout)
        self.assertNotIn("bf81", p.stdout.lower())

    def test_not_a_cache(self):
        with open(self.path, "wb") as f:
            f.write(b"\0" * 0x100)
        p = run(self.path)
        self.assertEqual(p.returncode, 2)
        self.assertIn("magic", p.stderr)

    def test_header_zero_decodes_the_header(self):
        rc, r = self.report("--header", "0")
        self.assertEqual(rc, 1)
        self.assertEqual(r["invalid_words"], 1 + 3 * 9)  # entry 2 plus nine header words per entry

    def test_short_entry(self):
        with open(self.path, "wb") as f:
            f.write(build([[]], HEADER[:32]))  # an entry of 32 bytes, shorter than the header
        rc, r = self.report()
        self.assertEqual(r["entries_shorter_than_header"], 1)

    def test_bad_header(self):
        self.assertEqual(run(self.path, "--header", "3").returncode, 2)

    def test_bad_limit(self):
        p = run(self.path, "--limit", "0")
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
