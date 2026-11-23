"""Synthetic DXBC magic and XBCH header-length validation tests for tools/fssm_unpack.py. No game data."""
from __future__ import annotations

import os
import struct
import sys
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import fssm_unpack  # noqa: E402


def _make_dxbc(chunks: list[tuple[bytes, bytes]]) -> bytes:
    count = len(chunks)
    head_len = 32 + 4 * count
    offsets = []
    body = bytearray()
    for fourcc, payload in chunks:
        offsets.append(head_len + len(body))
        body.extend(fourcc)
        body.extend(struct.pack("<I", len(payload)))
        body.extend(payload)
    total_size = head_len + len(body)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, total_size, count)
    offset_table = b"".join(struct.pack("<I", off) for off in offsets)
    return header + offset_table + bytes(body)


def _make_xdat(entries: list[tuple[int, bytes]]) -> bytes:
    count = len(entries)
    table_len = 12 + 12 * count
    table_entries = []
    payloads = bytearray()
    for tag, payload in entries:
        off = table_len + len(payloads)
        table_entries.append(struct.pack("<III", tag, off, len(payload)))
        payloads.extend(payload)
    return b"XDAT" + struct.pack("<II", count, 8) + b"".join(table_entries) + bytes(payloads)


class TestFssmUnpackHeaderValidation(unittest.TestCase):
    def test_parts_rejects_non_dxbc_magic(self):
        for magic, count in ((b"NOPE", 0), (b"\x00\x00\x00\x00", 0), (b"FSSM", 1)):
            blob = magic + bytes(24) + struct.pack("<I", count)
            with self.subTest(magic=magic, count=count):
                with self.assertRaisesRegex(ValueError, r"invalid DXBC magic"):
                    fssm_unpack.parts(blob)

    def test_parts_accepts_valid_dxbc(self):
        self.assertEqual(fssm_unpack.parts(_make_dxbc([])), {})
        self.assertEqual(fssm_unpack.parts(_make_dxbc([(b"DXIL", b"payload")])), {b"DXIL": b"payload"})

    def test_inflate_xbch_rejects_truncated_header(self):
        for size in (4, 8, 12, 16, 20, 23):
            blob = (b"XBCH" + bytes(20))[:size]
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, r"^truncated XBCH header$"):
                    fssm_unpack._inflate_xbch(blob)

    def test_inflate_xbch_valid_roundtrip(self):
        fssm = b"FSSM" + struct.pack("<H", 0x0101) + bytes(10)
        comp = zlib.compress(fssm)
        xbch = struct.pack("<4sIIIII", b"XBCH", 0, len(comp), 0, len(fssm), 0) + comp
        self.assertEqual(fssm_unpack._inflate_xbch(xbch), fssm)

    def test_unpack_and_unpack_all_propagate_header_errors(self):
        dxbc_bad_magic = b"NOPE" + bytes(28)
        with self.assertRaisesRegex(ValueError, r"invalid DXBC magic"):
            fssm_unpack.unpack(dxbc_bad_magic)
        with self.assertRaisesRegex(ValueError, r"invalid DXBC magic"):
            fssm_unpack.unpack_all(dxbc_bad_magic)

        short_xbch = b"XBCH" + bytes(10)
        dxbc_short_xbch = _make_dxbc([(b"DXIL", _make_xdat([(0x0D, short_xbch)]))])
        with self.assertRaisesRegex(ValueError, r"^truncated XBCH header$"):
            fssm_unpack.unpack(dxbc_short_xbch)
        with self.assertRaisesRegex(ValueError, r"^truncated XBCH header$"):
            fssm_unpack.unpack_all(dxbc_short_xbch)


if __name__ == "__main__":
    unittest.main()
