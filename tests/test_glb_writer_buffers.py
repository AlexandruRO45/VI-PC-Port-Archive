"""Test external buffers in synthetic binary glTF containers."""
from __future__ import annotations

import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_writer


def _glb_from_doc(doc: dict, bin_payload: bytes = b"") -> bytes:
    json_bytes = json.dumps(doc).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    chunks = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    if bin_payload:
        pad = (-len(bin_payload)) % 4
        padded_bin = bin_payload + b"\x00" * pad
        chunks += struct.pack("<I4s", len(padded_bin), b"BIN\0") + padded_bin
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _external_buffer_glb(view_length: int) -> bytes:
    return _glb_from_doc({
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "sample.bin", "byteLength": 4}],
        "bufferViews": [{"buffer": 0, "byteLength": view_length}],
    })


class TestReadGlbSummaryBuffers(unittest.TestCase):
    def test_external_buffer_view_without_bin_chunk(self):
        summary = glb_writer.read_glb_summary(_external_buffer_glb(view_length=4))
        self.assertEqual(summary["bin_bytes"], 0)
        self.assertEqual(summary["json"]["bufferViews"][0]["byteLength"], 4)

    def test_external_buffer_view_exceeding_declared_length_rejected(self):
        with self.assertRaisesRegex(glb_writer.GlbError, r"bufferViews\[0\] exceeds buffer 0 \(0\+5 > 4\)"):
            glb_writer.read_glb_summary(_external_buffer_glb(view_length=5))

    def test_malformed_buffers_and_buffer_views_containers_rejected(self):
        for doc, pattern in (
            ({"asset": {"version": "2.0"}, "buffers": "bad"}, r"buffers must be a JSON array"),
            ({"asset": {"version": "2.0"}, "buffers": [123]}, r"buffers\[0\] is not an object"),
            ({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}, "bad"]}, r"buffers\[1\] is not an object"),
            ({"asset": {"version": "2.0"}, "bufferViews": "bad"}, r"bufferViews must be a JSON array"),
        ):
            with self.subTest(doc=doc):
                with self.assertRaisesRegex(glb_writer.GlbError, pattern):
                    glb_writer.read_glb_summary(_glb_from_doc(doc))

    def test_secondary_buffer_byte_length_validated(self):
        for extra_buf in (
            {"uri": "extra.bin"},
            {"uri": "extra.bin", "byteLength": -1},
            {"uri": "extra.bin", "byteLength": True},
            {"uri": "extra.bin", "byteLength": "4"},
        ):
            doc = {
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": 4}, extra_buf],
            }
            with self.subTest(extra_buf=extra_buf):
                with self.assertRaisesRegex(glb_writer.GlbError, r"invalid buffers\[1\]\.byteLength"):
                    glb_writer.read_glb_summary(_glb_from_doc(doc, bin_payload=b"\x00" * 4))

    def test_buffer_view_missing_or_empty_buffers_rejected(self):
        for doc in (
            {"asset": {"version": "2.0"}, "bufferViews": [{"buffer": 0, "byteLength": 4}]},
            {"asset": {"version": "2.0"}, "buffers": [], "bufferViews": [{"buffer": 0, "byteLength": 4}]},
        ):
            with self.subTest(doc=doc):
                with self.assertRaisesRegex(glb_writer.GlbError, r"bufferViews\[0\] references invalid buffer 0"):
                    glb_writer.read_glb_summary(_glb_from_doc(doc, bin_payload=b"\x00" * 4))

    def test_buffer_view_invalid_buffer_index_rejected(self):
        for bad_bv in (
            {"byteLength": 4},
            {"buffer": False, "byteLength": 4},
            {"buffer": True, "byteLength": 4},
            {"buffer": -1, "byteLength": 4},
            {"buffer": 1, "byteLength": 4},
            {"buffer": "0", "byteLength": 4},
        ):
            doc = {
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": 4}],
                "bufferViews": [bad_bv],
            }
            with self.subTest(bad_bv=bad_bv):
                with self.assertRaisesRegex(glb_writer.GlbError, r"bufferViews\[0\] references invalid buffer"):
                    glb_writer.read_glb_summary(_glb_from_doc(doc, bin_payload=b"\x00" * 4))

    def test_secondary_buffer_view_bounds_and_fields_validated(self):
        base_buffers = [{"byteLength": 4}, {"uri": "extra.bin", "byteLength": 8}]
        ok_doc = {
            "asset": {"version": "2.0"},
            "buffers": base_buffers,
            "bufferViews": [{"buffer": 1, "byteOffset": 2, "byteLength": 6}],
        }
        summary = glb_writer.read_glb_summary(_glb_from_doc(ok_doc, bin_payload=b"\x00" * 4))
        self.assertEqual(summary["bin_bytes"], 4)

        bad_bounds_doc = {
            "asset": {"version": "2.0"},
            "buffers": base_buffers,
            "bufferViews": [{"buffer": 1, "byteOffset": 4, "byteLength": 5}],
        }
        with self.assertRaisesRegex(glb_writer.GlbError, r"bufferViews\[0\] exceeds buffer 1 \(4\+5 > 8\)"):
            glb_writer.read_glb_summary(_glb_from_doc(bad_bounds_doc, bin_payload=b"\x00" * 4))

        for bad_bv in (
            {"buffer": 1, "byteOffset": -1, "byteLength": 4},
            {"buffer": 1, "byteOffset": 0, "byteLength": True},
            {"buffer": 1},
        ):
            bad_fields_doc = {
                "asset": {"version": "2.0"},
                "buffers": base_buffers,
                "bufferViews": [bad_bv],
            }
            with self.subTest(bad_bv=bad_bv):
                with self.assertRaisesRegex(glb_writer.GlbError, r"invalid byteOffset/byteLength in bufferViews\[0\]"):
                    glb_writer.read_glb_summary(_glb_from_doc(bad_fields_doc, bin_payload=b"\x00" * 4))


if __name__ == "__main__":
    unittest.main()
