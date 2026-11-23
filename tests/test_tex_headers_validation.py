"""Synthetic input-validation tests for tools/tex_headers.py. No game data."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import tex_headers  # noqa: E402


class TestTexHeadersValidation(unittest.TestCase):
    def test_decode_etpf_chunk_pages_rejects_bool_and_non_int(self):
        for meta, extra in ((True, 0), (False, 0), (1, False), (1, True), (1.0, 0), ("1", 0)):
            with self.subTest(meta=meta, extra=extra):
                self.assertIsNone(tex_headers.decode_etpf_chunk_pages(meta, extra))
        self.assertEqual(tex_headers.decode_etpf_chunk_pages(1, 0), (1,))

    def test_compact_surface_layout_rejects_bool_and_non_int(self):
        for kw in (
            {"width": True, "height": 4, "mips": 1, "element_bytes": 4},
            {"width": 4, "height": True, "mips": 1, "element_bytes": 4},
            {"width": 4.0, "height": 4, "mips": 1, "element_bytes": 4},
            {"width": 4, "height": 4, "mips": True, "element_bytes": 4},
            {"width": 4, "height": 4, "mips": 1.0, "element_bytes": 4},
            {"width": 4, "height": 4, "mips": 1, "element_bytes": True},
            {"width": 4, "height": 4, "mips": 1, "element_bytes": 4, "block": True},
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    tex_headers.compact_surface_layout(**kw)

        layout = tex_headers.compact_surface_layout(4, 4, 1, 4, block=1)
        self.assertEqual(layout["mips"][0]["pixels_w"], 4)
        self.assertEqual(layout["mips"][0]["pixels_h"], 4)


if __name__ == "__main__":
    unittest.main()
