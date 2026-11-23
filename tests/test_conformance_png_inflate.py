"""Synthetic PNG inflation regressions; no renderer, GPU or game files."""
import contextlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import tracemalloc
import unittest
import zlib


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"
SPEC = importlib.util.spec_from_file_location("conformance_compare_inflate", SCRIPT)
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)


class InflateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def png(self, compressed, *, colour_type=2, width=1, height=1, split=False):
        header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
        parts = [compressed]
        if split:
            # Chunk boundaries can split both the zlib header and its checksum.
            parts = [b"", compressed[:1], compressed[1:-1], b"", compressed[-1:]]
        path = self.root / "synthetic.png"
        path.write_bytes(C.PNG_SIGNATURE + C.make_chunk(b"IHDR", header)
                         + b"".join(C.make_chunk(b"IDAT", part) for part in parts)
                         + C.make_chunk(b"IEND", b""))
        return path

    def test_excess_pixels_are_rejected_without_expanding_the_whole_payload(self):
        # Build before tracing: the reader must not materialize these 8 MiB for
        # a one-pixel image. A 1 MiB ceiling leaves room for zlib and PNG parsing.
        compressed = zlib.compress(b"\0" * (8 * 1024 * 1024))
        for colour_type in (2, 6):
            with self.subTest(colour_type=colour_type):
                path = self.png(compressed, colour_type=colour_type)
                tracemalloc.start()
                try:
                    with self.assertRaises(C.ToolError):
                        C.read_png(str(path))
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 1024 * 1024, f"PNG decode allocated {peak} bytes")

    def test_valid_rgb_and_rgba_with_split_idat_preserve_pixels(self):
        for colour_type, pixel in ((2, b"\x0a\x14\x1e"), (6, b"\x0a\x14\x1e\x28")):
            with self.subTest(colour_type=colour_type):
                rows = (b"\0" + pixel * 3) * 2
                path = self.png(zlib.compress(rows), colour_type=colour_type,
                                width=3, height=2, split=True)
                image = C.read_png(str(path))
                self.assertEqual((image.width, image.height), (3, 2))
                self.assertEqual(image.pixels, pixel * 6)

    def test_short_complete_stream_is_rejected(self):
        path = self.png(zlib.compress(b"\0\x0a\x14"))
        with self.assertRaisesRegex(C.ToolError, "expected 4"):
            C.read_png(str(path))

    def test_truncated_stream_is_rejected_even_after_all_pixels_arrive(self):
        compressed = zlib.compress(b"\0\x0a\x14\x1e")
        for removed in (1, 4, 6):
            with self.subTest(removed=removed):
                path = self.png(compressed[:-removed])
                with self.assertRaises(C.ToolError):
                    C.read_png(str(path))

    def test_bad_zlib_checksum_is_rejected(self):
        compressed = zlib.compress(b"\0\x0a\x14\x1e")
        path = self.png(compressed[:-1] + bytes([compressed[-1] ^ 1]))
        with self.assertRaisesRegex(C.ToolError, "does not decompress as zlib"):
            C.read_png(str(path))

    def test_huge_declared_dimensions_report_bad_length_without_overflow(self):
        # The expected scanline size exceeds a signed 64-bit max_length.
        path = self.png(zlib.compress(b"\0"), width=0x7fffffff, height=0x7fffffff)
        with self.assertRaisesRegex(C.ToolError, "expected"):
            C.read_png(str(path))

    def test_diff_reports_error_and_leaves_inputs_unchanged(self):
        image = self.png(zlib.compress(b"\0\x0a\x14\x1e\xff"))
        manifest = self.root / "manifest.json"
        C.write_selftest_manifest(str(manifest), "synthetic", "synthetic", [{
            "name": "scene", "png": image.name,
            "checks": [{"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255}],
        }])
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            rc = C.main(["diff", "--reference", str(manifest), "--candidate", str(manifest)])
        self.assertEqual(rc, C.EXIT_ERROR)
        self.assertIn("expected", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})


if __name__ == "__main__":
    unittest.main()
