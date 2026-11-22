"""Tests for tools/tile_probe.py using synthetic data (no game files).

    python -I tests/test_tile_probe.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "tile_probe.py")
    spec = importlib.util.spec_from_file_location("tile_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Try to load numpy and the module; skip if not available
try:
    import numpy as np
    tile_probe = load_module()
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


def bc1_block(c0, c1, rows):
    """One BC1 block; rows holds the four per-row index bytes."""
    return struct.pack("<HHI", c0, c1, int.from_bytes(bytes(rows), "little"))


def make_bc1_blob(width_blocks, height_blocks):
    """Create a synthetic BC1 blob with a smooth gradient."""
    blocks = []
    for by in range(height_blocks):
        for bx in range(width_blocks):
            # Smooth gradient: color varies with block position
            c0 = (bx * 31 // max(1, width_blocks - 1)) << 11 | (by * 63 // max(1, height_blocks - 1)) << 5 | 16
            c1 = c0 + 1
            rows = bytes([(bx + by * 4) & 3 for _ in range(4)])
            blocks.append(bc1_block(c0, c1, rows))
    return b"".join(blocks)


def make_bc3_blob(width_blocks, height_blocks):
    """Create a synthetic BC3 blob with a smooth gradient."""
    blocks = []
    for by in range(height_blocks):
        for bx in range(width_blocks):
            alpha0 = bx * 255 // max(1, width_blocks - 1)
            alpha1 = by * 255 // max(1, height_blocks - 1)
            alpha_indices = 0
            for i in range(16):
                alpha_indices |= ((bx + by + i) & 7) << (i * 3)
            c0 = 0xFFFF
            c1 = 0x0000
            color_bits = 0
            blocks.append(struct.pack("<BB", alpha0, alpha1) + alpha_indices.to_bytes(6, "little") + struct.pack("<HHI", c0, c1, color_bits))
    return b"".join(blocks)


@unittest.skipUnless(HAS_NUMPY, "numpy not available; tile_probe needs it")
class TileProbeFunctionTests(unittest.TestCase):
    def test_morton_decode(self):
        # morton_decode(0, 2) -> (0, 0)
        x, y = tile_probe.morton_decode(np.array([0]), 2)
        self.assertEqual(x[0], 0)
        self.assertEqual(y[0], 0)
        # morton_decode(1, 2) -> (1, 0) (bit 0 -> x)
        x, y = tile_probe.morton_decode(np.array([1]), 2)
        self.assertEqual(x[0], 1)
        self.assertEqual(y[0], 0)
        # morton_decode(2, 2) -> (0, 1) (bit 1 -> y)
        x, y = tile_probe.morton_decode(np.array([2]), 2)
        self.assertEqual(x[0], 0)
        self.assertEqual(y[0], 1)
        # morton_decode(3, 2) -> (1, 1)
        x, y = tile_probe.morton_decode(np.array([3]), 2)
        self.assertEqual(x[0], 1)
        self.assertEqual(y[0], 1)

    def test_tile_dims(self):
        # 4K tile, 8-byte blocks -> 512 blocks -> 9 bits -> 16x32
        tw, th, bits = tile_probe.tile_dims(4096, 8)
        self.assertEqual(bits, 9)
        self.assertEqual(tw, 16)
        self.assertEqual(th, 32)
        # 64K tile, 16-byte blocks -> 4096 blocks -> 12 bits -> 64x64
        tw, th, bits = tile_probe.tile_dims(65536, 16)
        self.assertEqual(bits, 12)
        self.assertEqual(tw, 64)
        self.assertEqual(th, 64)

    def test_block_feature_bc1(self):
        # Two BC1 blocks
        blocks = np.array([
            [0x00, 0x7C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],  # c0=0x7C00 (max red)
            [0xFF, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],  # c0=0x03FF (max blue)
        ], dtype=np.uint8)
        feat = tile_probe.block_feature(blocks, 8)
        self.assertEqual(feat.shape, (2, 3))
        # First block: max red -> r=255, g=0, b=0
        self.assertAlmostEqual(feat[0, 0], 255, delta=1)
        self.assertAlmostEqual(feat[0, 1], 0, delta=1)
        self.assertAlmostEqual(feat[0, 2], 0, delta=1)
        # Second block: max blue -> r=0, g=0, b=255
        self.assertAlmostEqual(feat[1, 0], 0, delta=1)
        self.assertAlmostEqual(feat[1, 1], 0, delta=1)
        self.assertAlmostEqual(feat[1, 2], 255, delta=1)

    def test_layout_linear(self):
        # 4 blocks, 2 wide, linear
        pos = tile_probe.layout(4, 2, "linear", 8)
        self.assertIsNotNone(pos)
        # block 0 -> (0,0) -> pos 0
        # block 1 -> (1,0) -> pos 1
        # block 2 -> (0,1) -> pos 2
        # block 3 -> (1,1) -> pos 3
        self.assertTrue(np.array_equal(pos, np.array([0, 1, 2, 3])))

    def test_layout_4k(self):
        # 4K tile, 8-byte blocks: 16x32 tile, 512 blocks per tile
        # Use 1024 blocks (2 tiles), 32 blocks wide
        pos = tile_probe.layout(1024, 32, "4k", 8)
        self.assertIsNotNone(pos)
        self.assertEqual(len(pos), 1024)

    def test_layout_64k(self):
        # 64K tile, 16-byte blocks: 64x64 tile, 4096 blocks per tile
        # Use 8192 blocks (2 tiles), 128 blocks wide
        pos = tile_probe.layout(8192, 128, "64k", 16)
        self.assertIsNotNone(pos)
        self.assertEqual(len(pos), 8192)

    def test_layout_invalid_width(self):
        # Width not multiple of tile width should return None
        # 4K tile with 8-byte blocks needs width multiple of 16
        pos = tile_probe.layout(512, 15, "4k", 8)
        self.assertIsNone(pos)

    def test_layout_invalid_count(self):
        # Block count not multiple of rows_per_tile * width should return None
        pos = tile_probe.layout(513, 16, "4k", 8)
        self.assertIsNone(pos)

    def test_score(self):
        # Perfectly smooth image -> score 0
        img = np.ones((4, 4, 3), dtype=np.float32) * 100
        s = tile_probe.score(img)
        self.assertAlmostEqual(s, 0.0, places=5)
        # Checkerboard -> high score
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[::2, ::2] = 255
        s = tile_probe.score(img)
        self.assertGreater(s, 0)

    def test_probe_bc1(self):
        blob = make_bc1_blob(16, 16)
        best = tile_probe.probe(blob)
        self.assertGreater(len(best), 0)
        # Best should be linear or 4k with low score
        s, bpe, wpx, mode = best[0]
        self.assertIn(bpe, (8, 16))
        self.assertIn(mode, ("linear", "4k", "64k"))

    def test_probe_bc3(self):
        blob = make_bc3_blob(16, 16)
        best = tile_probe.probe(blob)
        self.assertGreater(len(best), 0)
        s, bpe, wpx, mode = best[0]
        self.assertIn(bpe, (8, 16))
        self.assertIn(mode, ("linear", "4k", "64k"))


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TileProbeCLITests(unittest.TestCase):
    def run_tool(self, args, input_data=None):
        """Run tile_probe.py as a subprocess."""
        cmd = [sys.executable, "-I", os.path.join(ROOT, "tools", "tile_probe.py")] + args
        result = subprocess.run(cmd, input=input_data, capture_output=True, timeout=30)
        return result.returncode, result.stdout.decode(), result.stderr.decode()

    def test_missing_input(self):
        """Missing input file should exit with usage error, no traceback."""
        code, out, err = self.run_tool([])
        self.assertEqual(code, 2)  # argparse usage error
        self.assertIn("usage:", err.lower())
        self.assertNotIn("Traceback", err)

    def test_nonexistent_file(self):
        """Nonexistent file should exit cleanly with error message."""
        code, out, err = self.run_tool(["/nonexistent/path.bin"])
        self.assertEqual(code, 1)
        self.assertIn("No such file", err)
        self.assertNotIn("Traceback", err)

    def test_negative_offset(self):
        """Negative offset should exit with error."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x00" * 1024)
            path = f.name
        try:
            code, out, err = self.run_tool([path, "--offset", "-1"])
            self.assertEqual(code, 1)
            self.assertIn("offset", err.lower())
            self.assertNotIn("Traceback", err)
        finally:
            os.unlink(path)

    def test_offset_past_end(self):
        """Offset past end of file should exit with error."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x00" * 1024)
            path = f.name
        try:
            code, out, err = self.run_tool([path, "--offset", "2048"])
            self.assertEqual(code, 1)
            self.assertIn("offset", err.lower())
            self.assertNotIn("Traceback", err)
        finally:
            os.unlink(path)

    def test_good_bc1_input(self):
        """Synthetic BC1 blob should produce scored output."""
        blob = make_bc1_blob(16, 16)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(blob)
            path = f.name
        try:
            code, out, err = self.run_tool([path])
            self.assertEqual(code, 0)
            self.assertIn("score", out)
            self.assertIn("8-byte", out)
            self.assertNotIn("Traceback", err)
        finally:
            os.unlink(path)

    def test_good_bc3_input(self):
        """Synthetic BC3 blob should produce scored output."""
        blob = make_bc3_blob(16, 16)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(blob)
            path = f.name
        try:
            code, out, err = self.run_tool([path])
            self.assertEqual(code, 0)
            self.assertIn("score", out)
            self.assertNotIn("Traceback", err)
        finally:
            os.unlink(path)

    def test_out_preview(self):
        """--out should write a preview image."""
        blob = make_bc1_blob(16, 16)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(blob)
            path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            code, out, err = self.run_tool([path, "--out", out_path])
            self.assertEqual(code, 0)
            self.assertIn("preview", out)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
        finally:
            os.unlink(path)
            if os.path.exists(out_path):
                os.unlink(out_path)


if __name__ == "__main__":
    unittest.main()