"""Tests for tile_probe: synthetic tiled buffer decompression, block size calculations, and channel layouts.

    python -I tests/test_tile_probe.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "tile_probe.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import tile_probe as T  # noqa: E402

try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def make_gradient_blocks(wb, hb, bpe=8):
    """Generate deterministic smooth 2D gradient blocks for linear or tiled probing."""
    n = wb * hb
    x_coords = np.arange(wb, dtype=np.uint32) * 31 // wb
    y_coords = np.arange(hb, dtype=np.uint32) * 63 // hb
    gx, gy = np.meshgrid(x_coords, y_coords)
    c = (gx.ravel() << 11) | (gy.ravel() << 5)

    blocks = np.zeros((n, bpe), dtype=np.uint8)
    if bpe == 8:
        blocks[:, 0] = c & 0xFF
        blocks[:, 1] = (c >> 8) & 0xFF
    else:
        blocks[:, 8] = c & 0xFF
        blocks[:, 9] = (c >> 8) & 0xFF
        blocks[:, 0] = (gx.ravel() * 255 // 31).astype(np.uint8)
    return blocks


class TestMortonDecode(unittest.TestCase):
    def test_morton_decode_powers_of_two(self):
        # 1-bit base inputs
        x0, y0 = T.morton_decode(np.array(0), 1)
        self.assertEqual((int(x0), int(y0)), (0, 0))

        x1, y1 = T.morton_decode(np.array(1), 1)
        self.assertEqual((int(x1), int(y1)), (1, 0))

        x2, y2 = T.morton_decode(np.array(2), 1)
        self.assertEqual((int(x2), int(y2)), (0, 1))

        x3, y3 = T.morton_decode(np.array(3), 1)
        self.assertEqual((int(x3), int(y3)), (1, 1))

        # Higher bit bases
        x4, y4 = T.morton_decode(np.array(4), 2)
        self.assertEqual((int(x4), int(y4)), (2, 0))

        x8, y8 = T.morton_decode(np.array(8), 3)
        self.assertEqual((int(x8), int(y8)), (0, 2))

        x16, y16 = T.morton_decode(np.array(16), 3)
        self.assertEqual((int(x16), int(y16)), (4, 0))

    def test_morton_decode_vectorized_sequence(self):
        indices = np.arange(8)
        x, y = T.morton_decode(indices, 2)
        expected_x = np.array([0, 1, 0, 1, 2, 3, 2, 3])
        expected_y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)

    def test_morton_decode_roundtrip_reconstruction(self):
        bits = 4
        indices = np.arange(1 << (2 * bits), dtype=np.int64)
        x, y = T.morton_decode(indices, bits)

        reconstructed = np.zeros_like(indices)
        for b in range(bits):
            reconstructed |= ((x >> b) & 1) << (2 * b)
            reconstructed |= ((y >> b) & 1) << (2 * b + 1)
        np.testing.assert_array_equal(reconstructed, indices)


class TestBlockSizeAndTileDims(unittest.TestCase):
    def test_tile_dims_4k_and_64k_known_configurations(self):
        # 4 KB tile, 8 bytes per block (e.g. BC1): 512 blocks, 9 bits (odd -> tw=32, th=16)
        tw_4k_8, th_4k_8, bits_4k_8 = T.tile_dims(4096, 8)
        self.assertEqual((tw_4k_8, th_4k_8, bits_4k_8), (32, 16, 9))
        self.assertEqual(tw_4k_8 * th_4k_8, 512)

        # 4 KB tile, 16 bytes per block (e.g. BC3/BC7): 256 blocks, 8 bits (even -> tw=16, th=16)
        tw_4k_16, th_4k_16, bits_4k_16 = T.tile_dims(4096, 16)
        self.assertEqual((tw_4k_16, th_4k_16, bits_4k_16), (16, 16, 8))
        self.assertEqual(tw_4k_16 * th_4k_16, 256)

        # 64 KB tile, 8 bytes per block: 8192 blocks, 13 bits (odd -> tw=128, th=64)
        tw_64k_8, th_64k_8, bits_64k_8 = T.tile_dims(65536, 8)
        self.assertEqual((tw_64k_8, th_64k_8, bits_64k_8), (128, 64, 13))
        self.assertEqual(tw_64k_8 * th_64k_8, 8192)

        # 64 KB tile, 16 bytes per block: 4096 blocks, 12 bits (even -> tw=64, th=64)
        tw_64k_16, th_64k_16, bits_64k_16 = T.tile_dims(65536, 16)
        self.assertEqual((tw_64k_16, th_64k_16, bits_64k_16), (64, 64, 12))
        self.assertEqual(tw_64k_16 * th_64k_16, 4096)

    def test_tile_dims_invariants(self):
        for tile_bytes in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
            for bpe in (8, 16):
                tw, th, bits = T.tile_dims(tile_bytes, bpe)
                n = tile_bytes // bpe
                self.assertEqual(tw * th, n)
                self.assertEqual(bits, int(np.log2(n)))
                self.assertGreaterEqual(tw, th)
                if bits % 2 == 0:
                    self.assertEqual(tw, th)
                else:
                    self.assertEqual(tw, 2 * th)


class TestChannelLayoutAndBlockFeature(unittest.TestCase):
    def test_block_feature_bpe8_rgb565_primary_colors(self):
        # 5 blocks: black, white, pure red, pure green, pure blue
        blocks = np.zeros((5, 8), dtype=np.uint8)

        # White: 0xFFFF
        blocks[1, 0] = 0xFF
        blocks[1, 1] = 0xFF

        # Red: 0xF800 (bits 11..15 = 31) -> byte 0: 0x00, byte 1: 0xF8
        blocks[2, 0] = 0x00
        blocks[2, 1] = 0xF8

        # Green: 0x07E0 (bits 5..10 = 63) -> byte 0: 0xE0, byte 1: 0x07
        blocks[3, 0] = 0xE0
        blocks[3, 1] = 0x07

        # Blue: 0x001F (bits 0..4 = 31) -> byte 0: 0x1F, byte 1: 0x00
        blocks[4, 0] = 0x1F
        blocks[4, 1] = 0x00

        # Non-endpoint bytes 2..7 set to non-zero values to ensure they do not contaminate endpoints
        blocks[:, 2:] = 0xA5

        feat = T.block_feature(blocks, 8)
        self.assertEqual(feat.shape, (5, 3))
        self.assertEqual(feat.dtype, np.float32)

        np.testing.assert_array_equal(feat[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(feat[1], [255.0, 255.0, 255.0])
        np.testing.assert_array_equal(feat[2], [255.0, 0.0, 0.0])
        np.testing.assert_array_equal(feat[3], [0.0, 255.0, 0.0])
        np.testing.assert_array_equal(feat[4], [0.0, 0.0, 255.0])

    def test_block_feature_bpe8_intermediate_values(self):
        # R=10/31, G=20/63, B=30/31
        c = (10 << 11) | (20 << 5) | 30
        blocks = np.zeros((1, 8), dtype=np.uint8)
        blocks[0, 0] = c & 0xFF
        blocks[0, 1] = (c >> 8) & 0xFF

        feat = T.block_feature(blocks, 8)
        expected_r = float(10 * 255 // 31)
        expected_g = float(20 * 255 // 63)
        expected_b = float(30 * 255 // 31)
        np.testing.assert_array_equal(feat[0], [expected_r, expected_g, expected_b])

    def test_block_feature_bpe16_mix_and_endpoints(self):
        blocks = np.zeros((4, 16), dtype=np.uint8)

        # White 565 at bytes 8-9 (0xFFFF), byte 0 = 255 -> r = (255 + 255) // 2 = 255
        blocks[0, 8] = 0xFF
        blocks[0, 9] = 0xFF
        blocks[0, 0] = 255

        # White 565 at bytes 8-9, byte 0 = 0 -> r = (255 + 0) // 2 = 127
        blocks[1, 8] = 0xFF
        blocks[1, 9] = 0xFF
        blocks[1, 0] = 0

        # Black 565 at bytes 8-9, byte 0 = 100 -> r = (0 + 100) // 2 = 50, g = 0, b = 0
        blocks[2, 8] = 0x00
        blocks[2, 9] = 0x00
        blocks[2, 0] = 100

        # Green 565 (0x07E0) at bytes 8-9, byte 0 = 60 -> r = (0 + 60) // 2 = 30, g = 255, b = 0
        blocks[3, 8] = 0xE0
        blocks[3, 9] = 0x07
        blocks[3, 0] = 60

        # Other bytes (1..7, 10..15) set to junk values to verify independence
        blocks[:, 1:8] = 0x5A
        blocks[:, 10:] = 0xC3

        feat = T.block_feature(blocks, 16)
        self.assertEqual(feat.shape, (4, 3))
        self.assertEqual(feat.dtype, np.float32)

        np.testing.assert_array_equal(feat[0], [255.0, 255.0, 255.0])
        np.testing.assert_array_equal(feat[1], [127.0, 255.0, 255.0])
        np.testing.assert_array_equal(feat[2], [50.0, 0.0, 0.0])
        np.testing.assert_array_equal(feat[3], [30.0, 255.0, 0.0])


class TestBlockLayoutMapping(unittest.TestCase):
    def test_layout_linear(self):
        pos = T.layout(512, 32, "linear", 8)
        np.testing.assert_array_equal(pos, np.arange(512))

    def test_layout_invalid_dimensions_return_none(self):
        # 4k tile with bpe=8 has tw=32, th=16
        # Width not divisible by tw
        self.assertIsNone(T.layout(512, 30, "4k", 8))
        self.assertIsNone(T.layout(512, 33, "4k", 8))

        # Total blocks not divisible by (wblocks * th)
        self.assertIsNone(T.layout(32 * 15, 32, "4k", 8))
        self.assertIsNone(T.layout(32 * 17, 32, "4k", 8))

        # Valid dimensions must succeed
        self.assertIsNotNone(T.layout(32 * 16, 32, "4k", 8))

        # 64k tile with bpe=8 has tw=128, th=64
        self.assertIsNone(T.layout(128 * 64, 64, "64k", 8))
        self.assertIsNone(T.layout(128 * 60, 128, "64k", 8))
        self.assertIsNotNone(T.layout(128 * 64, 128, "64k", 8))

    def test_layout_tiled_permutation_properties(self):
        # 4k tile, 8 bpe: tw=32, th=16, 2x2 tiles = 64x32 blocks = 2048 blocks
        pos_4k_8 = T.layout(2048, 64, "4k", 8)
        self.assertEqual(len(pos_4k_8), 2048)
        self.assertEqual(len(np.unique(pos_4k_8)), 2048)
        self.assertEqual(pos_4k_8.min(), 0)
        self.assertEqual(pos_4k_8.max(), 2047)
        np.testing.assert_array_equal(np.sort(pos_4k_8), np.arange(2048))

        # 4k tile, 16 bpe: tw=16, th=16, 2x2 tiles = 32x32 blocks = 1024 blocks
        pos_4k_16 = T.layout(1024, 32, "4k", 16)
        self.assertEqual(len(pos_4k_16), 1024)
        np.testing.assert_array_equal(np.sort(pos_4k_16), np.arange(1024))

        # 64k tile, 8 bpe: tw=128, th=64, 1x1 tile = 8192 blocks
        pos_64k_8 = T.layout(8192, 128, "64k", 8)
        self.assertEqual(len(pos_64k_8), 8192)
        np.testing.assert_array_equal(np.sort(pos_64k_8), np.arange(8192))

    def test_layout_spatial_tile_and_morton_coordinates(self):
        wb, th, tw = 64, 16, 32
        pos = T.layout(wb * th, wb, "4k", 8)

        # Block 0 -> (x=0, y=0)
        self.assertEqual(pos[0], 0)
        # Block 1 -> (x=1, y=0)
        self.assertEqual(pos[1], 1)
        # Block 2 -> (x=0, y=1) -> pos = 1 * wb + 0
        self.assertEqual(pos[2], wb)
        # Block 3 -> (x=1, y=1) -> pos = 1 * wb + 1
        self.assertEqual(pos[3], wb + 1)

        # First tile blocks (indices 0..511) stay within tile 0 boundaries [0, 32) x [0, 16)
        tile0_indices = pos[:512]
        x_tile0 = tile0_indices % wb
        y_tile0 = tile0_indices // wb
        self.assertTrue((x_tile0 < tw).all())
        self.assertTrue((y_tile0 < th).all())

        # Second tile blocks (indices 512..1023) stay within tile 1 boundaries [32, 64) x [0, 16)
        tile1_indices = pos[512:1024]
        x_tile1 = tile1_indices % wb
        y_tile1 = tile1_indices // wb
        self.assertTrue(((x_tile1 >= tw) & (x_tile1 < 2 * tw)).all())
        self.assertTrue((y_tile1 < th).all())


class TestImageScore(unittest.TestCase):
    def test_score_uniform_image(self):
        img = np.full((32, 32, 3), 128.0, dtype=np.float32)
        self.assertEqual(T.score(img), 0.0)

    def test_score_gradient_significantly_lower_than_noise(self):
        x = np.linspace(0, 255, 32, dtype=np.float32)
        y = np.linspace(0, 255, 32, dtype=np.float32)
        gx, gy = np.meshgrid(x, y)
        smooth = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)

        rng = np.random.RandomState(20261121)
        scrambled = smooth.copy().reshape(-1, 3)
        rng.shuffle(scrambled)
        scrambled = scrambled.reshape(32, 32, 3)

        s_smooth = T.score(smooth)
        s_scrambled = T.score(scrambled)
        self.assertLess(s_smooth, 20.0)
        self.assertGreater(s_scrambled, 80.0)
        self.assertLess(s_smooth, s_scrambled)


class TestSyntheticBufferDecompression(unittest.TestCase):
    def test_probe_linear_gradient_blob(self):
        wb, hb = 64, 64
        blocks = make_gradient_blocks(wb, hb, bpe=8)
        res = T.probe(blocks.tobytes())

        self.assertGreater(len(res), 0)
        top_score, top_bpe, top_wpx, top_mode = res[0]
        self.assertEqual(top_bpe, 8)
        self.assertEqual(top_wpx, 256)
        self.assertEqual(top_mode, "linear")

    def test_probe_4k_tiled_bpe8_buffer(self):
        wb, hb = 64, 64
        n = wb * hb
        linear_blocks = make_gradient_blocks(wb, hb, bpe=8)

        pos = T.layout(n, wb, "4k", 8)
        self.assertIsNotNone(pos)
        tiled_blocks = linear_blocks[pos]

        res = T.probe(tiled_blocks.tobytes())
        self.assertGreater(len(res), 0)
        top_score, top_bpe, top_wpx, top_mode = res[0]
        self.assertEqual(top_bpe, 8)
        self.assertEqual(top_wpx, 256)
        self.assertEqual(top_mode, "4k")

    def test_probe_4k_tiled_bpe16_buffer(self):
        wb, hb = 64, 64
        n = wb * hb
        linear_blocks = make_gradient_blocks(wb, hb, bpe=16)

        pos = T.layout(n, wb, "4k", 16)
        self.assertIsNotNone(pos)
        tiled_blocks = linear_blocks[pos]

        res = T.probe(tiled_blocks.tobytes())
        self.assertGreater(len(res), 0)
        top_score, top_bpe, top_wpx, top_mode = res[0]
        self.assertEqual(top_bpe, 16)
        self.assertEqual(top_wpx, 256)
        self.assertEqual(top_mode, "4k")

    def test_probe_64k_tiled_bpe8_buffer(self):
        wb, hb = 128, 64
        n = wb * hb
        linear_blocks = make_gradient_blocks(wb, hb, bpe=8)

        pos = T.layout(n, wb, "64k", 8)
        self.assertIsNotNone(pos)
        tiled_blocks = linear_blocks[pos]

        res = T.probe(tiled_blocks.tobytes())
        self.assertGreater(len(res), 0)
        top_score, top_bpe, top_wpx, top_mode = res[0]
        self.assertEqual(top_bpe, 8)
        self.assertEqual(top_wpx, 512)
        self.assertEqual(top_mode, "64k")

    def test_probe_with_byte_offset(self):
        wb, hb = 64, 64
        n = wb * hb
        linear_blocks = make_gradient_blocks(wb, hb, bpe=8)
        pos = T.layout(n, wb, "4k", 8)
        tiled_bytes = linear_blocks[pos].tobytes()

        pad = b"PROBE_HEADER_OFFSET_FIXTURE_PADDING_1234"
        padded_blob = pad + tiled_bytes

        res = T.probe(padded_blob, offset=len(pad))
        self.assertGreater(len(res), 0)
        top_score, top_bpe, top_wpx, top_mode = res[0]
        self.assertEqual(top_bpe, 8)
        self.assertEqual(top_wpx, 256)
        self.assertEqual(top_mode, "4k")


class TestRenderAndCli(unittest.TestCase):
    @unittest.skipUnless(HAVE_PIL, "Pillow is not installed")
    def test_render_bc1_and_bc7(self):
        with tempfile.TemporaryDirectory() as td:
            # BC1 (bpe=8)
            blocks8 = make_gradient_blocks(64, 64, bpe=8)
            out_bc1 = os.path.join(td, "preview_bc1.png")
            size_bc1 = T.render(blocks8.tobytes(), 0, 8, 256, "linear", out_bc1)
            self.assertEqual(size_bc1, (256, 256))
            self.assertTrue(os.path.exists(out_bc1))
            self.assertGreater(os.path.getsize(out_bc1), 0)

            # BC7 (bpe=16)
            blocks16 = make_gradient_blocks(64, 64, bpe=16)
            out_bc7 = os.path.join(td, "preview_bc7.png")
            size_bc7 = T.render(blocks16.tobytes(), 0, 16, 256, "linear", out_bc7)
            self.assertEqual(size_bc7, (256, 256))
            self.assertTrue(os.path.exists(out_bc7))
            self.assertGreater(os.path.getsize(out_bc7), 0)

    @unittest.skipUnless(HAVE_PIL, "Pillow is not installed")
    def test_cli_execution_with_out(self):
        wb, hb = 64, 64
        blocks = make_gradient_blocks(wb, hb, bpe=8)
        pos = T.layout(wb * hb, wb, "4k", 8)
        tiled_bytes = blocks[pos].tobytes()

        with tempfile.TemporaryDirectory() as td:
            blob_path = os.path.join(td, "texture.bin")
            out_png = os.path.join(td, "preview.png")
            with open(blob_path, "wb") as f:
                f.write(tiled_bytes)

            p = subprocess.run([sys.executable, "-I", TOOL, blob_path, "--out", out_png],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("score", p.stdout)
            self.assertIn("8-byte blocks", p.stdout)
            self.assertIn("4k", p.stdout)
            self.assertIn("preview (256, 256)", p.stdout)
            self.assertTrue(os.path.exists(out_png))
            self.assertGreater(os.path.getsize(out_png), 0)


if __name__ == "__main__":
    unittest.main()
