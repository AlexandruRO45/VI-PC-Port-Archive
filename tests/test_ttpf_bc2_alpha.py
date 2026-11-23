"""Check explicit BC2 alpha through TTPF export and DDS decoding.

    python -I tests/test_ttpf_bc2_alpha.py

Set LEONIDA_GAME_DIR to an Xbox Series 1.0.2.1 package to also check the
two retail BC2_UNORM chains in place. Only aggregate counts are printed.
"""
from contextlib import closing
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ttpf_mips import pack_etpf_record, retile_to, ttpf, write_rds5_pool

import dds_png


BC2_UNORM = 74
RETAIL_CASES = (
    ("data/ttdb/maps/ttdb_maps_1_7.ttpf", 12656, (1024, 1024, 11)),
    ("data/ttdb/maps/ttdb_maps_2_7.ttpf", 14863, (256, 512, 9)),
)


def explicit_alpha(payload, width, height):
    """Read low nibble first in each 4x4 block, ignoring padding texels."""
    block_columns = (width + 3) // 4
    for y in range(height):
        block_row, row = divmod(y, 4)
        for x in range(width):
            block_col, col = divmod(x, 4)
            pair = payload[16 * (block_row * block_columns + block_col) + 2 * row + col // 2]
            yield ((pair >> (4 * (col % 2))) & 15) * 17


def synthetic_pool(path, width, height, mip_count):
    layout = ttpf.surface_layout(width, height, BC2_UNORM, mip_count, 4096)
    tiled = bytearray(layout["size"])
    expected = []
    for level, mip in enumerate(layout["mips"]):
        w, h = mip["pixels_w"], mip["pixels_h"]
        # Give the axes and each mip distinct alpha patterns. Padding is
        # opaque so accidentally copying it also changes visible alpha.
        plane = bytes(17 * ((3 * x + 5 * y + 7 * level) % 16)
                      for y in range(h) for x in range(w))
        expected.append(plane)
        blocks = bytearray()
        for ty in range(mip["blocks_h"]):
            for tx in range(mip["blocks_w"]):
                alpha = 0
                for index in range(16):
                    x, y = 4 * tx + index % 4, 4 * ty + index // 4
                    nibble = plane[y * w + x] // 17 if x < w and y < h else 15
                    alpha |= nibble << (4 * index)
                blocks.extend(struct.pack("<QHHI", alpha, 0xFFFF, 0, 0))
        retile_to(layout, mip, blocks, tiled)
    record = pack_etpf_record(width, height, mip_count, 0, BC2_UNORM, 1, 1, 2, 4096)
    write_rds5_pool(path, [record], [tiled])
    return expected


class Bc2AlphaTest(unittest.TestCase):
    def check_chain(self, data, shape, expected_planes=None):
        info = dds_png.parse_dds_header(data)
        self.assertEqual((info["width"], info["height"], info["mip_count"]), shape)
        self.assertEqual(info["dxgi_format"], BC2_UNORM)
        self.assertEqual(struct.unpack_from("<III", data, 132), (3, 0, 1))
        offset = info["data_offset"]
        header = bytearray(data[:offset])
        width, height, mip_count = shape
        clipped = pixels = 0
        for level in range(mip_count):
            w, h = max(1, width >> level), max(1, height >> level)
            size = ((w + 3) // 4) * ((h + 3) // 4) * 16
            payload = data[offset:offset + size]
            self.assertEqual(len(payload), size, f"mip {level} payload length")
            # Expose this mip as level zero to the public DDS decoder,
            # preserving the original DX10 format and alpha mode fields.
            struct.pack_into("<III", header, 12, h, w, size)
            struct.pack_into("<I", header, 28, 1)
            out_w, out_h, rgba = dds_png.decode_dds(bytes(header) + payload)
            self.assertEqual((out_w, out_h, len(rgba)), (w, h, w * h * 4))
            actual = rgba[3::4]
            mismatches = sum(a != b for a, b in zip(actual, explicit_alpha(payload, w, h)))
            self.assertEqual(mismatches, 0, f"mip {level} ({w}x{h}) alpha mismatches")
            if expected_planes is not None:
                self.assertEqual(actual, expected_planes[level], f"synthetic mip {level}")
            offset += size
            pixels += w * h
            clipped += bool(w % 4 or h % 4)
        self.assertEqual(offset, len(data), "DDS chain size")
        return mip_count, clipped, pixels

    def test_synthetic_chain_preserves_alpha_at_clipped_edges(self):
        shape = (5, 7, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-bc2.ttpf"
            expected = synthetic_pool(path, *shape)
            with closing(ttpf.Pool(path)) as pool:
                data = ttpf.texture_dds(pool, ttpf.analyze(pool, 0))
        self.assertEqual(self.check_chain(data, shape, expected), (3, 3, 42))

    @unittest.skipUnless(os.environ.get("LEONIDA_GAME_DIR"), "set LEONIDA_GAME_DIR for retail BC2")
    def test_retail_alpha_at_every_exported_mip(self):
        root = Path(os.environ["LEONIDA_GAME_DIR"])
        mips = clipped = pixels = 0
        for relative, index, shape in RETAIL_CASES:
            with self.subTest(index=index):
                with closing(ttpf.Pool(root / relative)) as pool:
                    data = ttpf.texture_dds(pool, ttpf.analyze(pool, index))
                n_mips, n_clipped, n_pixels = self.check_chain(data, shape)
                mips += n_mips
                clipped += n_clipped
                pixels += n_pixels
        print(f"Retail BC2 alpha: {mips} mips, {clipped} clipped, {pixels} samples checked")


if __name__ == "__main__":
    unittest.main()
