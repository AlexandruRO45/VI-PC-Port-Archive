"""Tests for DX10 alpha modes using synthetic DDS buffers (no game data).

    python3 -I tests/test_dds_png_dx10.py
"""
import importlib.util
import os
import struct
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()


class Dx10AlphaTests(unittest.TestCase):
    def test_rgba8_alpha_mode_in_png(self):
        header = bytearray(148)
        header[:4] = b"DDS "
        struct.pack_into("<IIIII", header, 4, 124, 0x100F, 1, 1, 4)
        struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
        struct.pack_into("<I", header, 108, 0x1000)  # DDSCAPS_TEXTURE
        struct.pack_into("<IIII", header, 128, 28, 3, 0, 1)  # RGBA8, 2D, one slice
        pixel = b"\xff\x00\x00\x00"

        for mode, alpha in ((3, 255), (1, 0)):  # Opaque and straight alpha
            with self.subTest(mode=mode):
                struct.pack_into("<I", header, 144, mode)
                png = dds.dds_to_png_bytes(bytes(header) + pixel)
                # The first chunk after IHDR contains the compressed row.
                pos = 8 + 12 + 13
                size, kind = struct.unpack_from(">I4s", png, pos)
                self.assertEqual(kind, b"IDAT")
                row = zlib.decompress(png[pos + 8:pos + 8 + size])
                self.assertEqual(row, bytes((0, 255, 0, 0, alpha)))


def _dx10(width, height, dxgi, payload, pitch=0):
    header = bytearray(148)
    header[:4] = b"DDS "
    struct.pack_into("<IIIII", header, 4, 124, 0x100F | (8 if pitch else 0), height, width, pitch)
    struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
    struct.pack_into("<IIII", header, 128, dxgi, 3, 0, 1)
    return bytes(header) + payload


class SingleChannelTests(unittest.TestCase):
    def test_r32_float_linear_values_with_padding(self):
        payload = (struct.pack("<4f", -1, 0, 0.5, 0.25) + b"skip"
                   + struct.pack("<4f", 2, float("inf"), float("-inf"), float("nan")))
        data = _dx10(4, 2, 41, payload, pitch=20)
        self.assertEqual(dds.parse_dds_header(data)["format"], "R32_FLOAT")
        width, height, rgba = dds.decode_dds(data)
        self.assertEqual((width, height), (4, 2))
        self.assertEqual(rgba, b"".join(bytes((r, 0, 0, 255)) for r in (0, 0, 128, 64, 255, 255, 0, 0)))

    def test_r16_unorm_red_opaque(self):
        px = struct.pack("<HHHH", 0, 65535, 32768, 257)
        _, _, rgba = dds.decode_dds(_dx10(2, 2, 56, px))
        self.assertEqual(rgba, bytes([0, 0, 0, 255, 255, 0, 0, 255, 128, 0, 0, 255, 1, 0, 0, 255]))

    def test_a8_unorm_alpha_only_with_pitch(self):
        payload = bytes([1, 2, 3, 99, 4, 5, 6, 99])  # 3x2, pitch 4
        _, _, rgba = dds.decode_dds(_dx10(3, 2, 65, payload, pitch=4))
        self.assertEqual(rgba[3::4], bytes([1, 2, 3, 4, 5, 6]))
        self.assertEqual(set(rgba[0::4]) | set(rgba[1::4]) | set(rgba[2::4]), {0})

    def test_truncated_rejected(self):
        for dxgi, n in ((56, 7), (65, 3), (41, 15), (11, 31)):
            with self.assertRaises(dds.DdsError):
                dds.decode_dds(_dx10(2, 2, dxgi, bytes(n)))


class Rgba16UnormTests(unittest.TestCase):
    def test_channels_and_alpha_with_padding(self):
        # 2x2 with a 20-byte pitch: 16 bytes of texels and 4 bytes of padding per row
        payload = (struct.pack("<8H", 0, 65535, 32768, 65535, 257, 128, 129, 0) + b"skip"
                   + struct.pack("<8H", 65535, 0, 0, 32767, 1, 2, 3, 65534) + b"skip")
        data = _dx10(2, 2, 11, payload, pitch=20)
        self.assertEqual(dds.parse_dds_header(data)["format"], "RGBA16_UNORM")
        width, height, rgba = dds.decode_dds(data)
        self.assertEqual((width, height), (2, 2))
        # 128 * 255 / 65535 is just under one half and 129 just over; alpha is kept, not forced opaque
        self.assertEqual(rgba, bytes((0, 255, 128, 255, 1, 0, 1, 0, 255, 0, 0, 127, 0, 0, 0, 255)))

    def test_every_code_rounds_like_r16_unorm(self):
        codes = list(range(65536))
        red = dds.decode_dds(_dx10(256, 256, 56, struct.pack("<65536H", *codes)))[2][0::4]
        rgba16 = dds.decode_dds(_dx10(256, 64, 11, struct.pack("<65536H", *codes)))[2]
        self.assertEqual(rgba16, red)

    def test_big_endian_path_matches(self):
        # The native 16-bit view only runs on little-endian hosts; the struct path must give the same bytes.
        tight = bytes(range(256)) * 4  # 4x16, pitch 64
        padded = b"".join(tight[y * 64:(y + 1) * 64] + b"pad!" for y in range(16))
        for payload, pitch in ((tight, 0), (padded, 68)):
            data = _dx10(8, 16, 11, payload, pitch=pitch)
            native = dds.decode_dds(data)
            with mock.patch.object(dds.sys, "byteorder", "big"):
                portable = dds.decode_dds(data)
            self.assertEqual(native, portable)

    def test_png_keeps_alpha(self):
        png = dds.dds_to_png_bytes(_dx10(1, 1, 11, struct.pack("<4H", 65535, 0, 0, 32768)))
        pos = 8 + 12 + 13
        size, kind = struct.unpack_from(">I4s", png, pos)
        self.assertEqual(kind, b"IDAT")
        self.assertEqual(zlib.decompress(png[pos + 8:pos + 8 + size]), bytes((0, 255, 0, 0, 128)))

    def test_short_pitch_rejected(self):
        with self.assertRaises(dds.DdsError):
            dds.decode_dds(_dx10(2, 1, 11, bytes(16), pitch=12))


if __name__ == "__main__":
    unittest.main()
