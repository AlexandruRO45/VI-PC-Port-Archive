"""DDS file previews use only the first slice's top mip, without retaining the file.

    python -I tests/test_dds_png_file.py
"""
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dds_png import (  # noqa: E402
    DDPF_FOURCC, DDPF_RGB, bc1_block, bc3_block, bc4_block, dds, dds_buffer,
)


def rgba_array(slices):
    """A standard DX10 RGBA8 array header and its first 256x256 slice."""
    pixel = bytes((17, 33, 129, 255))
    header = bytearray(dds_buffer(256, 256, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=28,
                                 pitch=1024, mips=1, alpha_mode=1))
    struct.pack_into("<I", header, 8, 0x100F)  # caps, dimensions, format, pitch
    struct.pack_into("<I", header, 108, 0x1008 if slices > 1 else 0x1000)
    struct.pack_into("<I", header, 140, slices)
    return bytes(header), pixel * (256 * 256)


def encoded_cases():
    """Exercise every current compressed decoder and uncompressed byte layout."""
    blocks = {
        "BC1": bc1_block(0xF800, 0x07E0, [0, 85, 170, 255]),
        "BC2": bytes.fromhex("0123456789abcdef") + bc1_block(0xF800, 0x07E0, [0, 85, 170, 255]),
        "BC3": bc3_block(192, 64, 0x123456789ABC),
        "BC4": bc4_block(192, 64, 0x123456789ABC),
        "BC4_SNORM": bc4_block(127, 192, 0x123456789ABC),
        "BC5": bc4_block(192, 64, 0x123456789ABC) * 2,
        "BC5_SNORM": bc4_block(127, 192, 0x123456789ABC) * 2,
        "BC7": b"\x40" + bytes(15),  # valid mode 6
    }
    for fourcc in (b"DXT1", b"DXT2", b"DXT3", b"DXT4", b"DXT5", b"ATI1", b"BC4U", b"BC4S",
                   b"ATI2", b"BC5U", b"BC5S"):
        fmt = dds._FOURCC_MAP[fourcc]
        yield fourcc.decode("ascii"), dds_buffer(5, 5, blocks[fmt] * 4,
                                               flags=DDPF_FOURCC, fourcc=fourcc)
    for dxgi in (27, 28, 29, 87, 90, 91, 88, 92, 93, 26, 56, 65, 70, 71, 72, 73, 74, 75,
                 76, 77, 78, 79, 80, 81, 82, 83, 84, 97, 98, 99):
        fmt = dds._DXGI_MAP[dxgi]
        if fmt in blocks:
            payload, pitch = blocks[fmt] * 4, None
        else:
            bpp = {"R16_UNORM": 2, "A8_UNORM": 1}.get(fmt, 4)
            payload = (bytes((23, 51, 107, 192)) * 25)[:25 * bpp]
            pitch = 5 * bpp
        yield str(dxgi), dds_buffer(5, 5, payload, flags=DDPF_FOURCC, fourcc=b"DX10",
                                   dxgi=dxgi, pitch=pitch)
    for bits, masks in (
        (24, (0xFF0000, 0xFF00, 0xFF, 0)),
        (24, (0xFF, 0xFF00, 0xFF0000, 0)),
        (32, (0xFF, 0xFF00, 0xFF0000, 0)),
        (32, (0x3FF, 0xFFC00, 0x3FF00000, 0)),
    ):
        row = bytes(range(3 * (bits // 8)))
        payload = row + b"padding" + row[::-1]
        yield f"legacy-{bits}-{masks[0]}", dds_buffer(
            3, 2, payload, flags=DDPF_RGB, bit_count=bits, masks=masks, pitch=len(row) + 7)
    # DX10 masks, row padding, and alpha handling all remain decoder decisions.
    for dxgi in (24, 26, 28, 56, 65, 87):
        for alpha_mode in (1, 2, 3):
            bpp = {56: 2, 65: 1}.get(dxgi, 4)
            row = (bytes((23, 51, 107, 192)) * 3)[:3 * bpp]
            yield f"padded-{dxgi}-{alpha_mode}", dds_buffer(
                3, 2, row + b"padding" + row, flags=DDPF_FOURCC, fourcc=b"DX10",
                dxgi=dxgi, pitch=len(row) + 7, alpha_mode=alpha_mode)


class DdsFileTests(unittest.TestCase):
    def test_file_and_buffer_previews_match_for_supported_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.dds"
            for name, data in encoded_cases():
                with self.subTest(format=name):
                    path.write_bytes(data)
                    self.assertEqual(dds.dds_file_to_png_bytes(str(path)), dds.dds_to_png_bytes(data))
                    self.assertEqual(dds.dds_file_to_png_bytes(str(path), max_size=2),
                                     dds.dds_to_png_bytes(data, max_size=2))

    def test_array_preview_memory_does_not_grow_with_unused_slices(self):
        peaks = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.dds"
            for slices in (1, 64):
                header, first = rgba_array(slices)
                expected = dds.dds_to_png_bytes(header + first)
                with path.open("wb") as handle:
                    handle.write(header)
                    handle.write(first)
                    # Other slices have a different color and are never previewed.
                    other = bytes((241, 17, 33, 255)) * (256 * 256)
                    for _ in range(slices - 1):
                        handle.write(other)
                del first, other
                tracemalloc.start()
                try:
                    actual = dds.dds_file_to_png_bytes(str(path))
                    peaks[slices] = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertEqual(actual, expected)
                self.assertEqual(path.stat().st_size, 148 + slices * 256 * 256 * 4)
        self.assertLess(peaks[64], peaks[1] + 2 * 1024 * 1024,
                        f"unused array slices retained: peak bytes {peaks}")

    def test_file_mapping_is_released_before_success_returns(self):
        mappings = []
        real_mmap = mmap.mmap

        def record_mapping(*args, **kwargs):
            result = real_mmap(*args, **kwargs)
            mappings.append(result)
            return result

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mmap, "mmap", record_mapping):
            path = Path(tmp) / "input.dds"
            for name, data in encoded_cases():
                with self.subTest(format=name):
                    path.write_bytes(data)
                    result = dds.dds_file_to_png_bytes(str(path))
                    self.assertEqual(result, dds.dds_to_png_bytes(data))
                    self.assertTrue(all(item.closed for item in mappings))
                    path.unlink()  # Windows also refuses this while a file mapping remains open.

    def test_errors_preserve_diagnostics_and_release_maps_while_held(self):
        valid = dds_buffer(4, 4, b"\x11\x22\x33\xff" * 16, flags=DDPF_FOURCC,
                           fourcc=b"DX10", dxgi=28)
        cases = [("empty", b"", 0), ("short", b"DDS ", 0), ("short-dx10", valid[:140], 0),
                 ("negative-max-size", valid, -1),
                 ("unsupported", dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"XXXX"), 0),
                 ("bc1", dds_buffer(4, 4, b"1234", flags=DDPF_FOURCC, fourcc=b"DXT1"), 0)]
        for name, data in encoded_cases():
            cases.append((name, data[:-1], 0))
        for dxgi in (26, 28, 56, 65):
            cases.append((f"bad-pitch-{dxgi}", dds_buffer(4, 4, bytes(64), flags=DDPF_FOURCC,
                                                        fourcc=b"DX10", dxgi=dxgi, pitch=1), 0))
        real_mmap = mmap.mmap
        mappings, held_errors = [], []

        def record_mapping(*args, **kwargs):
            result = real_mmap(*args, **kwargs)
            mappings.append(result)
            return result

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mmap, "mmap", record_mapping):
            path = Path(tmp) / "input.dds"
            for name, data, max_size in cases:
                with self.subTest(format=name):
                    with self.assertRaises(dds.DdsError) as expected:
                        dds.dds_to_png_bytes(data, max_size=max_size)
                    path.write_bytes(data)
                    try:
                        dds.dds_file_to_png_bytes(str(path), max_size=max_size)
                    except dds.DdsError as error:
                        held_errors.append(error)  # Keep its traceback, including decoder frames, alive.
                        self.assertEqual(str(error), str(expected.exception))
                    else:
                        self.fail("invalid input was accepted")
                    self.assertTrue(all(item.closed for item in mappings))
                    path.unlink()
        self.assertEqual(len(held_errors), len(cases))

    def test_real_cli_previews_first_slice_without_changing_input(self):
        header, first = rgba_array(2)
        data = header + first + bytes(len(first))
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "array.dds", Path(tmp) / "preview.png"
            source.write_bytes(data)
            result = subprocess.run([sys.executable, "-I", dds.__file__, str(source), str(output),
                                     "--max-size", "2"], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(output.read_bytes(), dds.dds_to_png_bytes(header + first, max_size=2))
            self.assertEqual(source.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
