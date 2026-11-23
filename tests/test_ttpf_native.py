"""The optional Rust kernels (tools/ttpf_native) against the Python code in tools/ttpf_mips.py.

Synthetic random surfaces and blocks only (no game data). Every comparison is on exact bytes, or
on the IEEE bits of every double, so -0.0 and 0.0 differ. The comparisons skip when the library
is not built:

    cargo build --release --locked --offline --manifest-path tools/ttpf_native/Cargo.toml
    python -I tests/test_ttpf_native.py
"""
import importlib.util
import os
import random
import struct
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from tex_headers import volume_surface_layout  # noqa: E402

spec = importlib.util.spec_from_file_location("ttpf_mips_native", os.path.join(ROOT, "tools", "ttpf_mips.py"))
ttpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttpf)
NATIVE = ttpf._native()

BC_FORMATS = {71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 83: 16, 95: 16, 96: 16, 98: 16, 99: 16}
SIZES = ((1, 1), (1, 37), (37, 1), (2, 3), (13, 7), (64, 64))


def python_only():
    """Run the Python code: the library reads as absent."""
    return mock.patch.object(ttpf, "_NATIVE", [None])


def bits(pixels):
    return b"".join(struct.pack("<3d", *p) for p in pixels)


def random_blocks(rng, fmt, count):
    """`count` random blocks of `fmt`, with the mode selector of each block walked through
    every value (BC6H mode codes, BC7 modes and the reserved one, both BC1 colour orders)."""
    size = BC_FORMATS[fmt]
    data = bytearray(rng.randbytes(count * size))
    for i in range(count):
        at = i * size
        if fmt in (95, 96):
            data[at] = (data[at] & 0xE0) | (i & 31)
        elif fmt in (98, 99):
            m = i % 9
            data[at] = 0 if m == 8 else ((1 << m) | (data[at] << (m + 1))) & 0xFF
        elif fmt in (71, 72, 74, 75, 77, 78) and i % 3 == 1:  # c0 == c1 (three-colour BC1)
            colour = at + (8 if size == 16 else 0)
            data[colour:colour + 2] = data[colour + 2:colour + 4]
        elif fmt == 80 and i % 2:  # e0 <= e1 (six-level BC4)
            data[at], data[at + 1] = sorted((data[at], data[at + 1]))
    return bytes(data)


@unittest.skipIf(NATIVE is None, "tools/ttpf_native is not built")
class DetileMatchesPython(unittest.TestCase):
    def assert_same(self, fn, *args):
        native = fn(*args)
        with python_only():
            expected = fn(*args)
        self.assertEqual(native, expected)
        return native

    def test_2d_mip_chains_every_element_size_and_tile(self):
        rng = random.Random(0x7A1)
        # DXGI 61/49/28 (1/2/4-byte elements), 10 and 2 (8/16-byte texels), BC1 and BC7 blocks.
        for fmt in (61, 49, 28, 10, 2, 71, 98):
            for tile in (4096, 65536):
                for width, height in ((1, 1), (3, 700), (1023, 5), (17, 13), (256, 256), (512, 64)):
                    block = 4 if fmt in ttpf.BC_FORMATS else 1
                    mips = max(1, min(6, (max(width, height) // block).bit_length()))
                    try:
                        layout = ttpf.surface_layout(width, height, fmt, mips, tile)
                    except ttpf.LayoutError:
                        continue
                    tiled = rng.randbytes(layout["size"])
                    for mip in layout["mips"]:
                        with self.subTest(fmt=fmt, tile=tile, size=(width, height), mip=mip["index"]):
                            region = tiled[mip["offset"]:mip["offset"] + mip["region"]]
                            self.assert_same(ttpf.detile, layout, mip, region)
                            self.assert_same(ttpf.detile, layout, mip, memoryview(region))

    def test_short_allocation_keeps_the_python_result(self):
        # The native copy refuses an element past the end; the Python code then gives its own
        # (shorter) result for that input, unchanged.
        layout = ttpf.surface_layout(64, 64, 71, 1, 4096)
        mip = layout["mips"][0]
        tiled = random.Random(5).randbytes(layout["size"])
        for cut in (1, 7, 100, layout["size"] // 2):
            with self.subTest(cut=cut):
                self.assert_same(ttpf.detile, layout, mip, tiled[:-cut])

    def test_volume_slices(self):
        rng = random.Random(0x3D)
        for eb in (1, 2, 4, 8, 16):
            for width, height, depth in ((1, 1, 1), (5, 3, 7), (33, 1, 2), (70, 40, 9), (64, 64, 16)):
                with self.subTest(eb=eb, size=(width, height, depth)):
                    layout = volume_surface_layout(width, height, depth, 1, eb)
                    tiled = rng.randbytes(layout["size"])
                    out = self.assert_same(ttpf.detile_volume, layout, tiled)
                    self.assertEqual(len(out), width * height * depth * eb)


@unittest.skipIf(NATIVE is None, "tools/ttpf_native is not built")
class DecodeMatchesPython(unittest.TestCase):
    def setUp(self):
        ttpf._NATIVE_BC_OK.clear()

    def decode_both(self, fmt, data, blocks_w, blocks_h):
        native = ttpf.decode_pixels(fmt, data, blocks_w, blocks_h)
        with python_only():
            expected = ttpf.decode_pixels(fmt, data, blocks_w, blocks_h)
        return native, expected

    def assert_same_pixels(self, native, expected, label):
        self.assertEqual(len(native), len(expected), label)
        if bits(native) != bits(expected):
            at = next(i for i, (a, b) in enumerate(zip(native, expected)) if struct.pack("<3d", *a) != struct.pack("<3d", *b))
            self.fail(f"{label}: pixel {at}: native {native[at]!r}, Python {expected[at]!r}")

    def test_every_format_and_size(self):
        rng = random.Random(0xDEC)
        for fmt in BC_FORMATS:
            for blocks_w, blocks_h in SIZES:
                with self.subTest(fmt=fmt, size=(blocks_w, blocks_h)):
                    data = random_blocks(rng, fmt, blocks_w * blocks_h)
                    native, expected = self.decode_both(fmt, data, blocks_w, blocks_h)
                    self.assert_same_pixels(native, expected, f"fmt {fmt}")

    def test_bc6h_edge_endpoints(self):
        # Constant fills (all zeros, all ones, sign bits only, alternating) under every 5-bit
        # mode prefix, so every endpoint field meets its extremes; SF16 then gives -0.0 too.
        fills = (0x00, 0xFF, 0x80, 0x7F, 0x55, 0xAA)
        data = b"".join(bytes([(fill & 0xE0) | code]) + bytes([fill] * 15) for code in range(32) for fill in fills)
        for fmt in (95, 96):
            with self.subTest(fmt=fmt):
                native, expected = self.decode_both(fmt, data, 32 * len(fills), 1)
                self.assert_same_pixels(native, expected, f"fmt {fmt}")

    def test_repeated_blocks_share_tuples(self):
        rng = random.Random(2)
        for fmt in (80, 83, 95, 98):
            pool = [random_blocks(rng, fmt, 1) for _ in range(6)]
            data = b"".join(rng.choice(pool) for _ in range(32 * 8))
            with self.subTest(fmt=fmt):
                native, expected = self.decode_both(fmt, data, 32, 8)
                self.assert_same_pixels(native, expected, f"fmt {fmt}")
                self.assertLessEqual(len({id(p) for p in native}), len({id(p) for p in expected}))

    def test_any_buffer_type(self):
        rng = random.Random(9)
        for fmt in (71, 80, 95, 98):
            data = random_blocks(rng, fmt, 15)
            with python_only():
                expected = ttpf.decode_pixels(fmt, data, 5, 3)
            for buf in (bytearray(data), memoryview(data), memoryview(b"xx" + data)[2:]):
                with self.subTest(fmt=fmt, buf=type(buf).__name__):
                    self.assert_same_pixels(ttpf.decode_pixels(fmt, buf, 5, 3), expected, f"fmt {fmt}")

    def test_truncated_input_fails_as_in_python(self):
        for fmt, size in ((71, 8), (74, 16), (80, 8), (83, 16), (95, 16), (98, 16)):
            short = bytes(size * 4 - 1)
            with self.subTest(fmt=fmt):
                try:
                    with python_only():
                        expected = ("ok", ttpf.decode_pixels(fmt, short, 2, 2))
                except Exception as exc:  # noqa: BLE001 - the error type is what is compared
                    expected = ("error", type(exc))
                try:
                    got = ("ok", ttpf.decode_pixels(fmt, short, 2, 2))
                except Exception as exc:  # noqa: BLE001
                    got = ("error", type(exc))
                self.assertEqual(got[0], expected[0])
                if got[0] == "ok":
                    self.assert_same_pixels(got[1], expected[1], f"fmt {fmt}")
                else:
                    self.assertIs(got[1], expected[1])


@unittest.skipIf(NATIVE is None, "tools/ttpf_native is not built")
class BlockMeansMatchesPython(unittest.TestCase):
    def setUp(self):
        ttpf._NATIVE_BC_OK.clear()

    def test_every_format_clip_and_block(self):
        rng = random.Random(0xB10C)
        for fmt in BC_FORMATS:
            for blocks_w, blocks_h in ((1, 1), (2, 3), (13, 7), (40, 40)):
                data = random_blocks(rng, fmt, blocks_w * blocks_h)
                px = ttpf.decode_pixels(fmt, data, blocks_w, blocks_h)
                self.assertIsInstance(px, ttpf._NativePixels)
                plain = list(px)
                w4, h4 = blocks_w * 4, blocks_h * 4
                for pw, ph in ((None, None), (w4 - 1, h4 - 3), (1, 1), (w4 + 9, h4 + 2), (w4 // 2 + 1, 1)):
                    for block in (4, 2, 3, 5, 8):
                        # Grids that are not the decoded one: the means still follow the flat list.
                        gw = max(1, blocks_w * 4 // block)
                        gh = max(1, blocks_h * 4 // block)
                        with self.subTest(fmt=fmt, size=(blocks_w, blocks_h), clip=(pw, ph), block=block):
                            try:
                                expected = ("ok", ttpf.block_means(plain, gw, gh, pw, ph, block))
                            except IndexError:
                                expected = ("error",)
                            try:
                                got = ("ok", ttpf.block_means(px, gw, gh, pw, ph, block))
                            except IndexError:
                                got = ("error",)
                            self.assertEqual(got[0], expected[0])
                            if got[0] == "ok":
                                self.assertEqual(bits(got[1]), bits(expected[1]))

    def test_native_path_is_taken(self):
        data = random_blocks(random.Random(1), 98, 64)
        px = ttpf.decode_pixels(98, data, 8, 8)
        with mock.patch.object(ttpf, "_native_means_raw", wraps=ttpf._native_means_raw) as spy:
            ttpf.block_means(px, 8, 8, 30, 29)
        self.assertEqual(spy.call_count, 1)

    def test_changed_list_averages_its_new_values(self):
        rng = random.Random(4)
        for fmt in (71, 80, 98):
            data = random_blocks(rng, fmt, 16)
            for change in ("set", "slice", "append", "iadd", "sort"):
                with self.subTest(fmt=fmt, change=change):
                    px = ttpf.decode_pixels(fmt, data, 4, 4)
                    if change == "set":
                        px[5] = (1.0, -2.0, 3.5)
                    elif change == "slice":
                        px[0:2] = [(9.0, 9.0, 9.0)] * 2
                    elif change == "append":
                        px.append((0.0, 0.0, 0.0))
                    elif change == "iadd":
                        px += [(7.0, 7.0, 7.0)]
                    else:
                        px.sort()
                    self.assertEqual(bits(ttpf.block_means(px, 4, 4, 15, 14)),
                                     bits(ttpf.block_means(list(px), 4, 4, 15, 14)))

    def test_plain_lists_and_other_inputs_stay_in_python(self):
        px = [(float(i), 0.5, -0.0) for i in range(64)]
        with mock.patch.object(ttpf, "_native_means_raw", side_effect=AssertionError("native used")):
            ttpf.block_means(px, 2, 2, 7, 6, 4)
            ttpf.block_means(ttpf.decode_pixels(28, bytes(64 * 4), 64, 1), 16, 1, 64, 1, 1)


@unittest.skipIf(NATIVE is None, "tools/ttpf_native is not built")
class SelfCheck(unittest.TestCase):
    def setUp(self):
        ttpf._NATIVE_BC_OK.clear()
        self.addCleanup(ttpf._NATIVE_BC_OK.clear)

    def test_a_failed_check_keeps_python(self):
        data = random_blocks(random.Random(6), 98, 4)
        with mock.patch.object(ttpf, "_native_self_check", return_value=False), \
                mock.patch.object(ttpf, "_native_decode_raw", side_effect=AssertionError("native used")), \
                mock.patch.object(ttpf, "_native_means_raw", side_effect=AssertionError("native used")):
            px = ttpf.decode_pixels(98, data, 2, 2)
            ttpf.block_means(px, 2, 2)
        with python_only():
            self.assertEqual(bits(px), bits(ttpf.decode_pixels(98, data, 2, 2)))

    def test_a_diverging_python_decoder_is_noticed(self):
        # If the Python decoder changes and the Rust port does not, the check fails.
        real = ttpf.decode_bc7_block

        def shifted(data, at):
            return [(r + 1.0, g, b) for r, g, b in real(data, at)]
        with mock.patch.dict(ttpf._BLOCK_DECODERS, {98: shifted}):
            self.assertFalse(ttpf._native_ok(NATIVE, 98))

    def test_every_kind_passes(self):
        for fmt in (71, 74, 80, 83, 95, 96, 98):
            with self.subTest(fmt=fmt):
                self.assertTrue(ttpf._native_ok(NATIVE, fmt))


class Fallback(unittest.TestCase):
    CODE = (
        "import importlib.util, os, random, struct, sys\n"
        "sys.path.insert(0, os.path.dirname(sys.argv[1]))\n"
        "from tex_headers import volume_surface_layout\n"
        "spec = importlib.util.spec_from_file_location('t', sys.argv[1])\n"
        "t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)\n"
        "out = [repr(t._native() is None)]\n"
        "rng = random.Random(1)\n"
        "lay = t.surface_layout(40, 24, 71, 3, 4096)\n"
        "tiled = rng.randbytes(lay['size'])\n"
        "out += [t.detile(lay, m, tiled[m['offset']:m['offset'] + m['region']]).hex() for m in lay['mips']]\n"
        "vol = volume_surface_layout(9, 5, 3, 1, 8)\n"
        "out.append(t.detile_volume(vol, rng.randbytes(vol['size'])).hex())\n"
        "for fmt, size in ((71, 8), (80, 8), (83, 16), (95, 16), (96, 16), (98, 16)):\n"
        "    px = t.decode_pixels(fmt, rng.randbytes(6 * size), 3, 2)\n"
        "    out.append(repr(type(px) is list))\n"
        "    out.append(b''.join(struct.pack('<3d', *p) for p in px).hex())\n"
        "    out.append(b''.join(struct.pack('<3d', *p) for p in t.block_means(px, 3, 2, 11, 7)).hex())\n"
        "sys.stdout.write('\\n'.join(out))\n"
    )

    def run_tool(self, **env):
        result = subprocess.run([sys.executable, "-I", "-c", self.CODE, os.path.join(ROOT, "tools", "ttpf_mips.py")],
                                env=dict(os.environ, **env), capture_output=True, text=True, check=True)
        return result.stdout.split("\n")

    def test_disabled_by_environment_gives_the_same_output(self):
        off = self.run_tool(LEONIDA_TTPF_NATIVE="0")
        self.assertEqual(off[0], "True")
        # Disabled, decode_pixels returns plain lists.
        self.assertEqual({off[i] for i in range(len(off)) if off[i] in ("True", "False") and i}, {"True"})
        if NATIVE is not None:
            on = self.run_tool(LEONIDA_TTPF_NATIVE="1")
            self.assertEqual(on[0], "False")
            self.assertEqual([v for v in on[1:] if v not in ("True", "False")],
                             [v for v in off[1:] if v not in ("True", "False")])

    def test_missing_library_falls_back(self):
        with mock.patch.object(ttpf, "_NATIVE", []), mock.patch.object(ttpf.os.path, "isfile", return_value=False):
            self.assertIsNone(ttpf._native())
            layout = ttpf.surface_layout(16, 16, 71, 1, 4096)
            self.assertEqual(len(ttpf.detile(layout, layout["mips"][0], bytes(layout["size"]))), 16 * 8)
            px = ttpf.decode_pixels(98, bytes(64), 2, 2)
            self.assertIs(type(px), list)
            self.assertEqual(ttpf.block_means(px, 2, 2), [(0.0, 0.0, 0.0)] * 4)

    def test_wrong_interface_version_is_not_used(self):
        if NATIVE is None:
            self.skipTest("tools/ttpf_native is not built")
        fake = mock.MagicMock()
        fake.ttpf_native_abi.return_value = 2
        with mock.patch.object(ttpf, "_NATIVE", []), mock.patch("ctypes.CDLL", return_value=fake):
            self.assertIsNone(ttpf._native())


if __name__ == "__main__":
    unittest.main()
