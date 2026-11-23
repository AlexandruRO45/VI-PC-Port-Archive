"""Synthetic read-only garment style fixtures with inherited and overridden resources."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]

import garment_style
from test_psin_data import make_container, ptr
from world_names import joaat


def array(count, block, offset=0):
    return struct.pack("<IIQ", count, count, ptr(block, offset))


def group(stride, entries):
    controls = bytearray([0x80] * 16)
    rows = bytearray(16 * stride)
    for slot, (key, record) in enumerate(entries):
        controls[slot] = 1
        struct.pack_into("<I", rows, slot * stride, key)
        rows[slot * stride + 8:(slot + 1) * stride] = record
    return controls + rows


def fixture(*, unknown_preset=False, bad_array=False, bad_count=False):
    tile_id, other_tile_id = 10, 11
    preset_id, extra_id, style_id = 20, 30, 40
    root = bytearray(64)
    for i, count in enumerate((2, 1, 2, 1)):
        struct.pack_into("<IIQ", root, i * 16, 1, count, ptr(i + 2))
    if bad_count:
        struct.pack_into("<I", root, 52, 2)
    tile = bytearray(56)
    tile[24:40] = array(2, 8)
    preset = bytearray(80)
    struct.pack_into("<I", preset, 4, tile_id)
    preset[8:24] = array(1, 6)
    struct.pack_into("<I", preset, 72, extra_id)
    extra = bytearray(48)
    extra[32:48] = array(1, 9)
    style = bytearray(168)
    struct.pack_into("<2I", style, 8, preset_id, 999 if unknown_preset else preset_id)
    struct.pack_into("<I", style, 40, other_tile_id)
    struct.pack_into("<I", style, 72, extra_id + 1)
    style[104:120] = array(1, 6, 12)
    style[120:136] = array(1, 8 if bad_array else 7)
    float_params = struct.pack("<IfBBxx", joaat("tintmaskopacity"), 0.0, 3, 255) + struct.pack(
        "<IfBBxx", joaat("tintmaskopacity"), 1.25, 3, 0)
    textures = b"".join(struct.pack("<I4xQBB6x", 101, value, i, 255)
                        for i, value in enumerate((0x1234567800000001, 0)))
    return make_container([
        (0x8419AB01, root), (0x115, group(64, [(tile_id, tile), (other_tile_id, bytes(56))])),
        (0x11B, group(88, [(preset_id, preset)])), (0x127, group(56, [(extra_id, extra), (extra_id + 1, bytes(48))])),
        (0x12D, group(176, [(style_id, style)])), (0x38F05860, float_params),
        (0xE9D99E0A, struct.pack("<I4BBBxx", 102, 12, 34, 56, 255, 2, 1)),
        (0xF1CB6289, textures), (0x4B2625BB, struct.pack("<IIBBxx", 103, 0xffffffff, 6, 255))
    ], [])


class GarmentStyleTests(unittest.TestCase):
    def test_sources_slots_and_typed_values(self):
        doc = garment_style.decode_style(fixture(), 40)
        self.assertEqual(doc["style"], "00000028")
        self.assertEqual(len(doc["slots"]), 8)
        first, second, empty = doc["slots"][:3]
        self.assertEqual((first["tile_override"], first["preset_tile"], first["tile"]),
                         ("0000000b", "0000000a", "0000000b"))
        self.assertEqual((second["tile_override"], second["tile"], second["preset_extra"]),
                         (None, "0000000a", "0000001e"))
        self.assertEqual((first["preset_extra"], first["extra_override"]), ("0000001e", "0000001f"))
        self.assertIsNone(second["extra_override"])
        self.assertIsNone(empty["preset"])
        self.assertIsNone(empty["tile"])
        records = doc["records"]
        preset_param, = records["preset"]["00000014"]["params"]
        style_float, color = records["style"]["00000028"]["params"]
        self.assertEqual((preset_param["value"], style_float["value"]), (0.0, 1.25))
        self.assertEqual(style_float["name"], "tintmaskopacity")
        self.assertNotIn("name", color)
        self.assertEqual((style_float["type"], style_float["index"], style_float["slot"]), ("float", 3, 0))
        self.assertEqual((color["type"], color["index"], color["slot"], color["value"]),
                         ("rgba8", 2, 1, [12, 34, 56, 255]))
        self.assertEqual([r["value"] for r in records["tile"]["0000000a"]["params"]],
                         ["1234567800000001", "0000000000000000"])
        self.assertEqual(records["extra"]["0000001e"]["params"][0]["value"], 0xffffffff)
        self.assertEqual(records["tile"]["0000000b"]["params"], [])

    def test_incomplete_data_or_unresolved_references(self):
        for data, key in [(fixture(), 99), (fixture()[:48], 40),
                          (fixture(unknown_preset=True), 40), (fixture(bad_array=True), 40),
                          (fixture(bad_count=True), 40)]:
            with self.subTest(key=key, size=len(data)), self.assertRaises(ValueError):
                garment_style.decode_style(data, key)

    def test_cli_decimal_and_hex(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = Path(folder) / "synthetic.pso"
            bank.write_bytes(fixture())
            command = [sys.executable, "-I", str(ROOT / "tools/garment_style.py"), str(bank)]
            listing = subprocess.run(command + ["--list-styles"], check=True, capture_output=True)
            listed, = json.loads(listing.stdout)["styles"]
            for style_id in ("40", listed):
                result = subprocess.run(command + ["--style", style_id], capture_output=True, check=True)
                self.assertEqual(json.loads(result.stdout), garment_style.decode_style(bank.read_bytes(), 40))
            for value in ("-1", "0x100000000", "0x63"):
                with self.subTest(value=value):
                    result = subprocess.run(command + ["--style", value], capture_output=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(result.stdout)
                    self.assertNotIn(b"Traceback", result.stderr)
            self.assertEqual(bank.read_bytes(), fixture())


if __name__ == "__main__":
    unittest.main()
