"""Fixed-count lane controls and a physical block-address oracle for packed tails."""
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bc7_tiles import linearize_4k_s_tail
from bcpack_probe import probe


class TextureChecks(unittest.TestCase):
    def test_constant_lane_with_two_refills(self):
        prefix = bytearray(48)
        prefix[4::8] = bytes.fromhex("de 05 03 0c 86 4e")

        result, = probe(prefix, 25000, prefetch_mask=0x10, prefetch=b"\xac")

        self.assertEqual(result["lane"], 4)
        self.assertEqual(result["refill_bytes"], 2)
        self.assertEqual(result["first_refill_offset"], 48)
        self.assertTrue(result["first_refill_matches"])

        # Lane 4 consumes AC then 7D. Other lanes shift its packet offsets.
        for byte, expected in ((0x7D, True), (0x7C, False)):
            result, = probe(prefix, 25000, prefetch_mask=0x11, prefetch=bytes([0, 0xAC]),
                            next_mask=0x14, next_bytes=bytes([0, byte]))
            self.assertEqual(result["second_refill_offset"], 51)
            self.assertTrue(result["first_refill_matches"])
            self.assertEqual(result["second_refill_matches"], expected)
        with self.assertRaises(ValueError):
            probe(prefix, 25000, next_mask=0x10)

    def test_constant_lane(self):
        prefix = bytearray(48)
        prefix[4::8] = bytes.fromhex("de 03 01 00 40 02")
        result, = probe(prefix, 2)
        self.assertEqual((result["lane"], result["status"]), (4, "verified_constant_lane"))
        self.assertFalse(probe(prefix, 3))
        prefix[4::8] = bytes.fromhex("de 04 01 00 11 30")
        result, = probe(prefix, 11358)
        self.assertEqual((result["refill_bytes"], result["status"]), (1, "header_match"))
        # Lanes 0, 2 and 4 read in this round. Lane 4 takes the third byte.
        result, = probe(prefix, 11358, prefetch_mask=0x15, prefetch=bytes([2, 17, 125]))
        self.assertEqual(result["first_refill_offset"], 50)
        self.assertTrue(result["first_refill_matches"])
        result, = probe(prefix, 11358, prefetch_mask=0x15, prefetch=bytes([2, 125, 17]))
        self.assertFalse(result["first_refill_matches"])
        with self.assertRaises(ValueError):
            probe(prefix, 11358, prefetch_mask=0x15, prefetch=bytes([125]))
        prefix[12] -= 1
        self.assertFalse(probe(prefix, 11358))
        with self.assertRaises(ValueError):
            probe(prefix[:-1], 2)

    def test_packed_tail_slots(self):
        tile = b"".join(struct.pack("<4I", *([index] * 4)) for index in range(256))
        for level, first in enumerate((128, 96, 80, 64, 48, 32, 16, 0)):
            side = max(1, 32 >> level)
            data = linearize_4k_s_tail(tile, side, side, level)
            count = (side + 3) // 4
            self.assertEqual(len(data), count * count * 16)
            self.assertEqual(struct.unpack_from("<I", data)[0], first)
            if count >= 2:
                self.assertEqual(struct.unpack_from("<I", data, 16)[0], first + 4)
                self.assertEqual(struct.unpack_from("<I", data, count * 16)[0], first + 1)
        for size, level in ((64, 0), (8, 7), (1, 8)):
            with self.assertRaises(ValueError):
                linearize_4k_s_tail(tile, size, size, level)


if __name__ == "__main__":
    unittest.main()
