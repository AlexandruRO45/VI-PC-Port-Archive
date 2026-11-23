import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pe_inspect import build_image, place
from runtime_function_inventory import PeFormatError, PeImage, inventory


class InventoryTest(unittest.TestCase):
    def test_sorted_ranges_shared_unwind(self):
        image = bytearray(build_image())
        for index, begin in enumerate((0x1050, 0x1000, 0x1020)):
            place(image, 0x1800 + index * 12, struct.pack("<III", begin, begin + 8, 0x1860))
        place(image, 0x1860, bytes((1 | (4 << 3), 0, 0, 0)))
        self.assertEqual(inventory(PeImage(image)), [
            (rva, rva + 8, ".text", 8, 0x1860, 4) for rva in (0x1000, 0x1020, 0x1050)])

    def test_absent_directory(self):
        self.assertEqual(inventory(PeImage(build_image(with_directories=False))), [])

    def test_invalid_range(self):
        with self.assertRaises(PeFormatError):
            inventory(PeImage(build_image()))


if __name__ == "__main__":
    unittest.main()
