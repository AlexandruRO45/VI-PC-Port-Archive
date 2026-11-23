import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pe_inspect import build_image, place, IMAGE_BASE
from psin_root_descriptor import PeFormatError, PeImage, descriptor


class DescriptorTest(unittest.TestCase):
    def test_unaligned_offset_vector_and_member_index(self):
        buf = bytearray(build_image())
        place(buf, 0x1108, struct.pack("<QIIHII", IMAGE_BASE + 0x1300, 42, 24, 3 << 7, 0, 16))
        place(buf, 0x1301, struct.pack("<HHH", 2, 0x70, 0x20))
        place(buf, 0x1388, struct.pack("<IB", 123, 5))
        place(buf, 0x133c, bytes([9]))
        self.assertEqual(descriptor(PeImage(buf), 0x1100), (42, 24, 8, [(0, 5, 123), (16, 9, 0)]))
        place(buf, 0x1303, struct.pack("<H", 0xffff))
        with self.assertRaises(PeFormatError):
            descriptor(PeImage(buf), 0x1100)


if __name__ == "__main__":
    unittest.main()
