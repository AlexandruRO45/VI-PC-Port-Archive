"""The GPU readback oracle must reject missing, stale and invalid results."""
import importlib.util
from pathlib import Path
import struct
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_gather_dims.py"
spec = importlib.util.spec_from_file_location("gather_dims", SCRIPT)
gather = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gather)


class ReadbackTests(unittest.TestCase):
    def test_known_mip_one_red_footprint(self):
        # u=.375 on a 4-wide mip: x=1 then 2. Gather order: left,right,right,left.
        expected = [39/255, 71/255, 71/255, 39/255]
        self.assertEqual(gather.expected("2D_ARRAY", 0, 1, False, 0), expected)
        gather.check_pixels(struct.pack("<4f", *expected)*4096, expected)

    def test_cube_second_instance_face_five(self):
        self.assertEqual(gather.expected("CUBE", 13, 1, False, 2), [227/255]*4)

    def test_round_even_and_clamp(self):
        for coordinate, layer in [(-1, 0), (2.5, 2), (3.5, 4), (12, 11)]:
            self.assertEqual(gather.expected("2D_ARRAY", coordinate, 0, False, 2), [layer*20/255]*4)

    def test_reject_bad_size(self):
        for size in (0, 16, 64*64*16-4, 64*64*16+4):
            with self.assertRaises(AssertionError):
                gather.check_pixels(bytes(size), [0]*4)

    def test_require_all_taps(self):
        with self.assertRaises(AssertionError):
            gather.check_pixels(bytes(64*64*16), [0]*3)

    def test_reject_last_pixel_and_nonfinite(self):
        for bad in (1, float("nan"), float("inf")):
            raw = bytes(64*64*16-16) + struct.pack("<4f", 0, 0, 0, bad)
            with self.assertRaises(AssertionError):
                gather.check_pixels(raw, [0]*4)

    def test_reject_wrong_mip(self):
        old_mip = gather.expected("CUBE", 13, 0, False, 2)
        with self.assertRaises(AssertionError):
            gather.check_pixels(struct.pack("<4f", *old_mip)*4096,
                                gather.expected("CUBE", 13, 1, False, 2))

    def test_reject_no_draw_clear_even_when_expected_is_zero(self):
        # vk_texture_test clears RGBA32F to -1 before drawing, not zero.
        with self.assertRaises(AssertionError):
            gather.check_pixels(struct.pack("<4f", -1, -1, -1, -1)*4096, [0]*4)


if __name__ == "__main__":
    unittest.main()
