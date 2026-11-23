"""Synthetic readbacks for the sandbox's coverage and texture-detail check."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import sandbox


class SandboxFrame(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.frame = Path(temporary.name) / "frame.ppm"

    def write_frame(self, width, height, pixels):
        self.frame.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)

    def test_valid_frame_preserves_binary_pixels(self):
        pixels = bytes((10, 32, 35, 0, 255, 13))
        self.write_frame(2, 1, pixels)
        self.assertEqual(sandbox.read_ppm(self.frame), (2, 1, pixels))
        self.assertTrue(sandbox.check_frame(self.frame, False)[0])

    def test_invalid_frames_fail_cleanly(self):
        cases = (
            b"", b"P6", b"P6\n1 1", b"P6\n1 1\n255",
            b"junk\n1 1\n255\n\0\0\0", b"P3\n1 1\n255\n\0\0\0",
            b"P6\n0 0\n255\n", b"P6\n-1 -1\n255\n\0\0\0",
            b"P6\n1 0\n255\n", b"P6\n1\n255\n\0\0\0",
            b"P6\n1 1 1\n255\n\0\0\0", b"P6\nx 1\n255\n\0\0\0",
            b"P6\n+1 1\n255\n\0\0\0", b"P6\n0_1 1\n255\n\0\0\0",
            b"P6\n1 1\n65535\n\0\0\0", b"P6\n1 1\n255\n\0\0",
            b"P6\n1 1\n255\n\0\0\0\0", b"P6\n2 1\n255\n\0\0\0",
        )
        for data in cases:
            with self.subTest(data=data):
                self.frame.write_bytes(data)
                with self.assertRaises(ValueError):
                    sandbox.read_ppm(self.frame)
                ok, message = sandbox.check_frame(self.frame, False)
                self.assertFalse(ok)
                self.assertIn("frame", message)

    def test_missing_frame_fails_cleanly(self):
        ok, message = sandbox.check_frame(self.frame, False)
        self.assertFalse(ok)
        self.assertIn("frame", message)

    def test_clear_frame_has_no_coverage(self):
        self.write_frame(1, 1, sandbox.SKY)
        self.assertFalse(sandbox.check_frame(self.frame, False)[0])

    def test_coverage_requires_one_percent_without_rounding_down(self):
        for total, expected in ((100, True), (101, False)):
            with self.subTest(total=total):
                self.write_frame(total, 1, b"\0\0\0" + sandbox.SKY * (total - 1))
                self.assertEqual(sandbox.check_frame(self.frame, False)[0], expected)

    def test_texture_detail_requires_256_colours(self):
        for colours, expected in ((255, False), (256, True)):
            with self.subTest(colours=colours):
                pixels = b"".join(bytes((value, 0, 0)) for value in range(colours))
                self.write_frame(colours, 1, pixels)
                self.assertTrue(sandbox.check_frame(self.frame, False)[0])
                self.assertEqual(sandbox.check_frame(self.frame, True)[0], expected)


if __name__ == "__main__":
    unittest.main()
