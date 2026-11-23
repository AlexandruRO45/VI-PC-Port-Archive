"""Synthetic ptr() and invert_affine() validation tests for tools/xdxr_bvh.py. No game data."""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import xdxr_bvh  # noqa: E402


class TestXdxrBvhBounds(unittest.TestCase):
    def test_ptr_rejects_negative_offset_and_invalid_kind(self):
        with self.assertRaises(ValueError):
            xdxr_bvh.ptr(-64, xdxr_bvh.BOX32)
        for bad_kind in (-1, 8, 255):
            with self.subTest(kind=bad_kind):
                with self.assertRaises(ValueError):
                    xdxr_bvh.ptr(64, bad_kind)
        self.assertEqual(xdxr_bvh.ptr(64, xdxr_bvh.BOX32), (1 << 3) | xdxr_bvh.BOX32)

    def test_invert_affine_rejects_non_finite_entries_and_determinant(self):
        for bad in (math.nan, math.inf, -math.inf):
            m = [[1.0, 0.0, 0.0, bad], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, r"non-finite"):
                    xdxr_bvh.invert_affine(m)

        # Overflowing finite determinant (1e200 ** 3 -> inf) must also raise ValueError.
        huge = [[1e200, 0.0, 0.0, 0.0], [0.0, 1e200, 0.0, 0.0], [0.0, 0.0, 1e200, 0.0]]
        with self.assertRaisesRegex(ValueError, r"singular"):
            xdxr_bvh.invert_affine(huge)

        m_ok = [[2.0, 0.0, 0.0, 4.0], [0.0, 2.0, 0.0, -6.0], [0.0, 0.0, 2.0, 8.0]]
        inv = xdxr_bvh.invert_affine(m_ok)
        self.assertEqual(inv, [[0.5, 0.0, 0.0, -2.0], [0.0, 0.5, 0.0, 3.0], [0.0, 0.0, 0.5, -4.0]])


if __name__ == "__main__":
    unittest.main()
