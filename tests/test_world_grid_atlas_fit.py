"""Synthetic unit tests for linear_fit() validation and Pearson r bounds in tools/world_grid_atlas.py."""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import world_grid_atlas as wga  # noqa: E402


class TestWorldGridAtlasLinearFit(unittest.TestCase):
    def test_mismatched_lengths_rejected(self):
        for xs, ys in (([1.0, 2.0, 3.0], [1.0, 2.0]), ([1.0, 2.0], [1.0, 2.0, 3.0])):
            with self.subTest(xs=xs, ys=ys):
                with self.assertRaisesRegex(ValueError, r"lengths differ"):
                    wga.linear_fit(xs, ys)

    def test_non_finite_values_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad, where="x"):
                with self.assertRaisesRegex(ValueError, r"finite"):
                    wga.linear_fit([1.0, bad, 3.0], [2.0, 4.0, 6.0])
            with self.subTest(bad=bad, where="y"):
                with self.assertRaisesRegex(ValueError, r"finite"):
                    wga.linear_fit([1.0, 2.0, 3.0], [2.0, bad, 6.0])

    def test_constant_x_and_constant_y(self):
        with self.assertRaisesRegex(ValueError, r"all x values are equal"):
            wga.linear_fit([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])

        flat = wga.linear_fit([1.0, 2.0, 3.0], [7.0, 7.0, 7.0])
        self.assertEqual(flat["slope"], 0.0)
        self.assertEqual(flat["intercept"], 7.0)
        self.assertEqual(flat["r"], 0.0)
        self.assertEqual(flat["n"], 3)

    def test_exact_linear_and_pearson_clamping(self):
        pos = wga.linear_fit([1.0, 2.0, 4.0], [5.0, 7.0, 11.0])
        self.assertAlmostEqual(pos["slope"], 2.0)
        self.assertAlmostEqual(pos["intercept"], 3.0)
        self.assertEqual(pos["r"], 1.0)
        self.assertEqual(pos["n"], 3)

        neg = wga.linear_fit([0.0, 2.0, 4.0], [10.0, 4.0, -2.0])
        self.assertAlmostEqual(neg["slope"], -3.0)
        self.assertAlmostEqual(neg["intercept"], 10.0)
        self.assertEqual(neg["r"], -1.0)

        # Large offsets where floating-point rounding can push sxy / sqrt(sxx * syy) past +-1.0.
        xs = [1e6 + 0.1, 1e6 + 0.2, 1e6 + 0.3, 1e6 + 0.7]
        ys_pos = [3.0 * x + 5.0 for x in xs]
        ys_neg = [-3.0 * x + 5.0 for x in xs]
        self.assertLessEqual(wga.linear_fit(xs, ys_pos)["r"], 1.0)
        self.assertGreaterEqual(wga.linear_fit(xs, ys_neg)["r"], -1.0)


if __name__ == "__main__":
    unittest.main()
