"""world_grid_atlas.align: the windowed, comprehension-based alignment gives the plain n x m loop's path.

    python -I tests/test_world_grid_atlas_align.py

``reference_align`` below is the original loop, kept inline: every case compares ``wga.align`` with it directly, so
no hash or byte pin is involved and the result is the same on every platform (the inputs come from a seeded
``random.Random``, which is platform independent). No game data and no numpy are used.
"""
import os
import random
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import world_grid_atlas as wga  # noqa: E402


def reference_align(scores):
    """The row-by-row loop over every slot, with a back-pointer table per row."""
    n = len(scores)
    if not n:
        return []
    m = len(scores[0])
    if m < n:
        raise ValueError(f"{n} tiles do not fit {m} slots in order")
    neg = float("-inf")
    best = list(scores[0])
    back = []
    for i in range(1, n):
        row = scores[i]
        cur = [neg] * m
        arg = [-1] * m
        run, run_at = neg, -1
        for j in range(1, m):
            if best[j - 1] > run:
                run, run_at = best[j - 1], j - 1
            if j >= i and run > neg:
                cur[j] = run + row[j]
                arg[j] = run_at
        back.append(arg)
        best = cur
    j = max(range(m), key=lambda k: best[k])
    path = [j]
    for arg in reversed(back):
        j = arg[j]
        path.append(j)
    return path[::-1]


def table(rnd, n, m, kind):
    def value():
        if kind == "ties":
            return float(rnd.randint(0, 3))          # many equal sums: the earliest argmax must win
        if kind == "sparse":
            return rnd.random() if rnd.random() < 0.08 else 0.0
        if kind == "negative":
            return rnd.choice([-3.0, -1.5, -0.0, 0.0, 1.0])
        if kind == "mixed":
            return rnd.uniform(-1.0, 1.0)
        return rnd.random()
    return [[value() for _ in range(m)] for _ in range(n)]


class AlignParityTest(unittest.TestCase):
    def check(self, scores):
        self.assertEqual(wga.align(scores), reference_align(scores))

    def test_random_tables(self):
        rnd = random.Random(2112)
        for kind in ("ties", "sparse", "negative", "mixed", "dense"):
            for _ in range(250):
                n = rnd.randint(1, 30)
                m = rnd.choice([n, n + 1, rnd.randint(n, 60), rnd.randint(n, 150)])
                with self.subTest(kind=kind, n=n, m=m):
                    self.check(table(rnd, n, m, kind))

    def test_shapes_at_the_window_edges(self):
        rnd = random.Random(7)
        for n, m in ((1, 1), (1, 9), (2, 2), (2, 3), (5, 5), (9, 10), (3, 200), (40, 41)):
            for kind in ("dense", "ties", "sparse"):
                with self.subTest(n=n, m=m, kind=kind):
                    self.check(table(rnd, n, m, kind))

    def test_measured_scale(self):
        # a shoreline-sized table: 268 tiles on the 32 x 66 grid's 2,112 cells, mostly zero overlaps
        rnd = random.Random(268)
        self.check(table(rnd, 268, 2112, "sparse"))

    def test_scores_that_rise_along_the_slots(self):
        # every slot is a new running maximum: the most back pointers a row can have
        rnd = random.Random(5)
        s = [[j / 97 + rnd.random() * 1e-3 for j in range(97)] for _ in range(23)]
        self.check(s)
        self.check([[float(j) for j in range(64)] for _ in range(64)])
        self.check([[float(-j) for j in range(64)] for _ in range(32)])

    def test_constant_table_takes_the_earliest_slots(self):
        s = [[1.0] * 12 for _ in range(5)]
        self.assertEqual(wga.align(s), [0, 1, 2, 3, 4])
        self.check(s)

    def test_result_is_a_strictly_increasing_path_of_the_best_sum(self):
        rnd = random.Random(31)
        s = table(rnd, 14, 30, "dense")
        path = wga.align(s)
        self.assertEqual(len(path), 14)
        self.assertTrue(all(b > a for a, b in zip(path, path[1:])))
        self.assertTrue(0 <= path[0] and path[-1] < 30)
        ref = reference_align(s)
        self.assertEqual(sum(s[i][j] for i, j in enumerate(path)), sum(s[i][j] for i, j in enumerate(ref)))

    def test_scores_that_are_not_finite(self):
        inf, nan = float("inf"), float("nan")
        cases = [
            [[0.0, -inf, 1.0, 0.5], [1.0, 2.0, -inf, 0.0]],
            [[0.0, inf, 1.0, 0.5], [1.0, 2.0, 3.0, 0.0]],
            [[-inf, -inf, -inf], [-inf, -inf, -inf]],
            [[1e308, 1e308, 1e308, 1e308], [1e308, 1e308, 1e308, 1e308], [1e308, 1e308, 1e308, 1e308]],
            [[-1e308, -1e308, -1e308, -1e308], [-1e308, -1e308, -1e308, -1e308], [-1e308] * 4],
        ]
        for s in cases:
            with self.subTest(s=s):
                self.check(s)
        with_nan = [[0.0, nan, 1.0], [1.0, 2.0, 3.0]]
        self.assertEqual(wga.align(with_nan), reference_align(with_nan))

    def test_errors_match_the_loop(self):
        self.assertEqual(wga.align([]), [])
        for s in ([[1.0], [1.0]], [[0.0, 1.0], [1.0]], [[0.0, 1.0, 2.0], [1.0, 2.0]]):
            with self.subTest(s=s):
                with self.assertRaises((ValueError, IndexError)) as want:
                    reference_align(s)
                with self.assertRaises(type(want.exception)):
                    wga.align(s)

    def test_input_is_not_modified(self):
        rnd = random.Random(1)
        s = table(rnd, 9, 20, "dense")
        copy = [list(r) for r in s]
        wga.align(s)
        self.assertEqual(s, copy)


class AlignMemoryTest(unittest.TestCase):
    """The peak memory of an alignment does not exceed the loop's, for dense and for sparse tables."""

    def peak(self, fn, scores):
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            path = fn(scores)
            return tracemalloc.get_traced_memory()[1] - base, path
        finally:
            tracemalloc.stop()

    def compare(self, scores):
        ref_peak, ref_path = self.peak(reference_align, scores)
        new_peak, new_path = self.peak(wga.align, scores)
        self.assertEqual(new_path, ref_path)
        self.assertLessEqual(new_peak, ref_peak)

    def test_dense_tables(self):
        rnd = random.Random(10)
        self.compare(table(rnd, 120, 400, "dense"))
        self.compare([[j / 400 for j in range(400)] for _ in range(120)])     # every slot is a record

    def test_sparse_tables(self):
        rnd = random.Random(11)
        self.compare(table(rnd, 300, 500, "sparse"))
        self.compare([[0.0] * 500 for _ in range(300)])

    def test_square_table(self):
        rnd = random.Random(12)
        self.compare(table(rnd, 200, 200, "dense"))


class AlignWindowTest(unittest.TestCase):
    def test_slots_outside_a_rows_window_do_not_matter(self):
        # row i is only on a path in slots i .. i + (m - n); changing any other score cannot change the path
        rnd = random.Random(99)
        n, m = 6, 15
        s = table(rnd, n, m, "dense")
        want = wga.align(s)
        t = [list(r) for r in s]
        for i in range(n):
            for j in range(m):
                if not i <= j <= i + (m - n):
                    t[i][j] = 1e6 if (i + j) % 2 else -1e6
        self.assertEqual(wga.align(t), reference_align(t))
        self.assertEqual(wga.align(t), want)

    def test_back_pointers_do_not_grow_with_the_table(self):
        # a table with no score at all keeps one record per row, not a row of back pointers
        n, m = 200, 600
        s = [[0.0] * m for _ in range(n)]
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            wga.align(s)
            peak = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        self.assertLess(peak, n * m)          # bytes: far below even one byte per table cell


if __name__ == "__main__":
    unittest.main()
