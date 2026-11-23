"""tools/navmesh_cell_fit.py: the grid fit, its chance baseline and the per-cell summary, on synthetic vertices.
No game files."""
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_navmesh_decode import build_0drp, build_navmesh, make_rpf  # noqa: E402
import navmesh_cell_fit as cf  # noqa: E402


def quantise(points, ext):
    return [tuple(round(p[k] / ext[k] * 65535) if ext[k] else 0 for k in range(3)) for p in points]


class CellFit(unittest.TestCase):
    def test_points_on_a_cell_grid_fit_it_and_not_another(self):
        ext = (1.755, 5.1675, 0.0)                   # 54 x 159 cells of 0.0325
        pts = [(i * 0.0325, j * 0.0325, 0.0) for i in range(0, 55, 6) for j in range(0, 160, 7)]
        q = quantise(pts, ext)
        self.assertGreaterEqual(cf.fit_fraction(q, ext, 0.0325), 0.99)
        self.assertLess(cf.fit_fraction(q, ext, 0.03125), 0.5)

    def test_baseline_is_small_for_a_fine_step(self):
        self.assertLess(cf.baseline(2.0625, 0.0625), 0.06)
        # a 128 m extent's step is 128/65535, a little over 1/512 m: it drifts off a 1/32 m grid, so the static
        # tiles are a real control (3 levels in 16 land within 1.5 steps)
        self.assertAlmostEqual(cf.baseline(128.0, 0.03125), 0.1875, places=3)

    def test_summary_takes_the_coarsest_documented_cell(self):
        ext = (2.0, 4.0, 0.0)
        coarse = quantise([(i * 0.0625, j * 0.0625, 0) for i in range(33) for j in range(0, 65, 4)], ext)
        fine = quantise([(i * 0.03125 + 0.03125 * (i % 2 == 1), j * 0.03125, 0) for i in range(0, 63, 2)
                         for j in range(1, 128, 2)], ext)
        meshes = [{"n": len(q), "ext": ext, "fit": {c: cf.fit_fraction(q, ext, c) for c in cf.DOC_CELLS + cf.OTHER_CELLS}}
                  for q in (coarse, fine)]
        s = cf.summarise(meshes)
        self.assertEqual(s["cells"]["0.0625"]["meshes"], 1)
        self.assertEqual(s["cells"]["0.03125"]["meshes"], 1)
        self.assertEqual(s["cells"]["0.0325"]["meshes"], 0)
        self.assertEqual(s["fit_none"], 0)

    def test_name_hashes_cover_case_and_suffix(self):
        h = cf.name_hashes({"Wagon01"})
        self.assertIn(cf.nd.joaat_cs("wagon01"), h)
        self.assertIn(cf.nd.joaat_cs("Wagon01.yft"), h)
        self.assertEqual(len(h), 4)


class GatedPath:
    """Hold the real scanner at its first filesystem access until the test releases it."""

    def __init__(self, path):
        self.path = path
        self.entered = Event()
        self.resume = Event()

    def __fspath__(self):
        self.entered.set()
        if not self.resume.wait(10):
            raise TimeoutError("waiting to resume navmesh scan")
        return os.fspath(self.path)


class ScanIsolation(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.archive = root / "one.rpf"
        self.carve = root / "two"
        self.carve.mkdir()
        self.tile = build_navmesh(flags=7)[0]
        make_rpf(self.archive, [self.tile])
        for i, blob in enumerate([self.tile, *build_0drp(), self.tile], start=1):
            (self.carve / f"{i:012x}.bin").write_bytes(blob)

    def test_measurements_and_plain_scan_are_independent(self):
        decoder = cf.nd.decode_navmesh
        self.addCleanup(setattr, cf.nd, "decode_navmesh", decoder)
        one, two = cf.measure(self.archive), cf.measure(self.carve)
        self.assertEqual((one[0]["meshes"], one[1]), (1, 1))
        self.assertEqual((two[0]["meshes"], two[1]), (2, 2))
        self.assertIs(cf.nd.decode_navmesh, decoder)

        first, second = GatedPath(self.archive), GatedPath(self.carve)
        with ThreadPoolExecutor(max_workers=2) as workers:
            try:
                a = workers.submit(cf.measure, first)
                self.assertTrue(first.entered.wait(10))
                b = workers.submit(cf.measure, second)
                self.assertTrue(second.entered.wait(10))
                during_both = cf.nd.decode_navmesh
                plain = cf.nd.scan(self.carve)
                # Finish the older call while the newer call is still inside scan.
                first.resume.set()
                result_a = a.result(timeout=10)
                during_second = cf.nd.decode_navmesh
                second.resume.set()
                result_b = b.result(timeout=10)
            finally:
                first.resume.set()
                second.resume.set()

        self.assertEqual(plain["resources"]["navmesh"]["resources"], 2)
        self.assertEqual((result_a, result_b), (one, two))
        self.assertIs(during_both, decoder)
        self.assertIs(during_second, decoder)
        self.assertIs(cf.nd.decode_navmesh, decoder)

    def test_observer_preserves_results_and_propagates_errors(self):
        failure = RuntimeError("observer failed")

        def fail(tile):
            raise failure

        for path, count in ((self.archive, 1), (self.carve, 2)):
            with self.subTest(path=path):
                original = cf.nd.decode_navmesh
                before_rows, after_rows, tiles = [], [], []
                before = cf.nd.scan(path, None, None, True, before_rows)
                with self.assertRaises(RuntimeError) as raised:
                    cf.nd.scan(path, on_navmesh=fail)
                self.assertIs(raised.exception, failure)
                self.assertIs(cf.nd.decode_navmesh, original)
                after = cf.nd.scan(path, None, None, True, after_rows, on_navmesh=tiles.append)
                self.assertEqual(tiles, [self.tile] * count)
                self.assertEqual(before_rows, after_rows)
                del before["elapsed_s"], after["elapsed_s"]
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
