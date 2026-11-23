"""tools/navmesh_cell_fit.py: the grid fit, its chance baseline and the per-cell summary, on synthetic vertices.
No game files."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
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


class ReadError(unittest.TestCase):
    """An archive the tool cannot read is one error line and exit 2, not a traceback."""

    def run_main(self, path):
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["navmesh_cell_fit.py", path]), contextlib.redirect_stderr(err):
            return cf.main(), err.getvalue()

    def test_a_file_that_is_not_an_rpf_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            open(empty, "wb").close()
            notrpf = os.path.join(tmp, "notrpf.rpf")
            with open(notrpf, "wb") as f:
                f.write(b"\x01" * 32)
            for path, want in ((empty, "too small for an RPF header"), (notrpf, "not an RPF7/8/9 archive")):
                with self.subTest(name=os.path.basename(path)):
                    rc, err = self.run_main(path)
                    self.assertEqual(rc, 2)
                    self.assertIn("error: ", err)
                    self.assertIn(want, err)
                    self.assertNotIn("Traceback", err)

    def test_a_missing_archive_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, err = self.run_main(os.path.join(tmp, "missing.rpf"))
            self.assertEqual(rc, 2)
            self.assertIn("error: ", err)
            self.assertIn("No such file", err)
            self.assertNotIn("Traceback", err)

    def test_a_common_path_that_is_not_a_directory_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "tiny.rpf")
            with open(archive, "wb") as f:  # the magic and a u32: enough for the option check, never scanned
                f.write(b"9FPR" + b"\0" * 8)
            for name, common in (("missing", os.path.join(tmp, "missing")), ("a file", archive)):
                with self.subTest(name=name):
                    err, out = io.StringIO(), io.StringIO()
                    argv = ["navmesh_cell_fit.py", archive, "--common", common]
                    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err), \
                            contextlib.redirect_stdout(out):
                        rc = cf.main()
                    self.assertEqual(rc, 2)
                    self.assertIn("error: --common ", err.getvalue())
                    self.assertIn(common, err.getvalue())
                    self.assertNotIn("Traceback", err.getvalue())
                    self.assertEqual(out.getvalue(), "")   # no report with zero documents


if __name__ == "__main__":
    unittest.main()
