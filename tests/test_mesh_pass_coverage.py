"""Synthetic CPU checks for the mesh-pass coverage comparator."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "mesh_pass_coverage.py"
sys.path.insert(0, str(ROOT / "tools"))
import mesh_pass_coverage as mpc  # noqa: E402


def write_mask(path, width, height, covered):
    mask = bytearray(width * height)
    for index in covered:
        mask[index] = 1
    path.write_bytes(mask)


def run_tool(triangles, coverage, width, height, *extra):
    return subprocess.run([sys.executable, "-I", str(TOOL), str(triangles), str(coverage),
                           str(width), str(height), *extra], capture_output=True, text=True)


class MeshPassCoverageTest(unittest.TestCase):
    def test_matching_shifted_and_empty_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            triangles = directory / "triangles.json"
            coverage = directory / "coverage.mask"
            raw = directory / "output.raw"
            # Two triangles cover the right column. The diagonal passes through a pixel center.
            triangles.write_text(json.dumps([[[1, 0], [2, 0], [1, 2]],
                                             [[2, 0], [2, 2], [1, 2]]]), encoding="utf-8")
            raw.write_bytes(struct.pack("<4f", 1, 1, 1, 1) * 4)
            for covered, code, expected in (
                ({1, 3}, 0, "IoU 1.000000, missing 0, extra 0"),
                ({0, 2}, 1, "IoU 0.000000, missing 2, extra 2"),
                (set(), 1, "IoU 0.000000, missing 2, extra 0"),
                ({1}, 1, "IoU 0.500000, missing 1, extra 0"),
            ):
                with self.subTest(covered=covered):
                    write_mask(coverage, 2, 2, covered)
                    run = run_tool(triangles, coverage, 2, 2, "--raw", raw)
                    self.assertEqual(run.returncode, code, run.stderr)
                    self.assertEqual(run.stdout.strip(), expected)

    def test_fragment_clear_colour_is_still_covered(self):
        """editrice: a covered fragment writing (-1,-1,-1,-1) must not vanish from the mask."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            triangles = [[[0, 0], [2, 0], [0, 2]]]
            expected = mpc.rasterize(triangles, 2, 2, subpixel_bits=8)
            covered = {i for i, value in enumerate(expected) if value}
            self.assertTrue(covered)
            path = directory / "triangles.json"
            path.write_text(json.dumps(triangles), encoding="utf-8")
            coverage = directory / "coverage.mask"
            write_mask(coverage, 2, 2, covered)
            # The raw colour is the harness clear sentinel; the mask is the only coverage signal.
            raw = directory / "output.raw"
            raw.write_bytes(struct.pack("<4f", -1.0, -1.0, -1.0, -1.0) * 4)
            run = run_tool(path, coverage, 2, 2, "--raw", raw)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), "IoU 1.000000, missing 0, extra 0")

    def test_subpixel_snap_matches_kev_ball(self):
        """Kev-Ball: 4.5 + 1/1024 must snap to 4.5, not look like a binding error."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            snapped = [[4.5, 2], [13, 2], [4.5, 14]]
            unsnapped = [[4.5009765625, 2], [13, 2], [4.5009765625, 14]]
            expected = mpc.rasterize([snapped], 16, 16, subpixel_bits=0)
            covered = {i for i, value in enumerate(expected) if value}
            self.assertTrue(covered)
            path = directory / "triangles.json"
            path.write_text(json.dumps([unsnapped]), encoding="utf-8")
            coverage = directory / "coverage.mask"
            write_mask(coverage, 16, 16, covered)
            run = run_tool(path, coverage, 16, 16)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), "IoU 1.000000, missing 0, extra 0")
            # Without the snap the same mask is rejected, which is the reported failure mode.
            run = run_tool(path, coverage, 16, 16, "--subpixel-bits", "0")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertNotEqual(run.stdout.strip(), "IoU 1.000000, missing 0, extra 0")

    def test_half_ties_round_to_even_grid(self):
        """CuteBooty: exact half-step ties follow the even 1/256 grid coordinate on the measured driver."""
        self.assertEqual(mpc.snap(4.5 + 1 / 1024, 8), 4.5)
        self.assertEqual(mpc.snap(1152.5 / 256, 8), 1152 / 256)
        self.assertEqual(mpc.snap(1153.5 / 256, 8), 1154 / 256)
        self.assertEqual(mpc.snap(4.5, 0), 4.5)

    def test_512x512_peak_memory(self):
        """shdann: stream both inputs and keep the CPU mask in a bytearray."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            width = height = 512
            triangles = [[[0, 0], [width, 0], [0, height]],
                         [[width, 0], [width, height], [0, height]]]
            coverage = directory / "coverage.mask"
            coverage.write_bytes(b"\1" * (width * height))
            raw = directory / "output.raw"
            raw.write_bytes(struct.pack("<4f", 1.0, 1.0, 1.0, 1.0) * (width * height))
            tracemalloc.start()
            result = mpc.compare(triangles, coverage, width, height, raw=raw)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            self.assertEqual((result["iou"], result["missing"], result["extra"]),
                             (1.0, 0, 0))
            self.assertLessEqual(peak, 2_000_000, f"peak {peak} bytes")


if __name__ == "__main__":
    unittest.main()
