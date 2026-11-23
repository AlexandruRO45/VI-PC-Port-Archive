"""Synthetic CPU checks for the mesh-pass coverage comparator."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "mesh_pass_coverage.py"


class MeshPassCoverageTest(unittest.TestCase):
    def test_matching_shifted_and_empty_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            triangles = Path(directory) / "triangles.json"
            raw = Path(directory) / "output.raw"
            # Two triangles cover the right column. The diagonal passes through a pixel center.
            triangles.write_text(json.dumps([[[1, 0], [2, 0], [1, 2]],
                                             [[2, 0], [2, 2], [1, 2]]]), encoding="utf-8")
            for covered, code, expected in (
                ({1, 3}, 0, "IoU 1.000000, missing 0, extra 0"),
                ({0, 2}, 1, "IoU 0.000000, missing 2, extra 2"),
                (set(), 1, "IoU 0.000000, missing 2, extra 0"),
                ({1}, 1, "IoU 0.500000, missing 1, extra 0"),
            ):
                with self.subTest(covered=covered):
                    raw.write_bytes(b"".join(struct.pack("<4f", *((1.0 if i in covered else -1.0,) * 4))
                                             for i in range(4)))
                    run = subprocess.run([sys.executable, "-I", str(TOOL), str(triangles),
                                          str(raw), "2", "2"], capture_output=True, text=True)
                    self.assertEqual(run.returncode, code, run.stderr)
                    self.assertEqual(run.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
