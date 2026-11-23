"""Literal --common directory handling in navmesh_cell_fit, using synthetic vehicle documents only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import navmesh_cell_fit as cf  # noqa: E402


class VehicleDocuments(unittest.TestCase):
    @staticmethod
    def write_document(path, model="SyntheticWagon", cell="0.0625"):
        path.write_text(
            f'<CVehiclePoints><fNavOverrideRecastCellSize value="{cell}"/>'
            f'<ModelName>{model}</ModelName></CVehiclePoints>', encoding="utf-8", newline="\n")

    def test_brackets_in_the_carve_or_its_parent_are_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relative in ("common [v1]", "work [v1]/common"):
                with self.subTest(relative=relative):
                    carve = Path(tmp, relative)
                    carve.mkdir(parents=True)
                    self.write_document(carve / "document.bin")
                    cells, names = cf.vehicle_documents(str(carve))
                    self.assertEqual(cells, {0.0625: 1})
                    self.assertEqual(names, {"SyntheticWagon"})

    def test_a_glob_matching_sibling_cannot_supply_the_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve, sibling = Path(tmp, "common [v1]"), Path(tmp, "common v")
            carve.mkdir()
            sibling.mkdir()
            self.write_document(carve / "document.bin")
            self.write_document(sibling / "document.bin", model="SyntheticDecoy", cell="0.03125")
            cells, names = cf.vehicle_documents(carve)
            self.assertEqual(cells, {0.0625: 1})
            self.assertEqual(names, {"SyntheticWagon"})

    def test_plain_directory_keeps_the_flat_bin_document_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = Path(tmp)
            self.write_document(carve / "document.bin")
            self.write_document(carve / "ignored.txt", model="SyntheticText", cell="0.03125")
            (carve / "nested").mkdir()
            self.write_document(carve / "nested" / "ignored.bin", model="SyntheticNested", cell="0.0325")
            (carve / "unrelated.bin").write_text("<Unrelated/>", encoding="utf-8", newline="\n")
            cells, names = cf.vehicle_documents(str(carve))
            self.assertEqual(cells, {0.0625: 1})
            self.assertEqual(names, {"SyntheticWagon"})


if __name__ == "__main__":
    unittest.main()
