"""Synthetic export windows and keyed model observations."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import world_drawable_verify as verifier
import world_names


class VerifyTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        return path

    def test_join_counts_and_window_edges(self):
        candidates = []
        for name, offset in [("one", 10), ("two", 12), ("three", 13), ("four", 14), ("five", 20)]:
            candidates.append({"archetype": format(world_names.joaat(name), "x"),
                               "key": format(world_names.hash64(name), "x"), "offsets": [offset],
                               "status": "candidate", "placements": 3})
        # A 32-bit match alone cannot confirm the prediction.
        candidates[1]["archetype"] = candidates[0]["archetype"]
        candidates.append(dict(candidates[0], status="ambiguous", offsets=[10, 15]))
        predictions = self.write("lookup", {"archive": "source.rpf", "candidates": candidates})
        export = {"src": "source.rpf", "Scan": {"Start": 10, "Length": 10},
                  "Items": [{"Name": "one.#dr", "Offset": 10}, {"Name": "one", "Offset": 12},
                            {"Name": "three", "Offset": 18}, {"Name": "", "Offset": 14}]}
        first = self.write("first", export)
        second = self.write("second", dict(export, Items=[dict(export["Items"][0], Drawable=None)]))
        result = verifier.verify(predictions, [first, second, first])
        self.assertEqual([r["status"] for r in result["rows"]],
                         ["confirmed", "conflicting_name", "conflicting_offset", "missing_evidence",
                          "outside_windows"])
        self.assertEqual(result["placements"], {s: 3 for s in result["identities"]})
        self.assertEqual(result["rows"][0]["exact_observations"], 2)
        self.assertEqual(result["rows"][0]["duplicate_observations"], 1)
        self.assertEqual(result["rows"][0]["distinct_drawable_children"], 0)
        self.assertEqual(result["rows"][0]["other_identity_offsets"], [12])
        full = self.write("full", dict(export, Scan={}))
        self.assertEqual(verifier.verify(predictions, [full])["identities"]["missing_evidence"], 2)

    def test_distinguish_drawable_children_from_repeats(self):
        candidate = {"archetype": format(world_names.joaat("twin"), "x"),
                     "key": format(world_names.hash64("twin"), "x"), "offsets": [0x3010],
                     "status": "candidate", "placements": 3}
        predictions = self.write("lookup", {"archive": "source.rpf", "candidates": [candidate]})
        children = [{"Name": "twin.#dr", "Offset": 0x3010, "Drawable": drawable}
                    for drawable in (0, 0x700)]
        first = self.write("first", {"src": "source.rpf", "Scan": {"Start": 0, "Length": 0x5000},
                                     "Items": children})
        second = self.write("second", {"src": "source.rpf", "Scan": {"Start": 0x2010, "Length": 0x2000},
                                       "Items": children[:1]})

        for manifests, exact, repeats in [([first], 2, 0), ([first, second], 3, 1)]:
            with self.subTest(manifests=manifests):
                result = verifier.verify(predictions, manifests)
                self.assertEqual(result["identities"], {"confirmed": 1})
                self.assertEqual(result["placements"], {"confirmed": 3})
                row, = result["rows"]
                self.assertEqual(row["exact_observations"], exact)
                self.assertEqual(row["duplicate_observations"], repeats)
                self.assertEqual(row["distinct_drawable_children"], 2)
                self.assertEqual(result, verifier.verify(predictions, [*manifests, first]))

    def test_reject_invalid_drawable_offsets(self):
        predictions = self.write("lookup", {"archive": "source.rpf", "candidates": []})
        for drawable in ([1], {"x": 1}, False, -1, 1.0, "1"):
            with self.subTest(drawable=drawable):
                manifest = self.write("manifest.json", {"src": "source.rpf", "Scan": {},
                                                        "Items": [{"Name": "one.#dr", "Offset": 16,
                                                                   "Drawable": drawable}]})
                with self.assertRaises(ValueError) as error:
                    verifier.verify(predictions, [manifest])
                self.assertIn(os.path.normcase(os.path.realpath(manifest)), str(error.exception))
                self.assertIn("Drawable", str(error.exception))

    def test_reject_unrelated_archive_and_bad_bounds(self):
        predictions = self.write("lookup", {"archive": "source.rpf", "candidates": []})
        for source, scan in [("elsewhere/source.rpf", {}), ("source.rpf", {"Start": 0}),
                             ("source.rpf", {"Start": True, "Length": 5}),
                             ("source.rpf", {"Start": 0, "Length": 2})]:
            manifest = self.write("manifest", {"src": source, "Scan": scan, "Items": [{"Offset": 2}]})
            with self.subTest(source=source, scan=scan), self.assertRaises(ValueError):
                verifier.verify(predictions, [manifest])


if __name__ == "__main__":
    unittest.main()
