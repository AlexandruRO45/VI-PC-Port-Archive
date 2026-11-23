import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sgd5_pass_state_groups import inventory


def stored(effect=100, selector=0, **states):
    return dict(db=1, effect_start=effect, programs={"0": "synthetic_v", "1": "synthetic_p"},
                stages={"0": "vs", "1": "ps"}, raster=selector, depth=0, blend=0, **states)


class InventoryTest(unittest.TestCase):
    def test_complete_values_across_effects(self):
        common = {"depth_state": {"depth_enable": 0}, "blend_state": {"prefix_raw": [1]}}
        a = stored(raster_state={"cull": "back", "raw_from_2": [0]}, **common)
        b = stored(200, 1, raster_state={"raw_from_2": [0], "cull": "back"}, **common)
        c = stored(raster_state={"cull": "back", "raw_from_2": [1]}, **common)
        report = inventory([a, b, c])
        self.assertEqual(report, inventory([c, b, a]))
        self.assertEqual(sorted(v["passes"] for v in report["chains"][0]["variants"]), [1, 2])

    def test_local_identity_and_empty_tables(self):
        rows = [stored(), stored(), stored(101), stored(selector=1),
                dict(stored(), db=2), stored(raster_state=None),
                stored(101, raster_state=None), stored(raster_state={"fill_mode": 3})]
        self.assertEqual(inventory(rows)["summary"], {
            "graphics_passes": 8, "shader_chains": 1, "selection_groups": 7,
            "chains_with_multiple_selections": 1})
        with self.assertRaises(ValueError):
            inventory([stored(selector=1, raster_state=None)])

    def test_stage_chains(self):
        mesh = stored()
        mesh["stages"]["0"] = "ms"
        geom = stored()
        geom["programs"]["2"] = "synthetic_g"
        geom["stages"]["2"] = "gs"
        compute = dict(stored(), programs={"5": "synthetic_c"}, stages={"5": "cs"})
        vs = dict(stored(), programs={"0": "synthetic_v"}, stages={"0": "vs"})
        report = inventory([stored(), mesh, geom, compute, vs])
        self.assertEqual(report["summary"]["graphics_passes"], 3)
        self.assertEqual(report["summary"]["shader_chains"], 3)


if __name__ == "__main__":
    unittest.main()
