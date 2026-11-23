"""tools/gbuffer.py: shadow VS matching and checking, on hand-made inputs."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import gbuffer  # noqa: E402

SHADOW_TEXT = "\n".join([
    "\ts_mov_b32 s4, s9",
    "\ts_load_dwordx4 s[0:3], s[4:5], 0x40",
    "\ts_load_dwordx4 s[4:7], s[4:5], 0x60",
    "\ts_or_b32 s25, s12, 0x10000",
    "\ts_mov_b32 s24, s11",
    "\ts_mov_b32 s27, 0x11014fac",
    "\tbuffer_load_dwordx3 v[0:2], v0, s[24:27], 0 offen",
    "\ts_buffer_load_dwordx16 s[0:15], s[4:7], 0x160",
    "\texp pos0 v0, v1, v2, v3 done",
    "\texp param0 v0, v0, v0, v1",
]) + "\n"

ENTRY = {"stage": "vs", "bindings": [], "user_sgpr_reads": [{"kind": k, "sgpr": s} for k, s in gbuffer.SHADOW_USER],
         "input_signature": [{"semantic": "SV_VertexID", "semantic_index": 0}]}


class ShadowVsMatches(unittest.TestCase):
    def test_matches(self):
        self.assertTrue(gbuffer.shadow_vs_matches(ENTRY, SHADOW_TEXT))

    def test_rejects_bindings(self):
        entry = ENTRY.copy()
        entry["bindings"] = [0]
        self.assertFalse(gbuffer.shadow_vs_matches(entry, SHADOW_TEXT))

    def test_rejects_image_load(self):
        text = SHADOW_TEXT + "\timage_load v0, v0, s[0:7]\n"
        self.assertFalse(gbuffer.shadow_vs_matches(ENTRY, text))

    def test_rejects_extra_param_export(self):
        text = SHADOW_TEXT + "\texp param1 v0, v0, v0, v1\n"
        self.assertFalse(gbuffer.shadow_vs_matches(ENTRY, text))

    def test_rejects_user_sgpr_reads(self):
        entry = ENTRY.copy()
        entry["user_sgpr_reads"] = entry["user_sgpr_reads"][:-1]
        self.assertFalse(gbuffer.shadow_vs_matches(entry, SHADOW_TEXT))

    def test_rejects_input_signature(self):
        entry = ENTRY.copy()
        entry["input_signature"] = [{"semantic": "POSITION", "semantic_index": 0}]
        self.assertFalse(gbuffer.shadow_vs_matches(entry, SHADOW_TEXT))

    def test_rejects_missing_layout_line(self):
        text = "\n".join(line for line in SHADOW_TEXT.splitlines() if "0x160" not in line) + "\n"
        self.assertFalse(gbuffer.shadow_vs_matches(ENTRY, text))

    def test_rejects_missing_pos_export(self):
        text = SHADOW_TEXT.replace("\texp pos0 v0, v1, v2, v3 done\n", "")
        self.assertFalse(gbuffer.shadow_vs_matches(ENTRY, text))

    def test_rejects_stage(self):
        entry = ENTRY.copy()
        entry["stage"] = "ps"
        self.assertFalse(gbuffer.shadow_vs_matches(entry, SHADOW_TEXT))


class PickShadowVs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for sub in ("spirv", os.path.join("dxbc", "disasm")):
            os.makedirs(os.path.join(self.root, sub))
        self.index = {}

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, name, entry, text, spv=b"\x03\x02\x23\x07"):
        self.index[name] = entry
        with open(os.path.join(self.root, "spirv", name + ".spv"), "wb") as f:
            f.write(spv)
        with open(os.path.join(self.root, "dxbc", "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(self.root, "spirv", "index.json"), "w", encoding="utf-8") as f:
            json.dump(self.index, f)

    def test_returns_none_for_empty_folder(self):
        self.assertIsNone(gbuffer.pick_shadow_vs(self.root))

    def test_picks_in_name_order(self):
        self.add("b_vs", ENTRY, SHADOW_TEXT)
        self.add("a_vs", ENTRY, SHADOW_TEXT)
        path = gbuffer.pick_shadow_vs(self.root)
        self.assertTrue(path.endswith("a_vs.spv"))

    def test_skips_missing_spv(self):
        self.add("a_vs", ENTRY, SHADOW_TEXT)
        self.add("b_vs", ENTRY, SHADOW_TEXT)
        os.remove(os.path.join(self.root, "spirv", "a_vs.spv"))
        path = gbuffer.pick_shadow_vs(self.root)
        self.assertTrue(path.endswith("b_vs.spv"))


class CheckShadow(unittest.TestCase):
    def test_pass(self):
        runs = [{"map_texels": 10, "lit": 5, "shadowed": 3, "away": 1},
                {"map_texels": 10, "lit": 2, "shadowed": 6, "away": 4}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertTrue(ok)
        self.assertIn("5 lit", msg)

    def test_fail_equal_runs(self):
        runs = [{"map_texels": 10, "lit": 5, "shadowed": 3, "away": 1},
                {"map_texels": 10, "lit": 5, "shadowed": 3, "away": 4}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertFalse(ok)

    def test_fail_zero_shadowed(self):
        runs = [{"map_texels": 10, "lit": 5, "shadowed": 0, "away": 1},
                {"map_texels": 10, "lit": 2, "shadowed": 6, "away": 4}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertFalse(ok)

    def test_fail_zero_lit(self):
        runs = [{"map_texels": 10, "lit": 0, "shadowed": 3, "away": 1},
                {"map_texels": 10, "lit": 2, "shadowed": 6, "away": 4}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertFalse(ok)

    def test_fail_zero_map_texels(self):
        runs = [{"map_texels": 0, "lit": 5, "shadowed": 3, "away": 1},
                {"map_texels": 10, "lit": 2, "shadowed": 6, "away": 4}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertFalse(ok)

    def test_fail_few_shadowed(self):
        runs = [{"map_texels": 10, "lit": 600, "shadowed": 9, "away": 391},
                {"map_texels": 10, "lit": 200, "shadowed": 300, "away": 500}]
        self.assertFalse(gbuffer.check_shadow(runs)[0])
        runs[0]["shadowed"], runs[0]["away"] = 10, 390
        self.assertTrue(gbuffer.check_shadow(runs)[0])

    def test_fail_lit_barely_moves(self):
        runs = [{"map_texels": 10, "lit": 500, "shadowed": 100, "away": 400},
                {"map_texels": 10, "lit": 401, "shadowed": 199, "away": 400}]
        self.assertFalse(gbuffer.check_shadow(runs)[0])
        runs[1]["lit"], runs[1]["shadowed"] = 400, 200
        self.assertTrue(gbuffer.check_shadow(runs)[0])

    def test_single_pixel_difference_fails(self):
        runs = [{"map_texels": 10, "lit": 82767, "shadowed": 10, "away": 31726},
                {"map_texels": 10, "lit": 82766, "shadowed": 11, "away": 31726}]
        self.assertFalse(gbuffer.check_shadow(runs)[0])

    def test_fail_single_run(self):
        runs = [{"map_texels": 10, "lit": 5, "shadowed": 3, "away": 1}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertFalse(ok)

    def test_fail_empty_runs(self):
        runs = [{}, {}]
        ok, msg = gbuffer.check_shadow(runs)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
