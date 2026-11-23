"""CPU replay of the wave64 graphics runner on a fragment stage with fewer than 32 live lanes (TEST-433).

The frames copy what Intel Arc Graphics (Meteor Lake Xe-LPG, Windows 101.6314) drew where no fragment subgroup size
can be required: 16 live lanes (ps_live_width), every word of every tier A and B pixel fixture exact except the ones
NEEDS_LIVE_LANES lists, which read lane L ^ 16 and came back 0 (ds_bpermute, ds_swizzle) or another row's value
(v_permlanex16).
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import w64_graphics  # noqa: E402

W = w64_graphics.W


def _frame(name, broken=()):
    """The fixture's derived words, with the word indices in `broken` changed as a 16-lane stage leaves them."""
    expect = w64_graphics.FIXTURES[name][1]
    out = []
    for i in range(W * W):
        words = [0 if v is None else v for v in expect(i)]
        for k in broken:
            words[k] = 0 if words[k] else 1
        out.append(tuple(words))
    return out


class NarrowFragmentTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        for name in w64_graphics.FIXTURES:
            (Path(self.work.name) / f"{name}.spv").touch()
        for target, value in (("_device_min", [8]), ("_live_width", {})):
            patcher = patch.object(w64_graphics, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_check(self, name, frame, live):
        def draw(_tools, _spirv, _name, size, _tmp):
            return frame, None, False if size else None    # 64 and 32 could not be required

        with patch.object(w64_graphics, "draw", side_effect=draw), \
                patch.object(w64_graphics, "_measure_live_width", return_value=live) as probe, \
                patch.dict(os.environ, W64G_REF="", W64G_SAVE_REF=""):
            result = w64_graphics.check(self.work.name, self.work.name, name)
        self.probes = probe.call_count
        return result

    def test_fixture_inside_a_row_of_16_passes(self):
        for name in ("ps_w64_a_skip", "ps_w64_a_one_lane", "ps_w64_a_discard", "ps_w64_b_waterfall"):
            with self.subTest(name=name):
                self.assertEqual(self.run_check(name, _frame(name), 16), "")

    def test_only_the_words_that_need_32_lanes_are_skipped(self):
        for name, needs in w64_graphics.NEEDS_LIVE_LANES.items():
            with self.subTest(name=name):
                result = self.run_check(name, _frame(name, broken=tuple(needs)), 16)
                self.assertIsInstance(result, w64_graphics.Skip)
                self.assertIn("16 live lanes", result)
                for k in needs:
                    self.assertIn(f"r{k}", result)

    def test_a_wrong_word_inside_the_row_still_fails(self):
        for name, needs in w64_graphics.NEEDS_LIVE_LANES.items():
            inside = next(k for k in range(4) if k not in needs)
            with self.subTest(name=name):
                result = self.run_check(name, _frame(name, broken=tuple(needs) + (inside,)), 16)
                self.assertNotIsInstance(result, w64_graphics.Skip)
                self.assertIn("differ from the derived words", result)

    def test_below_16_live_lanes_or_unmeasured_is_a_skip(self):
        for live in (8, None):
            with self.subTest(live=live):
                w64_graphics._live_width.clear()
                result = self.run_check("ps_w64_a_skip", _frame("ps_w64_a_skip"), live)
                self.assertIsInstance(result, w64_graphics.Skip)
                self.assertIn("need at least 16", result)

    def test_32_live_lanes_checks_every_word(self):
        name = "ps_w64_a_bpermute"
        self.assertEqual(self.run_check(name, _frame(name), 32), "")
        w64_graphics._live_width.clear()
        result = self.run_check(name, _frame(name, broken=(2,)), 32)
        self.assertNotIsInstance(result, w64_graphics.Skip)
        self.assertIn("differ from the derived words", result)

    def test_the_width_is_measured_once(self):
        self.run_check("ps_w64_a_bpermute", _frame("ps_w64_a_bpermute", broken=(2,)), 16)
        self.run_check("ps_w64_a_skip", _frame("ps_w64_a_skip"), 16)
        self.assertEqual(self.probes, 0)       # the second check reuses the first one's measurement

    def test_listed_words_are_the_ones_that_read_lane_xor_16(self):
        here = Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"
        for name, needs in w64_graphics.NEEDS_LIVE_LANES.items():
            text = (here / f"{name}.asm").read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertIn(name, w64_graphics.FIXTURES)
                for k, lanes in needs.items():
                    self.assertEqual(lanes, 32)
                    line = next(l for l in text.splitlines() if l.startswith(f";  r{k} ="))
                    self.assertIn("16", line)


if __name__ == "__main__":
    unittest.main()
