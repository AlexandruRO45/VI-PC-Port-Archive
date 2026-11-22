"""runtime/test/shaders/w64_graphics.py: the TEST-363 variants' bookkeeping (no device needed).

The SPIR-V patch that adds MaximallyReconvergesKHR, the coverage of vs_tri_partial and the words a partial draw must
give, and the check names the runner accepts."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import w64_graphics as w

ENTRY_ID = 4


def _string(text):
    raw = text.encode() + b"\0"
    raw += b"\0" * (-len(raw) % 4)
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def _module():
    """A minimal fragment module: OpCapability Shader, OpMemoryModel, OpEntryPoint Fragment %4 "main",
    OpExecutionMode %4 OriginUpperLeft (the rest of a real module does not matter to the patch)."""
    name = _string("main")
    return [0x07230203, 0x00010500, 0, 5, 0,
            (2 << 16) | 17, 1,
            (3 << 16) | 14, 0, 1,
            ((3 + len(name)) << 16) | 15, 4, ENTRY_ID, *name,
            (3 << 16) | 16, ENTRY_ID, 7]


def _instructions(words):
    i, out = 5, []
    while i < len(words):
        count = words[i] >> 16
        out.append((words[i] & 0xFFFF, words[i + 1:i + count]))
        i += count
    return out


def _patch(words):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.spv", Path(tmp) / "out.spv"
        src.write_bytes(struct.pack("<%dI" % len(words), *words))
        w.with_maximal_reconvergence(src, dst)
        data = dst.read_bytes()
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


class MaximalReconvergencePatch(unittest.TestCase):
    def test_adds_the_extension_after_the_capabilities_and_the_mode_after_the_entry_point(self):
        ops = _instructions(_patch(_module()))
        self.assertEqual([op for op, _ in ops], [17, 10, 14, 15, 16, 16])
        self.assertEqual(ops[1][1], _string(w.MR_EXTENSION))
        self.assertEqual(ops[4][1], [ENTRY_ID, w.MAXIMALLY_RECONVERGES])
        self.assertEqual(ops[5][1], [ENTRY_ID, 7])          # the module's own mode is kept

    def test_a_module_that_has_the_mode_is_copied_unchanged(self):
        once = _patch(_module())
        self.assertEqual(_patch(once), once)

    def test_a_module_without_an_entry_point_is_refused(self):
        words = _module()[:10]                             # header, capability, memory model
        with self.assertRaises(ValueError):
            _patch(words)


class PartialCoverage(unittest.TestCase):
    def test_no_pixel_centre_lies_on_the_long_edge(self):
        # 3 (x + 0.5) + 4 (y + 0.5) == 192 would need 3x + 4y == 188.5.
        self.assertFalse(any(2 * (3 * x + 4 * y) == 377 for x in range(w.W) for y in range(w.H)))

    def test_the_edge_cuts_32_quads_with_64_covered_pixels(self):
        cut = [(qx, qy) for qy in range(0, w.H, 2) for qx in range(0, w.W, 2)
               if 0 < sum(w.covered(qx + dx, qy + dy) for dx in (0, 1) for dy in (0, 1)) < 4]
        self.assertEqual(len(cut), 32)
        self.assertEqual(sum(w.covered(qx + dx, qy + dy) for qx, qy in cut for dx in (0, 1) for dy in (0, 1)), 64)

    def test_an_uncovered_pixel_keeps_the_clear_value(self):
        expect = w.expectation("ps_w64_a_skip", "partial")
        self.assertFalse(w.covered(63, 63))
        self.assertEqual(expect(63 * w.W + 63), (w.CLEAR,) * 4)
        self.assertEqual(expect(0), w.FIXTURES["ps_w64_a_skip"][1](0))

    def test_quad_live_counts_the_covered_pixels_of_the_quad(self):
        full, partial = w.expectation("ps_w64_a_quad_live"), w.expectation("ps_w64_a_quad_live", "partial+mr")
        i = 0 * w.W + 62                                   # (62, 0): only it is covered in its quad
        self.assertEqual(full(i), (4, 62, 0, 0))
        self.assertEqual(partial(i), (1, 62, 0, 0))

    def test_a_skip_whole_quad_mask_counts_covered_pixels_only(self):
        expect = w.expectation("ps_w64_a_skip", "partial")
        # (60, 2): its quad's x % 4 == 1 column (61) is uncovered at rows 2 and 3, so the helper's 0 bit leaves r1 0.
        self.assertFalse(w.covered(61, 2) or w.covered(61, 3))
        self.assertEqual(expect(2 * w.W + 60)[1], 0)
        self.assertEqual(w.FIXTURES["ps_w64_a_skip"][1](2 * w.W + 60)[1], 1)   # full coverage: 1
        self.assertEqual(expect(0)[1], 1)                                       # (0, 0): (1, 0) is covered
        self.assertEqual(sum(expect(i)[1] != w.FIXTURES["ps_w64_a_skip"][1](i)[1]
                             for i in range(w.W * w.H) if w.covered(w._x(i), w._y(i))), 8)

    def test_only_covered_pixels_are_wanted_by_the_ballot_atomic(self):
        words = [(int(w._x(i) % 4 == 1), 0, 0, 0) if w.covered(w._x(i), w._y(i)) else (w.CLEAR,) * 4
                 for i in range(w.W * w.H)]
        want = sorted(i + 1 for i in range(w.W * w.H) if w._x(i) % 4 == 1 and w.covered(w._x(i), w._y(i)))
        heap = [len(want)] + [0] * 63 + want + [0] * (16384 - 64 - len(want))
        self.assertEqual(len(want), 392)
        self.assertEqual(w.expected_problem("ps_w64_b_ballot_atomic", words, heap, "partial"), "")
        heap[0] = 416
        self.assertIn("counter is 416", w.expected_problem("ps_w64_b_ballot_atomic", words, heap, "partial"))


class CheckNames(unittest.TestCase):
    def test_only_the_listed_pixel_fixtures_run_in_the_variants(self):
        # Listed by hand: a fixture joins once its partial-coverage words are derived (the permute ones are not).
        for name, (tier, _, _) in w.FIXTURES.items():
            for variant in w.VARIANTS:
                self.assertEqual(f"{name}@{variant}" in w.CHECKS, name in w.VARIANT_FIXTURES)
        self.assertTrue(all(n.startswith("ps_") and w.FIXTURES[n][0] in "AB" for n in w.VARIANT_FIXTURES))

    def test_an_unexpected_pass_is_reported_but_does_not_fail_the_run(self):
        # Whether a helper joins a ballot without MaximallyReconvergesKHR is up to the driver (TEST-363).
        name = next(iter(w.KNOWN))
        with mock.patch.object(w, "check", return_value=""), mock.patch.object(w, "CHECKS", [name]), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(w.main(["unused"]), 0)
        self.assertIn("XPASS", out.getvalue())

    def test_a_known_failure_does_not_fail_the_run(self):
        name = next(iter(w.KNOWN))
        with mock.patch.object(w, "check", return_value="64 of 4096 pixels differ"), \
                mock.patch.object(w, "CHECKS", [name]), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(w.main(["unused"]), 0)
        self.assertIn("XFAIL", out.getvalue())

    def test_split_check(self):
        self.assertEqual(w.split_check("ps_w64_b_waterfall@partial+mr"), ("ps_w64_b_waterfall", "partial+mr"))
        self.assertEqual(w.split_check("ps_w64_b_waterfall"), ("ps_w64_b_waterfall", ""))

    def test_known_failures_name_real_checks(self):
        self.assertTrue(set(w.KNOWN) <= set(w.CHECKS))

    def test_run_gpu_tests_leaves_the_partial_vertex_shader_here(self):
        self.assertIn(w.PARTIAL_VS, w.ONLY_HERE)
        self.assertTrue(set(w.FIXTURES) <= w.ONLY_HERE)


if __name__ == "__main__":
    unittest.main()
