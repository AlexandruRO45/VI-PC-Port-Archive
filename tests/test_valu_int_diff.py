"""Tests for tools/valu_int_diff.py: integer & packed-16bit VALU differential fuzzer (no game data, stdlib only)."""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import valu_int_diff  # noqa: E402


class ValuIntDiffTests(unittest.TestCase):
    def test_self_test_and_negative_controls(self):
        valu_int_diff.self_test()

    def test_seeded_100_programs_match(self):
        fail = valu_int_diff.run_fuzz(programs=100, seed=11, max_ops=16, spirv_val=False)
        self.assertIsNone(fail)

    def test_reference_catches_perturbed_translator_output(self):
        # Verify our comparison would catch a wrong clamp/wrap or op_sel output.
        lines = [
            "v_mov_b32_e32 v0, 0xffffffff",
            "v_mov_b32_e32 v1, 0x00000005",
            "v_add_nc_u32_e64 v0, v0, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        got = valu_int_diff.translate_and_run(lines, name="ps_valu_int_neg_ctrl")
        self.assertEqual(got, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF])
        wrong_unclamped = valu_int_diff.eval_ref_op("v_add_nc_u32", [0xFFFFFFFF, 5], 0, clamp=False)
        self.assertNotEqual(got[0], wrong_unclamped)


class ArgumentValidation(unittest.TestCase):
    """--programs and --max-ops must be at least 1.

    Before this, `--programs 0` and `--programs -2` ran no program and printed
    "0 programs (seed 1): 0 pass, 0 mismatch" with exit 0, which reads as a
    pass, and `--max-ops 0` died inside random.randint(1, 0), reported as a
    harness error (exit 2) instead of a usage error.
    """

    def _rejected(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                valu_int_diff.main(list(argv))
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--programs and --max-ops must be at least 1", err.getvalue())

    def test_programs_zero_is_rejected(self):
        self._rejected("--programs", "0")

    def test_programs_negative_is_rejected(self):
        self._rejected("--programs", "-1")

    def test_max_ops_zero_is_rejected(self):
        self._rejected("--max-ops", "0")

    def test_a_valid_small_run_still_works(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(valu_int_diff.main(["--programs", "2", "--max-ops", "2"]), 0)
        self.assertIn("2 programs (seed 1): 2 pass, 0 mismatch", out.getvalue())


if __name__ == "__main__":
    unittest.main()
