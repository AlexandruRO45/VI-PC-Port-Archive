"""Partial exports preserve earlier components; all-off exports emit no store.

    python -I tests/semantic/test_vs_partial_export.py

On RDNA2, EXP's EN bits gate each component (EN=0 for off components).
A second partial export to the same target (such as VS paramN or pos0)
preserves components written by earlier partial exports rather than clobbering
them with 0.0. An all-off export (all EN=0) writes nothing to memory/output.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401


class PartialExportTest(unittest.TestCase):
    def test_vs_partial_param_export_preserves_earlier_components(self):
        lines = [
            "v_mov_b32_e32 v0, 1.0",
            "v_mov_b32_e32 v1, 2.0",
            "exp param0 v0, off, off, off",
            "exp param0 off, v1, off, off",
            "exp pos0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("vs_partial_param", lines)
        # First export stores (v0, old.y, old.z, old.w)
        # Second export loads old, extracts old.x, and stores (old.x, v1, old.z, old.w)
        stores = [ln for ln in spv.splitlines() if "OpStore %out_param0" in ln]
        self.assertEqual(len(stores), 2)

        # Check that the second export extracts component 0 from the loaded prior value
        extracts = re.findall(r"(%\w+) = OpCompositeExtract %float (%\w+) 0", spv)
        self.assertTrue(extracts, "second export should extract component 0 from prior output")

    def test_vs_all_off_export_emits_no_store(self):
        lines = [
            "v_mov_b32_e32 v0, 1.0",
            "exp param0 off, off, off, off",
            "exp pos0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("vs_all_off_param", lines)
        # Variable declared in interface
        self.assertIn("%out_param0", spv)
        # But no OpStore to param0
        stores = [ln for ln in spv.splitlines() if "OpStore %out_param0" in ln]
        self.assertEqual(len(stores), 0)

    def test_ps_mrt_partial_export_marks_partial_color(self):
        lines = [
            "v_mov_b32_e32 v0, 1.0",
            "v_mov_b32_e32 v1, 2.0",
            "exp mrt0 v0, off, off, off done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_partial_mrt", lines)
        self.assertIn("OpStore %out_mrt0", spv)
        extracts = re.findall(r"(%\w+) = OpCompositeExtract %float (%\w+) 1", spv)
        self.assertTrue(extracts)


if __name__ == "__main__":
    unittest.main()
