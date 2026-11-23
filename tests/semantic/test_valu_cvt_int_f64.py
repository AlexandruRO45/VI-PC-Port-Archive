"""v_cvt_f64_i32, v_cvt_f64_u32, v_cvt_i32_f64 and v_cvt_u32_f64 are all implemented in tools/rdna2spirv.py:
- v_cvt_f64_i32 and v_cvt_f64_u32 since #2521 (tested by tests/semantic/test_int32_to_f64.py)
- v_cvt_i32_f64 and v_cvt_u32_f64 in #3325 (tested by tests/semantic/test_valu_f64_round_cvt.py)

All four ops have been dropped from the refused-by-name list.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402

CASES = []


class CvtIntF64Unsupported(unittest.TestCase):
    def test_each_op_is_refused_by_name(self):
        for name, ins in CASES:
            with self.subTest(ins=ins):
                lines = ["v_mov_b32 v1, 5", ins, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
                with self.assertRaises(rdna2spirv.Unsupported) as refused:
                    support.translate_lines("ps_cvt_int_f64", lines)
                self.assertEqual(str(refused.exception), name)

    def test_a_listing_without_them_still_translates(self):
        lines = ["v_mov_b32 v1, 5", "v_cvt_f32_i32_e32 v2, v1", "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
        self.assertIn("OpFunction", support.translate_lines("ps_cvt_i32_ok", lines))


if __name__ == "__main__":
    unittest.main()
