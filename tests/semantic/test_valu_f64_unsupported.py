"""Binary64 VALU instructions that tools/rdna2spirv.py does not translate yet are refused, not mistranslated.

On main the translator lowers `v_add_f64`, `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64`, `v_cvt_f32_f64` and the two
`v_frexp_*_f64` forms (their tests: test_valu_add_f64.py, test_valu_rcp_f64.py, test_valu_rsq_f64.py,
test_valu_sqrt_f64.py, test_valu_cvt_f32_f64.py, test_alu_frexp_f64.py). The instructions below are real RDNA2 VOP1 /
VOP2 / VOP3 instructions that it still raises `Unsupported` for, so a shader that uses one is not translated.

This file replaces test_valu_{mul,fma,fract,minmax}_f64.py, which built their own register file and binary64
arithmetic and never reached `rdna2spirv`, so they passed whether or not the instruction was translated. When one of
these gets a lowering, delete its entry here and give it a test that runs the translated shader (see
test_valu_sqrt_f64.py). `v_ldexp_f64` is left out on purpose: a lowering
for it is open (the f64 <-> 32-bit integer conversions are lowered on main, test_valu_cvt_int_f64.py).

Standard library only: no GPU, no LLVM, no game data.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
import rdna2spirv  # noqa: E402  (support puts tools/ on sys.path)

REFUSED = [
    "v_mul_f64 v[4:5], v[0:1], v[2:3]",
    "v_fma_f64 v[4:5], v[0:1], v[2:3], v[6:7]",
    "v_fract_f64 v[4:5], v[0:1]",
    "v_min_f64 v[4:5], v[0:1], v[2:3]",
    "v_max_f64 v[4:5], v[0:1], v[2:3]",
    "v_cvt_f64_f32 v[4:5], v0",
]


class UnsupportedF64(unittest.TestCase):
    def test_each_instruction_is_refused_by_name(self):
        for line in REFUSED:
            with self.subTest(line=line):
                prog = [line, "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"]
                with self.assertRaises(rdna2spirv.Unsupported) as caught:
                    support.translate_lines("ps_f64_unsupported", prog)
                self.assertIn(line.split()[0], str(caught.exception))

    def test_the_translated_f64_instructions_are_not_in_the_list(self):
        for line in ("v_add_f64 v[4:5], v[0:1], v[2:3]", "v_rcp_f64 v[4:5], v[0:1]", "v_rsq_f64 v[4:5], v[0:1]",
                     "v_sqrt_f64 v[4:5], v[0:1]", "v_cvt_f32_f64 v4, v[0:1]", "v_frexp_mant_f64 v[4:5], v[0:1]"):
            with self.subTest(line=line):
                support.translate_lines("ps_f64_supported", [line, "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
