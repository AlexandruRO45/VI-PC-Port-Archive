"""cs_salu_const_fold: the constant-input s_bfe_u32 folds at translation time (cs_salu_const_fold.asm, out[1]).

The GPU readback checks the eight words, but a lost fold computes the same value with a runtime extract, so the
readback stays green. This checks the translated module instead: no extract with the constant offset and width
survives, and the load for out[1] uses the folded byte offset 0x18.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from compute_fixture import compute_header  # noqa: E402

BFE_4_8 = r"OpBitFieldUExtract %uint %\w+ %c4 %c8\b"   # s_bfe_u32 control 0x00080004: offset 4, width 8
LOAD_AT_0X18 = r"OpIAdd %ulong %\w+ %cl18\b"


class SaluConstFoldFixture(unittest.TestCase):
    def test_constant_bfe_folds(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_salu_const_fold.asm")
        spv = support.translate_fixture("cs_salu_const_fold", fssm=bytes(compute_header(asm)))
        self.assertNotRegex(spv, BFE_4_8)
        self.assertRegex(spv, LOAD_AT_0X18)

    def test_dynamic_bfe_keeps_extract(self):
        spv = support.translate_lines("ps_bfe_dynamic", ["s_bfe_u32 s5, s1, 0x00080004", "v_mov_b32 v0, s5",
                                                         "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertRegex(spv, BFE_4_8)


if __name__ == "__main__":
    unittest.main()
