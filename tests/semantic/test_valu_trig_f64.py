"""v_sin_f64 and v_cos_f64 (not RDNA2 opcodes: the ISA has v_sin_f32 and v_cos_f32 only): the translator has no lowering for binary64 trigonometry, so a listing that uses either spelling is refused
with `Unsupported` instead of being translated to something that is not the instruction. This pins the refusal;
when a lowering lands, the test is the place for its cases. Translator-backed, standard library only: no GPU, no
LLVM, no game data.
"""
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402


def translate(op, srcs):
    return support.translate_lines("ps_trig_f64", [
        "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40000000", "v_mov_b32 v4, 0", "v_mov_b32 v5, 0x40080000",
        "v_mov_b32 v6, 0", "v_mov_b32 v7, 0x3ff00000",
        f"{op} v[0:1], {srcs}", "exp mrt0 v0, v1, v1, v1 done vm", "s_endpgm"])


class TrigF64Refused(unittest.TestCase):
    def test_refused_not_guessed(self):
        for op, srcs in (("v_sin_f64", "v[2:3]"), ("v_cos_f64", "v[2:3]"), ("v_trig_f64", "v[2:3]")):
            with self.subTest(op=op), self.assertRaises(Unsupported) as cm:
                translate(op, srcs)
            self.assertIn(op.split("_e64")[0], str(cm.exception))


if __name__ == "__main__":
    unittest.main()
