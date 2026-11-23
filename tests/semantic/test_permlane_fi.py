"""v_permlane16_b32 / v_permlanex16_b32 with the source lane off in EXEC (SHADER-74), wave32, CPU values.

The input is the GPU test runtime/test/shaders/cs_w32_permlane_fi (hand-written, no game data; fixture from
regen_fixtures.py). EXEC is 0x55555555, so the source lane of every active lane is off, except in the identity
case. Expected words are the RDNA2 measurement from issue #374 (op_sel:[FI, BOUND_CTRL]):
FI=1 returns the off lane's value; FI=0 leaves the destination alone (BOUND_CTRL=0) or writes 0 (BOUND_CTRL=1);
a source lane that is on gives its value whatever FI / BOUND_CTRL are.
"""
import struct
import unittest

import support
import spirv_mini_interp as si
from test_dpp_eager_subgroup import conditional_subgroup_ops

# 4 user SGPRs (the output V#), 32x1x1 threads: build.py's default header with a 32-wide group.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

# permlane16 / permlanex16 x (FI=0 BC=0, FI=0 BC=1, FI=1 BC=0, source on); see the .asm header.
EXPECTED = [1000, 0, 4, 1, 1000, 0, 52, 49]
NAMES = ["permlane16 FI=0 BC=0 (not written)", "permlane16 FI=0 BC=1 (0)", "permlane16 FI=1 (value)",
         "permlane16 source on", "permlanex16 FI=0 BC=0 (not written)", "permlanex16 FI=0 BC=1 (0)",
         "permlanex16 FI=1 (value)", "permlanex16 source on"]


def run(spv):
    memory = si.run_workgroup(
        spv, 32, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    return [memory.get(4 * k) for k in range(8)]


class PermlaneSourceOff(unittest.TestCase):
    def test_all_cases(self):
        got = run(support.translate_fixture("cs_w32_permlane_fi", fssm=FSSM))
        for i, name in enumerate(NAMES):
            with self.subTest(case=name):
                self.assertEqual(got[i], EXPECTED[i])

    def test_no_permlane_shuffle_is_conditional(self):
        # SPIRV-Cross (MoltenVK) inlines a value used once into the expression that reads it: in an arm of a select
        # or on the right of an &&, a shuffle runs only in some lanes, and a lane that skips it gives undefined
        # data to the lanes reading from it (SHADER-529). FI=1 reads exactly from lanes off in EXEC.
        spv = support.translate_fixture("cs_w32_permlane_fi", fssm=FSSM)
        # At least the 8 value shuffles and the activity bit of the 6 permlanes with FI=0 (FI=1 takes the value
        # either way); a later write gate may add its own.
        self.assertGreaterEqual(spv.count("OpGroupNonUniformShuffle "), 14)
        self.assertEqual(conditional_subgroup_ops(spv), [])


if __name__ == "__main__":
    unittest.main()
