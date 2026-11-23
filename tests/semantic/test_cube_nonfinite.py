"""v_cubeid/sc/tc/ma_f32 with NaN, infinite and denormal operands against the words measured natively on RDNA2
(cs_cube_nonfinite.asm): compares a NaN fails and that read a denormal as zero, raw words passed through."""
import os
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import CUBE_NONFINITE_WORDS
from compute_fixture import compute_header

OUT_V = [0, 0, 1024, 0x31016FAC]   # the output V# in s[0:3]: base 0, 1024 bytes, FORMAT 32_FLOAT, DST_SEL xyzw


class CubeNonfiniteTest(unittest.TestCase):
    def test_hardware_words(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_cube_nonfinite.asm")
        spv = support.translate_fixture("cs_cube_nonfinite", fssm=bytes(compute_header(asm)))
        for lane in (0, 1):
            with self.subTest(lane=lane):
                memory = {}
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                    push_constants=OUT_V + [0] * 28, memory=memory)
                self.assertEqual(memory, {32 * lane + 4 * k: w for k, w in enumerate(CUBE_NONFINITE_WORDS)})


if __name__ == "__main__":
    unittest.main()
