"""v_med3_i16 / v_med3_u16 with a constant operand against the words measured natively on RDNA2 (cs_med3_i16.asm).

The median is the sum minus the smallest and the largest; a clamp or a min/max form of it came out as a bound on the
AMD Windows driver when one operand was a constant. The interpreter cannot show that, the GPU test does; this runs the
fixture through the translator and the interpreter for the values."""
import os
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import MED3_I16_WORDS
from compute_fixture import compute_header

OUT_V = [0, 0, 512, 0x31016FAC]   # the output V# in s[0:3]: base 0, 512 bytes, FORMAT 32_FLOAT, DST_SEL xyzw
FILL = 0xEEEEEEEE


class Med3I16Test(unittest.TestCase):
    def test_hardware_words(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_med3_i16.asm")
        spv = support.translate_fixture("cs_med3_i16", fssm=bytes(compute_header(asm)))
        # Structure: the median is the sum minus the smallest and the largest, not a clamp or a min/max/select form
        # (those came out as a bound on the AMD driver with a constant operand; the values above cannot show it)
        self.assertNotIn("Clamp", spv)
        self.assertGreaterEqual(spv.count("OpISub %int"), 2)
        for lane in (0, 1):
            with self.subTest(lane=lane):
                memory = {16 * lane + 4 * k: FILL for k in range(4)}   # the buffer fill the shader reads back
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                    push_constants=OUT_V + [0] * 28, memory=memory)
                self.assertEqual([memory[16 * lane + 4 * k] for k in range(4)], MED3_I16_WORDS[:4])


if __name__ == "__main__":
    unittest.main()
