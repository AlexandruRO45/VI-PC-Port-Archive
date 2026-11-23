"""v_mul_legacy_f32 signed zeros and omod against the words measured natively on RDNA2 (cs_legacy_mul_zero.asm).

-0 * 3.0 and 3.0 * -0 are +0 (the DX9 zero override); a product that omod scales into the subnormal range is flushed
afterwards and keeps its sign (-0), while a flushed -0 that enters omod becomes +0.
"""
import os
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import LEGACY_MUL_ZERO_WORDS
from compute_fixture import compute_header

OUT_V = [0, 0, 512, 0x31016FAC]   # the output V# in s[0:3]: base 0, 512 bytes, FORMAT 32_FLOAT, DST_SEL xyzw


class LegacyMulZeroTest(unittest.TestCase):
    def test_hardware_words(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_legacy_mul_zero.asm")
        spv = support.translate_fixture("cs_legacy_mul_zero", fssm=bytes(compute_header(asm)))
        self.assertEqual(LEGACY_MUL_ZERO_WORDS[:4], [0, 0, 0x80000000, 0])
        for lane in (0, 1):
            with self.subTest(lane=lane):
                memory = {}
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                    push_constants=OUT_V + [0] * 28, memory=memory)
                self.assertEqual(memory, {16 * lane + 4 * k: w for k, w in enumerate(LEGACY_MUL_ZERO_WORDS[:4])})


if __name__ == "__main__":
    unittest.main()
