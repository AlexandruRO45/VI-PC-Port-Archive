"""cs_w32_movrel: the GPU test of v_movrels_b32, v_movreld_b32 and v_movrelsd_b32, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_movrel.asm has lane 0 of a wave32 group store one word per relative move (M0 = 0, 3 and 82
for v_movrels_b32, 2 and 100 for v_movreld_b32, 1 and 48 for v_movrelsd_b32; derivations in its header). This runs
the fixture (tests/semantic/regen_fixtures.py) through spirv_mini_interp, checks the eight words against a plain
Python model of the three ops (the whole of M0 is the offset, on the source side for movrels, the destination side
for movreld and both sides for movrelsd), and checks that check.py and scripts/run_gpu_tests.py expect the same words.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# 4 user SGPRs (the output V#), workgroup id in s4, no LDS, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def reference():
    v = [0] * 256
    for r, x in {10: 0xA0, 11: 0xA1, 12: 0xA2, 13: 0xA3, 14: 0xA4, 15: 0xA5, 30: 0x30, 32: 0x32, 41: 0x41,
                 58: 0x58, 92: 0x92, 98: 0x98, 131: 0x131}.items():
        v[r] = x
    for m0 in (0, 3, 82):
        v[{0: 20, 3: 21, 82: 22}[m0]] = v[10 + m0]                  # v_movrels_b32 vN, v10
    v[30 + 2] = v[14]                                               # v_movreld_b32 v30, v14, M0 = 2
    v[31 + 100] = v[15]                                             # v_movreld_b32 v31, v15, M0 = 100
    v[40 + 1] = v[10 + 1]                                           # v_movrelsd_b32 v40, v10, M0 = 1
    v[50 + 48] = v[10 + 48]                                         # v_movrelsd_b32 v50, v10, M0 = 48
    return [v[r] for r in (20, 21, 22, 32, 30, 131, 41, 98)]


EXPECTED = reference()


class MovrelFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0xA0, 0xA3, 0x92, 0xA4, 0x30, 0xA5, 0xA1, 0x58])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_movrel", fssm=FSSM)
        mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_movrel"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()