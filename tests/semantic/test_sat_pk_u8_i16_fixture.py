"""cs_w32_sat_pk: the GPU test of v_sat_pk_u8_i16, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_sat_pk.asm stores eight v_sat_pk_u8_i16 results (VGPR, literal and SGPR sources, VOP1 and VOP3
encodings) into destinations preloaded with 0xdeadbeef. This runs the fixture through spirv_mini_interp, checks the
words against hand constants (the .asm header, measured on gfx1036) and a plain-Python model of the ISA pseudocode,
and checks that check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

SOURCES = [0x1234007F, 0xFFFF0001, 0x80008000, 0x7FFF8001, 0x00FF0100, 0x010000FE, 0xFFFE0100, 0x00800080]
HAND = [0x0000FF7F, 0x00000001, 0x00000000, 0x0000FF00, 0x0000FFFF, 0x0000FFFE, 0x000000FF, 0x00008080]


def sat_pk_u8_i16(x):
    """D = {sat8(S0.i16[31:16]), sat8(S0.i16[15:0])} in the low 16 bits; bits 31:16 are 0."""
    def half(h):
        h -= 0x10000 if h & 0x8000 else 0
        return max(0, min(255, h))
    return half(x & 0xFFFF) | (half(x >> 16) << 8)


class SatPkFixture(unittest.TestCase):
    def test_model_matches_hand_constants(self):
        self.assertEqual([sat_pk_u8_i16(x) for x in SOURCES], HAND)

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_sat_pk", fssm=FSSM)
        mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(HAND):
            with self.subTest(word=k):
                self.assertEqual(f"{mem[4 * k]:#010x}", f"{want:#010x}")

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_sat_pk"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], HAND)


if __name__ == "__main__":
    unittest.main()
